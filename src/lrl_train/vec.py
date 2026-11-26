"""A vector of StageEnv processes over pipes, with explicit resets (LRL_PLAN.md 6.3).

Nothing resets by itself, and no environment waits for another. A call to step sends each live environment its
action and returns as soon as some environments have answered; the others are still stepping (a step that ends on
the force limit holds three more control steps) or resetting (up to a few seconds: postures by inverse kinematics,
the way in, sometimes a layout load). An environment whose episode ended returns that terminal step; its reset then
runs in its own process, and the first observation of its next episode comes back in a later call. Every call says
per environment what its row is:

  stepped  a transition was made: reward and ended are valid
  ended    that transition ended the episode; obs is the terminal observation, info the episode's summary
  fresh    obs is the first observation of a new episode (no transition, reward 0)
  live     obs is one to act on: the next call uses this environment's row of actions (stepped and not ended, or
           fresh). Rows that are none of these belong to environments still at work.

A process that asks to be retired (StageEnv max_loads) is replaced by a new one before its reset.

Any environment class with StageEnv's constructor and protocol runs here (env_cls, with its observation and critic
sizes in dims): the M1E teacher's TeacherEnv does.
"""
from __future__ import annotations

import multiprocessing as mp
import selectors
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .env import ACT_DIM, CFG, CRIT_DIM, OBS_DIM, REWARD, StageEnv

GATHER = 0.0004     # T  s: after the first answer, wait this long for more, so that the policy runs on batches


def _worker(conn, seed, out_dir, cfg, reward, randomize, dump_dir, env_cls):
    env = env_cls(seed, out_dir, cfg, reward, randomize, dump_dir)
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            conn.send(env.step(arg))
        elif cmd == "reset":
            conn.send(env.reset(arg))
        else:
            env.close()
            conn.close()
            return


@dataclass
class Step:
    obs: list                   # per environment, an (OBS_DIM,) array
    crit: list
    reward: np.ndarray
    stepped: np.ndarray
    ended: np.ndarray
    fresh: np.ndarray
    live: np.ndarray
    info: list


class VecEnv:
    def __init__(self, n, out_dir, seed=0, cfg=CFG, reward=REWARD, randomize=True, dump_dir=None, env_cls=StageEnv,
                 dims=(OBS_DIM, CRIT_DIM)):
        self.n, self.out, self.seed = n, Path(out_dir), seed
        self.args = (cfg, reward, randomize, dump_dir, env_cls)
        self.ctx = mp.get_context("fork")
        self.sel = selectors.DefaultSelector()      # every pipe, registered once: an answer is all that is ever read
        self.born = np.zeros(n, int)                # processes started in each slot: part of the seed
        self.procs, self.conns = [None] * n, [None] * n
        for i in range(n):
            self._spawn(i)
        self.live = np.zeros(n, bool)
        self.busy = np.zeros(n, bool)               # a step is under way
        self.resetting = np.zeros(n, bool)
        # one array per environment as it came out of its pipe (never written to): a Step hands out references, not
        # copies of an (n, OBS_DIM) block on every call, which cost the learner more than the policy at 192 workers
        self.obs, self.crit = [np.zeros(dims[0])] * n, [np.zeros(dims[1])] * n
        self.scale = 1.0
        self.weights = None                         # {kind: weight} for the layouts the workers load next

    def _spawn(self, i):
        here, there = self.ctx.Pipe()
        self.procs[i] = self.ctx.Process(
            target=_worker, daemon=True,
            args=(there, [self.seed, i, int(self.born[i])], self.out / f"w{i:03d}", *self.args))
        self.procs[i].start()
        there.close()
        self.conns[i] = here
        self.sel.register(here, selectors.EVENT_READ, i)
        self.born[i] += 1

    def _retire(self, i):
        self.conns[i].send(("close", None))
        self.sel.unregister(self.conns[i])
        self.conns[i].close()
        self.procs[i].join()
        self._spawn(i)

    def _take(self, i, out):
        """The answer of a reset that has finished."""
        self.obs[i], self.crit[i], out.info[i] = self.conns[i].recv()
        self.resetting[i], self.live[i], out.fresh[i] = False, True, True

    def _blank(self):
        n = self.n
        return Step(None, None, np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool), None,
                    [{} for _ in range(n)])

    def _filled(self, out):
        out.obs, out.crit, out.live = list(self.obs), list(self.crit), self.live.copy()
        return out

    def reset(self, scale=1.0):
        """Start an episode in every environment and wait for all of them."""
        self.scale = scale
        for i in np.flatnonzero(~self.resetting):
            self.conns[i].send(("reset", (scale, self.weights)))
        self.resetting[:], self.live[:] = True, False
        out = self._blank()
        for i in range(self.n):
            self._take(i, out)
        return self._filled(out)

    def step(self, actions=None):
        """Send the live environments their rows of `actions`, wait for the first answer of any environment, and
        return every answer that is there GATHER seconds later. Without actions nothing is sent: the call only
        collects (settle uses it), and the environments that were live stay live."""
        out = self._blank()
        if actions is not None:
            for i in np.flatnonzero(self.live):
                self.conns[i].send(("step", actions[i]))
            self.busy |= self.live
            self.live[:] = False
        ready = self.sel.select()
        if GATHER:
            time.sleep(GATHER)
            ready = self.sel.select(0)
        for key, _ in ready:
            i, conn = key.data, key.fileobj
            if self.resetting[i]:
                self._take(i, out)
                continue
            self.obs[i], self.crit[i], out.reward[i], out.ended[i], out.info[i] = conn.recv()
            self.busy[i], out.stepped[i] = False, True
            if out.ended[i]:
                self.resetting[i] = True
                if out.info[i]["retire"]:
                    self._retire(i)
                self.conns[i].send(("reset", (self.scale, self.weights)))
            else:
                self.live[i] = True
        return self._filled(out)

    def settle(self):
        """Collect until no step is under way: the answers, one Step per call. Afterwards every environment is
        live or resetting."""
        outs = []
        while self.busy.any():
            outs.append(self.step())
        return outs

    def close(self):
        for i in range(self.n):
            if self.resetting[i] or self.busy[i]:
                self.conns[i].recv()
            self.conns[i].send(("close", None))
        for p in self.procs:
            p.join()


__all__ = ["VecEnv", "Step", "ACT_DIM", "OBS_DIM", "CRIT_DIM"]
