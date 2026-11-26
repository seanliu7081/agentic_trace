"""DAgger: the M1E teacher distilled into the student (LRL_PLAN.md M1E step 3, E2).

The student acts on run-time inputs only (student_obs: the head and wrist point cloud in frame A with the D9 sensor
model, and lrl_e2e's proprio); the teacher labels every visited state with its action from its privileged observation
(teacher_env). Early on a share beta of the episodes is driven by the teacher (beta falls linearly to 0), the rest by
the student; every state of every episode is labelled and kept in one aggregated buffer (the latest `buffer` states,
in shared memory), on which a trainer process trains the student continuously on one GPU and publishes its weights
to the acting loop.

StudentEnv is TeacherEnv plus the cameras: it renders head and wrist after every step (or every render_every-th),
builds the student's observation, and takes the believed part's centroid from the degraded head-camera mask of the
part, as E3 takes it from SAM3's (the teacher env's own draw of the normal error and of the wrong face stays).
vec.VecEnv runs it (env_cls); each worker renders on one of the given GPUs.

Labels: the teacher's mean numbers and, for done, the probability of a positive draw (a PPO teacher raises done
through its noise, ppo.Policy). Without a teacher checkpoint (--teacher none) an untrained ActorCritic of the teacher's
shape stands in: same work per step, no meaning; for measuring the loop.

  cd src && CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m lrl_train.dagger --name d0 --teacher <ckpt.pt> \
      --minutes 60 --envs 32 --gpus 2,3 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

from atrace.lrl_e2e import ACT_DIM, PROPRIO_DIM, Belief

from . import ppo
from . import student as S
from . import student_obs as SO
from .render import HEAD, WRIST, Cameras, egl_device
from .teacher_env import CFG as TEACHER, CRIT_DIM, OBS_DIM as T_DIM, REWARD, TeacherEnv, span_of
from .train_teacher import load_teacher
from .vec import VecEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "lrl_m1e_20261125" / "e2"
S_DIM = SO.OBS_DIM

CFG = {
    "envs": 32,                 # T  simulator processes (this workflow's budget for E2)
    "gpus": (2, 3),             # T  nvidia-smi GPUs the workers render on, in turn
    "size": SO.CFG["size"],     # T  px of the rendered images
    "render_every": 1,          # T  control steps between two renders (the cloud is reused in between)
    "min_part_points": 20,      # T  part points below which the env's own belief centroid is kept (part not seen)
    "scale": 1.0,               # T  the teacher env's curriculum scale: the full hand-off error
    "beta0": 1.0,               # T  share of teacher-driven episodes at the start ...
    "beta_steps": 2e6,          # T  ... falling linearly to 0 over this many labelled steps
    "buffer": 2_000_000,        # T  labelled states kept (float16 observations, ~13 KB a state)
    "warmup": 20_000,           # T  labelled states before the first update
    "batch": 512,               # T
    "lr": 3e-4,                 # T  Adam
    "sync": 20,                 # T  updates between copies of the trained student to the acting one
    "act_share": 0.5,           # T  the acting loop may wait until this share of the stepping envs is live ...
    "act_wait": 0.0,            # T  s ... or this long (larger batches, fewer GPU calls; with 8 envs it only added
                                #    latency: loop/smoke8, NOTES.md)
    "reuse": 8.0,               # T  updates x batch per labelled state at most: bounds the trainer's share of the GIL
    "save_minutes": 10.0,       # T
}


# -------------------------------------------------------------------------------------------------- environment
class StudentEnv(TeacherEnv):
    """reset(arg) -> (obs, crit, info), step(action) -> (obs, crit, reward, ended, info) as TeacherEnv; obs is the
    student's observation (S_DIM) followed by the teacher's (teacher_env.OBS_DIM). cfg: the teacher env's settings
    with a "student" entry (CFG keys gpus, size, render_every, min_part_points, and "obs": student_obs settings)."""

    def __init__(self, seed, out_dir, cfg=None, reward=REWARD, randomize=True, dump_dir=None):
        cfg = cfg or dict(TEACHER, student=student_cfg(CFG))
        super().__init__(seed, out_dir, cfg, reward, randomize, dump_dir, source="student", belief_fn=self._seen)
        self.sc = cfg["student"]
        i = seed[1] if isinstance(seed, (list, tuple)) else 0          # VecEnv's seed: [seed, slot, born]
        egl_device(self.sc["gpus"][i % len(self.sc["gpus"])])
        self.so = SO.StudentObs(self.sc["obs"])
        self.cams, self.cloud, self.rows = None, np.zeros((0, 3)), None

    def _cameras(self, sim):
        if self.cams is None:
            self.cams = Cameras(sim.robot.m, sim.robot.d, self.sc["size"])
        elif self.cams.m is not sim.robot.m:
            self.cams.reload(sim.robot.m, sim.robot.d)          # a newly loaded layout: 0.15 s, measured
        return self.cams

    def _seen(self, sim, part):
        """TeacherEnv's belief_fn, once per start with the arm at rest: the part's degraded head-camera mask, as the
        harness segments S1's pointer. Its centroid replaces the drawn one (the drawn normal error and wrong face
        stay); None when too little of the part is seen, and the env keeps its own draw."""
        s_open, _, span = span_of(part)
        self.so.episode(sim.robot.m, part, self.rng, (sim.joint(part.joint) - part.closed) * s_open / span)
        self.cloud = self.so.part_cloud(self._cameras(sim).render((HEAD,)))
        if len(self.cloud) < self.sc["min_part_points"]:
            return None
        return Belief(self.cloud.mean(0), part.face_normal, self.cloud)

    def reset(self, arg=1.0):
        again = self.replays > 0            # the start of the last episode once more: same part points
        t_obs, crit, info = super().reset(arg)
        views = self._cameras(self.sim).render()
        if not again:
            wrist = [v for v in views if v.name == WRIST and WRIST in self.sc["obs"]["start_views"]]
            self.so.start(self.frame, np.concatenate([self.cloud, self.so.part_cloud(wrist)]))
        self.rows, self.k = self.so.points(views, self.frame), 0
        return self._join(t_obs), crit, info | {"part_points": int(len(self.cloud))}

    def step(self, action):
        t_obs, crit, r, ended, info = super().step(action)
        self.k += 1
        if not ended and self.k % self.sc["render_every"] == 0:     # a terminal observation is never acted on
            self.rows = self.so.points(self.cams.render(), self.frame)
        return self._join(t_obs), crit, r, ended, info

    def _join(self, t_obs):
        return np.concatenate([t_obs[:PROPRIO_DIM], self.rows.ravel(), t_obs]).astype(np.float32)

    def close(self):
        super().close()
        if self.cams is not None:
            self.cams.close()


def student_cfg(c):
    return {"gpus": tuple(c["gpus"]), "size": c["size"], "render_every": c["render_every"],
            "min_part_points": c["min_part_points"], "obs": dict(SO.CFG, size=c["size"])}


# ------------------------------------------------------------------------------------------------------ teacher
class Teacher:
    """A teacher checkpoint (train_teacher.load_teacher) as a labeller, on the learner's device: observations ->
    (mean numbers clipped to [-1, 1], probability of done = Phi(mean / done_std))."""

    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device
        self.done_std = float(model.log_std[-1].detach().exp())

    @classmethod
    def load(cls, path, device):
        return cls(load_teacher(path, device).model, device)

    @classmethod
    def stand_in(cls, device, seed=0):
        torch.manual_seed(seed)
        return cls(ppo.ActorCritic(T_DIM, CRIT_DIM, ACT_DIM), device)

    @torch.inference_mode()
    def __call__(self, obs):
        mean = self.model.actor(self.model.obs_norm(torch.as_tensor(obs, device=self.device)))
        p = torch.special.ndtr(mean[:, -1] / self.done_std)
        return torch.cat([mean[:, :-1].clamp(-1.0, 1.0), p[:, None]], -1)


class Acting:
    """Per acting call, for up to n rows: the teacher's labels (mean numbers, P(done)) and the action each env takes
    (the teacher's with its done drawn, or the student's with its done drawn), in ONE CUDA graph replay padded to n
    rows. As plain calls these were ~60 kernel launches and two syncs, 5-10 ms per call under the machine's load,
    and the acting loop was what held 30 envs back (loop30, NOTES.md)."""

    def __init__(self, teacher, student, n, device):
        self.teacher, self.student, self.n = teacher, student, n
        self.inp = torch.zeros(n, S_DIM + T_DIM, device=device)
        self.teach = torch.zeros(n, dtype=torch.bool, device=device)
        self.u = torch.zeros(n, 2, device=device)
        self.host = torch.zeros(n, S_DIM + T_DIM).pin_memory()
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):                  # warm-up before capture, as torch.cuda.graphs asks
                self._fn()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = self._fn()

    def _fn(self):
        t = self.teacher.model
        mean = t.actor(t.obs_norm(self.inp[:, S_DIM:]))
        p = torch.special.ndtr(mean[:, -1] / self.teacher.done_std)
        target = torch.cat([mean[:, :-1].clamp(-1.0, 1.0), p[:, None]], -1)
        a, logit = self.student(self.inp[:, :S_DIM])
        t_act = torch.cat([target[:, :-1], torch.where(p > self.u[:, 0], 1.0, -1.0)[:, None]], -1)
        s_act = torch.cat([a, torch.where(torch.sigmoid(logit) > self.u[:, 1], 1.0, -1.0)[:, None]], -1)
        return torch.cat([torch.where(self.teach[:, None], t_act, s_act), target], -1)

    def __call__(self, obs, teach):
        """obs (k, S_DIM + T_DIM), teach (k,) bool -> (actions (k, ACT_DIM), labels (k, ACT_DIM)), numpy."""
        k = len(obs)
        self.host[:k] = torch.from_numpy(obs)
        self.inp[:k].copy_(self.host[:k], non_blocking=True)
        self.teach[:k] = torch.from_numpy(teach)
        self.u.uniform_()
        self.graph.replay()
        out = self.out[:k].cpu().numpy()
        return out[:, :ACT_DIM], out[:, ACT_DIM:]


# ------------------------------------------------------------------------------------------------------- buffer
class Buffer:
    """The aggregated dataset in shared memory: the latest `cap` labelled states (a ring; observations in float16),
    written by the acting process and sampled by the trainer process. meta = (states held, next row, states added) is
    written after the rows, so no unwritten row is ever sampled (once the ring has wrapped, a row read while it is
    overwritten is a rare mixed sample). name: attach to an existing buffer."""

    def __init__(self, cap, dim, name=None):
        self.cap, self.dim = cap, dim
        o, t = cap * dim * 2, cap * ACT_DIM * 4
        self.shm = shared_memory.SharedMemory(name=name, create=name is None, size=o + t + 24)
        self.obs = np.ndarray((cap, dim), np.float16, self.shm.buf)
        self.target = np.ndarray((cap, ACT_DIM), np.float32, self.shm.buf, o)
        self.meta = np.ndarray(3, np.int64, self.shm.buf, o + t)
        if name is None:
            self.meta[:] = 0

    n = property(lambda self: int(self.meta[0]))
    i = property(lambda self: int(self.meta[1]))
    added = property(lambda self: int(self.meta[2]))

    def add(self, obs, target):
        n, i, added = (int(x) for x in self.meta)
        k = len(obs)
        idx = (i + np.arange(k)) % self.cap
        self.obs[idx], self.target[idx] = obs, target
        self.meta[:] = (min(n + k, self.cap), (i + k) % self.cap, added + k)

    def sample(self, k, rng):
        idx = np.sort(rng.integers(self.n, size=k))
        return self.obs[idx], self.target[idx]

    def close(self, unlink=False):
        del self.obs, self.target, self.meta
        self.shm.close()
        if unlink:
            self.shm.unlink()


# ------------------------------------------------------------------------------------------------------ training
class Weights:
    """The student's parameters as one float32 vector in shared memory, with a version number: the trainer process
    publishes, the acting process copies when the version changed."""

    def __init__(self, model, name=None):
        self.size = sum(p.numel() for p in model.parameters()) + sum(b.numel() for b in model.buffers())
        self.shm = shared_memory.SharedMemory(name=name, create=name is None, size=4 * self.size + 8 + 4 * 8)
        self.vec = np.ndarray(self.size, np.float32, self.shm.buf)
        self.version = np.ndarray(1, np.int64, self.shm.buf, 4 * self.size)
        self.stats = np.ndarray(4, np.float64, self.shm.buf, 4 * self.size + 8)    # updates, mse, done_bce, done_acc
        if name is None:
            self.version[:], self.stats[:] = 0, 0

    @staticmethod
    def _tensors(model):
        return list(model.parameters()) + list(model.buffers())

    def publish(self, model):
        self.vec[:] = torch.cat([t.detach().reshape(-1).float().cpu() for t in self._tensors(model)]).numpy()
        self.version[0] += 1

    @torch.no_grad()
    def load_into(self, model):
        flat, k = torch.from_numpy(self.vec.copy()), 0
        for t in self._tensors(model):
            t.copy_(flat[k:k + t.numel()].view_as(t))
            k += t.numel()

    def close(self, unlink=False):
        del self.vec, self.version, self.stats
        self.shm.close()
        if unlink:
            self.shm.unlink()


def _trainer(buf_name, w_name, cfg, device, out, stop):
    """The trainer process: Adam on batches of the shared buffer, at most `reuse` updates x batch per labelled state;
    publishes the weights every `sync` updates and saves them every save_minutes."""
    torch.set_num_threads(2)
    model = S.Student().to(device)
    buf, w = Buffer(cfg["buffer"], S_DIM, buf_name), Weights(model, w_name)
    w.load_into(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng, updates, last_save = np.random.default_rng(1), 0, time.time()
    while not stop.is_set():
        if buf.n < cfg["warmup"] or updates * cfg["batch"] > cfg["reuse"] * buf.added:
            time.sleep(0.01)
            continue
        o, t = buf.sample(cfg["batch"], rng)
        loss, st = model.loss(torch.as_tensor(o, device=device).float(), torch.as_tensor(t, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        updates += 1
        w.stats[:] = (updates, st["mse"], st["done_bce"], st["done_acc"])
        if updates % cfg["sync"] == 0:
            w.publish(model)
        if time.time() - last_save > 60 * cfg["save_minutes"]:
            S.save(Path(out) / "student_latest.pt", model, {"updates": updates, "labelled": buf.added})
            last_save = time.time()
    S.save(Path(out) / "student_latest.pt", model, {"updates": updates, "labelled": buf.added})
    w.publish(model)
    buf.close()
    w.close()


def run(a):
    cfg = dict(CFG, envs=a.envs, gpus=tuple(int(g) for g in a.gpus.split(",")), size=a.size,
               render_every=a.render_every, buffer=a.buffer, beta_steps=a.beta_steps, warmup=a.warmup)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"args": vars(a), "cfg": cfg, "obs": SO.CFG, "student": S.CFG},
                                                indent=1, default=str))
    env_cfg = dict(TEACHER, student=student_cfg(cfg))
    # the workers are forked before this process touches CUDA (a forked child renders fine either way: tested)
    vec = VecEnv(cfg["envs"], out / "bddl", a.seed, cfg=env_cfg, reward=REWARD, env_cls=StudentEnv,
                 dims=(S_DIM + T_DIM, CRIT_DIM))
    device = torch.device(a.device)
    teacher = Teacher.stand_in(device) if a.teacher == "none" else Teacher.load(a.teacher, device)
    acting = S.Student().to(device).eval()
    buf, w = Buffer(cfg["buffer"], S_DIM), Weights(acting)
    w.publish(acting)
    act_fn = Acting(teacher, acting, cfg["envs"], device)
    # the trainer is a process of its own: as a thread its batch sampling took the acting loop's GIL (loop30 against
    # loop30_notrain, NOTES.md)
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    trainer = ctx.Process(target=_trainer, args=(buf.shm.name, w.shm.name, cfg, a.train_device, str(out), stop))
    trainer.start()
    version = int(w.version[0])
    rng = np.random.default_rng(a.seed)
    n = cfg["envs"]
    driver = np.zeros(n, bool)                  # this env's episode is driven by the teacher
    labelled, episodes, t0, last_n = 0, 0, time.time(), 0
    log = open(out / "train.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["min", "labelled", "per_s", "beta", "updates", "mse", "done_bce", "done_acc", "episodes",
                "succ_student", "succ_teacher", "buffer", "rows_per_act", "ms_wait", "ms_act"])
    eps = open(out / "episodes.csv", "w", newline="")
    we = csv.writer(eps)
    we.writerow(["min", "driver", "kind", "handle", "effect", "stop", "steps", "success", "label", "part_points"])
    recent = {True: [], False: []}
    step = vec.reset(cfg["scale"])
    beta = cfg["beta0"]
    driver[:] = rng.random(n) < beta
    last_log, part_pts = t0, np.zeros(n, int)
    for i in range(n):
        part_pts[i] = step.info[i].get("part_points", 0)
    clock = np.zeros(3)                         # s waiting for envs, s acting; acting calls (since the last log)

    def book(step):
        nonlocal episodes
        for i in np.flatnonzero(step.ended):
            info = step.info[i]
            episodes += 1
            recent[bool(driver[i])] = (recent[bool(driver[i])] + [bool(info["success"])])[-500:]
            we.writerow([round((time.time() - t0) / 60, 2), "teacher" if driver[i] else "student", info["kind"],
                         info["handle"], info["effect"], info["stop"], info["steps"], info["success"], info["label"],
                         part_pts[i]])
        for i in np.flatnonzero(step.fresh):
            driver[i] = rng.random() < beta
            part_pts[i] = step.info[i].get("part_points", 0)

    while time.time() - t0 < 60 * a.minutes and labelled < a.steps:
        live = np.flatnonzero(step.live)
        if not len(live):                       # only episode ends came back: wait for the next answers
            step = vec.step()
            book(step)
            continue
        t1 = time.perf_counter()
        if int(w.version[0]) != version:
            version = int(w.version[0])
            w.load_into(acting)
        obs = np.stack([step.obs[i] for i in live])
        executed, target = act_fn(obs, driver[live])
        actions = np.zeros((n, ACT_DIM))
        actions[live] = executed
        buf.add(obs[:, :S_DIM], target)
        labelled += len(live)
        beta = max(0.0, cfg["beta0"] * (1.0 - labelled / cfg["beta_steps"]))
        t2 = time.perf_counter()
        step = vec.step(actions)
        book(step)
        t_end = t2 + cfg["act_wait"]
        while (vec.live.sum() < cfg["act_share"] * (n - vec.resetting.sum()) and vec.busy.any()
               and time.perf_counter() < t_end):
            step = vec.step()
            book(step)
        clock += (time.perf_counter() - t2, t2 - t1, 1)
        now = time.time()
        if now - last_log > 30:
            st = w.stats.copy()
            wr.writerow([round((now - t0) / 60, 2), labelled, round(labelled / (now - t0), 1), round(beta, 3),
                         int(st[0]), st[1], st[2], st[3], episodes,
                         np.mean(recent[False]) if recent[False] else None,
                         np.mean(recent[True]) if recent[True] else None, buf.n,
                         round((labelled - last_n) / max(clock[2], 1), 2), round(1e3 * clock[0] / max(clock[2], 1), 2),
                         round(1e3 * clock[1] / max(clock[2], 1), 2)])
            clock[:], last_n = 0, labelled
            log.flush()
            eps.flush()
            last_log = now
    stop.set()
    trainer.join()
    dt = time.time() - t0
    updates = int(w.stats[0])
    summary = {"labelled": labelled, "seconds": round(dt, 1), "labelled_per_s": round(labelled / dt, 1),
               "updates": updates, "updates_per_s": round(updates / dt, 2), "episodes": episodes,
               "envs": n, "size": cfg["size"], "render_every": cfg["render_every"], "teacher": a.teacher}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    log.close()
    eps.close()
    vec.close()
    buf.close(unlink=True)
    w.close(unlink=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--teacher", default="none", help="teacher checkpoint (ppo.save); none: an untrained stand-in")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--steps", type=float, default=1e12)
    ap.add_argument("--envs", type=int, default=CFG["envs"])
    ap.add_argument("--gpus", default=",".join(map(str, CFG["gpus"])))
    ap.add_argument("--size", type=int, default=CFG["size"])
    ap.add_argument("--render-every", type=int, default=CFG["render_every"])
    ap.add_argument("--buffer", type=int, default=CFG["buffer"])
    ap.add_argument("--beta-steps", type=float, default=CFG["beta_steps"])
    ap.add_argument("--warmup", type=float, default=CFG["warmup"], help="labelled states before the first update")
    ap.add_argument("--device", default="cuda:0", help="acting: student and teacher")
    ap.add_argument("--train-device", default="cuda:0", help="the trainer process")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
