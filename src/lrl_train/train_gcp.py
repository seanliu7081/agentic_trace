"""Train and time the same end-to-end depth-policy PPO loop on procedural Panda scenes.

From agentic_trace: PYTHONPATH=src .venv/bin/python -m lrl_train.train_gcp
  --preset smoke_door_drawer [--workers 96 --gpus 0,1,2 --minutes 120]
Use --benchmark --preset all_5_families for measured, update-aligned training budgets.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from multiprocessing.connection import wait
from pathlib import Path

_MODULE_START = time.monotonic()
import numpy as np
import torch
from torch import nn

if not __package__:  # also permit "python src/lrl_train/train_gcp.py --preset ..."
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "lrl_train"

from atrace.gcp import ACTION_DIM as ACT_DIM, OBS_SHAPES
from .gcp_devices import configure_egl_worker, configure_learner, parse_gpus

ROOT = Path(__file__).resolve().parents[2]
OBS_DIM = sum(int(np.prod(shape)) for shape in OBS_SHAPES.values())
CRIT_DIM = 16
ROW_DIM = OBS_DIM + CRIT_DIM


def obs_views(flat):
    """Views of packed float32 observations, with the private critic tail always excluded."""
    fields, offset = {}, 0
    for name, shape in OBS_SHAPES.items():
        end = offset + int(np.prod(shape))
        fields[name] = flat[..., offset:end].reshape(flat.shape[:-1] + shape)
        offset = end
    return fields


def _write_obs(row, obs, info):
    for key, view in obs_views(row).items():
        if obs[key].shape != view.shape:
            raise ValueError(f"{key} has shape {obs[key].shape}, expected {view.shape}")
        view[:] = obs[key]
    row[OBS_DIM:] = info["critic"]
    if not np.isfinite(row).all():
        raise FloatingPointError("Non-finite GCP input")


def _worker(conn, shared, i, n, seed, gpu, directory, preset, horizon, env_options, use_counterfactual):
    directory.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        with (directory / "worker.log").open("a", buffering=1) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            configure_egl_worker(gpu)
            from .gcp_env import GCPEnv
            env = GCPEnv(seed=seed, out_dir=directory, gpu=gpu, preset=preset,
                         staged_reset_prob=0.5, horizon=horizon, **env_options)
            row = np.frombuffer(shared, np.float32).reshape(n, ROW_DIM)[i]
            obs, info = env.reset()
            _write_obs(row, obs, info)
            def priming(info):
                return tuple(float(info.get(k, 0)) for k in
                             ("priming_steps", "priming_dual_renders", "priming_seconds",
                              "priming_attempts", "priming_successes"))

            conn.send(("ready", info["render_count"], priming(info)))
            while True:
                command, action = conn.recv()
                if command == "close":
                    break
                expert = None if command == "ppo" else np.asarray(env.expert_action(), np.float32)
                alternative = env.counterfactual_expert() if use_counterfactual and expert is not None else None
                before_time = float(env.d.time)
                obs, reward, done, info = env.step(expert if command == "teacher" else action)
                advanced = float(env.d.time) >= before_time + .0499
                n_render = info["render_count"]
                step_kind = info["kind"]
                pen = float(info.get("max_pen_m", 0.0))
                episode = None
                reset_seconds, new_layout = 0., False
                if done:
                    episode = {k: info.get(k) for k in
                               ("kind", "goal_deg", "slip_injected", "regrasped", "success", "staged",
                                "staged_origin", "entry_ok", "max_pen_m", "reward_counts", "outcome")}
                    previous_layout = env.key
                    before_reset = time.monotonic()
                    obs, info = env.reset()
                    reset_seconds, new_layout = time.monotonic() - before_reset, env.key != previous_layout
                _write_obs(row, obs, info)
                conn.send(("step", float(reward), bool(done), pen, episode, expert, n_render,
                           info["render_count"], advanced, step_kind, priming(info), alternative, reset_seconds, new_layout))
    except (BrokenPipeError, EOFError):
        pass
    except BaseException:
        with contextlib.suppress(OSError):
            conn.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class Pool:
    """Workers write independent float32 rows. A reply is the ownership barrier for its row."""

    def __init__(self, args, path, physical_gpus):
        # The parent has no CUDA or GL context yet. Preloading CPU-side Python modules saves
        # private RSS and avoids 96 independent imports of robosuite and numba.
        import robosuite  # noqa: F401
        from . import sim  # noqa: F401

        ctx = mp.get_context("fork")
        self.n = args.workers
        self.shared = ctx.RawArray(ctypes.c_float, self.n * ROW_DIM)
        self.rows = np.frombuffer(self.shared, np.float32).reshape(self.n, ROW_DIM)
        self.conns, self.procs = [], []
        self.render_count = np.zeros(self.n, np.int64)
        self.steps = np.zeros(self.n, np.int64)
        self.physics_steps = np.zeros(self.n, np.int64)
        self.kind_steps = Counter()
        self.priming = np.zeros((self.n, 5))
        self.step_renders = 0
        self.measure_visual_occupancy = getattr(args, "benchmark", False)
        self.visual_points = defaultdict(list)
        self.paired_goals = 0
        self.paired_kinds = Counter()
        self.layout_rebuilds = 0
        self.reset_worker_seconds = 0.
        env_options = {}
        if getattr(args, "smoke_mix", None) is not None:
            env_options["smoke_sampling"] = args.smoke_mix
        if getattr(args, "reward_mode", "baseline") != "baseline":
            env_options["reward_mode"] = args.reward_mode
        self.use_counterfactual = getattr(args, "counterfactual_coef", 0.) > 0.
        self.rank_gpu = [physical_gpus[i % len(physical_gpus)] for i in range(self.n)]
        if args.preset == "all_5_families":
            rng = np.random.default_rng([args.seed, getattr(args, "rollout_generation", 0)])
            seeds = rng.choice(100_000, self.n, replace=False).tolist()
        else:
            seeds = [int(np.random.SeedSequence([args.seed, i]).generate_state(1)[0])
                     for i in range(self.n)]
        self.worker_seeds = seeds
        try:
            for i, gpu in enumerate(self.rank_gpu):
                recv, send = ctx.Pipe()
                seed = seeds[i]
                p = ctx.Process(target=_worker, args=(send, self.shared, i, self.n, seed, gpu,
                                path / "workers" / f"{i:03d}", args.preset, args.horizon,
                                env_options, self.use_counterfactual), daemon=True)
                p.start()
                send.close()
                self.conns.append(recv)
                self.procs.append(p)
            self.by_conn = {c: i for i, c in enumerate(self.conns)}
            remaining = set(range(self.n))
            while remaining:
                for i, (tag, initial_renders, priming) in self.receive(remaining):
                    if tag != "ready" or initial_renders < 1:
                        raise RuntimeError(f"worker {i} did not render its first dual-camera observation")
                    self.render_count[i] = initial_renders
                    self.priming[i] = priming
                    remaining.remove(i)
        except BaseException:
            self.close()
            raise

    def receive(self, ids):
        ready = wait([self.conns[i] for i in ids], timeout=240.0)
        if not ready:
            raise TimeoutError(f"No simulation replies for 240 seconds ({len(ids)} pending)")
        for conn in ready:
            i = self.by_conn[conn]
            try:
                result = conn.recv()
            except EOFError as e:
                raise RuntimeError(f"GCP worker {i} exited ({self.procs[i].exitcode})") from e
            if result[0] == "error":
                raise RuntimeError(f"GCP worker {i}:\n{result[1]}")
            yield i, result

    def close(self):
        for conn in self.conns:
            with contextlib.suppress(OSError):
                conn.send(("close", None))
        deadline = time.monotonic() + 5.0
        for p in self.procs:
            p.join(max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                p.terminate()
        for p in self.procs:
            p.join()
        for conn in self.conns:
            conn.close()


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        from atrace.gcp import UniversalContactPolicy
        self.policy = UniversalContactPolicy()
        # Asymmetric value baseline has privileged state + deployable goal and history.
        self.critic = nn.Sequential(nn.Linear(CRIT_DIM + 16 + 36, 192), nn.SiLU(),
                                    nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -1.0))

    def logits(self, rows):
        return self.policy.action_head(self.policy.encode(obs_views(rows))).float()

    def value(self, rows):
        obs = obs_views(rows)
        x = torch.cat((rows[:, OBS_DIM:], obs["goal"], obs["history"][:, -1]), dim=-1)
        return self.critic(x).flatten()

    def distribution(self, logits):
        return torch.distributions.Normal(logits, self.log_std.clamp(-3.5, 0.25).exp())

    @torch.no_grad()
    def act(self, rows):
        dist = self.distribution(self.logits(rows))
        raw = dist.sample()
        output = torch.cat((raw, raw.tanh(), dist.log_prob(raw).sum(-1, keepdim=True),
                            self.value(rows)[:, None]), -1)
        return output.cpu().numpy()


def collect(pool, model, count, device, mode, rng=None, teacher_probability=0.0):
    rows = np.empty((count, ROW_DIM), np.float32)
    actions, labels = (np.empty((count, ACT_DIM), np.float32) for _ in range(2))
    logp, values, rewards = (np.zeros(count, np.float32) for _ in range(3))
    done = np.zeros(count, bool)
    alternatives = {}
    if pool.use_counterfactual and mode != "ppo":
        alternatives = {"alternate_goal": np.zeros((count,16),np.float32),
                        "alternate_label": np.zeros((count,ACT_DIM),np.float32),
                        "has_alternate": np.zeros(count,bool)}
    episodes, trajectories = [], [[] for _ in range(pool.n)]
    ready, pending = deque(range(pool.n)), {}
    issued = 0
    while issued < count or pending:
        if ready and issued < count:
            n = min(len(ready), count - issued)
            ids = [ready.popleft() for _ in range(n)]
            rows[issued:issued+n] = pool.rows[ids]
            if mode == "mixed":
                with torch.no_grad():
                    deterministic = model.logits(torch.as_tensor(rows[issued:issued+n], device=device)).tanh().cpu().numpy()
                use_teacher = rng.random(n) < teacher_probability
            elif mode != "teacher":
                out = model.act(torch.as_tensor(rows[issued:issued+n], device=device))
            for j, i in enumerate(ids):
                k = issued + j
                a = None
                command = mode
                if mode == "mixed":
                    command, a = ("teacher" if use_teacher[j] else "label"), deterministic[j]
                elif mode != "teacher":
                    actions[k], a = out[j, :ACT_DIM], out[j, ACT_DIM:2*ACT_DIM]
                    logp[k], values[k] = out[j, -2:]
                trajectories[i].append(k)
                pending[i] = k
                pool.conns[i].send((command, a))
            issued += n
        # Coalesce into modest inference batches while slower processes keep stepping.
        target = min(len(pending), max(1, pool.n // 3))
        got = 0
        while got < target:
            for i, result in pool.receive(pending):
                (_, reward, terminal, pen, ep, expert, step_render, final_render, advanced, kind, priming,
                 alternative, reset_seconds, new_layout) = result
                if step_render != pool.render_count[i] + 1:
                    raise RuntimeError(f"Worker {i} did not render exactly once per GCP step")
                pool.step_renders += 1
                pool.render_count[i] = final_render
                pool.steps[i] += 1
                pool.physics_steps[i] += int(advanced)
                pool.kind_steps[kind] += int(advanced)
                pool.priming[i] = priming
                pool.layout_rebuilds += int(new_layout)
                pool.reset_worker_seconds += reset_seconds
                k = pending.pop(i)
                if pool.measure_visual_occupancy and advanced and pool.physics_steps[i] % 64 == 1:
                    obs = obs_views(rows[k])
                    scene, payload = obs["scene_pts"], obs["payload_pts"]
                    actual = np.any(scene[:, :3] != 0., axis=1)
                    wrist = scene[:, 4] > .5
                    pool.visual_points[kind].append((int(actual.sum()), int(np.sum(actual & ~wrist)),
                        int(np.sum(actual & wrist)), int(np.sum(payload[:, 3] > 0.)),
                        int(np.sum(actual & (scene[:, 3] > .01)))))
                rewards[k], done[k] = reward, terminal
                # SAPU downweights positive returns once a forbidden contact exceeds 1 mm.
                if pen >= 1e-3 and rewards[k] > 0.0:
                    rewards[k] *= np.exp(-min(20.0, (pen - 1e-3) * 1000.0))
                if expert is not None:
                    labels[k] = expert
                if alternative is not None:
                    other_goal, other_action = alternative
                    if np.max(np.abs(expert - other_action)) > .12:
                        alternatives["alternate_goal"][k] = other_goal
                        alternatives["alternate_label"][k] = other_action
                        alternatives["has_alternate"][k] = True
                        pool.paired_goals += 1
                        pool.paired_kinds[kind] += 1
                if ep is not None:
                    episodes.append(ep)
                ready.append(i)
                got += 1
    if mode in ("teacher", "mixed"):
        return {"rows": rows, "labels": labels, **alternatives}, episodes
    with torch.no_grad():
        bootstrap = model.value(torch.as_tensor(pool.rows, device=device)).cpu().numpy()
    adv = np.empty(count, np.float32)
    for i, trajectory in enumerate(trajectories):
        next_value, gae = bootstrap[i], 0.0
        for k in reversed(trajectory):
            active = float(not done[k])
            td = rewards[k] + .99 * active * next_value - values[k]
            gae = td + .99 * .95 * active * gae
            adv[k], next_value = gae, values[k]
    batch = {"rows": rows, "actions": actions, "logp": logp, "adv": adv, "returns": adv + values}
    if mode == "label":
        batch["labels"] = labels
    return {**batch, **alternatives}, episodes


def imitation_loss(logits, labels, rows, balanced):
    if not balanced:
        return nn.functional.mse_loss(logits.tanh(), labels)
    pred = logits.tanh()
    movement = (pred[:, :6] - labels[:, :6]).square().mean(-1)
    width = nn.functional.binary_cross_entropy_with_logits(
        2. * logits[:, 7], (labels[:, 7] + 1.) / 2., reduction="none")
    compliance = (pred[:, 6] - labels[:, 6]).square()
    error = .65 * movement + .3 * width + .05 * compliance
    # Only the training loss sees the critic's physical phase and target kind.
    c = rows[:, OBS_DIM:]
    close, near = labels[:, 7] < 0., c[:, 10] < .025
    phase = torch.where(c[:, 7] > .5, 2, torch.where(near & close, 1, 0))
    phase = torch.where((c[:, 13] > .5) & (c[:, 7] < .5), 3, phase)
    phase = torch.where((~close) & (c[:, 12] < .012), 4, phase)
    group = (c[:, 3].abs() > 1e-5).long() * 5 + phase
    counts = torch.bincount(group, minlength=10).to(error.dtype)
    weights = counts[group].rsqrt()
    return (error * weights).sum() / weights.sum()


def update(model, optimizer, batch, device, args, coefficient, bc_only=False):
    batch = {key: torch.as_tensor(value, device=device) for key, value in batch.items()}
    n = len(batch["rows"])
    if not bc_only:
        adv = batch["adv"]
        batch["adv"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    losses, divergences = [], []
    early_stop = False
    for _ in range(args.bc_epochs if bc_only else args.epochs):
        for idx in torch.randperm(n, device=device).split(args.minibatch):
            rows = batch["rows"][idx]
            logits = model.logits(rows)
            if bc_only:
                loss = imitation_loss(logits, batch["labels"][idx], rows, args.balanced_imitation)
            else:
                dist = model.distribution(logits)
                lp = dist.log_prob(batch["actions"][idx]).sum(-1)
                # Actions are tanh(u), and the PPO ratio in u-space is exactly the same:
                # the bijection's log-Jacobian cancels between old and new policy.
                ratio = (lp - batch["logp"][idx]).exp()
                divergence = (ratio - 1. - (lp - batch["logp"][idx])).mean().detach()
                divergences.append(divergence)
                if args.target_kl and losses and float(divergence) > args.target_kl:
                    early_stop = True
                    break
                adv = batch["adv"][idx]
                policy = -torch.minimum(ratio * adv, ratio.clamp(.8, 1.2) * adv).mean()
                value = nn.functional.smooth_l1_loss(model.value(rows), batch["returns"][idx])
                loss = policy + .5 * value - .001 * dist.entropy().sum(-1).mean()
                if coefficient > 0:
                    loss = loss + coefficient * imitation_loss(logits, batch["labels"][idx], rows, args.balanced_imitation)
            if "has_alternate" in batch:
                paired = torch.nonzero(batch["has_alternate"][idx]).flatten()
                if paired.numel():
                    different_rows = rows[paired].clone()
                    obs_views(different_rows)["goal"][:] = batch["alternate_goal"][idx[paired]]
                    other = model.logits(different_rows)
                    other_label = batch["alternate_label"][idx[paired]]
                    predicted_change = other.tanh() - logits[paired].tanh()
                    target_change = other_label - batch["labels"][idx[paired]]
                    paired_loss = imitation_loss(other, other_label, different_rows, args.balanced_imitation)
                    paired_loss += .5 * nn.functional.mse_loss(predicted_change, target_change)
                    loss = loss + args.counterfactual_coef * (1. if bc_only else coefficient) * paired_loss
            model.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), .7, error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.detach())
        if early_stop:
            break
    model.update_stats = {"gradient_steps": len(losses),
                          "mean_sample_kl": float(torch.stack(divergences).mean()) if divergences else None,
                          "stopped_on_kl": early_stop,
                          "action_std": model.log_std.detach().clamp(-3.5, .25).exp().cpu().tolist()}
    return float(torch.stack(losses).mean())


def gpu_used_mib(gpus):
    out = subprocess.check_output(["nvidia-smi", "-i", ",".join(map(str, gpus)), "--query-gpu=index,memory.used",
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    return {int(i): int(m) for i, m in (line.split(",") for line in out.splitlines())}


class MemoryMonitor:
    def __init__(self, gpus, baseline):
        self.gpus, self.baseline = gpus, baseline
        self.peak = dict(baseline)
        self.failure = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            while not self.stop.is_set():
                used = gpu_used_mib(self.gpus)
                for g in self.gpus:
                    self.peak[g] = max(self.peak[g], used[g])
                self.stop.wait(2.0)
        except Exception as error:
            self.failure = str(error)


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save(path, model, optimizer, supervised_optimizer, args, rng, counters):
    temp = path.with_suffix(".tmp")
    torch.save({"format": "gcp_ppo_v1", "policy": model.policy.state_dict(),
                "critic": model.critic.state_dict(), "log_std": model.log_std.detach(),
                "optimizer": optimizer.state_dict(), "supervised_optimizer": supervised_optimizer.state_dict(),
                **counters, "learner_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(model.log_std.device) if model.log_std.is_cuda else None,
                "training_manifest": model.training_manifest,
                "seed": args.seed, "preset": args.preset}, temp)
    temp.replace(path)


def read_resume(path, args):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"policy", "critic", "log_std", "optimizer", "supervised_optimizer", "steps", "updates",
                "bc_steps", "dagger_steps", "teacher_refresh_steps", "last_refresh", "learner_rng",
                "torch_rng", "cuda_rng", "rollout_generation"}
    if (state.get("format") != "gcp_ppo_v1" or state.get("preset") != args.preset
            or not required.issubset(state)):
        raise ValueError("Resume requires a full learner checkpoint from the same preset")
    if args.steps is not None and args.steps <= state["steps"]:
        raise ValueError("The PPO step limit must exceed the resumed PPO step count")
    return state


def restore_learner(state, model, optimizer, supervised_optimizer, rng):
    model.policy.load_state_dict(state["policy"])
    model.critic.load_state_dict(state["critic"])
    with torch.no_grad():
        model.log_std.copy_(state["log_std"])
    optimizer.load_state_dict(state["optimizer"])
    supervised_optimizer.load_state_dict(state["supervised_optimizer"])
    rng.bit_generator.state = state["learner_rng"]
    torch.set_rng_state(state["torch_rng"])
    if model.log_std.is_cuda and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], model.log_std.device)


def load_policy(path, device="cpu"):
    from atrace.gcp import UniversalContactPolicy
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != "gcp_ppo_v1":
        raise ValueError("Unrecognized GCP checkpoint format")
    policy = UniversalContactPolicy().to(device)
    policy.load_state_dict(state["policy"])
    return policy.eval()


def budget(transition_rate, physical_fraction, seconds, rollout, boot_seconds):
    steps = int(transition_rate * max(0., seconds - boot_seconds)) // rollout * rollout
    return {"available_wall_seconds": seconds, "ppo_steps": steps, "updates": steps // rollout,
            "reserved_startup_shutdown_seconds": boot_seconds, "assumed_transitions_per_second": transition_rate,
            "projected_real_20hz_steps": round(steps * physical_fraction),
            "projected_wall_seconds_including_boot_and_checkpoint_io": boot_seconds + steps / transition_rate}


def point_statistics(samples):
    result = {}
    names = ("scene_xyz", "head_scene_xyz", "wrist_scene_xyz", "payload_valid", "scene_heat_above_0.01")
    for kind, rows in sorted(samples.items()):
        counts = np.asarray(rows)
        metrics = {}
        for i, name in enumerate(names):
            col = counts[:, i]
            metrics[name] = {"min": int(col.min()), "p05": float(np.percentile(col, 5)),
                             "mean": float(col.mean()), "median": float(np.median(col)),
                             "p95": float(np.percentile(col, 95)), "max": int(col.max()),
                             "nonempty_fraction": float(np.mean(col > 0))}
        result[kind] = {"sampled_policy_observations": len(rows), **metrics}
    return result


def quantiles(values):
    return {f"p{q:02d}": float(np.percentile(values, q)) for q in (0, 10, 25, 50, 75, 90, 100)}


def arguments(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--preset", choices=("smoke_door_drawer", "all_5_families"))
    selector.add_argument("--benchmark", action="store_true")
    choice, _ = selector.parse_known_args(argv)
    selected = choice.preset or ("all_5_families" if choice.benchmark else "smoke_door_drawer")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=["smoke_door_drawer", "all_5_families"], default=selected)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--gpus", default="0,1,2", help="physical nvidia-smi render GPU indices")
    ap.add_argument("--device", default="cuda:0", help="PyTorch learner device")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--horizon", type=int, default=130)
    ap.add_argument("--rollout", type=int, default=3072, help="real visual transitions per PPO update")
    ap.add_argument("--steps", type=int, help="PPO transition limit; a partial final update is supported")
    ap.add_argument("--minutes", type=float, help="wall budget after boot; complete the current update")
    ap.add_argument("--min-ppo-minutes", type=float, default=0.,
                    help="reserve this much of the timed loop for on-policy batches, including their updates")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bc-epochs", type=int, default=6)
    ap.add_argument("--bc-steps", type=int, help="real teacher steps to warm-start the policy")
    ap.add_argument("--dagger-steps", type=int, default=0, help="optional labelled mixed teacher/student steps before PPO")
    ap.add_argument("--imitation-coef", type=float, default=None,
                    help="auxiliary geometric-teacher loss on student rollouts; disabled in benchmark")
    ap.add_argument("--imitation-floor", type=float, help="lower bound on decayed imitation coefficient")
    ap.add_argument("--balanced-imitation", action=argparse.BooleanOptionalAction, default=False,
                    help="balance teacher action loss across physical phases")
    ap.add_argument("--initial-std", type=float, default=np.exp(-1.), help="Gaussian action noise in unsquashed units")
    ap.add_argument("--gripper-std", type=float, help="optional separate initial aperture noise")
    ap.add_argument("--teacher-refresh-every", type=int, default=0,
                    help="run one additional mixed corrective demonstration batch every N PPO updates")
    ap.add_argument("--smoke-mix", type=float, nargs=3, metavar=("P_DOOR","P_75","P_SLIP75"),
                    help="training-only reset probabilities for the smoke preset")
    ap.add_argument("--reward-mode", choices=("baseline","centered"), default="baseline")
    ap.add_argument("--counterfactual-coef", type=float, default=0.,
                    help="auxiliary paired 40/75 imitation on identical observations near the decision boundary")
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ppo-lr", type=float, help="initial PPO actor LR; defaults to --lr")
    ap.add_argument("--target-kl", type=float, default=0., help="stop a PPO update if sampled policy KL exceeds this bound")
    starting = ap.add_mutually_exclusive_group()
    starting.add_argument("--init", type=Path, help="warm-start actor from a previous GCP checkpoint")
    starting.add_argument("--resume", type=Path, help="continue full learner state at a rollout boundary")
    ap.add_argument("--archive-minutes", type=float, default=30,
                    help="save an immutable intermediate checkpoint every N wall minutes (0 disables)")
    if selected == "smoke_door_drawer" and not choice.benchmark:
        ap.set_defaults(seed=1823, bc_steps=73728, dagger_steps=147456, bc_epochs=4,
                        lr=.00012, ppo_lr=.000012, target_kl=.04, initial_std=.16, gripper_std=.095,
                        balanced_imitation=True, imitation_coef=.8, teacher_refresh_every=10,
                        reward_mode="centered", smoke_mix=(.70,.65,.50), counterfactual_coef=.20, archive_minutes=10)
    elif selected == "all_5_families" and not choice.benchmark:
        ap.set_defaults(workers=72, bc_steps=73728, dagger_steps=110592, bc_epochs=4,
                        lr=1.2e-4, ppo_lr=1.5e-5, target_kl=.04, initial_std=.16, gripper_std=.095,
                        balanced_imitation=True, imitation_coef=.80, imitation_floor=.25,
                        teacher_refresh_every=12, reward_mode="centered", counterfactual_coef=.25,
                        archive_minutes=25, minutes=36*60)
    args = ap.parse_args(argv)
    try:
        parse_gpus(args.gpus)
    except ValueError as err:
        ap.error(str(err))
    if args.preset == "all_5_families" and not 0 <= args.seed < 100_000:
        ap.error("all_5_families training seeds must be in [0, 100000)")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = "step0" if args.benchmark else ("smoke" if args.preset == "smoke_door_drawer" else "all5")
        args.out = ROOT / "runs" / f"gcp_{label}_{stamp}"
    if args.minutes is None:
        args.minutes = 120.0 if args.preset == "smoke_door_drawer" else 48 * 60.0
    if args.imitation_coef is None:
        args.imitation_coef = 0.
    if args.imitation_floor is None:
        args.imitation_floor = min(.35, args.imitation_coef)
    if args.benchmark:
        args.counterfactual_coef = 0.
        args.bc_steps, args.dagger_steps, args.imitation_coef, args.imitation_floor, args.teacher_refresh_every = 0, 0, 0., 0., 0
        if args.steps is None:
            args.steps = 5 * args.rollout
    elif args.bc_steps is None:
        args.bc_steps = 3 * args.rollout if args.preset == "smoke_door_drawer" else 0
    if (min(args.workers, args.rollout, args.epochs, args.minibatch, args.horizon) < 1
            or args.minutes <= 0 or not 0 <= args.min_ppo_minutes <= args.minutes or (args.steps is not None and args.steps <= 0)
            or min(args.bc_steps, args.dagger_steps, args.archive_minutes, args.teacher_refresh_every, args.target_kl) < 0
            or args.bc_epochs <= 0 or not 0 <= args.imitation_floor <= args.imitation_coef
            or args.lr <= 0 or (args.ppo_lr is not None and args.ppo_lr <= 0)
            or args.counterfactual_coef < 0 or (args.smoke_mix is not None and
                (args.preset != "smoke_door_drawer" or not all(0. <= p <= 1. for p in args.smoke_mix)))
            or not 0 < args.initial_std <= 1. or (args.gripper_std is not None and not 0 < args.gripper_std <= 1.)):
        ap.error("Worker/step/time counts must be positive and BC budget/coefficient nonnegative.")
    return args


def main(argv=None):
    start = _MODULE_START if __name__ == "__main__" else time.monotonic()
    args = arguments(argv)
    gpus = list(parse_gpus(args.gpus))
    learner_uuid = configure_learner(args.device)
    resume = read_resume(args.resume, args) if args.resume else None
    args.rollout_generation = 1 + resume["rollout_generation"] if resume else 0
    run = args.out.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise FileExistsError(f"Run already exists: {run}")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["argv"] = list(sys.argv[1:] if argv is None else argv)
    sources = ["contact_contracts.py", "atrace/gcp_joint.py", "atrace/gcp_release.py",
               "lrl_train/gcp_teacher.py", "lrl_train/gcp_devices.py", "lrl_train/gcp_evaluation.py",
               "lrl_train/validate_gcp.py", "lrl_train/eval_gcp_smoke.py", "lrl_train/gcp_pilot_report.py",
               "atrace/gcp.py", "atrace/gcp_labels.py", "atrace/gcp_tracking.py", "atrace/gcp_entry.py",
               "atrace/geometry.py", "atrace/sim_iface.py",
               "atrace/guard.py", "lrl_train/gcp_env.py", "lrl_train/train_gcp.py",
               "lrl_train/fast_sim.py", "lrl_train/_fast_osc.cpp", "lrl_train/render.py",
               "lrl_train/scenes.py", "lrl_train/ucs_scenes.py", "lrl_train/sim.py",
               "lrl_train/student_obs.py"]
    config["source_sha256"] = {}
    config["learner_gpu_uuid"] = learner_uuid
    for name in sources:
        raw = (ROOT / "src" / name).read_bytes()
        config["source_sha256"][name] = hashlib.sha256(raw).hexdigest()
        archived = run / "source" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(raw)
    config["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "mujoco", "robosuite")}
    prior = args.init or args.resume
    config["initial_checkpoint_sha256"] = hashlib.sha256(prior.read_bytes()).hexdigest() if prior else None
    write_json(run / "config.json", config)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    gpu_before = gpu_used_mib(gpus)
    baseline = {g: gpu_before[g] for g in gpus}
    print(f"Starting {args.workers} visual workers on physical GPUs {gpus}: {run}", flush=True)
    pool = Pool(args, run, gpus)
    monitor = MemoryMonitor(gpus, baseline)
    monitor.thread.start()
    try:
        # FastSim may build its native library lazily on the first reset.
        name = "lrl_train/_fast_osc.so"
        binary = (ROOT / "src" / name).read_bytes()
        (run / "source" / name).write_bytes(binary)
        config["source_sha256"][name] = hashlib.sha256(binary).hexdigest()
        config["worker_seeds"] = pool.worker_seeds
        config["worker_physical_gpus"] = pool.rank_gpu
        write_json(run / "config.json", config)
        boot_priming = pool.priming.sum(0).copy()
        device = torch.device(args.device)
        model = Agent().to(device)
        model.training_manifest = {"initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
                                   "source_sha256": config["source_sha256"]}
        with torch.no_grad():
            model.log_std.fill_(np.log(args.initial_std))
            if args.gripper_std is not None:
                model.log_std[-1] = np.log(args.gripper_std)
        if args.init:
            model.policy.load_state_dict(load_policy(args.init, device).state_dict())
        # Keep teacher momentum out of the policy-gradient update (and vice versa).
        actor_lr = args.ppo_lr or args.lr
        optimizer = torch.optim.Adam([
            {"params": [*model.policy.parameters(), model.log_std], "lr": actor_lr},
            {"params": model.critic.parameters(), "lr": args.lr}])
        supervised_optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.lr)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        print(f"All workers ready. Trainable actor parameters: {sum(p.numel() for p in model.policy.parameters())}", flush=True)
        boot_seconds = time.monotonic() - start
        t0 = time.monotonic()
        updates, samples, bc_samples, dagger_samples = 0, 0, 0, 0
        last_refresh, refresh_samples, stop_reason = 0, 0, "configured_budget"
        rng = np.random.default_rng(args.seed)
        if resume is not None:
            restore_learner(resume, model, optimizer, supervised_optimizer, rng)
            updates, samples = resume["updates"], resume["steps"]
            bc_samples, dagger_samples = resume["bc_steps"], resume["dagger_steps"]
            last_refresh, refresh_samples = resume["last_refresh"], resume["teacher_refresh_steps"]
        next_archive = t0 + 60 * args.archive_minutes
        records, recent = [], deque(maxlen=500)
        phase_seconds = Counter()
        with (run / "updates.jsonl").open("x", buffering=1) as stream:
            while ((args.steps is None or samples < args.steps)
                   and (time.monotonic() - t0 < 60 * args.minutes or not records)):
                begin = time.monotonic()
                if updates and (run / "STOP").exists():
                    stop_reason = "requested_stop_after_update"
                    break
                physics_before = pool.physics_steps.sum()
                priming_before, renders_before = pool.priming.sum(0), int(pool.render_count.sum())
                layouts_before, resets_before = pool.layout_rebuilds, pool.reset_worker_seconds
                # Reserve the remaining wall budget at rollout granularity. Count
                # actual PPO cycles, excluding BC, DAgger and teacher refreshes.
                supervised_time = sum(phase_seconds.values()) - phase_seconds["ppo"]
                recent_cycle = max((record["cycle_seconds_including_checkpoint_io"]
                                    for record in records[-3:]), default=0.)
                supervised_allowed = (not args.min_ppo_minutes or
                    supervised_time + max(30., recent_cycle) < 60 * (args.minutes - args.min_ppo_minutes))
                is_bc = supervised_allowed and bc_samples < args.bc_steps
                is_dagger = supervised_allowed and not is_bc and dagger_samples < args.dagger_steps
                refresh = bool(args.teacher_refresh_every and updates and updates % args.teacher_refresh_every == 0
                               and last_refresh != updates and supervised_allowed and not (is_bc or is_dagger))
                supervised = is_bc or is_dagger or refresh
                coef = max(args.imitation_floor, args.imitation_coef / (1.0 + updates / 35.0))
                teacher_probability = 0.
                if is_bc:
                    n = min(args.rollout, args.bc_steps - bc_samples)
                    phase, mode = "teacher_bc", "teacher"
                elif is_dagger:
                    n = min(args.rollout, args.dagger_steps - dagger_samples)
                    phase, mode = "dagger", "mixed"
                    teacher_probability = .5 * (1. - dagger_samples / args.dagger_steps)
                elif refresh:
                    n = args.rollout
                    phase, mode, teacher_probability = "teacher_refresh", "mixed", .5
                else:
                    n = min(args.rollout, args.steps - samples) if args.steps else args.rollout
                    phase, mode = "ppo", ("label" if coef > 0 else "ppo")
                batch, episodes = collect(pool, model, n, device, mode, rng, teacher_probability)
                collected = time.monotonic()
                learner = supervised_optimizer if supervised else optimizer
                used_lr = learner.param_groups[0]["lr"]
                mean_loss = update(model, learner, batch, device, args, coef, supervised)
                model.update_stats["learning_rate"] = used_lr
                if args.target_kl and not supervised:
                    # A very narrow exploration distribution may exceed KL in one
                    # minibatch. Adapt the actor step so later updates see the rollout.
                    stats = model.update_stats
                    minibatches = max(1, int(np.ceil(n / args.minibatch)))
                    if stats["stopped_on_kl"] and stats["gradient_steps"] < minibatches:
                        optimizer.param_groups[0]["lr"] = max(actor_lr * .025, used_lr * .65)
                    elif not stats["stopped_on_kl"] and stats["mean_sample_kl"] < args.target_kl * .3:
                        optimizer.param_groups[0]["lr"] = min(actor_lr, used_lr * 1.12)
                finished = time.monotonic()
                bc_samples += n if is_bc else 0
                dagger_samples += n if is_dagger else 0
                if refresh:
                    refresh_samples += n
                    last_refresh = updates
                samples += 0 if supervised else n
                updates += int(not supervised)
                counters = dict(steps=samples, updates=updates, bc_steps=bc_samples, dagger_steps=dagger_samples,
                                last_refresh=last_refresh, teacher_refresh_steps=refresh_samples,
                                rollout_generation=args.rollout_generation)
                save(run / "policy.pt", model, optimizer, supervised_optimizer, args, rng, counters)
                if args.archive_minutes and finished >= next_archive and (args.preset == "all_5_families" or not supervised):
                    serial = samples + bc_samples + dagger_samples + refresh_samples if args.preset == "all_5_families" else samples
                    os.link(run / "policy.pt", run / f"policy_step{serial:09d}.pt")
                    next_archive = finished + 60 * args.archive_minutes
                recent.extend(episodes)
                record = {"phase": phase,
                          "steps": n, "ppo_total": samples, "dagger_total": dagger_samples,
                          "refresh_total": refresh_samples, "priming_steps": int(pool.priming[:, 0].sum()),
                          "paired_goal_labels_total": pool.paired_goals,
                          "paired_goal_labels_by_kind": dict(pool.paired_kinds),
                          "real_steps_by_kind_total": dict(pool.kind_steps),
                          "verified_dual_renders_total": pool.step_renders,
                          "prefix_delta": (pool.priming.sum(0)-priming_before).tolist(),
                          "dual_renders_in_resets": int(pool.render_count.sum()) - renders_before - n,
                          "layout_rebuilds": pool.layout_rebuilds - layouts_before,
                          "sum_worker_reset_seconds": pool.reset_worker_seconds - resets_before,
                          "real_20hz_steps": int(pool.physics_steps.sum() - physics_before),
                          "bc_total": bc_samples, "ppo_update": updates, "sample_seconds": collected - begin,
                          "update_seconds": finished - collected, "loop_seconds": finished - begin,
                          "cycle_seconds_including_checkpoint_io": time.monotonic() - begin,
                          "steps_per_s": n / (finished - begin), "loss": mean_loss,
                          "optimizer": model.update_stats,
                          "episode_count": len(episodes), "training_window_n": len(recent),
                          "training_window_success": np.mean([bool(e["success"]) for e in recent]).item() if recent else None,
                          "training_window_by_kind": {
                              kind: {"n": len(group), "success": sum(bool(e["success"]) for e in group),
                                     "entered": sum(bool(e["entry_ok"]) for e in group),
                                     "staged": sum(bool(e["staged"]) for e in group),
                                     "fresh_success": sum(bool(e["success"] and not e["staged"]) for e in group),
                                     "staged_success": sum(bool(e["success"] and e["staged"]) for e in group),
                                     "staged_sources": dict(Counter(e.get("staged_origin") for e in group if e["staged"]))}
                              for kind in sorted({str(e["kind"]) for e in recent})
                              for group in ([e for e in recent if str(e["kind"]) == kind],)},
                          "episode_kinds": dict(Counter(str(e["kind"]) for e in episodes))}
                for name, predicate in (("door40",lambda e:e["kind"]=="door" and e["goal_deg"]<60),
                                        ("door75",lambda e:e["kind"]=="door" and e["goal_deg"]>=60),
                                        ("slip75_triggered",lambda e:e["kind"]=="door" and e["goal_deg"]>=60
                                         and bool(e["slip_injected"]))):
                    group = [e for e in recent if predicate(e)]
                    record.setdefault("training_goal_windows", {})[name] = {
                        "n": len(group), "success": sum(bool(e["success"]) for e in group),
                        "regrasp": sum(bool(e["regrasped"]) for e in group)}
                record["reward_counts"] = dict(sum((Counter(e["reward_counts"] or {}) for e in episodes), Counter()))
                phase_seconds[phase] += time.monotonic() - begin
                record["phase_wall_seconds"] = dict(phase_seconds)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps(record), flush=True)
                if not supervised:
                    records.append(record)
        measured = records[1:] if len(records) > 1 else records
        elapsed = sum(r["loop_seconds"] for r in measured)
        cycle_elapsed = sum(r["cycle_seconds_including_checkpoint_io"] for r in measured)
        timed_transitions = sum(r["steps"] for r in measured)
        step_count = sum(r["real_20hz_steps"] for r in measured)
        if not step_count:
            raise RuntimeError("No dual-rendered transitions advanced the simulator by 50 ms")
        rate = step_count / elapsed
        window = min(10, len(measured))
        windows = [measured[i:i+window] for i in range(len(measured)-window+1)]
        physical_rolling = [sum(r["real_20hz_steps"] for r in group) /
                            sum(r["cycle_seconds_including_checkpoint_io"] for r in group) for group in windows]
        transition_rolling = [sum(r["steps"] for r in group) /
                              sum(r["cycle_seconds_including_checkpoint_io"] for r in group) for group in windows]
        report = {"preset": args.preset, "workers": args.workers, "render_gpus": gpus,
                  "phase_wall_seconds": dict(phase_seconds), "min_ppo_minutes": args.min_ppo_minutes,
                  "learner_device": str(device), "total_ppo_steps": samples,
                  "teacher_steps": bc_samples, "dagger_steps": dagger_samples,
                  "teacher_refresh_steps": refresh_samples, "stop_reason": stop_reason,
                  "priming": {k: float(v) for k,v in zip(("physics_steps", "dual_renders", "sum_worker_seconds",
                                                         "attempts", "successes"), pool.priming.sum(0))},
                  "priming_at_boot": dict(zip(("physics_steps", "dual_renders", "sum_worker_seconds",
                                             "attempts", "successes"),boot_priming.tolist())),
                  "timed_priming": dict(zip(("physics_steps", "dual_renders", "sum_worker_seconds",
                                            "attempts", "successes"),np.sum([r["prefix_delta"] for r in measured],axis=0).tolist())),
                  "timed_dual_renders_in_resets_including_priming": sum(r["dual_renders_in_resets"] for r in measured),
                  "timed_layout_rebuilds_after_initial_boot": sum(r["layout_rebuilds"] for r in measured),
                  "timed_resets": sum(r["episode_count"] for r in measured),
                  "timed_sum_worker_reset_s_including_priming": sum(r["sum_worker_reset_seconds"] for r in measured),
                  "ppo_updates": updates, "timed_updates": len(measured), "timed_steps": sum(r["steps"] for r in measured),
                  "timed_real_20hz_steps": step_count, "measured_steps_per_s": rate,
                  "ppo_transitions_per_wall_s": timed_transitions / cycle_elapsed,
                  "timed_full_cycles_physical_sps_quantiles": quantiles([
                      r["real_20hz_steps"] / r["cycle_seconds_including_checkpoint_io"] for r in measured]),
                  "rolling_full_cycles": {"window_updates": window, "windows": len(windows),
                      "real_20hz_sps_quantiles": quantiles(physical_rolling), "last_real_20hz_sps": physical_rolling[-1],
                      "ppo_transitions_sps_quantiles": quantiles(transition_rolling)},
                  "per_update_steps_per_s": [r["real_20hz_steps"] / r["loop_seconds"] for r in measured],
                  "boot_seconds": boot_seconds, "loop_wall_seconds": time.monotonic() - t0,
                  "timed_collection_seconds": sum(r["sample_seconds"] for r in measured),
                  "timed_ppo_seconds": sum(r["update_seconds"] for r in measured),
                  "per_worker_steps": pool.steps.tolist(), "each_worker_physical_gpu": pool.rank_gpu,
                  "per_worker_real_20hz_steps": pool.physics_steps.tolist(),
                  "real_20hz_steps_by_scene_kind": dict(pool.kind_steps),
                  "visual_point_counts_per_kind": point_statistics(pool.visual_points),
                  "visual_sampling": "One acted-on observation per 64 verified 20 Hz steps per worker. XYZ=nonzero metric coordinates, payload=valid mask>0, heat threshold=0.01.",
                  "verified_dual_renders_on_steps": pool.step_renders, "dual_renders_including_resets": int(pool.render_count.sum()),
                  "camera_size": [96, 96], "actor_obs_shapes": OBS_SHAPES, "memory_sampling_period_s": 2,
                  "device_memory_baseline_MiB": baseline, "device_memory_sampled_peak_MiB": monitor.peak,
                  "device_memory_sampling_error": monitor.failure,
                  "torch_max_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                  "torch_max_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
                  "boundary": "Excludes initial worker boot, first PPO update (when multiple), checkpoint I/O; includes resets, EGL, policy inference and PPO.",
                  "budget_basis": "Timed-update rates, including resets and checkpoint I/O; reserve measured boot + teardown, assume identical PPO and no teacher warmup, round down to whole updates. CLI transitions include failed entries."}
        before_teardown = time.monotonic()
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=12)
        pool.close()
    end = time.monotonic()
    teardown = end - before_teardown
    report["worker_teardown_seconds"] = teardown
    report["whole_run_wall_seconds_including_boot_and_teardown"] = end - start
    report["whole_run_real_20hz_steps_per_s_including_boot_and_teardown"] = float(pool.physics_steps.sum()) / (end-start)
    report["wall_budgets"] = {label: budget(timed_transitions / cycle_elapsed, step_count / timed_transitions,
                                    hours*3600, args.rollout, boot_seconds+teardown) for label, hours in
                             (("smoke_2h",2),("all5_36h",36),("all5_48h",48))}
    conservative = float(np.percentile(transition_rolling, 10))
    report["conservative_wall_budgets_at_rolling_p10"] = {
        label: budget(conservative,step_count/timed_transitions,hours*3600,args.rollout,boot_seconds+teardown)
        for label,hours in (("all5_36h",36),("all5_48h",48))}
    write_json(run / ("throughput.json" if args.benchmark else "training.json"), report)
    print(f"Finished {updates} PPO updates, {samples} PPO steps at {rate:.1f} measured steps/s", flush=True)


if __name__ == "__main__":
    main()
