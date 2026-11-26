"""Behaviour-cloning warm start of the M1E teacher (e1 step 2, accepted by the user as a training-time tool: the
deployed student stays fully learned, and PPO fine-tuning may leave the script wherever it can do better).

collect  The privileged expert (expert.py) drives TeacherEnv episodes over the full start distribution (curriculum
         scale drawn per episode, so near and grasped starts and the full hand-off error both occur). DART-style
         perturbation: the executed motion numbers are the expert's plus Gaussian noise of a per-episode size, the
         label is the expert's clean action at the state reached, so the clone sees recovery states. Kept: episodes
         that reach the goal, and injected-failure episodes that end with done short of it (giving up is right
         there). Per kept step: the teacher's observation, the critic's input, the label, the reward of the executed
         step, and the episode's discounted return from that step (Monte Carlo, the env's own reward).
train    The actor's mean is fitted to the labels (MSE; the done number to -2 off a done step and +1 on one, those
         weighted up: done is one step in a hundred), the critic to the returns; observation statistics are set from
         the data first. Written as a ppo checkpoint, so train_teacher --init fine-tunes it with PPO and
         eval_teacher evaluates it.

  python -m lrl_train.bc collect --episodes 12000 --procs 80 --out <dir>
  python -m lrl_train.bc train --data <dir> --out <ckpt.pt> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from atrace.lrl_e2e import ACT_DIM

from .teacher_env import CFG as ENV, CRIT_DIM, OBS_DIM, REWARD, TeacherEnv

CFG = {
    "noise": (0.0, 0.1, 0.2, 0.3),  # T  std of the noise on the six motion numbers, one per episode (DART)
    "scale": (0.0, 1.0),        # T  curriculum scale drawn uniformly per episode
    "shard": 20000,             # T  steps per file
    "min_giveup": 5,            # T  steps: an injected episode's give-up is kept only after this many
    "gamma": 0.99,              # T  (ppo.CFG)
    # training
    "hidden": 512, "layers": 3,     # T  train_teacher.CFG
    "std0": 0.12,               # T  the clone's exploration for fine-tuning: small, so that the first PPO
                                #    updates start from the clone's behaviour, not from noise around it
    "done_off": -2.0,           # T  done number's target off a done step (with std 0.3: never raised by chance)
    "done_on": 1.0,             # T
    "done_weight": 20.0,        # T  weight of done steps in the done loss
    "epochs": 30, "batch": 4096, "lr": 1e-3, "lr_end": 0.05,    # T
    "val_share": 0.05,          # T  episodes held out for the loss curve
}


def collect_job(job):
    """Expert episodes in one process: writes shards of kept steps to out/shard_<seed>_<k>.npz; returns stats."""
    from .expert import Expert
    seed, n, out, kinds = job
    out = Path(out)
    rng = np.random.default_rng([17, seed])
    env = TeacherEnv([17, seed], out / "bddl" / f"w{seed}", ENV | ({"kinds": tuple(kinds)} if kinds else {}))
    ex = Expert(env)
    buf = {k: [] for k in ("obs", "crit", "act", "rew", "ret", "first")}
    meta, stats, k_shard = [], {"episodes": 0, "kept": 0, "steps": 0, "kept_steps": 0}, 0

    def flush():
        nonlocal k_shard
        if not buf["obs"]:
            return
        np.savez(out / f"shard_{seed:04d}_{k_shard:03d}.npz",
                 **{k: np.concatenate(v).astype(np.float32 if k != "first" else bool) for k, v in buf.items()})
        for v in buf.values():
            v.clear()
        k_shard += 1

    for _ in range(n):
        obs, crit, info0 = env.reset(float(rng.uniform(*CFG["scale"])))
        ex.start()
        sigma = float(rng.choice(CFG["noise"]))
        ep = {k: [] for k in ("obs", "crit", "act", "rew")}
        while True:
            a = ex()
            a_run = a.copy()
            a_run[:6] = np.clip(a[:6] + sigma * rng.standard_normal(6), -1.0, 1.0)
            ep["obs"].append(obs), ep["crit"].append(crit), ep["act"].append(a)
            obs, crit, r, ended, info = env.step(a_run)
            ep["rew"].append(r)
            if ended:
                break
        stats["episodes"] += 1
        stats["steps"] += len(ep["rew"])
        # a give-up is kept only after an attempt: one at the start (no contact found) cannot be told from a start
        # that can be done, since the injection is hidden from the actor (bc1 learnt to raise done at step 0)
        keep = info["success"] or (info["inject"] and info["stop"] == "done" and info["steps"] > CFG["min_giveup"])
        meta.append({k: info[k] for k in ("kind", "effect", "handle", "near", "wrong", "inject", "success", "stop",
                                          "label", "steps", "scale")} | {"sigma": sigma, "kept": bool(keep),
                                                                         "cand": ex.cand.name if ex.cand else ""})
        if not keep:
            continue
        rew = np.array(ep["rew"])
        ret = np.zeros_like(rew)
        g = 0.0
        for t in range(len(rew) - 1, -1, -1):
            g = rew[t] + CFG["gamma"] * g
            ret[t] = g
        first = np.zeros(len(rew), bool)
        first[0] = True
        for k in ("obs", "crit", "act"):
            buf[k].append(np.array(ep[k]))
        buf["rew"].append(rew), buf["ret"].append(ret), buf["first"].append(first)
        stats["kept"] += 1
        stats["kept_steps"] += len(rew)
        if sum(len(x) for x in buf["rew"]) >= CFG["shard"]:
            flush()
    flush()
    env.close()
    return stats, meta


def collect(a):
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.seed0 + i, a.per_job, str(a.out), a.kinds) for i in range(-(-a.episodes // a.per_job))]
    t0, tot, metas = time.time(), {}, []
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as pool, open(a.out / "episodes.jsonl", "w") as f:
        for i, (stats, meta) in enumerate(pool.imap_unordered(collect_job, jobs)):
            for k, v in stats.items():
                tot[k] = tot.get(k, 0) + v
            metas += meta
            f.writelines(json.dumps(m, default=str) + "\n" for m in meta)
            f.flush()
            print(f"job {i + 1}/{len(jobs)} {tot} {(time.time() - t0) / 60:.1f} min", flush=True)
    (a.out / "collect.json").write_text(json.dumps(tot | {"minutes": (time.time() - t0) / 60, "cfg": CFG},
                                                   indent=1, default=str))


def load(data):
    """Every shard of one or more collection directories (a list, or one path), without the episodes that end
    within min_giveup steps (give-ups before any attempt; data1 was collected before the keep rule had that)."""
    files = sorted(f for d in ([data] if isinstance(data, (str, Path)) else data) for f in Path(d).glob("shard_*.npz"))
    parts = [dict(np.load(f)) for f in files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    first = np.flatnonzero(d["first"])
    lens = np.diff(np.r_[first, len(d["first"])])
    keep = np.repeat(lens > CFG["min_giveup"], lens)
    return {k: v[keep] for k, v in d.items()}


def train(a):
    import torch
    from . import ppo
    from .train_teacher import CFG as TRAIN
    torch.manual_seed(0)
    d = load(a.data)
    n = len(d["obs"])
    ep = np.cumsum(d["first"]) - 1                     # episode index of every step
    rng = np.random.default_rng(0)
    val_ep = rng.random(ep.max() + 1) < CFG["val_share"]
    val = val_ep[ep]
    print(f"{n} steps, {ep.max() + 1} episodes, {val.sum()} validation steps", flush=True)
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM, ppo.CFG | {
        "std0": CFG["std0"], "mean0": None, "lr": TRAIN["lr"], "minibatch": TRAIN["minibatch"],
        "hidden": CFG["hidden"], "layers": CFG["layers"]})
    model.obs_norm.update(torch.as_tensor(d["obs"][~val]))
    model.crit_norm.update(torch.as_tensor(d["crit"][~val]))
    dev = torch.device(a.device)
    model.to(dev)
    obs, crit = torch.as_tensor(d["obs"], device=dev), torch.as_tensor(d["crit"], device=dev)
    act, ret = torch.as_tensor(d["act"], device=dev), torch.as_tensor(d["ret"], device=dev)
    done = act[:, -1] > 0
    target = act.clone()
    target[:, -1] = torch.where(done, CFG["done_on"], CFG["done_off"])
    w_done = torch.where(done, CFG["done_weight"], 1.0)
    params = list(model.actor.parameters()) + list(model.critic.parameters())
    opt = torch.optim.Adam(params, lr=CFG["lr"])
    tr_idx = torch.as_tensor(np.flatnonzero(~val), device=dev)
    va_idx = torch.as_tensor(np.flatnonzero(val), device=dev)
    steps = max(1, CFG["epochs"] * (len(tr_idx) // CFG["batch"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 - (1.0 - CFG["lr_end"]) * min(1.0, s / steps))

    def losses(idx):
        mean = model.actor(model.obs_norm(obs[idx]))
        a_loss = ((mean[:, :-1] - target[idx, :-1]) ** 2).mean()
        d_loss = (w_done[idx] * (mean[:, -1] - target[idx, -1]) ** 2).mean()
        v_loss = ((model.value(obs[idx], crit[idx]) - ret[idx]) ** 2).mean()
        return a_loss, d_loss, v_loss

    log = []
    for epoch in range(CFG["epochs"]):
        model.train()
        for idx in tr_idx[torch.randperm(len(tr_idx), device=dev)].split(CFG["batch"]):
            a_loss, d_loss, v_loss = losses(idx)
            opt.zero_grad()
            (a_loss + d_loss + 0.5 * v_loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            tr = [x.item() for x in losses(tr_idx[:200000])]
            va = [x.item() for x in losses(va_idx)]
            # how often the clone's done would fire (P = Phi(mean / std)) off and on done steps of held-out episodes
            m = model.actor(model.obs_norm(obs[va_idx]))[:, -1]
            p = 0.5 * (1 + torch.erf(m / model.log_std[-1].exp() / 2 ** 0.5))
            dv = done[va_idx]
        row = {"epoch": epoch + 1, "train": tr, "val": va, "p_done_off": p[~dv].mean().item(),
               "p_done_on": p[dv].mean().item() if dv.any() else None}
        log.append(row)
        print(json.dumps(row), flush=True)
    model.cpu()
    ppo.save(a.out, model, {"bc": str(a.data), "steps": 0, "updates": 0, "scale": 1.0, "bc_steps": int(n),
                            "bc_cfg": CFG, "bc_log": log, "command": "lrl_train.bc train"})
    print("saved", a.out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("collect", "train"))
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--per-job", type=int, default=150)
    ap.add_argument("--procs", type=int, default=80)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kinds", nargs="+", help="collect: only these kinds (a second collection that evens the mix)")
    ap.add_argument("--seed0", type=int, default=0, help="collect: first job's seed (shard names, rng)")
    a = ap.parse_args()
    collect(a) if a.what == "collect" else train(a)


if __name__ == "__main__":
    main()
