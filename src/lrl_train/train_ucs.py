"""Asymmetric PPO trainer for Unified Contact Stage (`pi_exec`) + Outcome Detector (`D_out`).

Uses `VecEnv` over CPU processes (`FastSimRobot` C++ 500 Hz OSC) + PyTorch PPO on GPU (`cuda:0`).
Note: `VecEnv` (`fork`) is ALWAYS created BEFORE moving `ActorCritic` to CUDA so forked workers remain CUDA-clean.

Usage:
  cd /workspace/agentic_trace/src && CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
    ../.venv/bin/python -W ignore -m lrl_train.train_ucs --name g1_exec --steps 40000000 --envs 144 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from atrace.ucs_skill import EXEC_ACT_DIM, EXEC_MODES, EXEC_OBS_DIM, EXEC_STEP_DIM

from . import ppo
from .ucs_env import ENV_CFG, REWARD_CFG, UCS_CRIT_DIM, UCS_TRAIN_KINDS, UCSExecEnv
from .vec import VecEnv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs" / "ucs_g1_20261127"

TRAIN_CFG = {
    "hidden": 320,
    "layers": 3,
    "std0": (0.28,) * 6 + (0.25,),   # residual pos/ori + compliance alpha
    "mean0": (0.0,) * 6 + (0.45,),   # start alpha near ~0.81
    "done_mean0": 0.75,              # action[7] is release_gate (+0.75 = hold firmly until support gate or learned release!)
    "done_std0": 0.28,
    "obs_clip": 10.0,
    "gamma": 0.99,
    "lam": 0.95,
    "clip": 0.2,
    "epochs": 5,
    "minibatch": 4096,
    "lr": 3e-4,
    "lr_end": 0.25,
    "vf_coef": 0.5,
    "ent_coef": 0.002,
    "max_grad": 0.5,
    "target_kl": 0.03,
    # Sampling-Based Curriculum (SBC): uniform [s_low, 1.0], s_low advances from 0.0 -> 0.85 as success > 0.78
    "s_low_init": 0.0,
    "sbc_up_thresh": 0.78,
    "sbc_down_thresh": 0.55,
    "sbc_step": 0.04,
}

DET_CLASSES = ("reached_goal", "progress_short", "slipped_or_missed", "blocked_obstacle")


def _rows(arrays, idx):
    return np.stack([arrays[i] for i in idx])


def collect_rollout(vec: VecEnv, model: ppo.ActorCritic, out, target_steps: int, rng: np.random.Generator):
    c = model.cfg
    traj = [{k: [] for k in ("obs", "crit", "act", "logp", "val", "rew", "end")} for _ in range(vec.n)]
    episodes = []
    count = 0
    actions = np.zeros((vec.n, EXEC_ACT_DIM), dtype=np.float32)

    def book(step_out):
        for i in np.flatnonzero(step_out.stepped):
            traj[i]["rew"].append(step_out.reward[i])
            traj[i]["end"].append(float(step_out.ended[i]))
            if step_out.ended[i]:
                episodes.append(step_out.info[i])
        return int(step_out.stepped.sum())

    while count < target_steps:
        live = np.flatnonzero(out.live)
        if len(live):
            noise = rng.standard_normal((len(live), EXEC_ACT_DIM))
            a, logp, val = model.act(_rows(out.obs, live), _rows(out.crit, live), noise)
            actions[live] = a
            for j, i in enumerate(live):
                t = traj[i]
                t["obs"].append(out.obs[i])
                t["crit"].append(out.crit[i])
                t["act"].append(a[j])
                t["logp"].append(logp[j])
                t["val"].append(val[j])
        out = vec.step(actions)
        count += book(out)

    for s_out in vec.settle() or [out]:
        book(s_out)
        out = s_out

    last_val = np.zeros(vec.n)
    live = np.flatnonzero(out.live)
    if len(live):
        last_val[live] = model.act(_rows(out.obs, live), _rows(out.crit, live))[2]

    batch = {k: [] for k in ("obs", "crit", "act", "logp", "adv", "ret")}
    for i, t in enumerate(traj):
        if not t["rew"]:
            continue
        adv, ret = ppo.gae(t["rew"], np.asarray(t["val"]), t["end"], last_val[i], c["gamma"], c["lam"])
        for k, v in zip(batch, (t["obs"], t["crit"], t["act"], t["logp"], adv, ret)):
            batch[k].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in batch.items()}, episodes, out


class UCSOutcomeDetector(nn.Module):
    """4-layer 1D-CNN outcome detector with temperature scaling over 20 Hz UCSExecSkill traces."""

    def __init__(self, step_dim: int = EXEC_STEP_DIM, hidden: int = 128, n_cls: int = len(DET_CLASSES)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(step_dim, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + step_dim + len(EXEC_MODES) + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_cls),
        )
        self.log_temp = nn.Parameter(torch.zeros(1))

    def logits(self, x: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), mask: (B, T) bool, meta: (B, len(EXEC_MODES)+1)
        h = x.transpose(1, 2)
        keep = mask[:, None, :].float()
        for layer in self.conv:
            h = layer(h) * keep
        denom = keep.sum(-1).clamp(min=1.0)
        mean_pool = h.sum(-1) / denom
        max_pool = h.masked_fill(~mask[:, None, :], -1e4).amax(-1)
        # Extract last valid step per row
        lengths = mask.long().sum(-1).clamp(min=1) - 1
        last = x[torch.arange(len(x), device=x.device), lengths]
        return self.head(torch.cat([mean_pool, max_pool, last, meta], dim=-1))

    def predict_proba(self, x: torch.Tensor, mask: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(x, mask, meta) / self.log_temp.exp().clamp(min=0.2, max=5.0), dim=-1)


def train_detector_from_dumps(dump_dir: Path, out_ckpt: Path, device: str = "cuda:0") -> dict:
    files = sorted(Path(dump_dir).glob("ucs_trace_*.pkl"))
    rows = []
    for fp in files:
        try:
            with open(fp, "rb") as f:
                rows.extend(pickle.load(f))
        except Exception:
            continue
    if len(rows) < 200:
        return {"trained": False, "n_samples": len(rows)}

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(rows))
    n_val = max(100, int(0.15 * len(rows)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    max_T = 140
    def make_tensors(sel):
        N = len(sel)
        X = np.zeros((N, max_T, EXEC_STEP_DIM), dtype=np.float32)
        M = np.zeros((N, max_T), dtype=bool)
        G = np.zeros((N, len(EXEC_MODES) + 1), dtype=np.float32)
        Y = np.zeros(N, dtype=np.int64)
        for j, k in enumerate(sel):
            r = rows[k]
            tr = r["trace"][-max_T:]
            L = len(tr)
            X[j, :L] = tr
            M[j, :L] = True
            G[j, :len(EXEC_MODES)] = (np.array(EXEC_MODES) == r["mode"]).astype(np.float32)
            G[j, -1] = r["target_span"]
            Y[j] = DET_CLASSES.index(r["label"])
        return (
            torch.as_tensor(X, device=device),
            torch.as_tensor(M, device=device),
            torch.as_tensor(G, device=device),
            torch.as_tensor(Y, device=device),
        )

    X_tr, M_tr, G_tr, Y_tr = make_tensors(tr_idx)
    X_va, M_va, G_va, Y_va = make_tensors(val_idx)

    det = UCSOutcomeDetector().to(device)
    opt = torch.optim.AdamW([p for n, p in det.named_parameters() if n != "log_temp"], lr=1e-3, weight_decay=1e-4)
    bs = 256
    for _epoch in range(25):
        perm = torch.randperm(len(X_tr), device=device)
        for chunk in perm.split(bs):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(det.logits(X_tr[chunk], M_tr[chunk], G_tr[chunk]), Y_tr[chunk])
            loss.backward()
            opt.step()

    # Temperature calibration on validation split (minimizes NLL!)
    t_opt = torch.optim.LBFGS([det.log_temp], lr=0.1, max_iter=30)
    with torch.no_grad():
        val_logits = det.logits(X_va, M_va, G_va)

    def _closure():
        t_opt.zero_grad()
        loss_t = nn.functional.cross_entropy(val_logits / det.log_temp.exp().clamp(0.2, 5.0), Y_va)
        loss_t.backward()
        return loss_t

    t_opt.step(_closure)

    with torch.no_grad():
        probs = det.predict_proba(X_va, M_va, G_va)
        preds = probs.argmax(dim=-1)
        val_acc = float((preds == Y_va).float().mean().item())
        reached_mask = preds == 0
        reached_prec = float((Y_va[reached_mask] == 0).float().mean().item()) if int(reached_mask.sum()) > 0 else 0.0

    torch.save({
        "format": "ucs_outcome_detector",
        "classes": DET_CLASSES,
        "state_dict": {k: v.cpu() for k, v in det.state_dict().items()},
        "temperature": float(det.log_temp.exp().item()),
        "val_acc": round(val_acc, 4),
        "val_reached_precision": round(reached_prec, 4),
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
    }, out_ckpt)
    return {
        "trained": True,
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
        "val_acc": round(val_acc, 4),
        "val_reached_precision": round(reached_prec, 4),
        "temperature": round(float(det.log_temp.exp().item()), 4),
        "ckpt": str(out_ckpt),
    }


def load_ucs_actor(ckpt_path: str | Path):
    """Deterministic inference callable `obs (EXEC_OBS_DIM,) -> action (EXEC_ACT_DIM,)` for evaluation & deployment."""
    model, meta = ppo.load(ckpt_path)
    model.eval()

    @torch.no_grad()
    def policy_fn(obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float64)[None]
        a = model.actor(model.obs_norm(x))[0].numpy().astype(float)
        return np.clip(a, -1.0, 1.0)

    policy_fn.meta = meta
    return policy_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="exec_v1")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--envs", type=int, default=144)
    ap.add_argument("--rollout", type=int, default=32768)
    ap.add_argument("--steps", type=float, default=30e6)
    ap.add_argument("--minutes", type=float, default=50.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--init", default=None, help="Optional checkpoint to warm-start ActorCritic from")
    ap.add_argument("--no-dump", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = None if args.no_dump else (run_dir / "dump")

    # IMPORTANT: fork VecEnv BEFORE touching CUDA!
    t_boot = time.time()
    vec = VecEnv(
        n=args.envs,
        out_dir=run_dir / "workers",
        seed=args.seed,
        cfg=ENV_CFG,
        reward=REWARD_CFG,
        randomize=True,
        dump_dir=dump_dir,
        env_cls=UCSExecEnv,
        dims=(EXEC_OBS_DIM, UCS_CRIT_DIM),
    )

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.init:
        model, _ = ppo.load(args.init)
        model.train().to(args.device)
        print(f"[UCS Train] Warm-started ActorCritic from {args.init}", flush=True)
    else:
        model = ppo.ActorCritic(EXEC_OBS_DIM, UCS_CRIT_DIM, EXEC_ACT_DIM, cfg=TRAIN_CFG).to(args.device)
    learner = ppo.PPO(model, cfg=TRAIN_CFG)

    s_low = TRAIN_CFG["s_low_init"]
    out = vec.reset(scale=s_low)
    print(f"[UCS Train] {args.envs} workers booted & reset in {time.time() - t_boot:.1f}s. Starting PPO...", flush=True)

    csv_file = open(run_dir / "train.csv", "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "update", "total_steps", "elapsed_s", "steps_per_s", "s_low",
        "ep_count", "succ_all", "succ_pull", "succ_push", "succ_knob", "succ_cavity", "succ_overhang",
        "max_pen_p95_mm", "pi_loss", "v_loss", "kl", "explained",
    ])

    mode_hist = {m: deque(maxlen=400) for m in EXEC_MODES}
    kind_hist = {k: deque(maxlen=400) for k in UCS_TRAIN_KINDS}
    pen_hist = deque(maxlen=1000)

    total_steps = 0
    update = 0
    best_score = -1.0
    t_start = time.time()

    try:
        while total_steps < args.steps and (time.time() - t_start) < args.minutes * 60.0:
            if (run_dir / "STOP").exists():
                print("[UCS Train] STOP marker detected, finishing cleanly.", flush=True)
                break

            t_u0 = time.time()
            batch, eps, out = collect_rollout(vec, model, out, int(args.rollout), rng)
            n_new = len(batch["obs"])
            total_steps += n_new
            update += 1

            progress = min(1.0, max(total_steps / max(1.0, args.steps), (time.time() - t_start) / (args.minutes * 60.0)))
            stats = learner.update(
                batch["obs"], batch["crit"], batch["act"], batch["logp"], batch["adv"], batch["ret"], progress=progress
            )

            for e in eps:
                mode_hist[e["mode"]].append(float(e["success"]))
                kind_hist[e["kind"]].append(float(e["success"]))
                pen_hist.append(float(e["max_pen_mm"]))

            mode_rates = {m: (float(np.mean(h)) if len(h) else 0.0) for m, h in mode_hist.items()}
            succ_all = float(np.mean([v for h in mode_hist.values() for v in h])) if any(mode_hist.values()) else 0.0
            pen_p95 = float(np.percentile(pen_hist, 95)) if len(pen_hist) else 0.0

            # Sampling-Based Curriculum (SBC) update & hard-kind adaptive sampling
            if sum(len(h) for h in mode_hist.values()) >= 200:
                if succ_all >= TRAIN_CFG["sbc_up_thresh"]:
                    s_low = min(0.85, s_low + TRAIN_CFG["sbc_step"])
                elif succ_all < TRAIN_CFG["sbc_down_thresh"]:
                    s_low = max(0.0, s_low - 0.5 * TRAIN_CFG["sbc_step"])
                vec.scale = s_low
                k_weights = {
                    k: 0.4 + 0.6 * (1.0 - (float(np.mean(kind_hist[k])) if len(kind_hist[k]) >= 20 else 0.5))
                    for k in UCS_TRAIN_KINDS
                }
                vec.weights = k_weights

            dt_u = time.time() - t_u0
            sps = n_new / max(dt_u, 1e-6)
            el = time.time() - t_start
            writer.writerow([
                update, total_steps, round(el, 1), round(sps, 1), round(s_low, 3),
                len(eps), round(succ_all, 4),
                round(mode_rates["pull_open"], 4), round(mode_rates["push_close"], 4),
                round(mode_rates["knob_turn"], 4), round(mode_rates["cavity_insert"], 4),
                round(mode_rates["overhang_place"], 4),
                round(pen_p95, 3),
                round(stats.get("pi_loss", 0.0), 5), round(stats.get("v_loss", 0.0), 5),
                round(stats.get("kl", 0.0), 5), round(stats.get("explained", 0.0), 4),
            ])
            csv_file.flush()

            meta = {
                "modes": EXEC_MODES,
                "update": update,
                "total_steps": total_steps,
                "s_low": s_low,
                "mode_rates": mode_rates,
                "pen_p95_mm": pen_p95,
            }
            ppo.save(run_dir / "ckpt_latest.pt", model, meta=meta)
            min_mode = min(mode_rates.values()) if all(len(h) >= 30 for h in mode_hist.values()) else -1.0
            score = 0.6 * succ_all + 0.4 * max(0.0, min_mode) + 0.05 * s_low
            if min_mode >= 0.0 and score >= best_score:
                best_score = score
                ppo.save(run_dir / "ckpt_best.pt", model, meta=meta)

            if update == 1 or update % 5 == 0:
                print(
                    f"[upd {update:03d} | {total_steps/1e6:5.2f}M | {sps:6.0f} st/s | {el/60:4.1f}m] "
                    f"s_low={s_low:.2f} succ={succ_all:.3f} "
                    f"(pull={mode_rates['pull_open']:.2f} push={mode_rates['push_close']:.2f} "
                    f"knob={mode_rates['knob_turn']:.2f} cav={mode_rates['cavity_insert']:.2f} "
                    f"oh={mode_rates['overhang_place']:.2f}) pen95={pen_p95:.2f}mm exp={stats.get('explained',0):.2f}",
                    flush=True,
                )
    finally:
        csv_file.close()
        vec.close()

    det_summary = {}
    if dump_dir is not None and dump_dir.exists():
        print("[UCS Train] Training & temperature-calibrating Outcome Detector D_out on dumped traces...", flush=True)
        det_summary = train_detector_from_dumps(dump_dir, run_dir / "detector_ucs.pt", device=args.device)
        print("[UCS Train] Detector summary:", json.dumps(det_summary), flush=True)

    (run_dir / "summary.json").write_text(json.dumps({
        "total_steps": total_steps,
        "updates": update,
        "elapsed_s": round(time.time() - t_start, 2),
        "best_score": round(best_score, 4),
        "detector": det_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
