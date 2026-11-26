"""Outcome detector v0 (LRL_PLAN.md 5.3, 6.4): a temporal convolution net over the skill trace and the goal
(lrl_models.DetectorNet, features lrl_models.trace_features), trained on dumped episodes, calibrated on validation
layouts.

Data: the episodes StageEnv writes (training rollouts of train_move_part, and gen_detector_data passes with the
base law, random residuals and the policy, injected failures included); labels and travel are simulator truth.
Splits are by layout: training layouts (seeds below VAL_SEED) train; validation layouts (VAL_SEED on) are split by
seed parity into a calibration half (threshold["reached"]: the lowest value from which on the precision of
"reached" stays at least `precision` on it, tag C) and a report half; the unseen kinds are reported separately.

  python -m lrl_train.train_detector --train <dump dirs> --val <dump dir> --unseen <dump dir> --out <dir>
  python -m lrl_train.train_detector --publish <detector.pt>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import shutil
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from atrace.lrl_contract import Goal
from atrace.lrl_models import DETECTOR_CLASSES, FEAT, STEP_FEATS, DetectorNet, load_detector, trace_features

from .env import VAL_SEED

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights" / "lrl"

CFG = {
    "net": {"width": 96, "layers": 4, "kernel": 5, "hidden": 192},     # T  about 0.24 M parameters (5.3: ~0.2 M)
    "epochs": 30,               # T
    "batch": 512,               # T
    "lr": 1e-3,                 # T  AdamW, cosine to zero
    "weight_decay": 1e-4,       # T
    "travel_weight": 0.2,       # T  weight of the travel loss (Gaussian negative log likelihood) against the classes
    "class_power": 0.5,         # T  class weights: (count share) ** -power, so rare failures are not ignored
    "std_floor": 0.05,          # T  feature standard deviations below this are raised to it (constant columns)
    "precision": 0.98,          # T  the operating point of "reached" (LRL_PLAN.md G1: precision >= 98 %)
}


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _features(rec):
    g = rec["goal"]
    contract = SimpleNamespace(goal=Goal(g["mode"], g["direction"], g["axis_point"], g["travel"]))
    x, goal = trace_features(rec["trace"] | {"touch": rec["touch"]}, contract)
    unit = 1.0 if g["mode"] == "turn" else FEAT["p_unit"]
    return (x, goal, DETECTOR_CLASSES.index(rec["label"]), rec["travel"] / unit,
            {k: rec[k] for k in ("kind", "layout", "mode", "inject", "source", "label", "travel", "wanted")})


def load_set(dirs, procs=32):
    """Features of every dumped episode under `dirs` (*.pkl): list of (x, goal, class, travel in feature units, meta)."""
    files = sorted(f for d in dirs for f in Path(d).rglob("*.pkl"))
    with mp.get_context("fork").Pool(procs) as pool:
        recs = [r for chunk in pool.imap(_load, files, chunksize=4) for r in chunk]
        return pool.map(_features, recs, chunksize=64)


def _batch(items, mu, sd, dev):
    n, T = len(items), max(len(it[0]) for it in items)
    x = np.zeros((n, T, STEP_FEATS), np.float32)
    mask = np.zeros((n, T), bool)
    for i, it in enumerate(items):
        x[i, :len(it[0])] = (it[0] - mu) / sd
        mask[i, :len(it[0])] = True
    last = np.stack([(it[0][-1] - mu) / sd for it in items])
    t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)    # noqa: E731
    return (t(x), t(mask, torch.bool), t(np.stack([it[1] for it in items])), t(last),
            t([it[2] for it in items], torch.long), t([it[3] for it in items]))


@torch.no_grad()
def predict(net, items, mu, sd, dev, batch=1024):
    """Class probabilities (N, C) and travel (N,) in feature units."""
    net.eval()
    probs, travel = [], []
    for i in range(0, len(items), batch):
        x, mask, goal, last, _, _ = _batch(items[i:i + batch], mu, sd, dev)
        logits, t, _ = net(x, mask, goal, last)
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        travel.append(t.cpu().numpy())
    return np.concatenate(probs), np.concatenate(travel)


def threshold_for(p_reached, is_reached, precision):
    """The lowest threshold from which on the precision of "reached" stays at least `precision` (tag C)."""
    order = np.argsort(-p_reached)
    p, y = p_reached[order], is_reached[order]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)          # precision when the threshold is p[k]
    ok = prec >= precision
    # from the top down, the longest prefix whose every precision holds
    k = len(ok) if ok.all() else int(np.argmin(ok))
    return float(p[k - 1]) if k > 0 else 1.0


def report(probs, items, thr):
    """Labels as the Detector gives them (reached only above the threshold, else the most probable other class);
    precision and recall per class, confusion counts."""
    y = np.array([it[2] for it in items])
    r = DETECTOR_CLASSES.index("reached")
    other = probs.copy()
    other[:, r] = -1.0
    pred = np.where(probs[:, r] >= thr, r, other.argmax(1))
    out = {}
    for c, name in enumerate(DETECTOR_CLASSES):
        tp, fp, fn = int(((pred == c) & (y == c)).sum()), int(((pred == c) & (y != c)).sum()), int(((pred != c) & (y == c)).sum())
        out[name] = {"n": int((y == c).sum()), "predicted": tp + fp, "precision": tp / max(1, tp + fp),
                     "recall": tp / max(1, tp + fn)}
    out["confusion"] = {f"{DETECTOR_CLASSES[a]}>{DETECTOR_CLASSES[b]}": int(((y == a) & (pred == b)).sum())
                        for a in range(len(DETECTOR_CLASSES)) for b in range(len(DETECTOR_CLASSES))
                        if ((y == a) & (pred == b)).any()}
    out["n"] = len(items)
    return out


def train(a):
    t0 = time.time()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    tr = load_set(a.train)
    tr = [it for it in tr if it[4]["layout"] < VAL_SEED]
    val = load_set([a.val])
    cal = [it for it in val if it[4]["layout"] % 2 == 0]
    rep = [it for it in val if it[4]["layout"] % 2 == 1]
    unseen = load_set([a.unseen]) if a.unseen else []
    print(f"loaded {len(tr)} train, {len(cal)} calibration, {len(rep)} report, {len(unseen)} unseen episodes "
          f"in {time.time() - t0:.0f} s", flush=True)
    rows = np.concatenate([it[0] for it in tr])
    mu, sd = rows.mean(0), np.maximum(rows.std(0), CFG["std_floor"])
    counts = Counter(it[2] for it in tr)
    share = np.array([counts.get(c, 0) + 1 for c in range(len(DETECTOR_CLASSES))], float)
    share /= share.sum()
    w = torch.as_tensor(share ** -CFG["class_power"], dtype=torch.float32, device=dev)
    w = w / (w * torch.as_tensor(share, dtype=torch.float32, device=dev)).sum()
    net = DetectorNet(**CFG["net"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps = CFG["epochs"] * (-(-len(tr) // CFG["batch"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ce = nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(a.seed)
    for ep in range(CFG["epochs"]):
        net.train()
        order = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(tr), CFG["batch"]):
            x, mask, goal, last, y, t = _batch([tr[j] for j in order[i:i + CFG["batch"]]], mu, sd, dev)
            logits, tp, log_sd = net(x, mask, goal, last)
            nll = (0.5 * ((t - tp) / log_sd.exp()) ** 2 + log_sd).mean()
            loss = ce(logits, y) + CFG["travel_weight"] * nll
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
        p, _ = predict(net, cal, mu, sd, dev)
        acc = float((p.argmax(1) == np.array([it[2] for it in cal])).mean())
        print(f"epoch {ep + 1} loss {tot / len(tr):.4f} calibration accuracy {acc:.3f} ({time.time() - t0:.0f} s)",
              flush=True)
    r = DETECTOR_CLASSES.index("reached")
    p_cal, t_cal = predict(net, cal, mu, sd, dev)
    thr = threshold_for(p_cal[:, r], np.array([it[2] == r for it in cal]), CFG["precision"])
    out = {"threshold": thr, "calibration": report(p_cal, cal, thr)}
    for name, items in (("validation", rep), ("unseen", unseen)):
        if not items:
            continue
        p, t = predict(net, items, mu, sd, dev)
        out[name] = report(p, items, thr)
        travel = np.array([it[3] for it in items])
        out[name]["travel_mae_units"] = float(np.abs(t - travel).mean())
        for mode in ("push", "pull", "turn"):
            sel = [k for k, it in enumerate(items) if it[4]["mode"] == mode]
            if sel:
                out[name][f"mode_{mode}"] = report(p[sel], [items[k] for k in sel], thr)
        sel = [k for k, it in enumerate(items) if it[4]["kind"] in ("wooden_cabinet", "white_cabinet", "microwave",
                                                                   "flat_stove")]
        if sel:
            out[name]["task_fixtures"] = report(p[sel], [items[k] for k in sel], thr)
    out["data"] = {"train": len(tr), "train_classes": {DETECTOR_CLASSES[c]: n for c, n in sorted(counts.items())},
                   "train_sources": dict(Counter(it[4]["source"] for it in tr)), "calibration": len(cal),
                   "validation": len(rep), "unseen": len(unseen), "dirs": [str(d) for d in a.train] + [str(a.val)]}
    a.out.mkdir(parents=True, exist_ok=True)
    ckpt = a.out / "detector.pt"
    torch.save({"format": "lrl_detector", "cfg": CFG["net"], "state": {k: v.cpu() for k, v in net.state_dict().items()},
                "classes": list(DETECTOR_CLASSES), "threshold": {"reached": thr}, "feat_mean": mu, "feat_std": sd,
                "meta": {"train_cfg": CFG, "report": out, "seed": a.seed}}, ckpt)
    (a.out / "detector_report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("threshold",)}), flush=True)
    return ckpt


def publish(ckpt, note=None):
    """Copy a detector to weights/lrl under its content hash, manifest entry marked "m1" (lrl_models.find_weights)."""
    load_detector(ckpt)                     # it loads as the harness loads it
    sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    name = f"detector_move_part_{date.today():%Y%m%d}_{sha[:8]}.pt"
    shutil.copyfile(ckpt, WEIGHTS / name)
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)["meta"]
    path = WEIGHTS / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[name] = {"sha256": sha, "part": "detector", "kind": "move_part", "modes": ["push", "pull", "turn"],
                      "m1": True, "source": str(ckpt), "published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "threshold": meta["report"]["threshold"], "data": meta["report"]["data"]} | (
                          {"note": note} if note else {})
    path.write_text(json.dumps(manifest, indent=1))
    return WEIGHTS / name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+")
    ap.add_argument("--val", type=Path)
    ap.add_argument("--unseen", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--publish", type=Path)
    ap.add_argument("--note")
    a = ap.parse_args()
    if a.publish:
        print(publish(a.publish.resolve(), a.note))
        return
    train(a)


if __name__ == "__main__":
    main()
