"""Training of the lrl move_part policy (LRL_PLAN.md M1): one network for push, pull and turn (the mode is an input
of the observation), PPO over a vector of StageEnv processes on the training kinds. The unseen kinds are never
loaded here; layout seeds stay below VAL_SEED. Acting and the update run on one GPU when one is given.

  python -m lrl_train.train_move_part --name m1a --minutes 100 --envs 192 --device cuda:0   # train, then publish
  python -m lrl_train.train_move_part --publish <checkpoint.pt> [--note "..."]               # publish as it is

Per run: <out>/<name>/train.csv (one row per update), episodes.csv (one row per finished episode),
ckpt_latest.pt and numbered checkpoints, dump/ (a share of the episodes for the detector). Published weights:
weights/lrl/move_part_<date>_<sha8>.pt, with an entry in weights/lrl/manifest.json marked "m1".

--free trains with the skill's base law off (M1E step 1, the control run E0). Its weights are published as
move_part_free_<date>_<sha8>.pt, kind "move_part_free", not marked "m1": the harness never loads them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import ppo, vec as vec_module
from .env import ACT_DIM, CFG as ENV, CRIT_DIM, OBS_DIM, REWARD, TRAIN_KINDS, UNSEEN_KINDS, VAL_SEED
from .vec import VecEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "lrl_m1_20261125" / "train2"
MODES = ("push", "pull", "turn")
WEIGHTS = ROOT / "weights" / "lrl"

CFG = {
    "envs": 96,                 # T  simulator processes
    "rollout": 32768,           # T  transitions per update
    "minibatch": 4096,          # T
    "curriculum": 0.3,          # T  share of the run over which the hand-off error grows from 0 to its full range
    "save_every": 25,           # T  updates between numbered checkpoints
    # Exploration: white noise, the recipe of run 2 (run 3's noise correlated in time moved the mean policy without
    # improving it, its log probability scored as independent draws: train/REPORT.md).
    # per action number: position 3, orientation 3, effort, steer 2, grip, done
    "noise_rho": (0.0,) * 11,   # T  step-to-step correlation of the noise
    # initial standard deviation and mean of the residual numbers (done: ppo). The gripper number opens the pads
    # below -0.33 (lrl_skill grip_band): with 0.4 of noise about 0 a held handle was let go in a fifth of the steps
    # and pulls never succeeded (smoke run, NOTES.md); it starts at "close" with little noise.
    "std0": (0.4,) * 9 + (0.15,),           # T
    "mean0": (0.0,) * 9 + (0.6,),           # T
    "lr": 3e-4,                 # T  run 2's
    "threads": 16,              # T  torch threads of an update on CPU (acting on a few rows at a time runs on one)
    # kinds of new layouts sampled by low success: weight = mix / n + (1 - mix) x (1 - success) / sum
    "kind_mix": 0.5,            # T
    "kind_memory": 2000,        # T  latest episodes per kind the success is read from
}
EPISODE = ("kind", "layout", "family", "mode", "inject", "scale", "left0", "way_in", "stop", "steps", "return",
           "success", "closed_at_end", "left", "label", "travel", "body_steps", "outside_steps", "force_steps")


class Noise:
    """Standard normal exploration noise per environment, carried from step to step with correlation noise_rho and
    drawn afresh at the start of an episode."""

    def __init__(self, n, seed):
        self.rng, self.eps, self.new = np.random.default_rng(seed), np.zeros((n, ACT_DIM)), np.ones(n, bool)

    def draw(self, live):
        keep = np.array(CFG["noise_rho"]) * ~self.new[live, None]
        self.eps[live] = keep * self.eps[live] + np.sqrt(1.0 - keep ** 2) * self.rng.standard_normal(keep.shape)
        self.new[live] = False
        return self.eps[live]


def rows(arrays, idx):
    return np.stack([arrays[i] for i in idx])


def collect(vec, model, out, n_steps, noise):
    """Run the vector until n_steps transitions exist. Returns the batch, the episodes that ended, and the vector's
    last answer (the next call starts from it)."""
    c = model.cfg
    traj = [{k: [] for k in ("obs", "crit", "act", "logp", "val", "rew", "end")} for _ in range(vec.n)]
    episodes, count = [], 0
    actions = np.zeros((vec.n, ACT_DIM))

    def book(out):
        noise.new |= out.fresh
        for i in np.flatnonzero(out.stepped):
            traj[i]["rew"].append(out.reward[i]), traj[i]["end"].append(float(out.ended[i]))
            if out.ended[i]:
                episodes.append(out.info[i])
        return int(out.stepped.sum())

    while count < n_steps:
        live = np.flatnonzero(out.live)
        if len(live):
            a, logp, val = model.act(rows(out.obs, live), rows(out.crit, live), noise.draw(live))
            actions[live] = a
            for j, i in enumerate(live):
                t = traj[i]
                t["obs"].append(out.obs[i]), t["crit"].append(out.crit[i]), t["act"].append(a[j])
                t["logp"].append(logp[j]), t["val"].append(val[j])
        out = vec.step(actions)
        count += book(out)
    for out in vec.settle() or [out]:       # the steps still under way belong to this batch
        book(out)
    last = np.zeros(vec.n)
    live = np.flatnonzero(out.live)
    last[live] = model.act(rows(out.obs, live), rows(out.crit, live))[2]
    batch = {k: [] for k in ("obs", "crit", "act", "logp", "adv", "ret")}
    for i, t in enumerate(traj):
        if not t["rew"]:
            continue
        adv, ret = ppo.gae(t["rew"], np.array(t["val"]), t["end"], last[i], c["gamma"], c["lam"])
        for k, v in zip(batch, (t["obs"], t["crit"], t["act"], t["logp"], adv, ret)):
            batch[k].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in batch.items()}, episodes, out


def publish(ckpt, note=None):
    """Copy a checkpoint to weights/lrl under its content hash and enter it in the manifest, with the command that
    trained it (kept in the checkpoint), marked "m1" (lrl_models.find_weights)."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    meta = ppo.load(ckpt)[1]
    modes = list(meta.get("modes", ("push",)))
    kind, mark = ("move_part", {"m1": True}) if meta["env"].get("base", True) else ("move_part_free", {})
    name = f"{kind}_{date.today():%Y%m%d}_{sha[:8]}.pt"
    shutil.copyfile(ckpt, WEIGHTS / name)
    path = WEIGHTS / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[name] = {"sha256": sha, "part": "skill", "kind": kind, "modes": modes, **mark,
                      "source": str(ckpt), "published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      **{k: meta.get(k) for k in ("command", "seed", "layout_seeds", "kinds", "steps", "updates",
                                                  "init", "env", "reward")}} | ({"note": note} if note else {})
    path.write_text(json.dumps(manifest, indent=1))
    return WEIGHTS / name


def train(a):
    assert not set(ENV["kinds"]) & set(UNSEEN_KINDS) and ENV["seeds"][1] <= VAL_SEED
    env = ENV | {"base": not a.free}
    torch.manual_seed(a.seed)
    run = a.out / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {"command": " ".join(sys.argv), "train": CFG, "ppo": ppo.CFG, "env": env, "reward": REWARD,
         "args": {k: str(v) for k, v in vars(a).items()}}, indent=1))
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM, ppo.CFG | {"std0": CFG["std0"], "mean0": CFG["mean0"],
                                                                   "lr": CFG["lr"], "minibatch": CFG["minibatch"]})
    if a.init:                  # a run continued: the weights and statistics of a checkpoint, the full error at once
        state = ppo.load(a.init)[0].state_dict()
        state["log_std"] = model.log_std.data       # exploration as set for this run, not where the last one left it
        model.load_state_dict(state)
    model.to(a.device)
    vec_module.GATHER = a.gather
    algo = ppo.PPO(model, model.cfg)
    noise = Noise(a.envs, a.seed)
    vec = VecEnv(a.envs, run / "bddl", a.seed, cfg=env, dump_dir=None if a.no_dump else run / "dump")
    recent = {k: [] for k in ENV["kinds"]}

    def weights():
        s = {k: np.mean(v) if v else 0.0 for k, v in recent.items()}
        lack = {k: 1.0 - x for k, x in s.items()}
        tot = sum(lack.values()) or 1.0
        return {k: CFG["kind_mix"] / len(s) + (1 - CFG["kind_mix"]) * x / tot for k, x in lack.items()}

    def progress():
        return max(steps / a.steps, (time.time() - t0) / (60 * a.minutes))

    def scale():
        return 1.0 if a.init else min(1.0, progress() / CFG["curriculum"])

    def meta():
        return {"seed": a.seed, "layout_seeds": list(ENV["seeds"]), "kinds": list(ENV["kinds"]), "steps": steps,
                "updates": update, "env": dict(env), "reward": REWARD, "modes": list(MODES),
                "command": " ".join(sys.argv), "init": str(a.init) if a.init else None}

    steps = update = 0
    out = vec.reset(0.0)
    t0 = time.time()
    cols = (["update", "steps", "minutes", "sps", "scale", "episodes", "return", "success", "ep_steps"]
            + [f"success_{k}" for k in TRAIN_KINDS] + [f"success_{m}" for m in MODES] + [f"weight_{k}" for k in TRAIN_KINDS]
            + ["stop_done", "stop_force", "stop_time", "done_closed", "done_inject", "inject_share",
               "body_share", "outside_share", "force_share", "learn_s"]
            + ["pi_loss", "v_loss", "entropy", "clipped", "kl", "epochs", "explained", "std", "done_std"])
    with open(run / "train.csv", "w", newline="") as f_train, open(run / "episodes.csv", "w", newline="") as f_ep:
        w_train, w_ep = csv.DictWriter(f_train, cols), csv.DictWriter(f_ep, ("update",) + EPISODE)
        w_train.writeheader(), w_ep.writeheader()
        while progress() < 1.0:
            vec.scale = scale()
            vec.weights = weights()
            torch.set_num_threads(1)
            batch, episodes, out = collect(vec, model, out, CFG["rollout"], noise)
            torch.set_num_threads(CFG["threads"])
            t_learn = time.time()
            stats = algo.update(**batch, progress=progress())
            t_learn = time.time() - t_learn
            for e in episodes:
                if not e["inject"]:             # an injected failure says nothing about how well a kind is done
                    recent[e["kind"]] = (recent[e["kind"]] + [e["success"]])[-CFG["kind_memory"]:]
            steps += len(batch["obs"])
            update += 1
            w_ep.writerows({"update": update} | {k: e[k] for k in EPISODE} for e in episodes)
            n = max(1, len(episodes))
            ep_steps = max(1, sum(e["steps"] for e in episodes))
            done = [e for e in episodes if e["stop"] == "done"]
            row = {"update": update, "steps": steps, "minutes": (time.time() - t0) / 60,
                   "sps": steps / (time.time() - t0), "scale": vec.scale, "episodes": len(episodes),
                   "return": sum(e["return"] for e in episodes) / n, "success": sum(e["success"] for e in episodes) / n,
                   "ep_steps": ep_steps / n,
                   "done_closed": sum(e["closed_at_end"] for e in done) / max(1, len(done)),
                   "body_share": sum(e["body_steps"] for e in episodes) / ep_steps,
                   "outside_share": sum(e["outside_steps"] for e in episodes) / ep_steps,
                   "force_share": sum(e["force_steps"] for e in episodes) / ep_steps, "learn_s": t_learn,
                   "inject_share": sum(bool(e["inject"]) for e in episodes) / n,
                   "done_inject": sum(bool(e["inject"]) for e in done) / max(1, len(done))} | stats
            clean = [e for e in episodes if not e["inject"]]
            for k in TRAIN_KINDS:
                of = [e["success"] for e in clean if e["kind"] == k]
                row[f"success_{k}"] = sum(of) / max(1, len(of))
                row[f"weight_{k}"] = vec.weights.get(k, 0.0)
            for m in MODES:
                of = [e["success"] for e in clean if e["mode"] == m]
                row[f"success_{m}"] = sum(of) / max(1, len(of))
            for s in ("done", "force", "time"):
                row[f"stop_{s}"] = sum(e["stop"] == s for e in episodes) / n
            w_train.writerow({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in row.items()})
            f_train.flush(), f_ep.flush()
            ppo.save(run / "ckpt_latest.pt", model, meta())
            if update % CFG["save_every"] == 0:
                shutil.copyfile(run / "ckpt_latest.pt", run / f"ckpt_{update:05d}.pt")
            print(f'update {update} steps {steps} min {row["minutes"]:.1f} sps {row["sps"]:.0f} scale {vec.scale:.2f} '
                  f'ret {row["return"]:+.3f} succ {row["success"]:.3f} push/pull/turn {row["success_push"]:.2f}/'
                  f'{row["success_pull"]:.2f}/{row["success_turn"]:.2f} done {row["stop_done"]:.2f}/'
                  f'{row["done_closed"]:.2f}/{row["done_inject"]:.2f} kl {stats["kl"]:.4f} std {stats["std"]:.3f} '
                  f'learn {t_learn:.1f}s', flush=True)
    vec.close()
    if not a.no_publish:
        print("published", publish(run / "ckpt_latest.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--steps", type=float, default=50e6, help="transitions; the run ends here or at --minutes")
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--envs", type=int, default=CFG["envs"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, help="continue from this checkpoint, without the curriculum")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--publish", type=Path, help="publish this checkpoint and stop")
    ap.add_argument("--note", help="with --publish: a remark for the manifest entry")
    ap.add_argument("--device", default="cpu", help="torch device of acting and the update, e.g. cuda:0")
    ap.add_argument("--no-publish", action="store_true", help="a throughput test: keep the checkpoint local")
    ap.add_argument("--no-dump", action="store_true", help="write no episodes for the detector")
    ap.add_argument("--gather", type=float, default=vec_module.GATHER,
                    help="s the vector waits after the first answer, so that the policy acts on larger batches")
    ap.add_argument("--free", action="store_true", help="the skill's base law off (M1E step 1, the control run E0)")
    a = ap.parse_args()
    if a.publish:
        print(publish(a.publish.resolve(), a.note))
        return
    train(a)


if __name__ == "__main__":
    main()
