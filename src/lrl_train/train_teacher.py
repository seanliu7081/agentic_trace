"""Training of the M1E teacher (LRL_PLAN.md M1E step 2): PPO on privileged simulator state over a vector of TeacherEnv
processes. The actor acts through lrl_e2e's action space from its proprio plus privileged geometry; the critic also
sees the hidden physics. The curriculum scale (start distance, start-error width, wrong-face share, shaping weight)
grows with success; kinds of new layouts are sampled by low success. Unseen kinds are never loaded here; layout
seeds stay below VAL_SEED.

  python -m lrl_train.train_teacher --name t1 --minutes 60 --envs 64 --device cuda:0     # train, checkpoints
  python -m lrl_train.train_teacher --name f1 --init <bc.pt> --keep-std --bc-data <dir> --critic-warmup 3 ...
                                                                    # PPO from a behaviour clone (bc.py)
  python -m lrl_train.train_teacher --publish <checkpoint.pt> [--note "..."]              # publish as it is

Fine-tuning a clone: the expert's labelled steps (bc.py collect) stay in the actor's loss as an anchor whose weight
falls linearly to 0 over --bc-decay transitions, so that the first updates do not unlearn the clone and later ones
may leave the script where it does badly; --critic-warmup updates fit the critic alone first (it was fitted on the
expert's returns, which the clone does not earn).

Per run: <out>/<name>/train.csv (one row per update), episodes.csv, config.json, ckpt_latest.pt, ckpt_<minute>.pt
about every save_minutes, dump/ (a share of the episodes, labelled, for a later detector). A file <out>/<name>/STOP
ends the run after the update under way (e1/watch_plateau.py writes it). Published weights:
weights/lrl/move_part_teacher_<date>_<sha8>.pt, manifest kind "move_part_teacher", "m1e": true (never "m1").
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

from atrace.lrl_e2e import ACT_DIM, E2E_CFG, EFFECTS

from . import ppo, vec as vec_module
from .teacher_env import (CFG as ENV, CRIT_DIM, HANDLE_KINDS, OBS_DIM, REWARD, TEACHER_KINDS, UNSEEN_KINDS, VAL_SEED,
                          TeacherEnv)
from .vec import VecEnv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "lrl_m1e_20261125" / "e1"
WEIGHTS = ROOT / "weights" / "lrl"

CFG = {
    "envs": 64,                 # T  simulator processes
    "rollout": 32768,           # T  transitions per update
    "minibatch": 4096,          # T
    "hidden": 512, "layers": 3,  # T  actor and critic MLPs (the actor's input is 4x the M1 policy's)
    "lr": 3e-4,                 # T
    # exploration: initial standard deviation and mean per action number (lrl_e2e ACT_NAMES; done: ppo's)
    "std0": (0.3,) * 6 + (0.25,),   # T  motion leads (signed square: 0.3 -> 2 mm of lead at 0), gripper
    # x: no drift (s2-s5 drifted at -0.35 towards the part: contact at once, but opening needs the other sign and
    # s5's grasped starts were still pushed in after 1 M steps); gripper: towards closing, so that pads around a
    # handle close on it and a held one stays held (with mean 0 and std 0.25 a fifth of the steps opened or closed
    # the fingers by 1 cm each: grasped starts lost the handle within ~10 steps, s5's lost_grip labels)
    "mean0": (-0.15,) + (0.0,) * 5 + (0.4,),     # T  (s7: a mild drift to the part, 1 cm/s)
    "threads": 16,              # T  torch threads of an update on CPU
    "save_minutes": 20.0,       # T  numbered checkpoints this often
    # curriculum: the scale moves by `step` per update while the smoothed clean success is above `up` or below
    # `down`; kinds sampled by low success as M1 (weight = mix / n + (1 - mix) x (1 - success) / sum)
    "scale0": 0.0,              # T
    "scale_up": 0.55, "scale_down": 0.25, "scale_step": 0.02, "scale_ema": 0.3,    # T
    "kind_mix": 0.5,            # T
    "kind_memory": 2000,        # T  latest episodes per kind the success is read from
}
EPISODE = ("kind", "layout", "part", "effect", "handle", "scale", "near", "inject", "amount", "sense", "wrong", "err_pos",
           "err_angle", "dist", "stop", "steps", "t_done", "return", "success", "ever", "libero", "left0", "left",
           "label", "travel", "fixture_any", "outside_any", "fixture_steps", "outside_steps", "force_steps",
           "held_steps")


def rows(arrays, idx):
    return np.stack([arrays[i] for i in idx])


def collect(vec, model, out, n_steps, rng):
    """Run the vector until n_steps transitions exist (white exploration noise). Returns the batch, the episodes that
    ended, and the vector's last answer."""
    c = model.cfg
    traj = [{k: [] for k in ("obs", "crit", "act", "logp", "val", "rew", "end")} for _ in range(vec.n)]
    episodes, count = [], 0
    actions = np.zeros((vec.n, ACT_DIM))

    def book(out):
        for i in np.flatnonzero(out.stepped):
            traj[i]["rew"].append(out.reward[i]), traj[i]["end"].append(float(out.ended[i]))
            if out.ended[i]:
                episodes.append(out.info[i])
        return int(out.stepped.sum())

    while count < n_steps:
        live = np.flatnonzero(out.live)
        if len(live):
            a, logp, val = model.act(rows(out.obs, live), rows(out.crit, live), rng.standard_normal((len(live), ACT_DIM)))
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
    if len(live):
        last[live] = model.act(rows(out.obs, live), rows(out.crit, live))[2]
    batch = {k: [] for k in ("obs", "crit", "act", "logp", "adv", "ret")}
    for i, t in enumerate(traj):
        if not t["rew"]:
            continue
        adv, ret = ppo.gae(t["rew"], np.array(t["val"]), t["end"], last[i], c["gamma"], c["lam"])
        for k, v in zip(batch, (t["obs"], t["crit"], t["act"], t["logp"], adv, ret)):
            batch[k].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in batch.items()}, episodes, out


def load_teacher(path, device="cpu"):
    """A trained teacher as DAgger labels with it: act(obs (N, OBS_DIM)) -> the actor's mean actions (N, ACT_DIM),
    unclipped, with the checkpoint's observation normalization. The teacher raises done through the probability of a
    positive draw: P(done) = Phi(mean / act.done_std) per step."""
    model = ppo.load(path)[0].to(device)

    @torch.no_grad()
    def act(obs):
        return model.actor(model.obs_norm(torch.as_tensor(np.asarray(obs), device=device))).cpu().numpy()
    act.done_std, act.model = model.log_std[-1].exp().item(), model
    return act


def bc_anchor(data, device, batch=2048):
    """Loss of the actor's mean against the expert's labels on a random batch of bc.py's steps (bc.train's actor
    loss), as a callable for ppo.PPO.update(aux=...)."""
    from . import bc
    d = bc.load(data)
    obs = torch.as_tensor(d["obs"], device=device)
    act = torch.as_tensor(d["act"], device=device)
    done = act[:, -1] > 0
    target = act.clone()
    target[:, -1] = torch.where(done, bc.CFG["done_on"], bc.CFG["done_off"])
    w_done = torch.where(done, bc.CFG["done_weight"], 1.0)

    def loss(model):
        idx = torch.randint(len(obs), (batch,), device=device)
        mean = model.actor(model.obs_norm(obs[idx]))
        return ((mean[:, :-1] - target[idx, :-1]) ** 2).mean() + (w_done[idx] * (mean[:, -1] - target[idx, -1]) ** 2).mean()
    return loss


def publish(ckpt, note=None):
    """Copy a checkpoint to weights/lrl under its content hash and enter it in the manifest as kind
    "move_part_teacher", "m1e": true (the harness loads only "m1" entries)."""
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    meta = ppo.load(ckpt)[1]
    name = f"move_part_teacher_{date.today():%Y%m%d}_{sha[:8]}.pt"
    shutil.copyfile(ckpt, WEIGHTS / name)
    path = WEIGHTS / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[name] = {"sha256": sha, "part": "teacher", "kind": "move_part_teacher", "m1e": True,
                      "effects": list(EFFECTS), "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "source": str(ckpt),
                      "published": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      **{k: meta.get(k) for k in ("command", "seed", "layout_seeds", "kinds", "steps", "updates",
                                                  "init", "scale", "env", "reward", "e2e")}} | (
        {"note": note} if note else {})
    path.write_text(json.dumps(manifest, indent=1))
    return WEIGHTS / name


def train(a):
    assert not set(ENV["kinds"]) & set(UNSEEN_KINDS) and ENV["seeds"][1] <= VAL_SEED
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    run = a.out / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {"command": " ".join(sys.argv), "train": CFG, "ppo": ppo.CFG, "env": ENV, "reward": REWARD, "e2e": E2E_CFG,
         "args": {k: str(v) for k, v in vars(a).items()}}, indent=1, default=lambda o: np.asarray(o).tolist()))
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM, ppo.CFG | {k: CFG[k] for k in (
        "std0", "mean0", "lr", "minibatch", "hidden", "layers")} | ({"lr": a.lr} if a.lr else {})
        | ({"target_kl": a.target_kl} if a.target_kl else {}))
    scale = CFG["scale0"]
    if a.init:                  # a run continued: weights, statistics and the curriculum scale of a checkpoint
        init, meta0 = ppo.load(a.init)
        state = init.state_dict()
        if not a.keep_std:
            state["log_std"] = model.log_std.data
        model.load_state_dict(state)
        scale = meta0.get("scale", scale) if a.scale is None else a.scale
    elif a.scale is not None:
        scale = a.scale
    model.to(a.device)
    vec_module.GATHER = a.gather
    algo = ppo.PPO(model, model.cfg)
    vec = VecEnv(a.envs, run / "bddl", a.seed, ENV, REWARD, dump_dir=None if a.no_dump else run / "dump",
                 env_cls=TeacherEnv, dims=(OBS_DIM, CRIT_DIM))
    bc_loss = bc_anchor(a.bc_data, a.device) if a.bc_data else None
    recent = {k: [] for k in ENV["kinds"]}
    ema = None

    def weights():
        s = {k: np.mean(v) if v else 0.0 for k, v in recent.items()}
        lack = {k: 1.0 - x for k, x in s.items()}
        tot = sum(lack.values()) or 1.0
        return {k: CFG["kind_mix"] / len(s) + (1 - CFG["kind_mix"]) * x / tot for k, x in lack.items()}

    def progress():
        return max(steps / a.steps, (time.time() - t0) / (60 * a.minutes))

    def meta():
        return {"seed": a.seed, "layout_seeds": list(ENV["seeds"]), "kinds": list(ENV["kinds"]), "steps": steps,
                "updates": update, "scale": scale, "env": {k: v for k, v in ENV.items()}, "reward": REWARD,
                "e2e": E2E_CFG, "command": " ".join(sys.argv), "init": str(a.init) if a.init else None,
                "bc": {"data": [str(x) for x in a.bc_data], "weight": a.bc_weight, "decay": a.bc_decay,
                       "critic_warmup": a.critic_warmup} if a.bc_data else None}

    steps = update = 0
    vec.scale = scale
    out = vec.reset(scale)
    t0 = time.time()
    t_save = t0
    cols = (["update", "steps", "minutes", "sps", "scale", "episodes", "return", "success", "ever", "ep_steps"]
            + [f"success_{k}" for k in TEACHER_KINDS] + [f"success_{e}" for e in EFFECTS]
            + [f"success_{h}" for h in HANDLE_KINDS] + [f"weight_{k}" for k in TEACHER_KINDS]
            + ["success_wrong", "success_right", "stop_done", "stop_force", "stop_time", "done_reached",
               "done_inject", "touched", "held", "fixture_any", "outside_any", "force_share", "learn_s"]
            + ["pi_loss", "v_loss", "entropy", "clipped", "kl", "epochs", "explained", "std", "done_std", "aux",
               "bc_weight"])
    with open(run / "train.csv", "w", newline="") as f_train, open(run / "episodes.csv", "w", newline="") as f_ep:
        w_train, w_ep = csv.DictWriter(f_train, cols), csv.DictWriter(f_ep, ("update",) + EPISODE)
        w_train.writeheader(), w_ep.writeheader()
        while progress() < 1.0 and not (run / "STOP").exists():     # STOP: a plateau watcher ends the run
            vec.scale = scale
            vec.weights = weights()
            torch.set_num_threads(1)
            batch, episodes, out = collect(vec, model, out, CFG["rollout"], rng)
            torch.set_num_threads(CFG["threads"])
            t_learn = time.time()
            w_bc = a.bc_weight * max(0.0, 1.0 - steps / a.bc_decay) if bc_loss else 0.0
            stats = {"aux": 0.0} | algo.update(**batch, progress=progress(),
                                               aux=(lambda m: w_bc * bc_loss(m)) if w_bc > 0 else None,
                                               critic_only=update < a.critic_warmup)
            stats["bc_weight"] = w_bc
            t_learn = time.time() - t_learn
            clean = [e for e in episodes if not e["inject"]]
            for e in clean:
                recent[e["kind"]] = (recent[e["kind"]] + [e["success"]])[-CFG["kind_memory"]:]
            steps += len(batch["obs"])
            update += 1
            w_ep.writerows({"update": update} | {k: e[k] for k in EPISODE} for e in episodes)
            n = max(1, len(episodes))
            ep_steps = max(1, sum(e["steps"] for e in episodes))
            done = [e for e in episodes if e["stop"] == "done"]
            row = {"update": update, "steps": steps, "minutes": (time.time() - t0) / 60,
                   "sps": steps / (time.time() - t0), "scale": scale, "episodes": len(episodes),
                   "return": sum(e["return"] for e in episodes) / n,
                   "success": sum(e["success"] for e in clean) / max(1, len(clean)),
                   "ever": sum(e["ever"] for e in clean) / max(1, len(clean)), "ep_steps": ep_steps / n,
                   "done_reached": sum(e["success"] for e in done) / max(1, len(done)),
                   "done_inject": sum(bool(e["inject"]) for e in done) / max(1, len(done)),
                   "touched": sum(e["label"] != "no_contact" for e in episodes) / n,
                   "held": sum(e["held_steps"] > 0 for e in episodes) / n,
                   "fixture_any": sum(e["fixture_any"] for e in episodes) / n,
                   "outside_any": sum(e["outside_any"] for e in episodes) / n,
                   "force_share": sum(e["force_steps"] for e in episodes) / ep_steps, "learn_s": t_learn} | stats
            for key, values in (("kind", TEACHER_KINDS), ("effect", EFFECTS), ("handle", HANDLE_KINDS)):
                for v in values:
                    of = [e["success"] for e in clean if e[key] == v]
                    row[f"success_{v}"] = sum(of) / max(1, len(of))
            for w in (True, False):
                of = [e["success"] for e in clean if e["wrong"] == w]
                row["success_wrong" if w else "success_right"] = sum(of) / max(1, len(of))
            for k in TEACHER_KINDS:
                row[f"weight_{k}"] = vec.weights.get(k, 0.0)
            for s in ("done", "force", "time"):
                row[f"stop_{s}"] = sum(e["stop"] == s for e in episodes) / n
            w_train.writerow({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in row.items()})
            f_train.flush(), f_ep.flush()
            # curriculum: the smoothed clean success moves the scale (a step of 0 holds it, the per-start uniform
            # draw of a negative scale included)
            if clean and a.scale_step > 0:
                s_now = row["success"]
                ema = s_now if ema is None else (1 - CFG["scale_ema"]) * ema + CFG["scale_ema"] * s_now
                if ema > CFG["scale_up"]:
                    scale = min(1.0, scale + a.scale_step)
                elif ema < CFG["scale_down"]:
                    scale = max(0.0, scale - a.scale_step)
            ppo.save(run / "ckpt_latest.pt", model, meta())
            if time.time() - t_save > 60 * CFG["save_minutes"]:
                t_save = time.time()
                shutil.copyfile(run / "ckpt_latest.pt", run / f"ckpt_{int((t_save - t0) / 60):04d}min.pt")
            print(f'update {update} steps {steps} min {row["minutes"]:.1f} sps {row["sps"]:.0f} scale {row["scale"]:.2f} '
                  f'ret {row["return"]:+.3f} succ {row["success"]:.3f} open/close/on/off {row["success_open"]:.2f}/'
                  f'{row["success_close"]:.2f}/{row["success_turn_on"]:.2f}/{row["success_turn_off"]:.2f} '
                  f'touch {row["touched"]:.2f} held {row["held"]:.2f} out {row["outside_any"]:.2f} '
                  f'done {row["stop_done"]:.2f}/{row["done_reached"]:.2f} force {row["stop_force"]:.2f} '
                  f'kl {stats["kl"]:.4f} std {stats["std"]:.3f} ev {stats["explained"]:.2f} learn {t_learn:.1f}s',
                  flush=True)
    vec.close()
    print("finished", run / "ckpt_latest.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--steps", type=float, default=200e6, help="transitions; the run ends here or at --minutes")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--envs", type=int, default=CFG["envs"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=Path, help="continue from this checkpoint (its curriculum scale too)")
    ap.add_argument("--keep-std", action="store_true", help="with --init: keep the checkpoint's exploration")
    ap.add_argument("--scale", type=float, help="start the curriculum here")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--publish", type=Path, help="publish this checkpoint and stop")
    ap.add_argument("--note", help="with --publish: a remark for the manifest entry")
    ap.add_argument("--device", default="cpu", help="torch device of acting and the update, e.g. cuda:0")
    ap.add_argument("--no-dump", action="store_true", help="write no episodes for the detector")
    ap.add_argument("--gather", type=float, default=vec_module.GATHER)
    ap.add_argument("--bc-data", type=Path, nargs="+", help="bc.py collect's directories: the expert's steps as an anchor")
    ap.add_argument("--bc-weight", type=float, default=1.0, help="the anchor's weight at the start")
    ap.add_argument("--bc-decay", type=float, default=3e6, help="transitions over which it falls to 0")
    ap.add_argument("--critic-warmup", type=int, default=0, help="updates that fit the critic alone first")
    ap.add_argument("--lr", type=float, help="learning rate instead of CFG's")
    ap.add_argument("--target-kl", type=float, help="ppo target_kl instead of ppo.CFG's")
    ap.add_argument("--scale-step", type=float, default=CFG["scale_step"],
                    help="curriculum step per update; 0 holds the scale where --scale or --init put it (a clone "
                         "fine-tuned on the distribution it is judged on: f1 eased to scale 0.05 and did not improve "
                         "at the full error)")
    a = ap.parse_args()
    if a.publish:
        print(publish(a.publish.resolve(), a.note))
        return
    train(a)


if __name__ == "__main__":
    main()
