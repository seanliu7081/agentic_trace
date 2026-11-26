"""Evaluation of an M1E teacher (LRL_PLAN.md M1E step 2) in the stage simulator: first-attempt success of one
contact from the start region at the full measured hand-off error (curriculum scale 1), by kind, by effect (mode),
by handle class (the low-bar class separately: G1E asks that it be solved), by start mode (right or wrong face);
contact outside the part and its fixture; time to done. Validation layouts (seeds from VAL_SEED on) of the training
kinds, and layouts of the unseen fixtures (never trained on). Every episode is a new start (no replays), nominal
sensor model and physics randomization as in training, no dumps. The teacher acts with its mean; its done number is
drawn as in training (lrl_models.Policy: the policy raises done through the probability of a positive draw).

  python -m lrl_train.eval_teacher --ckpt <checkpoint.pt> --episodes 200 --procs 64 --out <dir>
  python -m lrl_train.eval_teacher --tables --out <dir>                  # tables.md again from episodes.jsonl
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=1,2 \
  python -m lrl_train.eval_teacher --ckpt <ckpt> --video 6 --gpu 1 --out <dir>   # rollout videos (EGL, GPU 1)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from atrace.lrl_e2e import ACT_DIM, EFFECTS

from .teacher_env import CFG as ENV, HANDLE_KINDS, TEACHER_KINDS, UNSEEN_KINDS, VAL_SEED, TeacherEnv

GPU_MINOR = {0: 3, 1: 2, 2: 1, 3: 0}    # R  nvidia-smi index -> /dev/nvidia minor, which EGL takes (scripts/run_episode)
TASK_KINDS = ("wooden_cabinet", "white_cabinet", "microwave", "flat_stove")
GROUPS = {"task fixtures (validation layouts)": TASK_KINDS,
          "generated mechanisms (validation layouts)": tuple(k for k in TEACHER_KINDS if k not in TASK_KINDS),
          "unseen fixtures": UNSEEN_KINDS}


def _policy(ckpt, seed):
    import torch
    from . import ppo
    torch.set_num_threads(1)
    model = ppo.load(ckpt)[0]
    rng, done_std = np.random.default_rng(seed), model.log_std[-1].exp().item()

    @torch.no_grad()
    def act(obs):
        a = model.actor(model.obs_norm(torch.as_tensor(obs)[None]))[0].numpy().astype(float)
        a[-1] += done_std * rng.standard_normal()
        return np.clip(a, -1.0, 1.0)
    return act


def run_job(job):
    """Episodes of one kind on validation layouts: list of the env's episode summaries."""
    kind, k, n, ckpt, out = job
    env = TeacherEnv([7, k], Path(out) / "bddl" / f"{kind}_{k}", ENV | {
        "kinds": (kind,), "seeds": (VAL_SEED + 1000 * k, VAL_SEED + 1000 * (k + 1)), "episodes_per_start": 1,
        "starts_per_layout": 4})
    act = _policy(ckpt, k)
    rows = []
    for _ in range(n):
        obs, _, _ = env.reset(1.0)
        while True:
            obs, _, _, ended, info = env.step(act(obs))
            if ended:
                break
        rows.append(info)
    env.close()
    return rows


def _rate(rows, key="success"):
    ok = [r[key] for r in rows if r[key] is not None]
    return {"n": len(ok), "k": int(sum(ok)), "rate": float(np.mean(ok)) if ok else None}


def summarize(rows):
    clean = [r for r in rows if not r["inject"]]
    by = defaultdict(list)
    for r in clean:
        for key in (("kind", r["kind"]), ("effect", r["effect"]), ("handle", r["handle"]),
                    ("kind_effect", f'{r["kind"]}|{r["effect"]}'), ("start", "wrong" if r["wrong"] else "right"),
                    ("kind_start", f'{r["kind"]}|{"wrong" if r["wrong"] else "right"}'),
                    ("group_effect", f'{_group(r["kind"])}|{r["effect"]}'),
                    ("group_handle", f'{_group(r["kind"])}|{r["handle"]}'), ("group", _group(r["kind"]))):
            by[key].append(r)
    done = [r for r in clean if r["stop"] == "done"]
    return {
        "episodes": len(rows), "clean": len(clean),
        "success": {f"{k[0]}:{k[1]}": _rate(v) for k, v in by.items()},
        "libero": {f"{k[0]}:{k[1]}": _rate(v, "libero") for k, v in by.items() if k[0] in ("kind", "kind_effect")},
        "outside": {f"{k[0]}:{k[1]}": _rate(v, "outside_any") for k, v in by.items() if k[0] in ("group", "effect")},
        "fixture": {f"{k[0]}:{k[1]}": _rate(v, "fixture_any") for k, v in by.items() if k[0] in ("group", "effect")},
        "stops": {s: sum(r["stop"] == s for r in clean) for s in ("done", "force", "time", "guard")},
        "time_to_done_s": {f"{k[0]}:{k[1]}": float(np.median([r["t_done"] * 0.05 for r in v if r["t_done"] >= 0]))
                           for k, v in by.items() if k[0] == "group" and any(r["t_done"] >= 0 for r in v)},
        "done_reached": _rate(done),
        "injected": {i: _rate([r for r in rows if r["inject"] == i], "success") for i in ("blocked", "slip")},
        "injected_giveup": {i: sum(r["stop"] == "done" for r in rows if r["inject"] == i) for i in ("blocked", "slip")},
        "labels": {lab: sum(r["label"] == lab for r in clean) for lab in sorted({r["label"] for r in clean})},
    }


def _group(kind):
    return next(g for g, ks in GROUPS.items() if kind in ks)


def tables(s):
    def cell(d):
        return "–" if not d or not d["n"] else f'{d["k"]}/{d["n"]} ({100 * d["rate"]:.0f}%)'
    sc = s["success"]
    out = [f'Episodes {s["episodes"]}, without injected failures {s["clean"]}. First-attempt success (truth: the '
           "goal's end of the range, or the amount, reached when the contact ends).", "",
           "| set | " + " | ".join(EFFECTS) + " | all | outside contact | fixture contact | median time to done, s |",
           "|---|" + "---|" * (len(EFFECTS) + 4)]
    for g in GROUPS:
        out.append(f"| {g} | " + " | ".join(cell(sc.get(f"group_effect:{g}|{e}")) for e in EFFECTS)
                   + f' | {cell(sc.get(f"group:{g}"))} | {cell(s["outside"].get(f"group:{g}"))} | '
                   f'{cell(s["fixture"].get(f"group:{g}"))} | {s["time_to_done_s"].get(f"group:{g}", float("nan")):.1f} |')
    out += ["", "| kind | " + " | ".join(EFFECTS) + " | right face | wrong face | LIBERO predicate |",
            "|---|" + "---|" * (len(EFFECTS) + 3)]
    for kind in TEACHER_KINDS + UNSEEN_KINDS:
        if f"kind:{kind}" in sc:
            out.append(f"| {kind} | " + " | ".join(cell(sc.get(f"kind_effect:{kind}|{e}")) for e in EFFECTS)
                       + f' | {cell(sc.get(f"kind_start:{kind}|right"))} | {cell(sc.get(f"kind_start:{kind}|wrong"))}'
                       f' | {cell(s["libero"].get(f"kind:{kind}"))} |')
    out += ["", "| handle class | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    for h in HANDLE_KINDS:
        cells = [sc.get(f"group_handle:{g}|{h}") for g in GROUPS]
        if any(cells):
            out.append(f"| {h} | " + " | ".join(cell(c) for c in cells) + " |")
    out += ["", f'Stops: {s["stops"]}; done raised with the goal reached: {cell(s["done_reached"])}; labels: '
            f'{s["labels"]}; injected failures, success {s["injected"]}, done raised {s["injected_giveup"]}.']
    return "\n".join(out)


def video(a):
    """A few rollouts rendered into mp4 files by a free camera aimed at the part. MuJoCo picks its GL backend at
    import, so the process starts with MUJOCO_GL=egl, MUJOCO_EGL_DEVICE_ID=<minor of the GPU> and that minor listed
    in CUDA_VISIBLE_DEVICES (robosuite checks it at import; scripts/run_episode.py --gpu does the same)."""
    assert os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("MUJOCO_EGL_DEVICE_ID") == str(GPU_MINOR[a.gpu])
    import cv2                      # the shared environment's imageio has no video writer
    import mujoco
    a.out.mkdir(parents=True, exist_ok=True)
    act = _policy(a.ckpt, 0)
    kinds = a.kinds or list(TEACHER_KINDS)
    for i in range(a.video):
        kind = kinds[i % len(kinds)]
        env = TeacherEnv([9, i], a.out / "bddl", ENV | {"kinds": (kind,), "seeds": (VAL_SEED, VAL_SEED + 1000),
                                                         "episodes_per_start": 1})
        obs, _, _ = env.reset(1.0)
        m, d = env.sim.robot.m, env.sim.robot.d
        r = mujoco.Renderer(m, 360, 480)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = env.part.face_point
        cam.distance, cam.azimuth, cam.elevation = 1.0, float(np.degrees(np.arctan2(*env.frame.R[1::-1, 0]))) + 180 - 35, -25
        frames = []
        while True:
            r.update_scene(d, cam)
            frames.append(r.render())
            obs, _, _, ended, info = env.step(act(obs))
            if ended:
                break
        name = f'{i:02d}_{kind}_{info["effect"]}_{info["handle"]}_{"ok" if info["success"] else "fail"}_{info["stop"]}.mp4'
        w = cv2.VideoWriter(str(a.out / name), cv2.VideoWriter_fourcc(*"mp4v"), 20, frames[0].shape[1::-1])
        for f in frames:
            w.write(f[:, :, ::-1])
        w.release()
        print(name, {k: info[k] for k in ("wrong", "err_angle", "steps", "label", "outside_any")}, flush=True)
        r.close()
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--episodes", type=int, default=200, help="per kind")
    ap.add_argument("--per-job", type=int, default=25)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--kinds", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--video", type=int, default=0, help="render this many rollouts instead")
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    if a.tables:
        rows = [json.loads(x) for x in open(a.out / "episodes.jsonl")]
        (a.out / "tables.md").write_text(tables(summarize(rows)) + "\n")
        print(tables(summarize(rows)))
        return
    if a.video:
        video(a)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    kinds = a.kinds or list(TEACHER_KINDS + UNSEEN_KINDS)
    jobs = [(kind, k, min(a.per_job, a.episodes - k * a.per_job), str(a.ckpt), str(a.out))
            for kind in kinds for k in range(-(-a.episodes // a.per_job))]
    rows = []
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as pool, open(a.out / "episodes.jsonl", "w") as f:
        for i, part in enumerate(pool.imap_unordered(run_job, jobs)):
            rows += part
            f.writelines(json.dumps(r) + "\n" for r in part)
            f.flush()
            print(f"job {i + 1}/{len(jobs)}  episodes {len(rows)}", flush=True)
    s = summarize(rows)
    (a.out / "summary.json").write_text(json.dumps(s, indent=1))
    (a.out / "tables.md").write_text(tables(s) + "\n")
    print(tables(s))


if __name__ == "__main__":
    main()
