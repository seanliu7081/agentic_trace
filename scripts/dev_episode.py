#!/usr/bin/env python
"""Dev runner: one episode of the integrated harness (CP-5 + TRACE) on a NON-TEST LIBERO scene (never use it on
Libero10-Collision states).

Same simulator settings as scripts/run_episode.py: libero10_collision.runtime.make_env (private LIBERO
config via LIBERO_CONFIG_PATH, seed 7, 512 px cameras, 20 Hz OSC_POSE), the saved initial state, then 10
hold steps. Success is LIBERO's original rule: the episode ends as a success at the first control step
where env.check_success() is true (LIBERO's `done`), the 10 hold steps after the policy ends included.
Evaluator-only, plus a light contact log (SceneCollisionMonitor needs the benchmark's added obstacle).

--s1 astra    real PROGRAM, REPAIR, VERIFY and effect calls, as in the eval;
--s1 stub     PROGRAM from scripts/s1_stub.py (simulator truth, a test fixture), the other calls real
              (the PROGRAM call still counts against --max-calls, as in the eval);
--s1 offline  PROGRAM from the stub and no model calls at all (REPAIR and VERIFY unavailable).
--dry-run     write the stub program and the head image with its pixels, then exit (no SAM, no harness).

Example (LIBERO-object, two non-test objects into the basket, no model calls, GPU 0):
  .venv/bin/python scripts/dev_episode.py --suite libero_object \\
      --task pick_up_the_ketchup_and_place_it_in_the_basket --init 3 --s1 offline \\
      --goal in:ketchup_1:basket_1_contain_region,in:bbq_sauce_1:basket_1_contain_region \\
      --instruction "put both the ketchup and the bbq sauce in the basket"
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_MINOR = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}   # nvidia-smi index -> /dev/nvidia minor (EGL)


class GoalReached(BaseException):
    """LIBERO's done: ends the episode from the step callback (BaseException passes the harness's handlers)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object", help="libero_object | libero_goal | libero_spatial | ...")
    ap.add_argument("--task", required=True, help="bddl stem")
    ap.add_argument("--init", type=int, default=0, help="index into the task's .pruned_init states")
    ap.add_argument("--suite-root", default=None,
                    help="LIBERO package dir with bddl_files/ and init_files/ for --suite (default: CP-5's LIBERO)")
    ap.add_argument("--goal", default=None,
                    help="dev goal override 'pred:arg[:arg],...' (scoring and the stub), e.g. in:milk_1:basket_1_contain_region")
    ap.add_argument("--instruction", default=None, help="language override (use with --goal)")
    ap.add_argument("--s1", choices=["astra", "stub", "offline"], default="stub")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--max-wall", type=float, default=900)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--hint", default=None, help="operator note added to every S1 prompt")
    ap.add_argument("--video", action="store_true", help="head + wrist video.mp4")
    ap.add_argument("--motion", choices=["trace", "v0", "cp5"], default="trace",
                    help="contact-free segments: TRACE search | TRACE contract with CP-5 seeds only | CP-5 planner")
    ap.add_argument("--trace-device", default="cuda:0")
    ap.add_argument("--trace-budget-ms", type=float, default=2000.0)
    ap.add_argument("--compiles", type=int, default=1, help="PROGRAM compiles for term confidence (1 = off)")
    ap.add_argument("--true-scene", action="store_true",
                    help="evaluator: save the simulator's true scene at every TRACE request (trace/true/)")
    args = ap.parse_args()

    minor = GPU_MINOR[args.gpu]
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(minor),
                      CUDA_VISIBLE_DEVICES=f"{args.gpu},{minor}" if args.gpu != minor else str(args.gpu))
    os.environ.setdefault("LIBERO_ROOT", str(ROOT / "third_party" / "LIBERO"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), "/workspace/Libero10-Collision"]

    import cv2
    import mujoco
    import numpy as np
    import torch
    from libero10_collision.runtime import HOLD, make_env, setup
    from atrace.harness import Harness, _jsonable
    from atrace.llm import AstraClient
    from atrace.segment import Sam3Segmenter
    from atrace.sim_iface import SimRobot
    from s1_stub import StubAstra, annotate, stub_program

    root = setup()
    base = Path(args.suite_root) if args.suite_root else root / "libero" / "libero"
    states = torch.load(base / "init_files" / args.suite / f"{args.task}.pruned_init", weights_only=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or ROOT / "runs" / "dev" / f"{args.suite}_{args.task}_i{args.init:02d}_{args.s1}_{stamp}")
    out.mkdir(parents=True, exist_ok=False)
    build = hashlib.sha256(b"".join(f.read_bytes() for d in ("atrace", "traceplan")
                                    for f in sorted((ROOT / "src" / d).glob("*.py")))
                           + (ROOT / "scripts" / "run_episode.py").read_bytes()).hexdigest()[:16]
    (out / "args.json").write_text(json.dumps({**vars(args), "build": build}, indent=1))

    env = make_env(base / "bddl_files" / args.suite / f"{args.task}.bddl", seed=7)
    env.set_init_state(np.asarray(states[args.init]))
    for _ in range(10):
        env.step(HOLD)
    if args.goal:
        env.env.parsed_problem["goal_state"] = [[p.lower(), *rest] for p, *rest in
                                               (g.split(":") for g in args.goal.split(","))]
    language = args.instruction or env.language_instruction

    prog = None
    if args.s1 != "astra" or args.dry_run:
        prog = stub_program(env, scene=f"dev stub: {args.suite}/{args.task} init {args.init}")
        (out / "stub_program.json").write_text(json.dumps(prog, indent=1))
        cv2.imwrite(str(out / "stub_program_head.png"), annotate(env, prog)[:, :, ::-1])
    if args.dry_run:
        print(json.dumps(prog, indent=1))
        print("out:", out)
        env.close()
        return

    # evaluator-side contact log (never visible to the policy)
    m, d = env.sim.model._model, env.sim.data._data
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "" for g in range(m.ngeom)]
    robot = np.array([n.startswith(("robot0_", "gripper0_", "mount0_")) for n in names])
    initial = {tuple(sorted((names[d.contact[i].geom1], names[d.contact[i].geom2]))) for i in range(d.ncon)}
    contacts, obj_contacts, frames = {}, {}, []
    onsets, active = [], set()             # contact onsets with their control step (certificate honesty)

    def on_step(rb, ob):
        now = set()
        for i in range(d.ncon):
            c = d.contact[i]
            r1, r2 = robot[c.geom1], robot[c.geom2]
            if not r1 and not r2:
                # object-object pairs that did not touch at the start (strikes, knock-overs)
                key = tuple(sorted((names[c.geom1], names[c.geom2])))
                if key not in initial and key[0] != key[1] and c.dist < 0:
                    k = "|".join(key)
                    obj_contacts[k] = max(obj_contacts.get(k, 0.0), -float(c.dist))
                    now.add(("object", k))
                continue
            if r1 == r2 or c.dist > -0.001:
                continue
            mine, other = (names[c.geom1], names[c.geom2]) if r1 else (names[c.geom2], names[c.geom1])
            contacts[f"{mine}|{other}"] = max(contacts.get(f"{mine}|{other}", 0.0), -float(c.dist))
            now.add(("robot", f"{mine}|{other}"))
        onsets.extend({"step": rb.steps, "kind": k, "pair": p} for k, p in sorted(now - active))
        active.clear()
        active.update(now)
        if args.video and rb.steps % 2 == 0:
            frames.append(np.concatenate([cv2.resize(ob["agentview_image"][::-1], (320, 320)),
                                          cv2.resize(ob["robot0_eye_in_hand_image"][::-1], (320, 320))], 1))
        if env.check_success():
            raise GoalReached

    seg = Sam3Segmenter(device="cuda:0")
    seg.load()
    real = None if args.s1 == "offline" else AstraClient(
        out / "model", effort=args.effort, max_calls=args.max_calls - (args.s1 == "stub"))
    astra = real if args.s1 == "astra" else StubAstra(prog, out / "model", real)
    rb = SimRobot(env, on_step=on_step)
    t0 = time.time()
    h = Harness(rb, language, out, astra, seg, max_wall_s=args.max_wall, verify=not args.no_verify,
                prompt_note=args.hint, motion=args.motion, trace_device=args.trace_device,
                trace_budget_ms=args.trace_budget_ms, compiles=args.compiles)
    if args.true_scene and h.tm is not None:
        # evaluator-side truth for the certificate measures: the harness hands over the request id only
        from true_scene import true_snapshot
        (out / "trace" / "true").mkdir(parents=True, exist_ok=True)
        h.tm.true_scene_hook = lambda rid: true_snapshot(env).save(out / "trace" / "true" / f"{rid}.npz")
    status, success_step = "stopped_at_libero_done", None
    try:
        status = h.run()
        rb.hold(10)
    except GoalReached:
        success_step = rb.steps
        if not h._events.closed:           # stopped inside the policy: end its log as Harness.run does
            h.log("end", status=status, timing=h.timing, astra=astra.stats())
            h._events.close()
    t_policy = time.time() - t0
    summary = {
        "suite": args.suite, "task": args.task, "init": args.init, "goal": env.env.parsed_problem["goal_state"],
        "language": language, "s1": args.s1, "build": build, "motion": args.motion, "status": status,
        "policy_finished": h.finished,
        "success": success_step is not None, "success_step": success_step,
        "robot_contacts_over_1mm": contacts, "new_object_contacts": obj_contacts,
        "wall_s_policy": round(t_policy, 1), "sim_steps": rb.steps, "model": astra.stats(), "timing": h.timing,
        "segmenter": {"calls": seg.calls, "s": round(seg.seconds, 1)},
        "program": h.program, "ledger": h.ledger, "repairs": h.repairs,
        "trace": h.tm.summary() if h.tm is not None else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    (out / "contacts.json").write_text(json.dumps({"onsets": onsets, "robot_max_depth": contacts,
                                                   "object_max_depth": obj_contacts}, indent=1))
    env.close()
    if frames:
        raw = out / "video_mpeg4.tmp.mp4"
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 20, (frames[0].shape[1], frames[0].shape[0]))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "video.mp4")], check=True)
        raw.unlink()
    print(json.dumps({k: summary[k] for k in ("task", "init", "s1", "motion", "status", "success", "success_step",
                                              "wall_s_policy", "trace",
                                              "robot_contacts_over_1mm", "new_object_contacts")}, default=str))
    print("model:", summary["model"])
    print("out:", out)


if __name__ == "__main__":
    main()
