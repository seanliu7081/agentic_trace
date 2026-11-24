#!/usr/bin/env python
"""Run one Libero10-Collision episode with the integrated harness (CP-5 + TRACE) and score it.

Evaluator-only (never visible to the policy): SceneCollisionMonitor on every
physics substep, and LIBERO's original success rule: LIBERO sets
done = check_success() at every control step, and the episode ends as a success
at the first done (LIBERO's own evaluation; the benchmark's reference loop).
After the policy ends, the episode continues for ten control steps with the same
gripper command. No step limit; the harness has its own wall budget.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_MINOR = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}   # nvidia-smi index -> /dev/nvidia minor (EGL)


class GoalReached(BaseException):
    """LIBERO's done: ends the episode from the step callback (BaseException passes the harness's handlers)."""


class CapReached(BaseException):
    """The test limit (control time or model calls, evaluator-side): ends the episode as a failure wherever the
    policy is (stepping, planning or waiting for a model answer)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--max-wall", type=float, default=900)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--hint", default=None, help="optional task-specific note added to every S1 prompt")
    ap.add_argument("--instruction", default=None,
                    help="task instruction the policy gets instead of the episode's (success check unchanged)")
    ap.add_argument("--no-render", action="store_true", help="with --video: record but skip the post-episode videos")
    ap.add_argument("--motion", choices=["trace", "v0", "cp5"], default="trace",
                    help="contact-free segments: TRACE search | TRACE contract with CP-5 seeds only | CP-5 planner")
    ap.add_argument("--placement", choices=["ladder", "v1", "v2", "unified"], default="ladder",
                    help="putting the held object down: careful planner with the margin ladder | the same planner "
                         "ranking instead of rejecting (tier 3) | no planner: System 1 points, contact ends the "
                         "descent, System 1 judges the result (r1's mechanism) | unified: System 1 points, geometry "
                         "finds the rest pose, the way down carries its expected contact, the gripper opens only when "
                         "it holds (CONTACT_STAGE_PLAN.md)")
    ap.add_argument("--in-release", choices=["program", "drop", "place"], default="program",
                    help="every \"in\" transfer releases this way (drop: let go at the lowest height up to 12 cm above "
                         "the rest where the open hand is clear; place: set down on the floor), whatever the program "
                         "says; program: as the program says")
    ap.add_argument("--trace-device", default="cuda:0")
    ap.add_argument("--trace-budget-ms", type=float, default=2000.0)
    ap.add_argument("--compiles", type=int, default=1, help="PROGRAM compiles for term confidence (1 = off)")
    ap.add_argument("--true-scene", action="store_true",
                    help="evaluator: save the simulator's true scene at every TRACE request (trace/true/)")
    ap.add_argument("--cap-control-s", type=float, default=None,
                    help="evaluator: stop the episode as a failure when the control time (policy wall clock, "
                         "model waits included) reaches this many seconds")
    ap.add_argument("--cap-calls", type=int, default=None,
                    help="evaluator: stop the episode as a failure when the policy asks for a model call beyond "
                         "this many")
    args = ap.parse_args()

    minor = GPU_MINOR[args.gpu]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(minor)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu},{minor}" if args.gpu != minor else str(args.gpu)
    os.environ.setdefault("LIBERO_ROOT", str(ROOT / "third_party" / "LIBERO"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.path.insert(0, "/workspace/Libero10-Collision")
    sys.path.insert(0, str(ROOT / "src"))

    import numpy as np
    import cv2
    from libero10_collision import load_episode, SceneCollisionMonitor
    from atrace.harness import Harness, _jsonable
    from atrace.llm import AstraClient
    from atrace.segment import Sam3Segmenter
    from atrace.sim_iface import SimRobot

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out or ROOT / "runs" / f"{args.task}_ep{args.episode:03d}_{args.effort}_{stamp}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    seg = Sam3Segmenter(device="cuda:0")
    seg.load()
    astra = AstraClient(out / "model", effort=args.effort, max_calls=args.max_calls)
    frames = []

    # Test limits (evaluator-side, the harness itself stays as it is): a timer thread or the call counter sends
    # SIGUSR1 to the main thread, whose handler raises CapReached there, also out of a wait for a model answer.
    import signal
    import threading
    from atrace.llm import AstraError
    cap = {"armed": False, "why": None, "t": None, "calls": None}
    main_ident = threading.main_thread().ident

    def cap_fire(why):
        if cap["armed"] and cap["why"] is None:
            cap["why"] = why
            signal.pthread_kill(main_ident, signal.SIGUSR1)

    def on_cap_signal(signum, frame):
        if cap["armed"]:
            cap["armed"] = False
            raise CapReached(cap["why"] or "control_time")

    signal.signal(signal.SIGUSR1, on_cap_signal)
    if args.cap_calls is not None:
        astra_call = astra.call

        def capped_call(kind, *a, **k):           # astra.submit looks up self.call, so pool calls pass here too
            with astra._lock:
                over = len(astra.calls) >= args.cap_calls
            if over:
                cap_fire("model_calls")
                raise AstraError("model call cap reached", code="cap")
            return astra_call(kind, *a, **k)
        astra.call = capped_call
    t_start = time.time()
    if args.cap_control_s is not None or args.cap_calls is not None:
        cap["armed"] = True
    if args.cap_control_s is not None:
        timer = threading.Timer(args.cap_control_s, cap_fire, args=("control_time",))
        timer.daemon = True
        timer.start()
    rec = None
    import hashlib
    build = hashlib.sha256(b"".join(f.read_bytes() for d in ("atrace", "traceplan")
                                    for f in sorted((ROOT / "src" / d).glob("*.py")))
                           + (ROOT / "scripts" / "run_episode.py").read_bytes()).hexdigest()[:16]
    with load_episode(args.task, args.episode) as (env, obs, manifest):
        language = args.instruction or manifest["language"]
        with SceneCollisionMonitor(env) as monitor:
            if args.video:
                # evaluator-side recording for the post-episode task / 3D-map videos (never read by the policy)
                sys.path.insert(0, str(ROOT / "scripts"))
                from recorder import Recorder
                rec = Recorder(out, stride=2, meta={"task": args.task, "episode": args.episode,
                                                    "instruction": language, "hint": args.hint,
                                                    "build": build, "effort": args.effort,
                                                    "harness": f"agentic_trace (CP-5 + TRACE, {args.motion})"})

            def on_step(robot, ob):
                monitor.sample_endpoint()
                if args.video and robot.steps % 2 == 0:
                    a = ob["agentview_image"][::-1]
                    w = ob["robot0_eye_in_hand_image"][::-1]
                    frames.append(np.concatenate([cv2.resize(a, (320, 320)), cv2.resize(w, (320, 320))], 1))
                if rec is not None:
                    rec.on_step(robot, ob)
                if env.check_success():
                    cap["armed"] = False               # LIBERO's done came first: the limit no longer applies
                    raise GoalReached
            robot = SimRobot(env, on_step=on_step)
            h = Harness(robot, language, out, astra, seg, max_wall_s=args.max_wall,
                        verify=not args.no_verify, prompt_note=args.hint, motion=args.motion,
                        trace_device=args.trace_device, trace_budget_ms=args.trace_budget_ms, compiles=args.compiles,
                        placement=args.placement,
                        in_release=None if args.in_release == "program" else args.in_release)
            if args.true_scene and h.tm is not None:
                # evaluator-side truth for the certificate measures: the harness hands over the request id only
                sys.path.insert(0, str(ROOT / "scripts"))
                from true_scene import true_snapshot
                (out / "trace" / "true").mkdir(parents=True, exist_ok=True)
                h.tm.true_scene_hook = lambda rid: true_snapshot(env).save(out / "trace" / "true" / f"{rid}.npz")
            if rec is not None:
                rec.attach(h)
            status, success_step = "stopped_at_libero_done", None
            try:
                status = h.run()
                robot.hold(10)
                cap["armed"] = False
            except GoalReached:
                success_step = robot.steps
                if not h._events.closed:           # stopped inside the policy: end its log as Harness.run does
                    h.log("end", status=status, timing=h.timing, astra=astra.stats())
                    h._events.close()
            except CapReached as e:
                status = f"cap_{e}"
                cap.update(t=round(time.time() - t_start, 1), calls=len(astra.calls))
                if not h._events.closed:
                    h.log("end", status=status, timing=h.timing, astra=astra.stats(), cap=cap)
                    h._events.close()
            cap["armed"] = False
            t_policy = time.time() - t_start
            col = monitor.result(raw_goal_success=bool(env.check_success()))
    wall = time.time() - t_start
    success = success_step is not None
    summary = {
        "task": args.task, "episode": args.episode, "language": language, "language_original": manifest["language"],
        "build": build,
        "motion": args.motion, "placement": args.placement, "in_release": args.in_release, "status": status,
        "policy_finished": h.finished,
        "success": success, "success_step": success_step,
        "cap": {"control_s": args.cap_control_s, "calls": args.cap_calls,
                "reached": cap["why"] if status.startswith("cap_") else None,
                "at_s": cap["t"], "calls_at_stop": cap["calls"]},
        "scene_collision_free": col.get("scene_collision_free"),
        "added_obstacle_collision_free": col.get("added_obstacle_collision_free"),
        "collision_free_success": success and bool(col.get("scene_collision_free")),
        "colliding_pairs": [{"owners": r.get("owners"), "onsets": r.get("distinct_onset_events"),
                             "max_pen": r.get("max_penetration_m"),
                             "first": next((e for e in col.get("onset_events", [])
                                            if sorted(e.get("owners", [])) == sorted(r.get("owners", []))), None)}
                            for r in col.get("colliding_pairs", [])],
        "allowed_contacts_by_reason": col.get("allowed_contacts_by_reason"),
        "wall_s_policy": round(t_policy, 1), "wall_s_total": round(wall, 1),
        "sim_steps": robot.steps, "sim_time_s": round(robot.steps * 0.05, 1),
        "model": astra.stats(), "timing": h.timing, "segmenter": {"calls": seg.calls, "s": round(seg.seconds, 1)},
        "program": h.program, "ledger": h.ledger, "repairs": h.repairs,
        "trace": h.tm.summary() if h.tm is not None else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_jsonable))
    (out / "collision.json").write_text(json.dumps(col, indent=1, default=_jsonable))
    if frames:
        # OpenCV can only write MPEG-4 Part 2 here, which browsers / VS Code cannot play: re-encode to H.264
        raw = out / "video_mpeg4.tmp.mp4"
        vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                             (frames[0].shape[1], frames[0].shape[0]))
        for f in frames:
            vw.write(f[:, :, ::-1])
        vw.release()
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / "video.mp4")],
                           check=True, timeout=300)
            raw.unlink()
        except Exception as e:                     # keep the playable-elsewhere original rather than nothing
            print("video re-encode failed:", repr(e))
            raw.rename(out / "video.mp4")
    if rec is not None:
        rec.close()
        if not args.no_render:
            import subprocess
            for script in ("render_task_video.py", "render_map_video.py"):
                sp = ROOT / "scripts" / script
                if sp.exists():
                    r = subprocess.run([sys.executable, str(sp), str(out)], capture_output=True, text=True)
                    print(script, "exit", r.returncode, (r.stdout + r.stderr).strip()[-400:])
    print(json.dumps({k: summary[k] for k in ("task", "episode", "status", "success", "success_step",
                                              "scene_collision_free", "wall_s_policy", "sim_steps")}, default=str))
    print("model:", summary["model"])
    print("out:", out)
    if status.startswith("cap_"):
        sys.stdout.flush()
        os._exit(0)          # everything is written; do not wait for a model answer still in flight


if __name__ == "__main__":
    main()
