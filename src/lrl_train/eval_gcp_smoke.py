"""Paired, pre-drawn Step 1A evaluation. Run with a frozen GCP PPO checkpoint."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .gcp_devices import parse_gpus, configure_egl_worker


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    files = ("atrace/gcp.py", "atrace/gcp_entry.py", "atrace/gcp_tracking.py", "atrace/gcp_labels.py",
             "atrace/robot_model.py", "atrace/geometry.py", "atrace/sim_iface.py", "lrl_train/gcp_env.py",
             "lrl_train/fast_sim.py", "lrl_train/_fast_osc.cpp", "lrl_train/_fast_osc.so",
             "lrl_train/render.py", "lrl_train/scenes.py", "lrl_train/ucs_scenes.py",
             "lrl_train/student_obs.py", "lrl_train/sim.py", "lrl_train/eval_gcp_smoke.py")
    hashes = {f: hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
    # Only the checkpoint loader is executed from the trainer. Its training
    # diagnostics can evolve independently of a frozen policy evaluation.
    source = (root/"lrl_train/train_gcp.py").read_text()
    loader = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="load_policy")
    hashes["lrl_train/train_gcp.py::load_policy"] = hashlib.sha256(ast.get_source_segment(source,loader).encode()).hexdigest()
    return hashes


def physics_hash(env):
    d = env.fast_rb.d
    return hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest()


def paired_state_hash(env, obs):
    """All branches share warmstart, servo, tactile tare, visual/proprio history and random state."""
    goal = obs["goal"].copy()
    goal[6] = 0.  # the angular instruction is the designated intervention
    state = env.ss.snapshot()["state"]
    arrays = (state, env.core.R_des, env.core.p_des, env.fast_rb._grip_cur, env.fast_rb._q_init,
              env.observer.delta_tau0, obs["scene_pts"], obs["payload_pts"], obs["history"], goal)
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps(env.rng.bit_generator.state, sort_keys=True).encode())
    return digest.hexdigest()


def paired_rgb(camera):
    """Evaluator-only RGB from the calibrated head and wrist; no actor RNG or physics calls."""
    import mujoco
    camera.gl.make_current()
    camera.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
    camera.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0
    frames = []
    for cid in camera.ids:
        camera.cam.fixedcamid = cid
        mujoco.mjv_updateScene(camera.m, camera.d, camera.opt, camera.pert, camera.cam,
                              mujoco.mjtCatBit.mjCAT_ALL, camera.scn)
        mujoco.mjr_render(camera.rect, camera.scn, camera.con)
        mujoco.mjr_readPixels(camera.rgb, None, camera.rect, camera.con)
        frames.append(camera.rgb[::-1].copy())
    return np.concatenate(frames,axis=1)


def visual_strip(frames, labels, out):
    from PIL import Image, ImageDraw

    if any(not strip for strip in frames):
        raise ValueError("No recorded frames in comparison")
    h, w = frames[0][0][0].shape[:2]
    size = (w * len(labels), h + 46)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", "10", "-i", "-",
           "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for step in range(max(map(len, frames))):
            canvas = Image.new("RGB", size)
            draw = ImageDraw.Draw(canvas)
            for col, (strip, label) in enumerate(zip(frames, labels)):
                frame, caption = strip[min(len(strip)-1, step)]
                canvas.paste(Image.fromarray(frame).resize((w, h)), (w * col, 0))
                draw.text((w * col + 4, h + 3), f"{label}    head | wrist", fill="white")
                draw.text((w * col + 4, h + 23), caption, fill="white")
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed for {out}")


def play(env, actor, obs, info, mask="", film=False):
    import torch
    frames, trajectory = [], []
    def record(info, policy_width=None, forced=False):
        facts = info.get("contact_facts", {})
        row = dict(t=int(info["steps"]), angle_deg=float(info["angle_deg"]),
                   travel_m=float(info["target_delta"]), goal_now=bool(info["goal_satisfied"]),
                   reference_remaining=float(env._target_distance()),
                   header_remaining_m=float(obs["goal"][5]), header_remaining_rad=float(obs["goal"][6]),
                   success=bool(info["success"]), slip=bool(info.get("slip_injected", False)),
                   regrasp=bool(info.get("regrasped", False)), left=bool(facts.get("left_target")),
                   right=bool(facts.get("right_target")), width_m=float(facts.get("width_m", 0.)),
                   cmd_width_m=float(facts.get("command_width_m", 0.)),
                   policy_width_m=policy_width, forced=forced, separation_m=float(facts["separation_m"]),
                   reference_joint_origin=float(env.initial_joint),
                   max_pen_m=float(info["max_pen_m"]))
        if info["kind"] == "drawer":
            part = env.part
            row["bottom_open_m"] = float((env.ss.joint(part.joint)-part.closed)*np.sign(part.open-part.closed))
            others = [pi for pi in env.ss.parts if pi.joint != part.joint]
            row["upper_open_fraction"] = min(float((env.ss.joint(pi.joint)-pi.closed)/(pi.open-pi.closed))
                                             for pi in others)
        trajectory.append(row)
        if film and row["t"] % 2 == 0:
            state = (f'{row["angle_deg"]:.1f} deg' if info["kind"]=="door"
                     else f'{1000*row["reference_remaining"]:.1f} mm left')
            text = f't={row["t"]/20:.1f}s    {state}    pen={1000*row["max_pen_m"]:.2f} mm'
            frames.append((film(), text))

    record(info)
    if info["entry_ok"]:
        initial_header = np.asarray(obs["goal"]).copy()
        done = False
        while not done:
            data = {k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in obs.items()}
            if mask in ("no_torque", "no_depth_or_torque"):
                data["history"] = data["history"].clone()
                data["history"][..., -8:] = 0.
            if mask in ("no_cloud", "no_depth_or_torque"):
                for k in ("scene_pts", "payload_pts"):
                    data[k] = torch.zeros_like(data[k])
                # The goal command survives, but a raw-cloud ablation alone
                # leaks online depth registration via remaining displacement
                # and the tracking-assisted holding bit. Drawer axis is static.
                data["goal"] = data["goal"].clone()
                data["goal"][...,5:10] = torch.from_numpy(initial_header[5:10])
                data["goal"][...,11] = float(initial_header[11])
            with torch.inference_mode():
                action = actor(data)[0].cpu().numpy()
            already_injected, release_left = info["slip_injected"], env.release_left
            obs, _, done, info = env.step(action)
            forced = release_left > 0 or (info["slip_injected"] and not already_injected)
            record(info, float(.04*(1.+action[7])), forced)
    return dict(entry=bool(info["entry_ok"]), final=trajectory[-1], trajectory=trajectory), frames


def trial(args):
    kind, idx, seed, path, out, film, gpus = args
    ident = multiprocessing.current_process()._identity
    gpu = gpus[(ident[0] - 1 if ident else 0) % len(gpus)]
    configure_egl_worker(gpu)
    from .render import Cameras
    from .gcp_env import GCPEnv
    from .train_gcp import load_policy
    import torch
    torch.set_num_threads(1)
    actor = load_policy(path, device="cpu")
    with tempfile.TemporaryDirectory() as td:
        env = GCPEnv(seed, td, gpu=gpu, staged_reset_prob=0., evaluation=True)
        camera = None
        try:
            opts = dict(seed=seed, kind=kind, staged=False)
            if kind == "door":
                opts["target_deg"] = 75.
            prepare = kind == "prepared_slip"
            if prepare:
                opts.update(kind="door", target_deg=75.)
            obs, info = env.reset(**opts)
            if prepare:
                # Select physical treatment eligibility before running the policy.
                # Both arms of the pair inherit the entire trajectory and time budget.
                for _ in range(env.horizon):
                    if not info["entry_ok"] or info["angle_deg"] > 18.:
                        break
                    if info["angle_deg"] >= 15. and info["contact_facts"]["firm"]:
                        break
                    obs, _, done, info = env.step(env.expert_action())
                    if done:
                        break
                if not (15. <= info["angle_deg"] <= 18. and info["contact_facts"]["firm"]
                        and info["steps"] < env.horizon-10):
                    return dict(kind=kind, idx=idx, seed=seed, eligible=False,
                                entry=info["entry_ok"], prep_angle_deg=info["angle_deg"])
            snap = env.snapshot()
            if film:
                camera = Cameras(env.m, env.d, size=256)
            digest = physics_hash(env)
            conditions = (("door40",40.,None,""),("door75",75.,None,""),("slip75",75.,15.,"")) if kind=="door" else (
                ("drawer",None,None,""),("no_torque",None,None,"no_torque"),("no_cloud",None,None,"no_cloud"),
                ("no_depth_or_torque",None,None,"no_depth_or_torque"))
            if prepare:
                conditions = (("intact",75.,None,""),("forced_slip",75.,15.,""))
            branches, clips = {}, {}
            paired_digest = None
            for label, angle, slip, mask in conditions:
                obs, info = env.restore(snap, target_deg=angle, slip_at_deg=slip)
                if physics_hash(env) != digest:
                    raise RuntimeError("Paired branch changed initial qpos/qvel")
                state_digest = paired_state_hash(env, obs)
                if paired_digest is not None and paired_digest != state_digest:
                    raise RuntimeError("Paired branches diverged before the first policy action")
                paired_digest = state_digest
                row, clips[label] = play(env, actor, obs, info, mask, (lambda: paired_rgb(camera)) if film else False)
                rows = row["trajectory"]
                if kind in ("door", "prepared_slip"):
                    tail = rows[-10:]
                    stable = len(tail)==10 and all(abs(r["angle_deg"] - angle)<=5 for r in tail)
                    matched_goal = stable and rows[-1]["success"]
                    pad_clear_open = not (rows[-1]["left"] or rows[-1]["right"]) and rows[-1]["cmd_width_m"]>.030
                    released = any(r["slip"] and not (r["left"] or r["right"]) for r in rows)
                    reacquired = [r for r in rows if r["regrasp"] and r["left"] and r["right"] and not r["forced"]]
                    regrasp = bool(reacquired)
                    open_close = returning = False
                    if regrasp:
                        first = reacquired[0]
                        autonomous = [r for r in rows if r["slip"] and not r["forced"] and r["t"] < first["t"]]
                        peak_gap = max((r["separation_m"] for r in rows if r["forced"]), default=0.)
                        returning = peak_gap-first["separation_m"] >= .005
                        open_close = (any(r["policy_width_m"] is not None and r["policy_width_m"]>.030
                                          for r in autonomous) and first["policy_width_m"]<.025)
                    origin_kept = all(r["reference_joint_origin"] == rows[0]["reference_joint_origin"] for r in rows)
                    forced_indices = [i for i,r in enumerate(rows) if r["forced"]]
                    tracking_error = None
                    if forced_indices:
                        window = rows[max(0,forced_indices[0]-1):forced_indices[-1]+13]
                        tracking_error = float(np.percentile([
                            abs(angle-np.rad2deg(r["header_remaining_rad"])-r["angle_deg"]) for r in window
                        ],95))
                        origin_kept = origin_kept and tracking_error<=5.
                    row.update(stable=stable, lost_after_injection=released, bilateral_regrasp=regrasp,
                               autonomous_width_open_close=open_close, returned_to_handle=returning,
                               final_pad_clear_and_open=pad_clear_open,
                               final_half_second_angle_range_deg=float(np.ptp([r["angle_deg"] for r in tail])),
                               reference_origin_kept=origin_kept, intervention_tracking_p95_error_deg=tracking_error)
                    row["passed"] = bool(row["entry"] and matched_goal and
                                         (slip is None or (released and regrasp and open_close and returning and origin_kept)))
                else:
                    if rows[0]["upper_open_fraction"] < .5:
                        raise RuntimeError("Protruding upper drawer fixture did not start open")
                    row["passed"] = bool(row["entry"] and len(rows)>=11 and all(
                        abs(r["bottom_open_m"]) <= env.contract.trans_tolerance and r["goal_now"]
                        for r in rows[-10:]))
                branches[label] = row
            if film:
                if prepare:
                    visual_strip(list(clips.values()), ["Intact grip at 15 deg", "Forced release at 15 deg"],
                                 out/"door_slip_staged.mp4")
                elif kind == "door":
                    visual_strip([clips["door40"], clips["door75"]], ["40 deg", "75 deg"], out/"door_goals.mp4")
                    visual_strip([clips["door75"], clips["slip75"]], ["75 deg", "75 deg | release at 15 deg"], out/"door_slip.mp4")
                else:
                    visual_strip(list(clips.values()), ["Full", "Torque residual masked", "Depth masked", "Both masked"],
                                 out/"drawer_comparison.mp4")
            return dict(kind=kind, idx=idx, seed=seed, initial_qpos_qvel_sha256=digest,
                        paired_physics_obs_control_rng_sha256=paired_digest,
                        eligible=bool(info["entry_ok"]), conditions=branches)
        finally:
            if camera is not None:
                camera.close()
            env.close()


def summarize(rows):
    scores = {}
    for name in ("door40","door75","slip75","drawer","no_torque","no_cloud","no_depth_or_torque"):
        trials = [r["conditions"][name] for r in rows if name in r["conditions"]]
        count, n = sum(r["passed"] for r in trials), len(trials)
        pens = [r["final"]["max_pen_m"] for r in trials]
        entered = [r["final"]["max_pen_m"] for r in trials if r["entry"]]
        scores[name] = dict(success=count, n=n, unconditional_rate=count/n, ci95=wilson95(count,n), entries=len(entered),
                            p95_trajectory_max_pen_m=float(np.percentile(pens,95)),
                            p95_entered_max_pen_m=float(np.percentile(entered,95)) if entered else None,
                            over_1mm_trials=sum(v>.001 for v in pens))
        if name in ("door40","door75","slip75"):
            scores[name]["success_with_clear_open_gripper"] = sum(r["passed"] and r["final_pad_clear_and_open"]
                                                                 for r in trials)
        if name=="slip75":
            scores[name].update(injections=sum(any(t["slip"] for t in r["trajectory"]) for r in trials),
                                witnessed_losses=sum(r["lost_after_injection"] for r in trials),
                                witnessed_regrasps=sum(r["bilateral_regrasp"] for r in trials),
                                voluntary_open_close=sum(r["autonomous_width_open_close"] for r in trials),
                                tracked_injections=sum(r["reference_origin_kept"] and
                                                       r["intervention_tracking_p95_error_deg"] is not None for r in trials),
                                returned_to_handle=sum(r["returned_to_handle"] for r in trials))
    d, s = scores["drawer"], scores["slip75"]
    gates = dict(goal40=scores["door40"]["unconditional_rate"]>=.85,
                 goal75=scores["door75"]["unconditional_rate"]>=.85,
                 slip_from_fresh_start=s["n"]>=50 and s["unconditional_rate"]>=.60,
                 drawer=d["unconditional_rate"]>=.85 and d["p95_trajectory_max_pen_m"] is not None
                    and d["p95_trajectory_max_pen_m"]<.001 and d["p95_entered_max_pen_m"]<.001)
    return scores, gates


def wilson95(k, n):
    if n == 0:
        return None
    z = 1.95996398454
    p = k/n
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    width = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,float(center-width)),min(1.,float(center+width))]


def paired_contrasts(rows):
    by_kind = {k:[r["conditions"] for r in rows if r["kind"] == k] for k in ("door","drawer")}
    delta = np.asarray([r["door75"]["final"]["angle_deg"]-r["door40"]["final"]["angle_deg"]
                        for r in by_kind["door"]])
    drawer = by_kind["drawer"]
    flags = {k:np.asarray([int(r[k]["passed"]) for r in drawer])
             for k in ("drawer","no_torque","no_cloud","no_depth_or_torque")}
    both, no_t, no_d, neither = [flags[k] for k in flags]
    return dict(door_both_commands_correct=sum(r["door40"]["passed"] and r["door75"]["passed"]
                                               for r in by_kind["door"]),
                door_mean_paired_delta_deg=float(delta.mean()),
                door_median_paired_delta_deg=float(np.median(delta)),
                drawer_rate_change_when_masking_torque=float(np.mean(no_t-both)),
                drawer_rate_change_when_masking_depth=float(np.mean(no_d-both)),
                drawer_paired_interaction=float(np.mean(both-no_t-no_d+neither)),
                drawer_full_only_vs_no_torque=int(np.sum((both==1)&(no_t==0))),
                drawer_no_torque_only_vs_full=int(np.sum((both==0)&(no_t==1))),
                depth_ablation="Zero both clouds, freeze registered remaining goal and vision-assisted holding bit at handoff; preserve commanded goal, proprioception, tracking errors and robot kinematics.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2"), help="Allowed physical EGL GPU indices")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=985402)
    p.add_argument("--isolated-slips", type=int, default=50,
                   help="Additional real interventions on independently prepared 15-degree grasp snapshots.")
    args = p.parse_args()
    if min(args.workers, args.n) < 1 or args.isolated_slips < 0:
        p.error("workers and n must be positive; isolated-slips must be nonnegative")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"smoke_protocol.json").exists():
        raise FileExistsError(f"Evaluation directory already contains a protocol: {args.out}")
    checkpoint = args.checkpoint.resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    starts = [(kind,i,args.seed+offset+i,checkpoint,args.out,i==0,args.gpus)
              for kind,offset in (("door",0),("drawer",1000000)) for i in range(args.n)]
    sources = source_hashes()
    (args.out/"smoke_protocol.json").write_text(json.dumps(dict(sha256=digest, source_sha256=sources, physical_gpus=args.gpus, camera_hz=20,
        success_tail_steps=10, slip_deg=15, aperture_tolerance_deg=5, trials=[s[:3] for s in starts],
        isolated_slip_trials=args.isolated_slips, isolated_slip_seed_base=args.seed+2000000,
        regrasp_evidence="Both pads lose contact. After the imposed release ends the policy commands >30 mm then <25 mm aperture, closes on both pads again, reduces handle separation by at least 5 mm, and reaches the final 75-degree goal. Reference origin stays fixed, and actor-estimated opening P95 error must be <=5 degrees through intervention + 12 subsequent steps.",
        isolated_selection="First N in seed order that the scripted prefix brings to 15..18 degrees with opposed pad contacts. Both treatment arms use the learned policy after that point.",
        video_selection="First pre-drawn fresh trial initially, first entering seed in order if it failed entry."),
        indent=2))
    rows = []
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool, (args.out/"smoke_trials.jsonl").open("w") as f:
        for row in pool.imap_unordered(trial, starts):
            rows.append(row)
            f.write(json.dumps(row, allow_nan=False)+"\n")
            f.flush()
        videos = {}
        for kind in ("door", "drawer"):
            group = sorted((r for r in rows if r["kind"]==kind), key=lambda r:r["idx"])
            chosen = next((r for r in group if r["eligible"]),group[0])
            videos[kind] = chosen["seed"]
            if chosen["idx"] and chosen["eligible"]:
                pool.apply(trial, ((kind,chosen["idx"],chosen["seed"],checkpoint,args.out,True,args.gpus),))
        staged = []
        prefixes = 0
        limit = args.isolated_slips * 20
        with (args.out/"slip_interventions.jsonl").open("w") as sf:
            while len(staged)<args.isolated_slips and prefixes<limit:
                jobs = [( "prepared_slip",i,args.seed+2000000+i,checkpoint,args.out,False,args.gpus)
                        for i in range(prefixes,min(limit,prefixes+min(args.workers,args.isolated_slips-len(staged))))]
                for row in pool.imap(trial,jobs):
                    prefixes += 1
                    if row["eligible"]:
                        staged.append(row)
                    sf.write(json.dumps(row,allow_nan=False)+"\n")
                    sf.flush()
        if staged:
            chosen = staged[0]
            videos["prepared_slip"] = chosen["seed"]
            pool.apply(trial, (("prepared_slip",chosen["idx"],chosen["seed"],checkpoint,args.out,True,args.gpus),))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Checkpoint changed during the evaluation")
    scores, gates = summarize(rows)
    if source_hashes() != sources:
        raise RuntimeError("Source files changed during the evaluation")
    result = dict(checkpoint=str(checkpoint), sha256=digest, source_sha256=sources, metrics=scores,
                  paired_contrasts=paired_contrasts(rows), gates=gates,
                  passed=all(gates.values()), video_seeds=videos,
                  penetration_population="P95 of measured per-trial maxima on all fixed starts, and separately on entries. Entry failures keep their measured initial penetration and count against success.",
                  trial_file="smoke_trials.jsonl")
    if args.isolated_slips:
        forced = [r["conditions"]["forced_slip"] for r in staged]
        n = len(staged)
        hits = sum(r["passed"] for r in forced)
        losses = sum(r["lost_after_injection"] for r in forced)
        gates["isolated_recovery"] = n>=50 and losses>=50 and hits/n>=.60
        result["passed"] = all(gates.values())
        result["isolated_15deg_recovery"] = dict(requested=args.isolated_slips,preparation_attempts=prefixes,
            eligible=n,injections=sum(r["final"]["slip"] for r in forced),witnessed_losses=losses,
            bilateral_regrasps=sum(r["bilateral_regrasp"] for r in forced),
            tracked_injections=sum(r["reference_origin_kept"] and
                                   r["intervention_tracking_p95_error_deg"] is not None for r in forced),
            voluntary_open_close=sum(r["autonomous_width_open_close"] for r in forced),
            returned_to_handle=sum(r["returned_to_handle"] for r in forced),success=hits,
            success_rate=hits/n if n else None,ci95=wilson95(hits,n),
            paired_intact_success=sum(r["conditions"]["intact"]["passed"] for r in staged),
            trial_file="slip_interventions.jsonl")
    result_file = args.out/"smoke_results.json"
    partial_file = result_file.with_suffix(".tmp")
    partial_file.write_text(json.dumps(result, indent=2, allow_nan=False))
    partial_file.replace(result_file)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
