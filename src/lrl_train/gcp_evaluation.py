"""Procedural Step 1B frozen-actor interventions and physical tracking diagnostics."""
import gzip
import hashlib
import json
import multiprocessing as mp
import tempfile
from pathlib import Path

import numpy as np

from .gcp_devices import configure_egl_worker

TASKS = ("door", "drawer", "turn", "pick", "place", "planar_push")
PAIRS = ("door_extent", "door_reverse", "drawer_reverse", "turn_extent")
SLIPS = ("slip_positive", "slip_negative")
PROBES = PAIRS + SLIPS
VISIBLE, FIRM, SINGLE, SENSED, SLIP = 1, 2, 4, 8, 16
FLAG_LEGEND = {"registered_depth_visible": VISIBLE, "bilateral_firm": FIRM,
               "single_pad": SINGLE, "sensed_hold": SENSED, "slipping": SLIP}


def init_worker(checkpoint, gpus):
    global _actor, _gpu
    ident = mp.current_process()._identity
    _gpu = gpus[(ident[0] - 1 if ident else 0) % len(gpus)]
    configure_egl_worker(_gpu)
    import torch
    from .train_gcp import load_policy
    torch.set_num_threads(1)
    _actor = load_policy(checkpoint, device="cpu").requires_grad_(False)


def common_state(env, obs):
    """Hash complete physics, servo/tare, sensor RNG and every non-command actor input."""
    arrays = [env.ss.snapshot()["state"], env.core.R_des, env.core.p_des, env.fast_rb._grip_cur,
              env.fast_rb._q_init, env.observer.delta_tau0]
    arrays.extend(obs[k] for k in sorted(obs) if k != "goal")
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(json.dumps(env.rng.bit_generator.state, sort_keys=True).encode())
    return digest.hexdigest()


def measurement(env, info, prev):
    facts = info["contact_facts"]
    firm = bool(facts["firm"])
    rotation, tcp = env.fast_rb.eef_pose()
    body_rot = env.d.xmat[env.bid].reshape(3, 3)
    grip_pos = rotation.T @ (env.target_pos() - tcp)
    grip_rot = rotation.T @ body_rot
    closed = env.requested_width < .030
    slip = info["slip_lost"] and not info["regrasped"]
    if prev is not None and prev[0] and closed:
        # Ground-truth relative motion distinguishes sliding from optical occlusion.
        relative_change = np.linalg.norm(grip_pos - prev[1])
        cos_change = (np.trace(grip_rot @ prev[2].T) - 1.) / 2.
        slip |= not firm or relative_change > .003 or cos_change < np.cos(np.deg2rad(5.))
    flags = (VISIBLE * bool(info["depth_tracking_visible"]) + FIRM * firm +
             SINGLE * (bool(facts["left_target"]) != bool(facts["right_target"])) +
             SENSED * bool(env.sensed_hold) + SLIP * bool(slip))
    if env.free:
        est = env.tracker.transform(env.q_initial_world) * 1000.
        truth = env.target_pos() * 1000.
    else:
        scale = np.rad2deg(1.) if env.hinge else 1000.
        est = env.tracker.progress_value * scale
        truth = env.progress()[int(env.hinge)] * scale
    err = float(np.linalg.norm(np.asarray(est) - truth))
    frame = [int(info["steps"]), err, flags, np.asarray(est).tolist(), np.asarray(truth).tolist()]
    return frame, (firm, grip_pos, grip_rot)


def rollout(env, obs, info, capture_contact=False):
    """Ground truth is recorded after inference and never enters the action function."""
    import torch
    frames, contact = [], None
    previous = None
    goal_tail = 0
    start_state = common_state(env, obs)
    goal_head = obs["goal"].tolist()
    if info["entry_ok"]:
        frame, previous = measurement(env, info, previous)
        frames.append(frame)
        while info["steps"] < env.horizon:
            if capture_contact and contact is None and env.grasp_streak >= 3:
                angle = info["angle_deg"] * np.sign(env.contract.target_rot_rad)
                if 15. <= angle <= 18. and info["steps"] < env.horizon - 10:
                    contact = env.snapshot()
            with torch.inference_mode():
                actor_input = {k: torch.from_numpy(v).float().unsqueeze(0) for k, v in obs.items()}
                action = _actor(actor_input)[0].numpy()
            obs, _, done, info = env.step(action)
            frame, previous = measurement(env, info, previous)
            frames.append(frame)
            goal_tail = goal_tail + 1 if info["goal_satisfied"] else 0
            if done:
                break
    success = bool(goal_tail >= 10 and info["success"])
    row = dict(entered=bool(info["entry_ok"]), success=success, goal_tail=goal_tail,
               safe_success=success and info["max_pen_m"] < .001, max_pen_m=float(info["max_pen_m"]),
               actual_steps=len(frames) - 1 if frames else 0, fully_invisible=bool(frames) and
               not any(f[2] & VISIBLE for f in frames), frame_data=frames,
               state_sha256=start_state, initial_goal=goal_head,
               final_joint=None if env.free else float(env.progress()[int(env.hinge)]) *
                    (np.rad2deg(1.) if env.hinge else 1000.),
               injected=bool(info["slip_injected"]), lost=bool(info["slip_lost"]),
               regained=bool(info["regrasped"]), policy_recovered=bool(env.causal_recovered),
               place_grasp_jitter=info.get("place_grasp_jitter"))
    return row, contact


def probe_spec(panel, part):
    travel = part.open - part.closed
    sign = float(np.sign(travel))
    lo, hi = sorted(part.range)
    if panel == "turn_extent":
        # A common numerical positive command on both orientations.
        if hi - lo < np.deg2rad(65.) + .01 or abs(travel) < 1e-8:
            return None
        start = np.clip(lo + .005, min(part.closed, part.open), max(part.closed, part.open))
        if start + np.deg2rad(65.) > hi:
            return None
        return (start - part.closed) / travel, (30., 65.), 1.
    if part.handle_point_b is None:
        return None
    if panel == "drawer_reverse":
        if part.protrusion <= .012 or 2 * min(part.handle_half) >= .065 or abs(travel) < .102:
            return None
        return .5, (sign * .05, -sign * .05), sign
    extent = np.rad2deg(abs(travel))
    if panel == "door_reverse":
        if extent < 61.:
            return None
        return 35. / extent, (sign * 25., -sign * 25.), sign
    if extent < 75. or (panel in SLIPS and sign != (1. if panel == "slip_positive" else -1.)):
        return None
    return 0., (sign * 40., sign * 75.), sign


def trial(job):
    from .gcp_env import GCPEnv
    panel, idx, base_seed, n = job
    kind = "door" if panel in SLIPS else panel.split("_")[0] if panel in PAIRS else panel
    with tempfile.TemporaryDirectory() as folder:
        env = GCPEnv(base_seed, folder, gpu=_gpu, evaluation=True, staged_reset_prob=0.,
                     preset="all_5_families", horizon=130, place_grasp_jitter=True)
        def forbidden(*args, **kwargs):
            raise RuntimeError("A frozen-actor development rollout called the scripted expert")
        env.expert_action = forbidden
        try:
            result = dict(panel=panel, kind=kind, idx=idx, rejected_geometry=[], branches={})
            if panel in TASKS:
                seed = base_seed + idx
                obs, info = env.reset(seed=seed, kind=kind, staged=False)
                result.update(seed=seed)
                result["branches"]["natural"], _ = rollout(env, obs, info)
                return result
            for attempt in range(64):
                seed = base_seed + n * (1 + PROBES.index(panel) + len(PROBES)*attempt) + idx
                env._load(kind, seed)
                pi = np.random.default_rng(seed).integers(len(env.ss.parts)) if kind == "drawer" else 0
                part = env.ss.parts[pi]
                spec = probe_spec(panel, part)
                if spec is not None:
                    break
                result["rejected_geometry"].append(seed)
            else:
                return result
            initial, targets, sign = spec
            field = "target_trans_m" if kind == "drawer" else "target_deg"
            # Always approach the accessible handle for the paired causal test.
            primary = targets[-1] if panel in SLIPS else targets[0]
            obs, info = env.reset(seed=seed, kind=kind, staged=False, initial_fraction=initial,
                                  **{field: primary})
            if part.joint != env.part.joint:
                raise RuntimeError("Pre-drawn drawer joint differs from the evaluated joint")
            snap = env.snapshot()
            result.update(seed=seed, targets=list(targets), initial_fraction=initial, sign=sign,
                          handle=part.handle_kind, physical_joint=part.joint)
            states = []
            if panel in PAIRS:
                for name, target in zip(("a", "b"), targets):
                    obs, info = env.restore(snap, **{field: target})
                    run, _ = rollout(env, obs, info)
                    result["branches"][name] = run
                    states.append(run["state_sha256"])
            else:
                obs, info = env.restore(snap, slip_at_deg=None)
                intact, at_contact = rollout(env, obs, info, capture_contact=True)
                result["branches"]["intact"] = intact
                obs, info = env.restore(snap, slip_at_deg=15.)
                fresh, _ = rollout(env, obs, info)
                result["branches"]["fresh"] = fresh
                states.extend((intact["state_sha256"], fresh["state_sha256"]))
                if at_contact is not None:
                    obs, info = env.restore(at_contact, slip_at_deg=15.)
                    result["prepared_at_deg"] = info["angle_deg"]
                    result["prepared_at_step"] = info["steps"]
                    result["branches"]["prepared"], _ = rollout(env, obs, info)
            if len(set(states)) != 1:
                raise RuntimeError(f"{panel} seed {seed}: goal override changed physics or non-goal actor inputs")
            if panel in PAIRS and result["branches"]["a"]["initial_goal"] == result["branches"]["b"]["initial_goal"]:
                raise RuntimeError(f"{panel} seed {seed}: goal override failed to change the actor command")
            return result
        finally:
            env.close()


def rate(hits, total):
    if not total:
        return {"count": 0, "n": 0, "rate": None, "ci95": None}
    fraction = hits / total
    z = 1.959963984540054
    scale = 1 + z*z / total
    center = (fraction + z*z/(2*total)) / scale
    radius = z * np.sqrt(fraction*(1-fraction)/total + z*z/(4*total**2)) / scale
    return {"count": int(hits), "n": int(total), "rate": fraction,
            "ci95": [max(0., center - radius), min(1., center + radius)]}


def funnel(rows):
    count = len(rows)
    entered = sum(r["entered"] for r in rows)
    return {"entry_coverage": rate(entered, count), "conditional_success": rate(sum(r["success"] for r in rows), entered),
            "unconditional_safe_success": rate(sum(r["safe_success"] for r in rows), count),
            "penetration_over_1mm": rate(sum(r["max_pen_m"] > .001 for r in rows), count),
            "penetration_p95_mm": float(np.percentile([r["max_pen_m"]*1000. for r in rows], 95))}


def pair_summary(rows, panel):
    pairs = [r for r in rows if r["branches"]]
    entered = [r for r in pairs if r["branches"]["a"]["entered"] and r["branches"]["b"]["entered"]]
    hits = sum(all(b["success"] for b in r["branches"].values()) for r in entered)
    deltas = [(r["branches"]["b"]["final_joint"] - r["branches"]["a"]["final_joint"]) * r["sign"]
              for r in entered]
    direction = -1 if panel.endswith("reverse") else 1
    deltas = [d * direction for d in deltas]
    return {"starts": len(rows), "geometry_eligible": len(pairs), "both_entered": len(entered),
            "both_goals_met": rate(hits, len(rows)), "both_goals_met_given_entry": rate(hits, len(entered)),
            "endpoint_separation_median": float(np.median(deltas)) if deltas else None,
            "unit": "mm" if panel == "drawer_reverse" else "degrees",
            "expected_separation": {"door_extent": 35., "door_reverse": 50., "drawer_reverse": 100.,
                                    "turn_extent": 35.}[panel],
            "separation_definition": "Entering pairs only. Positive = ordered in commanded direction."}


def slips_summary(rows):
    result = {}
    for arm in ("fresh", "prepared"):
        branches = [r["branches"][arm] for r in rows if arm in r["branches"]]
        entered = [b for b in branches if b["entered"]]
        lost = [b for b in entered if b["lost"]]
        hits = sum(b["policy_recovered"] and b["safe_success"] for b in lost)
        result[arm] = dict(starts=len(rows), eligible_starts=len(branches), entered=len(entered),
                           availability=rate(len(entered), len(rows)), injected=sum(b["injected"] for b in entered),
                           actual_losses=len(lost), regained=sum(b["regained"] for b in lost),
                           voluntary_recovery=sum(b["policy_recovered"] for b in lost),
                           safe_completed_after_loss=hits, recovery_all_starts=rate(hits, len(rows)),
                           recovery_given_actual_loss=rate(hits, len(lost)))
    return result


def tracking(rows):
    grouped = {}
    for scope in ("all_actor_trajectories", "natural_starts"):
        grouped[scope] = {}
        for kind in TASKS:
            runs = [b for r in rows if r["kind"] == kind for b in r["branches"].values()
                    if (scope != "natural_starts" or r["panel"] == kind) and b["frame_data"]]
            tagged = [(f[1], f[2], b["fully_invisible"]) for b in runs for f in b["frame_data"]]
            masks = {"all": lambda f: True, "not_firm": lambda f: not f[1] & FIRM,
                     "single_pad": lambda f: f[1] & SINGLE, "slipping": lambda f: f[1] & SLIP,
                     "invisible": lambda f: not f[1] & VISIBLE,
                     "sensed_hold_without_firm": lambda f: f[1] & SENSED and not f[1] & FIRM}
            if kind == "turn":
                masks["fully_invisible_turn"] = lambda f: f[2]
            summary = {}
            for name, accept in masks.items():
                values = [f[0] for f in tagged if accept(f)]
                summary[name] = dict(frames=len(values), p95=float(np.percentile(values, 95)) if values else None,
                                     max=max(values, default=None))
            grouped[scope][kind] = dict(unit="degrees" if kind in ("door", "turn") else "mm",
                                        trajectories=len(runs), fully_invisible_trajectories=sum(b["fully_invisible"] for b in runs),
                                        strata=summary)
    return grouped


def run_panel(checkpoint, frame_path, n, workers, seed, gpus):
    if seed < 450000:
        raise ValueError("all_5_families development seeds must be >= 450000")
    jobs = [(panel, i, seed, n) for panel in TASKS + PROBES for i in range(n)]
    with mp.get_context("spawn").Pool(workers, init_worker, (checkpoint, gpus)) as pool:
        rows = sorted(pool.imap_unordered(trial, jobs), key=lambda r: (r["panel"], r["idx"]))
    metrics = {
        "task_funnel": {k: funnel([r["branches"]["natural"] for r in rows if r["panel"] == k]) for k in TASKS},
        "same_snapshot": {p: pair_summary([r for r in rows if r["panel"] == p], p) for p in PAIRS},
        "slip_recovery": {p: slips_summary([r for r in rows if r["panel"] == p]) for p in SLIPS},
        "tracking": tracking(rows)}
    tmp = Path(str(frame_path) + ".tmp")
    with gzip.open(tmp, "wt", compresslevel=3, encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, allow_nan=False) + "\n")
    tmp.replace(frame_path)
    success = [v["unconditional_safe_success"]["rate"] for v in metrics["task_funnel"].values()]
    paired = [v["both_goals_met"]["rate"] for v in metrics["same_snapshot"].values()]
    recovered = [v["fresh"]["recovery_all_starts"]["rate"] for v in metrics["slip_recovery"].values()]
    return dict(summary=metrics, trajectories=str(frame_path),
                frame_columns=["step", "absolute_error", "flags", "estimate", "truth"],
                frame_flags=FLAG_LEGEND, selection_key=[min(success), float(np.mean(success)),
                                                        float(np.mean(paired)), float(np.mean(recovered))],
                rule="Maximize worst then mean per-task unconditional safe success, then paired-command and fresh recovery rates.",
                protocol={"actor": "deterministic frozen CPU actor for all actions, including slip preparation",
                          "horizon": 130, "retention_steps": 10, "penetration_limit_m": .001,
                          "frame_population": "Initial observation and every post-action frame, including unsuccessful contacts; rejected entries have zero actor frames. Interventions and natural starts are also reported separately.",
                          "slipping": "Forced actual grip loss before regrasp, or a previously firm grasp under a continuing <30mm command loses firm contact or changes its true hand-relative pose by >3mm or >5deg in one tick.",
                          "slip_recovery": "Both pads lose contact, policy subsequently opens and closes, regains stable contact with gap reduction, and safely holds the commanded 75-degree goal for the final 10 steps.",
                          "geometry_screen": "Up to 64 predetermined disjoint geometry candidates per controlled pair, screened on joint range, accessible handle and prescribed sign before observing entry or policy output. The first eligible layout is used even if entry fails. No replacements for failed starts.",
                          "units": "World object centers in mm for free objects; signed motion from fixed handoff in mm or degrees for mechanisms."})
