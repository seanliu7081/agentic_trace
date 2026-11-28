"""Centralized safety-margin mode switch (`--safety {default,surgical,relaxed}`) for controlled A/B ablations.

- `default`: 0% change to any constant or threshold in `agentic_trace`.
- `surgical`: ONLY applies the two empirically beneficial false-rejection fixes while keeping 100% of `v2`'s
  grasp/part-grasp prefilters, face plausibility vetoes, `StepGuard`, and `ContactDetector` at `default`:
    (1) fixes `CLOSE_WINDOW` narrow-side false rejection from RGB-D depth bloat (`(-0.022, 0.012)` with a
        `>= 42%` solid-fraction floor so corner slips are still caught before lift);
    (2) moderately trims TRACE's `perception` inflation (`5 mm -> 1.5 mm`, `SAFETY_SIGMA: 2 mm -> 1 mm`,
        `PERCEPTION sigma0: 3 mm -> 1.5 mm`) without touching `LADDER["guard"]` (`StepGuard`).
- `relaxed`: aggressively relaxes all 7 safety/veto layers across TRACE (`LADDER`, `LEVELS`, `REACH`),
  arm rollout / IK (`ROLLOUT`, `ARM`, `JOINT_SLACK`), runtime guards (`StepGuard`, `ContactDetector`,
  `FORCE_LIMIT`, `STALL_LAG`), and `Harness` grasp/actuate prefilters & `CLOSE_WINDOW`.
"""
from __future__ import annotations


def apply_safety_mode(mode: str = "default") -> dict:
    """Apply `mode` ('default' | 'surgical' | 'relaxed') to module-level tables and `Harness` flags in-place."""
    assert mode in ("default", "surgical", "relaxed"), mode
    if mode == "default":
        return {"mode": "default"}

    from . import articulate, body, guard, harness, levels, snapshot

    if mode == "surgical":
        # 1. Moderately trim TRACE perception padding ONLY (`LADDER["guard"]` / `StepGuard` stays 100% default!)
        levels.LADDER["perception"] = 0.0015
        levels.SAFETY_SIGMA = 0.001
        snapshot.PERCEPTION.update({
            "sigma0": 0.0015,
            "sigma_far": 0.0035,
        })
        levels.LADDER["look_floor"] = snapshot.PERCEPTION["sigma0"] + 1e-4

        # 2. Fix `CLOSE_WINDOW` narrow-side depth-bloat false rejection (with solid-ratio guard in `grasp_object`)
        harness.Harness.CLOSE_WINDOW = (-0.022, 0.012)
        harness.Harness.surgical_close_ratio = 0.42
        return {
            "mode": "surgical",
            "perception_m": levels.LADDER["perception"],
            "safety_sigma_m": levels.SAFETY_SIGMA,
            "sigma0_m": snapshot.PERCEPTION["sigma0"],
            "close_window": list(harness.Harness.CLOSE_WINDOW),
            "surgical_close_ratio": harness.Harness.surgical_close_ratio,
        }

    deg = levels.DEG
    # 1. TRACE safety ladder & model-facing constraint levels (`levels.py`)
    levels.LADDER["guard"].update({
        "arm": 0.001,
        "palm": 0.0,
        "finger": -0.002,
        "link": 0.001,
        "payload": -0.002,
    })
    levels.LADDER["guard_plane"].update({
        "arm": 0.002,
        "palm": 0.0,
        "finger": -0.003,
        "link": 0.002,
        "payload": -0.002,
    })
    levels.LADDER["perception"] = 0.0
    levels.LADDER["tube0"] = 0.003
    levels.LADDER["tube_k"] = 0.016
    levels.LADDER["tube_w"] = 0.002
    levels.SAFETY_SIGMA = 0.0005
    levels.REACH = 0.92
    levels.LEVELS["clearance"].update({
        "close": (0.003, 0.003),
        "normal": (0.010, 0.006),
        "wide": (0.035, 0.015),
    })
    levels.LEVELS["tilt_limit"].update({
        "upright": (22 * deg, 6 * deg),
        "tilted": (42 * deg, 10 * deg),
        "loose": (75 * deg, 15 * deg),
    })
    levels.LEVELS["approach"].update({
        "vertical": (25 * deg, 8 * deg),
        "steep": (48 * deg, 12 * deg),
    })

    # 2. Controller rollout & IK joint-limit slack (`body.py`)
    body.ROLLOUT["limit_margin"] = 0.002
    body.ARM.update({
        "err_p": 0.010,
        "err_r": 0.10,
        "at_limit": 0.001,
        "dq_max": 0.65,
    })
    body.JOINT_SLACK = 0.005

    # 3. Per-step geometric guard & residual contact detector (`guard.py`)
    guard.GUARD["joint_margin"] = 0.001
    guard.DETECT.update({
        "f_thr": 18.0,
        "e_min": 0.0025,
        "strong": 3.0,
        "hits": 4,
        "e_push": 0.0008,
    })
    guard.ContactDetector.__init__.__defaults__ = (
        guard.DETECT["f_thr"],
        guard.DETECT["e_min"],
        guard.DETECT,
    )

    # 4. Articulation force / rotary stall limits (`articulate.py`)
    articulate.FORCE_LIMIT = 65.0
    articulate.STALL_LAG = 35.0 * deg

    # 5. Harness static clearance dicts, joint slack, close-width window & inline veto flag (`harness.py`)
    harness.TRANSIT.update({
        "arm": 0.004,
        "palm": 0.002,
        "finger": 0.0,
        "payload": 0.001,
        "plane_arm": 0.004,
        "plane_palm": 0.001,
        "plane_finger": -0.002,
        "plane_payload": -0.001,
    })
    harness.NEAR.update({
        "arm": 0.002,
        "palm": 0.001,
        "finger": -0.002,
        "payload": -0.001,
        "plane_arm": 0.003,
        "plane_palm": 0.0,
        "plane_finger": -0.003,
        "plane_payload": -0.002,
    })
    harness.GRASP_NEAR.update({
        **harness.NEAR,
        "plane_finger": -0.004,
        "plane_palm": -0.001,
    })
    harness.GRASP_JOINT_SLACK = 0.005
    harness.Harness.CLOSE_WINDOW = (-0.045, 0.035)
    harness.Harness.relaxed_safety = True

    return {
        "mode": "relaxed",
        "ladder_guard": dict(levels.LADDER["guard"]),
        "close_window": list(harness.Harness.CLOSE_WINDOW),
        "grasp_joint_slack": harness.GRASP_JOINT_SLACK,
        "detect_f_thr": guard.DETECT["f_thr"],
        "force_limit": articulate.FORCE_LIMIT,
    }
