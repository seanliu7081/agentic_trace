"""Level table, term compiler, harness safety terms and the margin ladder (DESIGN §3.4).

The plan call names levels ("upright", "wide"); this table owns the numbers, so the model never writes a tolerance
in metres and can neither create nor weaken a safety term. Every value is a general default (configurable, none
fitted to a task or an episode).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from traceplan.contract import Term

from .snapshot import PERCEPTION

DEG = np.pi / 180.0

# model-facing levels: prim -> level -> (parameter value, sigma), in the parameter's unit (PARAM)
LEVELS = {
    "tilt_limit": {"upright": (10 * DEG, 3 * DEG), "tilted": (25 * DEG, 5 * DEG), "loose": (45 * DEG, 8 * DEG)},
    "clearance": {"close": (0.01, 0.005), "normal": (0.03, 0.01), "wide": (0.08, 0.02)},
    "approach": {"vertical": (10 * DEG, 4 * DEG), "steep": (25 * DEG, 6 * DEG)},
}
MODEL_LEVELS = [lv for table in LEVELS.values() for lv in table]
# parameter name, unit, and which way is looser: +1 an upper bound (looser = larger), -1 a distance (looser = smaller)
PARAM = {"tilt_limit": ("max", "rad", 1), "clearance": ("margin", "m", -1), "approach": ("half_angle", "rad", 1)}
KNOWN_SPACE = ("strict", "slow")      # harness-only levels of the unknown-space term

# default margin ladder: TRACE plans with guard + tube + perception, the per-step guard stops at guard alone. The tube
# (DESIGN values) holds the feedforward executor's measured dev-scene tracking (distance to the path: <= 1 mm on
# straight 20 cm moves at 0.2-0.4 m/s, <= 3 mm on 15 cm-radius arcs at 0.3-0.4 m/s, at most 39 % of the tube)
LADDER = {"guard": {"arm": 0.010, "palm": 0.008, "finger": 0.005, "link": 0.010, "payload": 0.005},
          "guard_plane": {"arm": 0.015, "palm": 0.008, "finger": 0.004, "link": 0.015, "payload": 0.003},
          "perception": 0.005, "tube0": 0.004, "tube_k": 0.022, "v_min": 0.05,
          # hand rotation drifts the position through the OSC's coupling (t00 e2e: 5-9 mm on in-place wrist turns
          # at 1.2 rad/s against a 4-5 mm speed-only tube), so the tube also grows with the angular rate
          "tube_w": 0.003, "w_min": 0.15,
          # a closer look cannot shrink an entity's uncertainty below the perception model's floor: an end window
          # flagged at the floor is closed by contact (guarded move, gripper closure), not by another look
          "look_floor": PERCEPTION["sigma0"] + 1e-4}
LIMITS = {"v_max": 0.40, "a_max": 0.5, "w_max": 1.2, "dt": 0.05, "v_slow": 0.05}
LIMITS_HOLDING = {**LIMITS, "v_max": 0.20, "a_max": 0.25}
ZONE = 0.05                          # "do not get closer" zone at both ends of a segment (m)
GOAL_TOL = (0.015, 6 * DEG)          # standoff ends: what the arm reaches under load, not the note's 5 mm
SAFETY_SIGMA = 0.002                 # safety margins carry +sigma, so v <= 1 still guarantees the ladder distance
SHOULDER = np.array([0.0, 0.0, 0.333])   # Panda shoulder above the base
REACH = 0.855
REG_SIGMA = {"smooth_pos": 0.5, "smooth_rot": 1.5}


def tube(ladder, v):
    """Smallest tube the retimer can fall back to at speed v (rotation at its slowest rate)."""
    return ladder["tube0"] + ladder["tube_k"] * v + ladder["tube_w"] * ladder["w_min"]


def entity_id(h, sid):
    """Plan object id -> snapshot entity id (the first plan id bound to the same tracked object); unknown ids stay
    as written so the program tests report them."""
    oid = h.id_map.get(sid)
    if oid is None:
        return sid
    return next(s for s, o in h.id_map.items() if o == oid)


def compile_terms(stage_terms, h, phase, holding, approach=None, entity=None, zone=ZONE):
    """Plan terms -> contract Terms (src "language"). A tilt limit applies to the held object while carrying it;
    an approach term applies to a segment that ends at a contact pose of its object (approach axis given); a
    clearance term keeps every body away from its object only. Term ids lang<i>_<prim>_<of> name the stage term."""
    out = []
    for i, t in enumerate(stage_terms):
        prim, ent = t["prim"], entity_id(h, t["of"])
        val, sig = LEVELS[prim][t["level"]]
        name, unit, _ = PARAM[prim]
        common = dict(role=t["role"], sigma=sig, kappa=t.get("kappa", 1.0), src="language", level=t["level"],
                      relax_to=t["relax_to"], param=name, unit=unit)
        tid = f"lang{i}_{prim}_{t['of']}"
        if prim == "tilt_limit":
            if phase != "carry" or ent != holding:
                continue
            # the object's up at grasp, carried into the current hand orientation
            u0 = h.rb.eef_pose()[0] @ h.checker.attached.up_site
            out.append(Term(tid, "tilt_limit", {"u": "payload.up", "v": "world.up", "max": val, "u0": u0},
                            "ineq", **common))
        elif prim == "clearance":
            out.append(Term(tid, "clearance", {"body": "all", "field": {"exclude": [], "only": ent, "unknown": False},
                                               "margin": val, "zone": zone},
                            "ineq", entity=ent, effect="repel", **common))
        elif approach is not None and ent == entity:
            out.append(Term(tid, "approach_cone", {"axis": np.asarray(approach, float), "half_angle": val, "zone": zone},
                            "ineq", **common))
    return out


def safety_terms(h, phase, ladder=LADDER, zone=ZONE, holding=None):
    """The harness's own terms: per-group clearance to occupied space and to the table plane (safety), the reach
    limit (safety), smoothness regularizers (prefer), and unknown space as a require term relaxable to slow."""
    groups = ["palm", "finger", "link"] + (["payload"] if holding is not None else [])
    excl = [holding] if holding is not None else []
    tv = tube(ladder, ladder["v_min"])
    tz = h.scene.table_z
    out = []
    for g in groups:
        out.append(Term(f"clear_{g}", "clearance",
                        {"body": g, "field": {"exclude": excl, "only": None, "unknown": False},
                         "margin": ladder["guard"][g] + tv + ladder["perception"] + SAFETY_SIGMA, "zone": zone},
                        "ineq", "safety", sigma=SAFETY_SIGMA, src="perception", param="margin", unit="m"))
        out.append(Term(f"plane_{g}", "plane_clearance",
                        {"body": g, "z": tz, "margin": ladder["guard_plane"][g] + tv + SAFETY_SIGMA, "zone": zone},
                        "ineq", "safety", sigma=SAFETY_SIGMA, src="perception", param="margin", unit="m"))
        out.append(Term(f"known_{g}", "clearance",
                        {"body": g, "field": {"exclude": excl, "only": None, "unknown": True},
                         "margin": ladder["guard"][g] + SAFETY_SIGMA, "zone": zone},
                        "ineq", "require", sigma=SAFETY_SIGMA, src="perception", level="strict", relax_to="slow",
                        param="margin", unit="m"))
    out.append(Term("reach", "reach_limit", {"center": h.rm.root_p + SHOULDER, "r_max": REACH}, "ineq", "safety",
                    sigma=0.01, src="robot", param="r_max", unit="m"))
    out.append(Term("smooth_pos", "smooth_pos", {}, "eq", "prefer", sigma=REG_SIGMA["smooth_pos"], src="regularizer",
                    unit="m"))
    out.append(Term("smooth_rot", "smooth_rot", {}, "eq", "prefer", sigma=REG_SIGMA["smooth_rot"], src="regularizer",
                    unit="rad"))
    return out


def table_key(term):
    if term.level in KNOWN_SPACE:
        return "known_space"
    return "approach" if term.prim == "approach_cone" else term.prim


def can_relax(quote_entry, term):
    """Whether the level the plan allowed in advance (term.relax_to) admits the quoted parameter value. Crossing
    unknown space can always become slow guarded motion."""
    if term.relax_to is None:
        return False
    key = table_key(term)
    if key == "known_space":
        return True
    val = LEVELS[key][term.relax_to][0]
    return val >= quote_entry["needed"] if PARAM[key][2] > 0 else val <= quote_entry["needed"]


def relax(term, limits=LIMITS):
    """The term at its relax_to level (no further relaxation without the model). known_space -> slow: prefer, and
    retiming caps the speed where it is violated."""
    if table_key(term) == "known_space":
        return replace(term, role="prefer", level="slow", relax_to=None, args={**term.args, "slow": limits["v_slow"]})
    key = table_key(term)
    val, sig = LEVELS[key][term.relax_to]
    return replace(term, args={**term.args, PARAM[key][0]: val}, sigma=sig, level=term.relax_to, relax_to=None)
