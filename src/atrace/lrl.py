"""lrl in the harness (LRL_PLAN.md M1): the parts of `--placement lrl` and the contact stages that move a part.

LrlMixin holds methods of the Harness (self.rb, self.checker, self.scene, self.reach, ...). With the part "skill":

- push and pull (Harness._actuate_linear hands each attempt to lrl_move): every face hypothesis of the part is a
  source of candidates, in v2's order (v2's plausibility vetoes order the faces, they do not drop them). On a face a
  push takes the hold of the grip search, then the press search's contact points and hand poses in the order of what
  closed a part in the stage simulator; a pull takes the holds of the grip search. The grip search (_lrl_grip) is
  v2's without its vetoes on what the contract allows: the hand may meet the part and its fixture (plan 5.1).
- rotate (Harness._actuate hands the stage to lrl_turn): every axis hypothesis (v2's mount frame first, then the
  normals of the other planes fitted around the part) with a grip on the part, each turning sense a candidate.

Each candidate: TRACE to the entry, the guarded line to the touch, then lrl_skill.Skill (the trained policy when
"policy" is on and the policy knows the mode, else the base law) under the per-step guard. What the contact did is
the detector's label ("detector" on) or the interim rule's (interim_label). A label in CFG["marking"] marks the
candidate and its near-duplicates (no_effect: the whole face or axis) and the next candidate is taken, with no
model call; a stage takes at most CFG["candidates"]. "reached" is confirmed by v2's observation after leaving and
its verdict; when they disagree both are logged and the candidate is marked.
"""
from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np

from . import articulate, manip
from .body import hand_spheres, payload_spheres
from .geometry import frame_from_axes
from .guard import StepGuard
from .levels import LADDER
from .lrl_contract import LIMITS, Contract, Entry, Goal
from .lrl_skill import CFG as SKILL, Skill
from .sim_iface import HEAD_CAMERA, WRIST_CAMERA

PARTS = ("skill", "policy", "detector")     # what --lrl-parts can switch on; a part that is off leaves v2's code
TRAINED = ("policy", "detector")            # parts read from weights/; both act inside the skill loop
KIND = "move_part"                          # the contact class of the weights this module loads

CFG = {
    # Entry order until the scorer exists: first-attempt closing by hand pose on the two task cabinets (method
    # base_touch, all error levels): tip 98-100 %, tip_lean 81-93 %, tip_steep 24-33 %, side 20-31 %.
    "order": ("tip", "tip_lean", "tip_steep", "side"),    # M  runs/lrl_m0_20261125/reference/REFERENCE.md
    # A hold on the part before a press, as v2 has it. The stage simulator has not measured grip against press yet.
    "kinds": ("grip", "press"),                           # T
    "time": 8.0,             # s, longest run of the skill (the reference's base_touch; the trainer's env time).    T
    # Candidates of a stage (LRL_PLAN.md 4, 5.6).
    "candidates": 8,         # candidates entered per stage (TRACE asked for an entry).                            T
    "iterations": 24,        # passes of the face loop per stage (faces without a candidate cost a pass).          T
    "marking": ("no_effect", "no_contact", "outside"),    # labels that mark a candidate and take the next.     T
    "dup_radius": 0.02,      # m, a candidate of the same family this near a marked one is its near-duplicate.     T
    "dup_angle": 10.0,       # deg, normals or axes this close are one (as Harness._same_face).                    T
    "open_travel": 0.15,     # m, travel wanted of an opening whose extent is unmeasured (training 0.03-0.22).     T
    "turn_travel": float(articulate.MAX_TURN),   # rad, travel wanted of a turn (v2's quarter turn per grasp).     T
    # Interim rule, while no detector is loaded: when to run again in place, and what a contact did.
    "again_travel": 0.005,   # m the part must have moved in a run for another run from where the hand is.         T
    "again_hand": 0.005,     # m the hand must have moved in a run for another run in place.                       T
    "turn_follow": float(articulate.MIN_FOLLOW),  # rad, a turn the hand followed this far moved the part (v2).     T
    "runs": 3,               # runs of the skill per candidate.                                                    T
    # The press search's candidates and approach, as harness._press_stroke has them (the reference measured
    # exactly these poses, so they are the setting the order above holds for).
    "offsets": ((0, 0), (0.015, 0), (-0.015, 0), (0, 0.015), (0, -0.015), (0, 0.03), (0, -0.03)),   # m       T
    "rolls": (0.0, np.pi / 2, np.pi / 4, -np.pi / 4),     # rad about the approach.                          T
    "steep": (65.0, 80.0),   # deg, fingertips tilted down onto the face.                                    T
    "lean": (30.0, 50.0),    # deg.                                                                          T
    "side_dz": (0.01, -0.01, -0.025),   # m, fingertip below the contact point, finger sides pushing.        T
    "side_out": 0.018,       # m, fingertip in front of the face in a side pose.                             T
    "side_ok": 0.7,          # the tilted families exist when the face is this far from level.               T
    "tip_gap": 0.002,        # m, fingertips off the face at the touch pose.                                 T
    "failed_radius": 0.008,  # m, contact points this close to one whose transit failed are skipped.         T
    "standoff": 0.06,        # m, pre-pose back along the approach.                                          T
    "tip_over": 0.01,        # m, a tip approach aims this far past the touch pose.                          T
    "speed": {"tip": 0.04, "side": 0.05},                 # m/s of the guarded line.                         T
    "contact_force": 8.0,    # N, wrist force change that ends the guarded line.                             T
    "arrive": 0.03,          # m past the standoff: a transit stopped by contact this near the touch arrived. T
    "arrive_stops": ("contact", "tube_exit"),   # stops of a certified path that mean the hand met something. T
    "grip_back": 0.08,       # m, a gripping hand's standoff and back-off (harness._grip_stroke).            T
    "grip_speed": 0.05,      # m/s of the guarded line of a grip (harness._grip_stroke).                     T
    "grip_contact": 10.0,    # N, wrist force change that ends it (harness._grip_stroke).                    T
    "grip_empty": 0.002,     # m, pads closed to less than this hold nothing (harness._grip_stroke).         T
    "grip_depth": 0.07,      # m, part points this deep behind its outer level are graspable (plan_part_grasp). T
    "grip_tries": 30,        # grasps visited after the prefilter (plan_part_grasp).                         T
    "search_s": 30.0,        # s, a candidate search gives up (harness.PRESS_SEARCH_S).                      T
    "back": 0.06,            # m, back-off along the approach after the contact.                             T
    "side_lift": 0.02,       # m, a side pose first comes this far off the face.                             T
    "back_speed": 0.12,      # m/s.                                                                          T
    "fixture_skin": 0.01,    # m, unlabelled surface this close to the fixture's is the fixture (as v1).     T
    "turn_back": np.deg2rad(2.0),   # rad, a turn left this far from where it started is turned back (v2).   T
    "again_turn": np.deg2rad(5.0),  # rad the hand must have turned in a run for another run in place.        T
    "limit_slack": 0.10,     # rad from a joint limit: the posture TRACE cannot leave (body.JOINT_SLACK).    T
}


def parse_parts(text):
    """--lrl-parts: a comma list of PARTS. The empty list switches every part off, and lrl then runs v2's code.
    The trained parts act inside the skill loop, so they need "skill"."""
    parts = tuple(p.strip() for p in text.split(",") if p.strip())
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        raise ValueError(f"unknown lrl part {unknown}: the parts are {PARTS}")
    if "skill" not in parts and any(p in parts for p in TRAINED):
        raise ValueError(f"lrl parts {parts}: policy and detector run inside the skill loop, add 'skill'")
    return parts


def load_weights(weights_dir, parts):
    """The trained parts that are on, from the manifest entries marked for M1 (lrl_models.find_weights):
    {"dir", "policy": Policy | None, "detector": Detector | None}. A part that is on and has no marked entry is an
    error: running it would silently be another configuration."""
    from . import lrl_models
    found = lrl_models.find_weights(weights_dir, KIND)
    out = {"dir": str(weights_dir), "policy": None, "detector": None}
    for part, load in (("policy", lrl_models.load_policy), ("detector", lrl_models.load_detector)):
        if part in parts:
            if found[part] is None:
                raise FileNotFoundError(f"lrl part {part} is on, but {weights_dir}/manifest.json marks no {KIND} "
                                        f"{part} with \"m1\": true")
            out[part] = load(found[part])
    return out


def weights_record(w):
    """What summary.json keeps of the loaded weights: path and sha256 of each."""
    w = w or {}
    rec = {"dir": w.get("dir")}
    for part in TRAINED:
        m = w.get(part)
        rec[part] = None if m is None else {"path": m.path, "sha256": m.sha256,
                                            **({"modes": list(m.modes)} if part == "policy" else
                                               {"threshold": dict(m.threshold)})}
    return rec


def build_suffix(w):
    """The weights' part of the build string: '+p<sha8>' for the policy, '+d<sha8>' for the detector."""
    w = w or {}
    return "".join(f"+{part[0]}{w[part].sha256[:8]}" for part in TRAINED if w.get(part) is not None)


def is_marked(marked, family, point, normal, cfg=CFG):
    """A near-duplicate of a marked candidate: same family, contact point within the capture range, same normal."""
    cos = np.cos(np.deg2rad(cfg["dup_angle"]))
    return any(f == family and np.linalg.norm(np.asarray(point) - c) < cfg["dup_radius"] and float(np.dot(normal, m)) > cos
               for f, c, m in marked)


def interim_label(mode, stop, part_m=None, met=None, peak_n=0.0, gap=None, turned=None, cfg=CFG):
    """What a contact did, without a detector, in the detector's classes: from the skill's stop, the part's travel
    read in place (push, pull; m along the goal, None when the views gave none), whether the reading shows the goal
    state (met), the peak force, the pad gap at the end (grips) and the hand's turn about the believed axis (turn,
    rad). Not measured is 'short': v2's observation after leaving decides."""
    if stop == "guard":
        return "outside"
    if mode == "turn":
        if gap is not None and gap < cfg["grip_empty"]:
            return "lost_grip" if turned is not None and abs(turned) >= cfg["turn_follow"] else "no_contact"
        return "reached" if turned is not None and abs(turned) >= cfg["turn_follow"] else "no_effect"
    if mode == "pull" and gap is not None and gap < cfg["grip_empty"]:
        return "no_contact"
    if part_m is None:
        return "short"
    if part_m >= cfg["again_travel"]:
        return "reached" if met else "short"
    return "no_contact" if peak_n < cfg["contact_force"] else "no_effect"


def press_point(P, n):
    """Centre of a part's outer face and the lateral window of a press on it (as harness._press_stroke: the middle
    of the extent of the slab within 1 cm of the outer level; half the face's smaller extent)."""
    proj = P @ n
    slab = P[proj >= np.percentile(proj, 90) - 0.01]
    Qm = articulate.mount_rotation(n)
    Ls = (slab - P.mean(0)) @ Qm.T
    lo, hi = np.percentile(Ls[:, :2], 5, axis=0), np.percentile(Ls[:, :2], 95, axis=0)
    return P.mean(0) + np.r_[0.5 * (lo + hi), np.median(Ls[:, 2])] @ Qm, 0.5 * float(np.min(hi - lo))


def candidates(c0, n, failed=(), marked=(), cfg=CFG):
    """Entry candidates of a push on a face with outward normal n and centre c0: the press search's contact points
    and hand poses, ordered by family (cfg order), then point, then roll. Each is (family, mode, approach, R, grip
    site position at the touch, contact point); mode 'tip' = closed fingertips on the face, 'side' = the hand
    pointing down in front of it. Points next to a failed transit and near-duplicates of marked candidates are out."""
    n = np.asarray(n, float)
    u = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    down = np.array([0.0, 0.0, -1.0]) + n * n[2]
    tilted = np.linalg.norm(down) > cfg["side_ok"]
    down = down / max(np.linalg.norm(down), 1e-9)
    reach = manip.TIP_Z + cfg["tip_gap"]
    out = []
    for du, dv in cfg["offsets"]:
        c = c0 + du * u + dv * v
        if any(np.linalg.norm(c - f) < cfg["failed_radius"] for f in failed):
            continue
        tips = [("tip", 0.0)]
        if tilted:
            tips += [("tip_steep", t) for t in cfg["steep"]] + [("tip_lean", t) for t in cfg["lean"]]
            if dv == 0:
                out += [("side", "side", down, frame_from_axes(down, x), c + n * cfg["side_out"]
                         + down * (dz - manip.TIP_Z), c) for dz in cfg["side_dz"] for x in (u, n)]
        for fam, t in tips:
            a = np.cos(np.deg2rad(t)) * -n + np.sin(np.deg2rad(t)) * down
            out += [(fam, "tip", a, frame_from_axes(a, np.cos(r) * u + np.sin(r) * v), c - a * reach, c)
                    for r in cfg["rolls"]]
    out = [k for k in out if not is_marked(marked, k[0], k[5], n, cfg)]
    return sorted(out, key=lambda k: cfg["order"].index(k[0]))        # stable: points and rolls keep their order


def stroke_stop(trace):
    """The skill's stop in the words of articulate.translate, which Harness._linear_verdict reads: a force stop is
    its force_limit; a contact that ran out of time with the hand at rest is its stalled (same window, same
    distance). Anything else keeps the skill's word and counts as not resisted."""
    if trace.stop == "force":
        return "force_limit"
    p = trace.arrays()["p"]
    if trace.stop in ("time", "done") and len(p) >= 12 and np.linalg.norm(p[-1] - p[-12]) < articulate.STALL_DISP:
        return "stalled"
    return trace.stop


def turned_about(trace, d, dt=SKILL["dt"]):
    """The hand's measured turn about d (rad) since the touch, summed from the angular velocity so that a turn past
    180 degrees does not wrap (as the skill sums it)."""
    a = trace.arrays()
    return float((a["w"][trace.touch:] @ np.asarray(d, float)).sum() * dt) if len(trace.rows) else 0.0


class LrlMixin:
    """Methods of the Harness for `--placement lrl`."""

    def lrl_setup(self, placement, parts=(), weights=None):
        """weights: load_weights' result (the trained parts that are on), or None."""
        self.lrl = placement == "lrl"                       # every branch that is not an lrl part behaves as v2
        self.lrl_parts = tuple(parts) if self.lrl else ()
        assert set(self.lrl_parts) <= set(PARTS), self.lrl_parts
        w = weights or {}
        self.lrl_policy = w.get("policy") if "policy" in self.lrl_parts else None
        self.lrl_detector = w.get("detector") if "detector" in self.lrl_parts else None
        for part in TRAINED:
            assert (part in self.lrl_parts) == (getattr(self, f"lrl_{part}") is not None), \
                f"lrl part {part}: switched on and loaded must agree"
        self.lrl_weights = weights_record(w) if self.lrl else None

    def lrl_entered(self, result):
        """Candidates this stage has entered so far."""
        return self._lrl_stage(result)["entered"]

    def _lrl_stage(self, result):
        """What the stage has learned: marked candidates (family, contact point, normal or axis), faces where the grip
        search found no hold, candidates entered. One record per stage (per result dict)."""
        m = getattr(self, "_lrl_mem", None)
        if m is None or m["result"] is not result:
            m = self._lrl_mem = {"result": result, "marked": [], "no_hold": [], "entered": 0, "escaped": False}
        return m

    def lrl_mark(self, result, stroke, why, scope="candidate", **info):
        """Mark a candidate: it and its near-duplicates are not entered again in this stage."""
        mem = self._lrl_stage(result)
        mem["marked"].append((stroke["family"], np.asarray(stroke["point"], float), np.asarray(stroke["normal"], float)))
        self.log("lrl_candidate_marked", stage=result["stage"], family=stroke["family"], point=stroke["point"],
                 normal=stroke["normal"], label=stroke.get("label"), why=why, scope=scope, marked=len(mem["marked"]),
                 **info)

    # ------------------------------------------------------------------------------------- push and pull
    def lrl_move(self, motion, part, parent, mf, failed, result, want, remaining, front_pt, fixture_pt):
        """One candidate of a push or pull stage on the face mf, in place of v2's grip stroke and press stroke, with
        _press_stroke's returns: the stroke (with the contact's label); 'no_plan' when this face has no candidate
        left; None when the transit failed (the point joins `failed`). want: 'closed' | 'open' | None. remaining: how
        far the part stands out (m) or None; front_pt / fixture_pt: points on the part's measured front and on the
        fixture face, or None."""
        n = np.asarray(mf["n"], float)
        s = -1.0 if motion == "push" else 1.0                 # the part moves along s * n
        mem = self._lrl_stage(result)
        P = part.points
        if len(P) < 20:
            result["attempts"].append({"status": "no_feasible_grip_or_press", "why": "no part model"})
            return "no_plan"
        fix_oids, fix_pts = self._fixture(part, part)
        allowed = (part.oid,) + fix_oids
        own = [(fix_pts, CFG["fixture_skin"], True)] if len(fix_pts) else []
        kinds = CFG["kinds"] if motion == "push" else ("grip",)
        cand, why, blame = None, {}, {}
        for kind in kinds:
            if kind == "grip":
                if any(self._same_face(f, mf) for f in mem["no_hold"]):
                    # a part moving into its fixture has less room around it, not more: a search that found no
                    # hold is not repeated on this face (it costs the search time at every candidate)
                    why["grip"] = {"no_hold_earlier": 1}
                    continue
                cand = self._lrl_hold(part, parent, mf, failed, motion, want, remaining, fixture_pt, why, allowed,
                                      own, mem["marked"])
                if cand is None:
                    mem["no_hold"].append(mf)
            else:
                c0, win = press_point(P, n)
                cand, w, blame = self._lrl_entry(part, parent, n, c0, failed, mem["marked"], allowed, own)
                why.update(w)
                cand = None if cand is None else dict(zip(("family", "mode", "a", "R", "p", "c", "q_pre"), cand),
                                                      win=win, finger=0.0)
            if cand is not None:
                break
        self.log("lrl_entry", stage=result["stage"], motion=motion, candidate=mem["entered"] + (cand is not None),
                 family=None if cand is None else cand["family"], passed=why, normal=n, face_source=mf.get("source"),
                 face_veto=mf.get("why_not"), point=None if cand is None else cand["c"],
                 approach=None if cand is None else cand["a"],
                 allowed=[self.objects[o].name for o in allowed], fixture_surface_points=int(len(fix_pts)))
        if cand is None:
            result["attempts"].append({
                "status": "no_feasible_grip" + ("_or_press" if motion == "push" else ""), "press_blame": blame,
                "rejected": {f"{f}_{k}": v for f, w in why.items() for k, v in w.items()}})
            return "no_plan"
        mem["entered"] += 1
        fam, mode, a, p_touch, c, f_open = (cand[k] for k in ("family", "mode", "a", "p", "c", "finger"))
        grip = mode == "grip"
        width = 2 * (f_open + manip.PAD_OFFSET)                              # pad gap of the approach (grip)
        stops = self._lrl_stops()
        if grip:
            self.ex.gripper_to(width)
            end = self.reach(cand["R"], p_touch, cand["q"], a, part.oid, (width - cand["g"].width) / 2, "to_part",
                             0.0, f_open, cand["q_pre"])
        else:
            self.ex.gripper(1.0, max_steps=25)                              # closed hand
            end = self.reach(cand["R"], p_touch, self._ik_near(cand["R"], p_touch, cand["q_pre"]), a, part.oid,
                             cand["win"], "to_press", 1.0, 0.0, cand["q_pre"])
        # the wrist before anything loads the hand: a run in place later tares against it (the grip's pads, a
        # press's contact load are not zero any more by then)
        w_free = tuple(np.array(x, float) for x in self.rb.wrench())
        if end is None:
            # A certified path that ended on contact next to the believed touch has brought the hand to the part
            # (the part stands a little before where it was believed): the skill starts from where the hand is.
            dist = float(np.linalg.norm(self.rb.eef_pose()[1] - p_touch))
            arrived = self._lrl_stops() > stops and dist <= CFG["standoff"] + CFG["arrive"]
            self.log("lrl_arrival", stage=result["stage"], family=fam, arrived=arrived, short_m=round(dist, 4),
                     stop=(self.last_move or {}).get("status"))
            if not arrived:
                failed.append(c)
                result["attempts"].append({"status": "transit_failed", "pressing": not grip, "pose": fam,
                                           "why": (self.last_move or {}).get("why")})
                self._lrl_unstick(result)
                return None
        elif grip:
            def early(robot, F, F0, k):
                return "early_contact" if np.linalg.norm(F - F0) > CFG["grip_contact"] and k > 2 else None
            self.move_line(np.asarray(end.R, float), p_touch, grip=0.0, speed=CFG["grip_speed"],
                           guard=self._guard("early_contact", early), finger=f_open)
        else:
            def touch(robot, F, F0, k):
                return "contact" if np.linalg.norm(F - F0) > CFG["contact_force"] and k > 2 else None
            # closed fingertips are symmetric: the twin presses the same
            self.move_line(np.asarray(end.R, float), p_touch + a * CFG["tip_over"] * (mode == "tip"), grip=1.0,
                           speed=CFG["speed"][mode], guard=self._guard("contact", touch), finger=0.0)

        p0 = np.asarray(mf["p0"], float)
        before = P.copy()
        fb = float((front_pt - p0) @ n) if front_pt is not None else float(np.percentile((before - p0) @ n, 90))
        fl = None if fixture_pt is None else float((fixture_pt - p0) @ n)
        band = (fb - 0.40, fb + 0.03) if motion == "push" else (fb - 0.03, fb + 0.40)
        # what lies in the part rides with it: its mapped place is stale as soon as the part moves
        moving = allowed + tuple(m.oid for m in self._contents_of(part))
        R_in, p_start = self.rb.eef_pose()
        p_start = p_start.copy()
        entry = Entry(R_in, p_start, a, (0.0, CFG["standoff"]), twin=True)
        # a grip that pushes is the skill's close-then-lead: with the closing direction it is a grip and a push
        gmode = "pull" if grip else "push"
        far, runs, front, held, read = p_start, [], fb, False, None
        done_m = 0.0                                          # the part's travel along s * n so far (as read)
        for k in range(CFG["runs"]):
            if want == "closed":
                left = 0.0 if remaining is None else max(0.0, remaining - done_m)
            else:
                left = max(0.0, CFG["open_travel"] - done_m) if motion == "pull" else 0.0
            contract = Contract("move_part", "pads" if grip else "tips", allowed, [entry],
                                Goal(gmode, s * n, travel=left), {**LIMITS, "time": CFG["time"]})
            tr, rec, hit = self._lrl_run(contract, moving, own, result["stage"], tare=None if k == 0 else w_free)
            p_run = tr.arrays()["p"]
            adv = (p_run - p_start) @ (s * n)
            if len(adv) and adv.max() > (far - p_start) @ (s * n):
                far = p_run[int(adv.argmax())]
            gap = float(self.rb.finger_opening())
            held = held or (grip and gap >= CFG["grip_empty"])
            went = None
            if self.lrl_detector is not None:
                read = self._lrl_detect(tr, contract)
                done_m += float(read["travel"] or 0.0)
            else:
                now = self._lrl_look(part, mf, before, band, fl)
                went = None if now.get("front_m") is None else s * (float(now["front_m"]) - front)
                met = None if want is None else self._lrl_met(want, now)
                read = {"label": interim_label(motion, tr.stop, went, met, rec["peak_force_n"],
                                               gap if motion == "pull" else None),
                        "probs": None, "travel": went, "rule": "interim", "part": self._state_words(now)}
                front = front if went is None else float(now["front_m"])
                done_m = s * (front - fb)
            self.log("lrl_skill", stage=result["stage"], family=fam, run=k, **rec, guard=hit or None,
                     **self._lrl_words(read), travel_wanted_m=round(left, 4),
                     **({"pad_gap_m": round(gap, 4)} if grip else {}))
            runs.append({"stop": tr.stop, "hand_m": rec["hand_travel_m"], "label": read["label"],
                         "part_m": None if read["travel"] is None else round(float(read["travel"]), 3)})
            # Only a contact that moved the part and has not got there yet is run again from where the hand is: a
            # guard stop would only be allowed closer to what stopped it, and the same law from the same place
            # does the same when the hand did not move.
            if read["label"] != "short" or want is None or tr.stop == "guard" \
                    or rec["hand_travel_m"] < CFG["again_hand"] or (self.lrl_detector is None and went is None):
                break
        # how far the hand pushed or pulled, not where the skill's back-off after a force stop left it
        disp = far - p_start
        travel, stop = float(np.linalg.norm(disp)), stroke_stop(tr)
        # pads that closed on nothing pressed the part with closed fingers: the verdict reads a push as a press
        how = "grip" if held or motion == "pull" else "press"
        self.log("leg", tag="lrl_" + motion, cause=stop, travel=round(travel, 4), runs=len(runs), mode=how,
                 label=read["label"])
        result["attempts"].append({"status": ("press_push" if not grip else "push") if motion == "push" else "pull",
                                   "mode": how, "pose": fam, "travel_m": round(travel, 3), "stop": stop,
                                   "label": read["label"], "runs": runs})
        Rs, ps = self.rb.eef_pose()
        if grip:
            self.ex.gripper_to(width)                                       # release only as wide as approached
            self.retreat(Rs, ps - Rs[:, 2] * CFG["grip_back"], grip=0.0, speed=CFG["back_speed"], finger=f_open,
                         tag="release_back")
        else:
            back = ps - a * CFG["back"] + n * CFG["side_lift"] * (mode == "side")
            self.retreat(Rs, back, grip=1.0, speed=CFG["back_speed"], finger=0.0, tag="press_back")
            self.ex.gripper(-1.0, max_steps=12)
        stroke = {"mode": how, "travel": travel, "stop": stop, "d": s * n, "disp": disp, "point": c, "family": fam,
                  "normal": n, "label": read["label"]}
        if read["label"] in CFG["marking"]:
            # no_effect: the part was touched and did not move along this face's axis, so every candidate on the
            # face is a near-duplicate (the caller drops the face); otherwise this pose and its near neighbours
            self.lrl_mark(result, stroke, read["label"], scope="face" if read["label"] == "no_effect" else "candidate")
        return stroke

    @staticmethod
    def _lrl_met(want, state):
        """The in-place reading shows the goal state (as Harness._linear_verdict reads 'closed' and 'opened')."""
        if state.get("closed") is None:
            return None
        if want == "closed":
            return bool(state["closed"])
        return (not state["closed"]) and ((state.get("offset_m") or 0.0) >= 0.02 or (state.get("angle_deg") or 0.0) >= 15.0)

    def _lrl_unstick(self, result):
        """A transit refused at a joint limit: the arm stands next to its limits (a look from above can leave it so,
        smoke run s0, the stove), and no path leaves that posture; back out toward the start posture
        (Harness.escape), once per stage."""
        why = (self.last_move or {}).get("why") or {}
        mem = self._lrl_stage(result)
        if why.get("reason") == "joint_limit" and not mem["escaped"]:
            mem["escaped"] = True
            self.log("lrl_escape", stage=result["stage"], why=why, moved=self.escape())

    def _lrl_stops(self):
        """How many certified paths have ended on contact so far (TraceMotion counts every stop by kind)."""
        return sum(self.tm.counts["stops"][k] for k in CFG["arrive_stops"])

    def _lrl_hold(self, part, parent, mf, failed, motion, want, remaining, fixture_pt, why, allowed, own, marked):
        """The hold the exempt grip search finds on the part for this stage (with v2's end check of a closing push),
        as a candidate. None when it finds none."""
        end_ok = self._closing_end_check(part, mf, remaining, fixture_pt) if want == "closed" else None
        plan, w = self._lrl_grip(part, parent, mf, failed, motion, allowed, own, marked, end_check=end_ok)
        if plan is None:
            why["grip"] = dict(w)
            return None
        g, q_pre, q_g = plan
        return {"family": "grip", "mode": "grip", "a": g.R[:, 2], "R": g.R, "p": g.p, "c": g.p, "q_pre": q_pre,
                "q": q_g, "finger": g.info["finger"], "g": g}

    def _lrl_grip(self, part, parent, mf, failed, motion, allowed, own, marked, end_check=None):
        """v2's grip search (Harness.plan_part_grasp: antipodal grasps on the part in the mount frame, a rigid-hand
        prefilter at the widest pre-shape, arm postures, the approach line, the turn's reach, the closing end check),
        bounded in time, without its vetoes on what the contract allows (plan 5.1): the hand may meet the part and
        its fixture; the table, other objects and the arm's clearance stay as v2 has them. Returns ((g, q_pre, q_g) or
        None, {reason: count})."""
        t_end = time.time() + CFG["search_s"]
        R0, _ = self.rb.eef_pose()
        n = np.asarray(mf["n"], float)
        cands = articulate.part_grasps(part, mf, R0, max_depth=CFG["grip_depth"] if motion in ("push", "pull") else None)
        cands = [g for g in cands if all(np.linalg.norm(g.p - f) > 0.006 for f in failed)
                 and not is_marked(marked, "grip", g.p, n)]
        excl = (part.oid,) + ((parent.oid,) if parent is not None else ())
        q_now = self.rb.q()[:7]
        why, hands, fams, blame = Counter(), {}, {}, {}
        for g in cands:
            if time.time() > t_end:
                why["search_time_limit"] = 1
                break
            bad = self._lrl_hand_bad(g, allowed, own, hands, blame)
            if bad:
                why[bad] += 1
                continue
            fams.setdefault((round(g.info["tilt"], 2), round(g.info["pitch"], 2)), []).append(g)
        order = []
        while any(fams.values()) and len(order) < CFG["grip_tries"]:          # families round-robin, best first
            for k in list(fams):
                if fams[k]:
                    order.append(fams[k].pop(0))
        self.log("part_grasp_prefilter", kept=len(order), rejected=dict(why), lrl=True, candidates=len(cands),
                 blame=blame, pitches=dict(Counter(round(float(np.rad2deg(g.info.get("pitch", 0.0)))) for g in order)))
        fallback = None
        for g in order:
            if time.time() > t_end:
                why["search_time_limit"] = 1
                break
            f = g.info["finger"]
            sols = self.ik_clear(g.R, g.p, q_now, exclude=excl, finger=f)
            if len(sols) == 0:
                why["no_ik"] += 1
            for q_g in sols[:3]:
                bad = self._lrl_meets(q_g[None], excl, allowed, own, finger=f)
                if bad is not None:
                    why["config_" + bad[0]] += 1
                    continue
                q_pre = self._ik_near(g.R, g.p - g.R[:, 2] * CFG["grip_back"], q_g)
                line = self.line_configs(g.R, g.p, q_pre, finger=f)
                if line is None or self._lrl_meets(np.array(line[::2]), excl, allowed, own, finger=f) is not None:
                    why["approach"] += 1
                    continue
                if motion == "rotate" and not self._turn_reachable(q_g, g, mf, excl):
                    why["turn"] += 1
                    continue
                m_end = None if end_check is None else end_check(g.R, g.p, line[-1], 0.5 * g.width)
                if m_end is not None and m_end < 0:
                    why["end_blocked"] += 1
                    if fallback is None or m_end > fallback[0]:
                        fallback = (m_end, (g, line[0], line[-1]))
                    continue
                self.log("part_grasp", p=g.p, yaw=g.info["yaw"], tilt=g.info.get("tilt"), pitch=g.info.get("pitch"),
                         width=g.width, finger=f, align=g.info.get("align"), lrl=True)
                return (g, line[0], line[-1]), why
        if fallback is not None:
            # every hold would meet the fixture before the part closes: the nearest-to-clear one (as v2)
            g = fallback[1][0]
            self.log("part_grasp", p=g.p, yaw=g.info["yaw"], width=g.width, finger=g.info["finger"], lrl=True,
                     end_check_relaxed=True, end_margin=round(fallback[0], 4))
            return fallback[1], why
        self.log("part_grasp_fail", why=dict(why), lrl=True)
        return None, why

    def _lrl_hand_bad(self, g, allowed, own, hands, blame=None):
        """v2's rigid-hand prefilter of a grasp (palm and fingers at the grasp and 4 cm back, against observed
        geometry and the table), at the widest pre-shape that passes, with the part and its fixture taken out:
        None (and the pre-shape set in g.info) or 'hand_scene' / 'hand_unknown' (only unobserved space is in the
        way) / 'hand_table'. blame: {reason: what is nearest}, filled for the first rejection of each reason."""
        from .harness import NEAR
        bad = None
        with self._lrl_free(own):
            for gap in manip.APPROACH_GAPS:            # widest pre-shape the surroundings allow
                f = round(manip.preshape_finger(g.width, gap), 3)
                if f not in hands:
                    hands[f] = self._hand_local(f)
                L, grp, rad = hands[f]
                bad = None
                for back in (0.0, 0.04):
                    Wh = g.p - g.R[:, 2] * back + L @ g.R.T
                    c = self.checker.point_clearance(Wh, tuple(allowed)) - rad
                    if c[grp == 1].min() < NEAR["palm"] or c[grp == 2].min() < NEAR["finger"]:
                        k = self.checker.point_clearance(Wh, tuple(allowed), unknown=False) - rad
                        seen = k[grp == 1].min() < NEAR["palm"] or k[grp == 2].min() < NEAR["finger"]
                        bad = "hand_scene" if seen else "hand_unknown"
                        if blame is not None and bad not in blame:
                            blame[bad] = {**self.nearest_obstacle(Wh[c < NEAR["palm"]] if seen else Wh, allowed),
                                          "pitch_deg": round(float(np.rad2deg(g.info.get("pitch", 0.0)))),
                                          "back_m": back}
                        break
                    zc = Wh[:, 2] - self.table_z - 0.5 * rad
                    if zc[grp == 1].min() < NEAR["plane_palm"] or zc[grp == 2].min() < NEAR["plane_finger"]:
                        bad = "hand_table"
                        break
                if bad is None:
                    g.info["finger"], g.info["gap"] = f, gap
                    return None
        return bad

    @contextmanager
    def _lrl_free(self, own):
        """The fixture's unlabelled surface (own) is not an obstacle inside the block."""
        ck = self.checker
        prev, ck.free_near = ck.free_near, list(ck.free_near or []) + own
        try:
            yield
        finally:
            ck.free_near = prev

    def _lrl_entry(self, part, parent, n, c0, failed, marked, allowed, own):
        """The first press candidate, in CFG order, with an arm posture at the standoff and a straight line to the
        touch on which hand and arm meet nothing but what the contract allows. Returns (candidate + (q_pre,) or None,
        {family: {reason: count}} for the candidates passed over, blame per reason)."""
        excl = (part.oid,) + ((parent.oid,) if parent is not None else ())
        q_now = self.rb.q()[:7]
        why, blame = {}, {}

        def note(fam, reason, who=None):
            why.setdefault(fam, {})[reason] = why.get(fam, {}).get(reason, 0) + 1
            if who is not None:
                blame.setdefault(f"{fam}_{reason[5:]}", who)
        cands = candidates(c0, n, failed, marked)
        fams = [f for f in CFG["order"] if any(k[0] == f for k in cands)]
        t_end = time.time() + CFG["search_s"]
        for i, fam in enumerate(fams):
            # a family that has no pose here (the table under a level hand, say) must not use up the search:
            # each gets an equal share of the time that is left
            t_fam = time.time() + (t_end - time.time()) / (len(fams) - i)
            for _, mode, a, R, p, c in (k for k in cands if k[0] == fam):
                if time.time() > t_fam:
                    note(fam, "search_time_limit")
                    break
                sols = self.ik_clear(R, p - a * CFG["standoff"], q_now, exclude=excl, finger=0.0, n_random=8)[:3]
                if len(sols) == 0:
                    note(fam, "no_clear_ik")
                    continue
                at = [self._lrl_meets(q[None], excl, allowed, own) for q in sols]      # the standoff before a line
                for q_pre, bad in zip(sols, at):
                    line = None if bad else self.line_configs(R, p, q_pre, finger=0.0)
                    if bad is None and line is None:
                        note(fam, "no_line")
                        continue
                    if bad is None:
                        bad = self._lrl_meets(np.array(list(line[2:-1:2]) + ([line[-1]] if mode == "side" else [])),
                                              excl, allowed, own)
                    if bad is None:
                        return (fam, mode, a, R, p, c, q_pre), why, blame
                    note(fam, "line_" + bad[0], bad[1])
        return None, why, blame

    def _lrl_meets(self, Q, excl, allowed, own, finger=0.0):
        """Whether hand or arm would meet something that is not theirs to touch at the configurations Q (harness
        NEAR margins): None, or (group, what is nearest and the clearances). The hand may touch what the contract
        allows and the unlabelled surface of that fixture (`own`); the arm keeps v2's check."""
        from .harness import NEAR
        ck = self.checker
        if len(Q) == 0:
            return None
        arm = ck.clearance(Q, exclude=excl, finger=finger)
        with self._lrl_free(own):
            hand = ck.clearance(Q, exclude=allowed, finger=finger)
        cl = {g: (arm if g.endswith("arm") else hand)[g]
              for g in ("arm", "plane_arm", "palm", "plane_palm", "finger", "plane_finger") if g in arm}
        short = np.array([v < NEAR[g] for g, v in cl.items()])            # (groups, configurations)
        if not short.any():
            return None
        k = int(short.any(0).argmax())
        g = list(cl)[int(short[:, k].argmax())]
        W, _ = ck.world_points(Q[k][None], finger)
        pts = W[0][ck.pg == ("arm", "palm", "finger").index(g.replace("plane_", ""))]
        return g, {**self.nearest_obstacle(pts, excl if g.endswith("arm") else allowed),
                   "clearance": {k_: round(float(v[k]), 4) for k_, v in cl.items()}}

    # ------------------------------------------------------------------------------------------------- turn
    def lrl_turn(self, st, part, parent, direction, result):
        """A rotate stage through the skill. Candidates: the axis hypotheses (_lrl_axes) in order, each with the
        exempt grip search's hold on the part, each turning sense (v2's turn_order or System 1's turn_direction, the
        other sense after it when the arm has room) a candidate; the senses of one hold run in place, the later ones
        tared against the wrist before the grip. A sense that ended in a marking label, or whose turn showed no
        expected effect, is marked; what it turned is turned back (as v2) before the next. Done: the label says
        reached and v2's checks agree: System 1's effect seen and confirmed (when the stage names one; it must still
        be there after the hand lets go), else the part followed the hand (v2's MIN_FOLLOW)."""
        mf0 = self._mount_frame(part, parent)
        q_seen = self.rb.q()[:7].copy()
        if self._look_at_face(part, parent, mf0) is not None:
            mf0 = self._mount_frame(part, parent)          # refit with the view along the mounting normal
        q = self.rb.q()[:7]
        if np.minimum(q - self.cm.lo, self.cm.hi - q).min() < CFG["limit_slack"]:
            # a look from above can leave the arm at its joint limits, where TRACE refuses every path to a hold and
            # escape() finds none (dev stove: smoke s0, s1, final/dev/turn_on_the_stove/init1): back to where the look
            # started, a posture TRACE came from
            self.log("lrl_after_look", q=np.round(q, 3),
                     moved=self.go_configs([q_seen], None, finger=self._finger(), tag="lrl_after_look"))
        axes = self._lrl_axes(part, parent, mf0)
        self.log("lrl_axes", stage=result["stage"], axes=[{"n": np.round(f["n"], 3), "source": f["source"],
                                                            "axis_point": np.round(f["axis_point"], 3)} for f in axes])
        self.checker.free_near = [(part.points, 0.015, True)] + \
            ([(parent.points, 0.010, False)] if parent is not None else [])
        mem = self._lrl_stage(result)
        fix_oids, fix_pts = self._fixture(part, part)
        allowed = (part.oid,) + fix_oids
        own = [(fix_pts, CFG["fixture_skin"], True)] if len(fix_pts) else []
        effect = bool(st.get("effect"))
        ref0, measure = None, None
        if effect:
            ref0 = self.capture(HEAD_CAMERA)              # before operating: S1's IMAGE 1 and the undo reference
            self.save_img("effect_ref", ref0.rgb)
            region = self._effect_region(ref0, part, parent, st)
            has_px = st.get("effect_x") is not None and st.get("effect_y") is not None

            def measure(cur):                             # as Harness._turn_until_effect measures
                n_ch, n_vis = self._effect_change(ref0, cur, region)
                seen = n_vis >= (max(self.EFFECT_MIN_VISIBLE, 0.5 * int(region.sum())) if has_px
                                 else self.EFFECT_MIN_VISIBLE)
                need = max(40, self.EFFECT_FRAC * n_vis) if has_px else 250
                return {"changed_px": n_ch, "visible_px": n_vis, "seen": bool(seen), "hit": bool(seen and n_ch >= need)}
        failed = []
        for ax, _ in ((a, k) for a in axes for k in range(CFG["candidates"])):
            # each axis keeps its turn until a hold on it has turned both ways without moving the part
            if mem["entered"] >= CFG["candidates"]:
                break
            if any(ax is d_ for d_ in mem.setdefault("dead_axes", [])):
                continue
            n, ap = np.asarray(ax["n"], float), np.asarray(ax["axis_point"], float)
            plan, why = self._lrl_grip(part, parent, ax, failed, "rotate", allowed, own, mem["marked"])
            if plan is None:
                self.log("lrl_entry", stage=result["stage"], motion="rotate", candidate=mem["entered"], family=None,
                         axis=n, axis_source=ax["source"], passed={"grip": dict(why)})
                result["attempts"].append({"status": "no_feasible_grip", "axis": np.round(n, 3), "rejected": dict(why)})
                mem["dead_axes"].append(ax)
                continue
            g, q_pre, q_g = plan
            senses = list(articulate.turn_order(self, g, q_g, ap, n, direction))
            if len(senses) == 1 and self._sense_room(g, q_g, ap, n, -senses[0]) > 0:
                senses.append(-senses[0])
            mem["entered"] += 1
            self.log("lrl_entry", stage=result["stage"], motion="rotate", candidate=mem["entered"], family="grip",
                     axis=n, axis_source=ax["source"], senses=senses, point=g.p, approach=g.R[:, 2],
                     allowed=[self.objects[o].name for o in allowed])
            f_open = g.info["finger"]
            width = 2 * (f_open + manip.PAD_OFFSET)
            self.ex.gripper_to(width)
            end = self.reach(g.R, g.p, q_g, g.R[:, 2], part.oid, (width - g.width) / 2, "to_part", 0.0, f_open, q_pre)
            if end is None:
                failed.append(g.p)
                result["attempts"].append({"status": "transit_failed", "why": (self.last_move or {}).get("why")})
                self._lrl_unstick(result)
                continue
            g, q_g = self._taken(g, end, q_g)
            w_free = tuple(np.array(x, float) for x in self.rb.wrench())    # before the pads load the hand

            def early(robot, F, F0, k):
                return "early_contact" if np.linalg.norm(F - F0) > CFG["grip_contact"] and k > 2 else None
            self.move_line(g.R, g.p, grip=0.0, speed=CFG["grip_speed"], guard=self._guard("early_contact", early),
                           finger=f_open)
            done = self._lrl_turn_senses(st, result, allowed, own, n, ap, g, senses, w_free, ref0, measure)
            failed.append(g.p)
            self.ex.gripper_to(width)                                       # release only as wide as approached
            Rs, ps = self.rb.eef_pose()
            self.retreat(Rs, ps - Rs[:, 2] * CFG["grip_back"], grip=0.0, speed=CFG["back_speed"], finger=f_open,
                         tag="release_back")
            head = self.capture(HEAD_CAMERA)
            self.frames.append(head)
            before = part.points.copy()
            if self.refresh_object(part, [head], [part.category, part.name]):
                result["measured_part_shift_m"] = round(float(np.linalg.norm(part.points.mean(0) - before.mean(0))), 4)
            self.rebuild_map(self.frames[-8:])
            if done is None:
                if any(kw.get("label") == "no_effect" for kw in result["attempts"][-len(senses):]):
                    mem["dead_axes"].append(ax)          # the part did not follow about this axis: the next axis
                continue
            moved, eff = done
            result["measured"] = moved
            if eff is None:
                result["relation_check"] = {"satisfied": True, **moved}
            else:
                # v2: the change must still be there once the hand has let go
                after = measure(head)
                eff["after_release"] = after
                result["effect"] = eff
                gone = after["seen"] and not after["hit"]
                if gone:
                    self.log("lrl_disagree", stage=result["stage"], label="reached", v2="effect gone after release",
                             after_release=after)
                result["relation_check"] = {"satisfied": not gone, **({"unverified": True} if gone else {}),
                                            "effect_observed": True if not gone else None, "confirmed_by_s1": True,
                                            **moved}
            self.log("actuated", result=result)
            return result
        from .harness import StageFailure        # harness imports this module: the name is only needed here
        if effect:
            raise StageFailure("no_visible_effect", expected_effect=st.get("effect"), effect_pixel=[
                st.get("effect_x"), st.get("effect_y")], attempts=result["attempts"][-4:],
                note="no axis and turning sense produced the expected change in the watched area of the head camera")
        raise StageFailure("actuation_failed", attempts=result["attempts"][-4:])

    def _lrl_turn_senses(self, st, result, allowed, own, n, ap, g, senses, w_free, ref0, measure):
        """The senses of one hold on the part, run in place. Returns (moved, effect report or None) when one is
        done, else None (each sense tried is marked)."""
        mem = self._lrl_stage(result)
        net = 0.0
        for i, s in enumerate(senses):
            if i > 0:
                if mem["entered"] >= CFG["candidates"]:
                    break
                mem["entered"] += 1
            d = s * n
            ang = 0.0                                    # turned in this sense, over its runs (rad, at the hand)
            for k in range(CFG["runs"]):
                left = max(CFG["turn_travel"] - ang, 0.0)
                contract = Contract("move_part", "pads", allowed, [Entry(*self.rb.eef_pose(), g.R[:, 2], (0.0, 0.0))],
                                    Goal("turn", d, axis_point=ap, travel=left), {**LIMITS, "time": CFG["time"]})
                pol = self._lrl_policy_for("turn")
                # the base law has no travel rule: it turns the travel wanted (after the pads close) and stops there
                steps = None if pol is not None else \
                    SKILL["close_steps"] + int(np.ceil(left / np.radians(SKILL["turn_rate"]) / SKILL["dt"]))
                tr, rec, hit = self._lrl_run(contract, allowed, own, result["stage"],
                                             tare=None if i == 0 and k == 0 else w_free, max_steps=steps, policy=pol)
                a_k = turned_about(tr, d)
                ang += a_k
                net += s * a_k
                gap = float(self.rb.finger_opening())
                read = self._lrl_detect(tr, contract) if self.lrl_detector is not None else \
                    {"label": interim_label("turn", tr.stop, gap=gap, turned=ang), "probs": None, "travel": ang,
                     "rule": "interim"}
                self.log("lrl_skill", stage=result["stage"], family="grip", run=k, sense=s, **rec, guard=hit or None,
                         **self._lrl_words(read), turned_deg=round(float(np.rad2deg(a_k)), 1), pad_gap_m=round(gap, 4))
                # a turn that moved the part and has not got there yet goes on from where the hand is (as lrl_move's
                # runs); marking it and turning it back would undo the progress (final/dev/turn_on_the_stove/init1)
                if read["label"] != "short" or tr.stop == "guard" or abs(a_k) < CFG["again_turn"]:
                    break
            att = {"status": "turn", "sign": s, "achieved_deg": round(float(np.rad2deg(ang)), 1), "label": read["label"],
                   "stop": tr.stop}
            result["attempts"].append(att)
            moved = {"rotation_deg": round(float(np.rad2deg(net)), 1),
                     "sense": "counterclockwise" if net > 0 else "clockwise"}
            stroke = {"family": "turn", "point": ap, "normal": d, "label": read["label"]}
            if read["label"] == "reached" and measure is None:
                if abs(ang) >= CFG["turn_follow"]:                  # v2's verdict: the part followed the hand
                    return moved, None
                self.log("lrl_disagree", stage=result["stage"], label="reached", v2="the hand did not turn",
                         turned_deg=att["achieved_deg"])
                self.lrl_mark(result, stroke, "reached, but the hand did not turn")
            elif read["label"] == "reached":
                self._hold(4)
                cur = self.capture(HEAD_CAMERA)
                m = measure(cur)
                ok, why = (None, "no change in the watched area") if not m["hit"] else \
                    (None, "no model budget left for an effect check") if not self._effect_budget_ok() else \
                    self._confirm_effect(st, ref0, cur, net)
                att["effect"] = {**m, "confirmed_by_s1": ok}
                if ok:
                    self.save_img("effect_seen", cur.rgb)
                    return moved, {"expected": st.get("effect"), "observed": True, "s1_evidence": why, **m}
                self.log("lrl_disagree", stage=result["stage"], label="reached", v2=why, **m)
                self.lrl_mark(result, stroke, "reached, but " + why)
            elif read["label"] in CFG["marking"]:
                self.lrl_mark(result, stroke, read["label"], scope="axis" if read["label"] == "no_effect" else "sense")
            else:
                self.lrl_mark(result, stroke, read["label"])
            if read["label"] == "lost_grip" or gap < CFG["grip_empty"]:
                break                                            # nothing held: the next hold, not the next sense
            if abs(net) > CFG["turn_back"]:
                back = articulate.twist_about(self, ap, n, -net, settle=6)   # leave the part as it was found (v2)
                net += back
                self.log("turn_back", turned_deg=round(float(np.rad2deg(back)), 1), net_deg=round(float(np.rad2deg(net)), 1))
        return None

    def _lrl_axes(self, part, parent, mf0):
        """Axis hypotheses of a turn, v2's mount frame first: the normals of the dominant planes fitted around the
        part (the faces Harness._face_hypotheses fits, without its plausibility vetoes), one per axis (n and -n are
        one axis: both senses are candidates)."""
        out = [dict(mf0, source="mount_frame")]
        cos = np.cos(np.deg2rad(CFG["dup_angle"]))
        for f in self._face_hypotheses(part, parent):
            if all(abs(float(np.dot(f["n"], g["n"]))) <= cos for g in out):
                out.append(f)
        return out

    # ------------------------------------------------------------------------------------- the skill run
    def _lrl_policy_for(self, mode):
        """The trained policy when it is on and was trained on this mode; otherwise the base law (None)."""
        p = self.lrl_policy
        return p if p is not None and mode in p.modes else None

    def _lrl_detect(self, trace, contract):
        """The detector's reading of one contact."""
        if not trace.rows:
            return {"label": "no_contact", "probs": None, "travel": 0.0, "rule": "no step was made"}
        a = trace.arrays()
        a["touch"] = trace.touch
        detect = self.lrl_detector                      # a model (an attribute), not a method of the harness
        return {**detect(a, contract), "rule": "detector"}

    @staticmethod
    def _lrl_words(read):
        """The reading of a contact for the event log."""
        return {"label": read["label"], "rule": read.get("rule"),
                "probs": None if read.get("probs") is None else {k: round(float(v), 3) for k, v in read["probs"].items()},
                "travel": None if read.get("travel") is None else round(float(read["travel"]), 4),
                **({"travel_sd": round(float(read["travel_sd"]), 4)} if read.get("travel_sd") is not None else {}),
                **({"part": read["part"]} if "part" in read else {})}

    def _lrl_guard(self, moving):
        """The per-step guard of a certified move (guard.StepGuard: joint limits; hand, arm and held object against
        observed geometry at the ladder's guard distances), built for a contact that starts here: what `moving`
        names is not geometry, and the rest must not come closer than it is at the entry."""
        ck = self.checker
        excl = tuple(moving) + ((self.held,) if self.held is not None else ())
        snap = SimpleNamespace(exact=lambda x, _, unknown: ck.point_clearance(x, excl, unknown=unknown),
                               table_z=self.scene.table_z)
        bodies = hand_spheres(self.rm, 0.0)
        if self.held is not None:
            bodies = bodies + payload_spheres(ck.attached)
        R, p = self.rb.eef_pose()
        q = self.rb.q()[:7]
        here = SimpleNamespace(R=[R, R], p=[p, p], q=[q, q])
        return StepGuard(self, snap, SimpleNamespace(bodies=bodies), here, LADDER, np.inf).check

    def _lrl_run(self, contract, moving, own, stage, tare=None, max_steps=None, policy=False):
        """One run of the skill from where the hand stands, under the per-step guard. tare: the wrist reading to tare
        against (a run on a loaded hand); policy: the actor, or False for the one _lrl_policy_for gives. Returns
        (trace, its numbers, the guard's finding). The trace is saved in a finally: LIBERO's done ends the episode by
        raising through the policy, and a run cut that way is the one that reached the goal."""
        ck = self.checker
        pol = self._lrl_policy_for(contract.goal.mode) if policy is False else policy
        sk = Skill(pol)
        hit, swept = {}, [self.rb.q()[:7]]
        prev, ck.free_near = ck.free_near, list(ck.free_near or []) + own
        check = self._lrl_guard(moving)

        def guard(io):
            if len(sk.trace.rows) % 4 == 0:
                swept.append(io.q()[:7])
            out = check(io.q())
            hit.update(out or {})
            return out is not None
        try:
            sk.run(self.rb, contract, guard, max_steps=max_steps, tare=tare)
        finally:
            ck.free_near = prev
            rec = self._lrl_save(sk.trace, stage, contract.goal.direction)
            rec.update(law="policy" if pol is not None else "base", tared=tare is not None)
            if sk.trace.stop is None:
                self.log("lrl_skill", stage=stage, **rec, cut="the episode ended inside the run")
        self.carve_robot(np.array(swept + [self.rb.q()[:7]]))      # the arm has been there (as articulate.translate)
        return sk.trace, rec, hit

    def _lrl_save(self, trace, stage, d):
        """Write one skill trace to <run_dir>/contacts/<stage>_<n>.npz; its numbers for the log."""
        out = self.run_dir / "contacts"
        out.mkdir(exist_ok=True)
        name = f"{stage}_{len(list(out.glob(f'{stage}_*.npz')))}.npz"
        a = trace.arrays()
        np.savez_compressed(out / name, **a, R_c=trace.R_c, p_c=trace.p_c, touch=trace.touch, stop=str(trace.stop),
                            direction=d)
        adv = (a["p"] - trace.p_c) @ d
        return {"file": f"contacts/{name}", "stop": trace.stop, "steps": len(trace.rows),
                "hand_travel_m": round(float(adv.max()), 4) if len(adv) else 0.0,
                "hand_end_m": round(float(adv[-1]), 4) if len(adv) else 0.0,
                "peak_force_n": round(float(np.linalg.norm(a["f"], axis=1).max()), 1) if len(adv) else 0.0}

    def _lrl_look(self, part, mf, before, band, fixture_level):
        """The part's state read from where the hand stands (head and wrist camera, no move): the state measurement
        over the pre-stroke footprint, with the fixture level of an earlier view when the surroundings are hidden,
        as _observe_after_stroke reads it after a stroke. Nothing enters the map: the hand is on the part. The
        interim reading between runs while no detector is loaded."""
        views = [self.capture(HEAD_CAMERA), self.capture(WRIST_CAMERA)]
        state = self._measure_state(part, mf, views, part_pts=before, front_band=band, allow_turned=False)
        if state.get("front_m") is not None and not state.get("measured") and fixture_level is not None \
                and (state.get("angle_deg") or 0.0) <= articulate.CLOSED_TILT:
            off = float(state["front_m"]) - fixture_level
            state.update(measured=True, face_m=round(fixture_level, 4), offset_m=round(max(off, 0.0), 4),
                         closed=bool(off <= articulate.FACE_TOL), why="fixture face from an earlier view")
        return state

