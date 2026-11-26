"""Reference curves of LRL_PLAN.md M0 step 3 and the numbers of gate G0: how today's push stroke and the lrl base
law do in the Tier A stage simulator when the believed contact frame carries a hand-off error.

Task: push an open part closed. A trial is one (layout, target part, opening, hand-off error, pose family); every
method starts it from the same simulator state:
  stroke      today's first attempt: the guarded line of harness._press_stroke from the standoff, then
              articulate.translate itself with the arguments _press_stroke passes. No re-seat, no second stroke.
  base        lrl_skill.Skill with policy None from the same standoff pose.
  base_touch  the same guarded line, then the base law: separates the motion law from the way in.
Hand poses are the press search's own (harness.py:3992-4020), built from the believed face frame; of its vetoes only
kinematics is kept: a posture at the standoff that touches nothing, and a Cartesian line to the touch pose.

Labels are simulator truth. Nothing here is an input of a skill.

  python -m lrl_train.eval_stroke --layouts 40 --procs 32      # trials.jsonl, summary.json
  python -m lrl_train.eval_stroke --tables                     # the tables of REFERENCE.md from summary.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

import numpy as np

from atrace import manip
from atrace.articulate import translate
from atrace.body import ControllerModel
from atrace.geometry import frame_from_axes, rotvec_to_mat
from atrace.guard import ContactDetector
from atrace.harness import Harness
from atrace.lrl_contract import Contract, Entry, Goal
from atrace.lrl_skill import CFG as SKILL, Skill
from atrace.motion import Executor

from . import scenes
from .sim import Shim, StageSim

OUT = Path("/workspace/agentic_trace/runs/lrl_m0_20261125/reference")

CFG = {
    # hand-off error, the prior range of M0 (LRL_PLAN.md M0 step 4); a level is a share of it
    "err_pos": 0.03,             # T  m: believed face point, lateral to the face normal
    "err_angle": 20.0,           # T  deg: believed normal against the true one
    "err_travel": 0.03,          # T  m: believed remaining travel
    "levels": {"none": 0.0, "half": 0.5, "full": 1.0},     # T
    # start of a trial
    "open_slide": (0.08, 0.16),  # T  m: opening of a drawer or sliding panel (its full range if shorter)
    "open_hinge": (60.0, 110.0),  # T  deg: opening of a door (its full range if shorter)
    "edge_depth": 0.006,         # T  m: a panel sliding along its face is pushed on its edge, this far behind the face
    # the press search's pose construction and approach, copied from harness._press_stroke (H = harness.py)
    "tip_gap": 0.002,            # T  m: fingertips this far off the face at the touch pose (H:4009)
    "rolls": (0.0, np.pi / 2, np.pi / 4, -np.pi / 4),      # T  rad about the approach (H:4006)
    "steep": (65.0, 80.0),       # T  deg: fingertips tilted down onto the face (H:3999)
    "lean": (30.0, 50.0),        # T  deg (H:4001)
    "side_dz": (0.01, -0.01, -0.025),   # T  m: fingertip below the contact point, finger sides pushing (H:4000)
    "side_out": 0.018,           # T  m: fingertip in front of the face in a side pose (H:4013)
    "side_ok": 0.7,              # T  the tilted families exist when the face is this far from level (H:3993)
    "standoff": 0.06,            # T  m: pre-pose back along the approach (H:4027)
    "tip_over": 0.01,            # T  m: a tip approach aims this far past the touch pose (H:4081)
    "speed": {"tip": 0.04, "side": 0.05},   # T  m/s of the guarded line (H:4081, 4084)
    "contact_force": 8.0,        # T  N: wrist force change that ends the guarded line (H:4079)
    # the base law
    "skill_time": 8.0,           # T  s: longest run of the base law
    # labels (evaluator side)
    "closed_tol": 0.005,         # T  m at the face: closed, where LIBERO has no predicate for the fixture
    "still": (2, 0.001),         # T  steps, m: the hand stands still when it moved less than this over the window
    "block_steps": 3,            # T  still steps in a contact that make it a standing block
    "end_window": 5,             # T  steps: "at the stop" (covers the stroke's 3 hold steps, the skill's back-off)
}

KINDS = ("drawer_stack", "wooden_cabinet", "white_cabinet", "microwave", "door", "sliding_panel",
         "short_cabinet", "short_fridge", "slide_cabinet", "window")
FAMILIES = ("tip", "tip_steep", "side", "tip_lean")       # the order the press search tries them in
METHODS = ("stroke", "base", "base_touch")
ROBOT_PARTS = ("finger", "hand", "link")                  # contact classes on the robot side ...
TOUCHED = ("target", "fixture", "scene")                  # ... and on the other: the part, the rest of its fixture
# why a trial failed. block_body: hand body or an arm link on the rest of the fixture at the stop; block_other:
# fingers on the rest of the fixture, or any robot body on the table or a loose object, at the stop; then the
# motion's own stop reason (translate: stalled, force_limit, max_travel; skill: time, force)
REASONS = ("no_pose", "block_body", "block_other", "missed", "slipped_off", "stalled", "force_limit", "max_travel",
           "time", "force")


class Hand(Shim):
    """The part of a Harness that its guarded line needs: the harness's own line_configs, move_line and contact
    guard run on this object, so the approach here is the code _press_stroke runs."""
    motion, held = "trace", None
    line_configs, move_line, _guard = Harness.line_configs, Harness.move_line, Harness._guard

    def __init__(self, robot):
        super().__init__(robot)
        self.rm = robot.model
        self.ex = Executor(robot, self.rm)
        self.detector = ContactDetector(ControllerModel(self.rm))
        manip.set_gripper_geometry(self.rm.gripper_geometry(robot.q()[:7]))     # TIP_Z, as Harness.__init__


def _unit(x):
    return x / np.linalg.norm(x)


def contact_face(part, q):
    """The face a push meets, from simulator truth: point, outward normal, travel left to closed (m at the point).
    A part that slides along its own front is pushed on its trailing edge."""
    n = part.face_normal
    if part.kind == "hinge":
        r = part.face_point - part.axis_point
        return part.face_point, n, abs(q - part.closed) * np.linalg.norm(r - (r @ part.axis) * part.axis)
    dc = part.axis * np.sign(part.closed - q)
    if abs(dc @ n) > 0.5:
        return part.face_point, n, abs(q - part.closed)
    v = np.cross(n, part.face_u)
    half = abs(dc @ part.face_u) * part.half_extents[0] + abs(dc @ v) * part.half_extents[1]
    return part.face_point - dc * half - n * CFG["edge_depth"], -dc, abs(q - part.closed)


def sample_error(rng, n, scale):
    """Hand-off error at a share of the prior range: lateral offset, tilt (rotation vector), travel error. Magnitudes
    are uniform up to the bound, directions uniform in the face plane."""
    t, w = (_unit(x - (x @ n) * n) for x in rng.normal(size=(2, 3)))
    return (scale * CFG["err_pos"] * rng.uniform() * t, scale * np.radians(CFG["err_angle"]) * rng.uniform() * w,
            scale * CFG["err_travel"] * rng.uniform(-1, 1))


def press_poses(c, n):
    """Hand poses of harness._press_stroke for one contact point c on a face with outward normal n, by family:
    lists of (mode, approach, R, touch position of the grip site), in the search's order."""
    up = np.array([0.0, 0.0, 1.0])
    u = _unit(np.cross(n, up) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0]))
    v = np.cross(n, u)
    down = -up + n * n[2]
    tilted = np.linalg.norm(down) > CFG["side_ok"]
    down = down / max(np.linalg.norm(down), 1e-9)
    reach = manip.TIP_Z + CFG["tip_gap"]

    def tips(degrees):
        out = []
        for a in (np.cos(t) * -n + np.sin(t) * down for t in np.radians(degrees)):
            out += [("tip", a, frame_from_axes(a, np.cos(r) * u + np.sin(r) * v), c - a * reach) for r in CFG["rolls"]]
        return out

    fam = {"tip": tips((0.0,))}
    if tilted:
        fam["tip_steep"], fam["tip_lean"] = tips(CFG["steep"]), tips(CFG["lean"])
        fam["side"] = [("side", down, frame_from_axes(down, x), c + n * CFG["side_out"] + down * (dz - manip.TIP_Z))
                       for dz in CFG["side_dz"] for x in (u, n)]
    return fam


def find_entry(sim, hand, base, cands):
    """Index of the first pose whose standoff has an arm posture that touches nothing and a Cartesian line to the
    touch pose (the arm is left at that standoff), or None; and why the poses before it were passed over."""
    why = Counter()
    for k, (mode, a, R, p) in enumerate(cands):
        sim.restore(base)                           # a posture that does not fit shoves the scene
        if not sim.entry(R, p - a * CFG["standoff"]):
            why["no_posture"] += 1                  # no IK solution inside the joint limits, or the hand was pushed off
        elif any(_robot(b) >= 0 for b, _, _ in sim.robot_contacts()):
            why["touching"] += 1
        elif hand.line_configs(R, _aim(mode, a, p), sim.robot.q()[:7], finger=0.0) is None:
            why["no_line"] += 1
        else:
            return k, dict(why)
    return None, dict(why)


def _aim(mode, a, p):
    return p + a * CFG["tip_over"] * (mode == "tip")


def _contact(robot, F, F0, k):
    return "contact" if np.linalg.norm(F - F0) > CFG["contact_force"] and k > 2 else None


def _robot(body):
    return (0 if "finger" in body else 1 if body.startswith("gripper0_") or body == "robot0_right_hand"
            else 2 if body.startswith("robot0_link") else -1)


class Recorder:
    """Simulator truth after every control step: hand position, wrist force against the reading at the standoff,
    the target's joint value, and which robot bodies touch what."""

    def __init__(self, sim, part):
        self.sim, self.joint, self.F0 = sim, part.joint, sim.robot.wrench()[0]
        own, rest = os.path.commonprefix([part.body, *part.fixture_bodies]), set(part.fixture_bodies)
        self.touched = lambda b: 1 if b in rest else 0 if b.startswith(own) else 2
        self.p, self.f, self.q, self.pairs = [], [], [], []

    def __call__(self, rb, obs):
        self.p.append(rb.eef_pose()[1])
        self.f.append(np.linalg.norm(rb.wrench()[0] - self.F0))
        self.q.append(self.sim.joint(self.joint))
        self.pairs.append({(a, b) for a, b, _ in self.sim.robot_contacts() if _robot(a) >= 0})


def label(rec, k0, part, closed, lever):
    """Labels of one motion; the stroke starts at step k0 (what comes before is the approach)."""
    p, f, q = np.array(rec.p), np.array(rec.f), np.array(rec.q)
    c = np.zeros((len(p), 3, 3), bool)              # step, robot side, other side
    for k, pairs in enumerate(rec.pairs):
        for a, b in pairs:
            c[k, _robot(a), rec.touched(b)] = True
    w, tol = CFG["still"]
    still = np.zeros(len(p), bool)
    still[k0 + w:] = np.linalg.norm(p[k0 + w:] - p[k0:len(p) - w], axis=1) < tol
    ok = closed(q)
    end = slice(len(p) - CFG["end_window"], None)
    # the block of the recorded t03 runs: hand body or an arm link on the fixture next to the part, hand at rest
    standing = still & ~ok & c[:, 1:, 1].any(1)
    left = float(abs(q[-1] - part.closed) * lever)
    return {
        "success": bool(ok.any()), "closed_at_end": bool(ok[-1]), "left": left, "near": left <= CFG["closed_tol"],
        "hand_travel": float(np.linalg.norm(p[-1] - p[k0])), "steps": int(len(p) - k0),
        "force_peak": float(f[k0:].max()), "force_final": float(f[-3:].mean()),
        "body_at_stop": bool(c[end, 1:, 1].any()), "other_at_stop": bool((c[end, 0, 1] | c[end, :, 2].any(1)).any()),
        "stop_bodies": sorted({f"{a}>{b}" for pairs in rec.pairs[end] for a, b in pairs
                               if _robot(a) > 0 and rec.touched(b) == 1}),
        "block_body": int(standing.sum()), "block_force": float(f[standing].mean()) if standing.any() else 0.0,
        "touched": bool(c[:, :, 0].any()), "touching_at_end": bool(c[end, :, 0].any()),
        "contact_steps": {f"{r}_{o}": int(c[k0:, i, j].sum()) for i, r in enumerate(ROBOT_PARTS)
                          for j, o in enumerate(TOUCHED) if c[k0:, i, j].any()},
    }


def closed_test(sim, part, lever):
    """LIBERO's own predicate where the fixture class has one, else within closed_tol of the closed end."""
    if sim.layout.fixtures:
        from libero.libero.envs.objects.articulated_objects import ArticulatedObject
        obj = sim.env.env.get_object(next(iter(sim.layout.fixtures)))
        if type(obj).is_close is not ArticulatedObject.is_close:
            return lambda q: np.array([obj.is_close(x) for x in q], bool)
    return lambda q: np.abs(q - part.closed) * lever <= CFG["closed_tol"]


def run_method(method, sim, hand, part, cand, n_b, left_b, closed, lever):
    mode, a, R, p = cand
    rb = sim.robot
    rec = rb.on_step = Recorder(sim, part)
    out = {}
    if method != "base":
        out["approach"] = hand.move_line(R, _aim(mode, a, p), grip=1.0, speed=CFG["speed"][mode],
                                         guard=hand._guard("contact", _contact), finger=0.0).status
    k0 = len(rec.p)
    if method == "stroke":
        travel, out["stop"], _ = translate(hand, n_b, "push", grip_check=False)
        out["stroke_travel"] = float(travel)
    else:
        R0, p0 = rb.eef_pose()
        contract = Contract("move_part", "tips", (part.name,), [Entry(R0, p0, a, (0.0, CFG["standoff"]))],
                            Goal("push", -n_b, travel=left_b))
        out["stop"] = Skill().run(rb, contract, max_steps=round(CFG["skill_time"] / SKILL["dt"]),
                                  way_in=CFG["standoff"] if method == "base" else 0.0).stop
    rb.on_step = None
    return out | label(rec, k0, part, closed, lever)


def run_layout(job, levels=tuple(CFG["levels"]), families=FAMILIES, methods=METHODS, rounds=3):
    """All trials of one random layout: every part of its fixture as the target (at least `rounds` targets), one
    group of trials per error level, the pose families of a group sharing one opening and one error sample."""
    kind, index, out = job
    rng = np.random.default_rng([KINDS.index(kind), index])
    sim = StageSim(scenes.sample_layout(rng, kind, Path(out) / "bddl"))
    hand = Hand(sim.robot)
    home = sim.snapshot()
    rank = np.argsort(np.argsort([p.face_point[2] for p in sim.parts]))
    n_parts = len(sim.parts)
    rows = []
    for rnd in range(-(-rounds // n_parts)):
        for part, z in zip(sim.parts, rank):
            height = "-" if n_parts == 1 else "bottom" if z == 0 else "top" if z == n_parts - 1 else "middle"
            for level in levels:
                sim.restore(home)
                for other in sim.parts:
                    sim.set_joint(other.joint, other.closed)
                q0 = part.closed + np.sign(part.open - part.closed) * min(
                    rng.uniform(*(CFG["open_slide"] if part.kind == "slide" else np.radians(CFG["open_hinge"]))),
                    abs(part.open - part.closed))
                sim.set_joint(part.joint, q0)
                base = sim.snapshot()
                c, n, left = contact_face(part, q0)
                lever = left / abs(q0 - part.closed)
                closed = closed_test(sim, part, lever)
                dc, dn, dt = sample_error(rng, n, CFG["levels"][level])
                n_b = rotvec_to_mat(dn) @ n
                poses = press_poses(c + dc, n_b)
                group = {"kind": kind, "layout": index, "part": part.name, "height": height, "round": rnd,
                         "level": level, "opening": float(q0), "left0": float(left), "protrusion": part.protrusion,
                         "err_pos": float(np.linalg.norm(dc)), "err_angle": float(np.degrees(np.linalg.norm(dn))),
                         "err_travel": float(dt)}
                for family in families:
                    if family not in poses:         # the search builds the tilted families on upright faces only
                        continue
                    k, passed = find_entry(sim, hand, base, poses[family])
                    if k is None:
                        rows += [group | {"family": family, "method": m, "pose": None, "passed": passed}
                                 for m in methods]
                        continue
                    start = sim.snapshot()
                    for m in methods:
                        sim.restore(start)
                        rows.append(group | {"family": family, "method": m, "pose": k, "passed": passed}
                                    | run_method(m, sim, hand, part, poses[family][k], n_b, left + dt, closed, lever))
    sim.close()
    return rows


def reason(row):
    """Why a failed trial failed: one class per trial, the first that applies."""
    if row["pose"] is None:
        return "no_pose"
    if row["body_at_stop"]:
        return "block_body"
    if row["other_at_stop"]:
        return "block_other"
    if not row["touched"]:
        return "missed"
    if not row["touching_at_end"]:
        return "slipped_off"
    return row["stop"]


def _rate(rows):
    ran = [r for r in rows if r["pose"] is not None]
    return {"n": len(rows), "ran": len(ran), "success": sum(r["success"] for r in ran),
            "near": sum(r["near"] for r in ran)}


def _by(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault("|".join(str(r[k]) for k in keys), []).append(r)
    return out


def summarize(rows):
    """Counts behind REFERENCE.md. `first_pose` replays the search's order without its vetoes: per group, the first
    family that has a pose is the first attempt."""
    order = {f: i for i, f in enumerate(FAMILIES)}
    first = []
    for g in _by(rows, "kind", "layout", "part", "round", "level", "method").values():
        ran = sorted((r for r in g if r["pose"] is not None), key=lambda r: order[r["family"]])
        first.append(ran[0] if ran else g[0])
    wc = [r for r in rows if r["kind"] == "white_cabinet" and r["pose"] is not None]
    return {
        "config": {k: v for k, v in CFG.items()},
        "trials": len(rows),
        "success": {k: _rate(g) for k, g in _by(rows, "kind", "family", "level", "method").items()},
        "success_by_height": {k: _rate(g) for k, g in _by(rows, "kind", "height", "family", "level", "method").items()},
        "first_pose": {k: _rate(g) for k, g in _by(first, "kind", "level", "method").items()},
        "failures": {k: dict(Counter(reason(r) for r in g if r["pose"] is None or not r["success"]), n=len(g))
                     for k, g in _by(rows, "kind", "method").items()},
        "white_cabinet_block": {k: _block(g) for k, g in _by(wc, "method", "family", "height").items()},
    }


def _block(rows):
    """The G0 fidelity numbers of one cell: failed trials that ended with the hand body or an arm link on the rest
    of the fixture, and those among them where the hand stood still in that contact."""
    hit = [r for r in rows if not r["success"] and r["body_at_stop"]]
    stand = [r for r in hit if r["block_body"] >= CFG["block_steps"]]
    return {"ran": len(rows), "failed": sum(not r["success"] for r in rows), "block": len(hit),
            "standing": len(stand), "force": float(np.median([r["block_force"] for r in stand] or [0.0])),
            "stops": dict(Counter(r["stop"] for r in hit)),
            "bodies": dict(Counter(b for r in hit for b in r["stop_bodies"]))}


def tables(s):
    """The tables of REFERENCE.md as markdown."""
    def cell(d):
        if d is None or not d["ran"]:
            return "–" if d is None else f'no pose ({d["n"]})'
        return f'{d["success"]}/{d["ran"]} ({100 * d["success"] / d["ran"]:.0f}%)'

    levels = list(CFG["levels"])
    out = ["| fixture | pose family | " + " | ".join(f"{m}, {lv}" for lv in levels for m in METHODS) + " |",
           "|---|---|" + "---|" * (len(levels) * len(METHODS))]
    for kind in KINDS:
        for fam in FAMILIES:
            cells = [s["success"].get(f"{kind}|{fam}|{lv}|{m}") for lv in levels for m in METHODS]
            if any(cells):
                out.append(f"| {kind} | {fam} | " + " | ".join(cell(d) for d in cells) + " |")
        cells = [s["first_pose"].get(f"{kind}|{lv}|{m}") for lv in levels for m in METHODS]
        if any(cells):
            out.append(f"| {kind} | first with a pose | " + " | ".join(cell(d) for d in cells) + " |")
    out += ["", "| fixture | method | trials | failed | " + " | ".join(REASONS) + " |",
            "|---|---|---|---|" + "---|" * len(REASONS)]
    for kind in KINDS:
        for m in METHODS:
            d = s["failures"].get(f"{kind}|{m}")
            if d:
                out.append(f'| {kind} | {m} | {d["n"]} | {sum(v for k, v in d.items() if k != "n")} | '
                           + " | ".join(str(d.get(r, 0)) for r in REASONS) + " |")
    out += ["", "| white cabinet: method | pose family | drawer | run | failed | hand body or arm link on the rest of "
            "the cabinet at the stop | of those, hand standing still | median force standing, N | stop reasons | "
            "robot body > cabinet body |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, d in sorted(s["white_cabinet_block"].items()):
        out.append("| " + " | ".join(k.split("|"))
                   + f' | {d["ran"]} | {d["failed"]} | {d["block"]} | {d["standing"]} | {d["force"]:.0f} | '
                   + ", ".join(f"{b} {n}" for b, n in sorted(d["stops"].items())) + " | "
                   + ", ".join(f"{b} {n}" for b, n in sorted(d["bodies"].items())).replace("white_cabinet_1_", "")
                   .replace("gripper0_", "").replace("robot0_", "") + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=40, help="random layouts per fixture kind")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tables", action="store_true", help="print the tables from summary.json and stop")
    a = ap.parse_args()
    if a.tables:
        print(tables(json.loads((a.out / "summary.json").read_text())))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(kind, i, str(a.out)) for i in range(a.layouts) for kind in a.kinds]
    rows = []
    # a fresh process per few layouts: robosuite keeps memory of every environment it built
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=4) as pool, open(a.out / "trials.jsonl", "w") as f:
        for k, part in enumerate(pool.imap_unordered(run_layout, jobs)):
            rows += part
            f.writelines(json.dumps(r) + "\n" for r in part)
            f.flush()
            print(f"layout {k + 1}/{len(jobs)}  trials {len(rows)}", flush=True)
    (a.out / "summary.json").write_text(json.dumps(summarize(rows), indent=1, default=lambda o: np.asarray(o).tolist()))


if __name__ == "__main__":
    main()
