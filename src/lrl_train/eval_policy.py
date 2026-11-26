"""Gate G1, simulator side (LRL_PLAN.md M1): the protocol of eval_stroke with the trained move_part policy as a
further method, for push, pull and turn. Same trials (layout, target part, start, hand-off error, pose or grip
family), each method from the same simulator state, nominal physics and clean sensors as in eval_stroke:

  push   stroke      today's first attempt (eval_stroke)
         base_touch  the harness's guarded line, then the lrl base law (eval_stroke)
         policy      from the standoff: the skill's own way in, then the base law with the policy's residual
  pull   stroke      today's grip stroke: close the pads, articulate.translate(..., "pull") along the believed direction
  turn   stroke      close the pads, articulate.twist_about the believed axis by the believed angle
         base        the skill with the base law from the grip pose (pads open around the believed handle)
         policy      the same with the policy's residual
The policy runs as the harness runs it: lrl_models.load_policy, the deterministic action. A further method, free
(M1E step 1), runs the checkpoint of the control run E0 (train_move_part --free) the way policy runs M1's, with the
skill's base law off; --methods picks the methods (the trials stay the same: no method draws from the layout's
random numbers).

Layouts are validation layouts (seeds from VAL_SEED on) of the training kinds and layouts of the kinds the policy
never saw. Labels are simulator truth.

  python -m lrl_train.eval_policy --weights weights/lrl/move_part_push_<date>_<sha8>.pt --procs 96
  python -m lrl_train.eval_policy --tables --out <dir>         # tables.md again from trials.jsonl
  python -m lrl_train.eval_policy --methods free --free-weights <E0 checkpoint> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from atrace.articulate import translate, twist_about
from atrace.geometry import rotvec_to_mat
from atrace.lrl_contract import Contract, Entry, Goal
from atrace.lrl_models import load_policy
from atrace.lrl_skill import CFG as SKILL, Skill

from . import env as V
from . import eval_stroke as E
from . import scenes
from .env import TRAIN_KINDS, UNSEEN_KINDS, VAL_SEED
from .sim import StageSim

OUT = Path("/workspace/agentic_trace/runs/lrl_m1_20261125/train2/eval")
METHODS = ("stroke", "base_touch", "policy")
GRIP_METHODS = ("stroke", "base", "policy")
LEARNED = ("policy", "free")        # free: an E0 checkpoint with the skill's base law off (M1E step 1)
TASK_KINDS = ("wooden_cabinet", "white_cabinet", "microwave", "flat_stove")    # the LIBERO fixtures of the test tasks
GROUPS = {"task fixtures, validation layouts": TASK_KINDS,
          "generated mechanisms, validation layouts": tuple(k for k in TRAIN_KINDS if k not in TASK_KINDS),
          "unseen fixtures": UNSEEN_KINDS}

CFG = {
    "layouts": 40,                              # T  per kind
    # T  kinds whose trials often have no reachable pose (REFERENCE.md: sliding panel four in five; the generated
    #    door two in three on the first validation layouts), or have one part only, so that each kind and mode
    #    has its 300 trials
    "layouts_more": {"sliding_panel": 80, "door": 80, "microwave": 60, "knob": 60, "flat_stove": 60,
                     "short_fridge": 60, "slide_cabinet": 60, "window": 60, "faucet": 60, "basin_faucet": 60},
    "grip_rounds": 4,                           # T  pull or turn trials per level: parts x rounds >= this
    "gate_level": "full",                       # T  G1 is read at the full prior error
}

_model = {}


def skill_for(method, policies):
    """The skill of a method: the base law alone, the base law with M1's policy, or E0's policy with the base off."""
    return Skill(policies.get(method), cfg=SKILL | {"base": method != "free"})


def run_method(method, policies, sim, hand, part, cand, n_b, left_b, closed, lever):
    """eval_stroke.run_method plus the learned methods. For those the motion starts at the touch, where the skill's
    way in ends, so their labels cover the same span as those of the other two methods."""
    mode, a, R, p = cand
    rb = sim.robot
    rec = rb.on_step = E.Recorder(sim, part)
    out = {}
    if method not in LEARNED:
        out["approach"] = hand.move_line(R, E._aim(mode, a, p), grip=1.0, speed=E.CFG["speed"][mode],
                                         guard=hand._guard("contact", E._contact), finger=0.0).status
    k0 = len(rec.p)
    if method == "stroke":
        travel, out["stop"], _ = translate(hand, n_b, "push", grip_check=False)
    else:
        R0, p0 = rb.eef_pose()
        contract = Contract("move_part", "tips", (part.name,), [Entry(R0, p0, a, (0.0, E.CFG["standoff"]))],
                            Goal("push", -n_b, travel=left_b))
        trace = skill_for(method, policies).run(
            rb, contract, max_steps=round(E.CFG["skill_time"] / SKILL["dt"]),
            way_in=E.CFG["standoff"] if method in LEARNED else 0.0)
        out["stop"] = trace.stop
        k0 = trace.touch if method in LEARNED else k0
    rb.on_step = None
    lab = E.label(rec, k0, part, closed, lever)
    steps = lab["contact_steps"]
    whole = {rec.touched(b) for pairs in rec.pairs for _, b in pairs}
    return out | lab | {
        "body_fixture": any(steps.get(f"{r}_fixture", 0) for r in ("hand", "link")),    # during the motion
        "outside": any(steps.get(f"{r}_scene", 0) for r in E.ROBOT_PARTS),              # during the motion
        "outside_trial": 2 in whole}                                                    # the way in included


def run_layout(job):
    """eval_stroke.run_layout with the methods of this module; the random draws are the same."""
    kind, index, out, weights, free, methods = job
    torch.set_num_threads(1)
    policies = {}
    for m, w in (("policy", weights), ("free", free)):
        if m in methods:
            if w not in _model:
                _model[w] = load_policy(w)
            policies[m] = _model[w]
    push = [m for m in METHODS + ("free",) if m in methods]
    rng = V.layout_rng(kind, index)
    sim = StageSim(scenes.sample_layout(rng, kind, Path(out) / "bddl"))
    hand = E.Hand(sim.robot)
    home = sim.snapshot()
    rank = np.argsort(np.argsort([p.face_point[2] for p in sim.parts]))
    n_parts = len(sim.parts)
    rows = []
    for rnd in range(-(-3 // n_parts)):
        for part, z in zip(sim.parts, rank):
            if "push" not in V.part_modes(part):
                continue
            height = "-" if n_parts == 1 else "bottom" if z == 0 else "top" if z == n_parts - 1 else "middle"
            for level in E.CFG["levels"]:
                sim.restore(home)
                for other in sim.parts:
                    sim.set_joint(other.joint, other.closed)
                q0 = part.closed + np.sign(part.open - part.closed) * min(
                    rng.uniform(*(E.CFG["open_slide"] if part.kind == "slide" else np.radians(E.CFG["open_hinge"]))),
                    abs(part.open - part.closed))
                sim.set_joint(part.joint, q0)
                base = sim.snapshot()
                c, n, left = E.contact_face(part, q0)
                lever = left / abs(q0 - part.closed)
                closed = E.closed_test(sim, part, lever)
                dc, dn, dt = E.sample_error(rng, n, E.CFG["levels"][level])
                n_b = rotvec_to_mat(dn) @ n
                poses = E.press_poses(c + dc, n_b)
                group = {"kind": kind, "layout": index, "part": part.name, "height": height, "round": rnd,
                         "level": level, "opening": float(q0), "left0": float(left), "mode": "push"}
                for family in E.FAMILIES:
                    if family not in poses:
                        continue
                    k, _ = E.find_entry(sim, hand, base, poses[family])
                    if k is None:
                        rows += [group | {"family": family, "method": m, "pose": None} for m in push]
                        continue
                    start = sim.snapshot()
                    for m in push:
                        sim.restore(start)
                        rows.append(group | {"family": family, "method": m, "pose": k}
                                    | run_method(m, policies, sim, hand, part, poses[family][k], n_b, left + dt,
                                                 closed, lever))
    rows += grip_trials(sim, hand, home, policies, rng, kind, index, rank,
                        [m for m in GRIP_METHODS + ("free",) if m in methods])
    sim.close()
    return rows


def grip_trials(sim, hand, home, policies, rng, kind, index, rank, methods=GRIP_METHODS):
    """Pull and turn trials of a layout: every part that has the mode as the target, each error level, the first grip
    pose of the generator's order (roll) that the arm can be set to, the grip methods from that state."""
    rows = []
    n_parts = len(sim.parts)
    cfg = V.CFG | {"err_widen": 1.0}            # levels are shares of the prior range itself
    for rnd in range(-(-CFG["grip_rounds"] // n_parts)):
        for part, z in zip(sim.parts, rank):
            mode = "turn" if "turn" in V.part_modes(part) else "pull" if "pull" in V.part_modes(part) else None
            if mode is None:
                continue
            height = "-" if n_parts == 1 else "bottom" if z == 0 else "top" if z == n_parts - 1 else "middle"
            for level, scale in E.CFG["levels"].items():
                sim.restore(home)
                for other in sim.parts:
                    sim.set_joint(other.joint, other.closed)
                q0 = part.closed
                base = sim.snapshot()
                goal_q, reached = V.goal_value(part, mode, rng, sim)
                hp = V.handle(part)[0]
                lever = 1.0 if mode == "turn" else V.lever_at(part, hp)
                want = abs(goal_q - q0) * lever
                cands, d_b, ap_b, err = V.believe_grip(part, mode, rng, scale, cfg)
                dt = rng.uniform(-1, 1) * scale * (cfg["err_turn"] if mode == "turn" else E.CFG["err_travel"])
                group = {"kind": kind, "layout": index, "part": part.name, "height": height, "round": rnd,
                         "level": level, "mode": mode, "left0": float(want), "opening": float(q0)} | err
                k = V.enter_grip(sim, base, cands, V.finger_for(V.handle(part)[4]))
                if k is None:
                    rows += [group | {"family": "grip", "method": m, "pose": None} for m in methods]
                    continue
                start = sim.snapshot()
                goal = Goal(mode, d_b, axis_point=ap_b if mode == "turn" else None, travel=max(0.0, want + dt))
                for m in methods:
                    sim.restore(start)
                    rows.append(group | {"family": f"roll{k}", "method": m, "pose": k}
                                | run_grip(m, policies, sim, hand, part, goal, reached, q0, goal_q, lever))
    return rows


def run_grip(method, policies, sim, hand, part, goal, reached, q0, goal_q, lever):
    """One pull or turn from the grip pose. Labels: reached at any step and at the end, travel toward the goal,
    contacts outside the part and its fixture, the pads' hold."""
    rb = sim.robot
    rec = rb.on_step = E.Recorder(sim, part)
    if method == "stroke":
        R, p = rb.eef_pose()
        for _ in range(SKILL["close_steps"]):           # the pads close on the handle, as the skill's base law does
            rb.step(R, p, grip=1.0)
        if goal.mode == "pull":
            stop = translate(hand, goal.direction, "pull")[1]
        else:
            twist_about(hand, goal.axis_point, goal.direction, goal.travel)
            stop = "twist"
    else:
        R0, p0 = rb.eef_pose()
        contract = Contract("move_part", "pads", (part.name,), [Entry(R0, p0, R0[:, 2], (0.0, 0.0))], goal)
        stop = skill_for(method, policies).run(
            rb, contract, max_steps=round(V.CFG["time"][goal.mode] / SKILL["dt"])).stop
    rb.on_step = None
    q = np.array(rec.q)
    ok = np.array([reached(x) for x in q])
    sgn = np.sign(goal_q - q0)
    c = np.zeros((len(q), 3, 3), bool)
    for k, pairs in enumerate(rec.pairs):
        for a, b in pairs:
            c[k, E._robot(a), rec.touched(b)] = True
    end = slice(len(q) - E.CFG["end_window"], None)
    return {"stop": stop, "success": bool(ok.any()), "closed_at_end": bool(ok[-1]),
            "travel": float((q[-1] - q0) * sgn * lever), "steps": len(q),
            "outside": bool(c[:, :, 2].any()), "outside_trial": bool(c[:, :, 2].any()),
            "body_fixture": bool(c[:, 1:, 1].any()), "touched": bool(c[:, :, 0].any()),
            "touching_at_end": bool(c[end, :, 0].any())}


# ------------------------------------------------------------------------------------------------- tables
def _cell(rows, key="success"):
    ran = [r for r in rows if r["pose"] is not None]
    return f"{sum(r[key] for r in ran)}/{len(ran)} ({100 * sum(r[key] for r in ran) / len(ran):.0f}%)" if ran else "–"


def _pct(rows, key="success"):
    ran = [r for r in rows if r["pose"] is not None]
    return 100 * sum(r[key] for r in ran) / max(1, len(ran))


def _first(rows, families=E.FAMILIES):
    """Per trial group and method, the first family of an order (the press search's by default) that has a pose."""
    order = {f: i for i, f in enumerate(families)}
    out = []
    for g in E._by(rows, "kind", "layout", "part", "round", "level", "method").values():
        ran = sorted((r for r in g if r["pose"] is not None), key=lambda r: order[r["family"]])
        out.append(ran[0] if ran else g[0])
    return out


def tables(rows):
    """REPORT.md tables per mode: push as before (tables_push), then pull and turn (tables_grip)."""
    push = [r for r in rows if r.get("mode", "push") == "push"]
    out = ["## Push", "", tables_push(push)]
    for mode in ("pull", "turn"):
        rs = [r for r in rows if r.get("mode") == mode]
        if rs:
            out += ["", f"## {mode.capitalize()}", "", tables_grip(rs, mode)]
    return "\n".join(out)


def tables_grip(rows, mode):
    """Gate cells of a pull or turn: success (the goal reached at any step) of stroke, base law and policy by kind
    and error level, the policy against the stroke, contacts outside, how the contacts end."""
    lv = CFG["gate_level"]
    by = lambda rs, **kw: [r for r in rs if all(r[k] == v for k, v in kw.items())]      # noqa: E731
    kinds = [k for k in TRAIN_KINDS + UNSEEN_KINDS if by(rows, kind=k)]
    out = [f"Gate G1, simulator side, {mode}: first-attempt success at the {lv} prior error (the first grip pose "
           "the arm can be set to; trials without one are not counted).", "",
           "| set | fixture | stroke | base | policy | policy − stroke | policy − base |", "|---|---|---|---|---|---|---|"]
    for name, group in GROUPS.items():
        for kind in [k for k in group if k in kinds] + ["all"]:
            sel = {m: [r for r in by(rows, level=lv, method=m)
                       if (r["kind"] in group if kind == "all" else r["kind"] == kind)] for m in GRIP_METHODS}
            if not any(r["pose"] is not None for r in sel["policy"]):
                continue
            out.append(f"| {name} | {kind} | " + " | ".join(_cell(sel[m]) for m in GRIP_METHODS)
                       + f' | {_pct(sel["policy"]) - _pct(sel["stroke"]):+.0f} | '
                       f'{_pct(sel["policy"]) - _pct(sel["base"]):+.0f} |')
    out += ["", f"{mode.capitalize()}: success by fixture and error level (pose: trials with a grip pose / all).", "",
            "| fixture | " + " | ".join(f"{m}, {l}" for l in E.CFG["levels"] for m in GRIP_METHODS) + " | with a pose |",
            "|---|" + "---|" * (len(E.CFG["levels"]) * len(GRIP_METHODS) + 1)]
    for kind in kinds:
        pol = by(rows, kind=kind, method="policy")
        out.append(f"| {kind} | " + " | ".join(_cell(by(rows, kind=kind, level=l, method=m))
                                               for l in E.CFG["levels"] for m in GRIP_METHODS)
                   + f" | {sum(r['pose'] is not None for r in pol)}/{len(pol)} |")
    out += ["", f"{mode.capitalize()}: contacts and endings, all levels, trials with a pose. `outside`: any robot body "
            "on something that is neither the part nor its fixture; `at the end`: goal reached at the end.", "",
            "| fixture | method | trials | success | at the end | outside | body on fixture | lost touch at the end "
            "| stop reasons |", "|---|---|---|---|---|---|---|---|---|"]
    for kind in kinds + ["all"]:
        for m in GRIP_METHODS:
            rs = [r for r in by(rows, method=m) if r["pose"] is not None and kind in (r["kind"], "all")]
            if not rs:
                continue
            lost = [dict(r, lost=not r["touching_at_end"]) for r in rs]
            stops = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["stop"] for r in rs).items()))
            out.append(f"| {kind} | {m} | {len(rs)} | {_cell(rs)} | {_cell(rs, 'closed_at_end')} | "
                       f"{_cell(rs, 'outside')} | {_cell(rs, 'body_fixture')} | {_cell(lost, 'lost')} | {stops} |")
    return "\n".join(out)


def tables_push(rows):
    """REPORT.md tables: the gate, the gate with the measured entry order, success by kind, family and error level,
    contacts, how the policy's contacts end."""
    lv = CFG["gate_level"]
    by = lambda rs, **kw: [r for r in rs if all(r[k] == v for k, v in kw.items())]      # noqa: E731
    first = _first(rows)
    kinds = [k for k in TRAIN_KINDS + UNSEEN_KINDS if by(rows, kind=k)]
    out = [f"Gate G1, simulator side: first-attempt success at the {lv} prior error. `all poses`: every trial that "
           "had a reachable pose, the four pose families pooled. `first family`: per trial group the first family of "
           "the press search's order that has a pose.", "",
           "| set | fixture | stroke, all poses | base_touch, all poses | policy, all poses | policy − stroke | "
           "stroke, first family | policy, first family | policy − stroke |", "|---|---|---|---|---|---|---|---|---|"]
    for name, group in GROUPS.items():
        for kind in [k for k in group if k in kinds] + ["all"]:
            sel = (lambda rs, m: [r for r in by(rs, level=lv, method=m)
                                  if (r["kind"] in group if kind == "all" else r["kind"] == kind)])
            a, f = {m: sel(rows, m) for m in METHODS}, {m: sel(first, m) for m in METHODS}
            out.append(f"| {name} | {kind} | " + " | ".join(_cell(a[m]) for m in METHODS)
                       + f' | {_pct(a["policy"]) - _pct(a["stroke"]):+.0f} | {_cell(f["stroke"])} | '
                       f'{_cell(f["policy"])} | {_pct(f["policy"]) - _pct(f["stroke"]):+.0f} |')
    # M1 step 5: the entry order is read from validation layouts of the training kinds, never from the unseen set
    val = [r for r in by(rows, method="policy") if r["kind"] in TRAIN_KINDS and r["pose"] is not None]
    order = sorted(E.FAMILIES, key=lambda f: -_pct(by(val, family=f)))
    best = _first(rows, order)
    out += ["", "Entry order measured: the pose families by the policy's success on validation layouts of the six "
            "training kinds, all error levels: " + ", ".join(f"{f} {_cell(by(val, family=f))}" for f in order)
            + f". The gate at the {lv} prior error with the first family of that order that has a pose:", "",
            "| set | fixture | " + " | ".join(METHODS) + " | policy − stroke |", "|---|---|---|---|---|---|"]
    for name, group in GROUPS.items():
        for kind in [k for k in group if k in kinds] + ["all"]:
            f = {m: [r for r in by(best, level=lv, method=m)
                     if (r["kind"] in group if kind == "all" else r["kind"] == kind)] for m in METHODS}
            out.append(f"| {name} | {kind} | " + " | ".join(_cell(f[m]) for m in METHODS)
                       + f' | {_pct(f["policy"]) - _pct(f["stroke"]):+.0f} |')
    out += ["", "First-attempt success by fixture, pose family and error level.", "",
            "| fixture | pose family | " + " | ".join(f"{m}, {l}" for l in E.CFG["levels"] for m in METHODS) + " |",
            "|---|---|" + "---|" * (len(E.CFG["levels"]) * len(METHODS))]
    for kind in kinds:
        for fam in E.FAMILIES + ("first family",):
            src = by(first, kind=kind) if fam == "first family" else by(rows, kind=kind, family=fam)
            if any(r["pose"] is not None for r in src):
                out.append(f"| {kind} | {fam} | " + " | ".join(
                    _cell(by(src, level=l, method=m)) for l in E.CFG["levels"] for m in METHODS) + " |")
    out += ["", "Contacts, all error levels and pose families, trials that had a pose. `body on fixture`: the hand "
            "body or an arm link touched the rest of the fixture during the motion. `outside`: any robot body touched "
            "something that is neither the part nor its fixture (table, loose objects), during the motion and over "
            "the whole trial with the way in.", "",
            "| fixture | method | trials run | success | body on fixture | outside, motion | outside, whole trial | "
            "stop reasons |", "|---|---|---|---|---|---|---|---|"]
    for kind in kinds + ["all"]:
        for m in METHODS:
            rs = [r for r in by(rows, method=m) if r["pose"] is not None and kind in (r["kind"], "all")]
            stops = ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["stop"] for r in rs).items()))
            out.append(f"| {kind} | {m} | {len(rs)} | {_cell(rs)} | {_cell(rs, 'body_fixture')} | "
                       f"{_cell(rs, 'outside')} | {_cell(rs, 'outside_trial')} | {stops} |")
    out += ["", "How the policy's contacts end. `done, closed`: the done flag was raised with the part closed.", "",
            "| fixture | trials run | done, closed | done, open | force stop, closed | force stop, open | "
            "time, closed | time, open |", "|---|---|---|---|---|---|---|---|"]
    for kind in kinds:
        rs = [r for r in by(rows, kind=kind, method="policy") if r["pose"] is not None]
        n = Counter((r["stop"], r["closed_at_end"]) for r in rs)
        out.append(f"| {kind} | {len(rs)} | " + " | ".join(
            str(n[(s, c)]) for s in ("done", "force", "time") for c in (True, False)) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--layouts", type=int, default=CFG["layouts"], help="layouts per fixture kind")
    ap.add_argument("--procs", type=int, default=192)
    ap.add_argument("--kinds", nargs="+", default=TRAIN_KINDS + UNSEEN_KINDS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tables", action="store_true", help="write tables.md from trials.jsonl and stop")
    ap.add_argument("--methods", nargs="+", default=list(METHODS + GRIP_METHODS),
                    help="methods to run, push and grip names alike (free: the E0 checkpoint of --free-weights)")
    ap.add_argument("--free-weights", type=Path, help="the checkpoint of train_move_part --free (method free)")
    a = ap.parse_args()
    if not a.tables:
        a.out.mkdir(parents=True, exist_ok=True)
        more = CFG["layouts_more"] if a.layouts == CFG["layouts"] else {}
        (a.out / "config.json").write_text(json.dumps({"weights": str(a.weights), "cfg": CFG, "levels": E.CFG["levels"],
                                                       "layouts": {k: more.get(k, a.layouts) for k in a.kinds}}
                                                      | {"methods": a.methods, "free_weights": str(a.free_weights)},
                                                      indent=1))
        jobs = [(kind, VAL_SEED + i, str(a.out), str(a.weights), str(a.free_weights), tuple(a.methods))
                for i in range(max([a.layouts, *more.values()])) for kind in a.kinds if i < more.get(kind, a.layouts)]
        done = 0
        # a fresh process per few layouts: robosuite keeps memory of every environment it built
        with mp.get_context("fork").Pool(a.procs, maxtasksperchild=4) as pool, open(a.out / "trials.jsonl", "w") as f:
            for part in pool.imap_unordered(run_layout, jobs):
                f.writelines(json.dumps(r) + "\n" for r in part)
                f.flush()
                done += 1
                print(f"layout {done}/{len(jobs)}", flush=True)
    if set(a.methods) != set(METHODS + GRIP_METHODS):
        return          # the tables here compare M1's three methods; E0's: runs/lrl_m1e_20261125/e0
    rows = [json.loads(line) for line in open(a.out / "trials.jsonl")]
    (a.out / "tables.md").write_text(tables(rows) + "\n")
    print(tables(rows))


if __name__ == "__main__":
    main()
