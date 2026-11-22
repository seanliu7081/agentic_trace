#!/usr/bin/env python
"""TRACE bench (DESIGN §4; TRACE_method.md §8.2 stages 1-4) on the synthetic standard scenes, no model calls.

Stage 1 compares explore x exploit at equal evaluation budgets per slot: every explorer gets n_samples x
explore_iters rows per seed and level, every exploiter the rows L-BFGS uses (exploit_iters x ls_steps line-search
rows; Adam evaluates one row per iteration, so it runs ls_steps times as many iterations). "none" leaves its slot
unused. The plan budget is large, so time never cuts a run; wall time is reported separately. Stages 2 and 3 hold
the default pair. Success: S1-S5 certified, S6 (contradictory) a conflict that quotes its tilt term.

    python scripts/trace_bench.py [--seeds 3] [--stages 1,2,3,4]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from traceplan.contract import Term  # noqa: E402
from traceplan.program import Program, run_program_tests  # noqa: E402
from traceplan.search import CFG as SEARCH  # noqa: E402
from traceplan.service import TraceService  # noqa: E402
from traceplan.solvers import DEFAULTS  # noqa: E402
from traceplan.spline import SE3Spline  # noqa: E402
from traceplan.synthetic import make_request, standard_scenes, trivial_arm_check  # noqa: E402

BUDGET_MS = 6e4
EXPLORE, EXPLOIT = ("none", "random", "cem", "mppi"), ("none", "adam", "lbfgs")
ERRORS = ("wrong_entity", "unit_error", "not_attached", "world_coordinate", "wrong_scope")


def success(sid, r):
    if sid == "S6":
        return r.result == "conflict" and any(q["term"] == "upright" for q in r.quote or [])
    return r.result == "certified"


def run_setting(name, search, scenes, seeds):
    svc = TraceService({"search": search})
    rows = []
    for sid, sc in scenes.items():
        for seed in seeds:
            t = time.perf_counter()
            r = svc.plan(make_request(sc, seed=seed, budget_ms=BUDGET_MS), sc["snap"], trivial_arm_check)
            hard = [row["max_violation"] for row in r.report if row["role"] != "prefer"]
            rows.append({"setting": name, "scene": sid, "seed": seed, "result": r.result,
                         "stage": r.stats.get("stage"), "success": success(sid, r),
                         "max_hard_violation": round(max(hard, default=0.0), 3), "evals": r.stats["evals"],
                         "wall_ms": round((time.perf_counter() - t) * 1e3, 1)})
    return rows


def summarize(rows):
    out = {}
    for name in dict.fromkeys(r["setting"] for r in rows):
        rs = [r for r in rows if r["setting"] == name]
        out[name] = {"success": sum(r["success"] for r in rs), "n": len(rs),
                     "per_scene": {s: sum(r["success"] for r in rs if r["scene"] == s)
                                   for s in dict.fromkeys(r["scene"] for r in rs)},
                     "evals_median": float(np.median([r["evals"] for r in rs])),
                     "wall_ms_median": float(np.median([r["wall_ms"] for r in rs])),
                     "wall_ms_max": float(max(r["wall_ms"] for r in rs)),
                     "max_hard_violation_median": float(np.median([r["max_hard_violation"] for r in rs]))}
    return out


def stage1(scenes, seeds):
    rows = []
    for e in EXPLORE:
        for x in EXPLOIT:
            iters = tuple(i * DEFAULTS["ls_steps"] for i in SEARCH["exploit_iters"]) if x == "adam" \
                else SEARCH["exploit_iters"]
            rows += run_setting(f"{e}+{x}", {"explore": e, "exploit": x, "exploit_iters": iters}, scenes, seeds)
    return rows


def stage2(scenes, seeds, base):
    rows = [dict(r, setting="default(astar,all,al)") for r in base]
    for name, cfg in (("seeding=straight", {"seeding": "straight"}), ("seeding=random", {"seeding": "random"}),
                      ("continuation=off", {"continuation": "off"}), ("hard=penalty", {"hard": "penalty"})):
        rows += run_setting(name, cfg, scenes, seeds)
    return rows


def stage3(scenes, seeds, base):
    rows = [dict(r, setting=f"K={SEARCH['k_schedule']}") for r in base]
    for ks in ((4, 4, 4, 4), (11, 11, 11, 11), (5, 7, 11, 19)):
        rows += run_setting(f"K={ks}", {"k_schedule": ks}, scenes, seeds)
    return rows


# ------------------------------------------------------------------------------------- stage 4: program checks
def _ref(sc, rng):
    """(reference string, its z) for terms: an entity's top centre, or the end pose when the scene has none."""
    ents = list(sc["snap"].entities)
    if not ents:
        return "end", float(sc["ends"][0].p[2])
    e = sc["snap"].entities[ents[rng.integers(len(ents))]]
    return f"{e.id}.top_center", float(e.keypoint("top_center")[2])


def correct_term(sc, rng, i):
    ref, _ = _ref(sc, rng)
    return [Term("ok", "above", {"a": "eef", "b": ref, "h": float(rng.uniform(0.0, 0.05))}, "ineq", "prefer"),
            Term("ok", "reach_point", {"a": "eef", "b": "end"}, "eq", "prefer", "terminal",
                 sigma=float(rng.uniform(0.005, 0.03))),
            Term("ok", "tilt_limit", {"u": "-eef.z", "v": "world.up", "max": float(rng.uniform(0.5, 1.2))}, "ineq",
                 "prefer", sigma=0.05, param="max", unit="rad")][i % 3]


def injected(kind, sc, rng, i):
    """(bad term, holding of the program) for one injected error."""
    ref, z = _ref(sc, rng)
    hold = sc["holding"]
    if kind == "wrong_entity":
        ghost = ["mug_2", "bowl_1", "plate_3", "cup_7", "lid_1"][i % 5]
        return Term("bad", "reach_point", {"a": "eef", "b": f"{ghost}.centroid"}, "eq", "prefer", "terminal"), hold
    if kind == "unit_error":
        return [Term("bad", "tilt_limit", {"u": "-eef.z", "v": "world.up", "max": float(rng.uniform(5, 60))}, "ineq",
                     "prefer", sigma=0.05, param="max", unit="rad"),                        # degrees for radians
                Term("bad", "above", {"a": "eef", "b": ref, "h": float(rng.uniform(3, 30))}, "ineq", "prefer"),  # cm
                Term("bad", "reach_point", {"a": "eef", "b": "end"}, "eq", "prefer", "terminal",
                     sigma=float(rng.uniform(0.6, 3.0)))][i % 3], hold                       # cm sigma
    if kind == "not_attached":
        return [Term("bad", "above", {"a": "payload.base", "b": ref, "h": float(rng.uniform(0.0, 0.05))}, "ineq",
                     "prefer"),
                Term("bad", "tilt_limit", {"u": "payload.up", "v": "world.up", "max": 0.2, "u0": [0.0, 0.0, 1.0]},
                     "ineq", "prefer", sigma=0.05, param="max", unit="rad")][i % 2], None    # nothing marked held
    if kind == "world_coordinate":
        end, mid = sc["ends"][0].p, 0.5 * (sc["ends"][0].p + sc["start"]["p"])     # a goal or a via point
        pt = lambda c: (c + rng.normal(0, 0.02, 3)).round(3).tolist()
        return [Term("bad", "reach_point", {"a": "eef", "b": pt(end)}, "eq", "prefer", "terminal"),
                Term("bad", "reach_point", {"a": "eef", "b": pt(mid)}, "eq", "prefer", [0.4, 0.6])][i % 2], hold
    need = float(sc["start"]["p"][2]) - z
    return Term("bad", "above", {"a": "eef", "b": ref, "h": need + float(rng.uniform(0.05, 0.2))}, "ineq",
                "require"), hold


def stage4(scenes, n_programs=20):
    names = list(scenes)
    rows = []
    for i in range(n_programs):
        sc = scenes[names[i % len(names)]]
        rng = np.random.default_rng(1000 + i)
        st, e = sc["start"], sc["ends"][0]
        make = lambda: SE3Spline(st["R"], st["p"], e.R, e.p, 4, False, device="cpu", dtype=torch.float64)
        base = list(sc["terms"]) + [correct_term(sc, rng, i)]
        cases = [("correct", base, sc["holding"])] + \
            [(k, base + [t], h) for k in ERRORS for t, h in [injected(k, sc, rng, i)]]
        for kind, terms, hold in cases:
            found = run_program_tests(Program(terms, hold), sc["snap"], make, sc["bodies"], np.random.default_rng(i))
            hit = [f for f in found if f["term"] == "bad"]
            rows.append({"program": i, "scene": sc["name"], "kind": kind, "detected": bool(hit),
                         "tests": sorted({f["test"] for f in hit}), "alarm": kind == "correct" and bool(found),
                         "findings": found})
    summ = {k: {"detected": sum(r["detected"] for r in rows if r["kind"] == k), "n": n_programs,
                "by_test": {t: sum(t in r["tests"] for r in rows if r["kind"] == k)
                            for t in sorted({t for r in rows if r["kind"] == k for t in r["tests"]})}}
            for k in ERRORS}
    summ["false_alarms"] = {"count": sum(r["alarm"] for r in rows), "n": n_programs}
    return rows, summ


# ------------------------------------------------------------------------------------------------------- main
def table(title, summ):
    lines = [f"### {title}", "", "| setting | success | per scene | evals (median) | wall ms median / max | "
             "max hard viol. (median) |", "|---|---|---|---|---|---|"]
    for k, s in summ.items():
        ps = " ".join(f"{a}:{b}" for a, b in s["per_scene"].items())
        lines.append(f"| {k} | {s['success']}/{s['n']} | {ps} | {s['evals_median']:.0f} | "
                     f"{s['wall_ms_median']:.0f} / {s['wall_ms_max']:.0f} | {s['max_hard_violation_median']:.2f} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--stages", default="1,2,3,4")
    args = ap.parse_args()
    stages = {int(s) for s in args.stages.split(",")}
    out = REPO / "runs" / f"bench_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    scenes, seeds = standard_scenes(), list(range(args.seeds))
    TraceService().plan(make_request(scenes["S2"], budget_ms=BUDGET_MS), scenes["S2"]["snap"], trivial_arm_check)
    res, lines, t0 = {}, [f"# TRACE bench, {len(seeds)} seeds, equal evaluation budgets per slot", ""], time.time()
    base = None
    if 1 in stages:
        rows = stage1(scenes, seeds)
        res["stage1"] = {"rows": rows, "summary": summarize(rows)}
        d = f"{SEARCH['explore']}+{SEARCH['exploit']}"
        base = [r for r in rows if r["setting"] == d]
        best = max(res["stage1"]["summary"].items(), key=lambda kv: (kv[1]["success"], -kv[1]["wall_ms_median"]))
        res["stage1"]["default"], res["stage1"]["best"] = d, best[0]
        lines += table("Stage 1: explore x exploit", res["stage1"]["summary"])
        clear = best[1]["success"] >= res["stage1"]["summary"][d]["success"] + 3    # of 6 scenes x seeds
        res["stage1"]["clear_winner"] = best[0] if clear else None
        lines += [f"default {d}: {res['stage1']['summary'][d]['success']}; most successes (then fastest) {best[0]}: "
                  f"{best[1]['success']}; {'clear winner' if clear else 'no clear winner, keep the default'}", ""]
        print(f"stage 1 done {time.time() - t0:.0f} s", flush=True)
    base = base or run_setting("default", {}, scenes, seeds)
    if 2 in stages:
        rows = stage2(scenes, seeds, base)
        res["stage2"] = {"rows": rows, "summary": summarize(rows)}
        lines += table("Stage 2: seeding, continuation, hard terms", res["stage2"]["summary"])
        print(f"stage 2 done {time.time() - t0:.0f} s", flush=True)
    if 3 in stages:
        rows = stage3(scenes, seeds, base)
        res["stage3"] = {"rows": rows, "summary": summarize(rows)}
        lines += table("Stage 3: K schedule", res["stage3"]["summary"])
        print(f"stage 3 done {time.time() - t0:.0f} s", flush=True)
    if 4 in stages:
        rows, summ = stage4(scenes)
        res["stage4"] = {"rows": rows, "summary": summ}
        lines += ["### Stage 4: program checks (20 programs per error type)", "",
                  "| error | detected | by test |", "|---|---|---|"]
        lines += [f"| {k} | {summ[k]['detected']}/{summ[k]['n']} | {summ[k]['by_test']} |" for k in ERRORS]
        lines += [f"| false alarms on correct programs | {summ['false_alarms']['count']}/"
                  f"{summ['false_alarms']['n']} | |", ""]
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "bench.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("out:", out)


if __name__ == "__main__":
    main()
