#!/usr/bin/env python
"""Replay every logged TRACE request of a run on its saved snapshot (DESIGN §4), at no model or simulator cost.

The snapshot comes back with grid geometry: its `exact` (the harness's live map) is not saved. The arm and IK checks
belong to the harness; the run logged their answers with each request (meta["callbacks"]), and the replay returns
the logged answer for a query at the same positions (within ARM_TOL). A query that matches none (the replayed search
went elsewhere) gets an unconstrained answer (q = q_ref) and is counted. Compares result, end index and the largest
position difference between the logged and the replayed path.

    python scripts/trace_replay.py RUN_DIR [--device cuda:0]   -> RUN_DIR/trace/replay.json, replay.txt
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from traceplan.contract import ArmResult, SceneSnapshot  # noqa: E402
from traceplan.service import TraceService  # noqa: E402
from traceplan.spline import SE3Spline  # noqa: E402

ARM_TOL = 1e-3          # m: largest position difference at which a logged arm answer is reused


def recorded(calls):
    arm = [(p, r) for k, p, r in calls if k == "arm"]
    ik = [(p, r) for k, p, r in calls if k == "ik"]
    miss = {"arm": 0, "ik": 0}

    def arm_check(R, p, s, q_ref):
        for pr, r in arm:
            if pr.shape == p.shape and np.abs(pr - p).max() <= ARM_TOL:
                return r
        miss["arm"] += 1
        return ArmResult(True, np.asarray(q_ref, float))

    def ik_check(R, p, q_seed=None):
        for pr, r in ik:
            if np.abs(pr - p).max() <= 1e-9:
                return r
        miss["ik"] += 1
        return True, q_seed
    return arm_check, ik_check, miss


def path(resp, n=128):
    sp, th = SE3Spline.from_dict(resp.spline)
    return sp.sample_np(th, np.linspace(0.0, 1.0, n))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    tdir = Path(args.run_dir) / "trace"
    svc = TraceService({"device": args.device})
    snaps, rows = {}, []
    for f in sorted(tdir.glob("req_*.pkl")):
        req, old = pickle.load(open(f, "rb"))
        if req.scene_version not in snaps:
            snaps[req.scene_version] = SceneSnapshot.load(tdir / "snapshots" / f"v{req.scene_version}.npz")
        arm, ik, miss = recorded(req.meta.get("callbacks", []))
        new = svc.plan(req, snaps[req.scene_version], arm, ik)
        dp = float(np.abs(path(old) - path(new)).max()) if old.spline and new.spline else None
        rows.append({"request_id": req.request_id, "mode": req.mode, "logged": old.result, "replay": new.result,
                     "stage": [old.stats.get("stage"), new.stats.get("stage")],
                     "end_index": [old.end_index, new.end_index], "max_dp_mm": None if dp is None else dp * 1e3,
                     "solve_ms": [round(old.stats["solve_ms"], 1), round(new.stats["solve_ms"], 1)],
                     "unmatched": dict(miss)})
    both = [r for r in rows if r["max_dp_mm"] is not None]
    summ = {"requests": len(rows), "same_result": sum(r["logged"] == r["replay"] for r in rows),
            "same_end_index": sum(r["end_index"][0] == r["end_index"][1] for r in rows),
            "both_certified": len(both),
            "max_dp_mm": round(max((r["max_dp_mm"] for r in both), default=0.0), 3),
            "unmatched_arm_queries": sum(r["unmatched"]["arm"] for r in rows)}
    lines = ["| request | mode | logged | replay | stage (log/replay) | end idx | max dp mm | solve ms | unmatched |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['request_id']} | {r['mode']} | {r['logged']} | {r['replay']} | {r['stage'][0]}/{r['stage'][1]} | "
              f"{r['end_index'][0]}/{r['end_index'][1]} | "
              f"{'' if r['max_dp_mm'] is None else round(r['max_dp_mm'], 3)} | {r['solve_ms'][0]}/{r['solve_ms'][1]} | "
              f"{r['unmatched']['arm']}/{r['unmatched']['ik']} |" for r in rows]
    lines += ["", json.dumps(summ)]
    (tdir / "replay.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, default=str))
    (tdir / "replay.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
