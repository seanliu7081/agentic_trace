#!/usr/bin/env python
"""TRACE measures of one run (DESIGN §4), from its logs at no model or simulator cost.

Requests by result, fast-path share, solve time p50/p95 for the fast path and the search, conflicts resolved without a
model call, looks, guard stops per certified path, tube exits, and certificate honesty: certified executions whose
step range contains a forbidden contact onset (contacts.json in dev runs, collision.json in eval runs; robot contact
with anything but the held object, or the held object striking something). With trace/true/ snapshots each certified
trajectory is re-checked on the true scene (hand and payload spheres), which splits defects into perception error
(the path collides in truth, or the guard stopped the robot on a path that is clean in truth) and tracking error.

    python scripts/trace_measures.py RUN_DIR      -> RUN_DIR/trace/measures.json
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from scipy.spatial import cKDTree  # noqa: E402
from traceplan.contract import GROUPS, SceneSnapshot  # noqa: E402
from true_scene import exact_from  # noqa: E402


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


def onsets(run):
    """[(step, kind, a, b)] forbidden-candidate contact onsets on the control-step clock."""
    if (run / "contacts.json").exists():
        return [(o["step"], o["kind"], *o["pair"].split("|")) for o in json.loads((run / "contacts.json").read_text())
                ["onsets"]]
    col = json.loads((run / "collision.json").read_text())
    ratio = json.loads((run / "summary.json").read_text())["sim_steps"] / max(col["physics_steps"], 1)
    return [(int(e["physics_step"] * ratio), "robot", *e["owners"]) for e in col.get("onset_events", [])]


def forbidden(o, held):
    """Robot contact with anything but the held body; the held body touching anything. Contact logs name geom bodies,
    the true scene top-level bodies: one name extends the other."""
    same = lambda n: held is not None and (n.startswith(held) or held.startswith(n))
    _, kind, a, b = o
    return not same(b) if kind == "robot" else same(a) or same(b)


def true_check(path, req, resp):
    """({group: (minimum true clearance, minimum slack)} along the certified trajectory, the held body's name). Slack
    applies the certificate's zone rule: within req.zone of an end, the need drops to that end's clearance - 1 mm
    (a held object resting on its support starts in contact). The held body is the one around the payload spheres at
    the first sample (the harness's entity ids are not sim names)."""
    snap = SceneSnapshot.load(path)
    T, b = resp.timed, req.bodies
    X = T.p[:, None, :] + np.einsum("kij,sj->ksi", T.R, b.centers)
    held = None
    if req.holding is not None:
        pc = X[0, b.groups == GROUPS.index("payload")]
        held = min(snap.entities, key=lambda k: np.median(cKDTree(snap.entities[k].points).query(pc)[0]))
    c = exact_from(snap)(X.reshape(-1, 3), () if held is None else (held,)).reshape(X.shape[:2]) - b.radii
    if snap.table_z is not None:
        c = np.minimum(c, X[..., 2] - snap.table_z - b.radii)
    near = [np.linalg.norm(T.p - e, axis=1) < req.zone for e in (T.p[0], T.p[-1])]
    out = {}
    for g in np.unique(b.groups):
        cg = c[:, b.groups == g].min(1)
        need = np.zeros(len(cg))
        for m, c_end in zip(near, (cg[0], cg[-1])):
            need[m] = np.minimum(need[m], c_end - 0.001)
        out[GROUPS[g]] = (round(float(cg.min()), 4), round(float((cg - need).min()), 4))
    return out, held


def measures(run):
    run = Path(run)
    ev = [json.loads(line) for line in open(run / "events.jsonl")]
    log = [json.loads(line) for line in open(run / "trace" / "trace.jsonl")]
    summ = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    res = Counter(r["response"]["result"] for r in log)
    stage = [(r["response"]["stats"].get("stage"), r["response"]["stats"]["solve_ms"], r["request"]["budget_ms"])
             for r in log]
    fast = [ms for s, ms, _ in stage if s == "fast_path"]
    search = [ms for s, ms, _ in stage if s in ("search", "seed")]
    certified = [r for r in log if r["response"]["result"] == "certified"]
    react = Counter(e["reaction"] for e in ev if e["event"] == "trace_reaction")
    execs = [e for e in ev if e["event"] == "exec_timed"]
    status = Counter(e["status"] for e in execs)
    out = {"requests": len(log), "by_result": dict(res),
           "by_stage": dict(Counter(s for s, _, _ in stage)),
           "fast_path_share": round(sum(r["response"]["stats"].get("stage") == "fast_path" for r in certified)
                                    / max(len(log), 1), 3),
           "fast_path_share_of_certified": round(sum(r["response"]["stats"].get("stage") == "fast_path"
                                                     for r in certified) / max(len(certified), 1), 3),
           "solve_ms": {"fast_path": {"n": len(fast), "p50": pct(fast, 50), "p95": pct(fast, 95)},
                        "search": {"n": len(search), "p50": pct(search, 50), "p95": pct(search, 95),
                                   "max": round(max(search), 1) if search else None}},
           "over_budget": sum(ms > 1.5 * b for _, ms, b in stage),
           "max_solve_over_budget": round(max(ms / b for _, ms, b in stage), 2) if stage else None,
           "reactions": dict(react),
           "conflicts": {"all": res["conflict"], "resolved_without_model": react["relax"],
                         "repair_calls": react["repair"]},
           "looks": react["look"],
           "executions": dict(status), "n_executions": len(execs),
           "guard_stops_per_certified_path": round(status["guard_stop"] / max(len(execs), 1), 3),
           "tube_exits": status["tube_exit"],
           "guard_defects": dict(Counter(e["explanation"] for e in ev if e["event"] == "guard_defect"))}

    # certificate honesty and the true-scene split
    reqs = {e["request_id"]: pickle.load(open(run / "trace" / f"req_{e['request_id']}.pkl", "rb")) for e in execs}
    ons = onsets(run) if (run / "contacts.json").exists() or (run / "collision.json").exists() else None
    rows = []
    for e in execs:
        req, resp = reqs[e["request_id"]]
        row = {"request_id": e["request_id"], "status": e["status"], "steps": [e["first_step"], e["last_step"]],
               "holding": req.holding}
        tp = run / "trace" / "true" / f"{e['request_id']}.npz"
        if tp.exists():
            row["true_clearance"], row["held_body"] = true_check(tp, req, resp)
        if ons is not None:     # without a true snapshot the held body is unknown: only robot onsets count then
            row["onsets"] = [o for o in ons if e["first_step"] <= o[0] <= e["last_step"]
                             and forbidden(o, row.get("held_body"))]
        if e["status"] in ("guard_stop", "tube_exit", "contact") or row.get("onsets"):
            # the path collides in truth: perception missed geometry; a guard stop on the path (tracking inside the
            # tube) of a path clean in truth: perceived geometry that is not there; otherwise the robot left the path
            on_path = e["status"] == "guard_stop" and e["info"]["tracking_err"] <= e["info"]["tube"]
            row["defect"] = "unknown" if "true_clearance" not in row else \
                "perception_missed" if min(v[1] for v in row["true_clearance"].values()) < 0 else \
                "perception_phantom" if on_path else "tracking"
        rows.append(row)
    out["honesty"] = {"contact_log": ons is not None,
                      "certified_with_forbidden_onset": sum(bool(r.get("onsets")) for r in rows),
                      "true_checked": sum("true_clearance" in r for r in rows),
                      "true_violating_paths": sum(min(v[1] for v in r["true_clearance"].values()) < 0 for r in rows
                                                  if "true_clearance" in r),
                      "true_min_clearance_m": min((min(v[0] for v in r["true_clearance"].values()) for r in rows
                                                   if "true_clearance" in r), default=None),
                      "defects": dict(Counter(r["defect"] for r in rows if "defect" in r))}
    out["episode"] = {k: summ.get(k) for k in ("success", "success_step", "status", "sim_steps", "wall_s_policy",
                                                "robot_contacts_over_1mm", "new_object_contacts")}
    out["executions_detail"] = rows
    return out


def main():
    run = Path(sys.argv[1])
    out = measures(run)
    (run / "trace" / "measures.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "executions_detail"}, indent=1, default=str))


if __name__ == "__main__":
    main()
