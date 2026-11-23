"""Shared timeline / state helpers for the post-episode videos (task dashboard + 3D obstacle map).

Both renderers MUST build their frame list with `build_timeline` so the two videos stay frame-aligned
(same length, same hold frames) and can be played side by side.

Timeline: one output frame per recorded telemetry frame (every 2 simulator steps = 10 Hz simulated time,
played at 10 fps = real time). Where the harness worked without stepping the simulator for more than
`hold_gap` seconds of wall time (S2 planning, Astra calls), hold frames are inserted: the camera image and
robot state stay at the last recorded frame while the event state advances through the gap, so the planning /
model-call phases are visible. Hold frames are capped (they compress long waits).
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path

PHASES = ("PERCEIVE", "S1 MODEL CALL", "S2 PLAN", "EXECUTE / MONITOR")
PHASE_LABELS = ("1 PERCEIVE  (RGB-D + SAM)", "2 S1 PROGRAM / REPAIR / VERIFY", "3 S2 PLAN", "4 EXECUTE / MONITOR")
S1_EVENTS = ("program", "repair", "verify", "effect_check", "v2_pose", "v2_look")   # v2_*: placement v2 calls
S1_KIND = {"v2_pose": "place_pose", "v2_look": "place_look"}                       # event -> model call kind
PERCEIVE_EVENTS = ("scene", "survey", "grounded", "ground_fail", "ground_alias", "support_plane", "side_look",
                   "look_failed", "placed", "relocated", "mount_frame", "payload", "payload_centering",
                   "repair_ground_fail")
PLAN_EVENTS = ("grasp_cands", "grasp_prefilter", "grasp_plan", "place_cands", "place_precheck", "place_plan",
               "place_ik", "place_ik_fail", "part_grasp_cands", "part_grasp", "rrt", "rrt_fail", "plan_fail",
               "program_warnings", "place_cands_soft", "trace_request", "trace_result", "trace_seed",
               "trace_reaction")
PERCEIVE_MOVES = ("look", "look_toward", "side_look_back", "look_retry", "look_toward_retry")


def _jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_run(run_dir):
    """Everything the renderers need from a run directory (paths + parsed logs)."""
    d = Path(run_dir)
    rec = d / "rec"
    run = {
        "dir": d, "rec": rec,
        "meta": json.loads((rec / "meta.json").read_text()) if (rec / "meta.json").exists() else {},
        "telemetry": _jsonl(rec / "telemetry.jsonl"),
        "plans": _jsonl(rec / "plans.jsonl"),
        "maps": _jsonl(rec / "maps" / "index.jsonl"),
        "events": _jsonl(d / "events.jsonl"),
        "summary": json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None,
        "args": json.loads((d / "args.json").read_text()) if (d / "args.json").exists() else {},
    }
    run["event_t"] = [float(e.get("t", 0.0)) for e in run["events"]]
    run["plan_t"] = [float(p.get("t", 0.0)) for p in run["plans"]]
    run["map_t"] = [float(m.get("t", 0.0)) for m in run["maps"]]
    return run


def phase_of(event):
    """Pipeline phase (index into PHASES) of one harness event."""
    ev = event.get("event", "")
    if ev in S1_EVENTS:
        return 1
    if ev in PERCEIVE_EVENTS:
        return 0
    if ev in PLAN_EVENTS:
        return 2
    if ev == "move" and str(event.get("tag", "")) in PERCEIVE_MOVES:
        return 0
    return 3


def gap_label(events):
    """Short caption for a hold interval from the events that happened in it."""
    for e in events:
        if e.get("event") in S1_EVENTS:
            rec = e.get("record") or {}
            lat = rec.get("latency_s") if isinstance(rec, dict) else None
            kind = S1_KIND.get(e["event"], e["event"]).upper()
            return f"S1 {kind} call (Astra" + (f", {lat:.1f} s)" if isinstance(lat, (int, float)) else ")")
    names = [e.get("event", "") for e in events]
    if any(n in PLAN_EVENTS for n in names):
        key = next(n for n in names if n in PLAN_EVENTS)
        if key.startswith("grasp"):
            return "S2 grasp planning"
        if key.startswith("place"):
            return "S2 placement planning"
        if key.startswith("part"):
            return "S2 part-grasp planning"
        if key.startswith("trace"):
            return "S2 motion planning (TRACE)"
        return "S2 motion planning"
    if any(n in ("grounded", "support_plane", "ground_fail") for n in names):
        return "S2 grounding objects (SAM 3.1 + depth)"
    if any(n == "placed" for n in names):
        return "S2 checking the placement"
    return "S2 processing"


def build_timeline(run, hold_gap=1.0, hold_per_s=2.0, max_hold=30, tail_frames=20):
    """List of output frames. Each item:
       frame   recorded frame index (camera image / telemetry / robot points to show)
       step    simulator step of that frame
       t       harness clock (s) used to select the event / plan / map state shown
       hold    True for inserted hold frames (no simulator time passes)
       label   caption for hold frames (None otherwise)
       final   True for the closing result frames
    """
    tel = run["telemetry"]
    ev_t = run["event_t"]
    out = []
    for k, rec in enumerate(tel):
        t_k = float(rec["t"])
        out.append({"frame": k, "step": rec["step"], "t": t_k, "hold": False, "label": None, "final": False})
        if k + 1 < len(tel):
            gap = float(tel[k + 1]["t"]) - t_k
            if gap > hold_gap:
                lo, hi = bisect.bisect_right(ev_t, t_k), bisect.bisect_right(ev_t, t_k + gap)
                label = gap_label(run["events"][lo:hi]) + f"  |  {gap:.1f} s wall, simulator paused"
                n = int(min(max_hold, max(2, round(gap * hold_per_s))))
                for j in range(n):
                    out.append({"frame": k, "step": rec["step"], "t": t_k + gap * (j + 1) / (n + 1), "hold": True,
                                "label": label, "final": False})
    if tel:
        last = len(tel) - 1
        t_end = max([float(tel[-1]["t"])] + ev_t)
        for j in range(tail_frames):
            out.append({"frame": last, "step": tel[-1]["step"], "t": t_end, "hold": True, "label": None,
                        "final": True})
    return out


def events_upto(run, t):
    """Harness events with time <= t (list slice)."""
    return run["events"][:bisect.bisect_right(run["event_t"], t)]


def last_event(run, t, names=None):
    evs = events_upto(run, t)
    for e in reversed(evs):
        if names is None or e.get("event") in names:
            return e
    return None


def current_plan(run, t):
    """Latest planned move (plans.jsonl record) at time t, or None."""
    i = bisect.bisect_right(run["plan_t"], t)
    return run["plans"][i - 1] if i > 0 else None


def current_map(run, t):
    """Index record of the latest map snapshot at time t (its 'file' is under rec/maps/), or None."""
    i = bisect.bisect_right(run["map_t"], t)
    return run["maps"][i - 1] if i > 0 else (run["maps"][0] if run["maps"] else None)


def program_state(run, t):
    """Current S1 program and per-stage status at time t.
    Returns (program dict or None, {stage_id: 'pending'|'running'|'done'|'failed'}, current stage id or None,
             list of S1 calls [(kind, latency_s, ok, t)], last REPAIR diagnosis or None)."""
    prog, status, cur, calls, diag = None, {}, None, [], None
    for e in events_upto(run, t):
        ev = e.get("event")
        if ev in ("program", "repair") and isinstance(e.get("program"), dict) and e["program"].get("stages") is not None:
            prog = e["program"]
            status = {s["id"]: "pending" for s in prog.get("stages", [])}
            cur = None
            if ev == "repair":
                diag = e.get("diagnosis")
        if ev in S1_EVENTS:
            rec = e.get("record") or {}
            calls.append((ev, rec.get("latency_s") if isinstance(rec, dict) else None,
                          bool(rec.get("ok", True)) if isinstance(rec, dict) else True, e.get("t")))
        if ev == "stage_start":
            sid = (e.get("stage") or {}).get("id")
            if sid is not None:
                status[sid] = "running"
                cur = sid
        elif ev == "stage_done":
            status[e.get("stage")] = "done"
        elif ev == "stage_failed":
            status[e.get("stage")] = "failed"
    return prog, status, cur, calls, diag


def objects_at(run, t):
    """Grounded objects at time t: {object id (S1 id): summary dict from the latest 'grounded' event}."""
    objs = {}
    for e in events_upto(run, t):
        if e.get("event") == "grounded" and isinstance(e.get("summary"), dict):
            objs[str(e.get("object"))] = dict(e["summary"], how=e.get("how"))
    return objs
