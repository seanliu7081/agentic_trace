#!/usr/bin/env python
"""Hand-off error of the actuation stages on the development tasks (LRL_PLAN.md M1 step 6).

  handoff_error.py <dev dir> [--out HANDOFF_ERROR.md]

<dev dir> holds <task>/init<N>/ runs of scripts/run_episode.py --suite ... --stage-truth. From each run's
events.jsonl (what the harness believed) and truth.jsonl (what the simulator held at the same events) it takes every
actuation attempt and writes the distributions of the belief errors and one table per task.

Words:
  candidate   a contact the stage chose: lrl_entry with a family, or v2's part_grasp / press_plan
  attempt     a candidate the hand got to: TRACE handed off, or the transit counted as an arrival
  target      the joint the task's goal reads (the top drawer, the stove knob)
Per attempt:
  direction   angle between the believed motion direction (push, pull: the face normal; turn: the axis) and the
              target's true joint axis, as lines
  travel      the part's believed stand-out from its fixture face (the measurement before the stroke: for a closing
              push the remaining travel) minus its true opening
  position    where the believed contact point lies along the true axis relative to the part's first true surface
              on the hand's side: > 0 in front of it, < 0 behind it (stage_truth's probe)
  reached     LIBERO's own rule on the target joint after the attempt (closed: q above the closed range's lower
              end; open: q below the open range's upper end; on: q at least the on range's lower end), or LIBERO's
              done inside the attempt
"""
import argparse
import json
from pathlib import Path

import numpy as np

TRUTH_EVENTS = ("stage_start", "handoff", "lrl_arrival", "part_state_before", "part_state", "press_plan",
                "part_grasp", "lrl_entry", "lrl_skill", "stage_done", "stage_failed")
PRIOR = {"position": 0.03, "direction": 20.0, "travel": 0.03}     # LRL_PLAN.md M0 step 4: m, deg, m


def load(run):
    ev = [json.loads(l) for l in open(run / "events.jsonl")]
    tr = [json.loads(l) for l in open(run / "truth.jsonl")]
    head, tr = tr[0], tr[1:]
    k = 0
    for e in ev:                      # truth records are written in the order of the events they follow
        if e["event"] in TRUTH_EVENTS and k < len(tr):
            assert tr[k]["event"] == e["event"], (run, e["event"], tr[k]["event"])
            e["truth"], k = tr[k], k + 1
    return ev, head, json.loads((run / "summary.json").read_text())


def target_of(head):
    """(joint, what LIBERO asks of it: 'open' | 'close' | 'turnon' | None, its fixture's ranges)."""
    joint, pred = None, None
    for g in head["goal"]:
        for arg in g[1:]:
            if head["goal_joints"].get(arg):
                joint = head["goal_joints"][arg][0]
                pred = g[0] if g[0] in ("open", "close", "turnon") else pred
    fix = next(v for k, v in head["fixtures"].items() if v and joint.startswith(k))
    return joint, pred, fix


def holds(q, want, fix):
    if want == "close":
        return q > min(fix["default_close_ranges"])
    if want == "open":
        return q < max(fix["default_open_ranges"])
    return q >= min(fix["default_turnon_ranges"])


def opening(q, fix):
    """How far a sliding part truly stands out (m): its joint value from the closed end, toward the open range."""
    if not fix.get("default_open_ranges") or not fix.get("default_close_ranges"):
        return None
    c, o = min(fix["default_close_ranges"]), np.mean(fix["default_open_ranges"])
    return max(0.0, (q - c) * np.sign(o - c))


def episode(run):
    """The actuation stage calls of one run, each with its candidates."""
    ev, head, summary = load(run)
    joint, pred, fix = target_of(head)
    calls, cur, att, mount, state = [], None, None, None, None
    for e in ev:
        k, tr = e["event"], e.get("truth")
        if k == "stage_start":
            st = e["stage"]
            cur, att, mount, state = None, None, None, None
            if st["op"] == "actuate":
                want = {"closed": "close", "open": "open"}.get(st.get("goal")) or pred or \
                    ("turnon" if st.get("motion") == "rotate" else None)
                cur = {"id": st["id"], "motion": st["motion"], "want": want, "t0": e["t"], "t1": None, "how": None,
                       "cands": []}
                calls.append(cur)
            continue
        if cur is None:
            continue
        if tr is not None and att is not None:
            att["q_end"] = tr["joints"][joint]["q"]
        if k in ("stage_done", "stage_failed", "end"):
            cur["t1"], cur["how"] = e["t"], k
            cur, att = None, None
        elif k == "mount_frame":
            mount = e
        elif k in ("part_state_before", "part_state"):
            state = e
        elif k in ("part_grasp", "press_plan") or (k == "lrl_entry" and e["family"] is not None):
            if k == "lrl_entry" and att is not None and att["kind"] == "v2 grip" and not att["reached"]:
                att.update(kind="grip", n=e["normal"])      # lrl's grip candidate: the hold part_grasp just logged
                continue
            n = e["normal"] if k == "lrl_entry" else mount["normal"]
            probe = tr["probe"][joint]
            near = min(tr["probe"], key=lambda j: tr["probe"][j]["near_m"])
            att = {"kind": e["family"] if k == "lrl_entry" else {"part_grasp": "v2 grip", "press_plan": "v2 press"}[k],
                   "n": n, "point": e.get("point", e.get("p")), "probe": probe, "nearest": near, "target": joint,
                   "axis": tr["joints"][joint]["axis"], "q0": tr["joints"][joint]["q"], "q_end": None,
                   "state": state, "reached": False, "arrival": False, "runs": [], "step": e["step"], "t": e["t"]}
            cur["cands"].append(att)
        elif k == "handoff" and att is not None and e["tag"] in ("to_part", "to_press"):
            att["reached"] = True
        elif k == "lrl_arrival" and att is not None and e["arrived"]:
            att["reached"] = att["arrival"] = True
        elif k == "lrl_skill" and att is not None:
            att["runs"].append(e)
    for c in calls:
        if c["t1"] is None:
            c["t1"], c["how"] = ev[-1]["t"], "end"
    # LIBERO's done inside the last attempt of a stage call that the episode's end cut: the goal was reached there
    if summary["success"] and calls and calls[-1]["how"] == "end":
        tried = [a for a in calls[-1]["cands"] if a["reached"]]
        if tried:
            tried[-1]["done_inside"] = True
    for c in calls:
        for a in c["cands"]:
            n, ax = np.asarray(a["n"], float), np.asarray(a["axis"], float)
            a["angle"] = float(np.degrees(np.arccos(min(1.0, abs(n @ ax) / np.linalg.norm(n)))))
            s, b = a["state"], None
            if s is not None and s.get("measured") and s.get("offset_m") is not None:
                b = float(s["offset_m"])
            o = opening(a["q0"], fix)
            a["believed_out"], a["true_out"] = b, o
            a["travel"] = None if b is None or o is None else b - o
            # the hand comes from the side the believed face normal points to
            side = "front" if n @ ax > 0 else "back"
            at = a["probe"][side]
            a["on_line"] = at is not None
            at = at if at is not None else a["probe"]["hi" if side == "front" else "lo"]
            a["position"] = -at if side == "front" else at
            a["goal"] = a.get("done_inside", False) or \
                (a["q_end"] is not None and c["want"] is not None and bool(holds(a["q_end"], c["want"], fix)))
    return {"run": run, "summary": summary, "joint": joint, "calls": calls}


def dist(x, scale=1.0, absolute=True):
    x = np.array([v for v in x if v is not None], float) * scale
    if len(x) == 0:
        return "- | - | - | - | 0"
    y = np.abs(x) if absolute else x
    return f"{np.median(y):.2f} | {np.percentile(y, 90):.2f} | {y.max():.2f} | {np.median(x):+.2f} | {len(x)}"


def inside(x, limit, scale=1.0):
    x = np.abs([v for v in x if v is not None]) * scale
    if len(x) == 0:
        return "not measured"
    worst, p90 = x.max(), np.percentile(x, 90)
    n_out = int((x > limit * scale).sum())
    if worst <= limit * scale:
        return f"inside (largest {worst:.2f})"
    return f"OUTSIDE: {n_out} of {len(x)} beyond the prior, 90th percentile {p90:.2f}, largest {worst:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs = sorted(p for p in args.dev.glob("*/init[0-9]") if (p / "summary.json").exists() and
                  (p / "truth.jsonl").exists())
    eps = [episode(r) for r in runs]
    atts = [(e, c, a) for e in eps for c in e["calls"] for a in c["cands"] if a["reached"]]
    cands = [a for e in eps for c in e["calls"] for a in c["cands"]]
    firsts = []                           # the first attempt of each episode's first actuation stage call
    for e in eps:
        tried = [a for c in e["calls"][:1] for a in c["cands"] if a["reached"]]
        if e["calls"]:
            firsts.append((e, e["calls"][0], tried[0] if tried else None))
    L = []
    w = L.append
    builds = sorted({e["summary"]["build"] for e in eps})
    w("# Hand-off error on the development tasks (LRL_PLAN.md M1 step 6)")
    w("")
    w(f"Written by `scripts/handoff_error.py` from {len(eps)} episodes under `{args.dev}` (build {', '.join(builds)}; "
      f"lrl, parts: skill; gpt-6.2-astra, effort high; caps 600 s and 50 calls). Belief from `events.jsonl`, truth "
      f"from `truth.jsonl` (simulator state at the same events, development tasks only).")
    w("")
    w("- candidate: a contact the stage chose (an `lrl_entry` with a family, or v2's `part_grasp`).")
    w("- attempt: a candidate the hand got to (TRACE handed off, or the transit counted as an arrival).")
    w("- direction: angle between the believed motion direction (turn: axis) and the true joint axis, as lines.")
    w("- travel: believed stand-out of the part from its fixture face, measured before the stroke (for a closing "
      "push the remaining travel), minus the true opening.")
    w("- position: the believed contact point along the true axis, relative to the part's first true surface on "
      "the hand's side (+ in front of it, - behind it).")
    w("- reached: LIBERO's rule on the target joint after the attempt, or LIBERO's done inside it.")
    w("")
    n_succ = sum(e["summary"]["success"] for e in eps)
    w("## Totals")
    w("")
    w(f"- LIBERO success: {n_succ} of {len(eps)} episodes.")
    none = [e for e in eps if not e["calls"]]
    w(f"- Episodes that never came to an actuation stage: {len(none)}"
      + (" (" + ", ".join(f"{e['run'].parent.name} init {e['run'].name[4:]}" for e in none) + ")." if none else "."))
    dry = [e for e in eps if e["calls"] and not any(a["reached"] for c in e["calls"] for a in c["cands"])]
    w(f"- Episodes whose actuation stages made no attempt (no candidate, or none the hand got to): {len(dry)}"
      + (" (" + ", ".join(f"{e['run'].parent.name} init {e['run'].name[4:]}" for e in dry) + ")." if dry else "."))
    w(f"- Actuation stage calls: {sum(len(e['calls']) for e in eps)}; candidates {len(cands)}; attempts {len(atts)} "
      f"({sum(a['arrival'] for _, _, a in atts)} of them arrivals on contact).")
    ok = [f for f in firsts if f[2] is not None and f[2]["goal"]]
    w(f"- First attempt of the episode's first actuation stage reached the goal by truth: {len(ok)} of {len(firsts)} "
      f"episodes with an actuation stage ({100 * len(ok) / max(len(firsts), 1):.0f} %). Gate G1 asks for 80 %.")
    for label, sel in (("closing push (lrl skill)", lambda c: c["motion"] == "push"),
                       ("opening pull (v2's stroke)", lambda c: c["motion"] == "pull"),
                       ("turn (v2's turn)", lambda c: c["motion"] == "rotate")):
        grp = [f for f in firsts if sel(f[1])]
        if grp:
            w(f"  - {label}: {sum(1 for f in grp if f[2] is not None and f[2]['goal'])} of {len(grp)}.")
    wrong = [a for a in cands if a["nearest"] != a["target"]]
    w(f"- Candidates whose believed contact point lies nearest to another moving part than the target: {len(wrong)} "
      f"of {len(cands)}.")
    w(f"- Attempts whose line through the believed point along the true axis misses the part: "
      f"{sum(not a['on_line'] for _, _, a in atts)} of {len(atts)} (their position is taken to the part's extent).")
    w("")
    w("## Distributions")
    w("")
    w("Absolute values; the signed median shows a bias. Position and travel in cm, direction in degrees.")
    w("")
    w("| quantity | set | median | 90th pct | 100th pct | signed median | count |")
    w("|---|---|---|---|---|---|---|")
    sets = (("all attempts", [a for _, _, a in atts]), ("first attempts", [f[2] for f in firsts if f[2] is not None]),
            ("push attempts", [a for _, c, a in atts if c["motion"] == "push"]),
            ("pull attempts", [a for _, c, a in atts if c["motion"] == "pull"]),
            ("turn attempts", [a for _, c, a in atts if c["motion"] == "rotate"]))
    for q, scale in (("direction", 1.0), ("travel", 100.0), ("position", 100.0)):
        for name, A in sets:
            if A:
                key = "angle" if q == "direction" else q
                w(f"| {q} | {name} | {dist([a[key] for a in A], scale)} |")
    w("")
    w("Time in the actuation stage, s (stage calls that made at least one attempt, and all stage calls):")
    w("")
    w("| set | median | 90th pct | 100th pct | count |")
    w("|---|---|---|---|---|")
    for name, C in (("calls with an attempt", [c for e in eps for c in e["calls"] if any(a["reached"] for a in c["cands"])]),
                    ("all calls", [c for e in eps for c in e["calls"]])):
        t = np.array([c["t1"] - c["t0"] for c in C])
        if len(t):
            w(f"| {name} | {np.median(t):.0f} | {np.percentile(t, 90):.0f} | {t.max():.0f} | {len(t)} |")
    w("")
    w("Entry kind of the attempts, and how many reached the goal:")
    w("")
    w("| entry kind | attempts | arrivals on contact | reached the goal | first attempts | first attempts that reached |")
    w("|---|---|---|---|---|---|")
    for kind in sorted({a["kind"] for _, _, a in atts}):
        A = [a for _, _, a in atts if a["kind"] == kind]
        F = [f[2] for f in firsts if f[2] is not None and f[2]["kind"] == kind]
        w(f"| {kind} | {len(A)} | {sum(a['arrival'] for a in A)} | {sum(a['goal'] for a in A)} | {len(F)} | "
          f"{sum(a['goal'] for a in F)} |")
    w("")
    w("## Against the prior range (position 3 cm, direction 20 degrees, travel 3 cm)")
    w("")
    A = [a for _, _, a in atts]
    w(f"- direction: {inside([a['angle'] for a in A], PRIOR['direction'])} degrees, {len(A)} attempts.")
    w(f"- travel: {inside([a['travel'] for a in A], PRIOR['travel'], 100.0)} cm, "
      f"{sum(a['travel'] is not None for a in A)} attempts with a measured state before the stroke.")
    w(f"- position: {inside([a['position'] for a in A], PRIOR['position'], 100.0)} cm, {len(A)} attempts.")
    w("")
    w("A grip's contact point is the centre between the pads, which is meant to lie inside the handle: its position "
      "is negative by about half the handle's depth. A press's contact point is on the part's believed outer face.")
    w("")
    w("Attempts beyond the prior in at least one quantity:")
    w("")
    w("| task | init | stage call | entry kind | believed normal | direction deg | travel cm | position cm | "
      "reached the goal |")
    w("|---|---|---|---|---|---|---|---|---|")
    for e, c, a in atts:
        over = a["angle"] > PRIOR["direction"] or abs(a["position"]) > PRIOR["position"] or \
            (a["travel"] is not None and abs(a["travel"]) > PRIOR["travel"])
        if over:
            fmt = lambda v: "-" if v is None else f"{100 * v:+.1f}"
            w(f"| {e['run'].parent.name} | {e['run'].name[4:]} | {c['id']} ({c['motion']}) | {a['kind']} | "
              f"{np.round(a['n'], 2).tolist()} | {a['angle']:.1f} | {fmt(a['travel'])} | {fmt(a['position'])} | "
              f"{'yes' if a['goal'] else 'no'} |")
    w("")
    w("## Per task")
    for task in sorted({e["run"].parent.name for e in eps}):
        w("")
        w(f"### {task}")
        w("")
        w("| init | LIBERO success | control time, s | model calls | repairs | first attempt reached the goal | "
          "attempts / candidates | actuation stage calls | time in them, s | first attempt: kind, direction deg, "
          "travel cm, position cm |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for e in [e for e in eps if e["run"].parent.name == task]:
            s = e["summary"]
            f = next((f for f in firsts if f[0] is e), None)
            a = f[2] if f else None
            n_att = sum(a_["reached"] for c in e["calls"] for a_ in c["cands"])
            n_c = sum(len(c["cands"]) for c in e["calls"])
            fmt = lambda v, sc=100.0: "-" if v is None else f"{sc * v:+.1f}"
            first = "-" if a is None else f"{a['kind']}, {a['angle']:.1f}, {fmt(a['travel'])}, {fmt(a['position'])}"
            w(f"| {e['run'].name[4:]} | {'yes' if s['success'] else 'no'} ({s['status']}) | {s['wall_s_policy']:.0f} | "
              f"{s['model']['calls']} | {s['repairs']} | "
              f"{'-' if f is None else 'no attempt' if a is None else 'yes' if a['goal'] else 'no'} | {n_att} / {n_c} | "
              f"{len(e['calls'])} | {sum(c['t1'] - c['t0'] for c in e['calls']):.0f} | {first} |")
    text = "\n".join(L) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
