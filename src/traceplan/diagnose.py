"""Standoff, perception limits and diagnosis of a failed request (TRACE_method.md §7.3 plus the integration doc).

The integration doc's rule: an axis whose workable window is smaller than perception error plus tracking error must
be closed by contact (along the approach, through the standoff) or by a closer look (laterally).
"""
from __future__ import annotations

import numpy as np

from .contract import Term
from .primitives import PRIMITIVES
from .search import continuation

CFG = {"z": 1.64, "s_min": 0.02, "s_max": 0.10, "s_extra": 0.005, "top_k": 3, "look_sd": 0.005,
       "near_entity": 0.05}


def tube_vmin(m):
    return m["tube0"] + m["tube_k"] * m["v_min"]


def _cov(snap, entity):
    return np.asarray(snap.entities[entity].cov, float) if entity in snap.entities else np.zeros((3, 3))


def auto_standoff(a, entity, snap, margins, z=CFG["z"]):
    """s = clip(z sqrt(a^T Sigma a) + tube(v_min) + guard_finger + 5 mm, 2 cm, 10 cm): perception error along the
    approach plus tracking plus the guard distance, so the guarded move from there ends on contact."""
    S = _cov(snap, entity)
    s = z * np.sqrt(max(a @ S @ a, 0.0)) + tube_vmin(margins) + margins["guard"]["finger"] + CFG["s_extra"]
    return float(np.clip(s, CFG["s_min"], CFG["s_max"]))


def window_flag(a, entity, window, snap, margins, z=CFG["z"]):
    """Not certifiable with this perception: lateral perception error plus tracking exceed the contact window."""
    S = _cov(snap, entity)
    P = np.eye(3) - np.outer(a, a)
    lat = float(np.sqrt(max(np.linalg.eigvalsh(P @ S @ P).max(), 0.0)))
    if z * lat + tube_vmin(margins) > window:
        return {"entity": entity, "term": "end_window", "uncertainty": lat, "tolerance": window, "axis": "lateral"}
    return None


def end_window_term(a, window, entity):
    return Term("end_window", "reach_point", {"a": "eef", "b": "end", "perp": [float(v) for v in a]}, "eq", "require",
                scope="terminal", sigma=window, src="goal", entity=entity, effect="attract", param="sigma", unit="m")


def quote_entry(term, row):
    """{term, prim, level, relax_to, param, value, needed, unit}: needed = value + sign * max_t r(t) for a
    parameter, the residual's own size for a tolerance (equality terms)."""
    spec = PRIMITIVES[term.prim]
    param, unit = term.param or spec.param, term.unit or spec.unit
    if param == "sigma":
        value, needed = term.args.get("sigma_accept", term.sigma), row["r_max"]
    elif param is None:
        value = needed = None
    else:
        value = term.args.get("param0", term.args[param])
        needed = value + spec.sign * row["r_max"]
    return {"term": term.id, "prim": term.prim, "level": term.level, "relax_to": term.relax_to, "param": param,
            "value": value, "needed": needed, "unit": unit}


def nearest_entity(snap, x):
    best = (None, np.inf)
    for e in snap.entities.values():
        d = float(np.linalg.norm(e.points - x, axis=1).min())
        if d < best[1]:
            best = (e.id, d)
    return best


def conflict(obj, spline, theta, rows, gen, cfg, n_dense):
    """Leave-one-out (§7.3): demote each of the top require terms (by violation, then multiplier) to prefer and
    rerun the last level from the best path; a term whose removal makes the rest feasible is in the conflict."""
    terms = {t.id: t for t in obj.terms}
    ranked = sorted((r for r in rows if r["role"] == "require"),
                    key=lambda r: (r["max_violation"], obj.multiplier_size(r["term"])), reverse=True)
    quotes = []
    for r in ranked[:CFG["top_k"]]:
        o = obj.demote(r["term"])
        _, th, _, _ = continuation(o, spline, theta[None], gen, cfg, np.inf, warm=True)
        rep = o.dense_report(th[0], n_dense)
        if all(x["max_violation"] <= 1.0 for x in rep["rows"] if x["role"] != "prefer"):
            quotes.append(quote_entry(terms[r["term"]], next(x for x in rep["rows"] if x["term"] == r["term"])))
    return quotes


def look_rule(rows, snap, margins):
    """Only safety clearance is violated, each by no more than the perception part of the margin, next to a labelled
    entity whose position is uncertain by more than 5 mm: a closer look can open the passage."""
    viol = [r for r in rows if r["role"] != "prefer" and r["max_violation"] > 1.0]
    if not viol or any(r["role"] != "safety" or r["prim"] != "clearance" for r in viol):
        return None
    worst = None
    for r in viol:
        deficit = r["r_max"] - r["sigma"]
        eid, d = nearest_entity(snap, r["at_x"])
        if deficit > margins["perception"] or eid is None or d > CFG["near_entity"]:
            return None
        sd = float(np.sqrt(max(np.linalg.eigvalsh(np.asarray(snap.entities[eid].cov, float)).max(), 0.0)))
        if sd <= CFG["look_sd"]:
            return None
        if worst is None or deficit > worst["deficit"]:
            worst = {"entity": eid, "term": r["term"], "uncertainty": sd, "tolerance": margins["perception"],
                     "axis": "any", "deficit": deficit,
                     "remedy": "look" if deficit > margins["tube_k"] * margins["v_min"] else "look_or_slow"}
    return worst
