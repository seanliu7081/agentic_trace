"""Certificate (TRACE_method.md §7.2 as changed by the integration doc): dense exact term check, the harness's arm
check (its controller model rolled along the path), then retiming from the certified clearance."""
from __future__ import annotations

import numpy as np

from .retime import retime


def q_reference(p, start_q, cand):
    """Posture reference along the path: the candidate's q_path resampled by arc length, else the linear
    interpolation by arc length from start_q to the candidate's q (start_q when it has none)."""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = d / d[-1] if d[-1] > 1e-9 else np.linspace(0.0, 1.0, len(p))
    start_q = np.asarray(start_q, float)
    if cand.src is not None and cand.src.q_path is not None:
        Q = np.asarray(cand.src.q_path, float)
        dq = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
        sq = dq / dq[-1] if dq[-1] > 1e-12 else np.linspace(0.0, 1.0, len(Q))
        return np.stack([np.interp(s, sq, Q[:, j]) for j in range(Q.shape[1])], 1)
    qe = start_q if cand.q is None else np.asarray(cand.q, float)
    return start_q + s[:, None] * (qe - start_q)


def certify(obj, theta_1, req, cand, arm_check, n=128):
    """(accepted, cert). cert: report rows, flags, reason when rejected, arm summary, rolled-out q, slack, timed."""
    rep = obj.dense_report(theta_1, n)
    cert = {"report": rep["rows"], "flags": obj.flags}
    bad = [r["term"] for r in rep["rows"] if r["role"] != "prefer" and r["max_violation"] > 1.0]
    if bad:
        cert["reason"] = "terms:" + ",".join(bad)
        return False, cert
    arm = arm_check(rep["R"], rep["p"], rep["ts"], q_reference(rep["p"], req.start_q, cand))
    cert["arm"] = {"ok": arm.ok, "reason": arm.reason, "at_t": arm.at_t, "min_clearance": arm.min_clearance}
    if not arm.ok:
        cert["reason"] = f"arm:{arm.reason}"
        return False, cert
    m, lim = req.margins, req.limits
    slack = np.min(np.stack(list(rep["slack"].values())), 0) if rep["slack"] else np.full(n, np.inf)
    if arm.clearance is not None:          # arm clearance comes minus guard + tube(v_min) + perception
        slack = np.minimum(slack, np.asarray(arm.clearance, float) + m["tube0"] + m["tube_k"] * m["v_min"])
    cfg = {k: m[k] for k in ("tube0", "tube_k", "v_min") + tuple(k for k in ("tube_w", "w_min") if k in m)}
    cfg.update({k: lim[k] for k in ("v_max", "a_max", "w_max", "dt", "v_slow")})
    cfg["slow"] = rep["slow"] if rep["slow"].any() else None
    cert.update(timed=retime(rep["p"], rep["R"], arm.q, rep["ts"], slack, cfg), q=arm.q, slack=slack,
                arm_clearance=arm.clearance)
    return True, cert
