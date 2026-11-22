"""Retiming (DESIGN §2.9): time a certified geometric path from its clearance, fast where there is room and slow in
tight passages, because the tracking tube grows with speed.

Timing law, on the N dense samples of the certified path (a polyline in p, piecewise geodesic in R):

- Speed cap per sample: v_cap = clip((slack - tube0) / tube_k, v_min, v_max), so that the tube tube0 + tube_k * v
  fits inside the certified slack. Where cfg["slow"] is set, the cap is at most v_slow.
- Within a segment, rotation advances in proportion to arc length, so the angular-rate limit caps the segment's
  speed at w_max * ds / dtheta. A segment's cap (the lower of its two sample caps and its angular cap) applies to
  both of its end samples, so the speed between samples stays below the caps of both neighbours.
- Forward and backward passes with the acceleration a_max along the path, from rest to rest: v^2 changes by at
  most 2 a_max ds per segment. Acceleration is constant within a segment.
- Segment time = max(2 ds / (v_k + v_k+1), dtheta / w_max). The second term times rotations in place (ds = 0).
- Resampling on the control grid t = 0, dt, ..., n dt with n = ceil(T / dt). The pose is held at the end for the
  remainder of the last step. p is interpolated linearly and R by slerp, at the same within-segment fraction, which
  is the exact arc-length fraction of the constant-acceleration motion. q is interpolated linearly at that fraction,
  and speed exactly (it is linear in time within a segment).
- tube = tube0 + tube_k * speed per output sample.

Rotations use scipy's Rotation (float64), like the rest of certification.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import Timed


def retime(p, R, q, s, slack, cfg):
    """p (N,3), R (N,3,3), q (N,7), s (N,) path parameter, slack (N,) certified clearance slack (m) -> Timed.

    s is not used by the timing law, which needs only the geometry; it is part of the dense-sample tuple the
    certificate passes. cfg keys (all required, from the request's margins and limits): tube0 (m), tube_k (s),
    v_min, v_max (m/s), a_max (m/s^2, along the path), w_max (rad/s), dt (s), v_slow (m/s). Optional: slow, an (N,)
    bool mask of samples capped at v_slow (a relaxed known-space term is violated there).
    """
    tube0, tube_k, a_max, w_max, dt = (cfg[k] for k in ("tube0", "tube_k", "a_max", "w_max", "dt"))
    tube_w, w_min = cfg.get("tube_w", 0.0), cfg.get("w_min", 0.0)
    rot = Rotation.from_matrix(R)
    rel = (rot[:-1].inv() * rot[1:]).as_rotvec()                  # (N-1,3) segment rotation, in its start frame
    dth = np.linalg.norm(rel, axis=1)
    du = np.linalg.norm(np.diff(p, axis=0), axis=1)

    vc = np.clip((slack - tube0) / tube_k, cfg["v_min"], cfg["v_max"])
    if cfg.get("slow") is not None:
        vc = np.where(cfg["slow"], np.minimum(vc, cfg["v_slow"]), vc)
    # turning the hand drifts its position too (the OSC couples position and orientation): the tube is
    # tube0 + tube_k v + tube_w w, so rotation slows where the slack is tight, and while translating w = (dth/du) v
    room = np.minimum(slack[:-1], slack[1:]) - tube0
    wseg = np.clip(room / tube_w, w_min, w_max) if tube_w > 0 else np.full_like(du, w_max)
    rate = np.divide(dth, du, out=np.zeros_like(du), where=du > 0)
    vrot = np.maximum(room / (tube_k + tube_w * rate), cfg["v_min"])
    seg = np.minimum.reduce([vc[:-1], vc[1:], vrot,
                             np.divide(wseg * du, dth, out=np.full_like(du, np.inf), where=dth > 0)])
    cap = np.minimum(np.r_[seg[0], seg], np.r_[seg, seg[-1]])
    cap[[0, -1]] = 0.0                                             # start and end at rest

    # Both passes in closed form. Forward: v_k^2 = min_{j<=k} cap_j^2 + U_k - U_j, U = cumulative 2 a_max ds.
    # Backward: v_k^2 = min_{j>=k} v_j^2 + U_j - U_k.
    U = np.r_[0.0, np.cumsum(2.0 * a_max * du)]
    w = U + np.minimum.accumulate(cap ** 2 - U)
    w = np.minimum.accumulate((w + U)[::-1])[::-1] - U
    v = np.sqrt(np.maximum(w, 0.0))

    vs = v[:-1] + v[1:]
    seg_t = np.maximum(np.divide(2.0 * du, vs, out=np.zeros_like(du), where=vs > 0), dth / wseg)
    tk = np.r_[0.0, np.cumsum(seg_t)]
    n = max(1, int(np.ceil(tk[-1] / dt - 1e-9)))
    t = np.arange(n + 1) * dt

    k = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, len(du) - 1)
    x = np.clip((t - tk[k]) / np.maximum(seg_t[k], 1e-12), 0.0, 1.0)   # time fraction; zero-length segments -> 0/1
    v0, v1 = v[k], v[k + 1]
    # arc-length fraction under constant acceleration: u(x)/ds = x (2 v0 + (v1 - v0) x) / (v0 + v1); in-place
    # rotations (v0 = v1 = 0) advance linearly in time
    f = np.where(vs[k] > 0, x * (2.0 * v0 + (v1 - v0) * x) / np.where(vs[k] > 0, vs[k], 1.0), x)
    speed = v0 + (v1 - v0) * x
    omega = np.where(du[k] > 0, rate[k] * speed, np.divide(dth[k], seg_t[k], out=np.zeros_like(x), where=seg_t[k] > 0))
    return Timed(t=t,
                 p=(1.0 - f)[:, None] * p[k] + f[:, None] * p[k + 1],
                 R=(rot[k] * Rotation.from_rotvec(f[:, None] * rel[k])).as_matrix(),
                 q=(1.0 - f)[:, None] * q[k] + f[:, None] * q[k + 1],
                 tube=tube0 + tube_k * speed + tube_w * omega,
                 speed=speed)
