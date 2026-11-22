"""General articulation skill: operate any part System 1 names (knob, handle, lever, door, drawer ...).

Nothing here knows what the part is. System 1 gives the part, its parent object, the motion type
("rotate" about the part's own axis, "push" or "pull") and optionally a turn direction. System 2:
  1. measures the surface the part is mounted on (robust plane fit to the parent's points around
     the part) -> outward normal n; the rotation axis is n through the part's centre, the push / pull
     direction starts along -n / +n;
  2. finds a grasp on the part with the ordinary antipodal grasp generator, run in a "mount frame"
     where n is vertical (top-down grasps are the special case n = +z);
  3. executes the motion compliantly: rotation about the measured axis with stall detection, or
     translation that re-aims along the displacement the part actually allows (follows drawers and
     hinged doors without knowing which they are), bounded by force, travel and grip checks;
  4. measures how far the part moved and reports it; System 1's VERIFY judges the semantic state.
"""
from __future__ import annotations

import numpy as np

from .geometry import frame_from_axes, mat_to_rotvec, rotvec_to_mat
from .manip import GraspCand, grasp_candidates
from .perception import ObjectModel

MAX_TURN = np.deg2rad(90.0)      # one quarter turn per grasp
TURN_STEP = np.deg2rad(7.5)      # increment when turning until an expected visible effect appears (5-10 deg)
MIN_FOLLOW = np.deg2rad(15.0)    # the part counts as following the hand beyond this
STALL_LAG = np.deg2rad(18.0)     # commanded minus achieved rotation that means "blocked"
MAX_TRAVEL = 0.25                # m, push / pull bound per grasp
STALL_DISP = 0.003               # m over the stall window means "end of travel"
FORCE_LIMIT = 40.0               # N, wrist force change that stops a push / pull


def _plane_ransac(pts, iters=150, tol=0.004, rng=None):
    rng = rng or np.random.default_rng(0)
    best = None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        inl = np.abs((pts - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    p = pts[best] if best is not None and best.sum() >= 10 else pts
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt[2], (best if best is not None else np.ones(len(pts), bool))


def _planes(pts, k=3, tol=0.004):
    """Up to k dominant planes (sequential RANSAC)."""
    out, rest = [], pts
    for _ in range(k):
        if len(rest) < 30:
            break
        c, n, inl = _plane_ransac(rest, tol=tol)
        d = np.abs((rest - c) @ n)
        inl = d < tol
        if inl.sum() < 20:
            break
        out.append((c, n, rest[inl]))
        rest = rest[~inl]
    return out


def mount_faces(part, parent, support_z, k=3):
    """Candidate mounting faces of a part, best first: up to k planes fitted to the parent's surface around the part,
    each scored by how well the part sits on it: the part lies on its outer side (within a few cm) and the face's
    measured surface surrounds the part's projection onto it (a neighbouring face that merely passes close to the
    part, e.g. a cabinet top next to a handle near its edge, scores low). Each entry: outward normal n, a point p0 on
    the face, the axis point (part centre projected onto the face), protrusion (98th percentile height of the part
    above the face), lift (3rd percentile: how far the part's nearest points stand off the face), score."""
    c = part.points.mean(0)
    ext = float(np.linalg.norm(part.points[:, :2] - c[:2], axis=1).max())
    ring = None
    if parent is not None and len(parent.points) > 30:
        d = np.linalg.norm(parent.points - c, axis=1)
        cand = parent.points[d < ext + 0.06]
        if len(cand) > 30:
            cand = cand[part.surface_distance(cand) > 0.008]
        if len(cand) > 30:
            ring = cand
    out = []
    if ring is not None:
        for p0, n, inl in _planes(ring, k=k):
            if np.dot(n, c - p0) < 0:
                n = -n
            h = (part.points - p0) @ n
            on_side = float(np.mean((h > -0.01) & (h < 0.10)))
            cp = c - np.dot(c - p0, n) * n                       # part centre projected onto the face
            u = inl - cp
            u -= np.outer(u @ n, n)
            r = np.linalg.norm(u, axis=1)
            near = r < max(0.05, ext + 0.02)
            if near.sum() < 8:
                continue
            ang = np.arctan2(u[near] @ np.cross(n, u[near][0] / (np.linalg.norm(u[near][0]) + 1e-9)),
                             u[near] @ (u[near][0] / (np.linalg.norm(u[near][0]) + 1e-9)))
            sectors = len(np.unique(np.floor((ang + np.pi) / (np.pi / 4)).astype(int)))   # surround, of 8
            score = on_side * sectors * np.log1p(near.sum())
            out.append({"n": n, "p0": p0, "axis_point": cp, "protrusion": float(np.percentile(h, 98)),
                        "lift": float(np.percentile(h, 3)), "extent": ext, "plane_inliers": int(len(inl)),
                        "score": float(score), "sectors": int(sectors), "on_side": round(on_side, 3),
                        "inl_sample": inl[:: max(1, len(inl) // 150)]})
    out.sort(key=lambda f: -f["score"])            # stable: equal scores keep the fitting order
    return out


def mount_frame(part, parent, support_z):
    """Outward normal of the surface the part is mounted on, the axis point and the protrusion height: the best of
    mount_faces (3 planes), or a level face at the support height when the parent gives none (plane_inliers 0)."""
    faces = mount_faces(part, parent, support_z, k=3)
    if faces:
        f = faces[0]
        return {k_: f[k_] for k_ in ("n", "p0", "axis_point", "protrusion", "extent", "plane_inliers")}
    c = part.points.mean(0)
    ext = float(np.linalg.norm(part.points[:, :2] - c[:2], axis=1).max())
    n = np.array([0.0, 0.0, 1.0])
    p0 = np.array([c[0], c[1], support_z])
    h = (part.points - p0) @ n
    return {"n": n, "p0": p0, "axis_point": c - np.dot(c - p0, n) * n, "protrusion": float(np.percentile(h, 98)),
            "extent": ext, "plane_inliers": 0}


# ---------------------------------------------------------------- measured state of a part that slides / swings
FACE_TOL = 0.005                     # m: a front within this of the surrounding fixture face is flush (closed)
FACING_COS = np.cos(np.deg2rad(30.0))   # surface normals within this of the face normal count as facing outward
CLOSED_TILT = 10.0                   # deg: a moving front tilted more than this against the face is not closed
RING_MIN_PX = 200                    # outward-facing surroundings needed to judge the state (fewer: not measured)


def depth_points_normals(frame, stride=2, jump=0.015):
    """World points, unit surface normals (facing the camera) and their (row, col) from one depth image, by finite
    differences on the pixel grid; robot pixels and depth discontinuities are left out."""
    from scipy import ndimage
    z = frame.depth[::stride, ::stride].astype(float)
    H, W = z.shape
    rr, cc = np.mgrid[0:H, 0:W]
    x = (cc * stride + 0.5 - frame.K[0, 2]) / frame.K[0, 0] * z
    y = (rr * stride + 0.5 - frame.K[1, 2]) / frame.K[1, 1] * z
    P = np.stack([x, y, z], -1)
    valid = np.isfinite(z) & (z > 0.02) & (z < 5.0)
    valid &= ~ndimage.binary_dilation(frame.robot_mask, iterations=3)[::stride, ::stride]
    du = P[1:-1, 2:] - P[1:-1, :-2]
    dv = P[2:, 1:-1] - P[:-2, 1:-1]
    nrm = np.cross(du, dv)
    ln = np.linalg.norm(nrm, axis=-1)
    zc = z[1:-1, 1:-1]
    ok = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
    with np.errstate(invalid="ignore"):
        ok &= (np.abs(du[..., 2]) < jump + 0.02 * zc) & (np.abs(dv[..., 2]) < jump + 0.02 * zc) & (ln > 1e-12)
    nrm = nrm / np.maximum(ln, 1e-12)[..., None]
    Pc = P[1:-1, 1:-1]
    flip = (nrm * Pc).sum(-1) > 0                       # the camera sits at the origin: normals point back at it
    nrm[flip] *= -1.0
    R, t = frame.T_wc[:3, :3], frame.T_wc[:3, 3]
    rc = np.stack([rr[1:-1, 1:-1][ok] * stride, cc[1:-1, 1:-1][ok] * stride], 1)
    return Pc[ok] @ R.T + t, nrm[ok] @ R.T, rc


def articulation_state(obs, part_pts, n, p0, depth_band=0.35, front_band=None, allow_turned=True):
    """State of a part that slides or swings out of a fixture face (drawer, door, flap), measured from fresh views.

    obs: [(points, normals, on_part)] from depth_points_normals plus a part-pixel flag. The frame is the mounting face
    used for the stroke (outward normal n through p0). Nothing here knows what the part is:
      front: the moving front, i.e. the plane of the outward-facing surface over the part's footprint (for a small part
             such as a handle that is the face it sits on, which moves with it);
      face:  the surrounding fixture face, i.e. outward-facing surface in a 10 cm ring around the footprint (part
             pixels excluded); when enough of it lies >= FACE_TOL behind the front, its main level is the face and the
             part stands out by front - face; otherwise the surroundings are flush with (or proud of) the front.
    closed: flush (nothing of the surroundings lies a face-tolerance behind the front) and not tilted; a part whose own
    main plane is turned > 30 deg from the face (a swung door) is open by that angle (allow_turned: only when part_pts
    is a fresh observation of the part). None when the views do not show the front or its surroundings.
    front_band: (lo, hi) heights along n where the front may be (default: the part model's own height range); with
    it, part_pts only gives the part's footprint on the face, which sliding along n does not change."""
    out = {"measured": False, "closed": None, "front_m": None, "face_m": None, "offset_m": None, "angle_deg": None}
    part_pts = np.asarray(part_pts, float)
    if not obs or len(part_pts) < 20:
        out["why"] = "no view or no part model"
        return out
    n = np.asarray(n, float)
    Q = mount_rotation(n)
    P = np.concatenate([o[0] for o in obs])
    Nr = np.concatenate([o[1] for o in obs])
    onp = np.concatenate([o[2] for o in obs]).astype(bool)
    L = (P - p0) @ Q.T
    facing = Nr @ n >= FACING_COS
    pl = (part_pts - p0) @ Q.T
    uc, vc = float(np.median(pl[:, 0])), float(np.median(pl[:, 1]))
    au = max(0.02, 0.5 * float(np.percentile(pl[:, 0], 97) - np.percentile(pl[:, 0], 3)) + 0.015)
    av = max(0.02, 0.5 * float(np.percentile(pl[:, 1], 97) - np.percentile(pl[:, 1], 3)) + 0.015)
    du, dv = np.abs(L[:, 0] - uc), np.abs(L[:, 1] - vc)
    h_lo, h_hi = float(np.percentile(pl[:, 2], 2)) - 0.03, float(np.percentile(pl[:, 2], 99)) + 0.02
    if front_band is not None:
        h_lo, h_hi = float(front_band[0]), float(front_band[1])
    fw = facing & (du <= au) & (dv <= av) & (L[:, 2] > h_lo) & (L[:, 2] < h_hi)
    out["n_front_px"] = int(fw.sum())
    if fw.sum() < 30:
        # no outward-facing surface over the footprint: a part turned away from the face (an open door) shows it in its
        # own main plane; otherwise the view did not show the front
        sub = part_pts[:: max(1, len(part_pts) // 3000)]
        if allow_turned and len(sub) >= 30:
            _, pn, inl = _plane_ransac(sub, tol=0.004)
            tilt = float(np.rad2deg(np.arccos(min(1.0, abs(float(pn @ n))))))
            if inl.mean() >= 0.35 and tilt > 30.0:
                out.update(measured=True, closed=False, angle_deg=round(tilt, 1),
                           why="the part's own main plane is turned away from the face")
                return out
        out["why"] = "the part's front was not seen facing outward"
        return out
    # reference direction: the normal of the large surrounding fixture face (fitted on the ring; the mounting normal
    # can be off by a few degrees, and a plane fitted over the part's footprint alone mixes the front with the frame
    # strips at its border when the two are within a centimetre); levels are histogram peaks along it
    front0 = _peak(L[fw, 2])
    ring = facing & ~onp & (du <= au + 0.10) & (dv <= av + 0.10) & ~((du <= au + 0.01) & (dv <= av + 0.01)) \
        & (L[:, 2] > front0 - depth_band) & (L[:, 2] < front0 + 0.10)
    out["n_ring_px"] = int(ring.sum())
    nref = n
    if ring.sum() >= RING_MIN_PX:
        _, rn, rin = _plane_ransac(P[ring], tol=0.003)
        if rin.mean() >= 0.3 and abs(float(rn @ n)) > np.cos(np.deg2rad(20.0)):
            nref = rn if float(rn @ n) > 0 else -rn
    _, mn, _ = _plane_ransac(P[fw], tol=0.004)
    tilt = float(np.rad2deg(np.arccos(min(1.0, abs(float(mn @ nref))))))
    cw = P[fw].mean(0)
    hW = (P[fw] - cw) @ nref
    f_rel = _peak(hW)                                        # the moving front, relative to the footprint centroid
    front = float((cw + nref * f_rel - p0) @ n)              # its height along n (the stroke frame)
    out.update(front_m=round(front, 4), angle_deg=round(tilt, 1))
    if ring.sum() < RING_MIN_PX:
        out["why"] = "the fixture face around the part was not seen well enough"
        return out
    hR = (P[ring] - cw) @ nref - f_rel                       # the surroundings' heights relative to the front
    behind = hR < -FACE_TOL
    out["behind_share"] = round(float(behind.mean()), 3)
    out["measured"] = True
    if behind.sum() >= max(40, 0.15 * len(hR)):
        face_rel = _peak(hR[behind])
        out.update(face_m=round(front + face_rel, 4), offset_m=round(-face_rel, 4), closed=False)
        return out
    ahead = hR[hR > FACE_TOL]
    if len(ahead) >= max(40, 0.5 * len(hR)) and float(np.median(ahead)) > 0.02:
        # the surroundings stand well in front of the measured front: that is not the part's front (an end stop sits
        # within a centimetre or so behind the face), so the state is not known
        out.update(measured=False, why="the surroundings stand in front of the measured front")
        return out
    face_rel = float(np.median(ahead)) if len(ahead) >= max(40, 0.5 * len(hR)) else 0.0
    out.update(face_m=round(front + face_rel, 4), offset_m=round(min(0.0, -face_rel), 4),
               closed=bool(tilt <= CLOSED_TILT))
    if tilt > CLOSED_TILT:
        out["why"] = "the front is flush at its centre but tilted against the face"
    return out


def _peak(h, bin_=0.002):
    """Main level of a set of heights: the fullest 2 mm bin (smoothed over its neighbours), refined to the median of
    the heights within 3 mm of it."""
    h = np.asarray(h, float)
    if len(h) == 0:
        return 0.0
    lo, hi = float(h.min()), float(h.max())
    if hi - lo < 2 * bin_:
        return float(np.median(h))
    cnt, edges = np.histogram(h, bins=np.arange(lo, hi + bin_, bin_))
    sm = np.convolve(cnt, np.ones(3), mode="same")
    k = int(np.argmax(sm))
    mid = 0.5 * (edges[k] + edges[k + 1])
    sel = np.abs(h - mid) <= 0.003
    return float(np.median(h[sel])) if sel.any() else mid


def mount_rotation(n):
    """Rotation Q (rows = mount axes) mapping world vectors into a frame where n is +z."""
    z = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - z * np.dot(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def part_grasps(part, mf, cur_R, max_depth=None):
    """Antipodal grasps on the part, generated in the mount frame and mapped back to the world. max_depth: only the
    part's points within this depth behind its outer level (along the face normal) are used (fingers and palm reach no
    deeper; a deep part such as an open drawer box otherwise costs minutes of grasp search)."""
    Q = mount_rotation(mf["n"])
    p0 = mf["p0"]
    P = part.points
    if max_depth is not None and len(P) > 30:
        h = (P - p0) @ np.asarray(mf["n"], float)
        P = P[h >= np.percentile(h, 98) - max_depth]
    local = (P - p0) @ Q.T
    tmp = ObjectModel(-2, part.name, part.category, "part", support_z=0.0)
    tmp.set_points(local)
    if len(tmp.points) < 10:
        return []
    R_loc = Q @ cur_R
    cur_yaw = float(np.arctan2(R_loc[1, 0], R_loc[0, 0]))
    ap = (mf["axis_point"] - p0) @ Q.T
    # parts sit on faces of any orientation: also lean the approach along the squeeze direction (pitch),
    # e.g. onto a bar from above-front when the space straight in front of the face is crowded
    cands = grasp_candidates(tmp, 0.0, cur_yaw=cur_yaw, center=ap[:2], max_cands=120,
                             pitches=(np.deg2rad(25), -np.deg2rad(25), np.deg2rad(45), -np.deg2rad(45)))
    out = []
    for g in cands:
        out.append(GraspCand(Q.T @ g.R, Q.T @ g.p + p0, g.width, g.score, dict(g.info)))
    return out


def rot_about(n, a):
    return rotvec_to_mat(np.asarray(n, float) * a)


def achieved_about(R_now, R0, n):
    return float(np.dot(mat_to_rotvec(R_now @ R0.T), n))


def twist_about(h, axis_point, n, angle, rate=np.deg2rad(30), settle=10):
    """Rotate the grasped part about the measured axis; returns the achieved angle (signed)."""
    R0, p0 = h.rb.eef_pose()
    steps = max(8, int(abs(angle) / rate / 0.05))
    achieved = 0.0
    swept = [h.rb.q()[:7]]
    for k in range(steps + settle):
        if k % 4 == 0:
            swept.append(h.rb.q()[:7])
        a = angle * min(1.0, (k + 1) / steps)
        Rr = rot_about(n, a)
        p = axis_point + Rr @ (p0 - axis_point)
        h.rb.step(Rr @ R0, p, grip=1.0)
        Ra, _ = h.rb.eef_pose()
        achieved = achieved_about(Ra, R0, n)
        if abs(a - achieved) > STALL_LAG and k > 3:
            h.log("twist_stall", commanded=np.rad2deg(a), achieved=np.rad2deg(achieved))
            # hold where the part stopped instead of pressing on
            h.rb.step(Ra, h.rb.eef_pose()[1], grip=1.0)
            break
    swept.append(h.rb.q()[:7])
    h.carve_robot(np.array(swept))          # the arm has been there: that space holds nothing else
    return achieved


def turn_order(h, g, q_g, axis_point, n, direction):
    """Directions to try: System 1's if given, otherwise the kinematically roomier one first."""
    if direction in ("counterclockwise", "ccw"):
        return [1.0]
    if direction in ("clockwise", "cw"):
        return [-1.0]
    room = []
    for s in (1.0, -1.0):
        Rr = rot_about(n, s * MAX_TURN)
        p = axis_point + Rr @ (g.p - axis_point)
        q, ok, _, _ = h.rm.ik(Rr @ g.R, p, q_g[None], q_rest=q_g, iters=80)
        lim = h.rm.limits[:7]
        margin = float(np.min(np.minimum(q[0] - lim[:, 0], lim[:, 1] - q[0]))) if ok[0] else -1.0
        room.append((margin, s))
    room.sort(reverse=True)
    return [s for m, s in room if m > 0] or [1.0, -1.0]


def translate(h, n, motion, max_travel=MAX_TRAVEL, grip_check=True):
    """Push / pull with compliance: re-aim along the displacement the mechanism allows. grip_check=False when the
    hand presses the part with closed fingers instead of gripping it (then a closed hand is no lost grip).
    Returns (hand travel, stop reason, final direction); the stop reason is 'stalled' or 'force_limit' when the part
    resisted (an end stop or a block), 'max_travel', or 'lost_grip'."""
    d = -n if motion == "push" else n
    R0, start = h.rb.eef_pose()
    F0, _ = h.rb.wrench()
    hist = [start.copy()]
    swept = [h.rb.q()[:7]]
    why = "max_travel"
    for k in range(int(max_travel / 0.003) + 80):
        R, p = h.rb.eef_pose()
        h.rb.step(R, p + d * 0.015, grip=1.0)        # orientation follows the hand (compliant)
        hist.append(h.rb.eef_pose()[1])
        if k % 4 == 0:
            swept.append(h.rb.q()[:7])
        if len(hist) > 12:
            disp = hist[-1] - hist[-12]
            nd = np.linalg.norm(disp)
            if nd < STALL_DISP:
                why = "stalled"
                break
            d = 0.5 * d + 0.5 * disp / nd
            d /= np.linalg.norm(d)
        if np.linalg.norm(hist[-1] - start) > max_travel:
            break
        if grip_check and h.rb.finger_opening() < 0.002:
            why = "lost_grip"
            break
        F, _ = h.rb.wrench()
        if np.linalg.norm(F - F0) > FORCE_LIMIT:
            why = "force_limit"
            break
    R, p = h.rb.eef_pose()
    for _ in range(3):
        h.rb.step(R, p, grip=1.0)
    swept.append(h.rb.q()[:7])
    h.carve_robot(np.array(swept))          # the arm has been there: that space holds nothing else
    return float(np.linalg.norm(hist[-1] - start)), why, d
