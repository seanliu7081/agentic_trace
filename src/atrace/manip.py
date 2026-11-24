"""S2 candidate generators: grasps and placements.

All candidates come from measured point clouds; no object-specific constants.
Gripper geometry (Panda / robosuite, grip-site frame: x = closing axis,
z = approach): pads span z in [0.0034, 0.0194], |y| <= 0.008; pad inner face
at |x| = finger joint + PAD_OFFSET (measured); finger bodies reach back to z = -0.0446; the
palm starts at z ~ -0.045 and spans |x| <= 0.105, |y| <= 0.035.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay, cKDTree

from .geometry import top_down_rotation
from .perception import voxel_down

# Defaults (overwritten at start-up from the robot's collision model via set_gripper_geometry)
PAD_Z = (-0.007, 0.0092)
PAD_Y = 0.008
FINGER_Y = 0.0105
FINGER_ROOT_Z = -0.0445
PALM_Z = -0.031
PALM_X, PALM_Y = 0.10, 0.0315
TIP_Z = 0.0092
MAX_JOINT = 0.0387                   # practical full-open finger joint (m)
FULL_JOINT = 0.04                    # finger joint after a binary open (joint limit; pad gap ~79 mm)
MAX_GAP = 2 * (-0.0005 + MAX_JOINT)  # practical full-open inner pad gap (m), set from the model at start-up
FINGER_T = 0.027                     # finger body thickness along the closing axis (measured at start-up)
FINGER_T_TIP = 0.015                 # ... from the pads down to the tip
PAD_OFFSET = -0.0005                 # pad inner face = finger joint + PAD_OFFSET (measured at start-up)
TIP_CLEAR = 0.0035                   # lowest fingertip sample above the support plane (m)
APPROACH_GAP = 0.010                 # pre-shaped opening: pads at least this far from the slab on each side
APPROACH_GAPS = (0.020, 0.015, 0.010)  # widest first: observed slabs under-estimate bevelled / flared sides


def set_gripper_geometry(g):
    global PAD_Z, PAD_Y, FINGER_Y, FINGER_ROOT_Z, PALM_Z, PALM_X, PALM_Y, TIP_Z, FINGER_T, PAD_OFFSET, MAX_GAP
    global FINGER_T_TIP
    if "finger_x_open" in g:
        FINGER_T = g["finger_x_open"] - g["pad_inner_open"]
    if "finger_x_open_tip" in g:
        FINGER_T_TIP = g["finger_x_open_tip"] - g["pad_inner_open"]
    PAD_OFFSET = g["pad_inner_open"] - 0.04
    MAX_GAP = 2 * (PAD_OFFSET + MAX_JOINT)
    PAD_Z = tuple(g["pad_z"])
    PAD_Y = g["pad_y"]
    FINGER_Y = g["finger_y"] + 0.0015
    FINGER_ROOT_Z = g["finger_root_z"]
    PALM_Z = g["palm_z"]
    PALM_X, PALM_Y = g["palm_x"] + 0.004, g["palm_y"] + 0.004
    TIP_Z = g["tip_z"]


@dataclass
class GraspCand:
    R: np.ndarray
    p: np.ndarray
    width: float
    score: float
    info: dict = field(default_factory=dict)


def _wrap_pi(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def _rx(t):
    c, s_ = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def _ry(t):
    c, s_ = np.cos(t), np.sin(t)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])


def preshape_finger(width, gap=APPROACH_GAP):
    """Finger joint value that leaves `gap` between each pad and a slab of `width`."""
    return float(np.clip(width / 2 + gap - PAD_OFFSET, 0.004, MAX_JOINT))


def grasp_candidates(obj, support_z, cur_yaw=0.0, n_yaw=18, side_clear=0.006, center=None,
                     center_radius=None, depth_options=(0.008, 0.015, 0.028, 0.04, 0.05), max_cands=80,
                     tilts=(0.0, np.deg2rad(20), -np.deg2rad(20), np.deg2rad(35), -np.deg2rad(35)),
                     pitches=(), slab_gap=0.004, xy_step=0.01, near_r=None, voxel=0.004):
    """Antipodal grasps on a fused object point cloud: top-down plus approach tilted about the
    closing axis (lets the wrist and forearm come in from the side of nearby obstacles).

    Approach families: tilt (rotation about the closing axis) and, if given, pitch (rotation about
    the pad-width axis: the approach leans toward the squeeze direction, e.g. onto a bar from above).

    The pads close on a *local slab*: the run of material (within the finger band) that contains the
    grasp centre. Material further out along the closing axis is allowed as long as each finger has a
    free channel next to the slab (finger thickness + clearance), so wall-like parts of larger objects
    (rims, handles, edges) are graspable as well as whole small objects. The fingers approach
    pre-shaped (APPROACH_GAP from the slab), not fully open. Each pad must meet a face that is
    aligned with it (contact patch spread across the pad width), not a corner.

    Returns candidates sorted by score (higher is better).
    """
    pts_all = voxel_down(obj.points, voxel)
    if len(pts_all) < 10:
        return []
    fc = obj.footprint_center() if center is None else np.asarray(center[:2], float)
    xy = voxel_down(np.c_[obj.footprint_xy(), np.zeros(len(obj.footprint_xy()))], xy_step)[:, :2]
    if center_radius is not None:
        xy = xy[np.linalg.norm(xy - fc, axis=1) <= center_radius]
    xy = np.vstack([fc[None], xy])
    yaws = np.arange(n_yaw) * np.pi / n_yaw
    obj_h = obj.top_z - support_z
    pad_c = 0.5 * (PAD_Z[0] + PAD_Z[1])
    tip_corners = np.array([[sx * 0.05, sy * FINGER_Y, TIP_Z] for sx in (-1, 1) for sy in (-1, 1)])
    out = []
    for tilt, pitch in [(t, 0.0) for t in tilts] + [(0.0, pt) for pt in pitches]:
        Rs = np.array([top_down_rotation(y) @ _rx(tilt) @ _ry(pitch) for y in yaws])          # (Y,3,3)
        for cxy in xy:
            # (near_r: only the points within this horizontal distance can meet the fingers and the palm here; the
            # caller's hand checks see the whole object)
            pts = pts_all if near_r is None else pts_all[np.linalg.norm(pts_all[:, :2] - cxy, axis=1) < near_r]
            near = np.linalg.norm(pts[:, :2] - cxy, axis=1) < 0.06
            if not near.any():
                continue
            local_top = float(np.percentile(pts[near, 2], 98))
            for depth in depth_options:
                d_eff = min(depth, max(0.004, obj_h - 0.006))
                g = np.array([cxy[0], cxy[1], local_top - d_eff])
                P = g[None, :] - np.einsum("yij,j->yi", Rs, np.array([0.0, 0.0, pad_c]))   # site positions (Y,3)
                # fingertips must stay above the support: raise the grasp where needed
                tips_z = P[:, None, 2] + np.einsum("yj,kj->yk", Rs[:, 2, :], tip_corners)
                lift_needed = np.maximum(0.0, support_z + TIP_CLEAR - tips_z.min(axis=1))
                P = P + np.c_[np.zeros((len(P), 2)), lift_needed]
                ok_y = lift_needed < 0.03
                D = pts[None, :, :] - P[:, None, :]                                    # (Y,N,3)
                L = np.einsum("ynj,yjk->ynk", D, Rs)                                  # local coords
                lx, ly, lz = L[..., 0], L[..., 1], L[..., 2]
                band = (np.abs(ly) <= FINGER_Y) & (lz >= FINGER_ROOT_Z - 0.004) & (lz <= TIP_Z + 0.002)
                for k in range(len(yaws)):
                    if not ok_y[k]:
                        continue
                    b = band[k]
                    if b.sum() < 4:
                        continue
                    xs = np.sort(lx[k, b])
                    brk = np.nonzero(np.diff(xs) > slab_gap)[0]
                    lo_r = xs[np.r_[0, brk + 1]]
                    hi_r = xs[np.r_[brk, len(xs) - 1]]
                    d0 = np.where((lo_r <= 0) & (hi_r >= 0), 0.0, np.minimum(np.abs(lo_r), np.abs(hi_r)))
                    j = int(np.argmin(d0))
                    if d0[j] > 0.008:
                        continue                         # the grasp centre must be on material
                    xmin, xmax = lo_r[j], hi_r[j]
                    width = xmax - xmin
                    if width > MAX_GAP - 2 * side_clear or width < 0.002:
                        continue
                    # free channels for the pre-shaped fingers on both sides of the slab (finger thickness
                    # near the tip for material at pad height and below, full thickness above)
                    if j > 0 or j < len(lo_r) - 1:
                        lzb = lz[k, b]
                        xbb = lx[k, b]
                        need = np.where(lzb >= PAD_Z[0], FINGER_T_TIP, FINGER_T) + APPROACH_GAP + 0.003
                        left = xbb < xmin - 1e-6
                        right = xbb > xmax + 1e-6
                        if (left & (xmin - xbb < need)).any() or (right & (xbb - xmax < need)).any():
                            continue
                    local = (j > 0) or (j < len(lo_r) - 1)
                    shift = 0.5 * (xmin + xmax)
                    x_c = lx[k] - shift
                    in_slab = (lx[k] >= xmin - 1e-6) & (lx[k] <= xmax + 1e-6)
                    pad = b & in_slab & (lz[k] >= PAD_Z[0] - 0.002) & (np.abs(ly[k]) <= PAD_Y)
                    if pad.sum() < 4:
                        continue
                    xp = x_c[pad]
                    if (xp < -width / 2 + 0.006).sum() < 2 or (xp > width / 2 - 0.006).sum() < 2:
                        continue
                    # the pads, not the finger bodies above them, must make first contact
                    if xp.max() - xp.min() < width - 0.002:
                        continue
                    # contact alignment: each pad must meet a face spread across its width (not a corner)
                    yp = ly[k][pad]
                    cov = []
                    for side in (xp <= -width / 2 + 0.003, xp >= width / 2 - 0.003):
                        bins = np.unique(np.clip(np.floor((yp[side] + PAD_Y) / (PAD_Y / 2)), 0, 3))
                        cov.append((len(bins), int(side.sum())))
                    if any(n_b <= 1 and n_p >= 4 for n_b, n_p in cov):
                        continue                         # a well-observed corner contact: would slip / rotate
                    palm = (np.abs(x_c) <= PALM_X) & (np.abs(ly[k]) <= PALM_Y) & (lz[k] <= PALM_Z + 0.010)
                    if palm.any():
                        continue
                    R = Rs[k]
                    pc = P[k] + R[:, 0] * shift
                    off = np.linalg.norm((g[:2] + R[:2, 0] * shift) - fc)
                    margin = (MAX_GAP - width) / 2 - side_clear
                    dyaw = abs(_wrap_pi(yaws[k] - cur_yaw))
                    contact = min(pad.sum(), 40) / 40.0
                    align = min(c[0] for c in cov)
                    score = (1.0 * min(margin, 0.004) / 0.004 + 0.5 * contact - 40.0 * off
                             - 0.3 * dyaw + 1.5 * min(d_eff, 0.04) / 0.04 - 0.8 * (abs(tilt) + abs(pitch))
                             + 0.25 * align)
                    out.append(GraspCand(R, pc, float(width), float(score),
                                         {"yaw": float(yaws[k]), "tilt": float(tilt), "pitch": float(pitch),
                                          "depth": float(d_eff),
                                          "offset": float(off), "margin": float(margin),
                                          "pad_points": int(pad.sum()), "local": bool(local),
                                          "align": int(align), "finger": preshape_finger(width)}))
    out.sort(key=lambda g: -g.score)
    keep, per_tilt = [], {}
    quota = max(8, max_cands // max(1, len(tilts) + len(pitches)))
    for g in out:
        t = (round(g.info["tilt"], 3), round(g.info["pitch"], 3))
        if per_tilt.get(t, 0) >= quota:
            continue
        if all(np.linalg.norm(g.p - k.p) > 0.006 or abs(_wrap_pi(g.info["yaw"] - k.info["yaw"])) > 0.15
               or abs(g.info["tilt"] - k.info["tilt"]) > 0.1 or abs(g.info["pitch"] - k.info["pitch"]) > 0.1
               for k in keep):
            keep.append(g)
            per_tilt[t] = per_tilt.get(t, 0) + 1
    keep.sort(key=lambda g: -g.score)
    return keep


def up_frame(n):
    """Rotation Q (rows = frame axes) mapping world vectors into a frame where the unit direction n is +z."""
    z = np.asarray(n, float) / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = helper - z * np.dot(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def side_directions(n_ring=8, elevations=(0.0, np.deg2rad(45.0))):
    """Reference directions (from the object toward the hand) for grasps other than from straight above: a ring of
    horizontal ones and a ring at 45 degrees (with the generator's tilts of up to 35 degrees about the closing axis they
    cover every approach from horizontal to the top-down family)."""
    out = []
    for el in elevations:
        for k in range(n_ring):
            a = 2 * np.pi * k / n_ring
            out.append(np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]))
    return out


def side_grasp_candidates(points, name, dirs, cur_R=None, max_per_dir=8,
                          tilts=(0.0, np.deg2rad(25), -np.deg2rad(25))):
    """Grasps against reference directions n other than straight down (CONTACT_STAGE_PLAN 4.5): the cloud is turned so
    that n is up, the top-down generator runs there (its tilts lean the approach about the closing axis), and the
    grasps are turned back (articulate.part_grasps does the same for parts). The table is not modelled here: the
    hand-fit checks of the caller hold the hand above it."""
    from .perception import ObjectModel
    P = voxel_down(np.asarray(points, float), 0.005)
    if len(P) < 10:
        return []
    c = P.mean(0)
    out = []
    for n in dirs:
        Q = up_frame(n)
        loc = (P - c) @ Q.T
        tmp = ObjectModel(-3, name, name, "target", support_z=float(loc[:, 2].min()) - 0.10)
        tmp.set_points(loc)
        if len(tmp.points) < 10:
            continue
        cur_yaw = 0.0
        if cur_R is not None:
            Rl = Q @ np.asarray(cur_R, float)
            cur_yaw = float(np.arctan2(Rl[1, 0], Rl[0, 0]))
        cands = grasp_candidates(tmp, tmp.support_z, cur_yaw=cur_yaw, n_yaw=8, depth_options=(0.01, 0.025),
                                 tilts=tilts, max_cands=12 * max_per_dir, xy_step=0.015, near_r=0.13, voxel=0.006)
        kept = []
        for g in cands:                       # spread over the object (a handle as well as the rim), best first
            if len(kept) >= max_per_dir:
                break
            if any(np.linalg.norm(g.p - k.p) < 0.02 for k in kept):
                continue
            kept.append(g)
            info = dict(g.info)
            info["side"] = [round(float(v), 3) for v in n]
            out.append(GraspCand(Q.T @ g.R, Q.T @ g.p + c, g.width, g.score, info))
    out.sort(key=lambda g: -g.score)
    return out


# --------------------------------------------------------------------- placement
def _hull_mask(cells_xy, grid_xy):
    if len(cells_xy) < 4:
        return np.zeros(len(grid_xy), bool)
    try:
        tri = Delaunay(cells_xy)
    except Exception:
        return np.zeros(len(grid_xy), bool)
    return tri.find_simplex(grid_xy) >= 0


def container_region(dest, cell=0.005, band=0.018):
    """Opening of a container: interior enclosed by its rim (top band of points), plus the floor that the
    container's own walls enclose where they stay below that band (a caddy's walls and dividers slope down to a low
    front, so the rim band alone encloses only its tallest compartment). A wall here is any surface of the container
    rising more than max(2 cm, a quarter of its depth) above the floor.
    Compartments ("compartments", labels over the height-map grid, 0 = wall or outside): the parts of the opening
    that such walls separate (internal dividers need not reach the rim); parts under 20 cells (5 cm2) are dropped.

    The floor is the lowest observed surface inside the opening; every other observed surface of the
    container is a wall. Clearance is the exact xy distance to those wall points.
    """
    pts = dest.points
    rim_z = float(np.percentile(pts[:, 2], 97))
    top = pts[pts[:, 2] > rim_z - band]
    hm = dest.hm
    shape = hm.shape
    ring = np.zeros(shape, bool)
    ij = np.floor((top[:, :2] - dest.lo_xy) / dest.cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    ring[ij[ok, 0], ij[ok, 1]] = True
    ring = ndimage.binary_closing(ring, iterations=2)
    filled = ndimage.binary_fill_holes(ring)
    interior = filled & ~ring
    if interior.sum() < 20:
        # open ring (occlusion): fall back to the convex hull of the footprint minus the ring
        fin = np.isfinite(hm)
        cxy = dest.lo_xy + (np.argwhere(fin) + 0.5) * dest.cell
        gxy_all = dest.lo_xy + (np.argwhere(np.ones(shape, bool)) + 0.5) * dest.cell
        interior = _hull_mask(cxy, gxy_all).reshape(shape) & ~ndimage.binary_dilation(ring, iterations=1)
        filled = interior | ring
    pij = np.floor((pts[:, :2] - dest.lo_xy) / dest.cell).astype(int)
    pok = (pij[:, 0] >= 0) & (pij[:, 1] >= 0) & (pij[:, 0] < shape[0]) & (pij[:, 1] < shape[1])
    in_mask = np.zeros(len(pts), bool)
    in_mask[pok] = interior[pij[pok, 0], pij[pok, 1]]
    inner_pts = pts[in_mask]
    floor_z = float(np.percentile(inner_pts[:, 2], 8)) if len(inner_pts) > 10 else float(np.percentile(pts[:, 2], 5))
    # walls relative to the floor: they enclose compartments whose walls stay below the rim band, and they split the
    # opening into compartments
    wall_h = max(0.02, 0.25 * (rim_z - floor_z))
    tall = np.isfinite(hm) & (hm > floor_z + wall_h)
    tall_c = ndimage.binary_closing(tall, iterations=1)
    interior = interior | (ndimage.binary_fill_holes(tall_c) & ~tall_c)
    comp, n = ndimage.label(interior & ~ndimage.binary_dilation(tall, iterations=1))
    if n:
        sizes = ndimage.sum(np.ones_like(comp), comp, np.arange(1, n + 1))
        comp[np.isin(comp, np.flatnonzero(sizes < 20) + 1)] = 0
    out_ij = np.argwhere(interior)
    gxy = dest.lo_xy + (out_ij + 0.5) * dest.cell
    # every observed surface of the container above its floor is a wall (rim, sides, anything inside)
    walls = np.concatenate([top, pts[pts[:, 2] > floor_z + 0.006]])
    from scipy.spatial import cKDTree
    wall_tree = cKDTree(walls[:, :2])
    # coarse prefilter clearance: distance to the container's upper walls (low floor relief is judged later,
    # against the height the payload would rest at)
    upper = pts[pts[:, 2] > floor_z + 0.5 * (rim_z - floor_z)]
    clr, _ = cKDTree((upper if len(upper) else walls)[:, :2]).query(gxy)
    return {"floor_z": floor_z, "rim_z": rim_z, "grid_xy": gxy, "clearance": clr, "kind": "in",
            "wall_pts": walls, "wall_tree": wall_tree, "interior": interior, "compartments": comp,
            "lo_xy": dest.lo_xy, "cell": dest.cell,
            "all_pts": pts, "all_tree": cKDTree(pts[:, :2]), "_level_trees": {}}


REST_STEP = 0.003          # resting-height quantisation for height-aware wall tests (m)
STEP_OVER = 0.006          # a surface counts as a wall only where it rises this far above the resting height


def _rest_heights(region, C, fp_hull):
    """Height each footprint would rest at: the highest container surface inside the footprint (convex)."""
    pts, tree = region["all_pts"], region["all_tree"]
    r = float(np.linalg.norm(fp_hull, axis=1).max()) + 1e-3
    H = fp_hull
    E = np.roll(H, -1, axis=0) - H
    sgn = np.sign(np.sum(H[:, 0] * np.roll(H[:, 1], -1) - np.roll(H[:, 0], -1) * H[:, 1])) or 1.0
    out = np.full(len(C), region["floor_z"])
    lists = tree.query_ball_point(C, r)
    ci = np.repeat(np.arange(len(C)), [len(l) for l in lists])
    if len(ci) == 0:
        return out
    pi = np.fromiter((j for l in lists for j in l), dtype=np.int64, count=len(ci))
    inside = np.ones(len(ci), bool)
    for e in range(len(H)):                      # convex polygon: on the inner side of every edge
        rel = pts[pi, :2] - C[ci] - H[e]
        inside &= sgn * (E[e, 0] * rel[:, 1] - E[e, 1] * rel[:, 0]) >= -1e-9
    np.maximum.at(out, ci[inside], pts[pi[inside], 2])
    return out


def _level_tree(region, level):
    """2D tree of the container surfaces that rise above a resting level (the walls for that level)."""
    k = int(np.ceil((level - region["floor_z"]) / REST_STEP - 1e-9))
    trees = region["_level_trees"]
    if k not in trees:
        z = region["floor_z"] + k * REST_STEP + STEP_OVER
        w = region["all_pts"][region["all_pts"][:, 2] > z]
        trees[k] = cKDTree(w[:, :2]) if len(w) else None
    return trees[k]


def support_region(dest, point=None, cell=0.005, radius=0.06):
    """Flat top region of a support object (optionally around a pointed location)."""
    hm = dest.hm
    fin = np.isfinite(hm)
    if point is not None:
        ij = np.floor((np.asarray(point[:2]) - dest.lo_xy) / dest.cell).astype(int)
        ij = np.clip(ij, 0, np.array(hm.shape) - 1)
        win = hm[max(0, ij[0] - 3):ij[0] + 4, max(0, ij[1] - 3):ij[1] + 4]
        wv = win[np.isfinite(win)]
        top = float(np.median(wv)) if len(wv) else float(point[2])
    else:
        vals = hm[fin]
        hist, edges = np.histogram(vals, bins=np.arange(vals.min(), vals.max() + 0.006, 0.004))
        k = int(np.argmax(hist))
        top = float(0.5 * (edges[k] + edges[k + 1]))
    flat = fin & (np.abs(hm - top) <= 0.008)
    lab, n = ndimage.label(flat)
    if point is not None and n > 1:
        ij = np.floor((np.asarray(point[:2]) - dest.lo_xy) / dest.cell).astype(int)
        ij = np.clip(ij, 0, np.array(hm.shape) - 1)
        l0 = lab[ij[0], ij[1]]
        if l0 > 0:
            flat = lab == l0
    edge_free = ndimage.distance_transform_edt(flat) * dest.cell
    out_ij = np.argwhere(flat)
    gxy = dest.lo_xy + (out_ij + 0.5) * dest.cell
    clear = edge_free[flat]
    if point is not None:
        keep = np.linalg.norm(gxy - np.asarray(point[:2]), axis=1) <= radius
        gxy, clear = gxy[keep], clear[keep]
    walls = dest.points[dest.points[:, 2] > top + 0.006]
    from scipy.spatial import cKDTree
    wall_tree = cKDTree(walls[:, :2]) if len(walls) else None
    return {"floor_z": top, "rim_z": top, "grid_xy": gxy, "clearance": clear, "kind": "on",
            "wall_pts": walls, "wall_tree": wall_tree}


def placement_candidates(region, payload_xy_local, payload_r, yaw_options, scene_block_xy=None,
                         target_xy=None, need_clear=0.008, edge_ok=False, max_cands=40, step_m=0.005,
                         block_clear=None):
    """Score payload-centre placements inside a region (exact xy clearances, vectorised). need_clear: from the
    region's walls; block_clear (default need_clear): from other things (scene_block_xy)."""
    block_clear = need_clear if block_clear is None else block_clear
    gxy = region["grid_xy"]
    if len(gxy) == 0:
        return []
    from scipy.spatial import cKDTree
    block_tree = cKDTree(scene_block_xy) if scene_block_xy is not None and len(scene_block_xy) else None
    wall_tree = region.get("wall_tree")
    key = np.floor(gxy / step_m).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    C = gxy[first]
    fp0 = np.asarray(payload_xy_local, float)
    if region["kind"] == "in":
        # a centre nearer to the walls than the payload's narrowest extent from its centre (any yaw) cannot hold it
        a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        reach_min = float((fp0 @ np.stack([np.cos(a), np.sin(a)])).max(axis=0).min())
        C = C[region["clearance"][first] >= reach_min - region.get("cell", step_m)]
    if len(C) == 0:
        return []
    cands = []
    for yaw in yaw_options:
        cs, sn = np.cos(yaw), np.sin(yaw)
        fp = fp0 @ np.array([[cs, -sn], [sn, cs]]).T
        P = C[:, None, :] + fp[None, :, :]                 # (C,K,2)
        flat = P.reshape(-1, 2)
        ok = np.ones(len(C), bool)
        if region["kind"] == "in":
            interior, lo_xy, cell = region["interior"], region["lo_xy"], region["cell"]
            ij = np.floor((flat - lo_xy) / cell).astype(int)
            inb = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < interior.shape[0]) & (ij[:, 1] < interior.shape[1])
            ins = np.zeros(len(flat), bool)
            ins[inb] = interior[ij[inb, 0], ij[inb, 1]]
            ok &= ins.reshape(len(C), -1).all(axis=1)
        rest = None
        if region["kind"] == "in" and "all_pts" in region and ok.any():
            # height-aware walls: the payload rests on the highest surface under its footprint; only surfaces
            # rising above that (not low bumps / folds of the floor it rests on) constrain it sideways
            hull = outline(fp, step=0.01)
            rest = np.full(len(C), np.inf)
            rest[ok] = _rest_heights(region, C[ok], hull if len(hull) > 3 else fp)
            ok &= rest <= region["floor_z"] + 0.03
            wall_c = np.zeros(len(C))
            lev = np.ceil((rest - region["floor_z"]) / REST_STEP - 1e-9)
            for k in np.unique(lev[ok]):
                sel = np.nonzero(ok & (lev == k))[0]
                t = _level_tree(region, region["floor_z"] + k * REST_STEP)
                wall_c[sel] = t.query(P[sel].reshape(-1, 2))[0].reshape(len(sel), -1).min(axis=1) \
                    if t is not None else 1.0
        else:
            wall_c = wall_tree.query(flat)[0].reshape(len(C), -1).min(axis=1) if wall_tree is not None else np.ones(len(C))
        block_c = block_tree.query(flat)[0].reshape(len(C), -1).min(axis=1) if block_tree is not None else np.ones(len(C))
        mc = np.minimum(wall_c, block_c)
        ok &= (wall_c >= need_clear) & (block_c >= block_clear)
        for i in np.nonzero(ok)[0]:
            score = 3.0 * min(mc[i], 0.03) / 0.03
            if target_xy is not None:
                score -= 15.0 * float(np.linalg.norm(C[i] - target_xy))
            score -= 0.2 * abs(yaw)
            cands.append({"xy": C[i], "yaw": float(yaw), "clearance": float(mc[i]), "wall_clear": float(wall_c[i]),
                          "block_clear": float(block_c[i]), "score": float(score),
                          "rest_z": float(rest[i]) if rest is not None else float(region["floor_z"])})
    cands.sort(key=lambda t: -t["score"])
    keep = []
    for cnd in cands:
        if all(np.linalg.norm(cnd["xy"] - k["xy"]) > 0.004 or abs(cnd["yaw"] - k["yaw"]) > 0.2 for k in keep):
            keep.append(cnd)
        if len(keep) >= max_cands:
            break
    return keep


def soft_candidates(region, payload_xy_local, yaw_options, hard_block_xy=None, target_xy=None, max_cands=24,
                    step_m=0.005):
    """Placements ranked instead of rejected (placement "v1"): every region cell and yaw is a candidate, whatever
    the destination's own geometry says. Ranked by the share of the footprint over the opening ("inside"), its xy
    clearance from the destination's walls, and nearness to the target. Only things the payload may not touch
    (hard_block_xy) rule a candidate out. The payload is aimed at the floor; contact ends the descent."""
    gxy = region["grid_xy"]
    if len(gxy) == 0:
        return []
    from scipy.spatial import cKDTree
    block_tree = cKDTree(hard_block_xy) if hard_block_xy is not None and len(hard_block_xy) else None
    wall_tree = region.get("wall_tree")
    key = np.floor(gxy / step_m).astype(int)
    _, first = np.unique(key, axis=0, return_index=True)
    C = gxy[first]
    fp0 = np.asarray(payload_xy_local, float)
    interior = region.get("interior") if region["kind"] == "in" else None
    rows = []                                          # (score, cell index, yaw, inside, wall_c, block_c)
    for yaw in yaw_options:
        cs, sn = np.cos(yaw), np.sin(yaw)
        fp = fp0 @ np.array([[cs, -sn], [sn, cs]]).T
        flat = (C[:, None, :] + fp[None, :, :]).reshape(-1, 2)
        inside = np.ones(len(C))
        if interior is not None:
            ij = np.floor((flat - region["lo_xy"]) / region["cell"]).astype(int)
            inb = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < interior.shape[0]) & (ij[:, 1] < interior.shape[1])
            ins = np.zeros(len(flat), bool)
            ins[inb] = interior[ij[inb, 0], ij[inb, 1]]
            inside = ins.reshape(len(C), -1).mean(axis=1)
        wall_c = wall_tree.query(flat)[0].reshape(len(C), -1).min(axis=1) if wall_tree is not None else np.ones(len(C))
        block_c = block_tree.query(flat)[0].reshape(len(C), -1).min(axis=1) if block_tree is not None else np.ones(len(C))
        score = 3.0 * inside + np.minimum(wall_c, 0.03) / 0.03 - 0.2 * abs(yaw)
        if target_xy is not None:
            score = score - 15.0 * np.linalg.norm(C - np.asarray(target_xy, float), axis=1)
        ok = np.nonzero(block_c >= 0.002)[0]
        rows.append(np.c_[score[ok], ok, np.full(len(ok), yaw), inside[ok], wall_c[ok], block_c[ok]])
    rows = np.concatenate(rows) if rows else np.zeros((0, 6))
    rows = rows[np.argsort(-rows[:, 0])][:20 * max_cands]          # the best ones only: the rest cannot be kept anyway
    keep = []
    for score, i, yaw, inside, wall_c, block_c in rows:
        xy = C[int(i)]
        if all(np.linalg.norm(xy - k["xy"]) > 0.008 or abs(yaw - k["yaw"]) > 0.2 for k in keep):
            keep.append({"xy": xy, "yaw": float(yaw), "clearance": float(min(wall_c, block_c)),
                         "wall_clear": float(wall_c), "block_clear": float(block_c), "score": float(score),
                         "rest_z": float(region["floor_z"]), "inside": float(inside), "soft": True})
        if len(keep) >= max_cands:
            break
    return keep


def support_centres(sup, cell=0.005):
    """Candidate payload centres over a support: the cells its points cover from above, holes filled."""
    lo = sup[:, :2].min(0) - 2 * cell
    shape = tuple(np.ceil((sup[:, :2].max(0) + 2 * cell - lo) / cell).astype(int) + 1)
    m = np.zeros(shape, bool)
    ij = np.floor((sup[:, :2] - lo) / cell).astype(int)
    m[ij[:, 0], ij[:, 1]] = True
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    return lo + (np.argwhere(m) + 0.5) * cell


def bottom_profile(pts, cell=0.004):
    """Footprint cells of a payload (xy relative to its footprint centre) with the payload's lowest point in each cell,
    relative to its overall bottom: a mug's handle sits at its own height, not at the base's."""
    ij = np.floor(pts[:, :2] / cell).astype(int)
    key, inv = np.unique(ij, axis=0, return_inverse=True)
    b = np.full(len(key), np.inf)
    np.minimum.at(b, inv.ravel(), pts[:, 2])
    return (key + 0.5) * cell, b - b.min()


def _height_grid(pts, lo, shape, cell):
    g = np.full(shape, -np.inf)
    if pts is None or len(pts) == 0:
        return g
    ij = np.floor((pts[:, :2] - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.maximum.at(g, (ij[ok, 0], ij[ok, 1]), pts[ok, 2])
    return g


def rest_candidates(centres, fp_xy, fp_b, g_xy, support_pts, obstacle_pts, yaws, need_clear, target_xy=None,
                    base_z=None, cell=0.004, max_cands=40, stable_margin=0.005, need_support=None, touch_pts=None,
                    soft=False):
    """'on' placements that may stack (2.5D). The payload rests where its bottom profile first meets the support
    surface (the destination and whatever already lies on it); parts of it that sit higher may overhang lower surfaces
    (a mug's handle over a plate rim). Every surface within need_clear sideways, support or obstacle, must stay below
    the payload's local bottom (STEP_OVER tolerance). Stable: the payload's centroid lies inside the hull of its contact
    cells by stable_margin (16 directions). Score: nearness to the target, small yaw, resting low (stacking only when
    nothing lower fits). need_support (default need_clear): sideways clearance from the support itself; touch_pts:
    things the payload may touch (rest against) but not overlap, no sideways clearance.
    soft (placement "v1"): ranked instead of rejected. The support's own surfaces and the stability test no longer
    rule a candidate out (an unstable rest only lowers its score); obstacles and touch_pts still do."""
    need_support = need_clear if need_support is None else need_support
    R = float(np.linalg.norm(fp_xy, axis=1).max()) + need_clear + 2 * cell
    lo = centres.min(0) - R
    shape = tuple(np.ceil((centres.max(0) + R - lo) / cell).astype(int) + 1)
    S = _height_grid(support_pts, lo, shape, cell)

    def dilate(G, need):
        r = int(np.ceil(need / cell - 1e-9))
        if r <= 0:
            return G
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return ndimage.maximum_filter(G, footprint=(xx ** 2 + yy ** 2) <= r * r + 1e-9, mode="constant", cval=-np.inf)
    Hd = dilate(_height_grid(obstacle_pts, lo, shape, cell), need_clear)
    if not soft:
        Hd = np.maximum(dilate(S, need_support), Hd)
    if touch_pts is not None and len(touch_pts):
        Hd = np.maximum(Hd, _height_grid(touch_pts, lo, shape, cell))
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    U = np.stack([np.cos(a), np.sin(a)], 1)
    base = base_z if base_z is not None else -np.inf
    cands = []
    for yaw in yaws:
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        F, g = fp_xy @ rot.T, rot @ g_xy
        ij = np.floor((centres[:, None, :] + F[None] - lo) / cell).astype(int)
        s = S[ij[..., 0], ij[..., 1]] - fp_b[None]                      # rest height each cell asks for
        zr = s.max(axis=1)
        ok = np.isfinite(zr)
        ok &= ((Hd[ij[..., 0], ij[..., 1]] - fp_b[None]) <= zr[:, None] + STEP_OVER).all(axis=1)
        contact = s >= zr[:, None] - 0.003
        lean = np.zeros(len(centres))                                   # how far the centroid lies outside the hull
        for u in U:                                                     # centroid inside the contact hull
            if not ok.any():
                break
            d = np.where(contact, (F - g) @ u, -np.inf).max(axis=1)
            if soft:
                lean = np.maximum(lean, np.where(np.isfinite(d), stable_margin - d, 1.0))
            else:
                ok &= d >= stable_margin
        for i in np.nonzero(ok)[0]:
            score = 3.0 * min(need_clear, 0.03) / 0.03 - 0.2 * abs(yaw) - 30.0 * max(0.0, zr[i] - base)
            if target_xy is not None:
                score -= 15.0 * float(np.linalg.norm(centres[i] - target_xy))
            score -= 40.0 * float(np.clip(lean[i], 0.0, 0.05))
            cands.append({"xy": centres[i], "yaw": float(yaw), "clearance": float(need_clear), "score": float(score),
                          "rest_z": float(zr[i]), "stacked": bool(zr[i] > base + 0.01)})
            if soft:
                cands[-1].update(soft=True, lean=float(max(lean[i], 0.0)))
    cands.sort(key=lambda t: -t["score"])
    keep = []
    for cnd in cands:
        if all(np.linalg.norm(cnd["xy"] - k["xy"]) > 0.004 or abs(cnd["yaw"] - k["yaw"]) > 0.2 for k in keep):
            keep.append(cnd)
        if len(keep) >= max_cands:
            break
    return keep


def outline(fp_xy, step=0.005):
    """Convex outline of a 2D footprint sampled every `step` (for fast clearance tests)."""
    from scipy.spatial import ConvexHull
    fp_xy = np.asarray(fp_xy, float)
    if len(fp_xy) < 4:
        return fp_xy
    if len(fp_xy) > 50:
        r = np.linalg.norm(fp_xy - np.median(fp_xy, axis=0), axis=1)
        fp_xy = fp_xy[r <= np.percentile(r, 99)]
    try:
        hull = fp_xy[ConvexHull(fp_xy).vertices]
    except Exception:
        return fp_xy
    out = []
    for a_, b_ in zip(hull, np.roll(hull, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b_ - a_) / step)))
        out.extend(a_ + (b_ - a_) * k / n for k in range(n))
    return np.array(out)


def reserve_space(region, cands, payload_xy_local, future_fps, yaw_options, scene_block_xy, need_clear,
                  max_eval=70):
    """Re-rank placements so that objects still to come into the same region also fit.

    Evaluates a spatially diverse subset of feasible first placements (corners included) and
    scores each by the best clearance still available to the next object.
    """
    if not future_fps or not cands:
        return cands
    pts = np.array([c["xy"] for c in cands])
    key = np.floor(pts / 0.01).astype(int)
    best = {}
    for i, k in enumerate(map(tuple, key)):
        if k not in best or cands[i]["clearance"] > cands[best[k]]["clearance"]:
            best[k] = i
    cells = sorted(best.values())
    stride = max(1, int(np.ceil(len(cells) / max_eval)))
    subset = [cands[i] for i in cells[::stride]]
    fp_out = outline(payload_xy_local)
    scored = []
    for c in subset:
        cs, sn = np.cos(c["yaw"]), np.sin(c["yaw"])
        fp = fp_out @ np.array([[cs, -sn], [sn, cs]]).T + c["xy"]
        blocks = fp if scene_block_xy is None else np.concatenate([scene_block_xy, fp])
        worst = 1.0
        for ffp in future_fps:
            fo = outline(ffp)
            r2 = float(np.linalg.norm(fo, axis=1).max())
            nxt = placement_candidates(region, fo, r2, yaw_options[:2], scene_block_xy=blocks,
                                       need_clear=need_clear, max_cands=2, step_m=0.005)
            worst = min(worst, max((n["clearance"] for n in nxt), default=-1.0))
        scored.append((min(c["clearance"], worst), worst, c))
    scored.sort(key=lambda t: -t[0])
    out = []
    for joint, fut, c in scored:
        c = dict(c)
        c["future_clearance"] = fut
        c["score"] = 10.0 * joint + 0.1 * c["score"]
        out.append(c)
    return out
