"""The student's input (LRL_PLAN.md M1E, E2): what the end-to-end contact policy sees at run time, built from the head
and wrist depth images the harness captures, with the D9 sensor model applied in training.

Observation (OBS_DIM) = lrl_e2e's proprio (PROPRIO_DIM: hand in frame A, tared wrench, finger gap, previous action,
time left, goal) followed by a point cloud in frame A of N_POINTS rows of POINT_DIM numbers:
  live points  every valid, non-robot pixel of the current head and wrist depth images, back-projected with the
               believed camera calibration, cropped to a box around the believed part, evened on a voxel grid and
               sampled to a fixed count with more weight near the believed part;
  part points  the part S1 named: its mask in the hand-off views (the head camera, where S1 points, and the wrist
               camera at the start pose), back-projected once and kept in A for the whole contact. SAM3 costs 93 ms a
               call (NOTES.md step 1), too slow for the control rate. Mask mode "live" instead flags the live points
               through a mask of every frame, which E3 could give only with a tracker.
Per point: x, y, z in A (lrl_e2e's position scale), part flag, wrist-camera flag. Robot pixels are dropped through
the robot mask, which the harness has exactly (rendered from the robot model at its measured joints).

D9 sensor model (SensorModel, one draw per episode): depth noise growing with distance, a depth scale error, holes at
depth edges and scattered dropouts, a calibration error of the head camera (extrinsic) and of the wrist camera
(hand-eye) of a few millimetres and a fraction of a degree, and the part mask as SAM3 gives it from S1's pointer, in
modes measured on the M1 dev episodes: the part without its handle, its handle alone, a region mostly on the fixture
beside it, or the whole fixture; boundaries pixel-accurate.

In the harness (E3) the same functions run on SimRobot frames: depth at 512 px taken every (512 / size)-th pixel with
K scaled accordingly, the robot mask, the SAM3 part mask, and no SensorModel.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from atrace.geometry import rotvec_to_mat
from atrace.lrl_e2e import E2E_CFG, PROPRIO_DIM

from .render import HEAD, WRIST
from .scenes import ROBOT, _subtree, turns

CFG = {
    "size": 96,                  # T  px of the rendered square images (render.Cameras)
    "live_points": 1024,         # T  rows of live points
    "part_points": 256,          # T  rows of part points
    "crop_lo": (-0.45, -0.50, -0.35),   # T  m in A: lrl_e2e's workspace box (box_lo, box_hi) with 5 cm around it
    "crop_hi": (0.65, 0.50, 0.50),
    "voxel": 0.01,               # T  m: grid that evens the cloud's density (the wrist camera samples ~10x denser)
    "near": 0.20,                # T  m: sampling weight 1 / (1 + (r / near)^2), r from A's origin (the believed part)
    "robot_dilate": 1,           # T  px: robot mask grown by this before its pixels are dropped (mask_points: 2 at 512)
    "mask_mode": "start",        # T  "start": part points from the hand-off views; "live": per-frame part flags
    "start_views": (HEAD, WRIST),  # T  views whose part mask makes the part points (head: setup, wrist: start pose)
    # --- D9: depth (assumed RGB-D camera figures, tag T: no real sensor was measured)
    "noise_a0": 0.0005,          # T  m: depth noise std = scale x (a0 + a2 z^2), z in m (stereo-like growth)
    "noise_a2": 0.0025,          # T  m per m^2 (2.5 mm at 1 m, 0.6 mm at 15 cm)
    "noise_scale": (0.5, 2.0),   # T  per-episode factor on the noise, log-uniform
    "depth_scale": 0.002,        # T  std of a per-episode relative depth scale error per camera
    "edge_jump": 0.01,           # T  m: a depth step to a 4-neighbour larger than this makes an edge pixel
    "edge_hole": (0.3, 0.8),     # T  share of edge pixels dropped, per episode
    "dropout": (0.0, 0.02),      # T  share of all pixels dropped, per episode
    # --- D9: calibration, per episode, std per axis (the plan's "a few millimetres and a fraction of a degree")
    "calib_head": (0.003, 0.3),  # T  m, deg: head camera extrinsic error
    "calib_wrist": (0.002, 0.3),  # T  m, deg: wrist camera hand-eye error
    # --- D9: the part mask as SAM3 gives it from S1's pointer, fitted to the 100 part pointings on the head image of the
    #     M1 dev episodes (runs/lrl_m1e_20261125/e2/sam3_masks.py, analyze_sam3.py; NOTES.md step 3). The mode is drawn
    #     once per episode and depends on what S1 sees: an open part is named as such, a closed one by its handle.
    "mask_modes": ("part", "handle", "spill", "fixture"),
    "mask_p_open": (0.96, 0.04, 0.0, 0.0),      # M  part open (46): the part without its handle 44, the handle 2
    "mask_p_closed": (0.0, 0.57, 0.39, 0.04),   # M  part closed (54): the handle 31; a region mostly on the fixture
                                                #    21 (SAM3 there 7, no SAM3 mask and region growing 14); fixture 2
    "closed_share": 0.15,        # T  a part open less than this share of its span counts as closed (teacher start_open)
    "proud": 0.004,              # T  m: part pixels this far out of its face are its handle (scenes handle_tol)
    "mask_sigma": 0.7,           # M  px at 512: boundary error (mean |offset| 1.2 px, 0.5 of it the pixel grid)
    "mask_corr": 5.0,            # M  px at 512: its correlation length along the boundary
    "spill_recall": 0.29,        # M  share of the part a spill mask covers (median of the 7 measured)
    "spill_precision": 0.35,     # M  share of a spill mask that is on the part
}
POINT_DIM = 5
N_POINTS = CFG["live_points"] + CFG["part_points"]
OBS_DIM = PROPRIO_DIM + N_POINTS * POINT_DIM
OTHER, ROBOT_C, PART, FIXTURE = 0, 1, 2, 3          # pixel classes of the truth


def body_classes(m, part=None):
    """Class of every body (OTHER, ROBOT_C, PART, FIXTURE), with one extra entry for "no body" (index -1)."""
    import mujoco
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    cls = np.zeros(m.nbody + 1, np.int8)
    cls[:-1][[n.startswith(ROBOT) for n in names]] = ROBOT_C
    if part is not None:
        own = _subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part.body))
        cls[:-1][np.isin(names, list(part.fixture_bodies)) & ~own] = FIXTURE
        cls[:-1][own] = PART
    return cls


def _smooth_noise(shape, corr, rng):
    """Zero-mean, roughly unit-std random field correlated over `corr` px."""
    g = np.maximum(2, np.ceil(np.array(shape) / max(corr, 1.0)).astype(int) + 1)
    f = ndimage.zoom(rng.normal(size=g), np.array(shape) / g, order=1, mode="nearest")[:shape[0], :shape[1]]
    return f / max(float(f.std()), 1e-6)


def _sample(pts, n, weights, rng):
    """n rows of pts drawn by weight without repetition (Efraimidis-Spirakis keys: one pass, no loop), all of them
    and then repeats when there are fewer."""
    if len(pts) == 0:
        return np.zeros((n, pts.shape[1]), np.float32)
    if len(pts) <= n:
        return pts[np.concatenate([np.arange(len(pts)), rng.integers(len(pts), size=n - len(pts))])]
    key = np.log(rng.random(len(pts))) / weights
    return pts[np.argpartition(-key, n)[:n]]


class SensorModel:
    """The D9 draw of one episode. None of it reaches the harness: there the camera itself is the sensor model."""

    def __init__(self, rng, size, cfg=CFG, closed=False):
        self.cfg, self.s = cfg, size / 512.0
        self.mode = str(rng.choice(cfg["mask_modes"], p=cfg["mask_p_closed" if closed else "mask_p_open"]))
        self.noise = float(np.exp(rng.uniform(*np.log(cfg["noise_scale"]))))
        self.hole, self.drop = rng.uniform(*cfg["edge_hole"]), rng.uniform(*cfg["dropout"])
        self.scale, self.err = {}, {}
        for cam, key in ((HEAD, "calib_head"), (WRIST, "calib_wrist")):
            t, r = cfg[key]
            E = np.eye(4)
            E[:3, :3] = rotvec_to_mat(np.radians(r) * rng.normal(size=3))
            E[:3, 3] = t * rng.normal(size=3)
            self.err[cam] = E                       # believed camera pose = true pose @ E
            self.scale[cam] = 1.0 + cfg["depth_scale"] * rng.normal()

    def holes(self, depth, rng):
        """Pixels the camera returns no depth for: a share of the edge pixels (a step to a 4-neighbour larger than
        edge_jump) and scattered dropouts (one uniform draw serves both)."""
        c, d = self.cfg, depth
        jump = np.zeros(d.shape, bool)
        dv, dh = np.abs(d[1:] - d[:-1]) > c["edge_jump"], np.abs(d[:, 1:] - d[:, :-1]) > c["edge_jump"]
        jump[1:] |= dv
        jump[:-1] |= dv
        jump[:, 1:] |= dh
        jump[:, :-1] |= dh
        u = rng.random(d.shape)
        return (jump & (u < self.hole)) | (u > 1.0 - self.drop)

    def noisy(self, z, cam, rng):
        """Depth readings of the kept pixels: the scale error, then noise growing with distance."""
        c = self.cfg
        z = z * self.scale[cam]
        return z + self.noise * (c["noise_a0"] + c["noise_a2"] * z * z) * rng.standard_normal(len(z))

    def pose(self, view):
        return view.T_wc @ self.err[view.name]

    def mask(self, cls_img, proud, rng):
        """The part mask SAM3 would give for S1's pointer, from the truth classes of a view and the part's proud
        pixels (its handle), in this episode's mode."""
        c, s = self.cfg, self.s
        part = cls_img == PART
        if not part.any():
            return part
        if self.mode == "handle" and proud.any():
            out = proud
        elif self.mode == "fixture":
            out = part | (cls_img == FIXTURE)
        elif self.mode == "spill":
            # the pixels of part and fixture nearest to a point on the part's edge, as many as the measured spill
            # masks hold (recall / precision times the part), so that about a third of the part is covered
            cand = np.flatnonzero(part | (cls_img == FIXTURE))
            edge = np.flatnonzero(part & ~ndimage.binary_erosion(part))
            r0, c0 = np.divmod(edge[rng.integers(len(edge))], part.shape[1])
            rr, cc = np.divmod(cand, part.shape[1])
            k = min(len(cand), int(part.sum() * c["spill_recall"] / c["spill_precision"]))
            out = np.zeros(part.size, bool)
            out[cand[np.argpartition((rr - r0) ** 2 + (cc - c0) ** 2, k - 1)[:k]]] = True
            out = out.reshape(part.shape)
        else:
            out = part & ~proud if (part & ~proud).any() else part
        sd = np.where(out, ndimage.distance_transform_edt(out) - 0.5, 0.5 - ndimage.distance_transform_edt(~out))
        out = sd + s * c["mask_sigma"] * _smooth_noise(out.shape, c["mask_corr"] * s, rng) > 0
        return out & (cls_img != ROBOT_C)          # the harness takes robot pixels out of every SAM3 mask


class StudentObs:
    """episode(m, part, rng) at every new episode (its D9 draw and the truth classes); part_cloud(views) gives the
    world points of the part mask in hand-off views (they make the belief's centroid and the part points);
    start(frame, part_world) fixes the part points in A; obs(views, proprio) is one observation."""

    def __init__(self, cfg=CFG, sensors=True):
        self.cfg, self.sensors = cfg, sensors
        self.box = (np.array(cfg["crop_lo"]), np.array(cfg["crop_hi"]))
        self.part_A = np.zeros((0, POINT_DIM), np.float32)
        self.dims = np.ceil((self.box[1] - self.box[0]) / cfg["voxel"]).astype(np.int64) + 1
        self.grid = np.zeros(int(np.prod(self.dims)), np.int64)    # voxel -> last point written there
        self.rays = {}

    def episode(self, m, part, rng, share=1.0):
        """share: how far the part stands open (share of its span): decides how S1 names it (D9 mask modes)."""
        self.cls, self.rng, self.part = body_classes(m, part), rng, part
        self.sm = SensorModel(rng, self.cfg["size"], self.cfg, share < self.cfg["closed_share"]) \
            if self.sensors else None
        self.part_A = np.zeros((0, POINT_DIM), np.float32)

    def _proud(self, view, cls_img):
        """Pixels of the part standing out of its face (its handle; all of a knob), from the clean depth."""
        part = cls_img == PART
        if turns(self.part) or not part.any():
            return part
        idx = np.flatnonzero(part)
        z = view.depth.ravel()[idx].astype(np.float64)
        pts = np.column_stack([self._rays(view)[idx] * z[:, None], z]) @ view.T_wc[:3, :3].T + view.T_wc[:3, 3]
        out = np.zeros(part.size, bool)
        out[idx] = (pts - self.part.face_point) @ self.part.face_normal > self.cfg["proud"]
        return out.reshape(part.shape)

    def _mask(self, view, cls_img):
        if self.sm is None:
            return cls_img == PART
        return self.sm.mask(cls_img, self._proud(view, cls_img), self.rng)

    def _rays(self, view):
        """Per pixel (x / z, y / z) in the camera frame, cached per camera and intrinsics."""
        H, W = view.depth.shape
        key = (view.name, H, W, float(view.K[0, 0]))
        if key not in self.rays:
            r, c = np.mgrid[0:H, 0:W]
            K = view.K
            self.rays[key] = np.stack([((c + 0.5 - K[0, 2]) / K[0, 0]).ravel(),
                                       ((r + 0.5 - K[1, 2]) / K[1, 1]).ravel()], 1)
        return self.rays[key]

    def _points(self, view, mask=None):
        """World points of the valid, non-robot pixels of a view (within `mask`), back-projected with the believed
        calibration (render.backproject's depth window); their flat pixel indices; the view's pixel classes."""
        cls_img = self.cls[view.body]
        d = view.depth
        keep = (d > 0.02) & (d < 5.0) & ~ndimage.binary_dilation(cls_img == ROBOT_C,
                                                                 iterations=self.cfg["robot_dilate"])
        if mask is not None:
            keep &= mask
        T = view.T_wc
        if self.sm is not None:
            keep &= ~self.sm.holes(d, self.rng)
            T = self.sm.pose(view)
        idx = np.flatnonzero(keep)
        z = d.ravel()[idx].astype(np.float64)
        if self.sm is not None:
            z = self.sm.noisy(z, view.name, self.rng)
        pc = np.column_stack([self._rays(view)[idx] * z[:, None], z])
        return pc @ T[:3, :3].T + T[:3, 3], idx, cls_img

    def part_cloud(self, views):
        """World points of the (degraded) part mask in these views."""
        out = [self._points(v, self._mask(v, self.cls[v.body]))[0] for v in views]
        return np.concatenate(out) if out else np.zeros((0, 3))

    def _even(self, xyz):
        """Indices of one point per occupied voxel of the crop box (the last one written there: no sort)."""
        k = np.floor((xyz - self.box[0]) / self.cfg["voxel"]).astype(np.int64)
        flat = (k[:, 0] * self.dims[1] + k[:, 1]) * self.dims[2] + k[:, 2]
        order = np.arange(len(flat))
        self.grid[flat] = order
        return np.flatnonzero(self.grid[flat] == order)

    def _rows(self, xyz, part, wrist, n):
        c = self.cfg
        keep = np.all((xyz >= self.box[0]) & (xyz <= self.box[1]), axis=1)
        xyz, part, wrist = xyz[keep], part[keep], wrist[keep]
        idx = self._even(xyz)
        xyz, part, wrist = xyz[idx], part[idx], wrist[idx]
        rows = np.column_stack([E2E_CFG["p_scale"] * xyz, part, wrist]).astype(np.float32)
        w = 1.0 / (1.0 + (xyz * xyz).sum(1) / c["near"] ** 2)
        return _sample(rows, n, w, self.rng)

    def start(self, frame, part_world):
        self.part_A = self._rows(frame.point(part_world), np.ones(len(part_world)), np.zeros(len(part_world)),
                                 self.cfg["part_points"])

    def points(self, views, frame):
        xyz, part, wrist = [], [], []
        live = self.cfg["mask_mode"] == "live"
        for v in views:
            pts, idx, cls_img = self._points(v)
            xyz.append(frame.point(pts))
            part.append(self._mask(v, cls_img).ravel()[idx] if live else np.zeros(len(pts)))
            wrist.append(np.full(len(pts), float(v.name == WRIST)))
        rows = self._rows(np.concatenate(xyz), np.concatenate(part), np.concatenate(wrist), self.cfg["live_points"])
        n_part = self.cfg["part_points"]
        part_rows = self.part_A if len(self.part_A) == n_part else np.zeros((n_part, POINT_DIM), np.float32)
        return np.concatenate([rows, part_rows])

    def obs(self, views, proprio, frame):
        return np.concatenate([np.asarray(proprio, np.float32), self.points(views, frame).ravel()])


def split(obs):
    """(proprio (..., PROPRIO_DIM), points (..., N_POINTS, POINT_DIM)) of observations."""
    obs = np.asarray(obs) if not hasattr(obs, "shape") else obs
    return obs[..., :PROPRIO_DIM], obs[..., PROPRIO_DIM:].reshape(*obs.shape[:-1], N_POINTS, POINT_DIM)


__all__ = ["CFG", "POINT_DIM", "N_POINTS", "OBS_DIM", "SensorModel", "StudentObs", "body_classes", "split",
           "OTHER", "ROBOT_C", "PART", "FIXTURE"]
