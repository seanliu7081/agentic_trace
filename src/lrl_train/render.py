"""Depth and segmentation of the harness's head and wrist cameras in the stage simulator (LRL_PLAN.md M1E, E2).

StageSim builds its environment without cameras (Tier A). Cameras adds an offscreen EGL context of its own to a loaded
model and renders what atrace.sim_iface.SimRobot.capture renders - the same two cameras, pinhole model, image
orientation, depth conversion and robot pixels (robot geoms and robot sites) - at the student's resolution. One render
pass per camera: mjr_readPixels reads the segmentation colours and the depth buffer of the same frame, where capture
makes two passes (RGB-D, then segmentation). Each pixel carries the body it shows (-1: nothing of the model), from
which student_obs derives the robot mask the harness has and the part mask that stands in for SAM3's.

GPU choice: egl_device(gpu) before the process makes its first GL context (MuJoCo opens the EGL display once per
process, by /dev/nvidia minor, and those are numbered in reverse of nvidia-smi on this machine).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

HEAD, WRIST = "agentview", "robot0_eye_in_hand"     # atrace.sim_iface.HEAD_CAMERA, WRIST_CAMERA

CFG = {
    "cameras": (HEAD, WRIST),      # R  the harness's head and wrist cameras
    "size": 96,                    # T  px, square images (the harness captures 512)
    "max_geom": 2000,              # T  scene buffer (robosuite's context: 1000; LIBERO layouts hold a few hundred)
}
GPU_MINOR = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}   # M  nvidia-smi index -> /dev/nvidia minor (EGL device);
                                                                #    scripts/run_episode.py


def egl_device(gpu):
    """Render on nvidia-smi GPU `gpu`. robosuite asserts at import that MUJOCO_EGL_DEVICE_ID appears in a non-empty
    CUDA_VISIBLE_DEVICES; a process that renders only (no CUDA) gets that device listed after its own."""
    minor = str(GPU_MINOR[int(gpu)])
    os.environ["MUJOCO_EGL_DEVICE_ID"] = minor
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd and minor not in cvd.split(","):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{cvd},{minor}"


@dataclass
class View:
    name: str
    depth: np.ndarray       # (H, W) float32 metric z along the optical axis, row 0 = top (as Frame.depth)
    body: np.ndarray        # (H, W) int32 body shown at the pixel (geom or site owner), -1 where nothing is
    K: np.ndarray           # (3, 3) pinhole intrinsics, pixel centres at i + 0.5 (as Frame.K)
    T_wc: np.ndarray        # (4, 4) camera -> world, OpenCV axes (as Frame.T_wc)


class Cameras:
    """render() -> [View] of the configured cameras at the model's current state (mj_forward done by the caller's
    step). The GL context and the model's MjrContext are made once; reload(m, d) moves to a newly loaded model and
    keeps the GL context."""

    def __init__(self, m, d, size=CFG["size"], cameras=CFG["cameras"], cfg=CFG):
        from mujoco.egl import GLContext          # by module: the process may have imported mujoco without MUJOCO_GL
        self.size, self.names, self.cfg = int(size), tuple(cameras), cfg
        self.gl = GLContext(self.size, self.size)
        self.gl.make_current()
        self.rect = mujoco.MjrRect(0, 0, self.size, self.size)
        self.rgb = np.empty((self.size, self.size, 3), np.uint8)
        self.zbuf = np.empty((self.size, self.size), np.float32)
        self.con = None
        self.reload(m, d)

    def reload(self, m, d):
        self.gl.make_current()
        if self.con is not None:
            self.con.free()
        self.m, self.d = m, d
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), self.size)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), self.size)
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scn = mujoco.MjvScene(m, maxgeom=self.cfg["max_geom"])
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1
        self.opt, self.pert = mujoco.MjvOption(), mujoco.MjvPerturb()   # defaults, as robosuite's context
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in self.names]
        ext = m.stat.extent
        self.near, self.far = m.vis.map.znear * ext, m.vis.map.zfar * ext
        self.K = {}
        for n, cid in zip(self.names, self.ids):
            f = 0.5 * self.size / np.tan(np.deg2rad(float(m.cam_fovy[cid])) / 2)
            self.K[n] = np.array([[f, 0, self.size / 2], [0, f, self.size / 2], [0, 0, 1.0]])
        self.lut = None

    def _lut(self):
        """Scene geom index + 1 (the segmentation colour) -> body. The scene lists the same geoms in the same order
        at every update of one model (default options: no contact or frame decorations), so it is built once."""
        m, scn = self.m, self.scn
        lut = np.full(scn.ngeom + 1, -1, np.int32)
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                lut[g.segid + 1] = m.geom_bodyid[g.objid]
            elif g.objtype == mujoco.mjtObj.mjOBJ_SITE:
                lut[g.segid + 1] = m.site_bodyid[g.objid]
        return lut

    def render(self, names=None, *, data=None):
        self.gl.make_current()
        m, d, out = self.m, self.d if data is None else data, []
        for n, cid in zip(self.names, self.ids):
            if names is not None and n not in names:
                continue
            self.cam.fixedcamid = cid
            mujoco.mjv_updateScene(m, d, self.opt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            if self.lut is None:
                self.lut, self.ngeom = self._lut(), self.scn.ngeom
            assert self.scn.ngeom == self.ngeom, "scene changed: the segmentation table no longer applies"
            mujoco.mjr_render(self.rect, self.scn, self.con)
            mujoco.mjr_readPixels(self.rgb, self.zbuf, self.rect, self.con)
            seg = (self.rgb[::-1, :, 0].astype(np.int32) | (self.rgb[::-1, :, 1].astype(np.int32) << 8)
                   | (self.rgb[::-1, :, 2].astype(np.int32) << 16))
            body = self.lut[np.where(seg > self.ngeom, 0, seg)]
            z = self.zbuf[::-1].astype(np.float64)
            depth = (self.near / (1.0 - z * (1.0 - self.near / self.far))).astype(np.float32)
            T = np.eye(4)
            T[:3, :3] = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            T[:3, 3] = d.cam_xpos[cid]
            out.append(View(n, depth, body, self.K[n], T))
        return out

    def close(self):
        if self.con is not None:
            self.gl.make_current()
            self.con.free()
            self.con = None
        self.gl.free()


def backproject(view, stride=1, mask=None):
    """Pixels (optionally masked) -> world points (N, 3) and their (row, col), as atrace.sim_iface.Frame.backproject
    (same depth window)."""
    H, W = view.depth.shape
    rows, cols = np.mgrid[0:H:stride, 0:W:stride]
    d = view.depth[::stride, ::stride]
    sel = np.isfinite(d) & (d > 0.02) & (d < 5.0)
    if mask is not None:
        sel &= mask[::stride, ::stride]
    r, c, z = rows[sel], cols[sel], d[sel]
    K = view.K
    pc = np.stack([(c + 0.5 - K[0, 2]) / K[0, 0] * z, (r + 0.5 - K[1, 2]) / K[1, 1] * z, z], axis=1)
    return pc @ view.T_wc[:3, :3].T + view.T_wc[:3, 3], np.stack([r, c], axis=1)


__all__ = ["CFG", "Cameras", "View", "egl_device", "backproject", "GPU_MINOR", "HEAD", "WRIST"]
