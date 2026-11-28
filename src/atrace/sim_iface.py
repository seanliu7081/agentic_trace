"""Policy-side robot interface: the ONLY object the harness uses to touch the simulator.

Allowed run-time inputs (real-robot equivalents):
  * head (agentview) and wrist (eye-in-hand) RGB-D images with calibration
    (intrinsics from the camera model, extrinsics = fixed head calibration /
    hand-eye calibration through forward kinematics);
  * a robot mask rendered from the robot's own model at its measured joint
    state (only robot/gripper/mount geoms are identified, never scene objects);
  * joint encoders, gripper finger encoders, wrist force/torque sensor.
Commands: absolute end-effector pose (OSC_POSE with control_delta=False),
the OSC null-space posture reference, and the binary gripper command.
Object poses, contacts, task predicates and the collision monitor are never
read here; the runner owns those for scoring only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .geometry import mat_to_rotvec
from .robot_model import RobotModel

HEAD_CAMERA = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"


@dataclass
class Frame:
    name: str
    rgb: np.ndarray          # (H,W,3) uint8, row 0 = top of image
    depth: np.ndarray        # (H,W) float32 metric z-depth along the optical axis
    K: np.ndarray            # (3,3) pinhole intrinsics (pixel centers at i+0.5)
    T_wc: np.ndarray         # (4,4) camera->world, OpenCV axes (x right, y down, z forward)
    robot_mask: np.ndarray   # (H,W) bool
    step: int
    q: np.ndarray            # robot joint state at capture
    meta: dict = field(default_factory=dict)

    def project(self, pts):
        """World points (N,3) -> pixel coords (N,2) float, camera z (N,)."""
        R, t = self.T_wc[:3, :3], self.T_wc[:3, 3]
        pc = (np.asarray(pts, float) - t) @ R
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.K[0, 0] * pc[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * pc[:, 1] / z + self.K[1, 2]
        return np.stack([u, v], axis=1), z

    def backproject(self, mask=None, stride=1):
        """Pixels (optionally masked) -> world points (N,3) and their (row, col)."""
        H, W = self.depth.shape
        rows, cols = np.mgrid[0:H:stride, 0:W:stride]
        sel = np.ones(rows.shape, bool) if mask is None else mask[::stride, ::stride]
        d = self.depth[::stride, ::stride]
        sel = sel & np.isfinite(d) & (d > 0.02) & (d < 5.0)
        r, c, z = rows[sel], cols[sel], d[sel]
        x = (c + 0.5 - self.K[0, 2]) / self.K[0, 0] * z
        y = (r + 0.5 - self.K[1, 2]) / self.K[1, 1] * z
        pc = np.stack([x, y, z], axis=1)
        pw = pc @ self.T_wc[:3, :3].T + self.T_wc[:3, 3]
        return pw, np.stack([r, c], axis=1)

    def pixel_ray_point(self, u, v):
        """3D world point at pixel (u=x col, v=y row) using the depth image (median of a 5x5 patch)."""
        H, W = self.depth.shape
        c, r = int(np.clip(round(u), 0, W - 1)), int(np.clip(round(v), 0, H - 1))
        patch = self.depth[max(0, r - 2):r + 3, max(0, c - 2):c + 3]
        z = float(np.median(patch[np.isfinite(patch)]))
        x = (c + 0.5 - self.K[0, 2]) / self.K[0, 0] * z
        y = (r + 0.5 - self.K[1, 2]) / self.K[1, 1] * z
        return self.T_wc[:3, :3] @ np.array([x, y, z]) + self.T_wc[:3, 3]


class SimRobot:
    """Wraps a loaded Libero10-Collision env; exposes only policy-legal I/O."""

    def __init__(self, env, image_size=512, on_step=None):
        self.env = env
        self.sim = env.sim
        self.m = env.sim.model._model
        self.d = env.sim.data._data
        self.robot = env.robots[0]
        self.ctrl = self.robot.controller
        self.ctrl.use_delta = False          # absolute EEF pose channel (user-approved D1)
        self.model = RobotModel(self.m, self.d)
        self.size = int(image_size)
        self.on_step = on_step
        self.steps = 0
        self.grip_cmd = -1.0
        self.sensorless = False
        body_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(self.m.nbody)]
        robot_bodies = {b for b, n in enumerate(body_names)
                        if n.startswith(("robot0_", "gripper0_", "mount0_"))}
        self._robot_geom = np.array([int(self.m.geom_bodyid[g]) in robot_bodies for g in range(self.m.ngeom)])
        # rendered sites of the robot (e.g. the gripper's site markers) are robot pixels too
        self._robot_site = np.array([int(self.m.site_bodyid[s]) in robot_bodies for s in range(self.m.nsite)], bool)
        self._cam_id = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n)
                        for n in (HEAD_CAMERA, WRIST_CAMERA)}
        ext = self.m.stat.extent
        self._near = self.m.vis.map.znear * ext
        self._far = self.m.vis.map.zfar * ext
        self.posture = self.q()[:7]
        self.last_goal = self.eef_pose()

    # ----------------------------------------------------------- proprioception
    def q(self):
        return self.d.qpos[self.model.qpos_adr].copy()

    def joint_vel(self):
        adr = [self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]
               for n in self.model.joint_names]
        return self.d.qvel[adr].copy()

    def eef_pose(self):
        R, p = self.model.site_pose(self.q()[None])
        return R[0], p[0]

    def finger_opening(self):
        """Inner pad gap (m) from finger encoders (pad face = joint + measured offset; ~0 when closed)."""
        q = self.q()
        if not hasattr(self, "_pad_off"):
            g = self.model.gripper_geometry(q[:7])
            self._pad_off = float(g["pad_inner_open"] - 0.04)
        return float(q[7] - q[8] + 2 * self._pad_off)

    def wrench(self):
        """Wrist F/T (sensor frame) -> world-frame force and torque."""
        if self.sensorless:
            raise RuntimeError("Wrist F/T is disabled for this sensorless robot")
        f = np.array(self.robot.ee_force, dtype=float)
        t = np.array(self.robot.ee_torque, dtype=float)
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_ft_frame")
        R = self.d.site_xmat[sid].reshape(3, 3) if sid >= 0 else np.eye(3)
        return R @ f, R @ t

    # ------------------------------------------------------------------ command
    def step(self, R_goal, p_goal, grip=None, posture=None):
        if grip is not None:
            self.grip_cmd = float(np.clip(grip, -1, 1))
        if posture is not None:
            self.posture = np.asarray(posture, float)[:7]
            self.ctrl.update_initial_joints(self.posture)
        action = np.concatenate([np.asarray(p_goal, float), mat_to_rotvec(R_goal), [self.grip_cmd]])
        self.last_goal = (np.asarray(R_goal, float), np.asarray(p_goal, float))
        obs, _, _, _ = self.env.step(action)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self, obs)
        return obs

    def hold(self, n=1, grip=None):
        R, p = self.last_goal
        for _ in range(n):
            self.step(R, p, grip)

    # ------------------------------------------------------------------ cameras
    def capture(self, camera=HEAD_CAMERA):
        size = self.size
        cid = self._cam_id[camera]
        rgb, depth = self.sim.render(width=size, height=size, camera_name=camera, depth=True)
        seg = self.sim.render(width=size, height=size, camera_name=camera, segmentation=True)
        rgb = np.ascontiguousarray(rgb[::-1])
        depth = depth[::-1].astype(np.float64)
        depth = (self._near / (1.0 - depth * (1.0 - self._near / self._far))).astype(np.float32)
        seg = seg[::-1]
        geom_ids = seg[..., 1]
        is_geom = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        robot_mask = np.zeros(geom_ids.shape, bool)
        valid = is_geom & (geom_ids >= 0)
        robot_mask[valid] = self._robot_geom[geom_ids[valid]]
        site = (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_SITE)) & (geom_ids >= 0) & (geom_ids < len(self._robot_site))
        robot_mask[site] = self._robot_site[geom_ids[site]]
        fovy = float(self.m.cam_fovy[cid])
        f = 0.5 * size / np.tan(np.deg2rad(fovy) / 2)
        K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = self.d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = self.d.cam_xpos[cid]
        return Frame(camera, rgb, depth, K, T, robot_mask, self.steps, self.q())
