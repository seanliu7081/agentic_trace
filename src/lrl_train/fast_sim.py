"""Zero-gap C++ 500 Hz OSC stepper wrapper for MuJoCo 3.2.6 (`_fast_osc.so`).

Wraps `SimRobot` (or `StageSim`) with an identical `io` interface (`step`, `eef_pose`, `wrench`, `finger_opening`,
`q`, `joint_vel`) while executing all 25 physics substeps + 500 Hz Operational Space Control + PandaGripper action
integration directly inside C++ (`libmujoco.so.3.2.6`).

Parity with `robosuite 1.4.1` `OperationalSpaceController(use_delta=False)`:
  - Single 20 Hz control step (25 physics substeps): max |qpos_cpp - qpos_robosuite| ~ 1e-13
  - 40-step contact trajectory (1000 physics substeps): max |qpos_cpp - qpos_robosuite| < 1e-7
  - Also records maximum physical contact penetration (-contact.dist) across every 2 ms physics substep.
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import tempfile
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T

from atrace.geometry import mat_to_rotvec

_DIR = Path(__file__).resolve().parent
_CPP = _DIR / "_fast_osc.cpp"
_SO = _DIR / "_fast_osc.so"
_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    def needs_build():
        return not _SO.exists() or _CPP.stat().st_mtime_ns > _SO.stat().st_mtime_ns

    if needs_build():
        # Linux training workers share this file. Recheck under the lock:
        # one compile per revision, with a fully linked library published at once.
        with _CPP.open("rb") as src:
            fcntl.flock(src, fcntl.LOCK_EX)
            if needs_build():
                mj_pkg = Path(mujoco.__file__).resolve().parent
                fd, built = tempfile.mkstemp(dir=_DIR, suffix=".so")
                os.close(fd)
                try:
                    subprocess.run([
                        "g++", "-O3", "-march=native", "-shared", "-fPIC",
                        f"-I{mj_pkg / 'include'}", str(_CPP), str(mj_pkg / "libmujoco.so.3.2.6"),
                        f"-Wl,-rpath,{mj_pkg}", "-o", built,
                    ], check=True)
                    os.replace(built, _SO)
                finally:
                    Path(built).unlink(missing_ok=True)
    lib = ctypes.CDLL(str(_SO))
    lib.ucs_osc_step.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte),
    ]
    lib.ucs_osc_step.restype = ctypes.c_int
    _LIB = lib
    return lib


class FastSimRobot:
    """Drop-in fast replacement for `atrace.sim_iface.SimRobot` during blind/depth stage simulation and RL training."""

    def __init__(self, sim_robot, substeps: int = 25, split_step: bool = True):
        self.base = sim_robot
        self.m = sim_robot.m
        self.d = sim_robot.d
        self.model = sim_robot.model
        self.substeps = int(substeps)
        self.split_step = int(bool(split_step))
        self._lib = _load_lib()

        ctrl = sim_robot.ctrl
        self._site_id = int(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, ctrl.eef_name))
        self._ft_site_id = int(mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_ft_frame"))
        self._qpos_idx = np.ascontiguousarray(ctrl.qpos_index, dtype=np.int32)
        self._qvel_idx = np.ascontiguousarray(ctrl.qvel_index, dtype=np.int32)
        self._arm_act = np.ascontiguousarray(sim_robot.robot._ref_joint_actuator_indexes, dtype=np.int32)
        self._grip_act = np.ascontiguousarray(
            [sim_robot.env.env.sim.model.actuator_name2id(a) for a in sim_robot.robot.gripper.actuators],
            dtype=np.int32,
        )
        f_name = sim_robot.robot.gripper.important_sensors["force_ee"]
        t_name = sim_robot.robot.gripper.important_sensors["torque_ee"]
        self._f_adr = int(self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, f_name)])
        self._t_adr = int(self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, t_name)])
        self._grip_speed = float(sim_robot.robot.gripper.speed)
        self._joint_kp = 10.0

        # Pre-allocate contiguous C buffers and pointers so step() allocates nothing
        self._p_buf = np.zeros(3, dtype=np.float64)
        self._R_buf = np.zeros((3, 3), dtype=np.float64)
        self._q_init = np.ascontiguousarray(ctrl.initial_joint, dtype=np.float64)
        self._kp = np.ascontiguousarray(ctrl.kp, dtype=np.float64)
        self._kd = np.ascontiguousarray(ctrl.kd, dtype=np.float64)
        ga = np.asarray(sim_robot.robot.gripper.current_action, dtype=np.float64).ravel()
        self._grip_cur = np.ascontiguousarray(
            ga if ga.size == 2 else np.array([-float(ga[0]), float(ga[0])]), dtype=np.float64
        )
        self._pen_buf = np.zeros(1, dtype=np.float64)

        self._p_qpos = self._qpos_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._p_qvel = self._qvel_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._p_aact = self._arm_act.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._p_gact = self._grip_act.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self._p_p = self._p_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_R = self._R_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_qinit = self._q_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_kp = self._kp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_kd = self._kd.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_gcur = self._grip_cur.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self._p_pen = self._pen_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        g = self.model.gripper_geometry(self.q()[:7])
        self._pad_off = float(g["pad_inner_open"] - 0.04)
        self.last_substep_max_pen_m = 0.0
        self._contact_mask = None
        self._mask_ptr = None

    @property
    def sensorless(self):
        return self.base.sensorless

    def set_penetration_pairs(self, forbidden):
        """Filter scene penetration across *every* inner substep. None counts all contacts."""
        if forbidden is not None and np.shape(forbidden) != (self.m.ngeom, self.m.ngeom):
            raise ValueError("Expected a geom-by-geom collision matrix")
        self._contact_mask = None if forbidden is None else np.ascontiguousarray(forbidden, np.uint8)
        self._mask_ptr = None if forbidden is None else self._contact_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    def sync_from_robosuite(self):
        """Call after StageSim.set_arm(), restore(), or randomize() so gains, posture, and gripper state match."""
        ctrl = self.base.ctrl
        self._q_init[:] = ctrl.initial_joint
        self._kp[:] = ctrl.kp
        self._kd[:] = ctrl.kd
        ga = np.asarray(self.base.robot.gripper.current_action, dtype=np.float64).ravel()
        self._grip_cur[:] = ga if ga.size == 2 else np.array([-float(ga[0]), float(ga[0])])

    def sync_to_robosuite(self):
        """Write gripper action & posture back to robosuite before snapshot()."""
        self.base.robot.gripper.current_action = self._grip_cur.copy()
        self.base.posture = self._q_init.copy()

    def q(self):
        return self.d.qpos[self.model.qpos_adr].copy()

    def joint_vel(self):
        return self.d.qvel[self._qvel_idx].copy()

    def eef_pose(self):
        return (
            self.d.site_xmat[self._site_id].reshape(3, 3).copy(),
            self.d.site_xpos[self._site_id].copy(),
        )

    def finger_opening(self):
        q = self.d.qpos[self.model.qpos_adr]
        return float(q[7] - q[8] + 2.0 * self._pad_off)

    def wrench(self):
        if self.sensorless:
            raise RuntimeError("Wrist F/T is disabled for this sensorless robot")
        f = self.d.sensordata[self._f_adr:self._f_adr + 3]
        t = self.d.sensordata[self._t_adr:self._t_adr + 3]
        R = self.d.site_xmat[self._ft_site_id].reshape(3, 3) if self._ft_site_id >= 0 else np.eye(3)
        return R @ f, R @ t

    def step(self, R_goal, p_goal, grip=None, posture=None):
        if grip is not None:
            self.base.grip_cmd = float(np.clip(grip, -1.0, 1.0))
        if posture is not None:
            self._q_init[:] = np.asarray(posture, float)[:7]
            self.base.posture = self._q_init.copy()
        self._p_buf[:] = p_goal
        self._R_buf[:] = T.quat2mat(T.axisangle2quat(mat_to_rotvec(R_goal)))
        rc = self._lib.ucs_osc_step(
            self.m._address, self.d._address, self.substeps, self._site_id,
            self._p_qpos, self._p_qvel, self._p_aact, self._p_gact,
            self._p_p, self._p_R, self._p_qinit, self._p_kp, self._p_kd,
            self._joint_kp, self.base.grip_cmd, self._grip_speed,
            self._p_gcur, self._p_pen, self.split_step, self._mask_ptr,
        )
        if rc != 0:
            raise RuntimeError(f"ucs_osc_step failed with code {rc}")
        self.last_substep_max_pen_m = float(self._pen_buf[0])
        self.base.steps += 1
        self.base.last_goal = (np.asarray(R_goal, float), np.asarray(p_goal, float))
        if self.base.on_step is not None:
            self.base.on_step(self.base, None)

    def hold(self, n=1, grip=None):
        R, p = self.base.last_goal
        for _ in range(n):
            self.step(R, p, grip)

    def capture(self, camera="agentview"):
        return self.base.capture(camera)
