"""Short-horizon contact prediction for the privileged simulation demonstrator."""
from copy import copy
from itertools import product
from types import SimpleNamespace

import mujoco
import numpy as np

from atrace.gcp import LeakyPLAI
from atrace.geometry import mat_to_rotvec, rotvec_to_mat
from .fast_sim import FastSimRobot


class GripRollout:
    """Compare pad loads on an isolated integration state with the actual OSC."""

    def __init__(self, env):
        robot = env.fast_rb
        base = copy(robot.base)
        base.d = mujoco.MjData(robot.m)
        base.on_step = None
        base.ctrl = copy(base.ctrl)
        base.ctrl.initial_joint = robot._q_init.copy()
        base.ctrl.kp, base.ctrl.kd = robot._kp.copy(), robot._kd.copy()
        # FastSimRobot writes the grip command and controller registers. Both
        # belong to the trial, even when the model and actuator maps are shared.
        gripper = copy(base.robot.gripper)
        gripper.current_action = robot._grip_cur.copy()
        base.robot = SimpleNamespace(gripper=gripper,
                                     _ref_joint_actuator_indexes=base.robot._ref_joint_actuator_indexes)
        self.robot = FastSimRobot(base, robot.substeps, robot.split_step)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(robot.m, self.state_spec))
        self.qadr, _ = env.ss._adr(env.part.joint)

    def drive(self, env, robot, core, params):
        side, preload, compression, lead = params[:4]
        tangential = params[4] if len(params) > 4 else .020
        # Older demonstration snapshots store the tangent load without an arc.
        depth, arc = params[5:] if len(params) > 5 else (0., 0.)
        rotation, _ = robot.eef_pose()
        data = robot.d
        angle = data.qpos[self.qadr] - env.initial_joint
        error = env.contract.target_rot_rad - angle
        contact = data.xpos[env.bid] + data.xmat[env.bid].reshape(3, 3) @ env.hint_local
        jaw = rotation[:, 0] - env.axis * (rotation[:, 0] @ env.axis)
        jaw /= max(1e-9, np.linalg.norm(jaw))
        tangent = np.sign(error) * np.cross(env.axis, -jaw)
        orbit = np.cross(env.axis, contact - env.anchor)
        load = side * (preload*jaw + tangential*tangent)
        if arc:
            # Move the actual pad site, with its axial seating depth, on the
            # shaft's rigid arc. Zero side load is a symmetric pinch.
            site = contact + depth*rotation[:, 2] + load
            target = env.anchor + rotvec_to_mat(env.axis*np.clip(error, -arc, arc)) @ (site-env.anchor)
        else:
            target = contact + load + np.clip(error, -.18, .18)*orbit
        desired = rotvec_to_mat(env.axis*(angle + np.clip(error, -lead, lead))) @ env.contact_R
        width = max(0., robot.finger_opening() - compression)
        return np.clip(np.r_[rotation.T @ (target - core.p_des)/.008,
                             rotation.T @ mat_to_rotvec(desired @ core.R_des.T)/np.deg2rad(4.5),
                             1., width/.04 - 1.], -1., 1.)

    def select(self, env, nominal):
        robot = self.robot
        robot.set_penetration_pairs(env.pairs)
        data = robot.d
        mujoco.mj_getState(env.m, env.d, self.state, self.state_spec)
        angle = env.d.qpos[self.qadr] - env.initial_joint
        error = env.contract.target_rot_rad - angle
        best, selected = -np.inf, None
        opposed = [(*params, .020, 0., 0.) for params in
                   product((1., -1.), (.006, .018), (.005, .016, .035), (.050, .11))]
        # Probe weaker, stronger and more nearly normal pad loading. Each
        # candidate is subject to the same command and penetration limits.
        oblique = [(side, preload, compression, .080, tangential, 0., 0.)
                   for side, compression, (preload, tangential) in
                   product((1., -1.), (.003, .010, .025), ((.012, .009), (.020, .032), (.026, .016)))]
        centered = [(0., 0., compression, lead, 0., depth, arc)
                    for compression, lead, depth, arc in
                    product((.008, .028), (.065, .16), (-.005, .005), (.18, .42))]
        loaded_arc = [(side, .008, .018, .11, .010, depth, .32)
                      for side, depth in product((1., -1.), (-.005, .006))]
        for params in (None, *opposed, *oblique, *centered, *loaded_arc):
            mujoco.mj_setState(env.m, data, self.state, self.state_spec)
            mujoco.mj_forward(env.m, data)
            robot._q_init[:] = env.fast_rb._q_init
            robot._grip_cur[:] = env.fast_rb._grip_cur
            core = LeakyPLAI(env.core.R_des, env.core.p_des, env.core.retention)
            initial_rotation, _ = robot.eef_pose()
            bilateral_steps = 0
            peak_slip = 0.
            penetration = 0.
            for _ in range(12):
                action = nominal if params is None else self.drive(env, robot, core, params)
                rotation, tcp = robot.eef_pose()
                rd, pd, alpha, width = core.command(action, rotation, tcp)
                core.apply_gains(robot, alpha)
                grip = np.clip((width - 2*robot._pad_off)/.04 - 1., -1., 1.)
                robot._grip_cur[:] = grip, -grip
                robot.step(rd, pd, 0.)
                first, second = data.contact.geom[:data.ncon].T
                on_first, on_second = env.grasp_g[first], env.grasp_g[second]
                left = np.any((on_first & env.left_g[second]) | (on_second & env.left_g[first]))
                right = np.any((on_first & env.right_g[second]) | (on_second & env.right_g[first]))
                bilateral_steps += left and right
                wrist_travel = env.axis @ mat_to_rotvec(robot.eef_pose()[0] @ initial_rotation.T)
                shaft_travel = data.qpos[self.qadr] - env.initial_joint - angle
                peak_slip = max(peak_slip, abs(wrist_travel-shaft_travel))
                penetration = max(penetration, robot.last_substep_max_pen_m)
            final = data.qpos[self.qadr] - env.initial_joint
            rotation, tcp = robot.eef_pose()
            shear = env.axis @ mat_to_rotvec(rotation @ env.contact_R.T) - final
            contact = data.xpos[env.bid] + data.xmat[env.bid].reshape(3, 3) @ env.hint_local
            gap = np.linalg.norm(tcp - contact)
            # Prefer measured progress that keeps the opposed contact on the
            # part. Unproductive wrist rotation, contact loss and collisions cost.
            score = (abs(error) - abs(env.contract.target_rot_rad-final)
                     - .8*max(0., abs(shear)-.057) - 5.*max(0., gap-.017)
                     - 35.*max(0., penetration-.001)
                     + .024*bilateral_steps/12. - .75*max(0., peak_slip-.035))
            if score > best:
                best, selected = score, params
        return selected
