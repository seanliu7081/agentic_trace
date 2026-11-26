"""Tier A of LRL_PLAN.md 6.1: the blind stage simulator. A training layout with the Panda, built without cameras
and driven through the harness's own robot interface (atrace.sim_iface.SimRobot on robosuite's OSC at 20 Hz), so a
skill trained here meets at run time the controller, rate and command format it was trained on.

An episode starts with the arm set to an entry pose by inverse kinematics and held for 10 control steps. Joint values
of scene parts and contacts are read here for labels, rewards and scoring; they are never an input of a skill.
"""
from __future__ import annotations

import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("LIBERO_ROOT", str(ROOT / "third_party" / "LIBERO"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/workspace/Libero10-Collision")     # runtime.setup only: it points LIBERO at its files

from atrace.sim_iface import SimRobot  # noqa: E402

from .scenes import FIXTURES, ROBOT, _subtree, place_parts, read_parts, seat_fixture  # noqa: E402

CFG = {
    "control_freq": 20,        # R  Hz, the harness's control rate (robosuite OSC_POSE, 25 physics steps each)
    "hold_steps": 10,          # T  control steps the arm is held at an entry pose (LRL_PLAN.md 6.1)
    "hold_tol": 0.001,         # T  m: a hand that moved further than this during the hold was not at rest
    "limit_slack": 0.05,       # T  rad: an entry posture keeps this far inside every joint limit
    "ik_seeds": (8, 32),       # T  random IK seeds, few first, as atrace.body.make_ik_check
    "finger_open": 0.04,       # R  m: Panda finger joint when open
    "finger_closed": 0.0,      # R  m: finger joint target of a closed hand
}


_STATE = mujoco.mjtState.mjSTATE_INTEGRATION    # everything MuJoCo integrates, solver warm start included


def make_env(bddl_path, seed=0):
    """LIBERO ControlEnv for a layout, without cameras or renderer."""
    from libero10_collision.runtime import setup
    setup()
    from libero.libero.envs.env_wrapper import ControlEnv
    random.seed(seed)           # LIBERO's placement samplers draw from the global generators
    np.random.seed(seed)
    env = ControlEnv(bddl_file_name=str(bddl_path), use_camera_obs=False, has_offscreen_renderer=False,
                     has_renderer=False, control_freq=CFG["control_freq"], horizon=math.inf, ignore_done=True,
                     controller="OSC_POSE", robots=["Panda"])
    env.seed(seed)
    env.reset()
    return env


class Shim:
    """Stands in for the Harness where atrace.articulate.translate and twist_about use it: the robot interface and
    the two bookkeeping calls, which have nothing to record here."""

    def __init__(self, robot):
        self.rb = robot

    def carve_robot(self, qs):
        pass

    def log(self, event, **fields):
        pass


class StageSim:
    def __init__(self, layout):
        self.layout = layout
        self.env = make_env(layout.bddl, layout.seed)
        sim = self.env.env.sim
        for instance, category in layout.fixtures.items():
            if FIXTURES[category].get("seat"):
                seat_fixture(sim.model._model, sim.data._data, instance)
        if layout.mjcf:
            self._insert(layout.mjcf)
        self.robot = SimRobot(self.env)         # after the last reload: every reset makes a new controller
        m, d = self.robot.m, self.robot.d
        self._body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
        self._is_robot = np.array([n.startswith(ROBOT) for n in self._body])
        self.q_home = self.robot.q()[:7]
        self.parts = place_parts(list(layout.parts), m, d)
        for instance, category in layout.fixtures.items():
            self.parts += read_parts(m, d, instance, category)
        self._start = {p.joint: self.joint(p.joint) for p in self.parts}
        # what randomize() scales: the values this layout was loaded with
        self._loaded = (m.dof_damping.copy(), m.dof_frictionloss.copy(), m.body_mass.copy(), m.body_inertia.copy(),
                        m.geom_friction.copy(), self.robot.ctrl.kp.copy())

    def _insert(self, fragments):
        """Add generated bodies through the model XML and reload. The XML keeps the fixtures where LIBERO put them
        but not the joint values, so those are carried over by joint name."""
        sim = self.env.env.sim
        m, d = sim.model._model, sim.data._data
        keep = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j):
                d.qpos[m.jnt_qposadr[j]:(m.jnt_qposadr[j + 1] if j + 1 < m.njnt else m.nq)].copy()
                for j in range(m.njnt)}
        root = ET.fromstring(sim.model.get_xml())
        world = root.find("worldbody")
        for xml in fragments:
            world.append(ET.fromstring(xml))
        self.env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
        sim = self.env.env.sim
        m, d = sim.model._model, sim.data._data
        for name, q in keep.items():
            adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
            d.qpos[adr:adr + len(q)] = q
        d.qvel[:] = 0
        sim.forward()

    # ------------------------------------------------------------------------------------------------ arm
    def set_arm(self, q7, finger=CFG["finger_open"]):
        """Put the arm at q7 at rest with the fingers at `finger` (joint value: 0.04 open, 0 closed), make the
        controller and SimRobot agree with that state, then hold."""
        rb = self.robot
        d = rb.d
        d.qpos[rb.model.qpos_adr] = rb.model.full_q(q7, finger)[0]
        d.qvel[self._robot_dofs()] = 0
        d.qacc_warmstart[:] = 0
        self.env.env.sim.forward()
        # OSC keeps its goal and its null-space posture between steps; this resets both to the present state
        rb.ctrl.update_initial_joints(np.asarray(q7, float))
        # the gripper integrates a finger target in [-1, 1] per finger; +-1 are closed and open
        a = (finger - 0.02) / 0.02
        rb.robot.gripper.current_action = np.array([a, -a])
        rb.grip_cmd = 1.0 if finger <= CFG["finger_closed"] else -1.0 if finger >= CFG["finger_open"] else 0.0
        rb.posture = np.asarray(q7, float).copy()
        rb.last_goal = rb.eef_pose()
        rb.hold(CFG["hold_steps"])

    def _robot_dofs(self):
        m = self.robot.m
        return np.flatnonzero(self._is_robot[m.dof_bodyid])

    def entry(self, R, p, q_seed=None, grip=1.0, finger=None):
        """Start an episode at hand pose (R, p): the IK solution nearest q_seed (default: the home posture) that
        stays limit_slack inside the joint limits. grip > 0: closed fingers; finger: an opening (joint value) instead.
        False when there is none (the arm is then left as it was), or when the scene pushed the hand off the pose
        during the hold: it does not fit there."""
        rb = self.robot
        lim = rb.model.limits[:7]
        if finger is None:
            finger = CFG["finger_closed"] if grip > 0 else CFG["finger_open"]
        q0 = self.q_home if q_seed is None else q_seed
        for n in CFG["ik_seeds"]:
            sols = rb.model.solve_ik(R, p, q0, n_random=n, finger=finger)
            sols = sols[np.minimum(sols - lim[:, 0], lim[:, 1] - sols).min(1) >= CFG["limit_slack"]]
            if len(sols):
                self.set_arm(sols[0], finger)
                p_set = rb.model.site_pose(rb.model.full_q(sols[0], finger))[1][0]
                return bool(np.linalg.norm(rb.eef_pose()[1] - p_set) < CFG["hold_tol"])
        return False

    def start_at(self, R, p, finger=CFG["finger_open"], n_random=8):
        """M1E start: of the IK solutions for hand pose (R, p) inside the joint limits, nearest the home posture
        first, the first whose posture touches nothing (checked when set, before the hold, which costs 10 control
        steps) is held. False when there is none, or when the hold moved the hand or brought a touch."""
        rb = self.robot
        lim = rb.model.limits[:7]
        sols = rb.model.solve_ik(R, p, self.q_home, n_random=n_random, finger=finger)
        for q in sols:
            if np.minimum(q - lim[:, 0], lim[:, 1] - q).min() < CFG["limit_slack"]:
                continue
            rb.d.qpos[rb.model.qpos_adr] = rb.model.full_q(q, finger)[0]
            mujoco.mj_forward(rb.m, rb.d)
            if self._touching():
                continue
            self.set_arm(q, finger)
            p_set = rb.model.site_pose(rb.model.full_q(q, finger))[1][0]
            return bool(np.linalg.norm(rb.eef_pose()[1] - p_set) < CFG["hold_tol"]) and not self._touching()
        return False

    def _touching(self):
        """Any arm, hand or finger body in contact with something (the base and mount excepted)."""
        return any(b.startswith(("robot0_link", "robot0_right_hand", "gripper0_")) for b, _, _ in self.robot_contacts())

    # -------------------------------------------------------------------------------------- scene joints
    def _adr(self, name):
        m = self.robot.m
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])

    def joint(self, name):
        return float(self.robot.d.qpos[self._adr(name)[0]])

    def set_joint(self, name, value):
        """Set a scene joint at rest; part_travel counts from here."""
        qa, va = self._adr(name)
        self.robot.d.qpos[qa], self.robot.d.qvel[va] = value, 0.0
        self.env.env.sim.forward()
        self._start[name] = float(value)
        place_parts(self.parts, self.robot.m, self.robot.d)

    # ------------------------------------------------------------------------------ physics of an episode
    def randomize(self, rng, part, cfg, slip=None, block=None):
        """Physics randomization of LRL_PLAN.md 6.2 for one episode: damping, friction loss and mass of the part,
        friction of the fingers, gain of the controller, each drawn about the value the layout was loaded with
        (so a LIBERO joint is randomized like a generated one). Every joint and geom starts from its loaded value,
        so nothing of an earlier episode (a blocked joint) carries over. Injected failures: slip, a sliding friction
        given to the fingers and the part alike (MuJoCo takes the larger of a pair, so lowering one side alone does
        nothing); block, a friction loss and a damping that hold the joint where it stands. Returns what was drawn, for the critic."""
        m, ctrl = self.robot.m, self.robot.ctrl
        damping, loss, mass, inertia, friction, kp = self._loaded
        dof = self._adr(part.joint)[1]
        own = _subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part.body))
        fingers = np.array(["finger" in n for n in self._body])[m.geom_bodyid]
        f = {k: float(np.exp(rng.uniform(*np.log(cfg[k])))) for k in ("damping", "mass", "finger_friction", "gain")}
        # friction loss follows the mass: a flap on a horizontal hinge is held against gravity by it
        f["loss"] = float(loss[dof] * f["mass"] * rng.uniform(*cfg["loss"])
                          + rng.uniform(0, cfg["loss_add"][part.kind]))
        m.dof_damping[:], m.dof_frictionloss[:] = damping, loss
        m.dof_damping[dof], m.dof_frictionloss[dof] = damping[dof] * f["damping"], f["loss"]
        if block is not None:               # (friction loss, damping): MuJoCo's soft friction alone lets a part creep
            m.dof_frictionloss[dof], m.dof_damping[dof] = block
        m.body_mass[:], m.body_inertia[:] = mass, inertia
        m.body_mass[own] *= f["mass"]
        m.body_inertia[own] *= f["mass"]
        m.geom_friction[:] = friction
        m.geom_friction[fingers, 0] *= f["finger_friction"]
        if slip is not None:
            m.geom_friction[fingers | own[m.geom_bodyid], 0] = slip
        ctrl.kp = kp * f["gain"]
        ctrl.kd = 2 * np.sqrt(ctrl.kp)          # damping ratio 1, as robosuite sets it
        return f | {"damping_value": float(m.dof_damping[dof])}

    # ----------------------------------------------------------------------------- evaluator-side truth
    def part_travel(self, part):
        """Signed joint travel (m or rad) since the part was last set, positive towards part.goal."""
        q0 = self._start[part.joint]
        return (self.joint(part.joint) - q0) * float(np.sign(part.goal - q0))

    def robot_contacts(self):
        """Current contacts between the robot and anything else: (robot body, other body, penetration in m)."""
        m, d = self.robot.m, self.robot.d
        b = m.geom_bodyid[d.contact.geom[:d.ncon]].reshape(-1, 2)
        r = self._is_robot[b]
        sel = r[:, 0] ^ r[:, 1]
        mine, other = np.where(r[:, 0], b[:, 0], b[:, 1])[sel], np.where(r[:, 0], b[:, 1], b[:, 0])[sel]
        return [(self._body[i], self._body[k], float(-x)) for i, k, x in zip(mine, other, d.contact.dist[:d.ncon][sel])]

    # ---------------------------------------------------------------------------------- snapshot, restore
    def snapshot(self):
        """Simulator state plus what lives outside it: the controller's goal and posture, the gripper's finger
        target, SimRobot's bookkeeping and the travel references."""
        rb = self.robot
        state = np.zeros(mujoco.mj_stateSize(rb.m, _STATE))
        mujoco.mj_getState(rb.m, rb.d, state, _STATE)
        return {"state": state, "goal": (rb.ctrl.goal_pos.copy(), rb.ctrl.goal_ori.copy()), "posture": rb.posture.copy(),
                "finger_target": rb.robot.gripper.current_action.copy(), "grip_cmd": rb.grip_cmd,
                "last_goal": (rb.last_goal[0].copy(), rb.last_goal[1].copy()), "steps": rb.steps,
                "start": dict(self._start)}

    def restore(self, s):
        rb, sim = self.robot, self.env.env.sim
        mujoco.mj_setState(rb.m, rb.d, s["state"], _STATE)
        sim.forward()
        rb.ctrl.update_initial_joints(s["posture"])
        rb.ctrl.goal_pos, rb.ctrl.goal_ori = s["goal"][0].copy(), s["goal"][1].copy()
        rb.robot.gripper.current_action = s["finger_target"].copy()
        rb.grip_cmd, rb.posture, rb.steps = s["grip_cmd"], s["posture"].copy(), s["steps"]
        rb.last_goal = (s["last_goal"][0].copy(), s["last_goal"][1].copy())
        self._start = dict(s["start"])
        place_parts(self.parts, rb.m, rb.d)

    def close(self):
        self.env.close()
