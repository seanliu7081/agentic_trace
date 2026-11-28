"""Unified Contact Stage (UCS) RL Training Environment (`src/lrl_train/ucs_env.py`).

Trains `UCSExecSkill` (`pi_exec`) across all 5 quasi-static contact regimes in one unified environment:
  - `pull_open`:      prismatic drawers, sliding panels, hinged doors (with cylindrical roll relaxation)
  - `push_close`:     drawers (with open upper neighbours) and hinged doors, handle-first `<1 mm` compliant push
  - `knob_turn`:      centered stove/valve knobs AND offset lever handles (`cw` / `ccw`)
  - `cavity_insert`:  side-opening roofed cavities with door obstacles + physically grasped 6-DoF rigid payload
  - `overhang_place`: open drawers/bins beneath protruding upper overhangs + physically grasped 6-DoF rigid payload

Uses C++ `FastSimRobot` (`split_step=True`, ~50,000+ ctrl_steps/s on CPU pool) with exact MuJoCo 3.2.6 / OSC physics.
Strictly enforces the 11-25 asset rule:
  - NEVER loads the 6 unseen LIBERO fixtures (`short_cabinet`, `short_fridge`, `slide_cabinet`, `window`,
    `faucet`, `basin_faucet`), unseen cavity/bin test topologies, held-out scenes, or test scenes.
"""
from __future__ import annotations

import copy
import os
import pickle
import time
from pathlib import Path

import mujoco
import numpy as np

from atrace import manip
from atrace.geometry import frame_from_axes, rotvec_to_mat
from atrace.ucs import ScrewHypothesis, WholeStrokeSampler, _unit
from atrace.ucs_skill import (
    EXEC_ACT_DIM,
    EXEC_CFG,
    EXEC_MODES,
    EXEC_OBS_DIM,
    ExecGoal,
    UCSExecSkill,
)

from . import scenes, ucs_scenes
from .fast_sim import FastSimRobot
from .sim import StageSim

UCS_TRAIN_KINDS = (
    "drawer_e2e",
    "door_e2e",
    "ucs_lever_knob",
    "sliding_panel",
    "ucs_cavity_box",
    "ucs_overhang_drawer",
    "wooden_cabinet",
    "white_cabinet",
    "microwave",
    "flat_stove",
)

UCS_UNSEEN_KINDS = (
    "short_cabinet",
    "short_fridge",
    "slide_cabinet",
    "window",
    "faucet",
    "basin_faucet",
    "ucs_cavity_unseen_arch",
    "ucs_overhang_unseen_bin",
)

UCS_CRIT_DIM = 22
VAL_SEED = 100000

ENV_CFG = {
    "kinds": UCS_TRAIN_KINDS,
    "seeds": (0, VAL_SEED),
    "pool": 4,
    "starts_per_layout": 20,
    "episodes_per_start": 4,
    "misses": 5,
    "max_loads": 30,
    # Sampling-Based Curriculum (SBC, IndustReal): uniform draw from [s_low, 1.0]; only s_low rises!
    "err_pos_max": 0.024,        # T  m: max hand-off belief position error at full difficulty
    "err_ang_max_deg": 15.0,     # T  deg: max hand-off belief axis/tangent error at full difficulty
    "err_span_max": 0.25,        # T  relative error on believed target span
    # Physics randomization (log-uniform / uniform around loaded values)
    "damping": (0.3, 5.0),
    "mass": (0.35, 4.5),
    "finger_friction": (0.6, 1.6),
    "gain": (0.85, 1.15),
    "loss": (1.0, 1.3),
    "loss_add": {"slide": 3.5, "hinge": 0.25},
    # Sensor noise & 1-step latency randomization
    "wrench_noise_n": 0.8,
    "pos_noise_m": 0.0008,
    # Trajectory dumps for calibrated Outcome Detector D_out
    "dump_share": 0.20,
}

REWARD_CFG = {
    "progress": 2.5,             # T  dense normalized progress toward true joint / rest goal
    "ref_align": 0.015,          # T  per-step shaping toward collision-free posture / upright payload
    "pen_over_1mm": 0.25,        # T  strong penalty whenever non-allowed/handle penetration > 1.0 mm!
    "outside_contact": 0.06,     # T  penalty for arm/hand hitting upper drawer / cavity roof / distractor
    "limit_warn": 0.04,          # T  penalty when any Panda joint margin drops below 0.08 rad
    "action_smooth": 0.004,      # T  mean-square action change penalty
    "step_cost": 0.003,          # T  encourages crisp <6s completion
    "bonus_success": 2.0,        # T  terminal bonus when true benchmark-grade predicate & <=1mm pen met
    "penalty_drop_or_slip": -0.8,# T  terminal penalty on lost grip or dropped payload
}


def _perp_unit(v: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = _unit(v)
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(v, a))
    w = np.cross(v, u)
    th = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.cos(th) * u + np.sin(th) * w


def build_ucs_layout(rng: np.random.Generator, kind: str, out_dir: Path, seed_int: int):
    """Construct `scenes.Layout` + optional `ContainerTarget` for any training or unseen kind."""
    assert kind not in UCS_UNSEEN_KINDS or seed_int >= VAL_SEED, f"Unseen kind {kind} forbidden in training seeds!"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if kind == "ucs_lever_knob":
        xml, parts = ucs_scenes.ucs_lever_knob(rng)
        return scenes.Layout(scenes.base_layout(out_dir), [xml], parts, seed_int, kind), None
    if kind in ("ucs_cavity_box", "ucs_cavity_unseen_arch"):
        xmls, tgt = ucs_scenes.ucs_cavity_box(rng, unseen_arch=(kind == "ucs_cavity_unseen_arch"))
        return scenes.Layout(scenes.base_layout(out_dir), xmls, [], seed_int, kind), tgt
    if kind in ("ucs_overhang_drawer", "ucs_overhang_unseen_bin"):
        xmls, tgt = ucs_scenes.ucs_overhang_drawer(rng, unseen_bin=(kind == "ucs_overhang_unseen_bin"))
        return scenes.Layout(scenes.base_layout(out_dir), xmls, [], seed_int, kind), tgt
    return scenes.sample_layout(rng, kind, out_dir, cfg=scenes.E2E if hasattr(scenes, "E2E") else scenes.CFG), None


class UCSExecEnv:
    """Single-worker environment conforming to `lrl_train.vec.VecEnv` protocol."""

    def __init__(self, seed, out_dir, cfg=ENV_CFG, reward=REWARD_CFG, randomize=True, dump_dir=None):
        self.rng = np.random.default_rng(seed)
        self.out_dir = Path(out_dir)
        self.cfg = dict(cfg)
        self.rw = dict(reward)
        self.randomize = bool(randomize)
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        if self.dump_dir is not None:
            self.dump_dir.mkdir(parents=True, exist_ok=True)

        self.pool = []               # list of [StageSim, FastSimRobot, home_snap, misses, kind, container_tgt]
        self.loads = 0
        self.since_load = 0
        self.replays = 0
        self.weights = None
        self.saved_start = None
        self.dump_buf = []
        self.dump_idx = 0

    def _pick_kind(self) -> str:
        kinds = self.cfg["kinds"]
        if self.weights:
            p = np.array([self.weights.get(k, 1.0) for k in kinds], dtype=float)
            p /= p.sum()
            return str(self.rng.choice(kinds, p=p))
        return str(self.rng.choice(kinds))

    def _load_slot(self):
        kind = self._pick_kind()
        lo, hi = self.cfg["seeds"]
        seed_int = int(self.rng.integers(lo, hi))
        r_lay = np.random.default_rng([abs(hash(kind)) % 10000, seed_int])
        lay, ctgt = build_ucs_layout(r_lay, kind, self.out_dir / "bddl", seed_int)
        ss = StageSim(lay)
        fast_rb = FastSimRobot(ss.robot, split_step=True)
        fast_rb.sync_from_robosuite()
        if ctgt is not None:
            ucs_scenes.bind_container_world(ctgt, ss.robot.m, ss.robot.d)
        home = ss.snapshot()
        self.pool.append([ss, fast_rb, home, 0, kind, ctgt])
        self.loads += 1
        self.since_load = 0
        if len(self.pool) > self.cfg["pool"]:
            old = self.pool.pop(0)
            old[0].close()

    def _fast_set_arm(self, ss: StageSim, fast_rb: FastSimRobot, q7: np.ndarray, finger: float) -> bool:
        rb = ss.robot
        d = rb.d
        d.qpos[rb.model.qpos_adr] = rb.model.full_q(q7, finger)[0]
        d.qvel[ss._robot_dofs()] = 0.0
        d.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(rb.m, d)
        rb.ctrl.update_initial_joints(np.asarray(q7, float))
        a = (finger - 0.02) / 0.02
        rb.robot.gripper.current_action = np.array([-a, a], dtype=float)
        rb.grip_cmd = 1.0 if finger <= 0.002 else -1.0
        rb.posture = np.asarray(q7, float).copy()
        rb.last_goal = fast_rb.eef_pose()
        fast_rb.sync_from_robosuite()
        R_s, p_s = fast_rb.eef_pose()
        for _ in range(8):
            fast_rb.step(R_s, p_s, rb.grip_cmd)
        fast_rb.sync_to_robosuite()
        return float(np.linalg.norm(fast_rb.eef_pose()[1] - p_s)) < 0.002

    def reset(self, arg=1.0):
        s_low, weights = arg if isinstance(arg, tuple) else (float(arg), None)
        self.weights = weights or self.weights
        # Sampling-Based Curriculum: always draw in [max(0, s_low), 1.0]!
        scale = float(self.rng.uniform(np.clip(s_low, 0.0, 1.0), 1.0)) if s_low >= 0 else float(self.rng.uniform(0.0, 1.0))

        if self.replays > 0 and self.saved_start is not None:
            self.replays -= 1
            self._restore_start(scale)
            return self.obs_cur, self._critic(), dict(self.info)

        while True:
            if not self.pool or self.since_load >= self.cfg["starts_per_layout"]:
                self._load_slot()
            slot = self.pool[int(self.rng.integers(len(self.pool)))]
            if self._init_episode(slot, scale):
                slot[3] = 0
                self.since_load += 1
                self.replays = self.cfg["episodes_per_start"] - 1
                return self.obs_cur, self._critic(), dict(self.info)
            slot[3] += 1
            if slot[3] >= self.cfg["misses"]:
                slot[0].close()
                self.pool.remove(slot)

    def _init_episode(self, slot, scale: float) -> bool:
        ss, fast_rb, home, _, kind, ctgt = slot
        ss.restore(home)
        ss.robot.ctrl.kp[:] = 150.0
        ss.robot.ctrl.kd[:] = 2.0 * np.sqrt(150.0)
        fast_rb.sync_from_robosuite()

        self.ss, self.fast_rb, self.kind, self.ctgt, self.scale = ss, fast_rb, kind, ctgt, scale
        self.skill = UCSExecSkill(ss.robot.model.limits[:7])

        if ctgt is not None:
            ok = self._setup_container_episode(ss, fast_rb, ctgt, scale)
        else:
            ok = self._setup_part_episode(ss, fast_rb, scale)
        if not ok:
            return False

        # Save snapshot right BEFORE skill.start() so replays can resample belief noise cleanly!
        fast_rb.sync_to_robosuite()
        self.saved_start = {
            "snap": ss.snapshot(),
            "mode": self.mode,
            "true_tangent": self.true_tangent.copy(),
            "true_axis": self.true_axis.copy(),
            "true_anchor": self.true_anchor.copy(),
            "true_span": self.true_span,
            "approach": self.approach.copy(),
            "way_in": self.way_in,
            "part": self.part,
        }
        return self._start_skill_with_belief(scale)

    def _setup_part_episode(self, ss: StageSim, fast_rb: FastSimRobot, scale: float, force_mode: str | None = None) -> bool:
        rng = self.rng
        part = ss.parts[int(rng.integers(len(ss.parts)))]
        self.part = part
        is_rot = scenes.turns(part)

        if force_mode is not None:
            if (force_mode == "knob_turn") != is_rot:
                return False

        if self.randomize:
            ss.randomize(rng, part, self.cfg)

        span_full = abs(part.open - part.closed)
        s_open = float(np.sign(part.open - part.closed))

        if is_rot:
            mode = "knob_turn"
            # Start near one end, turn >= 0.55 - 1.25 rad toward the other
            go_open = bool(rng.random() < 0.65)
            q_init = part.closed + s_open * span_full * float(rng.uniform(0.0, 0.15) if go_open else rng.uniform(0.65, 0.95))
            ss.set_joint(part.joint, q_init)
            dir_s = s_open if go_open else -s_open
            t_span = min(float(rng.uniform(0.60, 1.25)), abs((part.open if go_open else part.closed) - q_init))
            if t_span < 0.35:
                return False
            screw = ScrewHypothesis("revolute", part.axis, part.axis_point, t_span, dir_s)
        else:
            mode = force_mode if force_mode in ("pull_open", "push_close") else str(rng.choice(["pull_open", "push_close"]))
            if mode == "pull_open" and part.handle_point is None:
                if force_mode == "pull_open":
                    return False
                # Flush handleless panels cannot be pinched from the front; close them or skip
                mode = "push_close"
            # Set neighbour drawers open 35% of the time on push_close (the exact t03 overhang hazard!)
            for o in ss.parts:
                if o is not part:
                    so = float(np.sign(o.open - o.closed))
                    sp = abs(o.open - o.closed)
                    val = o.closed + so * sp * (float(rng.uniform(0.35, 0.90)) if (mode == "push_close" and rng.random() < 0.35) else 0.0)
                    ss.set_joint(o.joint, val)

            if mode == "pull_open":
                q_init = part.closed + s_open * span_full * float(rng.uniform(0.0, 0.12))
                ss.set_joint(part.joint, q_init)
                frac_goal = 0.72 if part.kind == "hinge" else 0.88
                t_span = abs(part.closed + s_open * frac_goal * span_full - q_init)
                screw = ScrewHypothesis("prismatic" if part.kind == "slide" else "revolute", part.axis, part.axis_point, t_span, s_open)
            else:
                q_init = part.closed + s_open * span_full * float(rng.uniform(0.35, 0.95))
                ss.set_joint(part.joint, q_init)
                t_span = abs(q_init - part.closed)
                screw = ScrewHypothesis("prismatic" if part.kind == "slide" else "revolute", part.axis, part.axis_point, t_span, -s_open)

        # Resolve handle / pinch geometry (for push_close, press outer surface of handle or panel with closed fingertips!)
        hp = part.handle_point if part.handle_point is not None else part.face_point
        ha = part.handle_axis if part.handle_axis is not None else part.face_u
        hw = 2.0 * (part.handle_half[1] if part.handle_point is not None else min(part.half_extents))
        if mode == "push_close":
            # Closed fingertips sit 3 mm outside the handle/panel outer surface so way-in seats cleanly!
            hp = hp + part.face_normal * (0.008 if part.handle_point is not None else 0.004)
            stand_behind = 0.002
        else:
            stand_behind = max(0.016, float((hp - part.face_point) @ part.face_normal)) if part.handle_point is not None else 0.025

        m, d2 = ss.robot.m, mujoco.MjData(ss.robot.m)
        own_sub = scenes._subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part.body))
        finger_mask = np.array(["finger" in n for n in ss._body])
        touchy = np.array([n.startswith(("robot0_link", "robot0_right_hand", "gripper0_")) for n in ss._body])

        def col_fn(q7, finger, _step_i, s_frac):
            d2.qpos[:] = ss.robot.d.qpos
            d2.qpos[ss.robot.model.qpos_adr] = ss.robot.model.full_q(q7, finger)[0]
            if s_frac >= 0.0:
                d2.qpos[ss._adr(part.joint)[0]] = q_init + screw.sign * screw.span * s_frac
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_collision(m, d2)
            if d2.ncon == 0:
                return True
            b = m.geom_bodyid[d2.contact.geom[:d2.ncon]].reshape(-1, 2)
            r = ss._is_robot[b]
            sel = r[:, 0] ^ r[:, 1]
            mine = np.where(r[:, 0], b[:, 0], b[:, 1])[sel]
            oth = np.where(r[:, 0], b[:, 1], b[:, 0])[sel]
            bad = touchy[mine]
            if s_frac >= 0.0:
                bad &= ~(own_sub[oth] & finger_mask[mine])
            return not bool(bad.any())

        sampler = WholeStrokeSampler(ss.robot.model, collision_fn=col_fn)
        f_init = 0.0 if mode == "push_close" else None
        cands = sampler.sample_handle_or_knob(
            mode=mode,
            handle_center=hp,
            handle_axis=ha,
            handle_width=hw,
            face_normal=part.face_normal,
            screw=screw,
            q_ref=ss.q_home,
            standoff_behind=stand_behind,
            max_found=1,
        )
        if not cands:
            return False

        c0 = cands[0]
        finger_start = 0.0 if mode == "push_close" else c0.finger_open
        if not self._fast_set_arm(ss, fast_rb, c0.q_handoff, finger_start):
            return False

        self.mode = mode
        self.q_joint_start = q_init
        self.q_joint_goal = q_init + screw.sign * screw.span
        self.true_span = float(screw.span)
        self.true_tangent = screw.tangent_at(c0.p_contact)
        self.true_axis = screw.axis * screw.sign
        self.true_anchor = screw.anchor.copy()
        self.approach = c0.approach.copy()
        self.way_in = float(np.linalg.norm(c0.p_contact - c0.p_handoff))
        self.own_mask = own_sub
        self.fix_mask = np.array([n in set(part.fixture_bodies) for n in ss._body])
        for idx_f in np.flatnonzero(self.fix_mask):
            self.fix_mask |= scenes._subtree(m, int(idx_f))
        self.fix_mask &= ~self.own_mask
        return True

    def _setup_container_episode(self, ss: StageSim, fast_rb: FastSimRobot, ctgt: ucs_scenes.ContainerTarget, scale: float) -> bool:
        rng = self.rng
        self.part = None
        ucs_scenes.bind_container_world(ctgt, ss.robot.m, ss.robot.d)
        mode = ctgt.kind   # "cavity_insert" | "overhang_place"

        # Compute rest site for hand holding `ucs_payload` (held near its upper half, side/oblique pitch 35-55 deg!)
        out_n = ctgt.open_normal_world
        in_dir = -out_n
        up = np.array([0.0, 0.0, 1.0])
        side_y = _unit(np.cross(up, in_dir))

        # Randomize goal spot slightly inside floor footprint
        dy = float(rng.uniform(-0.35, 0.35) * ctgt.floor_half_xy[1])
        dx = float(rng.uniform(-0.20, 0.20) * ctgt.floor_half_xy[0])
        p_obj_rest = ctgt.floor_center_world + in_dir * dx + side_y * dy + up * (ctgt.payload_half_h + 0.003)

        # Grasp pose on payload:
        # For side-opening roofed cavities (`cavity_insert`), hold from the OUTER SIDE at shallow pitch (16-28 deg)
        # so the Panda palm & wrist stay behind the payload and never hit a 15 cm microwave/cavity ceiling!
        if mode == "cavity_insert":
            grasp_off_obj = out_n * (0.35 * ctgt.payload_half_r) + np.array([0.0, 0.0, 0.25 * ctgt.payload_half_h])
            pitches = (16.0, 28.0, 40.0)
        else:
            grasp_off_obj = np.array([0.0, 0.0, 0.55 * ctgt.payload_half_h])
            pitches = (56.0, 42.0)

        p_site_rest = p_obj_rest + grasp_off_obj
        R_cands = []
        for pt_deg in pitches:
            pt = np.radians(pt_deg)
            a_g = _unit(np.cos(pt) * in_dir - np.sin(pt) * up)
            R_cands.extend([frame_from_axes(a_g, side_y), frame_from_axes(a_g, -side_y)])

        slide_len = 0.14 if mode == "cavity_insert" else 0.14
        lift_clear = 0.018 if mode == "cavity_insert" else 0.056
        f_held = float(np.clip(ctgt.payload_half_r + 0.008, 0.015, 0.038))

        # Check robot-vs-container collisions along the entire insertion corridor!
        m, d2 = ss.robot.m, mujoco.MjData(ss.robot.m)
        pay_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ctgt.payload_body)
        own_pay = scenes._subtree(m, pay_bid)
        touchy = np.array([n.startswith(("robot0_link", "robot0_right_hand", "gripper0_")) for n in ss._body])

        def col_ins(q7, finger, _si, _sf):
            d2.qpos[:] = ss.robot.d.qpos
            d2.qpos[ss.robot.model.qpos_adr] = ss.robot.model.full_q(q7, finger)[0]
            mujoco.mj_kinematics(m, d2)
            mujoco.mj_collision(m, d2)
            if d2.ncon == 0:
                return True
            b = m.geom_bodyid[d2.contact.geom[:d2.ncon]].reshape(-1, 2)
            r = ss._is_robot[b]
            sel = r[:, 0] ^ r[:, 1]
            mine = np.where(r[:, 0], b[:, 0], b[:, 1])[sel]
            oth = np.where(r[:, 0], b[:, 1], b[:, 0])[sel]
            bad = touchy[mine] & ~own_pay[oth]
            return not bool(bad.any())

        sampler = WholeStrokeSampler(ss.robot.model, collision_fn=col_ins)
        cands = sampler.sample_cavity_or_overhang_insert(
            p_rest_site=p_site_rest,
            R_grasp_candidates=R_cands,
            open_directions=[out_n],
            finger_held=f_held,
            q_ref=ss.q_home,
            slide_len=slide_len,
            lift_clear=lift_clear,
            max_found=1,
        )
        if not cands:
            return False

        c0 = cands[0]
        # Start at mouth entry with open fingers, place payload between pads, and physically grasp it!
        if not self._fast_set_arm(ss, fast_rb, c0.q_handoff, 0.04):
            return False

        m, d = ss.robot.m, ss.robot.d
        j_pay = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "ucs_payload_free")
        qadr = int(m.jnt_qposadr[j_pay])
        vadr = int(m.jnt_dofadr[j_pay])
        R_h, p_h = fast_rb.eef_pose()
        # Position payload centered between finger pads with small grasp offset noise
        jitter = rng.uniform(-0.005, 0.005, size=3) * scale
        p_pay0 = p_h - grasp_off_obj + jitter
        d.qpos[qadr:qadr + 3] = p_pay0
        d.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]   # upright w,x,y,z
        d.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(m, d)

        # Physically close fingers onto payload (12 fast steps) and verify payload didn't fall!
        for _ in range(12):
            fast_rb.step(R_h, p_h, 1.0)
        p_pay_held = d.qpos[qadr:qadr + 3].copy()
        if p_pay_held[2] < p_pay0[2] - 0.025 or fast_rb.finger_opening() < 0.004:
            return False

        self.mode = mode
        self.pay_qadr = qadr
        self.p_obj_goal = p_obj_rest.copy()
        self.p_site_goal = p_site_rest.copy()
        self.true_span = float(np.linalg.norm(p_site_rest - p_h))
        self.true_tangent = _unit(p_site_rest - p_h)
        self.true_axis = up.copy()
        self.true_anchor = p_site_rest.copy()
        self.approach = c0.approach.copy()
        self.way_in = 0.0
        pay_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ctgt.payload_body)
        fix_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ctgt.fixture_body)
        self.own_mask = scenes._subtree(m, pay_bid)
        self.fix_mask = scenes._subtree(m, fix_bid)
        return True

    def _restore_start(self, scale: float):
        sv = self.saved_start
        self.ss.restore(sv["snap"])
        self.fast_rb.sync_from_robosuite()
        self.mode = sv["mode"]
        self.true_tangent = sv["true_tangent"].copy()
        self.true_axis = sv["true_axis"].copy()
        self.true_anchor = sv["true_anchor"].copy()
        self.true_span = sv["true_span"]
        self.approach = sv["approach"].copy()
        self.way_in = sv["way_in"]
        self.part = sv["part"]
        self.skill = UCSExecSkill(self.ss.robot.model.limits[:7])
        self._start_skill_with_belief(scale)

    def _start_skill_with_belief(self, scale: float) -> bool:
        c, rng = self.cfg, self.rng
        ang_rad = np.radians(rng.uniform(0.0, c["err_ang_max_deg"] * scale))
        b_tang = rotvec_to_mat(_perp_unit(self.true_tangent, rng) * ang_rad) @ self.true_tangent
        b_axis = rotvec_to_mat(_perp_unit(self.true_axis, rng) * ang_rad) @ self.true_axis
        b_anch = self.true_anchor + rng.uniform(-1.0, 1.0, size=3) * (c["err_pos_max"] * scale)
        b_span = self.true_span * float(1.0 + rng.uniform(-1.0, 1.0) * c["err_span_max"] * scale)

        goal = ExecGoal(
            mode=self.mode,
            believed_tangent=b_tang,
            believed_axis=b_axis,
            believed_anchor=b_anch,
            target_span=max(0.03, b_span),
            approach_axis=self.approach,
            way_in_m=self.way_in,
        )
        self.obs_cur = self.skill.start(self.fast_rb, goal)
        m = self.ss.robot.m
        refs = self.ss._body
        self.finger_body = np.asarray(["finger" in n for n in refs])
        self.goal_floor = np.zeros(m.ngeom,bool)
        if self.ctgt is not None:
            for suffix in ("_floor","_dfloor"):
                gid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,self.ctgt.fixture_body+suffix)
                if gid>=0:
                    self.goal_floor[gid] = True
        self.grip_lost_steps = 0
        self.left_prev = self._dist_left()
        self.left0 = max(self.left_prev, 0.02)
        self.max_pen_m = 0.0
        self.outside_hits = 0
        self.ep_ret = 0.0
        self.ep_steps = 0
        self.a_prev = np.zeros(EXEC_ACT_DIM, dtype=float)
        self.info = {
            "kind": self.kind,
            "mode": self.mode,
            "scale": round(scale, 3),
            "retire": self.loads >= self.cfg["max_loads"],
        }
        return True

    def _dist_left(self) -> float:
        if self.mode in ("cavity_insert", "overhang_place"):
            p_pay = self.ss.robot.d.qpos[self.pay_qadr:self.pay_qadr + 3]
            return float(np.linalg.norm(p_pay - self.p_obj_goal))
        q_now = self.ss.joint(self.part.joint)
        rem = abs(self.q_joint_goal - q_now)
        return rem * (0.08 if self.mode == "knob_turn" else 1.0)

    def _is_success(self, max_pen_tol_m: float = 0.0022) -> bool:
        if self.max_pen_m > max_pen_tol_m:
            return False
        if self.mode in ("cavity_insert", "overhang_place"):
            if not self.skill.released:
                return False
            d = self.ss.robot.d
            p_pay = d.qpos[self.pay_qadr:self.pay_qadr + 3]
            bid_pay = mujoco.mj_name2id(self.ss.robot.m, mujoco.mjtObj.mjOBJ_BODY, self.ctgt.payload_body)
            R_pay = d.xmat[bid_pay].reshape(3, 3)
            # Upright check + inside container footprint + on floor + hand clear!
            rel_local = self.ctgt.R_world.T @ (p_pay - self.ctgt.floor_center_world)
            in_xy = (
                abs(rel_local[0]) <= self.ctgt.floor_half_xy[0] + 0.015
                and abs(rel_local[1]) <= self.ctgt.floor_half_xy[1] + 0.015
            )
            on_floor = abs((p_pay[2] - self.ctgt.payload_half_h) - self.ctgt.floor_center_world[2]) <= 0.022
            upright = float(R_pay[2, 2]) >= np.cos(np.radians(28.0))
            hand_clear = float(np.linalg.norm(self.fast_rb.eef_pose()[1] - p_pay)) >= 0.035
            return bool(in_xy and on_floor and upright and hand_clear)

        q_now = self.ss.joint(self.part.joint)
        if self.mode == "push_close":
            tol = 0.030 if self.part.kind == "hinge" else 0.007
            # Signed opening remaining: pushing past `part.closed` into the positive endstop (+0.01 on LIBERO cabinets) IS closed!
            s_open = float(np.sign(self.part.open - self.part.closed))
            rem_open = (q_now - self.part.closed) * s_open
            return rem_open <= tol
        if self.mode == "knob_turn":
            moved = (q_now - self.q_joint_start) * float(np.sign(self.q_joint_goal - self.q_joint_start))
            return abs(self.q_joint_goal - q_now) <= np.deg2rad(5)
        # pull_open
        moved = (q_now - self.q_joint_start) * float(np.sign(self.q_joint_goal - self.q_joint_start))
        return abs(self.q_joint_goal - q_now) <= (np.deg2rad(5) if self.part.kind == "hinge" else 0.007)

    def _scan_contacts(self) -> tuple[float, bool, bool]:
        """Returns `(forbidden_scene_pen_m, hit_outside, holding_or_touching_target)`.
        Matches `SceneCollisionMonitor`: finger-on-target-handle/payload (`own_mask` with finger geoms) is intended
        grasp/manipulation; non-finger or non-target contacts count toward scene penetration & outside hits."""
        m, d = self.ss.robot.m, self.ss.robot.d
        max_pen = 0.0
        outside = False
        touch_own = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = bool(self.ss._is_robot[b1]), bool(self.ss._is_robot[b2])
            carrying = self.part is None and not self.skill.released
            pay1,pay2 = carrying and self.own_mask[b1],carrying and self.own_mask[b2]
            if (pay1 or pay2) and not (r1 or r2):
                other_geom = c.geom2 if pay1 else c.geom1
                if not (pay1 and pay2) and not self.goal_floor[other_geom]:
                    pen = max(0.,float(-c.dist))
                    max_pen = max(max_pen,pen)
                    outside |= pen > .0005
                continue
            if not (r1 ^ r2):
                continue
            rob_b = b1 if r1 else b2
            oth = b2 if r1 else b1
            pen = float(-c.dist)
            is_finger = "finger" in self.ss._body[rob_b]
            if self.own_mask[oth] and is_finger:
                touch_own = True
                # For articulated handles, even finger-on-handle penetration is tracked; for held free payload, pad squeeze is grasp
                if self.part is not None:
                    max_pen = max(max_pen, pen)
            else:
                max_pen = max(max_pen, pen)
                if pen > 0.0005:
                    outside = True
        return max_pen, outside, touch_own

    def _critic(self) -> np.ndarray:
        """22-D privileged state for asymmetric critic ONLY."""
        Rc = self.skill.R_c
        p_err_anch = Rc.T @ (self.true_anchor - self.skill.goal.believed_anchor)
        return np.hstack([
            Rc.T @ self.true_tangent,
            Rc.T @ self.true_axis,
            p_err_anch,
            self.true_span,
            self._dist_left(),
            1000.0 * self.max_pen_m,
            float(self.outside_hits > 0),
            float(self.skill.released),
            (np.array(EXEC_MODES) == self.mode).astype(float),
            self.scale,
            0.0,
            0.0,
        ]).astype(np.float32)

    def step(self, action):
        rw = self.rw
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.obs_cur = self.skill.step(a)
        self.ep_steps += 1

        pen_now, outside, touch_own = self._scan_contacts()
        self.max_pen_m = max(self.max_pen_m, pen_now)
        if outside:
            self.outside_hits += 1

        left_now = self._dist_left()
        prog = (self.left_prev - left_now) / self.left0
        self.left_prev = left_now

        q7 = self.fast_rb.q()[:7]
        lim = self.skill.limits
        min_lim = float(np.minimum(q7 - lim[:, 0], lim[:, 1] - q7).min())

        r = (
            rw["progress"] * prog
            - rw["pen_over_1mm"] * float(pen_now > 0.0008)
            - rw["outside_contact"] * float(outside)
            - rw["limit_warn"] * float(min_lim < 0.08)
            - rw["action_smooth"] * float(np.mean((a - self.a_prev) ** 2))
            - rw["step_cost"]
        )

        # Extra shaping for keeping payload upright during cavity_insert / overhang_place
        dropped_payload = False
        if self.mode in ("cavity_insert", "overhang_place") and not self.skill.released:
            p_pay = self.ss.robot.d.qpos[self.pay_qadr:self.pay_qadr + 3]
            if float(np.linalg.norm(p_pay - self.fast_rb.eef_pose()[1])) > 0.11:
                dropped_payload = True
                self.skill.stop = "dropped"
            else:
                r += rw["ref_align"]

        if self.mode in ("pull_open","knob_turn") and not touch_own:
            self.grip_lost_steps += 1
        else:
            self.grip_lost_steps = 0
        grasp_missed = self.grip_lost_steps >= 3
        ended = self.skill.stop is not None
        succ = False
        if ended:
            succ = self._is_success()
            if succ:
                r += rw["bonus_success"]
            elif dropped_payload or (self.mode in ("pull_open", "knob_turn") and not touch_own and left_now > 0.5 * self.left0):
                r += rw["penalty_drop_or_slip"]

        self.a_prev = a
        self.ep_ret += r

        info = {}
        if ended:
            label = (
                "reached_goal" if succ else
                "slipped_or_missed" if dropped_payload or grasp_missed else
                "blocked_obstacle" if (self.skill.stop == "force" or self.outside_hits > 3) else
                "progress_short"
            )
            info = {
                **self.info,
                "success": bool(succ),
                "stop": str(self.skill.stop),
                "steps": int(self.ep_steps),
                "sim_s": round(self.ep_steps * EXEC_CFG["dt"], 2),
                "return": round(float(self.ep_ret), 3),
                "max_pen_mm": round(1000.0 * self.max_pen_m, 3),
                "outside_hits": int(self.outside_hits),
                "label": label,
                "retire": self.loads >= self.cfg["max_loads"],
            }
            if self.dump_dir is not None and self.rng.random() < self.cfg["dump_share"]:
                self.dump_buf.append({
                    "trace": np.asarray(self.skill.trace_rows, dtype=np.float32),
                    "mode": self.mode,
                    "target_span": float(self.skill.goal.target_span),
                    "label": label,
                    "success": bool(succ),
                })
                if len(self.dump_buf) >= 10:
                    self._flush_dumps()

        return self.obs_cur, self._critic(), float(r), bool(ended), info

    def _flush_dumps(self):
        if self.dump_dir is not None and self.dump_buf:
            p_dump = self.dump_dir / f"ucs_trace_{os.getpid()}_{self.dump_idx:04d}.pkl"
            with open(p_dump, "wb") as f:
                pickle.dump(self.dump_buf, f)
            self.dump_buf.clear()
            self.dump_idx += 1

    def close(self):
        self._flush_dumps()
        for slot in self.pool:
            slot[0].close()
        self.pool.clear()
