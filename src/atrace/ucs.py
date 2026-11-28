"""Unified Contact Stage (UCS) runtime module for `agentic_trace`.

Unifies all quasi-static contact-rich stages under one real-robot-legal interface:
  1. Articulated mechanisms (prismatic pull/push, revolute door open/close, rotary knob/valve cw/ccw)
  2. Constrained rigid-body insertion & dense placement (side-cavity slide, under-overhang drawer place, tight slot)
  3. Compliant Leaky-PLAI controller wrapper (caps normal steady-state force <= 16.5 N so contact penetration stays
     strictly <= 1.0 mm under `SceneCollisionMonitor`, and avoids 40 N force-stop spikes)
  4. Online Wrench-Displacement Screw Estimator (active probing in the first steps to lock the true instantaneous
     motion tangent and detect hard end-stops from compliance rather than open-loop thresholds)
  5. Whole-Stroke Kinematic-Chain Pre-Contact Sampler (ensures continuous IK + joint-limit margin + arm clearance
     across the ENTIRE predicted screw stroke / insertion corridor before handing <= 5 cm entries to TRACE)
  6. Frozen-Shape SE(3) ObjectStore (eliminates post-release re-segmentation drift & target loss).

Constants ledger tags (per LRL_PLAN.md Rule 7):
  R = Robot / controller specification
  M = Measured on stage simulator / physics probe
  C = Calibrated operating point
  T = Training / design prior
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import manip
from .geometry import frame_from_axes, mat_to_rotvec, rotvec_to_mat

UCS_MODES = (
    "pull_open",
    "push_close",
    "rotate_cw",
    "rotate_ccw",
    "cavity_insert",
    "surface_place",
)

UCS_CFG = {
    "dt": 0.05,                  # R  s: 20 Hz control period
    # Leaky-PLAI & <1mm penetration compliance (IndustReal V-D + measured k_contact ~ 9000 N/m, b_osc ~ 1730 Ns/m)
    "dp_max": 0.008,             # T  m: max incremental target step at alpha=1.0 (~16 cm/s max rate)
    "dr_max": 5.0,               # T  deg: max incremental orientation step at alpha=1.0
    "plai_pos_clamp": 0.007,     # M  m: max target lead |p_des - p_meas| at alpha=1.0 (bounds impact & steady force!)
    "plai_rot_clamp": 6.0,       # T  deg: max orientation lead |R_des * R_meas^T| at alpha=1.0
    "kp_contact_pos": 45.0,      # M  1/s^2: compliant operational-space position stiffness during contact stage
    "kp_contact_ori": 55.0,      # M  1/s^2: compliant operational-space orientation stiffness during contact stage
    "alpha_range": (0.25, 1.0),  # T  variable compliance scale controlled by policy / mode
    "f_touch": 3.5,              # M  N: tared force above which contact regulation engages
    "f_steady_cap": 15.5,        # M  N: max steady normal force (penetration ~ 0.1-0.4 mm << 1.0 mm!)
    "f_hard_stop": 36.0,         # T  N: sustained safety stop threshold (below 40 N monitor/wrapper limit)
    "force_stop_steps": 4,       # T  consecutive steps above f_hard_stop to trigger safe backoff
    "k_contact": 3400.0,         # M  N/m: measured OSC stiffness on rigid stop at kp_contact_pos = 45
    "b_osc": 850.0,              # M  N s/m: measured OSC damping credit at kp_contact_pos = 45
    "rot_contact_cap": 3.0,      # T  deg: max orientation lead while in contact (prevents lever-arm force spikes)
    "lowpass": 3,                # T  moving-average window on tared wrench
    # Whole-stroke kinematic-chain pre-contact sampler
    "handoff_standoff": (0.035, 0.055),  # T  m: <= 5.5 cm pre-contact handoff (wrist-cam sharp, minimal IK drift)
    "chain_steps": 5,            # T  check points along the predicted screw / insertion trajectory
    "approach_steps": 3,         # T  check points from handoff standoff to contact
    "joint_slack": 0.14,         # T  rad: strict margin from Panda joint limits (especially q7 +-2.897)
    "tilts_deg": (0.0, 28.0, 52.0, 68.0),  # T  deg: approach pitch angles (clears upper drawers & table)
    "pad_margins": (0.010, 0.006),         # T  m: finger opening margin per side around handle/knob
    # Online screw & endstop estimator
    "history_len": 12,           # T  steps of proprioceptive + wrench history (0.6 s)
    "endstop_vel": 0.004,        # M  m/s: hand speed below this under >10 N axial force signals mechanical endstop
    "endstop_force": 9.5,        # M  N: axial resistance confirming mechanical endstop
    "endstop_steps": 5,          # T  consecutive steps at mechanical endstop to declare reached_endstop
}


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


@dataclass
class ScrewHypothesis:
    """Unified kinematic twist hypothesis for 1-DoF mechanisms AND rigid cavity/surface insertions.

    pitch = inf: pure translation along `axis` (drawer slide, sliding door, cavity slide-in, vertical set-down)
    pitch = 0, radius > 0: revolute arc about line (`anchor`, `axis`) (hinged microwave/cabinet/fridge door)
    pitch = 0, radius = 0: pure in-place rotation about (`anchor`, `axis`) (stove knob, faucet valve)
    """
    kind: str                    # "prismatic" | "revolute" | "rotary"
    axis: np.ndarray             # unit 3D vector (direction of translation for prismatic; hinge/knob axis otherwise)
    anchor: np.ndarray           # 3D point on axis (or initial contact point for prismatic)
    span: float                  # expected travel (m for prismatic, rad for revolute/rotary; signed or positive)
    sign: float = 1.0            # +1 or -1 along/about axis

    def __post_init__(self):
        self.axis = _unit(self.axis)
        self.anchor = np.asarray(self.anchor, dtype=float)

    def propagate_pose(
        self,
        R0: np.ndarray,
        p0: np.ndarray,
        s: float,
        roll_axis_0: np.ndarray | None = None,
        roll_rad: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate end-effector pose (R0, p0) along the screw at progress fraction s in [0, 1].
        Optional `roll_axis_0` + `roll_rad` allows cylindrical handle roll relaxation around the grasped bar
        so wide-arc doors and low drawers never artificially hit wrist joint limits."""
        delta = float(s) * self.span * self.sign
        if self.kind == "prismatic":
            R_s, p_s = R0.copy(), p0 + self.axis * delta
            ax_s = _unit(roll_axis_0) if roll_axis_0 is not None else None
        else:
            # Exact SE(3) rotation about line (anchor, axis): covers both r=0 knobs and r>0 doors/levers!
            Rq = rotvec_to_mat(self.axis * delta)
            R_s = Rq @ R0
            p_s = Rq @ (p0 - self.anchor) + self.anchor
            ax_s = Rq @ _unit(roll_axis_0) if roll_axis_0 is not None else None

        if ax_s is not None and abs(roll_rad) > 1e-6:
            R_s = rotvec_to_mat(ax_s * roll_rad) @ R_s
        return R_s, p_s

    def tangent_at(self, p_hand: np.ndarray) -> np.ndarray:
        """Unit linear velocity direction of `p_hand` along this screw."""
        if self.kind == "prismatic":
            return self.axis * self.sign
        v = np.cross(self.axis * self.sign, np.asarray(p_hand, float) - self.anchor)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else self.axis * self.sign


class OnlineScrewTracker:
    """Online wrench-displacement estimator that adapts the believed motion direction during contact
    and detects mechanical end-stops without per-fixture constants."""

    def __init__(self, prior_tangent: np.ndarray, cfg: dict = UCS_CFG):
        self.cfg = cfg
        self.tangent = _unit(prior_tangent)
        self.positions: list[np.ndarray] = []
        self.forces: list[np.ndarray] = []
        self.endstop_count = 0

    def update(self, p_meas: np.ndarray, v_meas: np.ndarray, f_tared: np.ndarray) -> dict:
        self.positions.append(np.asarray(p_meas, float).copy())
        self.forces.append(np.asarray(f_tared, float).copy())
        if len(self.positions) >= 4:
            disp = self.positions[-1] - self.positions[max(0, len(self.positions) - 6)]
            d_norm = float(np.linalg.norm(disp))
            if d_norm > 0.006:
                emp = disp / d_norm
                if float(emp @ self.tangent) > 0.25:
                    # Smoothly fuse empirical displacement direction (follows curved door arcs automatically!)
                    self.tangent = _unit(0.65 * self.tangent + 0.35 * emp)

        # Mechanical endstop test: pressing along tangent, force opposes motion >= endstop_force, speed ~ 0
        # Note: wrist sensor reads force applied BY robot ON environment (or reaction depending on sign; project norm)
        f_mag = float(np.linalg.norm(f_tared))
        f_axial = abs(float(f_tared @ self.tangent))
        speed_axial = abs(float(v_meas @ self.tangent))
        if f_mag >= self.cfg["endstop_force"] and f_axial >= 0.6 * f_mag and speed_axial <= self.cfg["endstop_vel"]:
            self.endstop_count += 1
        else:
            self.endstop_count = 0

        travel = float((self.positions[-1] - self.positions[0]) @ self.tangent) if len(self.positions) > 1 else 0.0
        return {
            "tangent": self.tangent.copy(),
            "travel_m": travel,
            "at_endstop": self.endstop_count >= self.cfg["endstop_steps"],
            "endstop_count": self.endstop_count,
        }


class LeakyPLAICompliantController:
    """Compliant Leaky Policy-Level Action Integrator (PLAI) with normal-force projection cap.

    Why this exists:
      1. Eliminates steady-state tracking lag under drawer damping / gravity without high kp gains.
      2. Bounds maximum normal contact force to `f_steady_cap` (~15.5 N), guaranteeing simulated geom
         penetration stays < 1.0 mm (`scene-contact-v1` tolerance) and preventing 40 N safety trips.
      3. Allows policy-controlled variable compliance `alpha in [0.25, 1.0]`.
    """

    def __init__(self, cfg: dict = UCS_CFG):
        self.cfg = cfg
        self.reset_state(np.eye(3), np.zeros(3), np.zeros(6))

    def apply_contact_gains(self, ctrl_or_fast_rb, alpha: float = 0.85):
        """Switch controller into compliant contact impedance (`kp_contact_pos * alpha`).
        Supports `FastSimRobot` (`_kp`), `SimRobot` (`ctrl.kp`), and raw controllers (`kp`)."""
        c = self.cfg
        alpha = float(np.clip(alpha, *c["alpha_range"]))
        kp_vec = np.array([c["kp_contact_pos"] * alpha] * 3 + [c["kp_contact_ori"] * alpha] * 3, dtype=float)
        kd_vec = 2.0 * np.sqrt(kp_vec)
        if hasattr(ctrl_or_fast_rb, "_kp"):
            ctrl_or_fast_rb._kp[:] = kp_vec
            ctrl_or_fast_rb._kd[:] = kd_vec
            ctrl_or_fast_rb.base.ctrl.kp[:] = kp_vec
            ctrl_or_fast_rb.base.ctrl.kd[:] = kd_vec
        elif hasattr(ctrl_or_fast_rb, "ctrl") and hasattr(ctrl_or_fast_rb.ctrl, "kp"):
            ctrl_or_fast_rb.ctrl.kp[:] = kp_vec
            ctrl_or_fast_rb.ctrl.kd[:] = kd_vec
        elif hasattr(ctrl_or_fast_rb, "kp"):
            ctrl_or_fast_rb.kp[:] = kp_vec
            ctrl_or_fast_rb.kd[:] = kd_vec

    @staticmethod
    def restore_nominal_gains(ctrl_or_fast_rb, kp_nom: float = 150.0):
        """Restore robosuite's nominal `kp = 150` so subsequent TRACE certified transits keep exact tracking."""
        kp_vec = np.full(6, kp_nom, dtype=float)
        kd_vec = 2.0 * np.sqrt(kp_vec)
        if hasattr(ctrl_or_fast_rb, "_kp"):
            ctrl_or_fast_rb._kp[:] = kp_vec
            ctrl_or_fast_rb._kd[:] = kd_vec
            ctrl_or_fast_rb.base.ctrl.kp[:] = kp_vec
            ctrl_or_fast_rb.base.ctrl.kd[:] = kd_vec
        elif hasattr(ctrl_or_fast_rb, "ctrl") and hasattr(ctrl_or_fast_rb.ctrl, "kp"):
            ctrl_or_fast_rb.ctrl.kp[:] = kp_vec
            ctrl_or_fast_rb.ctrl.kd[:] = kd_vec
        elif hasattr(ctrl_or_fast_rb, "kp"):
            ctrl_or_fast_rb.kp[:] = kp_vec
            ctrl_or_fast_rb.kd[:] = kd_vec

    def reset_state(self, R0: np.ndarray, p0: np.ndarray, tare_wrench: np.ndarray):
        self.R_des = np.asarray(R0, float).copy()
        self.p_des = np.asarray(p0, float).copy()
        self.w0 = np.asarray(tare_wrench, float).copy()
        self.wbuf = np.zeros((self.cfg["lowpass"], 6), dtype=float)
        self.k = 0
        self.over_force = 0

    def compute_command(
        self,
        delta_pos_world: np.ndarray,
        delta_rot_world: np.ndarray,
        alpha: float,
        R_meas: np.ndarray,
        p_meas: np.ndarray,
        v_meas: np.ndarray,
        wrench_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        c = self.cfg
        alpha = float(np.clip(alpha, *c["alpha_range"]))
        raw = np.asarray(wrench_world, float) - self.w0
        self.wbuf[self.k % min(len(self.wbuf), max(1, self.k + 1)) if self.k < c["lowpass"] else self.k % c["lowpass"]] = raw
        n_valid = min(self.k + 1, c["lowpass"])
        lp = self.wbuf[:n_valid].mean(axis=0)
        self.k += 1

        # 1. Bound incremental step by alpha
        dp = np.asarray(delta_pos_world, float)
        dp_n = float(np.linalg.norm(dp))
        max_dp = c["dp_max"] * alpha
        if dp_n > max_dp:
            dp = dp * (max_dp / dp_n)

        dr = np.asarray(delta_rot_world, float)
        dr_n = float(np.linalg.norm(dr))
        max_dr = np.radians(c["dr_max"] * alpha)
        if dr_n > max_dr:
            dr = dr * (max_dr / dr_n)

        # 2. Integrate on previous desired pose (PLAI)
        p_next = self.p_des + dp
        R_next = rotvec_to_mat(dr) @ self.R_des

        # 3. Leaky clamp around measured pose (prevents windup on stalls/walls)
        lead = p_next - p_meas
        pos_clamp = c["plai_pos_clamp"] * alpha
        lead_n = float(np.linalg.norm(lead))
        if lead_n > pos_clamp:
            lead = lead * (pos_clamp / lead_n)

        # 4. Normal-force compliance projection (<1mm penetration guarantee!)
        # Use max of raw and low-passed force along reaction axis so transient spikes are caught immediately
        f_vec = raw[:3]
        f_mag = float(np.linalg.norm(f_vec))
        f_cap = c["f_steady_cap"] * (0.5 + 0.5 * alpha)
        if f_mag > c["f_touch"]:
            # Determine pushing unit vector: if lead has component opposing/matching force, cap the pressing direction
            # In robosuite Panda, pressing along +u produces wrist force along +u (or -u depending on frame convention);
            # align u_press with whichever sign of f_vec has positive dot product with `lead`!
            u_f = f_vec / f_mag
            u_press = u_f if float(lead @ u_f) >= 0.0 else -u_f
            max_press_lead = (f_cap + c["b_osc"] * max(0.0, float(v_meas @ u_press))) / c["k_contact"]
            excess = float(lead @ u_press) - max_press_lead
            if excess > 0.0:
                lead = lead - excess * u_press

        p_cmd = p_meas + lead

        # Orientation clamp (tighter when in contact so fingertip lever arms cannot gouge surfaces)
        rot_err = mat_to_rotvec(R_next @ R_meas.T)
        rot_lim = np.radians(c["rot_contact_cap"] if f_mag > c["f_touch"] else c["plai_rot_clamp"] * alpha)
        re_n = float(np.linalg.norm(rot_err))
        if re_n > rot_lim:
            rot_err = rot_err * (rot_lim / re_n)
        R_cmd = rotvec_to_mat(rot_err) @ R_meas

        self.p_des = p_cmd.copy()
        self.R_des = R_cmd.copy()

        lp_f = float(np.linalg.norm(lp[:3]))
        self.over_force = self.over_force + 1 if lp_f > c["f_hard_stop"] else 0

        return R_cmd, p_cmd, {
            "raw_force_n": f_mag,
            "lp_force_n": lp_f,
            "lead_mm": 1000.0 * float(np.linalg.norm(lead)),
            "force_stop": self.over_force >= c["force_stop_steps"],
        }


@dataclass
class ContactCandidate:
    """A pre-verified whole-stroke candidate ready for TRACE hand-off + local contact execution."""
    name: str
    mode: str
    R_handoff: np.ndarray
    p_handoff: np.ndarray
    R_contact: np.ndarray
    p_contact: np.ndarray
    approach: np.ndarray
    finger_open: float
    q_handoff: np.ndarray
    q_chain: np.ndarray          # (K, 7) continuous joint chain across approach + full stroke
    screw: ScrewHypothesis
    min_limit_margin_rad: float
    roll_axis_0: np.ndarray | None = None
    roll_schedule: np.ndarray | None = None
    prefer_handle: bool = True
    score: float = 0.0


class WholeStrokeSampler:
    """Samples and filters pre-contact hand-off poses (<= 5.5 cm from target) by requiring a continuous,
    collision-free, joint-limit-safe IK chain across BOTH the approach leg AND the entire contact stroke.

    Prioritizes explicit handles/knobs/affordances over panel faces so `SceneCollisionMonitor`
    (`fixture_finger_interaction` + `<= 1 mm` tolerance) is satisfied by construction.
    """

    def __init__(self, robot_model, collision_fn=None, cfg: dict = UCS_CFG):
        """
        robot_model: `atrace.robot_model.RobotModel`
        collision_fn: optional callable `(q7, finger, step_idx, s_frac) -> bool` returning True if clear
        """
        self.rm = robot_model
        self.collision_fn = collision_fn
        self.cfg = cfg
        self.limits = robot_model.limits[:7].copy()

    def _limit_margin(self, q7: np.ndarray) -> float:
        return float(np.minimum(q7 - self.limits[:, 0], self.limits[:, 1] - q7).min())

    def _nullspace_ik_step(self, R: np.ndarray, p: np.ndarray, q_seed: np.ndarray, finger: float) -> np.ndarray | None:
        # Bias rest posture toward joint midpoints so q7 stays away from +-2.897 during arcs/rotations
        q_mid = 0.5 * (self.limits[:, 0] + self.limits[:, 1])
        q_rest = 0.7 * q_seed + 0.3 * q_mid
        qs, ok, _, _ = self.rm.ik(
            R, p, q_seed[None], q_rest=q_rest, limit_margin=self.cfg["joint_slack"], finger=finger
        )
        return qs[0] if ok[0] else None

    def verify_chain(
        self,
        R_contact: np.ndarray,
        p_contact: np.ndarray,
        approach: np.ndarray,
        standoff: float,
        finger_approach: float,
        finger_stroke: float,
        screw: ScrewHypothesis,
        q_ref: np.ndarray,
        n_random: int = 12,
        roll_axis_0: np.ndarray | None = None,
        max_roll_deg: float = 36.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        """Verify continuous IK + collision clearance from handoff -> contact -> end of screw stroke.
        Returns `(q_handoff, q_chain, roll_schedule_rad, min_limit_margin_rad)` or None."""
        c = self.cfg
        p_handoff = p_contact - approach * standoff
        sols = self.rm.solve_ik(R_contact, p_handoff, q_ref, n_random=n_random, finger=finger_approach)
        if len(sols) == 0:
            return None

        margins0 = np.minimum(sols - self.limits[:, 0], self.limits[:, 1] - sols).min(axis=1)
        keep_mask = margins0 >= c["joint_slack"]
        sols = sols[keep_mask]
        margins0 = margins0[keep_mask]
        if len(sols) == 0:
            return None
        order = np.argsort(-margins0 + 0.15 * np.linalg.norm(sols - q_ref, axis=1))
        sols = sols[order[:2]]

        roll_deltas = (0.0,) if roll_axis_0 is None else np.radians((0.0, 18.0, -18.0))
        max_roll_rad = np.radians(max_roll_deg)

        for q0 in sols:
            if self.collision_fn is not None and not self.collision_fn(q0, finger_approach, 0, -1.0):
                continue
            chain = [q0]
            rolls = [0.0]
            q = q0
            ok = True
            # 1. Approach chain (handoff -> contact)
            for j in range(1, c["approach_steps"] + 1):
                frac = j / c["approach_steps"]
                pj = p_handoff + (p_contact - p_handoff) * frac
                q = self._nullspace_ik_step(R_contact, pj, q, finger_approach)
                if q is None:
                    ok = False
                    break
                if self.collision_fn is not None and not self.collision_fn(q, finger_approach, j, 0.0):
                    ok = False
                    break
                chain.append(q)
                rolls.append(0.0)
            if not ok:
                continue

            # 2. Full stroke chain (contact -> end of screw / cavity insertion), allowing smooth cylindrical handle roll
            cur_roll = 0.0
            for k in range(1, c["chain_steps"] + 1):
                s = k / c["chain_steps"]
                step_found = False
                for d_phi in roll_deltas:
                    cand_roll = cur_roll + float(d_phi)
                    if abs(cand_roll) > max_roll_rad + 1e-6:
                        continue
                    Rs, ps = screw.propagate_pose(R_contact, p_contact, s, roll_axis_0=roll_axis_0, roll_rad=cand_roll)
                    q_next = self._nullspace_ik_step(Rs, ps, q, finger_stroke)
                    if q_next is None:
                        continue
                    if self.collision_fn is not None and not self.collision_fn(
                        q_next, finger_stroke, c["approach_steps"] + k, s
                    ):
                        continue
                    q = q_next
                    cur_roll = cand_roll
                    chain.append(q)
                    rolls.append(cur_roll)
                    step_found = True
                    break
                if not step_found:
                    ok = False
                    break
            if ok:
                arr = np.stack(chain, axis=0)
                min_m = float(np.minimum(arr - self.limits[:, 0], self.limits[:, 1] - arr).min())
                return q0, arr, np.asarray(rolls, dtype=float), min_m
        return None

    def sample_handle_or_knob(
        self,
        mode: str,
        handle_center: np.ndarray,
        handle_axis: np.ndarray,
        handle_width: float,
        face_normal: np.ndarray,
        screw: ScrewHypothesis,
        q_ref: np.ndarray,
        tip_z: float = manip.TIP_Z,
        pad_offset: float = manip.PAD_OFFSET,
        standoff_behind: float = 0.025,
        max_found: int = 3,
        n_random: int = 4,
    ) -> list[ContactCandidate]:
        """Generate handle-first pinch/push candidates (including 180-deg symmetric twins) verified over full stroke."""
        c = self.cfg
        n = _unit(face_normal)
        e = _unit(handle_axis)
        up = np.array([0.0, 0.0, 1.0])
        is_rotary = screw.kind == "rotary"

        if abs(n[2]) > 0.75:
            tilts = [(0.0, -n)]
        else:
            h = _unit(np.array([n[0], n[1], 0.0]))
            tilts = [
                (t, _unit(-np.cos(np.radians(t)) * h - np.sin(np.radians(t)) * up))
                for t in c["tilts_deg"]
            ]

        out: list[ContactCandidate] = []
        for margin in c["pad_margins"]:
            f_open = float(np.clip(0.5 * handle_width + margin - pad_offset, 0.005, 0.04))
            f_stroke = 0.0 if mode != "surface_place" else f_open
            for tilt, a in tilts:
                cos_n = max(float(-(a @ n)), 0.25)
                # Keep fingertips 4 mm clear of the backing panel!
                back = max(0.0, tip_z - (standoff_behind - 0.004) / cos_n)
                p_cont = np.asarray(handle_center, float) - a * back

                x_base = np.cross(n, e)
                if np.linalg.norm(x_base) < 1e-3:
                    x_base = np.cross(up, a) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                x_base = _unit(x_base)
                # Include 180-deg symmetric twin (-x_base), plus orthogonal pair for round knobs!
                x_list = [x_base, -x_base]
                if is_rotary:
                    x_orth = _unit(np.cross(a, x_base))
                    x_list.extend([x_orth, -x_orth])

                for twin_i, x_dir in enumerate(x_list):
                    x_proj = x_dir - float(x_dir @ a) * a
                    if np.linalg.norm(x_proj) < 1e-3:
                        continue
                    R_cont = frame_from_axes(a, _unit(x_proj))
                    # Allow cylindrical roll relaxation around bar handles for pull/push on wide arcs/low drawers
                    r_ax = e if (not is_rotary and mode in ("pull_open", "push_close")) else None
                    for st_dist in c["handoff_standoff"]:
                        ver = self.verify_chain(
                            R_cont, p_cont, a, st_dist, f_open, f_stroke, screw, q_ref, n_random=n_random, roll_axis_0=r_ax
                        )
                        if ver is not None:
                            q0, q_chain, rolls, min_m = ver
                            roll_cost = 0.15 * float(np.max(np.abs(rolls)))
                            score = 2.0 * min_m - 0.01 * tilt - 0.2 * float(np.linalg.norm(q0 - q_ref)) - roll_cost
                            out.append(
                                ContactCandidate(
                                    name=f"handle_t{tilt:.0f}_w{int(margin*1000)}_tw{twin_i}",
                                    mode=mode,
                                    R_handoff=R_cont,
                                    p_handoff=p_cont - a * st_dist,
                                    R_contact=R_cont,
                                    p_contact=p_cont,
                                    approach=a,
                                    finger_open=f_open,
                                    q_handoff=q0,
                                    q_chain=q_chain,
                                    screw=screw,
                                    min_limit_margin_rad=min_m,
                                    roll_axis_0=r_ax,
                                    roll_schedule=rolls,
                                    prefer_handle=True,
                                    score=score,
                                )
                            )
                            break  # keep closest valid standoff for this pose
                    if len(out) >= max_found:
                        out.sort(key=lambda x: -x.score)
                        return out
        out.sort(key=lambda x: -x.score)
        return out

    def sample_cavity_or_overhang_insert(
        self,
        p_rest_site: np.ndarray,
        R_grasp_candidates: list[np.ndarray],
        open_directions: list[np.ndarray],
        finger_held: float,
        q_ref: np.ndarray,
        slide_len: float = 0.14,
        lift_clear: float = 0.025,
        max_found: int = 2,
    ) -> list[ContactCandidate]:
        """Generate whole-corridor-verified insertion candidates for side-opening cavities (t09 microwave)
        and overhang drawers (t03 bottom drawer)."""
        out: list[ContactCandidate] = []
        for d_idx, out_dir in enumerate(open_directions):
            u_in = -_unit(np.array([out_dir[0], out_dir[1], 0.0]))
            p_entry = np.asarray(p_rest_site, float) - u_in * slide_len + np.array([0.0, 0.0, lift_clear])
            screw = ScrewHypothesis(
                kind="prismatic",
                axis=_unit(np.asarray(p_rest_site, float) - p_entry),
                anchor=p_entry,
                span=float(np.linalg.norm(np.asarray(p_rest_site, float) - p_entry)),
                sign=1.0,
            )
            for r_idx, R_c in enumerate(R_grasp_candidates):
                for st_dist in self.cfg["handoff_standoff"]:
                    ver = self.verify_chain(
                        R_c, p_entry, u_in, st_dist, finger_held, finger_held, screw, q_ref, n_random=4
                    )
                    if ver is not None:
                        q0, q_chain, rolls, min_m = ver
                        score = 2.0 * min_m - 0.15 * float(np.linalg.norm(q0 - q_ref))
                        out.append(
                            ContactCandidate(
                                name=f"insert_d{d_idx}_r{r_idx}",
                                mode="cavity_insert",
                                R_handoff=R_c,
                                p_handoff=p_entry - u_in * st_dist,
                                R_contact=R_c,
                                p_contact=p_entry,
                                approach=u_in,
                                finger_open=finger_held,
                                q_handoff=q0,
                                q_chain=q_chain,
                                screw=screw,
                                min_limit_margin_rad=min_m,
                                roll_schedule=rolls,
                                prefer_handle=True,
                                score=score,
                            )
                        )
                        break
                if len(out) >= max_found:
                    out.sort(key=lambda x: -x.score)
                    return out
        out.sort(key=lambda x: -x.score)
        return out


@dataclass
class TrackedEntity:
    oid: str
    name: str
    local_points: np.ndarray      # (N, 3) frozen shape in object-centric frame
    R_world: np.ndarray           # (3, 3)
    p_world: np.ndarray           # (3,)
    status: str = "seen"          # seen | held | released_unverified | verified | lost
    T_hand_to_obj: tuple[np.ndarray, np.ndarray] | None = None  # (R_rel, p_rel) while held

    def world_points(self) -> np.ndarray:
        return self.local_points @ self.R_world.T + self.p_world


class ObjectPermanenceStore:
    """Deterministic SE(3) object permanence store across grasp, transit, contact, and release."""

    def __init__(self):
        self.entities: dict[str, TrackedEntity] = {}

    def register_seen(self, oid: str, name: str, world_pts: np.ndarray):
        pts = np.asarray(world_pts, dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return
        if oid in self.entities and self.entities[oid].status == "held":
            return  # never overwrite a frozen held shape!
        c = pts.mean(axis=0)
        self.entities[oid] = TrackedEntity(
            oid=oid, name=name, local_points=pts - c, R_world=np.eye(3), p_world=c, status="seen"
        )

    def on_grasp(self, oid: str, R_hand: np.ndarray, p_hand: np.ndarray):
        ent = self.entities.get(oid)
        if ent is None:
            return
        # Freeze rigid transform in hand frame: X_w = R_hand * X_h + p_hand
        R_rel = R_hand.T @ ent.R_world
        p_rel = R_hand.T @ (ent.p_world - p_hand)
        ent.T_hand_to_obj = (R_rel, p_rel)
        ent.status = "held"

    def update_held(self, R_hand: np.ndarray, p_hand: np.ndarray):
        for ent in self.entities.values():
            if ent.status == "held" and ent.T_hand_to_obj is not None:
                R_rel, p_rel = ent.T_hand_to_obj
                ent.R_world = R_hand @ R_rel
                ent.p_world = p_hand + R_hand @ p_rel

    def on_release_and_icp(self, oid: str, scene_depth_pts: np.ndarray, max_iters: int = 25) -> dict:
        """Update released object pose from depth cloud around last held pose using trimmed rigid ICP."""
        ent = self.entities.get(oid)
        if ent is None:
            return {"verified": False, "reason": "unknown_oid"}
        ent.T_hand_to_obj = None
        ent.status = "released_unverified"

        pred = ent.world_points()
        lo, hi = pred.min(axis=0) - 0.06, pred.max(axis=0) + 0.06
        S = np.asarray(scene_depth_pts, dtype=float).reshape(-1, 3)
        S = S[np.all((S >= lo) & (S <= hi), axis=1)]
        # Exclude other known objects so stove/plates/other moka pot do not hijack ICP!
        other_pts = [
            e.world_points() for k, e in self.entities.items() if k != oid and len(e.local_points) > 0
        ]
        if len(S) > 0 and other_pts:
            d_oth, _ = cKDTree(np.concatenate(other_pts, axis=0)).query(S, distance_upper_bound=0.006)
            S = S[~np.isfinite(d_oth)]

        if len(S) < 20:
            return {"verified": False, "reason": "insufficient_depth_points", "n_pts": len(S)}

        tree = cKDTree(S)
        X = pred.copy()
        R_acc, t_acc = np.eye(3), np.zeros(3)
        for it in range(max_iters):
            d, idx = tree.query(X, distance_upper_bound=0.035 if it < 5 else 0.018)
            ok = np.isfinite(d)
            if int(ok.sum()) < 15:
                break
            A, B = X[ok], S[idx[ok]]
            ca, cb = A.mean(axis=0), B.mean(axis=0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            det = float(np.sign(np.linalg.det(Vt.T @ U.T)))
            dR = Vt.T @ np.diag([1.0, 1.0, det]) @ U.T
            dt = cb - dR @ ca
            X = X @ dR.T + dt
            R_acc = dR @ R_acc
            t_acc = dR @ t_acc + dt
            if float(np.linalg.norm(dt)) < 1e-4:
                break

        d_final, _ = tree.query(X)
        inlier_frac = float(np.mean(d_final < 0.007))
        if inlier_frac >= 0.45:
            ent.R_world = R_acc @ ent.R_world
            ent.p_world = R_acc @ ent.p_world + t_acc
            ent.status = "verified"
            return {"verified": True, "inlier_frac": round(inlier_frac, 3), "shift_m": round(float(np.linalg.norm(t_acc)), 4)}
        return {"verified": False, "inlier_frac": round(inlier_frac, 3), "reason": "low_inliers"}
