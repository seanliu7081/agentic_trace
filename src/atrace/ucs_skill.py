"""Unified Contact Stage (UCS) Execution Skill (`UCSExecSkill`, `src/atrace/ucs_skill.py`).

Runs identically in C++ fast training (`UCSExecEnv`), Gate 1 unseen benchmark evaluation, and `agentic_trace`
deployment (`SimRobot`). Uses ONLY real-robot-legal observations:
  - 12-step history of local-frame EEF pose, velocity, Leaky-PLAI compliance error (`p_des - p_meas`),
    tared low-passed wrist F/T wrench, finger gap & rate, and previous action;
  - 7 arm joint angles (normalized) + 7 joint-limit margins (solves blind wrist-limit stalls);
  - 5-mode goal descriptor (`pull_open`, `push_close`, `knob_turn`, `cavity_insert`, `overhang_place`)
    with noisy believed tangent/axis and target travel.

Action (`EXEC_ACT_DIM = 8` in `[-1, 1]`):
  - `a[0:3]`: 3D position delta residual in contact frame `C` (`+-1.0 cm/step`) over online screw guide
  - `a[3:6]`: 3D orientation delta residual in contact frame `C` (`+-4.5 deg/step`, enables cylindrical handle roll)
  - `a[6]`:   variable compliance stiffness `alpha in [0.30, 1.00]` fed to `LeakyPLAICompliantController`
  - `a[7]`:   release gate trigger for `cavity_insert` / `overhang_place` (`< -0.3` initiates release + retreat)

No policy `done` action: premature `done` caused 20-30% of M1/M1E failures. Episode termination is governed
uniformly by measured physical end-stop / travel completion / clean release-retreat or safety limits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import mat_to_rotvec, rotvec_to_mat
from .ucs import LeakyPLAICompliantController, OnlineScrewTracker, _unit

EXEC_MODES = (
    "pull_open",
    "push_close",
    "knob_turn",
    "cavity_insert",
    "overhang_place",
)

EXEC_CFG = {
    "dt": 0.05,                  # R  s: 20 Hz control rate
    "history": 12,               # T  steps of proprioceptive-wrench history (0.6 s)
    "base_dp": 0.011,            # T  m/step: nominal screw translation guide (~15 cm/s)
    "base_dr_deg": 7.0,          # T  deg/step: nominal rotary turn guide
    "res_dp": 0.013,             # T  m/step: learned 3D residual (> base_dp, can reverse or steer 90 deg!)
    "res_dr_deg": 5.0,           # T  deg/step: learned 3D orientation residual (cylindrical roll / tilt)
    "alpha_lo": 0.35,            # T  minimum compliance scale
    "alpha_hi": 1.00,            # T  maximum compliance scale
    "close_steps": 10,           # T  steps to close pads on handle/knob after reaching contact
    "way_in_speed": 0.05,        # T  m/s: guarded approach from <=5.5 cm handoff to contact
    "release_steps": 6,          # T  steps to open fingers at placement support
    "retreat_steps": 10,         # T  steps to back out along reverse entry after release (ensures hand clear & object settled)
    "retreat_dist": 0.075,       # T  m: post-release retreat distance
    "f_support_min": 2.5,        # M  N: upward/axial force confirming payload touched support surface
    "max_steps": 130,            # T  6.5 s max contact execution (well inside <= 10 s stage budget!)
    # Compliant controller overrides per mode family:
    "kp_soft": (75.0, 60.0),     # M  (pos, ori) kp for push_close / cavity_insert / overhang_place (<0.4 mm pen!)
    "kp_firm": (105.0, 80.0),    # M  (pos, ori) kp for pull_open / knob_turn
    "f_cap_soft": 22.0,          # M  N: steady normal force cap (< 0.4 mm pen, overcomes 50 Ns/m drawer damping!)
    "f_cap_firm": 28.0,          # M  N: pulling force cap
}

EXEC_ACT_DIM = 8
EXEC_STEP_DIM = 37
EXEC_HEADER_DIM = 27
EXEC_OBS_DIM = EXEC_CFG["history"] * EXEC_STEP_DIM + EXEC_HEADER_DIM


@dataclass
class ExecGoal:
    mode: str                    # one of EXEC_MODES
    believed_tangent: np.ndarray # unit 3D vector in world: initial pull/push/insert direction
    believed_axis: np.ndarray    # unit 3D vector in world: hinge/knob axis (or upward normal for insert/place)
    believed_anchor: np.ndarray  # 3D world point: knob/door pivot or target rest point (for insert/place)
    target_span: float           # m (slide/insert) or rad (knob/door)
    approach_axis: np.ndarray    # unit 3D vector into contact from handoff
    way_in_m: float = 0.0        # m from handoff to contact (0 if already at contact)

    def __post_init__(self):
        assert self.mode in EXEC_MODES, self.mode
        self.believed_tangent = _unit(self.believed_tangent)
        self.believed_axis = _unit(self.believed_axis)
        self.believed_anchor = np.asarray(self.believed_anchor, dtype=float)
        self.approach_axis = _unit(self.approach_axis)


def neutral_exec_action() -> np.ndarray:
    """Zero residual, mid-high stiffness (`alpha ~ 0.82`), hold gripper (`+0.6`)."""
    a = np.zeros(EXEC_ACT_DIM, dtype=float)
    a[6] = 0.5
    a[7] = 0.6
    return a


class UCSExecSkill:
    """Closed-loop 20 Hz unified contact skill."""

    def __init__(self, arm_limits: np.ndarray, cfg: dict = EXEC_CFG):
        """`arm_limits`: `(7, 2)` array of Panda joint `[lo, hi]` in radians."""
        self.limits = np.asarray(arm_limits, dtype=float)[:7].copy()
        self.q_mid = 0.5 * (self.limits[:, 0] + self.limits[:, 1])
        self.q_half = 0.5 * (self.limits[:, 1] - self.limits[:, 0])
        self.cfg = cfg

    def start(self, io, goal: ExecGoal, max_steps: int | None = None, tare_wrench: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        self.io = io
        self.goal = goal
        self.n_max = int(c["max_steps"] if max_steps is None else min(c["max_steps"], max_steps))
        self.k = 0
        self.stop: str | None = None
        self.released = False
        self.trace_rows: list[np.ndarray] = []

        is_firm = goal.mode in ("pull_open", "knob_turn")
        kp_p, kp_o = c["kp_firm"] if is_firm else c["kp_soft"]
        f_cap = c["f_cap_firm"] if is_firm else c["f_cap_soft"]
        k_est = 6200.0 if is_firm else 3400.0
        plai_cfg = {
            **LeakyPLAICompliantController().cfg,
            "kp_contact_pos": kp_p,
            "kp_contact_ori": kp_o,
            "f_steady_cap": f_cap,
            "f_touch": 6.0 if goal.mode in ("cavity_insert", "overhang_place") else 3.5,
            "k_contact": 6500.0 if is_firm else 4800.0,
            "plai_pos_clamp": 0.028,
            "dp_max": 0.018,
        }
        self.plai = LeakyPLAICompliantController(cfg=plai_cfg)
        self.plai.apply_contact_gains(io, alpha=0.85)

        R0, p0 = io.eef_pose()
        w0 = np.hstack(io.wrench()) if tare_wrench is None else np.asarray(tare_wrench, float)

        # 1. Way-in from <= 5.5 cm handoff to contact (for push_close, pull_open, knob_turn)
        if goal.way_in_m > 0.002 and goal.mode in ("pull_open", "push_close", "knob_turn"):
            grip_in = 1.0 if goal.mode == "push_close" else -1.0
            step_len = c["way_in_speed"] * c["dt"]
            n_in = max(1, int(np.ceil((goal.way_in_m + 0.008) / step_len)))
            p_cur_cmd = p0.copy()
            for _ in range(n_in):
                wm = np.hstack(io.wrench()) - w0
                if float(np.linalg.norm(wm[:3])) > 6.5:
                    break
                p_cur_cmd = p_cur_cmd + goal.approach_axis * step_len
                io.step(R0, p_cur_cmd, grip_in)

        # 2. Close gripper on handle/knob for pull_open & knob_turn (skip if caller already closed and verified hold!)
        R_c, p_c = io.eef_pose()
        if goal.mode in ("pull_open", "knob_turn") and float(io.finger_opening()) > 0.032 and goal.way_in_m > 0.001:
            for _ in range(c["close_steps"]):
                io.step(R_c, p_c, 1.0)
            R_c, p_c = io.eef_pose()

        # Re-anchor contact frame C at the engaged pose (gravity-stabilized local frame)
        self.R_c = R_c.copy()
        self.p_c = p_c.copy()
        self.R, self.p = R_c.copy(), p_c.copy()
        self.gap = float(io.finger_opening())
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.a_prev = neutral_exec_action()
        self.rot_accum = 0.0

        # Tare wrench before loading for push/insert/place; for pull/knob use pre-grasp w0 so handle preload isn't hidden
        w_tare = w0 if goal.mode in ("pull_open", "knob_turn") else np.hstack(io.wrench())
        self.plai.reset_state(self.R_c, self.p_c, w_tare)
        self.screw_tracker = OnlineScrewTracker(goal.believed_tangent)
        self._prev_tang = goal.believed_tangent.copy()

        self.hist = np.zeros((c["history"], EXEC_STEP_DIM), dtype=np.float32)
        self._record_obs(np.zeros(3), np.zeros(3))
        return self.obs()

    def step(self, action: np.ndarray) -> np.ndarray:
        if self.stop is not None:
            return self.obs()

        c, g = self.cfg, self.goal
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        alpha = c["alpha_lo"] + 0.5 * (a[6] + 1.0) * (c["alpha_hi"] - c["alpha_lo"])
        self.plai.apply_contact_gains(self.io, alpha=alpha)

        # 1. Compute base guide increment + policy's 6D residual in contact frame C
        res_dp_w = self.R_c @ (c["res_dp"] * a[:3])
        res_dr_w = self.R_c @ (np.radians(c["res_dr_deg"]) * a[3:6])

        if g.mode == "knob_turn":
            # Nominal rotary guide about believed_axis around believed_anchor (works for BOTH centered knobs & offset levers!)
            dth = np.radians(c["base_dr_deg"])
            Rq = rotvec_to_mat(g.believed_axis * dth)
            p_base_step = (Rq @ (self.p - g.believed_anchor) + g.believed_anchor) - self.p
            dr_w = g.believed_axis * dth + res_dr_w
            dp_w = p_base_step + res_dp_w
        elif g.mode in ("cavity_insert", "overhang_place"):
            # Guide toward believed rest target `believed_anchor`: slide horizontally through the clear slot above
            # the front wall / sill and below the ceiling / overhang, then set down gently once over the floor!
            to_rest = g.believed_anchor - self.p
            horiz = np.array([to_rest[0], to_rest[1], 0.0])
            h_dist = float(np.linalg.norm(horiz))
            slot_z = 0.056 if g.mode == "overhang_place" else 0.020
            if h_dist > 0.022:
                z_hold = (g.believed_anchor[2] + slot_z) - self.p[2]
                dir_guide = _unit(np.array([horiz[0], horiz[1], np.clip(z_hold, -0.02, 0.02)]))
            else:
                dir_guide = _unit(np.array([horiz[0], horiz[1], min(-0.014, to_rest[2] - 0.010)]))
            dp_w = dir_guide * c["base_dp"] + res_dp_w
            dr_w = res_dr_w
        else:
            # `pull_open` or `push_close`: automatically handle BOTH prismatic slides (axis || tangent)
            # AND revolute hinged doors (axis _|_ tangent, curving around believed_anchor)!
            is_hinge_like = abs(float(g.believed_axis @ g.believed_tangent)) < 0.45
            wm_now = np.hstack(self.io.wrench()) - self.plai.w0
            f_now = float(np.linalg.norm(wm_now[:3]))
            if is_hinge_like:
                r_vec = self.p - g.believed_anchor
                r_perp = r_vec - float(r_vec @ g.believed_axis) * g.believed_axis
                rad = max(float(np.linalg.norm(r_perp)), 0.12)
                dth = c["base_dp"] / rad
                # Determine sign along believed_axis that moves self.p in the direction of OnlineScrewTracker.tangent
                v_pos = np.cross(g.believed_axis, r_perp)
                sgn = 1.0 if float(v_pos @ self.screw_tracker.tangent) >= 0.0 else -1.0
                Rq = rotvec_to_mat(g.believed_axis * (sgn * dth))
                arc_step = (Rq @ r_vec + g.believed_anchor) - self.p
                if g.mode == "push_close" and f_now < 2.5:
                    arc_step = arc_step + 0.0035 * g.approach_axis
                dp_w = arc_step + res_dp_w
                # Gently turn wrist (~55% of door angle) so cylindrical handle roll absorbs the rest without q7 limits!
                dr_w = g.believed_axis * (0.55 * sgn * dth) + res_dr_w
            else:
                prev_t = getattr(self, "_prev_tang", g.believed_tangent).copy()
                tang = self.screw_tracker.tangent
                self._prev_tang = tang.copy()
                if g.mode == "push_close" and f_now < 2.5:
                    tang = _unit(0.80 * tang + 0.30 * g.approach_axis)
                dp_w = tang * c["base_dp"] + res_dp_w
                # Purely physical arc compliance: if `pull_open` curves (a hinged door), rotate the hand by the exact
                # tangent rotation `prev_t x tang` so the pads stay square on the swinging handle past 80 deg; on a
                # straight drawer `prev_t x tang == 0`!
                arc_rot = np.cross(prev_t, self.screw_tracker.tangent) if g.mode == "pull_open" else np.zeros(3)
                dr_w = 1.15 * arc_rot + res_dr_w

        # 2. Null-space joint-limit repulsion ONLY when approaching a joint limit (< 0.25 rad margin)
        q_cur = self.io.q()[:7]
        min_m = float(np.minimum(q_cur - self.limits[:, 0], self.limits[:, 1] - q_cur).min())
        q_post = (0.85 * q_cur + 0.15 * self.q_mid) if min_m < 0.25 else q_cur

        # 3. Check placement release gate for cavity_insert / overhang_place
        wm = np.hstack(self.io.wrench())
        f_tared = wm[:3] - self.plai.w0[:3]
        dist_to_anchor_xy = float(np.linalg.norm((self.p - g.believed_anchor)[:2]))
        near_floor = (self.p[2] - g.believed_anchor[2]) < 0.026 and (self.p[2] - g.believed_anchor[2]) > -0.035
        # Upward support force (z > 0 or norm >= f_support_min) ONLY once horizontally inside target zone (< 3.2 cm)!
        support_touched = float(np.linalg.norm(f_tared)) >= c["f_support_min"] and near_floor and dist_to_anchor_xy < 0.032

        if g.mode in ("cavity_insert", "overhang_place") and (
            support_touched or (a[7] < -0.35 and dist_to_anchor_xy < 0.032 and near_floor)
        ):
            self._release_and_retreat(q_post)
            self.a_prev = a
            return self.obs()

        # 4. Step through Leaky-PLAI compliant wrapper
        R_cmd, p_cmd, plai_info = self.plai.compute_command(dp_w, dr_w, alpha, self.R, self.p, self.v, wm)
        self.io.step(R_cmd, p_cmd, 1.0, posture=q_post)
        self.k += 1
        self.a_prev = a

        # 5. Sense post-step state & check measured physical completion
        R_new, p_new = self.io.eef_pose()
        self.v = (p_new - self.p) / c["dt"]
        self.w = mat_to_rotvec(R_new @ self.R.T) / c["dt"]
        self.rot_accum += float(self.w @ g.believed_axis) * c["dt"]
        self.R, self.p = R_new, p_new

        wm_after = np.hstack(self.io.wrench())
        f_after = wm_after[:3] - self.plai.w0[:3]
        tr_info = self.screw_tracker.update(self.p, self.v, f_after)

        pos_err = self.plai.p_des - self.p
        rot_err = mat_to_rotvec(self.plai.R_des @ self.R.T)
        self._record_obs(pos_err, rot_err)

        # Termination rules (measured physical quantities, identical in training & deployment!)
        if plai_info["force_stop"]:
            # Safe unload hold
            for _ in range(3):
                self.io.step(self.R, self.p, 1.0 if g.mode != "push_close" else 1.0)
            self.stop = "force"
        elif g.mode == "push_close" and tr_info["at_endstop"] and self.k >= 8:
            self.stop = "endstop"
        elif g.mode == "pull_open" and (
            float(np.linalg.norm(self.p - self.p_c)) >= 1.18 * g.target_span or (tr_info["at_endstop"] and self.k >= 20)
        ):
            self.stop = "reached_span"
        elif g.mode == "knob_turn" and (
            abs(self.rot_accum) >= 1.38 * g.target_span or (tr_info["at_endstop"] and abs(self.rot_accum) >= 0.45)
        ):
            self.stop = "reached_span"
        elif self.k >= self.n_max:
            if g.mode in ("cavity_insert", "overhang_place") and dist_to_anchor_xy < 0.065:
                self._release_and_retreat(q_post)
            else:
                self.stop = "time"

        return self.obs()

    def _release_and_retreat(self, q_post: np.ndarray):
        c, g = self.cfg, self.goal
        R_rel, p_rel = self.io.eef_pose()
        for _ in range(c["release_steps"]):
            self.io.step(R_rel, p_rel, -1.0, posture=q_post)
            self.k += 1
        # Retreat gently upward/outward along -believed_tangent so hand leaves object undisturbed
        ret_dir = _unit(-0.75 * g.believed_tangent + np.array([0.0, 0.0, 0.35]))
        for j in range(1, c["retreat_steps"] + 1):
            p_ret = p_rel + ret_dir * (c["retreat_dist"] * j / c["retreat_steps"])
            self.io.step(R_rel, p_ret, -1.0, posture=q_post)
            self.k += 1
        self.R, self.p = self.io.eef_pose()
        self.released = True
        self.stop = "released"

    def _record_obs(self, pos_err_w: np.ndarray, rot_err_w: np.ndarray):
        Rc = self.R_c
        gap_now = float(self.io.finger_opening())
        gap_rate = (gap_now - self.gap) / self.cfg["dt"] if self.k > 0 else 0.0
        self.gap = gap_now

        wm = np.hstack(self.io.wrench()) - self.plai.w0
        Rr = Rc.T @ self.R
        row = np.hstack([
            5.0 * (Rc.T @ (self.p - self.p_c)),
            Rr[:, 0],
            Rr[:, 1],
            5.0 * (Rc.T @ self.v),
            0.5 * (Rc.T @ self.w),
            100.0 * (Rc.T @ pos_err_w),
            5.0 * (Rc.T @ rot_err_w),
            (Rc.T @ wm[:3]) / 10.0,
            Rc.T @ wm[3:6],
            10.0 * gap_now,
            5.0 * gap_rate,
            self.a_prev,
        ]).astype(np.float32)
        self.hist = np.vstack([self.hist[1:], row])
        self.trace_rows.append(row)

    def obs(self) -> np.ndarray:
        q7 = self.io.q()[:7]
        q_norm = (q7 - self.q_mid) / np.maximum(self.q_half, 1e-6)
        q_margin = np.minimum(q7 - self.limits[:, 0], self.limits[:, 1] - q7)
        Rc = self.R_c
        g = self.goal
        header = np.hstack([
            q_norm,
            q_margin,
            1.0 - self.k / max(1, self.n_max),
            (np.array(EXEC_MODES) == g.mode).astype(float),
            Rc.T @ self.screw_tracker.tangent,
            Rc.T @ g.believed_axis,
            g.target_span,
        ]).astype(np.float32)
        return np.concatenate([self.hist.ravel(), header]).astype(np.float32)
