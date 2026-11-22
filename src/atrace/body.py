"""Robot bodies and the controller model behind the TRACE contract (DESIGN §3.3).

TRACE knows no robot: the harness hands it collision spheres rigid in the grip-site frame (hand, link 7, payload)
and answers its arm check by rolling the controller's own redundancy resolution along the path, so the arm is
checked where the absolute-pose OSC will actually put it.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from traceplan.contract import GROUPS, ArmResult, Spheres

from .geometry import rot_err
from .levels import tube

BODY = {"palm": 0.012, "finger": 0.008, "link": 0.02, "payload": 0.015, "payload_r": 0.009, "payload_gap": 0.003}
ROLLOUT = {"lam": 0.03, "null_gain": 0.3, "stride": 8, "key_iters": 2, "iters": 4, "limit_margin": 0.02}
ARM = {"err_p": 0.003, "err_r": 0.03, "at_limit": 0.01, "dq_max": 0.35}


def _cover(pts, spacing, r0):
    """Voxel-downsample at `spacing`: one sphere per occupied voxel at the middle of its points' box, radius = its
    farthest point + r0, so the spheres cover every sample sphere."""
    _, inv = np.unique(np.floor(pts / spacing).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    lo, hi = np.full((n, 3), np.inf), np.full((n, 3), -np.inf)
    np.minimum.at(lo, inv, pts)
    np.maximum.at(hi, inv, pts)
    c = 0.5 * (lo + hi)
    r = np.zeros(n)
    np.maximum.at(r, inv, np.linalg.norm(pts - c[inv], axis=1))
    return c, r + r0


def hand_spheres(rm, finger, cfg=BODY):
    """Palm and finger collision samples at this finger opening plus link 7 (rigid with the hand), as spheres in
    the grip-site frame (groups palm, finger, link)."""
    qf = rm.full_q(np.zeros((1, 7)), finger)
    Rs, ps = rm.site_pose(qf)
    L = (rm.points_world(qf)[0] - ps[0]) @ Rs[0]
    g, rad, body = rm.points.group, rm.points.radius, rm.points.body
    parts = (("palm", g == 1), ("finger", g == 2), ("link", body == rm.body_names.index("robot0_link7")))
    C, R, G = [], [], []
    for name, sel in parts:
        c, r = _cover(L[sel], cfg[name], rad[sel].max())
        C.append(c)
        R.append(r)
        G.append(np.full(len(c), GROUPS.index(name)))
    return Spheres(np.concatenate(C), np.concatenate(R), np.concatenate(G))


def payload_spheres(attached, cfg=BODY):
    """The held object's closed collision model as spheres (group payload). Radius at least 0.9 cm and always the
    voxel's farthest sample plus half the hull-sample spacing, so the surface between samples is covered too."""
    c, r = _cover(attached.pts_site, cfg["payload"], cfg["payload_gap"])
    return Spheres(c, np.maximum(r, cfg["payload_r"]), np.full(len(c), GROUPS.index("payload")))


class ControllerModel:
    """Model of the absolute-pose OSC (robosuite OSC_POSE, position and orientation uncoupled). Per substep it applies
    tau = J_p^T L_p f + J_o^T L_o m + null space + bias, f = kp e_p - kd v, m = kp e_o - kd w, kd = 2 damping sqrt(kp),
    so the grip site closes x'' = A (kp e - kd x') with A = Lambda^-1 blockdiag(L_p, L_o), Lambda = (J M^-1 J^T)^-1,
    and L_p, L_o the inverses of its diagonal blocks. A is not the identity: on the Panda (large rotor armature) its
    eigenvalues span about 0.2-1.8, a slow, lightly damped position/orientation mode that a per-axis lead command
    excites (1 cm off the path at the end of a 20 cm radial move, measured). `predict` is the exact one-step
    discretization with the command held (the contact detector's prediction; its error is the contact signal),
    `command` its inverse along a timed path (the executor's feedforward), `rollout` the redundancy resolution."""

    def __init__(self, rm, kp=150.0, damping=1.0, dt=0.05, cfg=ROLLOUT, kp_null=10.0):
        self.rm, self.kp, self.dt, self.cfg = rm, kp, dt, cfg
        self.kd = 2.0 * damping * np.sqrt(kp)
        # robosuite's OSC null-space posture loop: tau_null = N^T M (kp_null (q_post - q) - 2 sqrt(kp_null) q')
        self.kp_null, self.kd_null = kp_null, 2.0 * np.sqrt(kp_null)
        lim = rm.limits[:7]
        self.lo, self.hi = lim[:, 0] + cfg["limit_margin"], lim[:, 1] - cfg["limit_margin"]

    def coupling(self, q):
        """A (6,6) and the grip-site Jacobian (6,7) at the robot configuration q (9,)."""
        J = self.rm.jacobian(np.asarray(q, float)[None])[0]
        Li = J @ np.linalg.solve(self.rm.mass_matrix(q), J.T)
        B = np.zeros((6, 6))
        B[:3, :3], B[3:, 3:] = np.linalg.inv(Li[:3, :3]), np.linalg.inv(Li[3:, 3:])
        return Li @ B, J

    def predict(self, A, e, v):
        """(e, v) one control step later under a held command: e = pose minus command (position, rotation vector) and
        v = (linear, angular) velocity, both (6,)."""
        F = np.block([[np.zeros((6, 6)), np.eye(6)], [-self.kp * A, -self.kd * A]])
        z = expm(F * self.dt) @ np.r_[e, v]
        return z[:6], z[6:]

    def posture(self, q, qd, qdd):
        """Posture command whose null-space loop tracks q(t): q + (q'' + kd_null q') / kp_null, the loop's inverse.
        Commanding q itself leaves the elbow ~kd_null/kp_null = 0.63 s behind the rolled-out arm the certificate checked."""
        return q + (qdd + self.kd_null * qd) / self.kp_null

    def command(self, A, J, v, qdd):
        """Pose offset (6,) of the command held over one step of a path, from the path's velocity v (6,) and joint
        acceleration qdd (7,) at the step's middle: (A^-1 J qdd + kd v) / kp, the closed loop's inverse (J qdd is the
        site acceleration the joint torques must produce; the per-axis lead x(t + kd/kp) is its decoupled first-order
        part)."""
        return (np.linalg.solve(A, J @ qdd) + self.kd * v) / self.kp

    def _dls(self, Q, R, P, Qref):
        """One damped least-squares step per row toward (R, P), with null-space attraction to Qref."""
        rm, c = self.rm, self.cfg
        qf = rm.full_q(Q)
        Rb, Pb = rm.fk(qf)
        Rs, ps = rm.site_pose(qf, Rb, Pb)
        ep, er = P - ps, rot_err(R, Rs)
        # bounded steps, as CP-5's IK: a far target must not throw the linearization away
        ep = ep * np.minimum(1.0, 0.08 / np.maximum(np.linalg.norm(ep, axis=1, keepdims=True), 1e-9))
        er = er * np.minimum(1.0, 0.4 / np.maximum(np.linalg.norm(er, axis=1, keepdims=True), 1e-9))
        J = rm.jacobian(qf, Rb, Pb)
        Jp = np.swapaxes(J, 1, 2) @ np.linalg.inv(J @ np.swapaxes(J, 1, 2) + c["lam"] ** 2 * np.eye(6))
        dq = (Jp @ np.concatenate([ep, er], 1)[..., None])[..., 0]
        dq += ((np.eye(7) - Jp @ J) @ (c["null_gain"] * (Qref - Q))[..., None])[..., 0]
        return np.clip(Q + dq, self.lo, self.hi)

    def rollout(self, q0, R, p, q_ref):
        """Joint configurations the controller reaches along the pose path: DLS warm-started from the previous
        configuration (sequentially on every stride-th sample, which keeps the branch), interpolated between them
        and refined together. Returns (q (N,7), err_p (N,), err_r (N,))."""
        c, N = self.cfg, len(p)
        keys = np.unique(np.r_[np.arange(0, N, c["stride"]), N - 1])
        q = np.clip(np.asarray(q0, float)[:7], self.lo, self.hi)[None]
        Qk = []
        for k in keys:
            for _ in range(c["key_iters"]):
                q = self._dls(q, R[k:k + 1], p[k:k + 1], q_ref[k:k + 1])
            Qk.append(q[0])
        Qk = np.array(Qk)
        Q = np.stack([np.interp(np.arange(N), keys, Qk[:, j]) for j in range(7)], 1)
        for _ in range(c["iters"]):
            Q = self._dls(Q, R, p, q_ref)
        Rs, ps = self.rm.site_pose(self.rm.full_q(Q))
        return Q, np.linalg.norm(p - ps, axis=1), np.linalg.norm(rot_err(R, Rs), axis=1)


def _zone_need(margin, c, p, zone):
    """Per-sample required clearance with the "do not get closer" rule within `zone` of either end."""
    need = np.full(len(c), margin)
    for end in (0, -1):
        near = np.linalg.norm(p - p[end], axis=1) < zone
        need[near] = np.minimum(need[near], c[end] - 0.001)
    return need


def make_arm_check(h, margins, zone, finger=0.04, cfg=ARM):
    """TRACE's arm check: the controller rollout from the measured joints (reach, joint limits, continuity) plus
    exact whole-arm clearance (CP-5 arm samples, scene and table) against guard + tube(v_min) + perception."""
    cm, rm, ck = h.cm, h.rm, h.checker
    q0 = h.rb.q()[:7].copy()
    arm = rm.points.group == 0
    pr = ck.pr[arm]
    held = (ck.attached.oid,) if ck.attached is not None else ()
    tz = h.scene.table_z
    tv = tube(margins, margins["v_min"])
    m_arm = margins["guard"]["arm"] + tv + margins["perception"]
    m_plane = margins["guard_plane"]["arm"] + tv

    def arm_check(R, p, s, q_ref):
        Q, ep, er = cm.rollout(q0, R, p, q_ref)
        W = rm.points_world(rm.full_q(Q, finger), groups=[0])
        c = (ck.point_clearance(W, held) - pr).min(1)
        cpl = (W[..., 2] - tz - 0.5 * pr).min(1)          # CP-5's plane rule: sample spacing counts half
        lim_gap = np.minimum(Q - cm.lo, cm.hi - Q).min(1)
        unreach = (ep > cfg["err_p"]) | (er > cfg["err_r"])
        at_lim = lim_gap < cfg["at_limit"]
        jump = np.r_[False, np.abs(np.diff(Q, axis=0)).max(1) > cfg["dq_max"]]
        close = (c < _zone_need(m_arm, c, p, zone)) | (cpl < _zone_need(m_plane, cpl, p, zone))
        bad = unreach | at_lim | jump | close
        slack = np.minimum(c - margins["guard"]["arm"] - margins["perception"], cpl - margins["guard_plane"]["arm"])
        if not bad.any():
            return ArmResult(True, Q, None, None, float(min(c.min(), cpl.min())), slack)
        i = int(np.argmax(bad))
        why = ("joint_limit" if at_lim[i] else "unreachable") if unreach[i] else \
            "joint_limit" if at_lim[i] else "discontinuous" if jump[i] else "arm_clearance"
        return ArmResult(False, Q, why, float(s[i]), float(min(c.min(), cpl.min())), slack)
    return arm_check


def make_ik_check(h, finger=0.04):
    """TRACE's ik_check for free ends: CP-5's multi-seed IK."""
    def ik_check(R, p, q_seed=None):
        sols = h.rm.solve_ik(R, p, h.rb.q()[:7] if q_seed is None else q_seed, n_random=8, finger=finger)
        return (True, sols[0]) if len(sols) else (False, None)
    return ik_check
