"""Contact detection and the per-step guard (DESIGN §3.6, §3.7).

ContactDetector: in free motion the measured hand follows the controller model's one-step prediction (the coupled
OSC loop of body.ControllerModel plus the velocity-product acceleration J'q') to within its noise; a contact stops
the hand, so the measured position falls behind the prediction along the predicted step. Measured on the dev scene:
free-motion residuals ~0.01 mm in guarded descents and below 0.5 mm in 0.3-0.4 m/s certified moves, while a blocked
hand falls short by most of its step (2-3 mm at 4-6 cm/s). One step past a low threshold, or two steps past the
noise floor, end a guarded move within a step or two of contact. The second signal is the wrist force (the force
the arm exerts on the hand) growing along the predicted step while the hand falls behind the prediction: a contact
does both within its first step (15-27 N and 0.2-0.5 mm in the dev-scene descents); leaving a jammed contact also
raises the force along the motion (20-30 N measured) but the hand keeps to the prediction (-0.03 mm).

StepGuard: on a certified path, robot and payload clearance on the snapshot's exact geometry against the guard
distances alone (the ladder's bottom rung), with the certificate's "do not get closer" zones, plus joint limits.
"""
from __future__ import annotations

import numpy as np
from traceplan.contract import GROUPS

from .geometry import rot_err

# e_min: residual floor (m), above the measured free-motion residuals (<= 0.38 mm) and far below a blocked step;
# strong: a single step past strong * threshold fires; win: free-motion steps the residual's mean and spread are taken
# over; e_push: how far behind the prediction the hand must be for the force signal to count (m)
DETECT = {"f_thr": 6.0, "e_min": 0.0005, "strong": 2.0, "hits": 2, "k_sigma": 4.0, "n_free": 4, "win": 8,
          "min_step": 1e-4, "h": 1e-3, "e_push": 1e-4}
GUARD = {"joint_margin": 0.01}


class ContactDetector:
    def __init__(self, controller_model, f_thr=DETECT["f_thr"], e_min=DETECT["e_min"], cfg=DETECT):
        self.cm, self.f_thr, self.e_min, self.cfg = controller_model, f_thr, e_min, cfg
        self.reset()

    def reset(self):
        self.prev = None          # (A, R, p, v6, J'q') measured after the previous step
        self.free = []            # residuals of the free-motion steps so far
        self.F_ref = None         # wrist force once the move is under way (absorbs a payload's weight coming on)
        self.hits = 0
        self.k = 0
        self.last = {}

    def update(self, q, qd, cmd, F, F0):
        """One control step: the command cmd = (R, p) was held during it; joints (q, qd) and the wrist force F (world
        frame, arm on hand) were measured after it; F0 is the force at the start of the move. True on contact."""
        c, cm = self.cfg, self.cm
        q, qd = np.asarray(q, float), np.asarray(qd, float)[:7]
        A, J = cm.coupling(q)
        Rs, ps = cm.rm.site_pose(q[None])
        # J'q' by a directional difference of the Jacobian along the joint velocity
        Jh = cm.rm.jacobian((q + c["h"] * np.r_[qd, 0.0, 0.0])[None])[0]
        state = (A, Rs[0], ps[0], J @ qd, (Jh - J) @ qd / c["h"])
        if self.prev is None:                # no measured state before the first step: calibrate from here
            self.prev = state
            return False
        (A0, R0, p0, v0, d0), self.prev = self.prev, state
        self.k += 1
        R_cmd, p_cmd = cmd
        # the controller realizes x'' = A (kp (cmd - x) - kd x') + J'q': the same loop with the command shifted
        sh = np.linalg.solve(A0, d0) / cm.kp
        e, _ = cm.predict(A0, np.r_[p0 - p_cmd, -rot_err(R_cmd, R0)] - sh, v0)
        p_pred = p_cmd + sh[:3] + e[:3]
        step = p_pred - p0
        n = np.linalg.norm(step)
        if self.k == c["n_free"]:
            self.F_ref = F
        if n <= c["min_step"]:               # not expected to move: nothing for a contact to stop
            self.hits = 0
            self.last = {"err": 0.0, "thr": 0.0, "mu": 0.0, "force": 0.0}
            return False
        u = step / n
        err = float((p_pred - ps[0]) @ u)
        free = self.free[-c["win"]:]
        mu, sd = (float(np.mean(free)), float(np.std(free))) if len(free) >= 2 else (0.0, 0.0)
        thr = max(self.e_min, c["k_sigma"] * sd)
        x = err - mu
        self.hits = self.hits + 1 if x > thr else 0
        if x <= thr:
            self.free.append(err)
        push = float((F - (F0 if self.F_ref is None else self.F_ref)) @ u)
        self.last = {"err": err, "thr": thr, "mu": mu, "force": push}
        return x > c["strong"] * thr or self.hits >= c["hits"] or (push > self.f_thr and err > c["e_push"])

    def guard(self, reason, rule=None):
        """A guard with CP-5's signature (robot, F, F0, k) -> reason | None: CP-5's own force rule `rule` is kept
        (its reason string wins), and this detector stops the move with `reason`. Fresh state per guarded move."""
        self.reset()

        def g(robot, F, F0, k):
            why = rule(robot, F, F0, k) if rule is not None else None
            if why:
                return why
            return reason if self.update(robot.q(), robot.joint_vel(), robot.last_goal, F, F0) else None
        return g


class StepGuard:
    """Per-step guard of one certified execution (built from its request, snapshot and timed path)."""

    def __init__(self, h, snap, req, timed, ladder, zone, cfg=GUARD):
        self.rm, self.exact, self.zone, self.cfg = h.rm, snap.exact, zone, cfg
        b = req.bodies
        self.c, self.r, self.g = b.centers, b.radii, b.groups
        self.names = [GROUPS[i] for i in np.unique(self.g)]
        self.arm = self.rm.points.group == 0
        self.pr = h.checker.pr[self.arm]
        self.tz = snap.table_z
        self.guard, self.plane = ladder["guard"], ladder["guard_plane"]
        self.ends = (timed.p[0], timed.p[-1])
        lim = self.rm.limits[:7]
        self.lo, self.hi = lim[:, 0] + cfg["joint_margin"], lim[:, 1] - cfg["joint_margin"]
        # clearance of every group at both ends of the certified path: near an end, only "do not get closer"
        self.at_ends = [self._clear(R, p, q) for R, p, q in ((timed.R[0], timed.p[0], timed.q[0]),
                                                            (timed.R[-1], timed.p[-1], timed.q[-1]))]

    def _clear(self, R, p, q):
        """Group -> (scene clearance, table clearance) at one pose (spheres) and joint configuration (arm)."""
        x = p + self.c @ R.T
        cs = self.exact(x) - self.r
        zs = x[:, 2] - self.tz - self.r
        out = {n: (cs[self.g == GROUPS.index(n)].min(), zs[self.g == GROUPS.index(n)].min()) for n in self.names}
        W = self.rm.points_world(self.rm.full_q(np.asarray(q, float)[None, :7]), groups=[0])[0]
        out["arm"] = ((self.exact(W) - self.pr).min(), (W[:, 2] - self.tz - 0.5 * self.pr).min())
        return out

    def check(self, q):
        """None when the measured configuration is clear, else {group, clearance, need} of the first violation."""
        q = np.asarray(q, float)[:7]
        if np.any(q < self.lo) or np.any(q > self.hi):
            return {"group": "joint_limit", "clearance": float(np.minimum(q - self.lo, self.hi - q).min()), "need": 0.0}
        Rs, ps = self.rm.site_pose(self.rm.full_q(q[None]))
        cur = self._clear(Rs[0], ps[0], q)
        near = [np.linalg.norm(ps[0] - e) < self.zone for e in self.ends]
        for n, (cs, zs) in cur.items():
            for val, need, j in ((cs, self.guard[n], 0), (zs, self.plane[n], 1)):
                for end, at in zip(near, self.at_ends):
                    if end:
                        need = min(need, at[n][j] - 0.001)
                if val < need:
                    return {"group": n if j == 0 else "plane_" + n, "clearance": float(val), "need": float(need)}
        return None
