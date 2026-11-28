"""S2 motion: whole-body clearance checks, joint-space planning, timing and execution.

Collision model: every moving robot collision geom is sampled (arm 1.6 cm,
palm 0.8 cm, fingers 0.4 cm) and checked against the voxel ESDF; an attached
payload is checked the same way. Wherever the voxel field reports less than
EXACT_BELOW of clearance, the distance is recomputed exactly (NearField:
observed surface points and unknown-space cubes), so near-contact margins mean
millimetres, not voxels. Execution streams absolute EEF poses (with a velocity
lead that cancels the OSC lag) and the null-space posture of the planned
joint path, while monitoring wrist force and tracking.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .geometry import rot_err
from .perception import NearField

TAU_LEAD = 2.0 * np.sqrt(150.0) / 150.0      # OSC kd/kp for kp=150, critically damped
DT = 0.05
# voxel clearance below which distances are recomputed exactly: up to the largest sphere (link, 23.6 mm) plus TRACE's
# largest margin (22.55 mm), so the conservative voxel value (up to ~1 voxel short) never decides a safety term
EXACT_BELOW = 0.06


@dataclass
class Attached:
    oid: int
    pts_site: np.ndarray        # closed payload collision model (observed points + hull surface), grip-site frame
    bottom_site: float          # lowest payload point along world -z at grasp (site frame z)
    model_site: np.ndarray | None = None   # the observed points alone (the object's model after release)
    up_site: np.ndarray | None = None      # the object's up (world +z at grasp) in the grip-site frame (tilt terms)


class Checker:
    def __init__(self, rm, scene, objects=None):
        self.rm = rm
        self.scene = scene
        self.objects = objects if objects is not None else {}
        self.attached: Attached | None = None
        self.pg = rm.points.group
        self.pr = rm.points.radius
        self.n_checks = 0
        self.near = NearField(lambda: self.scene, self.objects)
        self.free_near = None          # [(points, radius)]: unobserved space hugging these belongs to them

    def point_clearance(self, P, exclude=(), ignore_box=None, exact=True, unknown=True):
        """Clearance (m) of world points (...,3) to the scene (no point radius subtracted); unknown=False: unobserved
        space is not an obstacle."""
        P = np.asarray(P, float)
        shp = P.shape[:-1]
        flat = P.reshape(-1, 3)
        c = self.scene.clearance(flat, tuple(exclude), include_unknown=unknown)
        if exact:
            m = (c < EXACT_BELOW) & self.scene.inside(self.scene.index(flat))   # outside the map stays blocked
            if m.any():
                c = c.astype(float)
                c[m] = self.near.distance(flat[m], exclude, ignore_box, free_near=self.free_near, unknown=unknown)
        return c.reshape(shp)

    def world_points(self, Q, finger=0.04):
        Q = np.atleast_2d(Q)
        qf = self.rm.full_q(Q[:, :7], finger)
        R, P = self.rm.fk(qf)
        out = []
        for bi, pts in enumerate(self.rm.points.local):
            if len(pts):
                out.append(P[:, bi, None, :] + np.einsum("nij,kj->nki", R[:, bi], pts))
        W = np.concatenate(out, axis=1)
        att = None
        if self.attached is not None:
            Rs, ps = self.rm.site_pose(qf, R, P)
            att = ps[:, None, :] + np.einsum("nij,kj->nki", Rs, self.attached.pts_site)
        return W, att

    def clearance(self, Q, exclude=(), finger=0.04, dist=None, ignore_box=None, exact=True, unknown=True):
        """Per-config min clearance per group: dict(arm, palm, finger, payload) arrays (N,)."""
        self.n_checks += len(np.atleast_2d(Q))
        W, att = self.world_points(Q, finger)
        excl = tuple(exclude) + ((self.attached.oid,) if self.attached is not None else ())
        if exact and dist is None:
            c = self.point_clearance(W, excl, ignore_box, unknown=unknown) - self.pr[None, :]
        else:
            c = self.scene.clearance(W, excl, dist, include_unknown=unknown) - self.pr[None, :]
        tz = self.scene.table_z
        # the table is a plane: sample spacing matters only half as much against it as against points
        cp = W[..., 2] - tz - 0.5 * self.pr[None, :] if tz is not None else None
        res = {}
        for g, name in ((0, "arm"), (1, "palm"), (2, "finger")):
            sel = self.pg == g
            res[name] = c[:, sel].min(axis=1) if sel.any() else np.full(len(c), 1.0)
            if cp is not None and sel.any():
                res["plane_" + name] = cp[:, sel].min(axis=1)
        if att is not None:
            if exact and dist is None:
                ca = self.point_clearance(att, excl, ignore_box, unknown=unknown) - 0.004
            else:
                ca = self.scene.clearance(att, excl, dist, include_unknown=unknown) - 0.004
            res["payload"] = ca.min(axis=1)
            if tz is not None:
                res["plane_payload"] = (att[..., 2] - tz).min(axis=1)
        return res

    def ok(self, Q, margins, exclude=(), finger=0.04, dist=None, ignore_box=None, unknown=True):
        c = self.clearance(Q, exclude, finger, dist, ignore_box, unknown=unknown)
        good = np.ones(len(np.atleast_2d(Q)), bool)
        for k, v in c.items():
            good &= v >= margins.get(k, margins.get("default", 0.02))
        return good, c

    def fine(self, Q, oids, finger=0.04):
        """Exact signed clearance to specific objects: dict group -> (N,) min distance."""
        W, att = self.world_points(Q, finger)
        N = W.shape[0]
        out = {"arm": np.full(N, np.inf), "palm": np.full(N, np.inf), "finger": np.full(N, np.inf),
               "payload": np.full(N, np.inf)}
        for oid in oids:
            obj = self.objects.get(oid)
            if obj is None or len(obj.points) == 0:
                continue
            flat = W.reshape(-1, 3)
            d = obj.signed_clearance(flat).reshape(N, -1) - self.pr[None, :]
            for g, name in ((0, "arm"), (1, "palm"), (2, "finger")):
                sel = self.pg == g
                if sel.any():
                    out[name] = np.minimum(out[name], d[:, sel].min(axis=1))
            if att is not None and (self.attached is None or self.attached.oid != oid):
                da = obj.signed_clearance(att.reshape(-1, 3)).reshape(N, -1)
                out["payload"] = np.minimum(out["payload"], da.min(axis=1))
        return out


def densify(path, max_step=0.03, return_index=False):
    path = np.asarray(path, float)
    out = [path[0]]
    idx = [0]
    for a, b in zip(path[:-1], path[1:]):
        n = int(np.ceil(np.abs(b - a).max() / max_step))
        for k in range(1, max(n, 1) + 1):
            out.append(a + (b - a) * k / max(n, 1))
        idx.append(len(out) - 1)
    if return_index:
        return np.array(out), idx
    return np.array(out)


class Planner:
    def __init__(self, checker, rng=None, log=None):
        self.ck = checker
        self.rm = checker.rm
        self.rng = rng or np.random.default_rng(0)
        self.log = log or (lambda *a, **k: None)
        lim = self.rm.limits[:7]
        self.lo = lim[:, 0] + 0.03
        self.hi = lim[:, 1] - 0.03
        self.relax = None
        self.goal_relax = None

    def path_ok(self, path, margins, exclude=(), finger=0.04, max_step=0.03):
        d = densify(path, max_step)
        good, c = self.ck.ok(d, margins, exclude, finger)
        if self.relax is not None and not good.all():
            # configs near the start config may keep the (lower) clearance they already have
            q_ref, c_ref = self.relax
            near = np.abs(d - q_ref).max(axis=1) < 0.12
            if near.any():
                ok2 = np.ones(len(d), bool)
                for k, v in c.items():
                    need = min(margins.get(k, margins.get("default", 0.02)), c_ref.get(k, 1.0) - 0.004)
                    ok2 &= v >= need
                good = good | (near & ok2)
        if self.goal_relax is not None and not good.all():
            # the last few centimetres into a near-contact goal use near-contact margins
            q_goals, m_near = self.goal_relax
            near = np.zeros(len(d), bool)
            for qg in q_goals:
                near |= np.abs(d - qg).max(axis=1) < 0.15
            if near.any():
                ok3 = np.ones(len(d), bool)
                for k, v in c.items():
                    ok3 &= v >= m_near.get(k, 0.004)
                good = good | (near & ok3)
        return bool(good.all()), c, d

    def set_relax(self, q0, margins=None, exclude=(), finger=0.04):
        """Allow motions that start inside a low-clearance region to leave it."""
        if q0 is None:
            self.relax = None
            return
        c = self.ck.clearance(np.asarray(q0, float)[None, :7], exclude, finger)
        self.relax = (np.asarray(q0, float)[:7], {k: float(v[0]) for k, v in c.items()})

    def edge_ok(self, a, b, margins, exclude, finger):
        return self.path_ok([a, b], margins, exclude, finger)[0]

    def rrt_connect(self, q0, goals, margins, exclude=(), finger=0.04, max_iter=2500, step=0.25,
                    time_budget=6.0):
        t0 = time.time()
        Ta = [np.asarray(q0, float)]
        Pa = [-1]
        Tb = [np.asarray(g, float) for g in goals]
        Pb = [-1] * len(Tb)
        trees = [(Ta, Pa), (Tb, Pb)]
        start_side = 0

        def nearest(T, q):
            A = np.array(T)
            return int(np.argmin(np.abs(A - q).max(axis=1)))

        def extend(T, P, q, max_steps=1):
            i = nearest(T, q)
            a = T[i]
            d = q - a
            n = np.abs(d).max()
            new = a + d * min(1.0, step / max(n, 1e-9))
            if self.edge_ok(a, new, margins, exclude, finger):
                T.append(new)
                P.append(i)
                return len(T) - 1, np.abs(new - q).max() < 1e-6
            return None, False

        def connect(T, P, q):
            last = None
            while True:
                idx, reached = extend(T, P, q)
                if idx is None:
                    return last, False
                last = idx
                if reached:
                    return last, True

        for it in range(max_iter):
            if time.time() - t0 > time_budget:
                break
            qr = self.rng.uniform(self.lo, self.hi) if self.rng.random() > 0.1 else np.array(trees[1][0][0])
            (TA, PA), (TB, PB) = trees
            ia, _ = extend(TA, PA, qr)
            if ia is not None:
                ib, ok = connect(TB, PB, TA[ia])
                if ok:
                    def trace(T, P, i):
                        out = []
                        while i != -1:
                            out.append(T[i])
                            i = P[i]
                        return out
                    pa = trace(TA, PA, ia)[::-1]
                    pb = trace(TB, PB, ib)
                    path = pa + pb[1:] if np.allclose(pa[-1], pb[0]) else pa + pb
                    if start_side == 1:
                        path = path[::-1]
                    self.log("rrt", iters=it, nodes=len(Ta) + len(Tb), seconds=round(time.time() - t0, 2))
                    return np.array(path)
            trees = trees[::-1]
            start_side = 1 - start_side
        self.log("rrt_fail", iters=it, seconds=round(time.time() - t0, 2))
        return None

    def shortcut(self, path, margins, exclude=(), finger=0.04, iters=40):
        path = [np.asarray(p) for p in path]
        for _ in range(iters):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.edge_ok(path[i], path[j], margins, exclude, finger):
                path = path[:i + 1] + path[j:]
        return np.array(path)

    def plan(self, q0, goals, margins, exclude=(), finger=0.04, lift_z=None, time_budget=6.0, allow_rrt=True):
        """Joint path from q0 to the first reachable goal (list of configs)."""
        q0 = np.asarray(q0, float)[:7]
        goals = [np.asarray(g, float)[:7] for g in goals]
        if not goals:
            return None, "no_goal"
        goals.sort(key=lambda g: np.abs(g - q0).max())
        for g in goals[:6]:
            ok, c, _ = self.path_ok([q0, g], margins, exclude, finger)
            if ok:
                return np.array([q0, g]), "direct"
        if lift_z is not None:
            for lz in (lift_z if isinstance(lift_z, (list, tuple)) else [lift_z]):
                for g in goals[:4]:
                    via = self.lift_route(q0, g, lz, margins, exclude, finger)
                    if via is not None:
                        return via, "lift"
        if not allow_rrt:
            return None, "no_simple_route"
        path = self.rrt_connect(q0, goals[:8], margins, exclude, finger, time_budget=time_budget)
        if path is None:
            return None, "rrt_fail"
        path = self.shortcut(path, margins, exclude, finger)
        return path, "rrt"

    def lift_route(self, q0, g, lift_z, margins, exclude, finger):
        """Up-over-down route: raise EEF to lift_z, move above the goal, descend."""
        R0, p0 = self.rm.site_pose(self.rm.full_q(q0[None], finger))
        Rg, pg = self.rm.site_pose(self.rm.full_q(g[None], finger))
        R0, p0, Rg, pg = R0[0], p0[0], Rg[0], pg[0]
        za = max(lift_z, p0[2])
        zb = max(lift_z, pg[2])
        a = self.rm.solve_ik(R0, np.array([p0[0], p0[1], za]), q0, n_random=8, finger=finger)
        if len(a) == 0:
            return None
        b = self.rm.solve_ik(Rg, np.array([pg[0], pg[1], zb]), g, n_random=8, finger=finger)
        if len(b) == 0:
            return None
        path = [q0, a[0], b[0], g]
        ok, _, _ = self.path_ok(path, margins, exclude, finger)
        return np.array(path) if ok else None


def time_parameterize(rm, path, vmax=1.2, amax=2.5, eef_v=0.25, eef_w=1.2, finger=0.04, eef_a=0.5):
    """Return (times (K,), dense path (K,7)) with velocity/acceleration limits."""
    path = np.asarray(path, float)
    Q, wp = densify(path, 0.01, return_index=True)
    if len(Q) < 2:
        return np.array([0.0]), Q
    Rs, ps = rm.site_pose(rm.full_q(Q, finger))
    dq = np.abs(np.diff(Q, axis=0)).max(axis=1)
    dp = np.linalg.norm(np.diff(ps, axis=0), axis=1)
    dth = np.linalg.norm(rot_err(Rs[1:], Rs[:-1]), axis=1)
    dt_min = np.maximum.reduce([dq / vmax, dp / eef_v, dth / eef_w, np.full(len(dq), 1e-4)])
    vmax_s = 1.0 / dt_min                       # segments per second
    a_s = amax / np.maximum(dq, 1e-6)           # segments per s^2
    a_s = np.minimum(a_s, eef_a / np.maximum(dp, 1e-6))   # OSC (kp 150) tracks ~a/kp: keep it small
    n = len(dq)
    vcap = np.full(n + 1, np.inf)
    for j in range(1, len(path) - 1):
        a_dir = path[j] - path[j - 1]
        b_dir = path[j + 1] - path[j]
        na, nb = np.linalg.norm(a_dir), np.linalg.norm(b_dir)
        if na < 1e-6 or nb < 1e-6:
            continue
        cosang = float(a_dir @ b_dir / (na * nb))
        k = wp[j]
        if cosang < 0.5:
            vcap[k] = 0.0
        elif cosang < 0.92:
            vcap[k] = 0.3 * min(vmax_s[max(k - 1, 0)], vmax_s[min(k, n - 1)])
    v = np.zeros(n + 1)
    for i in range(n):
        v[i + 1] = min(vmax_s[i], np.sqrt(v[i] ** 2 + 2 * a_s[i]), vcap[i + 1])
    v[n] = 0.0
    for i in range(n - 1, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_s[i]), vmax_s[i] if i < n else np.inf)
    seg_t = 2.0 / np.maximum(v[:-1] + v[1:], 1e-6)
    seg_t = np.minimum(seg_t, 2.0 / np.sqrt(2 * a_s))  # start/stop segments
    seg_t = np.maximum(seg_t, dt_min)
    t = np.concatenate([[0.0], np.cumsum(seg_t)])
    return t, Q


def path_distance(p, P):
    """Distance from the point p to the polyline P (n,3), n >= 1."""
    if len(P) == 1:
        return float(np.linalg.norm(p - P[0]))
    a, s = P[:-1], np.diff(P, axis=0)
    w = np.clip(((p - a) * s).sum(1) / np.maximum((s * s).sum(1), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + w[:, None] * s - p, axis=1).min())


@dataclass
class ExecResult:
    status: str                 # done | contact | tracking | dropped | stopped
    steps: int = 0
    info: dict = field(default_factory=dict)


class Executor:
    def __init__(self, robot, rm, log=None):
        self.robot = robot
        self.rm = rm
        self.log = log or (lambda *a, **k: None)

    def _q_at(self, t, times, Q):
        if t <= times[0]:
            return Q[0]
        if t >= times[-1]:
            return Q[-1]
        k = int(np.searchsorted(times, t) - 1)
        a = (t - times[k]) / max(times[k + 1] - times[k], 1e-9)
        return Q[k] + a * (Q[k + 1] - Q[k])

    def run_path(self, path, grip=None, finger=0.04, vmax=1.2, eef_v=0.25, force_limit=25.0,
                 guard=None, settle=True, max_extra=40, track_limit=0.05, drop_check=None):
        """Execute a joint path. guard(robot, F, F0, step) -> reason string to stop early or None."""
        amax = 2.5 if drop_check is None else 1.2      # gentler when carrying something
        eef_a = 0.25 if (guard is not None or eef_v <= 0.08) else 0.5
        times, Q = time_parameterize(self.rm, path, vmax=vmax, amax=amax, eef_v=eef_v, finger=finger, eef_a=eef_a)
        if grip is not None:
            self.robot.grip_cmd = grip
        F0 = None if getattr(self.robot, "sensorless", False) else self.robot.wrench()[0]
        total = times[-1]
        nsteps = int(np.ceil(total / DT))
        k = 0
        steps = 0
        while k <= nsteps:
            t = k * DT
            q_cmd = self._q_at(t + TAU_LEAD, times, Q)
            Rc, pc = self.rm.site_pose(self.rm.full_q(q_cmd[None], finger))
            self.robot.step(Rc[0], pc[0], posture=q_cmd)
            steps += 1
            F = None if F0 is None else self.robot.wrench()[0]
            q_plan = self._q_at(t + DT, times, Q)
            _, pp = self.rm.site_pose(self.rm.full_q(q_plan[None], finger))
            R_act, p_act = self.robot.eef_pose()
            terr = float(np.linalg.norm(p_act - pp[0]))
            if guard is not None:
                why = guard(self.robot, F, F0, steps)
                if why:
                    self._stop()
                    return ExecResult(why, steps, {"t": t, "track_err": terr, "force": None if F is None else F.tolist()})
            if F is not None and np.linalg.norm(F - F0) > force_limit and k > 2:
                self._stop()
                return ExecResult("contact", steps, {"t": t, "force": (F - F0).tolist(), "track_err": terr})
            if terr > track_limit and k > 4:
                self._stop()
                return ExecResult("tracking", steps, {"t": t, "track_err": terr})
            if drop_check is not None and drop_check(self.robot):
                return ExecResult("dropped", steps, {"t": t})
            k += 1
        if settle:
            Rg, pg = self.rm.site_pose(self.rm.full_q(Q[-1][None], finger))
            for j in range(max_extra):
                R_act, p_act = self.robot.eef_pose()
                e = np.linalg.norm(p_act - pg[0])
                v = np.abs(self.robot.joint_vel()[:7]).max()
                if e < 0.004 and v < 0.05:
                    break
                self.robot.step(Rg[0], pg[0], posture=Q[-1])
                steps += 1
        return ExecResult("done", steps, {"duration": float(total)})

    def run_timed(self, timed, cm, grip=None, finger=0.04, guard=None, detector=None, drop_check=None, settle=True,
                  max_extra=40, settle_tol=0.015, window=1):
        """Execute a certified TRACE trajectory (contract Timed, on the control grid) with the controller model `cm`.
        Step k holds the model's inverse command for the path over (t_k, t_k+1): the pose at the step's middle offset
        by cm.command (feedforward through the OSC's position/orientation coupling, which a plain lead leaves to ring
        at the end of every move), with the posture feedforward cm.posture (the null-space loop's inverse). After the step it checks: the
        distance from the measured position to the path within `window` samples of t_k+1 against tube[k+1]
        ("tube_exit"; a lag along the path is not an exit, a blocked hand leaves the window), the per-step guard
        ("guard_stop"), the contact detector ("contact", a CP-5-signature guard), the drop check ("dropped").
        Settles like CP-5: error < 4 mm and slow, or 3 slow steps within the standoff tolerance, or max_extra steps."""
        from scipy.spatial.transform import Rotation
        if grip is not None:
            self.robot.grip_cmd = grip
        t, P, Q, tube = timed.t, timed.p, timed.q, timed.tube
        K = len(t)
        F0 = None if getattr(self.robot, "sensorless", False) else self.robot.wrench()[0]
        first = self.robot.steps + 1
        info = {"first_step": first, "ratio_max": 0.0, "err_max": 0.0}

        def stop(status, k, err, **kw):
            self._stop()
            info.update(step=k, last_step=self.robot.steps, tracking_err=round(err, 4), tube=round(float(tube[k]), 4),
                        **kw)
            return ExecResult(status, self.robot.steps - first + 1, info)

        if K > 1:
            dt = t[1] - t[0]
            rot = Rotation.from_matrix(timed.R)
            rel = (rot[1:] * rot[:-1].inv()).as_rotvec()             # world-frame rotation over each step
            Rm = Rotation.from_rotvec(0.5 * rel) * rot[:-1]           # pose at each step's middle
            Pm = 0.5 * (P[1:] + P[:-1])
            V = np.c_[np.diff(P, axis=0), rel] / dt                    # (K-1,6) velocity over each step
            Qp = np.r_[Q[:1], Q, Q[-1:]]                               # at rest before and after
            qdd = (Qp[2:] - 2.0 * Qp[1:-1] + Qp[:-2]) / dt ** 2
            qdd = 0.5 * (qdd[1:] + qdd[:-1])                           # (K-1,7) over each step
            for k in range(K):
                if k < K - 1:
                    off = cm.command(*cm.coupling(self.robot.q()), V[k], qdd[k])
                    R_c, p_c = (Rotation.from_rotvec(off[3:]) * Rm[k]).as_matrix(), Pm[k] + off[:3]
                else:
                    R_c, p_c = timed.R[-1], P[-1]
                if k < K - 1:     # null-space feedforward: the elbow follows the rolled-out arm, not ~0.6 s behind it
                    post = cm.posture(0.5 * (Q[k] + Q[k + 1]), (Q[k + 1] - Q[k]) / dt, qdd[k])
                else:
                    post = Q[-1]
                self.robot.step(R_c, p_c, posture=post)
                k1 = min(k + 1, K - 1)
                err = path_distance(self.robot.eef_pose()[1], P[max(k1 - window, 0):k1 + window + 1])
                info["ratio_max"] = max(info["ratio_max"], err / tube[k1])
                info["err_max"] = max(info["err_max"], err)
                if err > tube[k1]:
                    return stop("tube_exit", k1, err)
                if guard is not None:
                    bad = guard.check(self.robot.q())
                    if bad is not None:
                        return stop("guard_stop", k1, err, **bad)
                if detector is not None:
                    F = None if F0 is None else self.robot.wrench()[0]
                    if detector(self.robot, F, F0, k + 1):
                        return stop("contact", k1, err, force=[] if F is None else (F - F0).tolist())
                if drop_check is not None and drop_check(self.robot):
                    info.update(step=k1, last_step=self.robot.steps)
                    return ExecResult("dropped", self.robot.steps - first + 1, info)
        if settle:
            Rg, pg, qg = timed.R[-1], P[-1], Q[-1]
            stable = 0
            for _ in range(max_extra):
                e = np.linalg.norm(self.robot.eef_pose()[1] - pg)
                slow = np.abs(self.robot.joint_vel()[:7]).max() < 0.05
                if e < 0.004 and slow:
                    break
                stable = stable + 1 if (e < settle_tol and slow) else 0
                if stable >= 3:
                    break
                self.robot.step(Rg, pg, posture=qg)
        info.update(last_step=self.robot.steps, duration=float(t[-1]),
                    final_err=round(float(np.linalg.norm(self.robot.eef_pose()[1] - P[-1])), 4))
        return ExecResult("done", self.robot.steps - first + 1, info)

    def _stop(self):
        """Hold the current measured pose (stop moving)."""
        R, p = self.robot.eef_pose()
        q = self.robot.q()[:7]
        for _ in range(3):
            self.robot.step(R, p, posture=q)

    def gripper_to(self, opening, max_steps=30):
        """Open (or close) until the finger opening reaches `opening`, then hold the fingers there."""
        R, p = self.robot.last_goal
        cmd = -1.0 if opening > self.robot.finger_opening() else 1.0
        for i in range(max_steps):
            o = self.robot.finger_opening()
            if (cmd < 0 and o >= opening) or (cmd > 0 and o <= opening):
                break
            self.robot.step(R, p, grip=cmd)
        for _ in range(3):
            self.robot.step(R, p, grip=0.0)
        return self.robot.finger_opening()

    def close_on(self, max_steps=14, blocked=0.0015):
        """Close one finger-target step (~5 mm per finger) at a time and stop commanding as soon as the
        fingers are blocked: the target then sits at most one step inside contact, so the pads hold
        firmly without being driven into the object at the actuator's force limit."""
        R, p = self.robot.last_goal
        prev = self.robot.finger_opening()
        moved = False
        for i in range(max_steps):
            self.robot.step(R, p, grip=1.0)
            o = self.robot.finger_opening()
            if prev - o >= blocked:
                moved = True
            elif moved or i >= 2:
                break
            prev = o
        for _ in range(3):
            self.robot.step(R, p, grip=0.0)
        return self.robot.finger_opening()

    def soften_grip(self, squeeze=0.008):
        """After a close: back the finger position target off to `squeeze` inside contact.

        The Panda fingers are position servos (kp 1000, 20 N limit); a fully closed target
        squeezes at the force limit. A target just inside contact holds with ~kp*squeeze.
        The command integrator moves the target 5 mm per control step (robosuite speed 0.01).
        """
        g = self.robot.robot.gripper
        a1 = float(g.current_action[0])
        f_c = float(self.robot.q()[7])
        a_target = (f_c - squeeze - 0.02) / 0.02
        k = int(np.floor((a_target - a1) / 0.25))
        R, p = self.robot.last_goal
        for _ in range(max(0, k)):
            self.robot.step(R, p, grip=-1.0)
        for _ in range(3):
            self.robot.step(R, p, grip=0.0)
        return self.robot.finger_opening()

    def close_gently(self, expected_width, max_steps=40):
        """Close fast until near the expected width, then pulse the finger target (≈1.7 mm/step)
        so the pads meet the object slowly; stop as soon as the opening stops changing."""
        R, p = self.robot.last_goal
        for _ in range(25):
            if self.robot.finger_opening() <= expected_width + 0.014:
                break
            self.robot.step(R, p, grip=1.0)
        prev = self.robot.finger_opening()
        stable = 0
        for i in range(max_steps):
            self.robot.step(R, p, grip=1.0 if i % 3 == 0 else 0.0)
            o = self.robot.finger_opening()
            stable = stable + 1 if abs(o - prev) < 2e-4 else 0
            prev = o
            if stable >= 4 or o < 0.006:
                break
        return prev

    def gripper(self, cmd, max_steps=30, min_steps=4, tol=2e-4):
        """Actuate gripper until the finger opening is stable. Returns final opening (m)."""
        R, p = self.robot.last_goal
        prev = self.robot.finger_opening()
        stable = 0
        for i in range(max_steps):
            self.robot.step(R, p, grip=cmd)
            o = self.robot.finger_opening()
            stable = stable + 1 if abs(o - prev) < tol else 0
            prev = o
            if i + 1 >= min_steps and stable >= 3:
                break
        return prev
