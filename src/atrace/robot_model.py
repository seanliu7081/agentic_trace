"""Robot-only kinematic model extracted from the simulator's robot description.

Only the robot's own bodies (robot0_*, gripper0_*) are read: link transforms,
joint axes/limits, collision meshes, the grip site and the wrist camera mount.
This is the information a real robot has in its URDF; no scene state is used.

Batched numpy FK / Jacobian / multi-seed damped-least-squares IK, plus
surface point samples of every moving collision geom for clearance checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .geometry import axis_angle_batch, quat_wxyz_to_mat, rot_err

ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
GRIP_SITE = "gripper0_grip_site"
WRIST_CAMERA = "robot0_eye_in_hand"


@dataclass
class PointSet:
    local: list            # per-body (n_i,3) arrays in body frame
    group: np.ndarray      # (M,) 0=arm 1=palm 2=finger
    radius: np.ndarray     # (M,) sampling radius (conservative inflation)
    body: np.ndarray       # (M,) body index


def _sample_triangles(verts, faces, spacing, rng):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(np.clip(area.sum() / (spacing * spacing) * 6, 50, 20000))
    idx = rng.choice(len(tri), size=n, p=area / area.sum())
    u = rng.random((n, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    t = tri[idx]
    pts = t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])
    return np.concatenate([pts, verts], axis=0)


def _box_surface(size, spacing):
    pts = []
    for axis in range(3):
        for sign in (-1, 1):
            others = [a for a in range(3) if a != axis]
            g = [np.linspace(-size[o], size[o], max(2, int(np.ceil(2 * size[o] / spacing)) + 1)) for o in others]
            a, b = np.meshgrid(*g, indexing="ij")
            p = np.zeros((a.size, 3))
            p[:, axis] = sign * size[axis]
            p[:, others[0]] = a.ravel()
            p[:, others[1]] = b.ravel()
            pts.append(p)
    return np.concatenate(pts, axis=0)


def _voxel_downsample(pts, spacing):
    keys = np.floor(pts / spacing).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(idx)]


class RobotModel:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, arm_spacing=0.016, palm_spacing=0.008,
                 finger_spacing=0.004):
        self.m = model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(model.nbody)]
        self.bodies = [b for b, n in enumerate(names) if n.startswith("robot0_") or n.startswith("gripper0_")]
        self.body_names = [names[b] for b in self.bodies]
        index = {b: i for i, b in enumerate(self.bodies)}
        base = self.bodies[0]
        assert names[base] == "robot0_base", names[base]
        self.root_R = data.xmat[base].reshape(3, 3).copy()
        self.root_p = data.xpos[base].copy()
        self.parent = np.array([index.get(int(model.body_parentid[b]), -1) for b in self.bodies])
        self.local_p = np.array([model.body_pos[b] for b in self.bodies])
        self.local_R = np.array([quat_wxyz_to_mat(model.body_quat[b]) for b in self.bodies])
        jid = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(model.njnt)}
        self.joint_names = ARM_JOINTS + FINGER_JOINTS
        self.joint_of_body = [None] * len(self.bodies)
        lim = []
        for qi, jn in enumerate(self.joint_names):
            j = jid[jn]
            bi = index[int(model.jnt_bodyid[j])]
            jtype = int(model.jnt_type[j])
            assert jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            self.joint_of_body[bi] = (jtype, model.jnt_axis[j].copy(), model.jnt_pos[j].copy(), qi)
            lim.append(model.jnt_range[j].copy())
        self.limits = np.array(lim)
        self.qpos_adr = np.array([model.jnt_qposadr[jid[jn]] for jn in self.joint_names])
        self.arm_body_of_joint = [next(i for i, jb in enumerate(self.joint_of_body) if jb is not None and jb[3] == qi)
                                  for qi in range(7)]
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        self.site_body = index[int(model.site_bodyid[sid])]
        self.site_p = model.site_pos[sid].copy()
        self.site_R = quat_wxyz_to_mat(model.site_quat[sid])
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
        self.cam_body = index[int(model.cam_bodyid[cid])]
        self.cam_p = model.cam_pos[cid].copy()
        self.cam_R = quat_wxyz_to_mat(model.cam_quat[cid])
        self.cam_fovy = float(model.cam_fovy[cid])
        self.points = self._collision_points(arm_spacing, palm_spacing, finger_spacing)
        self.body_group = np.array([self._group(n) for n in self.body_names])
        # private data for the arm's mass matrix (link inertias and rotor armature; the arm's block of the joint-space
        # inertia depends on the robot's own joints only, never on scene state)
        self._md = mujoco.MjData(model)
        self._M = np.zeros((model.nv, model.nv))
        self.dof_adr = np.array([model.jnt_dofadr[jid[jn]] for jn in ARM_JOINTS])

    def mass_matrix(self, q):
        """Joint-space inertia (7,7) of the arm at the robot configuration q (9,), as the OSC computes it."""
        m, d = self.m, self._md
        d.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_fullM(m, self._M, d.qM)
        return self._M[np.ix_(self.dof_adr, self.dof_adr)]

    # ------------------------------------------------------------------ build
    @staticmethod
    def _group(name):
        if "finger" in name:
            return 2
        if name.startswith("gripper0_") or name == "robot0_right_hand":
            return 1
        return 0

    def _collision_points(self, arm_spacing, palm_spacing, finger_spacing):
        m = self.m
        rng = np.random.default_rng(0)
        local, group, radius, body = [], [], [], []
        for bi, b in enumerate(self.bodies):
            name = self.body_names[bi]
            pts_body = []
            if name in ("robot0_base", "robot0_link0", "robot0_link1"):
                local.append(np.zeros((0, 3)))
                continue
            grp = self._group(name)
            spacing = (arm_spacing, palm_spacing, finger_spacing)[grp]
            for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
                if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                    continue
                if m.geom_group[g] not in (0, 3):
                    continue
                gR = quat_wxyz_to_mat(m.geom_quat[g])
                gp = m.geom_pos[g]
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g]
                    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                    verts = m.mesh_vert[va:va + vn].astype(float)
                    faces = m.mesh_face[fa:fa + fn]
                    pts = _sample_triangles(verts, faces, spacing, rng)
                elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                    pts = _box_surface(m.geom_size[g], spacing * 0.7)
                else:
                    continue
                pts_body.append(pts @ gR.T + gp)
            if pts_body:
                pts = _voxel_downsample(np.concatenate(pts_body, 0), spacing)
            else:
                pts = np.zeros((0, 3))
            local.append(pts)
            group.append(np.full(len(pts), grp))
            radius.append(np.full(len(pts), 0.6 * spacing))
            body.append(np.full(len(pts), bi))
        return PointSet(local=local, group=np.concatenate(group), radius=np.concatenate(radius),
                        body=np.concatenate(body))

    # -------------------------------------------------------------- kinematics
    def full_q(self, q_arm, finger=0.04):
        q_arm = np.atleast_2d(np.asarray(q_arm, dtype=float))
        f = np.broadcast_to(np.asarray(finger, dtype=float), (q_arm.shape[0],))
        return np.concatenate([q_arm[:, :7], f[:, None], -f[:, None]], axis=1)

    def fk(self, q):
        """q (N,9) -> body rotations (N,B,3,3) and positions (N,B,3) in world."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        if q.shape[1] == 7:
            q = self.full_q(q)
        n, nb = q.shape[0], len(self.bodies)
        R = np.empty((n, nb, 3, 3))
        P = np.empty((n, nb, 3))
        for i in range(nb):
            pi = self.parent[i]
            if pi < 0:
                Rp = np.broadcast_to(self.root_R, (n, 3, 3)) @ np.eye(3)
                Pp = np.broadcast_to(self.root_p, (n, 3))
                Ri = np.broadcast_to(self.root_R, (n, 3, 3)).copy()
                Pi = np.broadcast_to(self.root_p, (n, 3)).copy()
            else:
                Rp, Pp = R[:, pi], P[:, pi]
                Ri = Rp @ self.local_R[i]
                Pi = Pp + Rp @ self.local_p[i]
            jb = self.joint_of_body[i]
            if jb is not None:
                jtype, axis, jpos, qi = jb
                if jtype == mujoco.mjtJoint.mjJNT_HINGE:
                    Rj = axis_angle_batch(axis, q[:, qi])
                    Pi = Pi + Ri @ jpos - (Ri @ Rj) @ jpos
                    Ri = Ri @ Rj
                else:
                    Pi = Pi + (Ri @ axis) * q[:, qi:qi + 1]
            R[:, i] = Ri
            P[:, i] = Pi
        return R, P

    def site_pose(self, q, R=None, P=None):
        if R is None:
            R, P = self.fk(q)
        Rb, Pb = R[:, self.site_body], P[:, self.site_body]
        return Rb @ self.site_R, Pb + Rb @ self.site_p

    def wrist_camera_pose(self, q):
        """World pose of the wrist camera (MuJoCo camera frame: looks along -z, y up)."""
        R, P = self.fk(q)
        Rb, Pb = R[:, self.cam_body], P[:, self.cam_body]
        return Rb @ self.cam_R, Pb + Rb @ self.cam_p

    def jacobian(self, q, R=None, P=None):
        """Geometric Jacobian (N,6,7) of the grip site w.r.t. the 7 arm joints."""
        if R is None:
            R, P = self.fk(q)
        _, ps = self.site_pose(q, R, P)
        n = R.shape[0]
        J = np.empty((n, 6, 7))
        for qi in range(7):
            bi = self.arm_body_of_joint[qi]
            _, axis, jpos, _ = self.joint_of_body[bi]
            a = R[:, bi] @ axis
            o = P[:, bi] + R[:, bi] @ jpos
            J[:, :3, qi] = np.cross(a, ps - o)
            J[:, 3:, qi] = a
        return J

    def points_world(self, q, groups=None):
        """Collision sample points for configs q -> (N,M,3) (all bodies concatenated)."""
        R, P = self.fk(q)
        out = []
        for bi, pts in enumerate(self.points.local):
            if len(pts) == 0:
                continue
            out.append(P[:, bi, None, :] + np.einsum("nij,kj->nki", R[:, bi], pts))
        W = np.concatenate(out, axis=1)
        if groups is not None:
            sel = np.isin(self.points.group, groups)
            return W[:, sel]
        return W

    def points_in_body(self, q, body_index, pts_world):
        """Express world points in a body frame for config q (single)."""
        R, P = self.fk(np.atleast_2d(q))
        return (pts_world - P[0, body_index]) @ R[0, body_index]

    def gripper_geometry(self, q_arm):
        """Gripper extents in the grip-site frame measured from the collision model."""
        out = {}
        for f in (0.04, 0.0):
            qf = self.full_q(np.asarray(q_arm)[None, :7], f)
            W = self.points_world(qf)[0]
            Rs, ps = self.site_pose(qf)
            L = (W - ps[0]) @ Rs[0]
            out[f] = L
        L = out[0.04]
        g = self.points.group
        tips = np.isin(self.points.body, [i for i, n in enumerate(self.body_names) if n.endswith("_tip")])
        palm = L[g == 1]
        fing = L[g == 2]
        pads = L[tips]
        low = palm[palm[:, 2] > palm[:, 2].max() - 0.012]
        return {
            "pad_z": (float(pads[:, 2].min()), float(pads[:, 2].max())),
            "pad_y": float(np.abs(pads[:, 1]).max()),
            "finger_y": float(np.abs(fing[:, 1]).max()),
            "finger_root_z": float(fing[:, 2].min()),
            "palm_z": float(palm[:, 2].max()),
            "palm_x": float(np.abs(low[:, 0]).max()),
            "palm_y": float(np.abs(palm[:, 1]).max()),
            "pad_inner_open": float(np.abs(pads[:, 0]).min()),
            "finger_x_open": float(np.abs(fing[:, 0]).max()),
            "finger_x_open_tip": float(np.abs(fing[fing[:, 2] >= pads[:, 2].min(), 0]).max()),
            "tip_z": float(fing[:, 2].max()),
        }

    # --------------------------------------------------------------------- IK
    def ik(self, R_target, p_target, seeds, q_rest=None, iters=120, tol_p=1e-3, tol_r=1e-2,
           limit_margin=0.04, null_gain=0.1, finger=0.04):
        """Batched DLS IK for the grip site. Returns (q (S,7), ok (S,), err_p, err_r)."""
        q = np.array(np.atleast_2d(seeds)[:, :7], dtype=float)
        lo = self.limits[:7, 0] + limit_margin
        hi = self.limits[:7, 1] - limit_margin
        q = np.clip(q, lo, hi)
        if q_rest is None:
            q_rest = 0.5 * (lo + hi)
        R_target = np.asarray(R_target, dtype=float)
        p_target = np.asarray(p_target, dtype=float)
        lam = 0.03
        for it in range(iters):
            qf = self.full_q(q, finger)
            R, P = self.fk(qf)
            Rs, ps = self.site_pose(qf, R, P)
            ep = p_target - ps
            er = rot_err(np.broadcast_to(R_target, Rs.shape), Rs)
            npn = np.linalg.norm(ep, axis=1, keepdims=True)
            nrn = np.linalg.norm(er, axis=1, keepdims=True)
            if it > 5 and np.all((npn[:, 0] < tol_p) & (nrn[:, 0] < tol_r)):
                break
            ep = ep * np.minimum(1.0, 0.08 / np.maximum(npn, 1e-9))
            er = er * np.minimum(1.0, 0.4 / np.maximum(nrn, 1e-9))
            e = np.concatenate([ep, er], axis=1)
            J = self.jacobian(qf, R, P)
            JJt = J @ np.swapaxes(J, 1, 2) + (lam ** 2) * np.eye(6)[None]
            x = np.linalg.solve(JJt, e[..., None])
            dq = (np.swapaxes(J, 1, 2) @ x)[..., 0]
            Jpinv = np.swapaxes(J, 1, 2) @ np.linalg.inv(JJt)
            Nmat = np.eye(7)[None] - Jpinv @ J
            dq += (Nmat @ (null_gain * (q_rest - q))[..., None])[..., 0]
            q = np.clip(q + dq, lo, hi)
        qf = self.full_q(q, finger)
        Rs, ps = self.site_pose(qf)
        err_p = np.linalg.norm(p_target - ps, axis=1)
        err_r = np.linalg.norm(rot_err(np.broadcast_to(R_target, Rs.shape), Rs), axis=1)
        ok = (err_p < tol_p * 2) & (err_r < tol_r * 2)
        return q, ok, err_p, err_r

    def solve_ik(self, R_target, p_target, q_ref, n_random=24, rng=None, q_rest=None, finger=0.04):
        """IK with the reference config plus random seeds; returns list of solutions sorted by distance to q_ref."""
        rng = rng or np.random.default_rng(0)
        lo = self.limits[:7, 0] + 0.05
        hi = self.limits[:7, 1] - 0.05
        q_ref = np.asarray(q_ref, dtype=float)[:7]
        seeds = [q_ref]
        noise = rng.normal(0, 0.35, (n_random // 2, 7))
        seeds += list(np.clip(q_ref + noise, lo, hi))
        seeds += list(rng.uniform(lo, hi, (n_random - n_random // 2, 7)))
        q, ok, ep, er = self.ik(R_target, p_target, np.array(seeds), q_rest=q_ref if q_rest is None else q_rest,
                                finger=finger)
        sols = q[ok]
        if len(sols) == 0:
            return np.zeros((0, 7))
        d = np.abs(sols - q_ref).max(axis=1)
        sols = sols[np.argsort(d)]
        # deduplicate
        keep = []
        for s in sols:
            if all(np.abs(s - k).max() > 0.05 for k in keep):
                keep.append(s)
        return np.array(keep)
