"""Small rotation / transform helpers (numpy, batched where useful).

Conventions: rotation matrices act on column vectors; quaternions are
MuJoCo order (w, x, y, z) unless a name says xyzw.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz_to_mat(q):
    q = np.asarray(q, dtype=float)
    return Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], axis=-1)).as_matrix()


def mat_to_quat_wxyz(m):
    xyzw = Rotation.from_matrix(np.asarray(m, dtype=float)).as_quat()
    return np.concatenate([xyzw[..., 3:], xyzw[..., :3]], axis=-1)


def rotvec_to_mat(v):
    return Rotation.from_rotvec(np.asarray(v, dtype=float)).as_matrix()


def mat_to_rotvec(m):
    return Rotation.from_matrix(np.asarray(m, dtype=float)).as_rotvec()


def axis_angle_batch(axis, angle):
    """Rodrigues for a fixed unit axis (3,) and angles (N,) -> (N,3,3)."""
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    s = np.sin(angle)[:, None, None]
    c = np.cos(angle)[:, None, None]
    eye = np.eye(3)[None]
    return eye + s * k[None] + (1 - c) * (k @ k)[None]


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def rot_err(r_target, r_current):
    """World-frame rotation vector taking r_current to r_target (batched ok)."""
    return Rotation.from_matrix(np.asarray(r_target) @ np.swapaxes(np.asarray(r_current), -1, -2)).as_rotvec()


def angle_between(r_a, r_b):
    return np.linalg.norm(rot_err(r_a, r_b), axis=-1)


def make_T(r, p):
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = p
    return t


def inv_T(t):
    r = t[:3, :3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t[:3, 3]
    return out


def transform_points(t, pts):
    pts = np.asarray(pts, dtype=float)
    return pts @ t[:3, :3].T + t[:3, 3]


def top_down_rotation(yaw):
    """Grip-site rotation with approach axis (local z) pointing down (-z world).

    Panda/robosuite grip site: local x = finger closing axis, local z = approach.
    yaw is the world direction of the closing axis in the table plane.
    """
    return frame_from_axes([0.0, 0.0, -1.0], [np.cos(yaw), np.sin(yaw), 0.0])


def frame_from_axes(z_axis, x_hint):
    """Right-handed rotation with local z = approach and local x ~ x_hint (closing axis)."""
    z = np.asarray(z_axis, dtype=float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, dtype=float)
    x = x - z * np.dot(x, z)
    n = np.linalg.norm(x)
    if n < 1e-6:
        x = np.cross([0.0, 1.0, 0.0], z)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross([1.0, 0.0, 0.0], z)
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
