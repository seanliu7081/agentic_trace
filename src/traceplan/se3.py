"""SO(3) maps (TRACE_method.md §3): batched torch for the search, float64 numpy twins for certification.

Every torch function stays finite under autograd at the singular points (zero rotation, half turn, parallel axes):
a NaN in an unused torch.where branch would still poison the gradient, so the unused branches read safe values.
"""
from __future__ import annotations

import numpy as np
import torch

PI_BRANCH = -0.99          # cos(angle) below this: read the axis from the symmetric part (sin(angle) ~ 0)
SMALL_BRANCH = 1 - 1e-6    # cos(angle) above this: Taylor series


# --------------------------------------------------------------------------------------------- torch
def hat(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def vee(S):
    return torch.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], -1)


_EXP = {}


def so3_exp(w):
    """I + sinc(th) K + (1 - cos th) / th^2 K^2 with the coefficients from torch.sinc (exact and smooth at 0; the
    clamp only keeps d th / d th2 finite there) and K = hat(w) as one matmul with a cached generator matrix: few
    kernels, since the search is launch bound."""
    key = (w.device, w.dtype)
    if key not in _EXP:
        G = np.zeros((3, 9))
        G[2, 1], G[1, 2], G[2, 3], G[0, 5], G[1, 6], G[0, 7] = -1, 1, 1, -1, -1, 1
        _EXP[key] = (torch.as_tensor(G, dtype=w.dtype, device=w.device), torch.eye(3, dtype=w.dtype, device=w.device))
    G, eye = _EXP[key]
    th = (w * w).sum(-1)[..., None, None].clamp(min=1e-30).sqrt()
    h = torch.sinc(th * (0.5 / np.pi))
    K = (w @ G).reshape(*w.shape[:-1], 3, 3)
    return eye + torch.sinc(th * (1 / np.pi)) * K + (0.5 * h * h) * (K @ K)


def so3_log(R):
    c = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    v = vee(R - R.transpose(-1, -2)) / 2                       # sin(angle) * axis
    s2 = (v * v).sum(-1)
    small, big = c > SMALL_BRANCH, c < PI_BRANCH
    w_small = v * (1 + s2 / 6)[..., None]                      # angle / sin(angle) ~ 1 + angle^2 / 6
    s = torch.where(small | big, torch.ones_like(s2), s2).sqrt()
    w_mid = v * (torch.atan2(s, c) / s)[..., None]
    # near a half turn: a a^T = (sym(R) - c I) / (1 - c); take the best-conditioned column, sign from v
    cb = torch.where(big, c, -torch.ones_like(c))
    M = ((R + R.transpose(-1, -2)) / 2 - cb[..., None, None] * torch.eye(3, dtype=R.dtype, device=R.device)) \
        / (1 - cb)[..., None, None]
    k = M.diagonal(dim1=-2, dim2=-1).argmax(-1)
    col = torch.gather(M, -1, k[..., None, None].expand(*k.shape, 3, 1))[..., 0]
    a = col / torch.gather(col, -1, k[..., None]).clamp_min(1e-12).sqrt()
    a = torch.where((a * v).sum(-1, keepdim=True) < 0, -a, a)
    w_big = a * torch.atan2((s2 + 1e-30).sqrt(), cb)[..., None]
    return torch.where(small[..., None], w_small, torch.where(big[..., None], w_big, w_mid))


def angle_between(u, v):
    """atan2(|u x v|, u.v): accurate at 0 and pi, unlike acos; the eps keeps d|.|/dx finite for parallel axes."""
    cr = torch.linalg.cross(u, v, dim=-1)
    return torch.atan2(((cr * cr).sum(-1) + 1e-12).sqrt(), (u * v).sum(-1))


def rotz(yaw, dtype=torch.float64):
    c, s = np.cos(yaw), np.sin(yaw)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


# --------------------------------------------------------------------------------------------- numpy
def hat_np(w):
    w = np.asarray(w, float)
    z = np.zeros(w.shape[:-1])
    return np.stack([np.stack([z, -w[..., 2], w[..., 1]], -1), np.stack([w[..., 2], z, -w[..., 0]], -1),
                     np.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def so3_exp_np(w):
    w = np.asarray(w, float)
    th2 = (w * w).sum(-1)[..., None, None]
    th = np.sqrt(th2)
    small = th2 < 1e-8
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(small, 1 - th2 / 6, np.sin(th) / th)
        b = np.where(small, 0.5 - th2 / 24, (1 - np.cos(th)) / th2)
    K = hat_np(w)
    return np.eye(3) + a * K + b * (K @ K)


def so3_log_np(R):
    R = np.asarray(R, float)
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    S = R - np.swapaxes(R, -1, -2)
    v = np.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], -1) / 2
    s = np.linalg.norm(v, axis=-1)
    th = np.arctan2(s, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where((c > SMALL_BRANCH)[..., None], v * (1 + s * s / 6)[..., None], v * (th / s)[..., None])
    big = c < PI_BRANCH
    if big.any():
        Rb, cb, vb = R[big], c[big], v[big]
        M = ((Rb + np.swapaxes(Rb, -1, -2)) / 2 - cb[:, None, None] * np.eye(3)) / (1 - cb)[:, None, None]
        k = M.diagonal(axis1=-2, axis2=-1).argmax(-1)
        col = M[np.arange(len(k)), :, k]
        a = col / np.sqrt(np.maximum(col[np.arange(len(k)), k], 1e-300))[:, None]
        a = np.where(((a * vb).sum(-1) < 0)[:, None], -a, a)
        w[big] = a * th[big][:, None]
    return w


def angle_between_np(u, v):
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (np.asarray(u) * np.asarray(v)).sum(-1))


def slerp_np(R0, R1, a):
    """Rotations R0 Exp(a Log(R0^T R1)) for arrays of pairs and fractions a (...,)."""
    return R0 @ so3_exp_np(np.asarray(a)[..., None] * so3_log_np(np.swapaxes(R0, -1, -2) @ R1))
