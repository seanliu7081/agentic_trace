"""Tangent-offset SE(3) B-spline (TRACE_method.md §3).

p(t) = sum_k B_k(t) p_k,   R(t) = R_ref(t) Exp(sum_k B_k(t) phi_k),   R_ref(t) = R0 Exp(t Log(R0^T Rg)).
Clamped uniform cubic basis with knots [0,0,0,0, j/(K-3) for j=1..K-4, 1,1,1,1]; the schedule K = 4, 5, 7, 11 has
nested knot vectors, so `refine` (Boehm insertion) reproduces T(t) exactly.

theta layout (the solvers' cfg["layout"] = (n_ctrl, 3, 3)): the free position control points (n_ctrl, 3), flattened,
followed by the free rotation offsets (n_ctrl, 3), flattened. The start is always pinned (p_1 = p0, phi_1 = 0); the
end (p_K = pg, phi_K = 0) when pin_end.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from .se3 import so3_exp, so3_exp_np, so3_log_np

K_SCHEDULE = (4, 5, 7, 11)
DEG = 3


def knots(K):
    return np.concatenate([np.zeros(DEG + 1), np.arange(1, K - DEG) / (K - DEG), np.ones(DEG + 1)])


def _div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.zeros(a.shape), where=b > 0)        # 0/0 := 0 at repeated knots


def _basis(ts, U, p, d):
    if d == 0:
        span = np.clip(np.searchsorted(U, ts, side="right") - 1, DEG, len(U) - DEG - 2)   # t = 1 -> last span
        B = np.zeros((len(ts), len(U) - 1))
        B[np.arange(len(ts)), span] = 1.0
        for q in range(1, p + 1):
            i = np.arange(len(U) - 1 - q)
            B = (_div(ts[:, None] - U[i], U[i + q] - U[i]) * B[:, :len(i)]
                 + _div(U[i + q + 1] - ts[:, None], U[i + q + 1] - U[i + 1]) * B[:, 1:len(i) + 1])
        return B
    L = _basis(ts, U, p - 1, d - 1)
    i = np.arange(len(U) - 1 - p)
    return p * (_div(1.0, U[i + p] - U[i]) * L[:, :len(i)] - _div(1.0, U[i + p + 1] - U[i + 1]) * L[:, 1:len(i) + 1])


def basis_matrix(ts, K, deriv=0):
    """(N, K) float64 matrix of the d-th derivative of the basis (Cox-de Boor), deriv in {0, 1, 2}."""
    return _basis(np.asarray(ts, float), knots(K), DEG, deriv)


def greville(K):
    U = knots(K)
    return (U[1:K + 1] + U[2:K + 2] + U[3:K + 3]) / 3


@lru_cache(maxsize=None)
def insert_matrix(K_from, K_to):
    """A (K_to, K_from) with ctrl_to = A @ ctrl_from: Boehm insertion of the knots of K_to missing from K_from."""
    U = knots(K_from)
    new = [u for u in knots(K_to)[DEG + 1:-DEG - 1] if not np.isclose(U, u).any()]
    assert len(new) == K_to - K_from, "knot vectors are not nested"
    A = np.eye(K_from)
    for u in new:
        k = int(np.searchsorted(U, u, side="right") - 1)
        Q = np.zeros((len(A) + 1, K_from))
        for i in range(len(A) + 1):
            if i <= k - DEG:
                Q[i] = A[i]
            elif i >= k + 1:
                Q[i] = A[i - 1]
            else:
                a = (u - U[i]) / (U[i + DEG] - U[i])
                Q[i] = (1 - a) * A[i - 1] + a * A[i]
        A, U = Q, np.insert(U, k + 1, u)
    return A


def insert_knots(ctrl, K_from, K_to):
    """Exact refinement of control points (..., K_from, d) -> (..., K_to, d); numpy or torch."""
    A = insert_matrix(K_from, K_to)
    if isinstance(ctrl, torch.Tensor):
        return torch.as_tensor(A, dtype=ctrl.dtype, device=ctrl.device) @ ctrl
    return A @ ctrl


def _clamp_norm(F, m):
    n = ((F * F).sum(-1, keepdim=True) + 1e-12).sqrt()
    return F * torch.clamp(m / n, max=1.0)


class SE3Spline:
    def __init__(self, R0, p0, Rg, pg, K, pin_end=True, phi_max=2.5, device="cpu", dtype=torch.float32):
        self.R0, self.p0, self.Rg, self.pg = (np.asarray(x, float) for x in (R0, p0, Rg, pg))
        self.K, self.pin_end, self.phi_max = int(K), bool(pin_end), float(phi_max)
        self.device, self.dtype = torch.device(device), dtype
        self.w_ref = so3_log_np(self.R0.T @ self.Rg)
        self.n_ctrl = self.K - 1 - int(self.pin_end)
        self.layout = (self.n_ctrl, 3, 3)
        self._p0 = self._t(self.p0)
        self._pg = self._t(self.pg)
        self._cache = {}

    def _t(self, x):
        return torch.as_tensor(np.asarray(x, float), dtype=self.dtype, device=self.device)

    @property
    def dim(self):
        return 6 * self.n_ctrl

    def like(self, K=None, device=None, dtype=None, pin_end=None):
        return SE3Spline(self.R0, self.p0, self.Rg, self.pg, self.K if K is None else K,
                         self.pin_end if pin_end is None else pin_end, self.phi_max,
                         self.device if device is None else device, self.dtype if dtype is None else dtype)

    # ------------------------------------------------------------------ control points
    def ctrl(self, theta):
        """theta (M, D) -> P (M, K, 3), Phi (M, K, 3) with pins inserted and |phi_k| clamped to phi_max."""
        M, n = theta.shape[0], self.n_ctrl
        P = theta[:, :3 * n].reshape(M, n, 3)
        F = _clamp_norm(theta[:, 3 * n:].reshape(M, n, 3), self.phi_max)
        z = theta.new_zeros(M, 1, 3)
        P = [self._p0.expand(M, 1, 3), P] + ([self._pg.expand(M, 1, 3)] if self.pin_end else [])
        F = [z, F] + ([z] if self.pin_end else [])
        return torch.cat(P, 1), torch.cat(F, 1)

    def theta_of(self, P, F):
        """Inverse of ctrl: the free entries of full control sets (M, K, 3)."""
        sl = slice(1, self.K - 1 if self.pin_end else self.K)
        M = P.shape[0]
        return torch.cat([P[:, sl].reshape(M, -1), F[:, sl].reshape(M, -1)], 1)

    def straight_theta(self):
        """Control points on the start-goal line at the Greville abscissae, phi = 0: exactly the straight geodesic."""
        g = greville(self.K)
        P = self.p0 + g[:, None] * (self.pg - self.p0)
        return self.theta_of(self._t(P)[None], self._t(np.zeros((self.K, 3)))[None])[0]

    def theta_from_path(self, p_path, R_path=None):
        """Least-squares fit at arc-length parameters with the pins respected (deviation from the straight seed, so a
        short path still gives a well-posed fit)."""
        p_path = np.asarray(p_path, float)
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p_path, axis=0), axis=1))]
        s = d / d[-1] if d[-1] > 1e-9 else np.linspace(0.0, 1.0, len(p_path))
        B = basis_matrix(s, self.K)
        free = np.arange(1, self.K - 1 if self.pin_end else self.K)
        A = B[:, free]
        A2 = A.T @ A + 1e-8 * np.eye(len(free))
        g = greville(self.K)
        P0 = self.p0 + g[:, None] * (self.pg - self.p0)
        P = P0.copy()
        P[free] += np.linalg.solve(A2, A.T @ (p_path - B @ P0))
        F = np.zeros((self.K, 3))
        if R_path is not None:
            Rref = self.R0 @ so3_exp_np(s[:, None] * self.w_ref)
            target = so3_log_np(np.swapaxes(Rref, -1, -2) @ np.asarray(R_path, float))
            F[free] = np.linalg.solve(A2, A.T @ target)
        return self.theta_of(self._t(P)[None], self._t(F)[None])[0]

    # ------------------------------------------------------------------ evaluation
    def _bases(self, ts):
        """Bases of the free control points for p, v, acc stacked (3N, n), the pins' constant share of them (3N, 3),
        and R_ref (N, 3, 3): eval is then two matmuls plus the exponential (the search is launch bound)."""
        key = ts.tobytes()
        if key not in self._cache:
            B = np.concatenate([basis_matrix(ts, self.K, d) for d in range(3)])
            free = np.arange(1, self.K - 1 if self.pin_end else self.K)
            pins = [0, self.K - 1] if self.pin_end else [0]
            c = B[:, pins] @ np.stack([self.p0, self.pg][:len(pins)])
            Rref = self.R0 @ so3_exp_np(ts[:, None] * self.w_ref)
            self._cache[key] = tuple(self._t(x) for x in (B[:, free], c, Rref))
        return self._cache[key]

    def eval(self, theta, ts):
        """theta (M, D), ts (N,) numpy -> dict R (M,N,3,3), p, v = dp/dt, acc = d2p/dt2 (M,N,3); differentiable."""
        ts = np.asarray(ts, float)
        B, c, Rref = self._bases(ts)
        M, n, N = theta.shape[0], self.n_ctrl, len(ts)
        X = B @ theta[:, :3 * n].reshape(M, n, 3) + c
        phi = B[:N] @ _clamp_norm(theta[:, 3 * n:].reshape(M, n, 3), self.phi_max)
        return {"R": Rref @ so3_exp(phi), "p": X[:, :N], "v": X[:, N:2 * N], "acc": X[:, 2 * N:], "phi": phi}

    def refine(self, theta, K_new):
        """(SE3Spline at K_new, theta') tracing the identical T(t)."""
        P, F = self.ctrl(theta)
        sp = self.like(K=K_new)
        return sp, sp.theta_of(insert_knots(P, self.K, K_new), insert_knots(F, self.K, K_new))

    def sample_np(self, theta_1, ts):
        """(R (N,3,3), p (N,3)) float64 numpy of one trajectory."""
        P, F = self.like(device="cpu", dtype=torch.float64).ctrl(
            torch.as_tensor(theta_1, dtype=torch.float64, device="cpu").reshape(1, -1))
        ts = np.asarray(ts, float)
        B = basis_matrix(ts, self.K)
        Rref = self.R0 @ so3_exp_np(ts[:, None] * self.w_ref)
        return Rref @ so3_exp_np(B @ F[0].numpy()), B @ P[0].numpy()

    # ------------------------------------------------------------------ serialization
    def to_dict(self, theta_1):
        P, F = self.like(device="cpu", dtype=torch.float64).ctrl(
            torch.as_tensor(theta_1, dtype=torch.float64, device="cpu").reshape(1, -1))
        return {"K": self.K, "pin_end": self.pin_end, "phi_max": self.phi_max, "R0": self.R0.tolist(),
                "p0": self.p0.tolist(), "Rg": self.Rg.tolist(), "pg": self.pg.tolist(),
                "P": P[0].tolist(), "Phi": F[0].tolist()}

    @staticmethod
    def from_dict(d, device="cpu", dtype=torch.float64):
        sp = SE3Spline(d["R0"], d["p0"], d["Rg"], d["pg"], d["K"], d["pin_end"], d["phi_max"], device, dtype)
        return sp, sp.theta_of(sp._t(d["P"])[None], sp._t(d["Phi"])[None])[0]
