"""Tests for traceplan.solvers and traceplan.retime (DESIGN §2.7, §2.9)."""
import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from traceplan.retime import retime
from traceplan.solvers import explore, exploit, portfolio

F64 = torch.float64
EXPLORE = ("random", "cem", "mppi", "cmaes")
LAYOUT = (5, 3, 3)                     # 5 free control points: theta = [P (5,3), Phi (5,3)] row-major, D = 30


def _gen(seed):
    return torch.Generator().manual_seed(seed)


_g = _gen(0)
TARGET = 0.1 * torch.randn(30, generator=_g, dtype=F64)
X_QUAD = TARGET + 0.1 * torch.randn(3, 30, generator=_g, dtype=F64)
_col = torch.sign(torch.randn(3, 1, 6, generator=_g, dtype=F64)).expand(3, 5, 6)     # one sign per coordinate
X_RIPPLE = TARGET + 0.1 * torch.cat([_col[..., :3].reshape(3, 15), _col[..., 3:].reshape(3, 15)], 1)  # path shift


def quad(th):
    """Smooth quadratic in the control-point layout: offset from TARGET plus its second differences along the
    control points, per coordinate."""
    e = th - TARGET
    P, F = e[:, :15].reshape(-1, 5, 3), e[:, 15:].reshape(-1, 5, 3)
    d2 = lambda X: X[:, 2:] - 2 * X[:, 1:-1] + X[:, :-2]
    return (P ** 2 + 0.5 * F ** 2).sum((1, 2)) + (4 * d2(P) ** 2 + 2 * d2(F) ** 2).sum((1, 2))


def ripple(th):
    """Multimodal: a local minimum every centimetre (X_RIPPLE sits in one) on a 5 cm-scale bowl."""
    e = th - TARGET
    return ((e / 0.05) ** 2 + (1 - torch.cos(2 * math.pi * e / 0.01))).sum(1)


def rastrigin(th):
    x = th / 0.05
    return (x ** 2 / 10 + 2 * (1 - torch.cos(2 * math.pi * x))).sum(1)


# ------------------------------------------------------------------------------------------------ solvers
@pytest.mark.parametrize("name", EXPLORE)
def test_explore_reduces_J(name):
    cfg = dict(layout=LAYOUT, alpha_pos=0.05, alpha_rot=0.05)
    for J, x0 in ((quad, X_QUAD), (ripple, X_RIPPLE)):
        ratio = J(explore(name, J, x0, 2560, _gen(1), cfg)) / J(x0)
        assert (ratio < (0.95 if name == "random" else 0.7)).all(), (J.__name__, ratio)


@pytest.mark.parametrize("name", ("adam", "lbfgs"))
def test_exploit_reduces_J(name):
    cfg = dict(layout=LAYOUT)
    ratio = quad(exploit(name, quad, X_QUAD, 20, cfg)) / quad(X_QUAD)
    assert (ratio < (0.2 if name == "adam" else 1e-4)).all(), ratio
    xs = X_RIPPLE + 0.0025 * torch.sign(X_RIPPLE - TARGET)            # on a ripple slope
    ratio = ripple(exploit(name, ripple, xs, 20, cfg)) / ripple(xs)
    assert (ratio < 0.9).all(), ratio


@pytest.mark.parametrize("budget", (1000, 100))
@pytest.mark.parametrize("name", EXPLORE)
def test_explore_respects_eval_budget(name, budget):
    rows = []

    def J(th):
        rows.append(th.shape[0])
        return quad(th)

    explore(name, J, X_QUAD, budget, _gen(2), dict(layout=LAYOUT))
    S = min(256, budget)
    assert rows == [3 * S] * (budget // S)            # one call of M x n_samples rows per iteration
    assert sum(rows) <= 3 * budget


@pytest.mark.parametrize("name", EXPLORE)
def test_explore_reproducible_from_generator_seed(name):
    cfg = dict(layout=LAYOUT)
    a, b, c = (explore(name, ripple, X_RIPPLE, 1024, _gen(s), cfg) for s in (5, 5, 6))
    assert torch.equal(a, b) and not torch.equal(a, c)


@pytest.mark.parametrize("name", ("adam", "lbfgs"))
def test_batched_exploit_equals_per_seed(name):
    x0 = torch.cat([X_QUAD[:2], X_RIPPLE[:1] + 0.003])
    cfg = dict(layout=LAYOUT)
    for J in (quad, ripple):
        batched = exploit(name, J, x0, 15, cfg)
        single = torch.cat([exploit(name, J, x0[m:m + 1], 15, cfg) for m in range(3)])
        torch.testing.assert_close(batched, single, rtol=0, atol=1e-10)


def test_portfolio_returns_best_arm():
    # Five arms in different Rastrigin basins, noise too small to hop basins. Arm 2 starts with the worst J but in
    # the global basin, so it must win on refined J. Arm 4 has its own cfg and generator (a separate group).
    cfg = dict(layout=(1, 2, 2), alpha_pos=0.005, alpha_rot=0.005, n_samples=64)
    starts = 0.05 * torch.tensor([[2., 1, 0, 1], [-1, 2, 2, 0], [0.2, 0.1, -0.1, 0], [1, 1, 1, 1], [0, 0, 3, 0]],
                                 dtype=F64)
    gen = _gen(0)
    arms = [dict(theta=starts[i], explore="mppi", exploit="lbfgs", cfg=cfg, gen=gen) for i in range(4)]
    arms.append(dict(theta=starts[4], explore="none", exploit="adam", cfg=dict(cfg), gen=_gen(1)))
    assert rastrigin(starts).argmax() == 2
    res = portfolio(arms, rastrigin, 128)
    assert res["index"] == 2 and res["theta"].shape == (4,) and res["J"] < 1e-6
    assert res["J"] == pytest.approx(rastrigin(res["theta"][None]).item())


# ------------------------------------------------------------------------------------------------- retime
CFG = dict(tube0=0.004, tube_k=0.022, v_min=0.05, v_max=0.4, a_max=0.5, w_max=1.2, dt=0.05, v_slow=0.05)
N = 128
S_PATH = np.linspace(0.0, 1.0, N)


def _check_timed(T, p, R, q):
    dt = CFG["dt"]
    assert T.t[0] == 0.0 and np.allclose(T.t, np.arange(len(T.t)) * dt, rtol=0, atol=1e-12)
    assert T.speed[0] == 0.0 and T.speed[-1] == 0.0
    for out, inp in ((T.p, p), (T.R, R), (T.q, q)):
        np.testing.assert_allclose(out[[0, -1]], inp[[0, -1]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(T.tube, CFG["tube0"] + CFG["tube_k"] * T.speed, rtol=0, atol=1e-15)
    v_fd = np.linalg.norm(np.diff(T.p, axis=0), axis=1) / dt
    w_fd = Rotation.from_matrix(np.einsum("nji,njk->nik", T.R[:-1], T.R[1:])).magnitude() / dt
    assert np.abs(np.diff(T.speed)).max() / dt <= CFG["a_max"] + 1e-9            # planned speed
    assert np.abs(np.diff(v_fd)).max() / dt <= 1.01 * CFG["a_max"] + 1e-6        # finite differences of positions
    assert w_fd.max() <= CFG["w_max"] + 1e-9
    assert (v_fd <= np.maximum(T.speed[:-1], T.speed[1:]) + CFG["a_max"] * dt + 1e-9).all()   # p follows speed


def test_retime_curved_path_caps_and_limits():
    ang = 0.5 * np.pi * S_PATH
    p = np.stack([0.4 + 0.3 * np.cos(ang), 0.3 * np.sin(ang), 0.2 + 0.1 * S_PATH], 1)
    R = Rotation.from_rotvec(np.outer(S_PATH, [0.3, -0.2, 2.4])).as_matrix()      # ~2.4 rad over 0.48 m: w_max binds
    q = np.outer(S_PATH, np.linspace(-0.5, 0.5, 7))
    slack = np.where(np.abs(S_PATH - 0.4) < 0.08, 0.008, 0.2)                    # a tight passage
    slow = (S_PATH > 0.7) & (S_PATH < 0.8)
    T = retime(p, R, q, S_PATH, slack, dict(CFG, slow=slow))
    _check_timed(T, p, R, q)
    cap = np.clip((slack - CFG["tube0"]) / CFG["tube_k"], CFG["v_min"], CFG["v_max"])
    k = np.linalg.norm(T.p[:, None] - p[None], axis=2).argmin(1)                 # nearest dense sample
    assert (T.speed <= cap[k] + 1e-12).all()
    inside = slow[k] & slow[np.clip(k - 1, 0, N - 1)] & slow[np.clip(k + 1, 0, N - 1)]
    assert inside.any() and (T.speed[inside] <= CFG["v_slow"] + 1e-12).all()


def test_retime_straight_path_time_optimal():
    p = np.stack([np.linspace(0.0, 0.5, N), np.zeros(N), np.zeros(N)], 1)
    R = np.repeat(np.eye(3)[None], N, 0)
    q = np.zeros((N, 7))
    T = retime(p, R, q, S_PATH, np.full(N, 1.0), CFG)
    _check_timed(T, p, R, q)
    # 0.8 s up to 0.4 m/s over 0.16 m, 0.45 s cruise, 0.8 s down = 2.05 s, plus under one step of rounding
    assert 2.05 - 1e-9 <= T.t[-1] <= 2.05 + CFG["dt"] + 1e-9
    assert T.speed.max() == pytest.approx(CFG["v_max"])


def test_retime_rotation_in_place():
    p = np.repeat(np.array([[0.5, 0.0, 0.3]]), N, 0)
    R = Rotation.from_rotvec(np.outer(S_PATH, [0.0, 0.0, 1.0])).as_matrix()
    q = np.outer(S_PATH, np.ones(7))
    T = retime(p, R, q, S_PATH, np.full(N, 0.2), CFG)
    _check_timed(T, p, R, q)
    assert (T.speed == 0).all()
    assert 1.0 / CFG["w_max"] - 1e-9 <= T.t[-1] <= 1.0 / CFG["w_max"] + CFG["dt"] + 1e-9
