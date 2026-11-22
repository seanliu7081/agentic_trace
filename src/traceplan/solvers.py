"""Solver slots (TRACE_method.md §6.3, DESIGN §2.7): explore (sampling), exploit (gradient), portfolio.

Generic: a solver sees only J, which maps a batch of parameter vectors theta (M', D) (torch) to costs (M',). It
knows nothing about robots. Each row of theta (M, D) is a seed, optimized independently of the others; batching the
seeds only saves J calls.

Conventions shared by every function here:

- theta layout. cfg["layout"] = (n_ctrl, n_pos_dims, n_rot_dims) means n_ctrl free control points, each with
  n_pos_dims position and n_rot_dims rotation coordinates, laid out as
  theta = concat(P_free.flatten(), Phi_free.flatten()), with P_free (n_ctrl, n_pos_dims) and
  Phi_free (n_ctrl, n_rot_dims) both row-major, so D = n_ctrl * (n_pos_dims + n_rot_dims). For the SE3Spline
  (DESIGN §2.2) that is (K - 2, 3, 3) with a pinned end and (K - 1, 3, 3) with a free end.
- J rows are seed-major. A call with M*k rows is an (M, k, D) tensor reshaped to (M*k, D): rows m*k ... m*k + k - 1
  belong to seed m. An objective with per-seed state (the augmented-Lagrangian multipliers) expands that state with
  repeat_interleave(k).
- Randomness comes only from the torch.Generator passed in (noise is drawn on the generator's device, then moved
  to theta's device), so the same generator seed reproduces the result exactly on CPU.
- Both slots are elitist: per seed they return the best point they evaluated, so J(output) <= J(input).

Not implemented: Levenberg-Marquardt and SLSQP (listed in §6.3, not needed per DESIGN §2.7), SVGD and
diffusion-style sampling.
"""
from __future__ import annotations

import math

import torch

DEFAULTS = {
    "n_samples": 256,        # rows per seed per explore iteration; sample 0 is the current mean
    "iters": 10,             # §6.4 explore iterations per level: callers pass budget_evals = n_samples * iters
    "exploit_iters": 20,     # §6.4 exploit iterations per level: callers pass budget_iters = exploit_iters
    "alpha_pos": 0.05,       # largest per-coordinate noise std of position control points (m); caller scales with s_l
    "alpha_rot": 0.2,        # same for rotation offsets (rad)
    "smooth_eps": 1e-3,      # eps of the smooth prior (D^T D + eps I)^-1
    "tau": 0.1,              # MPPI temperature, relative to the per-seed batch spread median(J) - min(J)
    "elite_frac": 0.1,       # CEM elite fraction of a batch
    "lr": 0.01,              # Adam step size and length of L-BFGS's first (gradient) step, in theta units (m, rad)
    "adam": (0.9, 0.999, 1e-8),   # Adam beta1, beta2, eps
    "history": 10,           # L-BFGS curvature pairs kept per seed
    "ls_steps": 10,          # L-BFGS step candidates 1, 1/2, ..., 2^-(ls_steps-1), evaluated in one J call
    "armijo": 1e-4,          # sufficient-decrease constant of the L-BFGS line search
}


def noise_factor(layout, alpha_pos, alpha_rot, eps):
    """A (D, D) float64 such that z @ A.T, z ~ N(0, I), is the smooth noise N(0, alpha^2 (D^T D + eps I)^-1) per
    coordinate.

    D is the second-difference operator over the K = n_ctrl + 2 control points made of the free ones plus a pinned
    neighbour on each side (start and end) whose perturbation is zero. Its free block is the tridiagonal (1, -2, 1)
    matrix (the STOMP prior), so the noise fades toward the pins. For a free-end spline the last free point is
    treated as if it had a pinned neighbour, which only damps its exploration noise. The covariance is scaled to a
    largest marginal variance of 1, so alpha_pos and alpha_rot are the largest per-coordinate standard deviations
    (m, rad).
    """
    n, dp, dr = layout
    f64 = dict(dtype=torch.float64)
    lap = -2.0 * torch.eye(n, **f64) + torch.diag(torch.ones(n - 1, **f64), 1) + torch.diag(torch.ones(n - 1, **f64), -1)
    cov = torch.linalg.inv(lap.T @ lap + eps * torch.eye(n, **f64))
    L = torch.linalg.cholesky(cov / cov.diagonal().max()).contiguous()      # kron needs a contiguous factor
    return torch.block_diag(alpha_pos * torch.kron(L, torch.eye(dp, **f64)), alpha_rot * torch.kron(L, torch.eye(dr, **f64)))


# ------------------------------------------------------------------------------------------------ explore
class _Random:
    """Random shooting: every batch is drawn around the input; the best sample wins."""

    def __init__(self, theta, A, c, S):
        self.theta, self.A = theta, A

    def propose(self, z):
        return self.theta[:, None] + z @ self.A.T

    def update(self, z, X, Jb):
        pass


class _MPPI:
    """theta <- sum_m w_m theta_m, w_m ∝ exp(-(J_m - min J) / tau'), tau' = tau * (median J - min J) per seed.

    A fixed absolute temperature cannot serve every continuation level: J scales differ by orders of magnitude from
    level 0 to level 3 (mu_l from 1 to 64, s_l from 8 to 1). Relative to the batch spread, tau = 0.1 gives the
    median sample a weight of e^-10 against the best one at every level.
    """

    def __init__(self, theta, A, c, S):
        self.mean, self.A, self.tau = theta, A, c["tau"]

    def propose(self, z):
        return self.mean[:, None] + z @ self.A.T

    def update(self, z, X, Jb):
        lo = Jb.min(1, keepdim=True).values
        temp = (self.tau * (Jb.median(1, keepdim=True).values - lo)).clamp_min(1e-12)   # ties at the minimum
        w = torch.softmax(-(Jb - lo) / temp, dim=1)
        self.mean = (w[..., None] * X).sum(1)


class _CEM:
    """Cross-entropy method with a diagonal Gaussian in the whitened prior frame: samples mean + (sd * z) A^T, then
    mean and sd are refit to the elites. Starting at sd = 1 keeps the smooth prior's correlations throughout."""

    def __init__(self, theta, A, c, S):
        self.mean, self.A = theta, A
        self.sd = torch.ones_like(theta)
        self.k = max(2, round(c["elite_frac"] * S))

    def propose(self, z):
        return self.mean[:, None] + (self.sd[:, None] * z) @ self.A.T

    def update(self, z, X, Jb):
        idx = Jb.topk(self.k, dim=1, largest=False).indices
        u = torch.gather(self.sd[:, None] * z, 1, idx[..., None].expand(-1, -1, z.shape[-1]))
        self.mean = self.mean + u.mean(1) @ self.A.T
        self.sd = u.std(1, correction=0)


class _SepCMA:
    """sep-CMA-ES (Ros & Hansen 2008: CMA-ES with a diagonal covariance and its scaled learning rates), run in the
    whitened prior frame theta = theta0 + u A^T so the smooth prior is its initial covariance. Sample 0 (the mean)
    is evaluated for the elitist best but left out of the update, which ranks the other S - 1 samples."""

    def __init__(self, theta, A, c, S):
        self.theta0, self.A = theta, A
        M, n = theta.shape
        mu = (S - 1) // 2
        w = math.log(mu + 0.5) - torch.log(torch.arange(1, mu + 1, dtype=theta.dtype, device=theta.device))
        self.w = (w / w.sum())[:, None]
        self.mueff = 1.0 / float((self.w ** 2).sum())
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.c1 = c1 * (n + 2) / 3
        self.cmu = min(1 - self.c1, cmu * (n + 2) / 3)
        self.chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.hs_bound = (1.4 + 2 / (n + 1)) * self.chi
        self.m, self.ps, self.pc = (torch.zeros_like(theta) for _ in range(3))
        self.C = torch.ones_like(theta)
        self.sigma = theta.new_ones(M)
        self.g = 0

    def propose(self, z):
        u = self.m[:, None] + self.sigma[:, None, None] * self.C.sqrt()[:, None] * z
        return self.theta0[:, None] + u @ self.A.T

    def update(self, z, X, Jb):
        order = Jb[:, 1:].argsort(dim=1)[:, :self.w.shape[0]]
        zs = torch.gather(z[:, 1:], 1, order[..., None].expand(-1, -1, z.shape[-1]))    # (M, mu, n), best first
        ys = self.C.sqrt()[:, None] * zs
        zw, yw = (self.w * zs).sum(1), (self.w * ys).sum(1)
        self.m = self.m + self.sigma[:, None] * yw
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        nps = self.ps.norm(dim=1)
        self.sigma = self.sigma * torch.exp(self.cs / self.ds * (nps / self.chi - 1))
        self.g += 1
        hs = (nps / math.sqrt(1 - (1 - self.cs) ** (2 * self.g)) < self.hs_bound).to(z.dtype)[:, None]
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * (self.w * ys ** 2).sum(1))


EXPLORERS = {"random": _Random, "cem": _CEM, "mppi": _MPPI, "cmaes": _SepCMA}


def explore(name, J, theta, budget_evals, gen, cfg):
    """Sampling solver. name: "none" | "random" | "cem" | "mppi" | "cmaes" (sep-CMA-ES, diagonal covariance).

    theta (M, D): M seeds. budget_evals: evaluation budget per seed, which sets the iteration count:
    budget_evals // S iterations of S = min(n_samples, budget_evals) rows per seed, each iteration one J call of
    M*S rows (seed-major). Sample 0 of every batch is the current mean (for "random": the input), so the input is
    always evaluated. Noise is the smooth control-point prior of noise_factor(). gen: torch.Generator, the only
    source of randomness. Returns (M, D), per seed the best evaluated point.

    cfg keys (defaults in DEFAULTS): layout (required, see the module docstring); n_samples 256; alpha_pos 0.05 m and
    alpha_rot 0.2 rad (largest per-coordinate noise std; the caller shrinks them with s_l); smooth_eps 1e-3; tau 0.1
    (MPPI, relative temperature, see _MPPI); elite_frac 0.1 (CEM). iters 10 is the caller's per-level iteration
    count (budget_evals = n_samples * iters).
    """
    if name == "none":
        return theta
    c = {**DEFAULTS, **cfg}
    theta = theta.detach()
    M, D = theta.shape
    S = min(c["n_samples"], budget_evals)
    A = noise_factor(c["layout"], c["alpha_pos"], c["alpha_rot"], c["smooth_eps"]).to(theta)
    method = EXPLORERS[name](theta, A, c, S)
    best, best_J = theta.clone(), theta.new_full((M,), math.inf)
    rows = torch.arange(M, device=theta.device)
    with torch.no_grad():
        for _ in range(budget_evals // S):
            z = torch.randn((M, S, D), generator=gen, device=gen.device, dtype=theta.dtype).to(theta.device)
            z[:, 0] = 0.0
            X = method.propose(z)
            Jb = J(X.reshape(M * S, D)).reshape(M, S)
            j, i = Jb.min(1)
            better = j < best_J
            best = torch.where(better[:, None], X[rows, i], best)
            best_J = torch.where(better, j, best_J)
            method.update(z, X, Jb)
    return best


# ------------------------------------------------------------------------------------------------ exploit
def _value_grad(J, x):
    with torch.enable_grad():          # callers may run the search under no_grad
        x = x.detach().requires_grad_(True)
        f = J(x)
        g, = torch.autograd.grad(f.sum(), x)     # J summed over seeds is separable: row m gets dJ_m/dtheta_m
    return f.detach(), g


def _adam(J, theta, iters, c):
    b1, b2, eps = c["adam"]
    x = theta.detach().clone()
    m, v = torch.zeros_like(x), torch.zeros_like(x)
    best, best_J = x.clone(), x.new_full((x.shape[0],), math.inf)
    for i in range(1, iters + 1):
        f, g = _value_grad(J, x)
        better = f < best_J
        best, best_J = torch.where(better[:, None], x, best), torch.where(better, f, best_J)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - c["lr"] * (m / (1 - b1 ** i)) / ((v / (1 - b2 ** i)).sqrt() + eps)
    with torch.no_grad():
        f = J(x)
    return torch.where((f < best_J)[:, None], x, best)


def _lbfgs(J, theta, iters, c):
    """L-BFGS with per-seed curvature histories and a per-seed backtracking (Armijo) line search whose step
    candidates are evaluated in one J call, so seeds never mix (torch.optim.LBFGS would couple them through shared
    inner products and one line search). That call also returns the candidates' gradients: J is launch bound, so
    the gradient at the accepted step comes at no extra call. Monotone per seed, so the last iterate is the best."""
    M, D = theta.shape
    h, ns = c["history"], c["ls_steps"]
    rows = torch.arange(M, device=theta.device)
    steps = 0.5 ** torch.arange(ns, dtype=theta.dtype, device=theta.device)
    Sh, Yh = theta.new_zeros(M, h, D), theta.new_zeros(M, h, D)
    rho = theta.new_zeros(M, h)                  # 0 marks an empty slot, which drops out of the recursion
    x = theta.detach().clone()
    f, g = _value_grad(J, x)
    for _ in range(iters):
        q, a = g.clone(), []
        for i in range(h - 1, -1, -1):           # newest pair is last
            a.append(rho[:, i] * (Sh[:, i] * q).sum(1))
            q = q - a[-1][:, None] * Yh[:, i]
        # initial inverse Hessian: s.y / y.y of the newest pair; without history a gradient step of length lr
        gamma = torch.where(rho[:, -1] > 0, (Sh[:, -1] * Yh[:, -1]).sum(1) / (Yh[:, -1] ** 2).sum(1),
                            c["lr"] / g.norm(dim=1).clamp_min(1e-30))      # zero gradient -> zero step
        r = gamma[:, None] * q
        for i in range(h):
            r = r + (a[h - 1 - i] - rho[:, i] * (Yh[:, i] * r).sum(1))[:, None] * Sh[:, i]
        d = -r
        fc, gc = _value_grad(J, (x[:, None] + steps[:, None] * d[:, None]).reshape(-1, D))
        fc, gc = fc.reshape(M, ns), gc.reshape(M, ns, D)
        ok = fc <= f[:, None] + c["armijo"] * steps * (g * d).sum(1, keepdim=True)
        found = ok.any(1)
        j = ok.int().argmax(1)                   # first True = largest step
        alpha = torch.where(found, steps[j], torch.zeros_like(f))
        x_new = x + alpha[:, None] * d
        f_new = torch.where(found, fc[rows, j], f)
        g_new = torch.where(found[:, None], gc[rows, j], g)
        s, y = x_new - x, g_new - g
        sy = (s * y).sum(1)
        keep = found & (sy > 1e-10)              # curvature condition keeps the inverse Hessian positive definite
        # failed line search: drop the history, so the next step is a plain gradient step
        Sh = torch.where(keep[:, None, None], torch.cat([Sh[:, 1:], s[:, None]], 1), Sh * found[:, None, None])
        Yh = torch.where(keep[:, None, None], torch.cat([Yh[:, 1:], y[:, None]], 1), Yh * found[:, None, None])
        rho = torch.where(keep[:, None], torch.cat([rho[:, 1:], (1.0 / sy)[:, None]], 1), rho * found[:, None])
        x, f, g = x_new, f_new, g_new
    return x


EXPLOITERS = {"adam": _adam, "lbfgs": _lbfgs}


def exploit(name, J, theta, budget_iters, cfg):
    """Gradient solver (torch autograd). name: "none" | "adam" | "lbfgs". Levenberg-Marquardt and SLSQP are not
    implemented.

    theta (M, D): M seeds, run in lockstep for budget_iters iterations; batched results equal per-seed runs because
    J summed over seeds is separable and nothing else couples the rows. Returns (M, D), per seed the best iterate.
    Adam: one J call with gradient per iteration. L-BFGS: one J call with gradient of M*ls_steps rows per
    iteration (the line-search candidates), plus one of M rows at the start.

    cfg keys (defaults in DEFAULTS): lr 0.01 (Adam step; length of L-BFGS's first gradient step); adam (0.9, 0.999,
    1e-8); history 10; ls_steps 10; armijo 1e-4. exploit_iters 20 is the caller's per-level count (budget_iters).
    """
    if name == "none":
        return theta
    return EXPLOITERS[name](J, theta, budget_iters, {**DEFAULTS, **cfg})


# ---------------------------------------------------------------------------------------------- portfolio
def portfolio(arms, J, budget_evals):
    """Successive halving over (seed, explore, exploit) arms (§6.3 portfolio mode).

    arms: list of dicts {"theta": (D,) tensor, "explore": name, "exploit": name, "cfg": solver cfg, "gen":
    torch.Generator}. Each round runs every surviving arm from its current theta with budget B: explore with B
    evaluations, then exploit with B * exploit_iters / (n_samples * iters) iterations (at least 1), the 10:20 ratio
    of §6.4's per-level budgets. Then it keeps the better half by J (rounded up, ties by arm order) and doubles B,
    starting from B = budget_evals, until one arm is left. Arms sharing explore, exploit, cfg object and generator
    run batched as the seeds of one call, in arm order; J must therefore not keep per-row state here.

    Returns the winning arm: its dict with "theta" (D,) refined, "J" (float) and "index" (position in `arms`).
    """
    alive = [dict(a, theta=a["theta"].reshape(1, -1), index=i) for i, a in enumerate(arms)]
    B = budget_evals
    while True:
        groups = {}
        for a in alive:
            groups.setdefault((a["explore"], a["exploit"], id(a["cfg"]), id(a["gen"])), []).append(a)
        for g in groups.values():
            a0, th = g[0], torch.cat([a["theta"] for a in g])
            c = {**DEFAULTS, **a0["cfg"]}
            th = explore(a0["explore"], J, th, B, a0["gen"], a0["cfg"])
            th = exploit(a0["exploit"], J, th, max(1, round(B * c["exploit_iters"] / (c["n_samples"] * c["iters"]))),
                         a0["cfg"])
            with torch.no_grad():
                Jg = J(th)
            for a, t, j in zip(g, th, Jg.tolist()):
                a["theta"], a["J"] = t[None], j
        alive = sorted(alive, key=lambda a: (a["J"], a["index"]))[:(len(alive) + 1) // 2]
        if len(alive) == 1:
            return dict(alive[0], theta=alive[0]["theta"][0])
        B *= 2
