"""The objective J_l of TRACE_method.md §5.2 and the certificate's dense term check (§7.2).

J_l(theta) = sum_prefer kappa_i / |T_i| sum_j rho_i(v_ij^2 / s_l^2) + sum_{safety, require} 1/|T_i| sum_j psi(u_ij / s_l)
with u = r / sigma, rho = Huber (identity for regularizers) and psi the augmented-Lagrangian term with one multiplier
per (seed, term, sample, component). J is called with seed-major rows (solvers.py): a call with M*k rows on M seeds
broadcasts the multipliers over the k rows of each seed.

J is launch bound (a few hundred tiny kernels per call), so the search evaluates it in units: clearance terms that
share primitive, role, sigma, path scope, geometry and zone are merged into one field lookup with per-sphere margins
and zone floors. Their per-term sample means add up to the merged one, so J is unchanged. Scopes are index tensors
(none for the full path), so no call synchronizes with the device.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch

from .contract import GROUPS as GROUP_NAMES
from .primitives import (PRIMITIVES, ZONE, Body, EvalContext, Source, pose_tensors, residual, scope_mask,
                         violation)

LEVELS = [{"s": 8, "beta": 0.08, "K": 4, "mu": 1}, {"s": 4, "beta": 0.04, "K": 5, "mu": 4},
          {"s": 2, "beta": 0.02, "K": 7, "mu": 16}, {"s": 1, "beta": 0.0, "K": 11, "mu": 64}]


def _sq(x):
    return x * x                      # not x ** 2: its backward copies the exponent to the device on every call


def huber(x):
    """Robust kernel on a squared normalized residual x = v^2: x inside the tolerance, 2 sqrt(x) - 1 beyond it."""
    return torch.where(x <= 1, x, 2 * x.clamp(min=1).sqrt() - 1)


@dataclass
class Unit:
    """One summand of J: a term, or merged clearance terms (they share role, shape, sigma, kappa and src)."""
    id: str
    term: object            # the (first) term: role, shape, sigma, kappa, src
    args: dict
    idx: object             # sample indices of the scope, None for the whole path
    slices: dict            # term id -> its components in the unit's residual


def _merge_key(t, a):
    if t.prim not in ("clearance", "plane_clearance") or t.role == "prefer" or t.scope != "path":
        return None
    f = a.get("field") or {}
    geo = ((tuple(sorted(f.get("exclude", ()))), f.get("only"), bool(f.get("unknown", False)))
           if t.prim == "clearance" else float(a["z"]))
    return t.prim, t.role, float(t.sigma), geo, float(a.get("zone", ZONE))


class Objective:
    def __init__(self, program, snap, bodies, spline, start, end, device, n_eval=32, margins=None, hard="al",
                 levels=LEVELS, dtype=torch.float32, flags=None):
        self.program, self.snap, self.bodies, self.spline = program, snap, bodies, spline
        self.terms = list(program.terms)
        self.start, self.end = start, end                     # numpy poses: {R, p} and {R, p, approach}
        self.device, self.dtype = torch.device(device), dtype
        self.ts = np.linspace(0.0, 1.0, n_eval)
        self.margins, self.hard, self.levels, self.flags = margins, hard, levels, flags or {}
        self.body = Body(bodies, self.device, dtype)
        self.st = pose_tensors(start["R"], start["p"], None, dtype, self.device)
        self.en = pose_tensors(end["R"], end["p"], end.get("approach"), dtype, self.device)
        self.idx = {}
        for t in self.terms:
            m = scope_mask(t.scope, self.ts)
            self.idx[t.id] = None if m.all() else torch.as_tensor(np.nonzero(m)[0], device=self.device)
        self.args = {t.id: t.args for t in self.terms}
        self.lam, self.M, self.level, self.evals, self.units = {}, None, None, 0, []

    # ------------------------------------------------------------------ levels, units and multipliers
    def set_level(self, level, spline):
        L = self.levels[level]
        self.level, self.s, self.mu, self.spline = level, L["s"], L["mu"], spline
        self.src = Source(self.snap, "grid", L["beta"], self.device, self.dtype)
        self.args = {t.id: self._level_args(t) for t in self.terms}
        self.units = self._units()

    def _level_args(self, t):
        """The zone floors ('do not get closer' than at the ends) measured on this level's own field, so the pinned
        ends are feasible in the search's geometry too; the certificate keeps the exact ones."""
        if t.prim != "clearance" or not ({"c_start", "c_end"} & set(t.args)):
            return t.args
        idx, idx_np = self.body.sel(t.args["body"])
        if len(idx_np) == 0:
            return t.args
        f, a = self.src.field(t.args.get("field")), dict(t.args)
        for key, pose in (("c_start", self.st), ("c_end", self.en)):
            if key in a:
                a[key] = (f(self.body.c[idx] @ pose["R"].T + pose["p"]) - self.body.r[idx]).min()
        return a

    def _units(self):
        units, groups = [], {}
        for t in self.terms:
            key = _merge_key(t, self.args[t.id])
            if key is None:
                units.append(Unit(t.id, t, self.args[t.id], self.idx[t.id], {t.id: slice(None)}))
            elif len(self.body.sel(t.args["body"])[1]):
                groups.setdefault(key, []).append(t)
        for ts in groups.values():
            sel = [self.body.sel(t.args["body"]) for t in ts]
            n = np.cumsum([0] + [len(s[1]) for s in sel])
            a0 = self.args[ts[0].id]
            full = lambda i, v: torch.as_tensor(v, dtype=self.dtype, device=self.device).expand(n[i + 1] - n[i])
            a = {"idx": torch.cat([s[0] for s in sel]), "idx_np": np.concatenate([s[1] for s in sel]),
                 "margin": torch.cat([full(i, self.args[t.id]["margin"]) for i, t in enumerate(ts)]),
                 "zone": a0.get("zone", ZONE), "field": a0.get("field"), "z": a0.get("z")}
            for key in ("c_start", "c_end"):
                if any(key in self.args[t.id] for t in ts):
                    a[key] = torch.cat([full(i, self.args[t.id].get(key, np.inf)) for i, t in enumerate(ts)])
            units.append(Unit("+".join(t.id for t in ts), ts[0], a, None,
                              {t.id: slice(n[i], n[i + 1]) for i, t in enumerate(ts)}))
        return units

    def reset(self, M):
        self.lam, self.M = {}, M

    def select(self, idx):
        self.lam = {k: v[idx] for k, v in self.lam.items()}
        self.M = len(idx)

    def add_seeds(self, n):
        self.lam = {k: torch.cat([v, v.new_zeros(n, *v.shape[1:])]) for k, v in self.lam.items()}
        self.M += n

    def _term_lam(self, term_id):
        for u in self.units:
            if term_id in u.slices and u.id in self.lam:
                return self.lam[u.id][..., u.slices[term_id]]
        return None

    def demote(self, term_id):
        """Copy with that term as prefer (leave-one-out, §7.3); the other terms keep their multipliers."""
        o = copy.copy(self)
        o.terms = [replace(t, role="prefer") if t.id == term_id else t for t in self.terms]
        o.units = o._units()
        o.lam = {}
        for u in o.units:
            parts = [self._term_lam(tid) for tid in u.slices]
            if u.term.role != "prefer" and all(p is not None for p in parts):
                o.lam[u.id] = torch.cat(parts, -1)
        return o

    def context(self, theta, ts, src):
        return EvalContext(self.spline.eval(theta, ts), ts, self.body, src, self.st, self.en,
                           bool(self.program.holding), self.program.allow_contact)

    def _u(self, ctx, u):
        """Normalized residual (M', samples in scope, d) of a unit at this level."""
        r = ctx.full(PRIMITIVES[u.term.prim].fn(ctx, u.args))
        if u.idx is not None:
            r = r.index_select(1, u.idx)
        return r * (1.0 / (u.term.sigma * self.s))

    # ------------------------------------------------------------------ J
    def __call__(self, theta):
        self.evals += theta.shape[0]
        ctx = self.context(theta, self.ts, self.src)
        J = 0.0
        for u in self.units:
            t, x = u.term, self._u(ctx, u)
            if t.role == "prefer":
                v2 = _sq(x if t.shape == "eq" else x.clamp(min=0)).sum(-1)
                J = J + t.kappa * (v2 if t.src == "regularizer" else huber(v2)).mean(1)
                continue
            x = x.reshape(self.M, -1, *x.shape[1:])            # (seeds, rows per seed, samples, d)
            lam = self.lam.get(u.id)
            if lam is None:
                psi = (0.5 * self.mu) * _sq(x if t.shape == "eq" else x.clamp(min=0))
            elif t.shape == "eq":
                psi = (lam[:, None] + (0.5 * self.mu) * x) * x
            else:
                psi = (_sq((lam[:, None] + self.mu * x).clamp(min=0)) - _sq(lam[:, None])) * (0.5 / self.mu)
            J = J + psi.sum(-1).mean(-1).reshape(-1)
        return J

    def update_multipliers(self, theta):
        """lambda <- lambda + mu u (eq), max(0, lambda + mu u) (ineq), at the current seeds; frozen at 0 for the
        fixed-penalty variant."""
        if self.hard != "al":
            return
        with torch.no_grad():
            ctx = self.context(theta, self.ts, self.src)
            for u in self.units:
                if u.term.role == "prefer":
                    continue
                lam = self.mu * self._u(ctx, u)
                if u.id in self.lam:
                    lam = lam + self.lam[u.id]
                self.lam[u.id] = lam if u.term.shape == "eq" else lam.clamp(min=0)

    def multiplier_size(self, term_id, i=0):
        lam = self._term_lam(term_id)
        return 0.0 if lam is None else float(lam[i].abs().mean())

    # ------------------------------------------------------------------ certificate
    def dense_report(self, theta_1, n=128):
        """Per term max_violation (original sigma; clearance on exact geometry), at_t, r_max (physical), plus the
        per-sample clearance slack per group (clearance - guard - perception; plane: - guard_plane) for retiming and
        the per-sample mask where a relaxed known-space term is violated."""
        f64, ts = torch.float64, np.linspace(0.0, 1.0, n)
        th = torch.as_tensor(theta_1, dtype=f64).reshape(1, -1).cpu()
        sp = self.spline.like(device="cpu", dtype=f64)
        body = Body(self.bodies, "cpu", f64)
        ctx = EvalContext(sp.eval(th, ts), ts, body, Source(self.snap, "exact", dtype=f64),
                          pose_tensors(self.start["R"], self.start["p"], None, f64, "cpu"),
                          pose_tensors(self.end["R"], self.end["p"], self.end.get("approach"), f64, "cpu"),
                          bool(self.program.holding), self.program.allow_contact)
        g = self.margins or {}
        rows, slack, slow = [], {}, np.zeros(n, bool)
        for t in self.terms:
            r = residual(ctx, t)[0]
            m = scope_mask(t.scope, ts)
            sig = t.args.get("sigma_accept", t.sigma)
            v = violation(r, t.shape, sig).numpy() * m
            j = int(v.argmax())
            rr = r.numpy()[m]
            row = {"term": t.id, "prim": t.prim, "role": t.role, "src": t.src, "kappa": t.kappa, "level": t.level,
                   "relax_to": t.relax_to, "entity": t.entity, "max_violation": float(v[j]), "at_t": float(ts[j]),
                   "r_max": float(np.linalg.norm(rr, axis=-1).max() if t.shape == "eq" else rr.max()),
                   "sigma": sig, "certifiable": self.flags.get(t.id, {}).get("certifiable", True)}
            if t.prim in ("clearance", "plane_clearance"):
                idx, idx_np = body.sel(t.args["body"])
                if len(idx_np):
                    row["at_x"] = ctx.x[0, j, idx[int(r[j].argmax())]].numpy()
                    spec = t.args.get("field") or {}
                    if t.role == "safety" and spec.get("only") is None and not spec.get("unknown", False):
                        # group clearance = margin - residual, on the certificate's own geometry
                        c = (ctx.zone_margin(t.args["margin"], t.args)[0] - r).numpy()
                        for gi in np.unique(body.groups[idx_np]):
                            name = GROUP_NAMES[gi]
                            sel = body.groups[idx_np] == gi
                            off = (g.get("guard_plane", {}).get(name, 0.0) if t.prim == "plane_clearance"
                                   else g.get("guard", {}).get(name, 0.0) + g.get("perception", 0.0))
                            sl = c[:, sel].min(1) - off
                            slack[name] = np.minimum(slack[name], sl) if name in slack else sl
            if t.args.get("slow"):
                slow |= v > 1
            rows.append(row)
        return {"rows": rows, "slack": slack, "slow": slow, "ts": ts, "R": ctx.R[0].numpy(), "p": ctx.p[0].numpy()}

