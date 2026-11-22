"""Residual program, uncertainty fusion and the numeric program tests (TRACE_method.md §5.1, §5.5, §7.1).

The program tests are the only checks on a program: numeric, model-free, milliseconds. Terms the harness writes
from the scene and the robot (role safety, src regularizer) are trusted and only linted; the others (language and
goal terms) get the range, theta-dependence, invariance and start-feasibility tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .contract import ROLES
from .primitives import PRIMITIVES, Body, EvalContext, Source, pose_tensors, residual, scope_mask, violation
from .se3 import rotz
from .spline import SE3Spline

CFG = {"sigma_bounds": {"m": (1e-4, 0.5), "rad": (1e-3, 1.6)},
       "param_bounds": {"m": (-2.0, 2.0), "rad": (0.0, np.pi)},     # cm-for-m / degree-for-radian in a parameter
       "max_violation": 1e3, "n_random": 8, "n_invariance": 4, "noise_pos": 0.03, "noise_rot": 0.15,
       "inv_tol": 1e-5, "inv_shift": 0.3, "n_eval": 32}
POINTS = ("eef", "payload.base", "payload.centroid")
AXES = ("eef.x", "eef.y", "eef.z", "world.up", "payload.up", "approach")
KEYPOINTS = ("centroid", "top_center", "base")


@dataclass
class Program:
    terms: list
    holding: str | None = None
    allow_contact: list = field(default_factory=list)   # [{body: "finger", entity, zone}]


def fuse(program, snap, z=1.64):
    """Asymmetric perception fusion (§5.5): attract terms get sigma^2 + lambda_max(Sigma), repel terms a parameter
    tightened by z sqrt(lambda_max(Sigma)); acceptance keeps the original sigma (args["sigma_accept"]). Safety terms
    already carry the ladder's perception bound. Terms naming an unknown entity are left for the lint test."""
    out, flags = [], {}
    for t in program.terms:
        if t.entity is None or t.effect is None or t.role == "safety" or t.entity not in snap.entities:
            out.append(t)
            continue
        S = np.asarray(snap.entities[t.entity].cov, float)
        if t.args.get("perp") is not None:
            a = np.asarray(t.args["perp"], float)
            P = np.eye(3) - np.outer(a, a) / (a @ a)
            S = P @ S @ P
        sd = float(np.sqrt(max(np.linalg.eigvalsh(S).max(), 0.0)))
        args, sigma = dict(t.args, sigma_accept=t.sigma), t.sigma
        if t.effect == "attract":
            sigma = float(np.sqrt(t.sigma ** 2 + sd ** 2))
        else:
            spec = PRIMITIVES[t.prim]
            p = t.param or spec.param
            args["param0"] = args[p]                         # quotes speak in the harness's level values
            args[p] = args[p] - spec.sign * z * sd           # the stricter direction of the parameter
        out.append(replace(t, args=args, sigma=sigma))
        flags[t.id] = {"certifiable": sd <= t.sigma, "uncertainty": sd, "tolerance": t.sigma, "entity": t.entity}
    return Program(out, program.holding, program.allow_contact), flags


# --------------------------------------------------------------------------------------------- tests
def _f(test, term, detail):
    return {"test": test, "term": term, "detail": detail}


def _scope_ok(s):
    if s in ("path", "terminal"):
        return True
    return isinstance(s, (list, tuple)) and len(s) == 2 and 0.0 <= s[0] <= s[1] <= 1.0


def _axis_ok(a):
    if isinstance(a, str):
        return a.lstrip("-") in AXES
    return np.asarray(a).shape == (3,)


def _ref_problem(b, snap):
    if not isinstance(b, str):
        return None if np.asarray(b).shape == (3,) else f"bad point {b!r}"
    if b in ("end", "start"):
        return None
    eid, _, kp = b.rpartition(".")
    if eid not in snap.entities:
        return f"unknown entity {eid!r}"
    if kp not in snap.entities[eid].keypoints and kp not in KEYPOINTS:
        return f"unknown keypoint {b!r}"
    return None


def lint(terms, snap):
    out = []
    for t in terms:
        spec = PRIMITIVES.get(t.prim)
        if spec is None:
            out.append(_f("lint", t.id, f"unknown primitive {t.prim!r}"))
            continue
        if t.shape != spec.shape:
            out.append(_f("lint", t.id, f"shape {t.shape!r}, {t.prim} is {spec.shape!r}"))
        if t.role not in ROLES:
            out.append(_f("lint", t.id, f"unknown role {t.role!r}"))
        if t.role == "safety" and t.src == "language":
            out.append(_f("lint", t.id, "a language term cannot be a safety term"))
        if not _scope_ok(t.scope):
            out.append(_f("lint", t.id, f"scope {t.scope!r} outside [0, 1]"))
        if t.entity is not None and t.entity not in snap.entities:
            out.append(_f("lint", t.id, f"unknown entity {t.entity!r}"))
        a = t.args
        if "a" in a and a["a"] not in POINTS:
            out.append(_f("lint", t.id, f"unknown point {a['a']!r}"))
        if "b" in a and _ref_problem(a["b"], snap):
            out.append(_f("lint", t.id, _ref_problem(a["b"], snap)))
        for k in ("u", "v", "axis", "n", "perp"):
            if a.get(k) is not None and not _axis_ok(a[k]):
                out.append(_f("lint", t.id, f"unknown axis {a[k]!r}"))
        only = (a.get("field") or {}).get("only")
        if only is not None and only not in snap.entities:
            out.append(_f("lint", t.id, f"unknown entity {only!r}"))
    return out


def _ctx(sp, theta, ts, body, src, approach, holding, allow, xf=None):
    f64 = torch.float64
    R0, p0, Rg, pg = sp.R0, sp.p0, sp.Rg, sp.pg
    return EvalContext(sp.eval(theta, ts), ts, body, src, pose_tensors(R0, p0, None, f64, "cpu"),
                       pose_tensors(Rg, pg, approach, f64, "cpu"), holding, allow)


def run_program_tests(program, snap, make_spline, bodies, rng, approach=None):
    """[{test, term, detail}] from lint, range, theta-dependence, invariance and start feasibility (§7.1).
    make_spline() -> a float64 CPU SE3Spline from the start to the (first) end with a free end, so terminal terms
    can depend on theta; approach: that end's approach axis (for "approach" references)."""
    found = lint(program.terms, snap)
    if found:
        return found
    f64, c = torch.float64, CFG
    sp, ts = make_spline(), np.linspace(0.0, 1.0, c["n_eval"])
    body, hold, allow = Body(bodies, "cpu", f64), bool(program.holding), program.allow_contact
    grid = Source(snap, "grid", 0.0, "cpu", f64)
    checked = [t for t in program.terms if t.role != "safety" and t.src != "regularizer"]
    mask = {t.id: scope_mask(t.scope, ts) for t in program.terms}

    # range: tolerances and parameters in sane bounds per unit; violations on the straight seed not astronomical
    for t in program.terms:
        spec = PRIMITIVES[t.prim]
        unit, p = t.unit or spec.unit, t.param or spec.param
        if unit in c["sigma_bounds"] and not c["sigma_bounds"][unit][0] <= t.sigma <= c["sigma_bounds"][unit][1]:
            found.append(_f("range", t.id, f"sigma {t.sigma:g} {unit} outside {c['sigma_bounds'][unit]}"))
        if unit in c["param_bounds"] and p not in (None, "sigma") and p in t.args \
                and not c["param_bounds"][unit][0] <= t.args[p] <= c["param_bounds"][unit][1]:
            found.append(_f("range", t.id, f"{p} {t.args[p]:g} {unit} outside {c['param_bounds'][unit]}"))
    th0 = sp.straight_theta()[None]
    ctx = _ctx(sp, th0, ts, body, grid, approach, hold, allow)
    for t in checked:
        v = float((violation(residual(ctx, t)[0], t.shape, t.sigma).numpy() * mask[t.id]).max())
        if v >= c["max_violation"]:
            found.append(_f("range", t.id, f"violation {v:.3g} on the straight seed (unit mistake?)"))

    # theta-dependence: the trajectory must be able to change the residual
    n = sp.n_ctrl * 3
    noise = np.concatenate([rng.normal(0, c["noise_pos"], (c["n_random"], n)),
                            rng.normal(0, c["noise_rot"], (c["n_random"], n))], 1)
    th = (th0 + torch.as_tensor(noise)).requires_grad_(True)
    ctx = _ctx(sp, th, ts, body, grid, approach, hold, allow)
    for t in checked:
        r = residual(ctx, t)[:, torch.as_tensor(mask[t.id])]
        g = torch.autograd.grad((r * torch.as_tensor(rng.normal(size=r.shape))).sum(), th, retain_graph=True,
                                allow_unused=True)[0] if r.requires_grad else None
        if g is None or not bool((g.abs() > 0).any()):
            found.append(_f("theta_dependence", t.id, "the trajectory cannot change this residual (static entities, "
                                                       "or a payload term while nothing is held?)"))

    # invariance: scene and trajectory moved together by a yaw and a horizontal shift
    yaw = float(rng.uniform(-np.pi, np.pi))
    shift = np.r_[rng.uniform(-c["inv_shift"], c["inv_shift"], 2), 0.0]
    Rz, tt = rotz(yaw), torch.as_tensor(shift)
    Rn = Rz.numpy()
    sp2 = SE3Spline(Rn @ sp.R0, Rn @ sp.p0 + shift, Rn @ sp.Rg, Rn @ sp.pg + shift, sp.K, sp.pin_end, sp.phi_max,
                    "cpu", f64)
    thb = th.detach()[:c["n_invariance"]]
    P, F = sp.ctrl(thb)
    th2 = sp2.theta_of(P @ Rz.T + tt, F)
    ctx1 = _ctx(sp, thb, ts, body, grid, approach, hold, allow)
    ctx2 = _ctx(sp2, th2, ts, body, Source(snap, "grid", 0.0, "cpu", f64, xf=(Rz, tt)),
                None if approach is None else Rn @ np.asarray(approach, float), hold, allow)
    for t in checked:      # equality residuals are world vectors that turn with the scene: compare their norms
        r1, r2 = residual(ctx1, t), residual(ctx2, t)
        if t.shape == "eq":
            r1, r2 = r1.norm(dim=-1), r2.norm(dim=-1)
        d = float((r1 - r2)[:, mask[t.id]].abs().max())
        if d > c["inv_tol"]:
            found.append(_f("invariance", t.id, f"residual changes by {d:.3g} when scene and path move together "
                                                 "(hard-coded world coordinate?)"))

    # start feasibility: a path-scope require term must hold at the fixed start pose
    ctx = _ctx(sp, th0, np.zeros(1), body, Source(snap, "exact", dtype=f64), approach, hold, allow)
    for t in checked:
        if t.role == "require" and (t.scope == "path" or (not isinstance(t.scope, str) and t.scope[0] == 0.0)):
            v = float(violation(residual(ctx, t)[0], t.shape, t.args.get("sigma_accept", t.sigma))[0])
            if v > 1.0:
                found.append(_f("start_feasibility", t.id, f"violated at the start pose ({v:.2f} tolerances)"))
    return found
