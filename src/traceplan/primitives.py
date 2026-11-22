"""Primitive library (TRACE_method.md §5.3, shared vocabulary of plan, guard and checks).

Every primitive maps an EvalContext to residuals (M, N, d) in physical units; constants may come back broadcastable
and are expanded by `EvalContext.full`. Scene access goes through a Source, so the same code serves the search
(differentiable blurred grids), the certificate (exact geometry, float64) and the invariance test (scene moved by
a rigid transform: query points get the inverse transform, entity points, directions and rotations the forward one,
literal world points nothing, which is what exposes hard-coded coordinates).

The search calls J thousands of times on small batches, so per-call overhead matters more than arithmetic: literal
constants are cached on the Source (no host-to-device copy per call), and the clearance primitives also accept
merged arguments (explicit sphere indices with per-sphere margins) so the objective can fuse terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .contract import GROUPS
from .scene import GridField, field_distance, grid_field
from .se3 import angle_between, so3_log, vee

ZONE = 0.05


@dataclass
class Spec:
    shape: str
    fn: Callable
    param: str | None       # args entry a conflict quote is expressed in ("sigma" for equalities)
    unit: str | None
    sign: int = 0           # residual moves with the parameter as -sign: needed = value + sign * max_t r(t)


# --------------------------------------------------------------------------------------------- scene access
class Source:
    def __init__(self, snap, kind="grid", beta=0.0, device="cpu", dtype=torch.float32, xf=None):
        self.snap, self.kind, self.beta, self.device, self.dtype = snap, kind, beta, torch.device(device), dtype
        self.xf = xf                                    # (Rz (3,3), t (3,)) tensors: the scene moved by x -> Rz x + t
        self._c = {}

    def const(self, x):
        a = np.asarray(x, float)
        key = (a.shape, a.tobytes())
        if key not in self._c:
            self._c[key] = torch.as_tensor(a, dtype=self.dtype, device=self.device)
        return self._c[key]

    def field(self, spec):
        if self.kind == "grid":
            f = grid_field(self.snap, spec, self.beta, self.device, self.dtype)
        else:
            def f(x):
                d = field_distance(self.snap, spec, x.detach().cpu().numpy().reshape(-1, 3))
                return torch.as_tensor(d, dtype=x.dtype, device=x.device).reshape(x.shape[:-1])
        if self.xf is None:
            return f
        Rz, t = self.xf
        return lambda x: f((x - t) @ Rz)

    def value(self, key):
        gf = self.snap.__dict__.setdefault("_trace_cache", {}).setdefault(
            ("value", key, str(self.device), self.dtype),
            GridField(np.asarray(self.snap.meta[key], np.float32), self.snap.origin, self.snap.res, self.device,
                      self.dtype))
        if self.xf is None:
            return gf
        Rz, t = self.xf
        return lambda x: gf((x - t) @ Rz)

    def keypoint(self, ref):
        eid, kp = ref.rsplit(".", 1)
        v = self.const(self.snap.entities[eid].keypoint(kp))
        return v if self.xf is None else self.xf[0] @ v + self.xf[1]

    def direction(self, d):
        d = np.asarray(d, float)
        v = self.const(d / np.linalg.norm(d))
        return v if self.xf is None else self.xf[0] @ v

    def rotation(self, R):
        R = self.const(R)
        return R if self.xf is None else self.xf[0] @ R


class Body:
    """Sphere set as tensors; groups stay numpy so selections need no device sync."""

    def __init__(self, spheres, device, dtype):
        self.c = torch.as_tensor(spheres.centers, dtype=dtype, device=device)
        self.r = torch.as_tensor(spheres.radii, dtype=dtype, device=device)
        self.groups = np.asarray(spheres.groups)
        self.spheres = spheres
        self._sel = {}

    def sel(self, body):
        if body not in self._sel:
            idx = np.arange(len(self.groups)) if body == "all" else np.nonzero(self.groups == GROUPS.index(body))[0]
            self._sel[body] = (torch.as_tensor(idx, device=self.c.device), idx)
        return self._sel[body]


# --------------------------------------------------------------------------------------------- context
class EvalContext:
    def __init__(self, out, ts, body, src, start, end, holding, allow=()):
        self.R, self.p, self.v, self.acc = out["R"], out["p"], out["v"], out["acc"]
        self.M, self.N = self.p.shape[:2]
        self.ts, self.body, self.src, self.holding, self.allow = np.asarray(ts, float), body, src, holding, allow
        self.R0, self.p0 = start["R"], start["p"]
        self.Re, self.pe, self.approach = end["R"], end["p"], end.get("approach")
        self._x, self._near = None, {}

    def full(self, r):
        if r.shape[-1] == 0:
            return self.p.new_zeros(self.M, self.N, 1)
        return r.expand(self.M, self.N, r.shape[-1])

    @property
    def x(self):
        if self._x is None:
            self._x = torch.einsum("mnij,sj->mnsi", self.R, self.body.c) + self.p[:, :, None]
        return self._x

    def near(self, which, zone):
        """(M, N) samples whose eef position is within `zone` of the start ("s") or end ("e") position."""
        if (which, zone) not in self._near:
            self._near[which, zone] = (self.p - (self.p0 if which == "s" else self.pe)).norm(dim=-1) < zone
        return self._near[which, zone]

    def _pl(self):
        return np.nonzero(self.body.groups == GROUPS.index("payload"))[0]

    def point(self, name):
        if name == "eef":
            return self.p
        pl = self._pl()
        if len(pl) == 0:
            return self.p.new_zeros(1, 1, 3)
        c, r = self.body.c[pl], self.body.r[pl]
        if name == "payload.base":
            i = torch.argmin((c @ self.R0.T)[:, 2] - r, 0, keepdim=True)   # lowest payload sphere at the start pose
            c = c[i] - r[i, None] * self.R0[2]                      # its bottom: world -z at the start, eef frame
        else:
            c = c.mean(0, keepdim=True)
        if not self.holding:                                       # not attached: the object stays put
            return (self.R0 @ c[0] + self.p0).reshape(1, 1, 3)
        return self.R @ c[0] + self.p

    def ref(self, b):
        if isinstance(b, str):
            if b == "end":
                return self.pe.reshape(1, 1, 3)
            if b == "start":
                return self.p0.reshape(1, 1, 3)
            return self.src.keypoint(b).reshape(1, 1, 3)
        return self.src.const(b).reshape(1, 1, 3)

    def axis(self, name, args):
        if isinstance(name, str):
            if name.startswith("-"):
                return -self.axis(name[1:], args)
            if name.startswith("eef."):
                return self.R[..., :, "xyz".index(name[-1])]
            if name == "world.up":
                return self.src.const([0.0, 0.0, 1.0]).reshape(1, 1, 3)
            if name == "approach":
                return self.approach.reshape(1, 1, 3)
            if name == "payload.up":
                u0 = self.src.direction(args.get("u0", [0.0, 0, 1]))
                return self.R @ (self.R0.T @ u0) if self.holding else u0.reshape(1, 1, 3)
            raise KeyError(name)
        return self.src.direction(name).reshape(1, 1, 3)

    def zone_margin(self, margin, a):
        """'Do not get closer' near both ends: within `zone` of the start (end) position the margin drops to the
        start (end) clearance minus 1 mm. Scalars or per-component vectors (merged terms)."""
        zone = a.get("zone", ZONE)
        m = margin if torch.is_tensor(margin) else self.p.new_full((1,), float(margin))
        for key, which in (("c_start", "s"), ("c_end", "e")):
            if key in a:
                c = a[key]
                lim = torch.minimum(m, c - 0.001) if torch.is_tensor(c) else m.clamp(max=c - 0.001)
                m = torch.where(self.near(which, zone)[..., None], lim, m)
        return m


# --------------------------------------------------------------------------------------------- primitives
def reach_point(ctx, a):
    r = ctx.point(a.get("a", "eef")) - ctx.ref(a["b"])
    if a.get("offset") is not None:
        r = r - ctx.src.const(a["offset"])
    if a.get("perp") is not None:
        ax = ctx.axis(a["perp"], a)
        r = r - (r * ax).sum(-1, keepdim=True) * ax
    return r


def reach_rot(ctx, a):
    Rg = ctx.Re if isinstance(a["R"], str) else ctx.src.rotation(a["R"])
    return so3_log(Rg.transpose(-1, -2) @ ctx.R)


def align_axis(ctx, a):
    return angle_between(*torch.broadcast_tensors(ctx.axis(a["u"], a), ctx.axis(a["v"], a)))[..., None]


def tilt_limit(ctx, a):
    return align_axis(ctx, a) - a["max"]


def above(ctx, a):
    return (ctx.ref(a["b"])[..., 2:] + a["h"]) - ctx.point(a.get("a", "eef"))[..., 2:]


def _spheres(ctx, a):
    return (a["idx"], a["idx_np"]) if "idx" in a else ctx.body.sel(a["body"])


def clearance(ctx, a):
    idx, idx_np = _spheres(ctx, a)
    x, r = ctx.x.index_select(2, idx), ctx.body.r[idx]
    spec = a.get("field") or {}
    m = ctx.zone_margin(a["margin"], a)
    res = m + r - ctx.src.field(spec)(x)
    if spec.get("only") is None and not spec.get("unknown", False):
        for e in ctx.allow:                       # intended finger contact with the step's target, end zone only
            g = ctx.body.groups[idx_np] == GROUPS.index(e["body"])
            if not g.any():
                continue
            phi_x = ctx.src.field({**spec, "exclude": list(spec.get("exclude", ())) + [e["entity"]]})(x)
            alt = torch.maximum(m + r - phi_x, 0.001 + r - ctx.src.field({"only": e["entity"]})(x))
            sel = ctx.near("e", e.get("zone", ZONE))[..., None] & torch.as_tensor(g, device=x.device)
            res = torch.where(sel, alt, res)
    return res


def plane_clearance(ctx, a):
    idx, _ = _spheres(ctx, a)
    return ctx.zone_margin(a["margin"], a) + ctx.body.r[idx] - (ctx.x[..., 2].index_select(2, idx) - a["z"])


def distance_band(ctx, a):
    d = (ctx.point(a.get("a", "eef")) - ctx.ref(a["b"])).norm(dim=-1, keepdim=True)
    return torch.maximum(a["lo"] - d, d - a["hi"])


def half_space(ctx, a):
    return a["d"] - (ctx.point(a.get("a", "eef")) * ctx.axis(a["n"], a)).sum(-1, keepdim=True)


def approach_cone(ctx, a):
    ang = angle_between(*torch.broadcast_tensors(ctx.v, ctx.axis(a["axis"], a)))
    on = ctx.near("e", a.get("zone", ZONE)) & (ctx.v.norm(dim=-1) > 1e-6)
    return torch.where(on, ang - a["half_angle"], torch.zeros_like(ang))[..., None]


def value_map(ctx, a):
    return ctx.src.value(a["grid_key"])(ctx.p)[..., None]


def smooth_pos(ctx, a):
    return ctx.acc


def smooth_rot(ctx, a):
    """Finite-difference angular acceleration. Consecutive samples differ by small rotations (~0.1 rad at N = 32),
    where the skew part sin(angle) * axis equals Log to 0.2 %, at a fraction of the cost."""
    dR = ctx.R[:, :-1].transpose(-1, -2) @ ctx.R[:, 1:]
    w = vee(dR - dR.transpose(-1, -2)) / (2 * ctx.src.const(np.diff(ctx.ts))[None, :, None])
    al = (w[:, 1:] - w[:, :-1]) / ctx.src.const(0.5 * (ctx.ts[2:] - ctx.ts[:-2]))[None, :, None]
    return torch.cat([al[:, :1], al, al[:, -1:]], 1)


def reach_limit(ctx, a):
    return (ctx.p - ctx.src.const(a["center"])).norm(dim=-1, keepdim=True) - a["r_max"]


PRIMITIVES = {
    "reach_point": Spec("eq", reach_point, "sigma", "m"),
    "reach_rot": Spec("eq", reach_rot, "sigma", "rad"),
    "align_axis": Spec("eq", align_axis, "sigma", "rad"),
    "tilt_limit": Spec("ineq", tilt_limit, "max", "rad", +1),
    "above": Spec("ineq", above, "h", "m", -1),
    "clearance": Spec("ineq", clearance, "margin", "m", -1),
    "plane_clearance": Spec("ineq", plane_clearance, "margin", "m", -1),
    "distance_band": Spec("ineq", distance_band, None, "m"),
    "half_space": Spec("ineq", half_space, "d", "m", -1),
    "approach_cone": Spec("ineq", approach_cone, "half_angle", "rad", +1),
    "value_map": Spec("eq", value_map, "sigma", None),
    "smooth_pos": Spec("eq", smooth_pos, None, "m/t2"),
    "smooth_rot": Spec("eq", smooth_rot, None, "rad/t2"),
    "reach_limit": Spec("ineq", reach_limit, "r_max", "m", +1),
}


def residual(ctx, term, args=None):
    return ctx.full(PRIMITIVES[term.prim].fn(ctx, term.args if args is None else args))


def scope_mask(scope, ts):
    ts = np.asarray(ts, float)
    if scope == "path":
        return np.ones(len(ts), bool)
    if scope == "terminal":
        return ts >= ts.max() - 1e-12
    a, b = scope
    m = (ts >= a - 1e-9) & (ts <= b + 1e-9)
    if not m.any():                                  # an interval between samples still gets its nearest sample
        m[np.argmin(np.abs(ts - 0.5 * (a + b)))] = True
    return m


def violation(r, shape, sigma):
    """Per-sample normalized violation (..., N): |r| / sigma (eq) or the worst component's max(0, r) / sigma."""
    return (r.norm(dim=-1) if shape == "eq" else r.clamp(min=0).amax(-1)) / sigma


def pose_tensors(R, p, approach, dtype, device):
    t = lambda x: torch.as_tensor(np.asarray(x, float), dtype=dtype, device=device)
    return {"R": t(R), "p": t(p), "approach": None if approach is None else t(approach)}
