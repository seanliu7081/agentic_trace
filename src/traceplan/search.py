"""Search (TRACE_method.md §6.1, §6.2, §6.4): terminal candidates, diverse path seeds, coupled continuation.

Every option in CFG exists because the method note or the bench asks for it: the explore/exploit pair (stage 1),
seeding / continuation / hard-term handling (stage 2) and the K schedule (stage 3).
"""
from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

from .contract import GROUPS, Term
from .primitives import Body, EvalContext, Source, pose_tensors, residual, violation
from .scene import coarse_occupancy, exact_distance
from .se3 import so3_exp, so3_exp_np, so3_log_np
from .solvers import exploit, explore
from .spline import K_SCHEDULE

CFG = {"G": 2, "H": 4, "explore": "mppi", "exploit": "lbfgs", "n_samples": 256, "explore_iters": (10, 10, 10, 10),
       "exploit_iters": (5, 5, 10, 20), "alpha_pos": 0.05, "alpha_rot": 0.2, "tau": 0.1, "elite_frac": 0.1,
       "lr": 0.01, "seeding": "astar", "continuation": "all", "hard": "al", "k_schedule": K_SCHEDULE,
       "n_eval": 32, "astar_res": 0.02, "astar_margin": 0.25, "astar_block_cost": 30.0, "astar_div_radius": 0.04,
       "astar_div_cost": 5.0, "astar_weight": 1.5, "astar_max_expand": 15000, "random_pos": 0.05,
       "random_rot": 0.2, "free_end_samples": 256, "free_end_iters": 60, "free_end_lr": 0.005,
       "free_end_distinct": (0.02, 0.3)}


@dataclass
class Cand:
    """One terminal candidate: the end pose after the standoff and its own compiled program."""
    index: int
    R: np.ndarray
    p: np.ndarray
    q: np.ndarray | None
    approach: np.ndarray | None
    standoff: float
    entity: str | None
    window: float | None
    src: object
    program: object = None
    flags: dict = field(default_factory=dict)
    flag: dict | None = None
    closed: dict | None = None      # window flag at the perception floor: the axis is closed by contact
    score: float = 0.0

    def end(self):
        return {"R": self.R, "p": self.p, "approach": self.approach}


def _pose_ctx(R, p, v, body, src, start, end, holding, allow, dtype, device):
    out = {"R": R[:, None], "p": p[:, None], "v": v[:, None], "acc": torch.zeros_like(p)[:, None]}
    return EvalContext(out, np.ones(1), body, src, pose_tensors(start["R"], start["p"], None, dtype, device),
                       pose_tensors(end["R"], end["p"], end.get("approach"), dtype, device), holding, allow)


# --------------------------------------------------------------------------------------------- terminal candidates
def end_test(snap, bodies, R, p, held=(), entity=None, contact=False):
    """The end test, one rule for TRACE and the harness: clearance (m) of every body sphere at the pose (R, p) to the
    observed scene and the table plane. A contact end (approach given) exempts the fingers from the step's named
    target. Unseen space is not part of the test; its deepest overlap is reported so the caller can look first.
    -> {"ok", "worst": {group, clearance, at}, "unseen": {group, depth, at} | None}"""
    held = tuple(held)
    x = np.asarray(R, float) @ bodies.centers.T + np.asarray(p, float)[:, None]
    x = x.T
    r = bodies.radii
    clear = exact_distance(snap, x, held) - r
    finger = bodies.groups == GROUPS.index("finger")
    excl = held
    if contact and entity is not None and finger.any():
        excl = held + (entity,)
        clear[finger] = exact_distance(snap, x[finger], excl) - r[finger]
    if snap.table_z is not None:
        clear = np.minimum(clear, x[:, 2] - snap.table_z - r)
    i = int(np.argmin(clear))
    out = {"ok": bool(clear[i] >= 0.0),
           "worst": {"group": GROUPS[int(bodies.groups[i])], "clearance": float(clear[i]), "at": x[i].copy()},
           "unseen": None}
    cu = exact_distance(snap, x, held, unknown=True) - r
    if contact and entity is not None and finger.any():
        cu[finger] = exact_distance(snap, x[finger], excl, unknown=True) - r[finger]
    j = int(np.argmin(cu))
    if cu[j] < 0.0 and cu[j] < clear[j] - 1e-6:
        out["unseen"] = {"group": GROUPS[int(bodies.groups[j])], "depth": float(-cu[j]), "at": x[j].copy()}
    return out


def score_candidates(cands, snap, bodies, start, holding, ik_check, G, dropped=None):
    """Drop ends that penetrate the observed scene (end_test) or fail IK; order the rest by their terminal-term
    violation (stable, so the harness's order decides ties) and keep G. dropped: list that receives why each end
    left ({end, why, group, clearance, at})."""
    f64 = torch.float64
    body = Body(bodies, "cpu", f64)
    held = (holding,) if holding else ()
    keep = []
    for c in cands:
        test = getattr(c, "test", None)
        if test is None:
            test = end_test(snap, bodies, c.R, c.p, held, c.entity, contact=c.approach is not None)
        if not test["ok"]:
            if dropped is not None:
                dropped.append({"end": c.index, "why": "no_posture" if test.get("no_posture") else "blocked",
                                **test["worst"]})
            continue
        if ik_check is not None and not getattr(c, "ik_done", False):
            ok, q = ik_check(c.R, c.p, c.q)
            if not ok:
                if dropped is not None:
                    dropped.append({"end": c.index, "why": "no_posture", "at": np.asarray(c.p, float).copy()})
                continue
            c.q = q if c.q is None else c.q
        c.unseen = test["unseen"]
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=f64)[None]
        ctx = _pose_ctx(t(c.R), t(c.p), t(c.p - start["p"]), body, Source(snap, "exact", dtype=f64), start, c.end(),
                        bool(holding), c.program.allow_contact, f64, "cpu")
        c.score = sum(max(0.0, float(violation(residual(ctx, tm)[0], tm.shape, tm.args.get("sigma_accept", tm.sigma))
                                     [0]) - 1.0)
                      for tm in c.program.terms if tm.role != "safety" and tm.scope == "terminal")
        keep.append(c)
    return sorted(keep, key=lambda c: c.score)[:G]


def free_end_candidates(fe, program, snap, bodies, start, holding, ik_check, gen, cfg, device):
    """§6.1 end-pose solve: a few hundred poses at/above the entity with random yaw about the approach axis,
    refined with Adam on a 6-D pose offset under the terminal terms, clearance (no end zone) and the relation."""
    dt = torch.float32
    ent = snap.entities[fe["entity"]]
    target = ent.keypoint("top_center" if fe["relation"] == "above" else "centroid") + np.r_[0, 0, fe.get("height", 0.0)]
    a = np.asarray(fe.get("approach", [0.0, 0, -1]), float)
    a = a / np.linalg.norm(a)
    z = np.r_[0.0, 0, 1]
    ax = np.cross(z, a)
    Ra = so3_exp_np(ax / max(np.linalg.norm(ax), 1e-12) * np.arctan2(np.linalg.norm(ax), z @ a))   # z -> a
    n = cfg["free_end_samples"]
    yaw = torch.rand(n, generator=gen, dtype=torch.float64).numpy() * 2 * np.pi
    jit = torch.randn(n, 3, generator=gen, dtype=torch.float64).numpy() * 0.02
    Rs = Ra @ so3_exp_np(yaw[:, None] * z)
    t = lambda x: torch.as_tensor(np.asarray(x, float), dtype=dt, device=device)
    R_i, p_i = t(Rs), t(target + jit)
    body = Body(bodies, device, dt)
    src = Source(snap, "grid", 0.0, device, dt)
    terms = [tm for tm in program.terms if tm.prim in ("clearance", "plane_clearance") and tm.role == "safety"
             or tm.scope == "terminal" and tm.prim != "approach_cone" and tm.args.get("b") != "end"]
    terms = [Term(tm.id, tm.prim, {k: v for k, v in tm.args.items() if k not in ("c_start", "c_end")}, tm.shape,
                  tm.role, "terminal", tm.sigma) for tm in terms]
    terms += [Term("fe_target", "reach_point", {"a": "eef", "b": list(target)}, "eq", "prefer", "terminal", 0.01),
              Term("fe_axis", "align_axis", {"u": "eef.z", "v": list(a)}, "eq", "prefer", "terminal", 0.1)]
    end = {"R": Rs[0], "p": target, "approach": a}
    x = torch.zeros(n, 6, dtype=dt, device=device, requires_grad=True)
    opt = torch.optim.Adam([x], lr=cfg["free_end_lr"])

    def cost(x):
        R, p = R_i @ so3_exp(x[:, 3:]), p_i + x[:, :3]
        ctx = _pose_ctx(R, p, torch.zeros_like(p), body, src, start, end, bool(holding), program.allow_contact, dt,
                        device)
        J = 0.0
        for tm in terms:
            r = residual(ctx, tm)[:, 0] / tm.sigma
            J = J + (r * r if tm.shape == "eq" else r.clamp(min=0) ** 2).sum(-1)
        return J, R, p

    for _ in range(cfg["free_end_iters"]):
        opt.zero_grad()
        cost(x)[0].sum().backward()
        opt.step()
    with torch.no_grad():
        J, R, p = cost(x)
    R, p = R.cpu().double().numpy(), p.cpu().double().numpy()
    out, dpos, drot = [], *cfg["free_end_distinct"]
    for i in np.argsort(J.cpu().numpy()):
        if any(np.linalg.norm(p[i] - c[1]) < dpos and np.linalg.norm(so3_log_np(R[i].T @ c[0])) < drot for c in out):
            continue
        q = None
        if ik_check is not None:
            ok, q = ik_check(R[i], p[i], None)
            if not ok:
                continue
        out.append((R[i], p[i], q))
        if len(out) == cfg["G"]:
            break
    return out


# --------------------------------------------------------------------------------------------- seeds
def _astar(cost, s, g, weight, max_expand):
    """Weighted A* on a cost grid (26-neighbourhood, edge cost = length x cost of the entered cell). The grid is
    padded with an infinite-cost border so neighbour indices never leave it; costs, heuristic and bookkeeping are flat
    lists, the fastest containers for this pure-Python loop."""
    shp = np.array(cost.shape) + 2
    C = np.full(shp, np.inf)
    C[1:-1, 1:-1, 1:-1] = cost
    Cf = C.ravel().tolist()
    s0, s1 = int(shp[1] * shp[2]), int(shp[2])
    offs = [(d[0] * s0 + d[1] * s1 + d[2], math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2))
            for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]
    flat = lambda c: int((c[0] + 1) * s0 + (c[1] + 1) * s1 + c[2] + 1)
    si, gi = flat(s), flat(g)
    Hf = (weight * np.linalg.norm(np.indices(shp).reshape(3, -1).T - (np.asarray(g) + 1), axis=1)).tolist()
    inf = math.inf
    dist, par, closed = [inf] * len(Cf), [-1] * len(Cf), bytearray(len(Cf))
    dist[si], heap, n = 0.0, [(Hf[si], si)], 0
    while heap:
        _, i = heapq.heappop(heap)
        if i == gi:
            break
        if closed[i]:
            continue
        closed[i] = 1
        n += 1
        if n > max_expand:
            return None
        di = dist[i]
        for off, L in offs:
            j = i + off
            nd = di + L * Cf[j]
            if nd < dist[j]:
                dist[j], par[j] = nd, i
                heapq.heappush(heap, (nd + Hf[j], j))
    else:
        return None
    path = [gi]
    while par[path[-1]] != -1:
        path.append(par[path[-1]])
    idx = np.array(path[::-1])
    return np.stack([idx // s0, (idx % s0) // s1, idx % s1], 1) - 1


def astar_paths(snap, p0, pg, inflate, n, cfg, deadline=math.inf):
    """Up to n diverse corridors: A* on the coarse occupancy (soft: blocked cells cost more, so an end inside the
    inflated obstacles still connects), then the cells within div_radius of each path are penalized. No new path is
    started after the deadline."""
    res = cfg["astar_res"]
    occ = coarse_occupancy(snap, inflate, res)
    shape, org = np.array(occ.shape), np.asarray(snap.origin, float)
    cell = lambda x: np.clip(np.floor((np.asarray(x) - org) / res).astype(int), 0, shape - 1)
    s, g = cell(p0), cell(pg)
    m = int(np.ceil(cfg["astar_margin"] / res))
    lo, hi = np.maximum(np.minimum(s, g) - m, 0), np.minimum(np.maximum(s, g) + m + 1, shape)
    lo[2], hi[2] = 0, shape[2]
    sub = occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    C = org + (np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1) + 0.5) * res
    cost = np.where(sub, cfg["astar_block_cost"], 1.0)
    # the hand may legitimately start or end next to objects: those surroundings are not blocked for seeding
    cost[(np.linalg.norm(C - p0, axis=-1) < inflate + res) | (np.linalg.norm(C - pg, axis=-1) < inflate + res)] = 1.0
    paths = []
    for _ in range(n):
        if time.perf_counter() >= deadline:
            break
        cells = _astar(cost, s - lo, g - lo, cfg["astar_weight"], cfg["astar_max_expand"])
        if cells is None:
            break
        pts = np.vstack([p0, org + (cells[1:-1] + lo + 0.5) * res, pg])
        paths.append(pts)
        d, _ = cKDTree(pts).query(C.reshape(-1, 3))
        cost = cost + cfg["astar_div_cost"] * (d.reshape(cost.shape) < cfg["astar_div_radius"])
    return paths


def path_seeds(spline, cand, snap, bodies, gen, cfg, deadline=math.inf):
    """Straight seed plus H-1 A* (or random) seeds, all at the spline's K (= 4 at level 0)."""
    th = [spline.straight_theta()]
    if cfg["seeding"] == "astar":
        inflate = float(bodies.radii[bodies.groups != GROUPS.index("payload")].max()) + 0.01
        th += [spline.theta_from_path(P) for P in astar_paths(snap, spline.p0, spline.pg, inflate, cfg["H"] - 1, cfg,
                                                                deadline)]
    elif cfg["seeding"] == "random":
        n = spline.n_ctrl * 3
        for _ in range(cfg["H"] - 1):
            z = torch.randn(2 * n, generator=gen, dtype=torch.float64)
            z = torch.cat([z[:n] * cfg["random_pos"], z[n:] * cfg["random_rot"]]).to(th[0])
            th.append(th[0] + z)
    return torch.stack(th)


# --------------------------------------------------------------------------------------------- continuation
def continuation(obj, spline, theta, gen, cfg, deadline, provided=None, warm=False):
    """Coupled continuation (§6.2) on one candidate: per level explore -> exploit -> multipliers -> best half ->
    exact knot refinement. `provided` (n, D at K = 11) joins at the level whose K matches; `warm` reruns only the
    last level (closed-loop reuse, leave-one-out). The clock is checked before each explore and exploit; past the
    deadline it stops and returns what exists, sorted by the J of the level reached."""
    nl = len(obj.levels)
    levels = [nl - 1] if warm else ([nl - 1] * nl if cfg["continuation"] == "off" else list(range(nl)))
    obj.reset(len(theta))
    s0 = obj.levels[0]["s"]
    info = {"levels_run": 0}
    for li, lev in enumerate(levels):
        L = obj.levels[lev]
        if spline.K != L["K"]:
            spline, theta = spline.refine(theta, L["K"])
        obj.set_level(lev, spline)
        if provided is not None and len(provided) and spline.K == provided.shape[1] // 6 + 1 + int(spline.pin_end):
            theta = torch.cat([theta, provided.to(theta)])
            obj.add_seeds(len(provided))
            provided = None
        if time.perf_counter() >= deadline:
            break
        sc = L["s"] / s0
        ecfg = {"layout": spline.layout, "n_samples": cfg["n_samples"], "alpha_pos": cfg["alpha_pos"] * sc,
                "alpha_rot": cfg["alpha_rot"] * sc, "tau": cfg["tau"], "elite_frac": cfg["elite_frac"]}
        theta = explore(cfg["explore"], obj, theta, cfg["n_samples"] * cfg["explore_iters"][lev], gen, ecfg)
        if time.perf_counter() < deadline:
            theta = exploit(cfg["exploit"], obj, theta, cfg["exploit_iters"][lev], {"lr": cfg["lr"]})
        obj.update_multipliers(theta)
        info["levels_run"] += 1
        if time.perf_counter() >= deadline:
            break
        if li < len(levels) - 1 and len(theta) > 1:
            with torch.no_grad():
                keep = torch.argsort(obj(theta))[:max(1, len(theta) // 2)]
            theta = theta[keep]
            obj.select(keep)
    with torch.no_grad():
        J = obj(theta)
    order = torch.argsort(J)
    obj.select(order)
    return spline, theta[order].detach(), J[order], info
