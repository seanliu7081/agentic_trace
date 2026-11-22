"""Evaluator-only true scene (DESIGN §4). Never a policy input.

true_snapshot(env, res) samples the surfaces of the MuJoCo collision geoms of every non-robot body at the current
state at 3 mm, voxelizes them into a SceneSnapshot and sets `exact` to the KD-tree distance to the samples. One
entity per top-level body; its `points` keep every surface sample (float32), so a saved snapshot rebuilds the same
`exact` with exact_from(). Box-shaped tables stay geometry; the highest upward-facing plane (the floor arenas' support)
becomes table_z, the analytic plane of the contract. Distances are unsigned (to the nearest surface sample).
"""
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from traceplan.contract import Entity, SceneSnapshot

CFG = {"step": 0.003, "robot": ("robot0_", "gripper0_", "mount0_"), "half_xy": 1.0, "z": (-0.2, 1.2)}
G = mujoco.mjtGeom


def _grid(a, b, step):
    U, V = np.meshgrid(np.linspace(-a, a, int(np.ceil(2 * a / step)) + 1),
                       np.linspace(-b, b, int(np.ceil(2 * b / step)) + 1), indexing="ij")
    return U.ravel(), V.ravel()


def _box(s, step):
    out = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        U, V = _grid(s[o[0]], s[o[1]], step)
        for sign in (-1.0, 1.0):
            P = np.empty((len(U), 3))
            P[:, o[0]], P[:, o[1]], P[:, ax] = U, V, sign * s[ax]
            out.append(P)
    return np.concatenate(out)


def _sphere(r, step):
    n = max(8, int(np.ceil(4 * np.pi * r * r / step ** 2)))
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _cylinder(r, h, step, caps=True):
    T, Z = np.meshgrid(np.linspace(0, 2 * np.pi, max(8, int(np.ceil(2 * np.pi * r / step))), endpoint=False),
                       np.linspace(-h, h, int(np.ceil(2 * h / step)) + 1), indexing="ij")
    side = np.c_[r * np.cos(T.ravel()), r * np.sin(T.ravel()), Z.ravel()]
    if not caps:
        return side
    U, V = _grid(r, r, step)
    m = U * U + V * V <= r * r
    return np.concatenate([side] + [np.c_[U[m], V[m], np.full(m.sum(), z)] for z in (-h, h)])


def _capsule(r, h, step):
    S = _sphere(r, step)
    S[:, 2] += np.where(S[:, 2] >= 0, h, -h)
    return np.concatenate([_cylinder(r, h, step, caps=False), S])


def _mesh(m, g, step, rng):
    i = m.geom_dataid[g]
    V = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]].astype(float)
    F = m.mesh_face[m.mesh_faceadr[i]:m.mesh_faceadr[i] + m.mesh_facenum[i]]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    n = int(np.ceil(area.sum() / step ** 2))
    k = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return np.concatenate([V, A[k] + u[:, None] * (B[k] - A[k]) + v[:, None] * (C[k] - A[k])])


def _local(m, g, step, rng):
    t, s = m.geom_type[g], m.geom_size[g]
    if t == G.mjGEOM_BOX:
        return _box(s, step)
    if t == G.mjGEOM_SPHERE:
        return _sphere(s[0], step)
    if t == G.mjGEOM_CAPSULE:
        return _capsule(s[0], s[1], step)
    if t == G.mjGEOM_CYLINDER:
        return _cylinder(s[0], s[1], step)
    if t == G.mjGEOM_ELLIPSOID:
        return _sphere(1.0, step / s.max()) * s
    if t == G.mjGEOM_MESH:
        return _mesh(m, g, step, rng)
    return None                                          # planes, height fields: outside the workspace box


def exact_from(snap):
    """exact(points, exclude) of a (saved) true snapshot: KD-tree distance to the entities not excluded."""
    trees = {k: cKDTree(e.points) for k, e in snap.entities.items() if len(e.points)}

    def exact(pts, exclude=()):
        d = [t.query(np.asarray(pts, float))[0] for k, t in trees.items() if k not in exclude]
        return np.min(d, 0) if d else np.full(len(pts), np.inf)
    return exact


def true_snapshot(env, res=0.01, cfg=CFG):
    m, d = env.sim.model._model, env.sim.data._data
    name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
    base = next(b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and name(b).startswith(cfg["robot"]))
    lo = d.xpos[base] + np.r_[-cfg["half_xy"], -cfg["half_xy"], cfg["z"][0]]
    hi = d.xpos[base] + np.r_[cfg["half_xy"], cfg["half_xy"], cfg["z"][1]]
    rng = np.random.default_rng(0)
    pts, planes = {}, []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        root = name(m.body_rootid[b])
        if not (m.geom_contype[g] or m.geom_conaffinity[g]) or name(b).startswith(cfg["robot"]) \
                or root.startswith(cfg["robot"]):
            continue
        c, r = d.geom_xpos[g], m.geom_rbound[g]
        if m.geom_type[g] == G.mjGEOM_PLANE and d.geom_xmat[g][8] > 0.99 and lo[2] <= c[2] < hi[2]:
            planes.append(float(c[2]))                   # infinite: no bounding-sphere test
        if np.any(c + r < lo) or np.any(c - r > hi):
            continue
        L = _local(m, g, cfg["step"], rng)
        if L is None:
            continue
        W = L @ d.geom_xmat[g].reshape(3, 3).T + c
        pts.setdefault(root, []).append(W[np.all((W >= lo) & (W < hi), 1)])
    shape = tuple(np.ceil((hi - lo) / res).astype(int))
    occ, labels = np.zeros(shape, bool), np.zeros(shape, np.int16)
    entities = {}
    for lab, (k, P) in enumerate(sorted(pts.items()), 1):
        P = np.concatenate(P)
        if not len(P):
            continue
        ijk = np.floor((P - lo) / res).astype(int).T
        occ[tuple(ijk)], labels[tuple(ijk)] = True, lab
        entities[k] = Entity(k, lab, P.astype(np.float32), np.zeros((3, 3)))
    snap = SceneSnapshot(0, lo, res, occ, np.zeros(shape, bool), labels, np.zeros(shape, np.int16),
                         max(planes) if planes else None, entities,
                         None, {"true": True, "step": cfg["step"], "sim_time": float(d.time)})
    snap.exact = exact_from(snap)
    return snap
