"""Scene fields TRACE derives from one SceneSnapshot (TRACE_method.md §4 as changed by the integration doc).

The harness owns the snapshot; TRACE only reads it and caches what it derives on the snapshot object, keyed by the
arguments, so one snapshot version is processed once.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree

# open_air: distance (m) reported above the grid's top (open air is free). device: where distance transforms and blurs
# run; on a real map (1.7 M voxels) scipy needs ~0.3 s per transform and ~0.1 s per blur, the GPU a few ms
CFG = {"open_air": 1.0, "device": "cuda:0" if torch.cuda.is_available() else None, "chunk": 1 << 25}


def edt(target):
    """Euclidean distance (voxels, float32) from every voxel to the nearest True voxel of the bool grid `target`,
    like scipy's distance_transform_edt(~target). On the GPU: the exact separable min-plus transform
    g(i) = min_j f(j) + (i - j)^2 per axis, evaluated by brute force over each grid line."""
    if CFG["device"] is None:
        return ndimage.distance_transform_edt(~target).astype(np.float32)
    f = torch.where(torch.as_tensor(target, device=CFG["device"]), 0.0, float("inf"))
    for ax in range(3):
        g = f.movedim(ax, -1)
        n = g.shape[-1]
        i = torch.arange(n, device=f.device, dtype=torch.float32)
        d2 = (i[:, None] - i[None, :]) ** 2
        lines = g.reshape(-1, n)
        step = max(1, CFG["chunk"] // (n * n))
        out = torch.cat([(lines[k:k + step, None, :] + d2).amin(-1) for k in range(0, len(lines), step)])
        f = out.reshape(g.shape).movedim(-1, ax)
    return f.sqrt().cpu().numpy()


def _cache(snap):
    return snap.__dict__.setdefault("_trace_cache", {})


def _cached(snap, key, make):
    c = _cache(snap)
    if key not in c:
        c[key] = make()
    return c[key]


def field_mask(snap, exclude=(), only=None, unknown=False):
    ex = list(snap.label_of(exclude))
    if unknown:
        m = snap.unknown & ~np.isin(snap.shadow, ex)
        return m & (snap.shadow == snap.entities[only].label) if only is not None else m
    m = snap.occ & ~np.isin(snap.labels, ex)
    return m & (snap.labels == snap.entities[only].label) if only is not None else m


def sdf_grid(snap, exclude=(), only=None, unknown=False):
    """Conservative signed distance (m, float32) like CP-5's SceneMap.clearance: one voxel is subtracted outside."""
    def make():
        m = field_mask(snap, exclude, only, unknown)
        if not m.any():                       # nothing of this kind in the scene: free everywhere
            return np.full(m.shape, CFG["open_air"], np.float32)
        return (np.where(m, -edt(~m), edt(m) - 1) * snap.res).astype(np.float32)
    return _cached(snap, ("sdf", tuple(sorted(exclude)), only, bool(unknown)), make)


def blurred(grid, beta, res):
    """Gaussian-blurred copy for the coarse continuation levels (sigma = beta / res voxels)."""
    if beta <= 0:
        return grid
    if CFG["device"] is None:
        return ndimage.gaussian_filter(grid, beta / res, mode="nearest", truncate=3.0)
    sigma = beta / res                        # scipy's kernel (radius int(3 sigma + 0.5)) and "nearest" border
    r = int(3.0 * sigma + 0.5)
    x = torch.arange(-r, r + 1, device=CFG["device"], dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum())[None, None]
    g = torch.as_tensor(grid, device=CFG["device"])
    for ax in range(3):
        h = g.movedim(ax, -1)
        g = F.conv1d(F.pad(h.reshape(-1, 1, h.shape[-1]), (r, r), mode="replicate"), k).reshape(h.shape).movedim(-1, ax)
    return g.cpu().numpy()


def field_grid(snap, spec, beta=0.0):
    spec = spec or {}
    key = (tuple(sorted(spec.get("exclude", ()))), spec.get("only"), bool(spec.get("unknown", False)))
    return _cached(snap, ("blur",) + key + (float(beta),),
                   lambda: blurred(sdf_grid(snap, *key), beta, snap.res))


class GridField:
    """Trilinear, differentiable lookup phi(x (...,3)) -> (...). Voxel centres sit at origin + (ijk + 0.5) res, which
    align_corners=True maps to -1..1; grid_sample reads the last coordinate dim as (W, H, D) = (z, y, x)."""

    def __init__(self, grid, origin, res, device, dtype=torch.float32):
        n = np.array(grid.shape)
        self.g = torch.as_tensor(grid, dtype=dtype, device=device)[None, None]
        self.lo = torch.as_tensor(np.asarray(origin, float) + 0.5 * res, dtype=dtype, device=device)
        self.scale = torch.as_tensor(2.0 / ((n - 1) * res), dtype=dtype, device=device)
        self.top = float(origin[2] + n[2] * res)

    def __call__(self, x):
        shp = x.shape[:-1]
        u = (x.reshape(-1, 3) - self.lo) * self.scale - 1
        val = F.grid_sample(self.g, u.flip(-1).reshape(1, -1, 1, 1, 3), mode="bilinear", padding_mode="border",
                            align_corners=True).reshape(shp)
        return torch.where(x[..., 2] >= self.top, torch.full_like(val, CFG["open_air"]), val)


def grid_field(snap, spec, beta, device, dtype=torch.float32):
    spec = spec or {}
    key = ("gf", tuple(sorted(spec.get("exclude", ()))), spec.get("only"), bool(spec.get("unknown", False)),
           float(beta), str(device), dtype)
    return _cached(snap, key, lambda: GridField(field_grid(snap, spec, beta), snap.origin, snap.res, device, dtype))


def trilinear_np(snap, grid, pts):
    """float64 trilinear value of a snapshot grid at points (N,3), border-clamped, open air above the top."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    idx = (pts - np.asarray(snap.origin, float)) / snap.res - 0.5
    val = ndimage.map_coordinates(grid.astype(np.float64), idx.T, order=1, mode="nearest")
    return np.where(pts[:, 2] >= snap.origin[2] + grid.shape[2] * snap.res, CFG["open_air"], val)


def exact_distance(snap, pts, exclude=()):
    """Distance (m) of points (N,3) to the scene's solid geometry, table plane excluded: the harness's exact near
    field when the snapshot carries one, else the trilinear grid value."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    if snap.exact is not None:
        return np.asarray(snap.exact(pts, tuple(exclude)), float)
    return trilinear_np(snap, sdf_grid(snap, tuple(exclude)), pts)


def field_distance(snap, spec, pts):
    """Certification geometry for a clearance field: exact for the plain occupied field, the unblurred grid for
    entity-only and unknown-space fields (the exact near field cannot select those)."""
    spec = spec or {}
    if spec.get("only") is None and not spec.get("unknown", False):
        return exact_distance(snap, pts, spec.get("exclude", ()))
    return trilinear_np(snap, field_grid(snap, spec), pts)


def coarse_occupancy(snap, inflate, res=0.02):
    """Bool grid (coarse voxel (ijk) centred at origin + (ijk + 0.5) res) of cells whose centre is closer than
    `inflate` to occupied or unknown space or to the table plane: the A* seeding grid."""
    def make():
        obs = snap.occ | snap.unknown
        d = (edt(obs) - 1) * snap.res if obs.any() else np.full(obs.shape, 1.0)
        n = np.maximum(np.round(np.array(snap.shape) * snap.res / res).astype(int), 1)
        c = [np.asarray(snap.origin)[i] + (np.arange(n[i]) + 0.5) * res for i in range(3)]
        C = np.stack(np.meshgrid(*c, indexing="ij"), -1).reshape(-1, 3)
        dist = trilinear_np(snap, d.astype(np.float32), C)
        if snap.table_z is not None:
            dist = np.minimum(dist, C[:, 2] - snap.table_z)
        return (dist < inflate).reshape(n)
    return _cached(snap, ("coarse", round(float(inflate), 6), float(res)), make)


def changed_inside(snap_old, snap_new, pts, radius):
    """True if occupancy or unknown state differs in any voxel within `radius` of the points (the 'scene changed
    inside the tube' rule)."""
    if snap_old.shape != snap_new.shape or not np.allclose(snap_old.origin, snap_new.origin) \
            or snap_old.res != snap_new.res:
        return True
    diff = (snap_old.occ != snap_new.occ) | (snap_old.unknown != snap_new.unknown)
    if not diff.any():
        return False
    centres = np.asarray(snap_new.origin, float) + (np.argwhere(diff) + 0.5) * snap_new.res
    d, _ = cKDTree(np.asarray(pts, float).reshape(-1, 3)).query(centres, distance_upper_bound=radius
                                                                  + 0.87 * snap_new.res)
    return bool(np.isfinite(d).any())
