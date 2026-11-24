"""S2 perception: multi-view RGB-D fusion into a voxel scene map + object models.

* SceneMap: 1 cm voxels with UNKNOWN / FREE / OCC state, object labels, shadow
  labels (unknown space hidden under an object's visible surface belongs to it),
  ESDF queries with per-object exclusions.
* ObjectModel: fused surface points + a 2.5D solid (per-cell top height above a
  support) for inside tests, extents, principal axes.
* Region growing on depth as a segmentation fallback.
* NearField: exact near-contact distances (observed surface points, unknown voxel cubes) for the
  places where the 1 cm voxel ESDF is too coarse (grasping, placing, operating parts).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

UNKNOWN, FREE, OCC = -1, 0, 1


def voxel_down(pts, s):
    if len(pts) == 0:
        return pts
    k = np.floor(pts / s).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return pts[idx]


def hull_surface_samples(pts, spacing=0.006):
    """Points on the surface of the convex hull of `pts` (N,3), about `spacing` apart, plus the hull
    vertices. Returns an empty array when the points are degenerate (flat / collinear)."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(pts, float)
    if len(pts) < 8:
        return np.zeros((0, 3))
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.zeros((0, 3))
    out = [pts[hull.vertices]]
    for tri in hull.simplices:
        a, b, c = pts[tri]
        n = int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing))
        if n < 1:
            continue
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        u, v = i[keep] / n, j[keep] / n
        out.append(a + u[:, None] * (b - a) + v[:, None] * (c - a))
    return voxel_down(np.concatenate(out), 0.5 * spacing)


def estimate_table_z(frame, base_xy, reach=1.2):
    pts, _ = frame.backproject(mask=~frame.robot_mask, stride=2)
    dx = pts[:, 0] - base_xy[0]
    sel = (dx > 0.05) & (dx < reach) & (np.abs(pts[:, 1] - base_xy[1]) < 0.8)
    z = pts[sel, 2]
    bins = np.arange(z.min(), z.max() + 0.004, 0.004)
    h, e = np.histogram(z, bins=bins)
    k = int(np.argmax(h))
    near = z[(z > e[k] - 0.004) & (z < e[k + 1] + 0.004)]
    return float(np.median(near))


@dataclass
class ObjectModel:
    oid: int
    name: str
    category: str
    role: str
    support_z: float
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    part_of: int | None = None
    cell: float = 0.005
    pixel_hint: tuple | None = None
    views: list = field(default_factory=list)
    held: bool = False
    prompts: list = field(default_factory=list)      # all of S1's segmenter prompts for it (category = the first)

    def add_points(self, pts, max_points=40000):
        pts = np.asarray(pts, float)
        if len(pts) == 0:
            return
        pts = pts[pts[:, 2] > self.support_z + 0.003]
        self.points = voxel_down(np.concatenate([self.points, pts], 0), 0.003)
        if len(self.points) > max_points:
            self.points = self.points[np.random.default_rng(0).choice(len(self.points), max_points, replace=False)]
        self._finalize()

    def set_points(self, pts):
        self.points = voxel_down(np.asarray(pts, float), 0.003)
        self._finalize()

    def _finalize(self):
        p = self.points
        if len(p) == 0:
            self.hm = None
            return
        self.tree = cKDTree(p)
        self.lo_xy = p[:, :2].min(0) - 2 * self.cell
        n = np.ceil((p[:, :2].max(0) + 2 * self.cell - self.lo_xy) / self.cell).astype(int) + 1
        hm = np.full(n, -np.inf)
        ij = np.floor((p[:, :2] - self.lo_xy) / self.cell).astype(int)
        np.maximum.at(hm, (ij[:, 0], ij[:, 1]), p[:, 2])
        # close small holes (unseen cells inside the footprint)
        filled = ndimage.grey_closing(np.where(np.isfinite(hm), hm, -10.0), size=(3, 3))
        filled[filled < -5] = -np.inf
        self.hm = filled
        self.bottom_z = self.support_z if not self.held else float(p[:, 2].min())

    # ------------------------------------------------------------ geometry
    @property
    def centroid(self):
        return self.points.mean(0)

    @property
    def top_z(self):
        return float(np.percentile(self.points[:, 2], 99.5))

    @property
    def bbox(self):
        return self.points.min(0), self.points.max(0)

    def footprint_center(self):
        """Centre of the top-view footprint (mid-range of xy extents of the solid)."""
        if self.hm is None:
            return self.centroid[:2]
        ij = np.argwhere(np.isfinite(self.hm))
        xy = self.lo_xy + (ij + 0.5) * self.cell
        return 0.5 * (xy.min(0) + xy.max(0))

    def footprint_xy(self):
        ij = np.argwhere(np.isfinite(self.hm))
        return self.lo_xy + (ij + 0.5) * self.cell

    def principal_axes_xy(self, pts=None):
        p = (self.points if pts is None else pts)[:, :2]
        c = p.mean(0)
        w, v = np.linalg.eigh(np.cov((p - c).T) + 1e-9 * np.eye(2))
        return c, v[:, 1], v[:, 0], np.sqrt(np.maximum(w[::-1], 0))

    def inside(self, pts, tol=0.0):
        """Points (N,3) inside the 2.5D solid (cell occupied and bottom <= z <= top)."""
        pts = np.asarray(pts, float)
        if self.hm is None or len(pts) == 0:
            return np.zeros(len(pts), bool)
        ij = np.floor((pts[:, :2] - self.lo_xy) / self.cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < self.hm.shape[0]) & (ij[:, 1] < self.hm.shape[1])
        res = np.zeros(len(pts), bool)
        top = np.full(len(pts), -np.inf)
        top[ok] = self.hm[ij[ok, 0], ij[ok, 1]]
        res = ok & np.isfinite(top) & (pts[:, 2] <= top + tol) & (pts[:, 2] >= self.bottom_z - tol)
        return res

    def surface_distance(self, pts):
        if len(self.points) == 0:
            return np.full(len(pts), np.inf)
        d, _ = self.tree.query(np.asarray(pts, float), k=1)
        return d

    def signed_clearance(self, pts):
        """Distance to the object's surface, negative if the point is inside its solid."""
        d = self.surface_distance(pts)
        ins = self.inside(pts, tol=-0.002)
        return np.where(ins, -d, d)

    def summary(self):
        lo, hi = self.bbox
        return {"id": self.oid, "name": self.name, "category": self.category, "role": self.role,
                "centroid": np.round(self.centroid, 3).tolist(), "size": np.round(hi - lo, 3).tolist(),
                "top_z": round(self.top_z, 3), "n_points": int(len(self.points)), "held": self.held}


class SceneMap:
    def __init__(self, lo, hi, res=0.01, table_z=None):
        self.lo = np.asarray(lo, float)
        self.res = float(res)
        self.shape = tuple(int(x) for x in np.ceil((np.asarray(hi, float) - self.lo) / res))
        self.hi = self.lo + np.array(self.shape) * res
        self.state = np.full(self.shape, UNKNOWN, np.int8)
        self.label = np.zeros(self.shape, np.int16)
        self.shadow = np.zeros(self.shape, np.int16)
        self.skin = np.zeros(self.shape, bool)        # voxels straddling / just in front of an observed surface
        self.seen = np.zeros(self.shape, bool)        # voxels inside at least one camera frustum
        self.out_of_view_free_z = None                # never-in-view space above this height is assumed free
        self.cloud = np.zeros((0, 3))                 # observed surface points (3 mm), free-space carved
        self._chunks = []
        self.cloud_version = 0
        self.table_z = table_z
        g = [self.lo[i] + (np.arange(self.shape[i]) + 0.5) * res for i in range(3)]
        self.centers = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
        self.version = 0
        self._cache = {}
        self.free_above = None if table_z is None else table_z + 0.30
        self.robot_zone = None          # (cx, cy, radius, z_lo, z_hi): robot-owned volume around the base

    # ------------------------------------------------------------ indexing
    def index(self, pts):
        return np.floor((np.asarray(pts, float) - self.lo) / self.res).astype(int)

    def inside(self, idx):
        return np.all((idx >= 0) & (idx < np.array(self.shape)), axis=-1)

    def layer(self, z):
        """Index of the voxel layer whose lower face is at height z, for heights that sit on a layer boundary by
        construction (table top, free-above rules). Snapped: (z - lo) / res is computed in floating point and an
        exact 6.0 can come out as 5.999..., which a bare floor turns into the layer below."""
        return int(np.floor((z - self.lo[2]) / self.res + 1e-6))

    def _touch(self):
        self.version += 1
        self._cache.clear()

    # ------------------------------------------------------------ updates
    def integrate(self, frame, labels=None, extra_mask=None, robot_shadow_free=None, held_box=None):
        """Fuse one RGB-D frame. labels: (H,W) int16 object ids (0 = unlabeled).

        robot_shadow_free: optional (z_min, x_max, z_min_near_base): unknown voxels hidden only
        behind the robot's own body (or the object it holds: extra_mask) in this view are assumed free above z_min
        (or above z_min_near_base when x < x_max). Later observations override this.
        held_box: optional (R, p, lo, hi): depth points inside this box (the held object's volume in the hand frame
        at capture) are the held object itself and never enter the map.
        """
        H, W = frame.depth.shape
        st = self.state.reshape(-1)
        lb = self.label.reshape(-1)
        uv, z = frame.project(self.centers)
        u = np.floor(uv[:, 0])
        v = np.floor(uv[:, 1])
        ok = (z > 0.03) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        self.seen.reshape(-1)[idx] = True
        ui = u[idx].astype(int)
        vi = v[idx].astype(int)
        d = frame.depth[vi, ui]
        zz = z[idx]
        cur = st[idx]
        carve = ((cur != OCC) & (zz < d - 1.5 * self.res)) | ((cur == OCC) & (zz < d - 3.0 * self.res))
        sel = idx[carve]
        st[sel] = FREE
        lb[sel] = 0
        # skin: voxels the carving test leaves unknown only because an observed surface passes through or
        # right behind them; their obstacle content is that surface, which the point cloud represents exactly
        band = (zz >= d - 1.5 * self.res) & (zz < d + 0.5 * self.res)
        self.skin.reshape(-1)[idx[band]] = True
        if robot_shadow_free is not None:
            z_min, x_max, z_min_base = robot_shadow_free
            rmask = ndimage.binary_dilation(frame.robot_mask if extra_mask is None else frame.robot_mask | extra_mask,
                                            iterations=3)
            behind = rmask[vi, ui] & (zz >= d - 1.5 * self.res) & (st[idx] == UNKNOWN)
            c = self.centers[idx]
            ok_z = (c[:, 2] > z_min) | ((c[:, 0] < x_max) & (c[:, 2] > z_min_base))
            st[idx[behind & ok_z]] = FREE
        mask = frame.robot_mask
        if extra_mask is not None:
            mask = mask | extra_mask
        mask = ndimage.binary_dilation(mask, iterations=2)
        pts, rc = frame.backproject(mask=~mask)
        if held_box is not None and len(pts):
            Rb, pb, blo, bhi = held_box
            L = (pts - pb) @ Rb
            keep = ~np.all((L >= blo) & (L <= bhi), axis=1)
            pts, rc = pts[keep], rc[keep]
        self._update_cloud(frame, mask, pts)
        vix = self.index(pts)
        ins = self.inside(vix)
        vix, rc = vix[ins], rc[ins]
        lin = np.ravel_multi_index(vix.T, self.shape)
        st[lin] = OCC
        if labels is not None:
            lab = labels[rc[:, 0], rc[:, 1]]
            has = lab != 0
            lb[lin[has]] = lab[has]
        self._touch()

    def _update_cloud(self, frame, mask, pts):
        """Remove earlier cloud points this view sees through (they are gone), then add this view's points."""
        H, W = frame.depth.shape
        dmin = ndimage.minimum_filter(np.where(np.isfinite(frame.depth), frame.depth, 0.0), size=3)
        chunks = []
        for c in self._chunks:
            if len(c) == 0:
                continue
            uv, z = frame.project(c)
            u = np.floor(uv[:, 0])
            v = np.floor(uv[:, 1])
            inb = (z > 0.03) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            gone = np.zeros(len(c), bool)
            ii = np.nonzero(inb)[0]
            ui, vi = u[ii].astype(int), v[ii].astype(int)
            gone[ii] = (~mask[vi, ui]) & (z[ii] < dmin[vi, ui] - 0.006)
            chunks.append(c[~gone])
        tz = getattr(self, "visual_table_z", None)
        tz = self.table_z if tz is None else tz
        keep = np.all((pts >= self.lo) & (pts < self.hi), axis=1)
        if tz is not None:
            keep &= pts[:, 2] > tz + 0.003
        chunks.append(voxel_down(pts[keep], 0.003))
        self._chunks = chunks
        self.cloud_version += 1

    def merge_cloud(self):
        if len(self._chunks) > 1 or (self._chunks and len(self.cloud) != len(self._chunks[0])):
            allp = np.concatenate(self._chunks, 0) if self._chunks else np.zeros((0, 3))
            self._chunks = [voxel_down(allp, 0.003)]
        self.cloud = self._chunks[0] if self._chunks else np.zeros((0, 3))

    def carve_points(self, pts, radius):
        """Mark voxels within radius of points as FREE (robot body occupies them now)."""
        if len(pts) == 0:
            return
        r = int(np.ceil(radius / self.res))
        idx = self.index(pts)
        idx = idx[self.inside(idx)]
        if len(idx) == 0:
            return
        m = np.zeros(self.shape, bool)
        m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        if r > 0:
            # full 26-neighbourhood: a diamond-shaped (6-connected) dilation leaves diagonal neighbours unknown
            m = ndimage.binary_dilation(m, structure=np.ones((3, 3, 3), bool), iterations=max(1, r - 1))
        clear = m & (self.state == UNKNOWN)
        if clear.any():                  # a new version only when the geometry changed (TRACE certificates key on it)
            self.state[clear] = FREE
            self._touch()

    def carve_hulls(self, hulls):
        """Mark non-occupied voxels inside posed convex hulls as FREE.
        hulls: list of (delaunay_in_body_frame, R (3,3), p (3,), corners_world (K,3))."""
        changed = False
        for tri, R, p, corners in hulls:
            lo = np.clip(self.index(corners.min(0)), 0, np.array(self.shape) - 1)
            hi = np.clip(self.index(corners.max(0)), 0, np.array(self.shape) - 1)
            if np.any(hi < lo):
                continue
            sub = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + 1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
            ctr = self.lo + (sub + 0.5) * self.res
            ins = tri.find_simplex((ctr - p) @ R) >= 0
            sel = sub[ins]
            if len(sel):
                st = self.state[sel[:, 0], sel[:, 1], sel[:, 2]]
                keep = st != OCC
                self.state[sel[keep, 0], sel[keep, 1], sel[keep, 2]] = FREE
                changed = True
        if changed:
            self._touch()

    def clear_box(self, R, p, lo, hi, margin):
        """Unlabeled voxels and cloud points inside an oriented box (frame R, p; local corners lo, hi; grown by margin)
        become free (a held object's own volume). Returns the number of cloud points removed."""
        lo = np.asarray(lo, float) - margin
        hi = np.asarray(hi, float) + margin
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = p + corners @ R.T

        def inside(P):
            L = (P - p) @ R
            return np.all((L >= lo) & (L <= hi), axis=1)
        vlo = np.clip(self.index(W.min(0)), 0, np.array(self.shape) - 1)
        vhi = np.clip(self.index(W.max(0)), 0, np.array(self.shape) - 1)
        if np.all(vhi >= vlo):
            sub = np.stack(np.meshgrid(*[np.arange(vlo[i], vhi[i] + 1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
            sel = sub[inside(self.lo + (sub + 0.5) * self.res)]
            if len(sel):
                free = self.label[sel[:, 0], sel[:, 1], sel[:, 2]] == 0
                self.state[sel[free, 0], sel[free, 1], sel[free, 2]] = FREE
        removed = 0
        chunks = []
        for ch in self._chunks:
            gone = inside(ch) if len(ch) else np.zeros(0, bool)
            removed += int(gone.sum())
            chunks.append(ch[~gone] if len(ch) else ch)
        if removed:
            self._chunks = chunks
            self.cloud_version += 1
            self.merge_cloud()
        self._touch()
        return removed

    def add_surface(self, pts):
        """Observed surface points from touch (a contact the cameras did not show): into the surface cloud and as
        occupied, unlabelled voxels."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        idx = self.index(pts)
        ok = self.inside(idx)
        if not ok.any():
            return
        self._chunks.append(voxel_down(pts[ok], 0.003))
        self.cloud_version += 1
        self.merge_cloud()
        ii = tuple(idx[ok].T)
        self.state[ii] = OCC
        self.skin[ii] = True
        self._touch()

    def clear_label(self, oid):
        sel = (self.label == oid) | (self.shadow == oid)
        self.state[sel] = FREE
        self.label[sel] = 0
        self.shadow[sel] = 0
        self._touch()

    def insert_object(self, obj: ObjectModel, seen_free=None, force=False, hollow=False, shell=0.02):
        """Insert an object model's solid (surface points + 2.5D fill) with its label. The fill below the observed
        surface is an inference: it never overwrites space a camera saw free (now: FREE in this map; earlier:
        seen_free, the remembered free space), so a cavity under a roof that a camera has looked into stays free.
        force: the object has just been put there (released): its whole solid is occupied whatever was seen before.
        hollow: a container (something is put in it): the fill is only a wall's thickness (`shell`) below its
        observed surface; the unseen space its walls and roof enclose stays as it was (unknown until a camera looks
        in), not inferred solid."""
        pts = obj.points
        if len(pts) == 0:
            return
        idx = self.index(pts)
        idx = idx[self.inside(idx)]
        self.state[idx[:, 0], idx[:, 1], idx[:, 2]] = OCC
        self.label[idx[:, 0], idx[:, 1], idx[:, 2]] = obj.oid
        # fill the column below the surface down to the object's bottom
        lo, hi = obj.bbox
        a = self.index(np.array([lo[0], lo[1], obj.bottom_z]))
        b = self.index(np.array([hi[0], hi[1], hi[2]]))
        a = np.clip(a, 0, np.array(self.shape) - 1)
        b = np.clip(b, 0, np.array(self.shape) - 1)
        sub = self.centers.reshape(*self.shape, 3)[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1].reshape(-1, 3)
        ins = obj.inside(sub, tol=0.0)
        if ins.any():
            idx2 = self.index(sub[ins])
            idx2 = idx2[self.inside(idx2)]
            if hollow and not force and len(idx2):
                kt = np.full(self.shape[:2], -1, int)            # the observed surface's top voxel per column
                np.maximum.at(kt, (idx[:, 0], idx[:, 1]), idx[:, 2])
                depth = kt[idx2[:, 0], idx2[:, 1]] - idx2[:, 2]
                idx2 = idx2[(depth >= 0) & (depth <= max(1, int(round(shell / self.res))))]
            if not force:
                ii = (idx2[:, 0], idx2[:, 1], idx2[:, 2])
                keep = self.state[ii] != FREE
                if seen_free is not None:
                    keep &= ~seen_free[ii]
                idx2 = idx2[keep]
            self.state[idx2[:, 0], idx2[:, 1], idx2[:, 2]] = OCC
            self.label[idx2[:, 0], idx2[:, 1], idx2[:, 2]] = obj.oid
        self._touch()

    def propagate_label(self, obj, radius_vox=1):
        """Unlabelled occupied / unknown voxels touching an object's points belong to that object."""
        if len(obj.points) == 0:
            return
        idx = self.index(obj.points)
        idx = idx[self.inside(idx)]
        m = np.zeros(self.shape, bool)
        m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        m = ndimage.binary_dilation(m, iterations=radius_vox)
        occ = m & (self.state == OCC) & (self.label == 0)
        self.label[occ] = obj.oid
        unk = m & (self.state == UNKNOWN) & (self.shadow == 0)
        self.shadow[unk] = obj.oid
        self._touch()

    def finalize(self):
        """Shadow attribution + free-above assumption. Call after a batch of integrations."""
        self.merge_cloud()
        st, lab = self.state, self.label
        shadow = np.zeros_like(lab)
        cur = np.zeros(self.shape[:2], np.int16)
        for k in range(self.shape[2] - 1, -1, -1):
            s = st[:, :, k]
            l = lab[:, :, k]
            cur = np.where((s == OCC) & (l > 0), l, cur)
            cur = np.where((s == FREE) | ((s == OCC) & (l == 0)), 0, cur)
            shadow[:, :, k] = np.where(s == UNKNOWN, cur, 0)
        self.shadow = shadow
        if self.free_above is not None:
            kz = int(np.clip(self.layer(self.free_above), 0, self.shape[2]))
            top = self.state[:, :, kz:]
            top[top == UNKNOWN] = FREE
        if self.out_of_view_free_z is not None:
            # space no camera has ever had in view (beside / above the scene) can only hold what the robot moves
            # into; occluded space inside a view stays unknown
            kz = int(np.clip(self.layer(self.out_of_view_free_z), 0, self.shape[2]))
            sub = self.state[:, :, kz:]
            sub[(sub == UNKNOWN) & ~self.seen[:, :, kz:]] = FREE
        if self.robot_zone is not None:
            cx, cy, r, z0, z1 = self.robot_zone
            c = self.centers.reshape(*self.shape, 3)
            zone = ((c[..., 0] - cx) ** 2 + (c[..., 1] - cy) ** 2 <= r * r) & (c[..., 2] >= z0) & (c[..., 2] <= z1)
            self.state[zone] = FREE
            self.label[zone] = 0
            self.shadow[zone] = 0
        self._touch()

    # ------------------------------------------------------------ queries
    def obstacles(self, exclude=(), include_unknown=True):
        obs = self.state == OCC
        if include_unknown:
            obs = obs | (self.state == UNKNOWN)
        if self.table_z is not None:
            # the table top is handled analytically as a plane (exact), not by 1 cm voxels:
            # drop everything below the table surface voxel, and the surface voxel itself
            # unless something stands on it (object bases keep their lowest slice)
            tz = self.visual_table_z if getattr(self, "visual_table_z", None) is not None else self.table_z
            k0 = int(np.clip(self.layer(tz), 0, self.shape[2] - 2))
            obs[:, :, :k0] = False
            above = (self.state[:, :, k0 + 1] == OCC)
            obs[:, :, k0] &= above
        for oid in exclude:
            obs &= ~((self.label == oid) & (self.state == OCC))
            obs &= ~(self.shadow == oid)
        return obs

    def esdf(self, exclude=(), include_unknown=True):
        key = tuple(sorted(int(e) for e in exclude)) + (() if include_unknown else ("known_only",))
        if key not in self._cache:
            obs = self.obstacles(exclude, include_unknown)
            self._cache[key] = (ndimage.distance_transform_edt(~obs) * self.res).astype(np.float32)
        return self._cache[key]

    def esdf_from(self, obs):
        return (ndimage.distance_transform_edt(~obs) * self.res).astype(np.float32)

    def support_slab(self, xy_lo, xy_hi, z_top):
        """Boolean grid of voxels forming a support surface slab (z <= z_top) inside an xy box."""
        m = np.zeros(self.shape, bool)
        a = np.clip(self.index(np.array([xy_lo[0], xy_lo[1], self.lo[2]])), 0, np.array(self.shape) - 1)
        b = np.clip(self.index(np.array([xy_hi[0], xy_hi[1], z_top])), 0, np.array(self.shape) - 1)
        m[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = True
        return m

    def clearance(self, pts, exclude=(), dist=None, include_unknown=True):
        """Conservative distance from points (...,3) to the nearest obstacle voxel surface (unobserved space counts
        unless include_unknown is False)."""
        dist = self.esdf(exclude, include_unknown) if dist is None else dist
        shp = pts.shape[:-1]
        p = pts.reshape(-1, 3)
        idx = self.index(p)
        ok = self.inside(idx)
        out = np.zeros(len(p), np.float32)
        out[ok] = dist[idx[ok, 0], idx[ok, 1], idx[ok, 2]] - self.res
        above = (~ok) & (p[:, 2] >= self.hi[2])
        out[above] = 1.0
        return out.reshape(shp)

    def occupied_labels_near(self, pts, radius=0.02):
        idx = self.index(pts)
        idx = idx[self.inside(idx)]
        return np.unique(self.label[idx[:, 0], idx[:, 1], idx[:, 2]])

    def heightmap(self, x_rng, y_rng, exclude=(), z_min=None):
        """Max occupied height per xy cell (voxel columns) within ranges; -inf if empty."""
        obs = self.state == OCC
        for oid in exclude:
            obs &= ~(self.label == oid)
        a = self.index(np.array([x_rng[0], y_rng[0], self.lo[2]]))
        b = self.index(np.array([x_rng[1], y_rng[1], self.lo[2]]))
        a = np.clip(a, 0, np.array(self.shape) - 1)
        b = np.clip(b, 0, np.array(self.shape) - 1)
        sub = obs[a[0]:b[0] + 1, a[1]:b[1] + 1, :]
        zs = self.lo[2] + (np.arange(self.shape[2]) + 1.0) * self.res
        h = np.where(sub, zs[None, None, :], -np.inf).max(axis=2)
        return h, (self.lo[0] + a[0] * self.res, self.lo[1] + a[1] * self.res)


def mask_points(frame, mask, erode=2):
    if erode > 0:
        mask = ndimage.binary_erosion(mask, iterations=erode)
    mask = mask & ~ndimage.binary_dilation(frame.robot_mask, iterations=2)
    pts, _ = frame.backproject(mask=mask)
    return pts


def grow_region(frame, seed_rc, table_z, radius=0.12, step=0.012, min_height=0.004):
    """Depth-continuous region growing from a seed pixel (segmentation fallback)."""
    H, W = frame.depth.shape
    pts, rc = frame.backproject()
    P = np.full((H, W, 3), np.nan)
    P[rc[:, 0], rc[:, 1]] = pts
    r0, c0 = int(np.clip(seed_rc[0], 0, H - 1)), int(np.clip(seed_rc[1], 0, W - 1))
    seed = P[r0, c0]
    if not np.all(np.isfinite(seed)):
        return np.zeros((H, W), bool)
    valid = np.isfinite(P[..., 2]) & (P[..., 2] > table_z + min_height) & ~frame.robot_mask
    valid &= np.linalg.norm(np.nan_to_num(P, nan=1e3) - seed, axis=2) < radius
    out = np.zeros((H, W), bool)
    if not valid[r0, c0]:
        # snap to nearest valid pixel
        cand = np.argwhere(valid)
        if len(cand) == 0:
            return out
        k = np.argmin(np.abs(cand - [r0, c0]).sum(1))
        r0, c0 = cand[k]
    stack = [(r0, c0)]
    out[r0, c0] = True
    while stack:
        r, c = stack.pop()
        p = P[r, c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not out[rr, cc] and valid[rr, cc]:
                if np.linalg.norm(P[rr, cc] - p) < step:
                    out[rr, cc] = True
                    stack.append((rr, cc))
    return out


def project_mask(frame, pts, radius_px=3):
    H, W = frame.depth.shape
    m = np.zeros((H, W), bool)
    if len(pts) == 0:
        return m
    uv, z = frame.project(pts)
    ok = (z > 0.03) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not ok.any():
        return m
    u = uv[ok, 0].astype(int)
    v = uv[ok, 1].astype(int)
    # keep only points that are visible (not occluded) in this frame
    vis = z[ok] <= frame.depth[v, u] + 0.02
    m[v[vis], u[vis]] = True
    if radius_px > 0:
        m = ndimage.binary_dilation(m, iterations=radius_px)
        m = ndimage.binary_closing(m, iterations=2)
    return m


class NearField:
    """Exact near-contact distances (used where the 1 cm voxel ESDF is too coarse).

    Obstacles: the fused surface cloud (3 mm samples, carved by later views), labelled by proximity to
    object models; unknown voxels except the skin of observed surfaces, as exact cubes; occupied voxels
    without observed points (inferred solids). The table top stays an analytic plane (checked
    separately). Points of held objects are dropped (the object is in the hand, not where it was seen).
    """
    SAMPLING = 0.002           # half the spacing of the 3 mm surface samples

    def __init__(self, get_scene, objects):
        self._get_scene = get_scene
        self.objects = objects
        self._lab_key = None
        self._struct = {}
        self._struct_ver = None

    def _labels(self, scene):
        key = (id(scene), scene.cloud_version,
               tuple((oid, len(m.points), bool(m.held), float(m.points[:, 0].sum()) if len(m.points) else 0.0)
                     for oid, m in sorted(self.objects.items())))
        if key == self._lab_key:
            return
        cloud = scene.cloud
        lab = np.zeros(len(cloud), np.int32)
        best = np.full(len(cloud), 0.006)
        held = []
        for oid, m in self.objects.items():
            if len(m.points) == 0 or len(cloud) == 0 or getattr(m, "virtual", False):
                continue
            d = m.surface_distance(cloud)
            sel = d < best
            lab[sel] = oid
            best[sel] = d[sel]
            if m.held:
                held.append(oid)
        self.held = tuple(held)
        self.cloud, self.lab = cloud, lab
        self._lab_key = key
        self._struct = {}

    def _structs(self, exclude, ignore_box, free_near=None, unknown=True):
        scene = self._get_scene()
        self._labels(scene)
        ver = (id(scene), scene.version)
        if ver != self._struct_ver:
            self._struct = {}
            self._struct_ver = ver
            idx = scene.index(self.cloud)
            ok = scene.inside(idx)
            has = np.zeros(scene.shape, bool)
            has[tuple(idx[ok].T)] = True
            self._cube_base = ((scene.state == UNKNOWN) & ~scene.skin) | ((scene.state == OCC) & ~has)
        ex = tuple(sorted(set(int(e) for e in exclude) | set(self.held)))
        fk = None if not free_near else tuple((len(q), round(float(r), 4), round(float(np.asarray(q).sum()), 3), t)
                                              for q, r, t in free_near)
        key = (ex, None if ignore_box is None else tuple(np.round(np.r_[ignore_box[0][:2], ignore_box[1][:2],
                                                                          ignore_box[2]], 4)), fk, unknown)
        if key in self._struct:
            return self._struct[key]
        keep = ~np.isin(self.lab, ex) if ex else np.ones(len(self.lab), bool)
        cubes = scene.obstacles(ex) & self._cube_base
        if not unknown:
            cubes &= scene.state != UNKNOWN          # unobserved space is not an obstacle here (inferred solids stay)
        if ignore_box is not None:
            lo, hi, ztop = ignore_box
            c = self.cloud
            inb = (c[:, 0] >= lo[0]) & (c[:, 0] <= hi[0]) & (c[:, 1] >= lo[1]) & (c[:, 1] <= hi[1]) & (c[:, 2] <= ztop)
            keep &= ~inb
            cubes &= ~scene.support_slab(lo, hi, ztop)
        for q, r, pts_too in (free_near or ()):
            # observed but unsegmented surface hugging a part (e.g. the rest of a handle) belongs to it
            q = np.asarray(q, float)
            if not pts_too or len(q) == 0:
                continue
            c = self.cloud
            lo_, hi_ = q.min(0) - r, q.max(0) + r
            cand = np.nonzero(keep & np.all((c >= lo_) & (c <= hi_), axis=1))[0]
            if len(cand):
                dq, _ = cKDTree(q).query(c[cand], distance_upper_bound=r)
                keep[cand[np.isfinite(dq) & (self.lab[cand] == 0)]] = False
        pts = self.cloud[keep]
        ct = cKDTree(pts) if len(pts) else None
        if free_near:
            # ... and so does the space hugging them that no surface explains (occluded / unlabelled voxels): a point
            # in such a voxel is not inside an obstacle
            cubes = cubes.copy()
            for q, r, _ in free_near:
                q = np.asarray(q, float)
                if len(q) == 0:
                    continue
                a = np.clip(scene.index(q.min(0) - r), 0, np.array(scene.shape) - 1)
                b = np.clip(scene.index(q.max(0) + r), 0, np.array(scene.shape) - 1)
                sub = cubes[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1]
                cid = np.argwhere(sub)
                if len(cid):
                    cen = scene.lo + (cid + a + 0.5) * scene.res
                    dq, _ = cKDTree(q).query(cen, distance_upper_bound=r + 0.5 * scene.res)
                    hit = cid[np.isfinite(dq)]
                    sub[hit[:, 0], hit[:, 1], hit[:, 2]] = False
        full = cubes
        # only boundary cubes can be nearest to a point outside the unknown region (inside ones: distance 0)
        cubes = cubes & ~ndimage.binary_erosion(cubes, structure=np.ones((3, 3, 3), bool), border_value=1)
        cidx = np.argwhere(cubes)
        centers = scene.lo + (cidx + 0.5) * scene.res
        for q, r, _ in (free_near or ()):
            # unobserved space hugging these surfaces belongs to them (occluded gaps of a part / fixture)
            q = np.asarray(q, float)
            if len(q) == 0 or len(centers) == 0:
                continue
            lo_, hi_ = q.min(0) - r - scene.res, q.max(0) + r + scene.res
            cand = np.nonzero(np.all((centers >= lo_) & (centers <= hi_), axis=1))[0]
            if len(cand):
                dq, _ = cKDTree(q).query(centers[cand], distance_upper_bound=r + 0.5 * scene.res)
                centers = np.delete(centers, cand[np.isfinite(dq)], axis=0)
        ut = cKDTree(centers) if len(centers) else None
        self._struct[key] = (ct, ut, centers, scene.res, full)
        return self._struct[key]

    def distance(self, P, exclude=(), ignore_box=None, horizon=0.08, free_near=None, unknown=True):
        """Exact distance (m) from points (N,3) to the nearest obstacle (capped at `horizon`)."""
        ct, ut, centers, res, full = self._structs(exclude, ignore_box, free_near, unknown)
        P = np.asarray(P, float).reshape(-1, 3)
        d = np.full(len(P), horizon)
        if len(P) == 0:
            return d
        scene = self._get_scene()
        idx = scene.index(P)
        ins = scene.inside(idx)
        within = np.zeros(len(P), bool)
        within[ins] = full[idx[ins, 0], idx[ins, 1], idx[ins, 2]]
        if ct is not None:
            dc, _ = ct.query(P, k=1, distance_upper_bound=horizon)
            d = np.minimum(d, dc - self.SAMPLING)
        if ut is not None:
            k = min(6, len(centers))
            du, iu = ut.query(P, k=k, distance_upper_bound=horizon + res)
            if k == 1:
                du, iu = du[:, None], iu[:, None]
            valid = iu < len(centers)
            if valid.any():
                C = centers[np.where(valid, iu, 0)]
                gap = np.maximum(np.abs(P[:, None, :] - C) - 0.5 * res, 0.0)
                dd = np.where(valid, np.linalg.norm(gap, axis=2), np.inf)
                d = np.minimum(d, dd.min(axis=1))
        d[within] = 0.0                       # inside unknown / inferred-solid space
        return d
