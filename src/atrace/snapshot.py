"""Scene snapshot export (DESIGN §3.2): the harness's voxel map and object models as one versioned SceneSnapshot.

TRACE and the per-step guard must check the same geometry, so the snapshot carries CP-5's exact near field
(observed surface cloud + unknown-space cubes, refined below 3.5 cm) as `exact`, and the grids follow CP-5's
obstacle rules: the table top is the analytic plane (its voxel layer is not an obstacle) and the held object
travels as payload spheres, never as occupied space.
"""
from __future__ import annotations

import numpy as np
from traceplan.contract import Entity, SceneSnapshot

from .perception import UNKNOWN
from .sim_iface import WRIST_CAMERA

PERCEPTION = {"sigma0": 0.003, "sigma_far": 0.006, "max_points": 600}   # perception model (configurable, not tuned)


def entity_cov(obj, cfg=PERCEPTION):
    """Position covariance of an object model: 3 mm everywhere, plus 6 mm until the wrist camera has seen it close."""
    close = any(v[0] == WRIST_CAMERA for v in obj.views)
    return np.eye(3) * (cfg["sigma0"] ** 2 + (0.0 if close else cfg["sigma_far"] ** 2))


def entity_names(h):
    """oid -> entity id: the plan's object id (the first plan id bound to it), else oid<n>."""
    out = {}
    for sid, oid in h.id_map.items():
        out.setdefault(oid, sid)
    return {oid: out.get(oid, f"oid{oid}") for oid in h.objects}


def entity_of(h, oid):
    return entity_names(h)[oid]


def _model_key(h):
    return tuple((oid, len(m.points), bool(m.held), float(m.points[:, 0].sum()) if len(m.points) else 0.0)
                 for oid, m in sorted(h.objects.items())), h.held


def build_snapshot(h, version_key=None, cfg=PERCEPTION):
    """The current scene as a SceneSnapshot, cached: a new version exactly when the map or an object model changed
    (or version_key differs), so a version names one geometry."""
    scene = h.scene
    key = (scene.version, _model_key(h), version_key)
    cache = getattr(h, "_snap_cache", None)
    if cache is not None and cache[0] is scene and cache[1] == key:
        return cache[2]
    held = (h.held,) if h.held is not None else ()
    occ = scene.obstacles(held, include_unknown=False)
    unknown = scene.obstacles(held, include_unknown=True) & (scene.state == UNKNOWN)
    names = entity_names(h)
    Rs, ps = h.rb.eef_pose()
    entities = {}
    for oid, m in h.objects.items():
        if len(m.points) == 0:
            continue
        if m.held:
            att = h.checker.attached
            P = ps + (att.model_site if att.model_site is not None else att.pts_site) @ Rs.T   # where it is now
            fc, base = 0.5 * (P[:, :2].min(0) + P[:, :2].max(0)), float(P[:, 2].min())
        else:
            P, fc, base = m.points, m.footprint_center(), float(m.support_z)
        pts = P if len(P) <= cfg["max_points"] else \
            P[np.random.default_rng(oid).choice(len(P), cfg["max_points"], replace=False)]
        top = float(np.percentile(P[:, 2], 99.5))
        entities[names[oid]] = Entity(names[oid], int(oid), pts, entity_cov(m, cfg),
                                      {"top_center": np.r_[fc, top], "base": np.r_[fc, base], "centroid": P.mean(0)})
    labels = {e.id: e.label for e in entities.values()}
    checker = h.checker

    def exact(pts, exclude=(), unknown=True):
        # unknown: unseen space counts as an obstacle (the guard's and TRACE's safety terms pass False: observed
        # geometry only; unseen space is a relaxable require term, crossed slowly after a look)
        return checker.point_clearance(np.asarray(pts, float), tuple(labels[e] for e in exclude if e in labels) + held,
                                       unknown=unknown)

    h._snap_version = getattr(h, "_snap_version", 0) + 1
    snap = SceneSnapshot(h._snap_version, scene.lo.copy(), scene.res, occ, unknown,
                         np.where(occ, scene.label, 0).astype(np.int16),
                         np.where(unknown, scene.shadow, 0).astype(np.int16), scene.table_z, entities, exact,
                         {"map_version": int(scene.version), "held_oid": h.held,
                          "table_z_visual": getattr(h, "table_z_visual", None)})
    h._snap_cache = (scene, key, snap)
    return snap
