"""Depth evidence for a moving surface along a calibrated joint screw."""
import numpy as np
from scipy.spatial import cKDTree


def _projection(points, view):
    depth, intrinsic, pose = view
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    focal = intrinsic.diagonal()[:2]
    uv = camera[:, :2] / np.maximum(z[:, None], .001) * focal + intrinsic[:2, 2]
    xy = np.floor(uv).astype(int)
    valid = (z > .02) & (xy >= 0).all(1) & (xy < [depth.shape[1], depth.shape[0]]).all(1)
    xy = np.clip(xy, 0, [depth.shape[1]-1, depth.shape[0]-1])
    return uv, xy.T, z, valid


def visible_normals(points, screws, normals, axis, origin, surfaces, views, gain_threshold):
    """Validate screw normals against the depth resolution of their own camera.

    The open side of a depth discontinuity measures the outward image normal.
    Points on an occluding hand and holes provide no such evidence. Comparing
    a displaced sample to its own handoff camera also avoids treating the
    offset between two cameras' calibrations as a moving edge.
    """
    if not views or surfaces is None or not len(points):
        return normals, np.ones(len(points), dtype=bool)
    result = normals.copy()
    quality = np.zeros(len(points))
    verified = np.zeros(len(points), bool)
    arm = points-origin
    radial = arm-(arm@axis)[:, None]*axis
    base = points-radial
    for surface, view in zip(surfaces, views):
        if len(surface) < 6:
            continue
        depth, intrinsic, pose = view
        uv, (x, y), z, inside = _projection(points, view)
        distance = cKDTree(surface).query(points)[0]
        measured = depth[y, x]
        # Pixel footprint in metres plus metric range uncertainty.
        pixel = z / min(intrinsic[0, 0], intrinsic[1, 1])
        band = .0015 + .0025*z*z
        use = inside & (distance < .0015) & (np.abs(measured-z) < band)
        if not use.any():
            continue
        grad = np.zeros((len(points), 2))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if not (dx or dy):
                    continue
                xx = np.clip(x+dx, 0, depth.shape[1]-1)
                yy = np.clip(y+dy, 0, depth.shape[0]-1)
                difference = depth[yy, xx]-measured
                weight = np.where(difference > 2*band, np.minimum(1., difference/.015), 0.)
                grad += weight[:, None] * np.array([dx,dy]) / (dx*dx+dy*dy)
        image_normal = np.column_stack((grad * intrinsic.diagonal()[:2],
                                        -np.sum(grad*(uv-intrinsic[:2,2]), axis=1)))
        edge_normal = image_normal @ pose[:3, :3].T
        edge_normal /= np.maximum(np.linalg.norm(edge_normal, axis=1, keepdims=True), 1.e-10)
        edge_gain = np.abs(np.sum(edge_normal * screws, axis=1))
        intrinsic_gain = np.abs(np.sum(normals * screws, axis=1))
        edge = (np.linalg.norm(grad, axis=1) > .8) & (edge_gain > intrinsic_gain)
        candidate = np.where(edge[:, None], edge_normal, normals)
        gain = np.abs(np.sum(candidate*screws, axis=1))
        resolved = use & (.25*gain > .40*pixel)
        # A surface that rotates onto the same measured shell is locally
        # symmetric. Test both directions without using any object parameters.
        votes = np.zeros(len(points), bool)
        for angle in (-.24, -.12, .12, .24):
            displaced = base + np.cos(angle)*radial + np.sin(angle)*screws
            _, (u,v), zz, valid = _projection(displaced, view)
            votes |= valid & (depth[v,u] - zz > band + .001)
        valid = resolved & votes & (gain > gain_threshold)
        better = valid & (gain > quality)
        result[better] = candidate[better]
        quality[better] = gain[better]
        verified |= valid
    return result, verified
