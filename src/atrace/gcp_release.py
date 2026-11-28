"""Rigid registration to camera-specific, measured handoff or held surfaces."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .geometry import mat_to_rotvec, rotvec_to_mat
from .gcp_tracking import _above_plane, _depth_pixels, _estimate_normals, _sample_surface, _unoccluded


@dataclass(frozen=True)
class Surface:
    """Measured points and normals in the original object frame about center0."""
    points: np.ndarray
    normals: np.ndarray
    spacing: float
    view: object
    measured_center: np.ndarray
    measured_rotation: np.ndarray
    background: object = None


@dataclass(frozen=True)
class DepthPatch:
    """Last acquisition's associated camera returns for checking a pose innovation."""
    points: np.ndarray
    view: object
    measured_center: np.ndarray
    measured_rotation: np.ndarray


def initial_surface(points, center, plane, rng, view=None, rotation=None, measured_center=None, background=None):
    if plane is not None:
        points = points[_above_plane(points, plane) > .005]
    points = _sample_surface(points, 900, rng, scale=.002)
    normals = _estimate_normals(points, points.mean(axis=0)) if len(points) else points.copy()
    spacing = np.median(cKDTree(points).query(points, k=2)[0][:, 1]) if len(points) > 1 else .008
    return Surface(points - center, normals, float(spacing), view,
                   center.copy() if measured_center is None else measured_center.copy(),
                   np.eye(3) if rotation is None else rotation.copy(), background)


def refresh_surfaces(tracker, clouds, camera_views, rng):
    tracker.previous_depth_patches = ()
    center = tracker.center0 + tracker.translation
    predicted = tracker._free_surface()
    tracker.anchor_center, tracker.anchor_rotation = center.copy(), tracker.rotation.copy()
    surfaces, observed = [], []
    for ref, live, views in zip(tracker.reference_surfaces, clouds, camera_views):
        visible = predicted[_unoccluded(predicted, views)]
        static = None
        if len(visible):
            distance = cKDTree(visible).query(live)[0]
            static = cKDTree(live[distance > .014])
            live = live[distance < .010]
        else:
            live = live[:0]
        if len(live) >= 6:
            observed.append(live)
            canonical = tracker.center0 + (live - center) @ tracker.rotation
            ref = initial_surface(canonical, tracker.center0, None, rng, views[0] if views else None,
                                  tracker.rotation, center, static)
        surfaces.append(ref)
    tracker.reference_surfaces = tuple(surfaces)
    if observed:
        tracker.observed = np.concatenate(observed)
        tracker.visible = len(tracker.observed) >= 6


_ANGLES = 2. * np.pi * np.arange(24) / 24
_DIRECTIONS = np.vstack((np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES), np.zeros(len(_ANGLES)))),
                         [0., 0., 1.], [0., 0., -1.]))


@dataclass
class SurfaceFit:
    jacobian: np.ndarray
    residual: np.ndarray
    weights: np.ndarray
    within_noise: bool
    observed: np.ndarray
    translation_directions: np.ndarray
    vertical_gain: float


def _correspondences(ref, live, tree, center, rotation, views, envelope):
    """Truncated forward/backward normal distances and measured exterior support."""
    model = ref.points @ rotation.T + center
    seen = _unoccluded(model, views)
    if seen.sum() < 3 or len(live) < 4:
        return None
    visible, normals = model[seen], (ref.normals @ rotation.T)[seen]
    # The handoff camera also defines a visibility test. A point whose
    # putative source was hidden by the robot or the object at handoff is
    # newly revealed geometry, and cannot be matched to a different face.
    crossed = np.ones(len(live), dtype=bool)
    if ref.view is not None:
        at_reference = ((live - center) @ rotation @ ref.measured_rotation.T
                        + ref.measured_center)
        source_visible = _unoccluded(at_reference, [ref.view])
        # Only a depth ray that previously passed through this point provides
        # evidence of a new exterior edge. A co-depth background patch or
        # previously unsampled portion of the surface supplies no edge motion.
        uv, z, valid = _depth_pixels(at_reference, ref.view)
        crossed = np.zeros(len(live), dtype=bool)
        u, v = uv[valid].T
        crossed[valid] = ref.view[0][v, u] > z[valid] + .004
        crossed = crossed[source_visible]
        live = live[source_visible]
        if len(live) < 3:
            return None
        tree = cKDTree(live)
    data_distance, to_model = cKDTree(visible).query(live)
    model_distance, to_data = tree.query(visible)
    # A measurement can cover several finer template samples. Require their
    # reciprocal partners to lie within the local sampling radius.
    radius = np.clip(1.5 * ref.spacing, .003, .012)
    data_valid = ((data_distance < .021) &
                  (np.linalg.norm(live - live[to_data[to_model]], axis=1) < radius))
    model_valid = ((model_distance < .021) &
                   (np.linalg.norm(visible - visible[to_model[to_data]], axis=1) < radius))
    if data_valid.sum() < 3 or model_valid.sum() < 3:
        return None
    x = np.vstack((visible[to_model[data_valid]], visible[model_valid]))
    y = np.vstack((live[data_valid], live[to_data[model_valid]]))
    n = np.vstack((normals[to_model[data_valid]], normals[model_valid]))
    residual = np.einsum("ij,ij->i", y - x, n)
    jacobian = np.column_stack((n, np.cross(x - center, n)))
    weights = np.minimum(1., .003 / np.maximum(np.abs(residual), 1e-8))
    weight = min(len(x), 100) / len(x)
    weights *= weight
    normal_error = np.quantile(np.abs(residual), .7)
    # Test the entire observed footprint. Holes and disappearing edges are
    # interior to these half-spaces and supply no residual.
    inliers = live[(data_distance < .026) & crossed]
    support = envelope @ _DIRECTIONS.T
    extreme = support.argmax(axis=0)
    proj = inliers @ _DIRECTIONS.T
    measured = proj.max(axis=0, initial=-np.inf)
    excess = measured - support[extreme, np.arange(len(extreme))]
    active = excess > -.0015
    edge_error = float(np.quantile(np.maximum(0., excess), .87))
    if active.any():
        directions = _DIRECTIONS[active]
        edge_x = envelope[extreme[active]]
        edge_j = np.column_stack((directions, np.cross(edge_x - center, directions)))
        jacobian = np.vstack((jacobian, edge_j))
        residual = np.r_[residual, np.maximum(0., excess[active])]
        weights = np.r_[weights, np.full(len(directions), weight * len(x) / len(directions))]
    observed = live[data_distance < .012]
    # A visible perimeter constrains tangential motion even at zero residual.
    # An aperture exposing only the middle of a flat face has no such support.
    edge_directions = np.empty((0, 3))
    if len(observed):
        reaches = (observed @ _DIRECTIONS.T).max(axis=0)
        perimeter = np.abs(reaches - support[extreme, np.arange(len(extreme))]) < max(.0025, ref.spacing)
        perimeter &= _unoccluded(envelope[extreme], views)
        edge_directions = _DIRECTIONS[perimeter]
    if len(edge_directions):
        edge_directions = edge_directions * np.sqrt(len(n) / len(edge_directions))
    translation_directions = np.vstack((n, edge_directions))
    within_noise = (normal_error < np.clip(.55 * ref.spacing, .0015, .0045) and
                    edge_error < np.clip(.55 * ref.spacing, .0015, .005))
    return SurfaceFit(jacobian, residual, weights, within_noise, observed, translation_directions,
                      float(np.mean(n[:, 2] ** 2)))


def _ray_cost(points, view):
    """Change in free-space error on rays shared with the unshifted pose."""
    shape = points.shape[:2]
    pixels, z, inside = _depth_pixels(points.reshape(-1, 3), view)
    u, v = pixels[inside].T
    difference = np.full(len(z), np.nan)
    difference[inside] = view[0][v, u] - z[inside]
    difference = difference.reshape(shape)
    visible = np.isfinite(difference) & (difference > -.007)
    residual = np.where(visible, np.clip(difference - .002, 0., .025), 0.)
    # The first proposal is the current center. A point that disappears
    # behind an occluder under another proposal cannot certify motion.
    shared = visible & visible[:1]
    change = np.where(shared, residual - residual[:1], 0.)
    return change.sum(axis=1) / max(5, visible[0].sum())


def _ray_translation(usable, center, rotation):
    """Coarse translation from rays that traverse the model in either frame."""
    cameras = []
    for ref, live, _, views in usable:
        if not views or ref.view is None:
            continue
        model = ref.points @ rotation.T
        near = cKDTree(model + center).query(live)[0] < .050
        surface = live[near]
        if len(surface) >= 4:
            # Subsample each view independently so the fine-grained camera
            # cannot swamp a geometrically complementary distant view.
            model = model[::max(1, len(model) // 180)]
            surface = surface[::max(1, len(surface) // 220)]
            weight = min(1., len(model) / 20) * min(1., len(surface) / 20)
            cameras.append((model, surface, ref, views[0], weight))
    if not cameras:
        return center.copy()
    weight_sum = sum(camera[-1] for camera in cameras)
    offsets = np.vstack((np.eye(3), -np.eye(3), _DIRECTIONS[[3, 9, 15, 21]]))
    start = center.copy()
    for distance in (.010, .005, .0025):
        centers = np.vstack((center, center + distance * offsets))
        costs = np.zeros(len(centers))
        for model, surface, ref, current, weight in cameras:
            live_model = model + centers[:, None, :]
            former_surface = ((surface - centers[:, None, :]) @ rotation @
                              ref.measured_rotation.T + ref.measured_center)
            costs += weight * (_ray_cost(live_model, current) + _ray_cost(former_surface, ref.view))
        costs /= weight_sum
        costs += .027 * np.linalg.norm(centers - start, axis=1)
        center = centers[np.argmin(costs)]
    return center


def _anchor_consistent(tracker, usable):
    """Check both directions of depth evidence at the measured anchor pose."""
    center = tracker.anchor_center
    rotation = tracker.anchor_rotation
    current_center = tracker.center0 + tracker.translation
    separated = np.linalg.norm(current_center - center) > .0025
    current_contradiction = False
    observed = 0
    for ref, live, tree, views in usable:
        if not views or ref.view is None:
            continue
        points = ref.points @ rotation.T + center
        uv, depth, valid = _depth_pixels(points, views[0])
        u, v = uv[valid].T
        difference = views[0][0][v, u] - depth[valid]
        seen = np.isfinite(difference) & (difference > -.007)
        if seen.sum() < 4:
            continue
        matched = np.count_nonzero(tree.query(points[valid][seen])[0] < max(.005, ref.spacing))
        contradiction = np.count_nonzero(difference > .011)
        if contradiction > max(2, .09 * seen.sum()) or matched < .60 * seen.sum():
            return False
        # New points can be previously occluded even in a static scene.
        # Only well-separated exterior points confirmed by a former free
        # depth ray provide independent evidence against a stationary anchor.
        distance = cKDTree(points).query(live)[0]
        outside = live[(distance > max(.007, 1.7 * ref.spacing)) & (distance < .030)]
        at_ref = (outside - center) @ rotation @ ref.measured_rotation.T + ref.measured_center
        uv, depth, valid = _depth_pixels(at_ref, ref.view)
        u, v = uv[valid].T
        invading = np.count_nonzero(ref.view[0][v, u] - depth[valid] > .009)
        if invading > max(3, .1 * seen.sum()):
            return False
        if separated:
            current = ref.points @ tracker.rotation.T + current_center
            uv, z, valid = _depth_pixels(current, views[0])
            u, v = uv[valid].T
            gap = views[0][0][v, u] - z[valid]
            current_contradiction |= (np.count_nonzero(gap > .012) >
                                      max(5, .25 * np.count_nonzero(gap > -.007)))
        observed += matched
    return observed >= 6 and (not separated or current_contradiction)


def _measured_points(usable, center, rotation):
    observed = []
    for ref, live, _, views in usable:
        points = ref.points @ rotation.T + center
        seen = points[_unoccluded(points, views)]
        if len(seen):
            observed.append(live[cKDTree(seen).query(live)[0] < .012])
    return np.concatenate(observed) if observed else np.empty((0, 3))


def _fit_reference(tracker, clouds, camera_views):
    usable = []
    for ref, live, views in zip(tracker.reference_surfaces, clouds, camera_views):
        if ref.background is not None and len(live):
            live = live[ref.background.query(live)[0] > .005]
        if len(ref.points) >= 3 and len(live) >= 4:
            usable.append((ref, live, cKDTree(live), views))
    if not usable:
        return
    start = tracker.center0 + tracker.translation
    center, rotation = start.copy(), tracker.rotation.copy()
    canonical = np.concatenate([ref.points for ref in tracker.reference_surfaces])
    if _anchor_consistent(tracker, usable):
        center, rotation = tracker.anchor_center.copy(), tracker.anchor_rotation.copy()
        tracker.observed = _measured_points(usable, center, rotation)
        tracker.visible = len(tracker.observed) >= 6
        if tracker.visible:
            tracker.rotation, tracker.translation = rotation, center - tracker.center0
        return
    if len(_measured_points(usable, center, rotation)) < 4:
        return
    ray_center = _ray_translation(usable, center, rotation)
    ray_shift = np.linalg.norm(ray_center - center)
    center = ray_center.copy()
    has_depth = any(views for _, _, _, views in usable)
    accumulated_angle = np.zeros(3)
    for iteration in range(6):
        envelope = canonical @ rotation.T + center
        pairs = [_correspondences(ref, live, tree, center, rotation, view, envelope)
                 for ref, live, tree, view in usable]
        pairs = [pair for pair in pairs if pair is not None]
        if not pairs:
            return
        observed = np.concatenate([pair.observed for pair in pairs])
        if len(observed) >= 6 and all(pair.within_noise for pair in pairs):
            if iteration == 0 and ray_shift < .0001:
                # A featureless interior patch behind an aperture cannot
                # locate the body in its tangent plane. Keep the last pose.
                rows = np.concatenate([pair.translation_directions for pair in pairs])
                singular = np.linalg.svd(rows, compute_uv=False)
                tracker.observed = observed
                tracker.visible = singular[1] > .08 * singular[0]
                return
            # A ray proposal can slide along a partially visible surface.
            # Validate it against the differential constraints before acceptance.
            if iteration > 0:
                break
        jacobian = np.concatenate([pair.jacobian for pair in pairs])
        residual = np.concatenate([pair.residual for pair in pairs])
        weights = np.concatenate([pair.weights for pair in pairs])
        # Penalize changes in directions the current view cannot measure.
        regularizer = np.array([2., 2., 2., .10, .10, .05])
        ray_weight = np.r_[np.full(3, 15. if has_depth else 0.), np.zeros(3)]
        # Horizontal faces directly observe height, whereas an isolated vertical
        # wall leaves height ambiguous over its entire length.
        vertical_gain = np.average([pair.vertical_gain for pair in pairs],
                                   weights=[min(100, len(pair.residual)) for pair in pairs])
        height_weight = 2. + 48. * (1. - vertical_gain) ** 2
        anchor_weight = np.array([.6, .6, height_weight, .45, .45, .18])
        anchor_offset = np.r_[center - tracker.anchor_center,
                              mat_to_rotvec(rotation @ tracker.anchor_rotation.T)]
        jtw = jacobian.T * weights
        step = np.linalg.solve(jtw @ jacobian + np.diag(regularizer + anchor_weight + ray_weight),
                               jtw @ residual - regularizer * np.r_[center - start, accumulated_angle]
                               - anchor_weight * anchor_offset - ray_weight * np.r_[center - ray_center, np.zeros(3)])
        center += np.clip(step[:3], -.012, .012)
        angle = np.clip(step[3:], -.08, .08)
        rotation = rotvec_to_mat(angle) @ rotation
        accumulated_angle += angle
        if np.linalg.norm(step[:3]) < .00015 and np.linalg.norm(angle) < .003:
            break
    observed = _measured_points(usable, center, rotation)
    if len(observed) >= 6 and np.linalg.norm(center - start) <= .05:
        tracker.observed = observed
        tracker.visible = True
        tracker.rotation, tracker.translation = rotation, center - tracker.center0


def register_surfaces(tracker, clouds, camera_views):
    start = tracker.center0 + tracker.translation
    old_rotation = tracker.rotation.copy()
    _fit_reference(tracker, clouds, camera_views)
    if not tracker.visible or not any(camera_views):
        tracker.previous_depth_patches = ()
        return
    delta = tracker.center0 + tracker.translation - start
    previous = getattr(tracker, "previous_depth_patches", ())
    corrected = False
    length = np.linalg.norm(delta)
    if length > .004 and np.linalg.norm(start-tracker.anchor_center) > .0005:
        pairs = []
        for patch, live, views in zip(previous, clouds, camera_views):
            if patch is None or len(live) < 8 or not views:
                continue
            close = cKDTree(patch.points + start).query(live)[0] < .025
            if np.count_nonzero(close) >= 8:
                pairs.append((patch, live[close], None, views))
        if pairs:
            # Opposing faces can match the same old template under occlusion.
            # Rays from consecutive acquisitions independently constrain how
            # far the surface actually travelled during this interval.
            optical = _ray_translation(pairs, start, np.eye(3)) - start
            difference = delta - optical
            mismatch = np.linalg.norm(difference)
            if mismatch > .005:
                # The adjacent acquisition can limit travel along the fitted
                # direction. Its ambiguous rays cannot create a displacement
                # in a direction the reference surfaces did not support.
                supported = optical @ (delta / length)
                delta *= np.clip((supported + .004) / length, 0., 1.)
                tracker.translation = start - tracker.center0 + delta
                corrected = True
    center = tracker.center0 + tracker.translation
    surface = tracker._free_surface()
    patches, measured = [], []
    for live, views in zip(clouds, camera_views):
        seen = surface[_unoccluded(surface, views)] if views else surface[:0]
        matches = live[cKDTree(seen).query(live)[0] < .008] if len(seen) >= 6 else live[:0]
        if len(matches) >= 8:
            patches.append(DepthPatch(matches-center, views[0], center, np.eye(3)))
            measured.append(matches)
        else:
            patches.append(None)
    tracker.previous_depth_patches = tuple(patches)
    if corrected:
        if measured:
            tracker.observed = np.concatenate(measured)
        else:
            tracker.rotation, tracker.translation = old_rotation, start-tracker.center0
            tracker.visible = False
            tracker.observed = surface[:0]
