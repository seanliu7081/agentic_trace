"""Register a handoff target cloud against subsequent *unlabelled* depth.

All points are metric and share the fixed world/reference axes. An initial
target/background segmentation is the sole object-identification input.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.spatial import cKDTree

from .geometry import mat_to_rotvec, rotvec_to_mat


def _sample_surface(points, count, rng, scale=.004):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(points):
        return points
    _, index = np.unique(np.floor(points / scale).astype(np.int32), axis=0, return_index=True)
    if len(index) > count:
        index = rng.choice(index, size=count, replace=False)
    return points[index]


def _estimate_normals(points, center):
    """Outward unit surface normals via local PCA on the sampled template."""
    n = len(points)
    if n < 4:
        out = points - center
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.where(norms > 1e-9, out / np.maximum(norms, 1e-9), np.array([[0., 0., 1.]]))
    k = min(12, n)
    _, idx = cKDTree(points).query(points, k=k)
    nbrs = points[idx] - points[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nbrs, nbrs) + 1e-9 * np.eye(3)
    _, vecs = np.linalg.eigh(cov)
    normals = vecs[:, :, 0]
    flip = np.sum(normals * (points - center), axis=1, keepdims=True) < 0.
    return np.where(flip, -normals, normals)


def _detect_support_z(target, background, center):
    """Detect a static horizontal support plane occluded beneath the t=0 target footprint."""
    if len(target) < 8 or len(background) < 30:
        return None
    z_min, z_max = float(target[:, 2].min()), float(target[:, 2].max())
    if z_max - z_min < .008:
        return None
    xy_dist = np.linalg.norm(background[:, :2] - center[:2], axis=1)
    near = background[(xy_dist < .28) & (background[:, 2] >= z_min - .020) & (background[:, 2] <= z_min + .015)]
    if len(near) < 25:
        return None
    z_vals = near[:, 2]
    bins = np.arange(z_vals.min() - .002, z_vals.max() + .005, .003)
    if len(bins) < 2:
        return float(np.median(z_vals)) if z_max - float(np.median(z_vals)) >= .008 else None
    hist, edges = np.histogram(z_vals, bins=bins)
    peak_z = .5 * (edges[int(np.argmax(hist))] + edges[int(np.argmax(hist)) + 1])
    inliers = z_vals[np.abs(z_vals - peak_z) <= .004]
    if len(inliers) < 25:
        return None
    sup_z = float(np.median(inliers))
    return sup_z if z_max - sup_z >= .008 else None


class _LegacyDepthTracker:
    """1-DoF joint registration (with screw observability) or SE(2)+Z ICP + rigid FK carry for free objects."""
    def __init__(self, target, background, origin, axis, hinge, rng, *, count=200, scale=.004):
        self.origin = np.asarray(origin, dtype=np.float64).copy()
        self.axis = np.asarray(axis, dtype=np.float64).copy()
        self.hinge = bool(hinge)
        self.model = _sample_surface(target, count, rng, scale)
        self.center0 = self.model.mean(axis=0) if len(self.model) else np.zeros(3)
        self._local_model = self.model - self.center0 if len(self.model) else np.empty((0, 3))
        self.normals0 = _estimate_normals(self.model, self.center0) if len(self.model) else np.empty((0, 3))
        self.center_in_grip = None
        self.rot_in_grip = None
        self.grip_R0 = None
        self.grip_val0 = 0.
        bg_arr = np.asarray(background, dtype=np.float64).reshape(-1, 3)
        self.support_z = None if (self.hinge or self.axis.any()) else _detect_support_z(self.model, bg_arr, self.center0)
        bg_sample = _sample_surface(bg_arr, 7000, rng, scale=.005)
        self.background = cKDTree(bg_sample if len(bg_sample) else np.full((1, 3), 1e6))
        self.value = 0.
        self.translation = np.zeros(3)
        self.rotation = np.eye(3)
        self.visible = len(self.model) >= 8
        self.observed = self.model.copy()
        self._centered = self.model - self.origin if len(self.model) else np.empty((0, 3))
        parallel = (self._centered @ self.axis)[:, None] * self.axis if len(self.model) else np.empty((0, 3))
        self._arm = self._centered - parallel
        self._cross = np.cross(self.axis, self._arm) if len(self.model) else np.empty((0, 3))
        self._base = self.origin + parallel if len(self.model) else np.empty((0, 3))
        distance_bg = self.background.query(self.model)[0] if len(self.model) else np.empty(0)
        self.weights = .2 + np.minimum(distance_bg / .025, 1.)
        if self.hinge and len(self.weights):
            # Screw-normal observability |n_i . (axis x (x_i - origin))|^2: zeroes out
            # rotationally invariant cylinder barrels on knobs while keeping ridges/levers/panels.
            obs_gain = np.abs(np.sum(self.normals0 * self._cross, axis=1))
            if obs_gain.max() > 1e-4:
                self.weights *= (obs_gain / max(float(obs_gain.max()), .008)) ** 2 + 1e-3
            else:
                r = np.linalg.norm(self._arm, axis=1)
                self.weights *= .2 + r / max(float(r.max()), .015)
        elif self.axis.any() and len(self.weights):
            # Prismatic screw observability |n_i . axis|^2: only faces with normal component along
            # the slide axis carry displacement signal (prevents drawer side walls matching carcass).
            slide_gain = np.abs(self.normals0 @ self.axis)
            if slide_gain.max() > .15:
                self.weights *= (slide_gain / float(slide_gain.max())) ** 2 + 1e-3

    def __deepcopy__(self, memo):
        other = copy.copy(self)
        memo[id(self)] = other
        other.translation = self.translation.copy()
        other.rotation = self.rotation.copy()
        other.observed = self.observed.copy()
        other.center_in_grip = None if self.center_in_grip is None else self.center_in_grip.copy()
        other.rot_in_grip = None if self.rot_in_grip is None else self.rot_in_grip.copy()
        other.grip_R0 = None if self.grip_R0 is None else self.grip_R0.copy()
        return other

    def record_grip(self, carrying, rotation, position):
        if not carrying:
            self.center_in_grip = None
            self.rot_in_grip = None
            self.grip_R0 = None
        elif self.center_in_grip is None and (self.visible or (not getattr(self, "legacy", True) and len(self.observed) >= 6)):
            R = np.asarray(rotation, dtype=np.float64)
            p = np.asarray(position, dtype=np.float64)
            self.center_in_grip = R.T @ (self.center0 + self.translation - p)
            self.rot_in_grip = R.T @ self.rotation
            self.grip_R0 = R.copy()
            self.grip_val0 = float(self.value)

    def carried_translation(self, rotation, position):
        if self.center_in_grip is not None:
            return np.asarray(position) + np.asarray(rotation) @ self.center_in_grip - self.center0
        return None

    def carried_pose(self, rotation, position):
        if self.center_in_grip is not None and self.rot_in_grip is not None:
            R = np.asarray(rotation, dtype=np.float64)
            p = np.asarray(position, dtype=np.float64)
            return R @ self.rot_in_grip, p + R @ self.center_in_grip - self.center0
        return None

    def carried_angle(self, rotation):
        if self.grip_R0 is not None and self.hinge:
            drot = mat_to_rotvec(np.asarray(rotation, dtype=np.float64) @ self.grip_R0.T)
            return float(self.grip_val0 + drot @ self.axis)
        return None

    def _free_surface(self, rot=None, trans=None):
        R = self.rotation if rot is None else rot
        t = self.translation if trans is None else trans
        return self.center0 + t + self._local_model @ R.T

    def _model_at(self, values):
        values = np.asarray(values)[:, None, None]
        if self.hinge:
            return self._base + np.cos(values) * self._arm + np.sin(values) * self._cross
        return self.model + values * self.axis

    def transform(self, point):
        p = np.asarray(point, dtype=np.float64)
        if self.hinge:
            return self.origin + rotvec_to_mat(self.axis * self.value) @ (p - self.origin)
        if self.axis.any():
            return p + self.axis * self.value
        return self.center0 + self.translation + self.rotation @ (p - self.center0)

    def _filter_foreground(self, points, expected, pad=.042, free=False):
        lo, hi = expected.min(axis=0) - pad, expected.max(axis=0) + pad
        local = points[np.all((points >= lo) & (points <= hi), axis=1)]
        if len(local) < 6:
            return np.empty((0, 3))
        local = local[self.background.query(local)[0] > (.0055 if free else .006)]
        if free and self.support_z is not None and len(local):
            local = local[np.abs(local[:, 2] - self.support_z) > .0045]
        return local

    def update(self, points, rng, free=False, grasp_prediction=None, angle_prior=None):
        self.visible = False
        self.observed = np.empty((0, 3))
        has_carry_prior = free and grasp_prediction is not None
        if has_carry_prior:
            if isinstance(grasp_prediction, tuple):
                self.rotation = np.asarray(grasp_prediction[0], dtype=np.float64).copy()
                self.translation = np.asarray(grasp_prediction[1], dtype=np.float64).copy()
            else:
                self.translation = np.asarray(grasp_prediction, dtype=np.float64).copy()
        elif not free and angle_prior is not None:
            # Sensorless wrist-encoder prior when holding a small occluded rotary knob/lever.
            self.value = float(angle_prior)

        if len(self.model) < 8 or not len(points):
            return

        if has_carry_prior:
            # Rigidly clamped payload: preserve exact FK pose without partial-view depth bias,
            # while still verifying camera visibility honestly.
            predicted = self._free_surface()
            local = _sample_surface(self._filter_foreground(points, predicted, pad=.035, free=True), 1600, rng)
            if len(local) >= 6:
                associated = cKDTree(predicted).query(local)[0] < .014
                self.observed = local[associated]
                self.visible = len(self.observed) >= 6
            return

        if free:
            expected = self._free_surface()
            local = _sample_surface(self._filter_foreground(points, expected, pad=.045, free=True), 1600, rng)
            if len(local) < 8:
                return
            R_cur = self.rotation.copy()
            t_cur = self.translation.copy()
            R0_step, t0_step = R_cur.copy(), t_cur.copy()
            yaw_accum = 0.
            tree_local = cKDTree(local)
            reg_diag = np.array([1.2, 1.2, 2.5, .015])
            for _ in range(6):
                X = self._free_surface(R_cur, t_cur)
                N = self.normals0 @ R_cur.T
                c_cur = self.center0 + t_cur

                # 1. Data-to-Model Point-to-Plane: occluded template faces generate no live points,
                # eliminating the single-view front-surface pull bias.
                d_dm, idx_dm = cKDTree(X).query(local)
                keep_dm = d_dm < .026
                if keep_dm.sum() < 8:
                    return
                Y_dm = local[keep_dm]
                X_dm = X[idx_dm[keep_dm]]
                N_dm = N[idx_dm[keep_dm]]
                res_dm = np.sum(N_dm * (Y_dm - X_dm), axis=1)
                rx_dm = X_dm - c_cur
                J_dm = np.column_stack((N_dm, rx_dm[:, 0] * N_dm[:, 1] - rx_dm[:, 1] * N_dm[:, 0]))
                w_dm = 1. / np.maximum(1., np.abs(res_dm) / .006)

                # 2. Visible-only Model-to-Data Point-to-Plane + Tangential Boundary Anchor
                # (locks in-plane XY translation on flat plates whose rim edge pixels were dropped!).
                d_md, idx_md = tree_local.query(X)
                keep_md = d_md < .016
                if keep_md.sum() >= 6:
                    Y_md = local[idx_md[keep_md]]
                    X_md = X[keep_md]
                    N_md = N[keep_md]
                    res_md = np.sum(N_md * (Y_md - X_md), axis=1)
                    rx_md = X_md - c_cur
                    J_md = np.column_stack((N_md, rx_md[:, 0] * N_md[:, 1] - rx_md[:, 1] * N_md[:, 0]))
                    w_md = .6 / np.maximum(1., np.abs(res_md) / .006)
                    J_list, r_list, w_list = [J_dm, J_md], [res_dm, res_md], [w_dm, w_md]
                else:
                    J_list, r_list, w_list = [J_dm], [res_dm], [w_dm]

                # Visibility-matched centroid + bidirectional XY lock: compares live foreground XY mean
                # against ONLY the visible subset of model points (`d_md < 2.2 cm`), giving exact 1:1
                # tracking on flat rimless plates AND zero front-surface bias inside dark cavities!
                close_dm = d_dm < .022
                vis_md = d_md < .022
                if close_dm.sum() >= 10 and vis_md.sum() >= 6:
                    inlier_pts = local[close_dm]
                    vis_shift = inlier_pts[:, :2].mean(axis=0) - X[vis_md, :2].mean(axis=0)
                    nn_dm = (inlier_pts[:, :2] - X[idx_dm[close_dm], :2]).mean(axis=0)
                    nn_md = (local[idx_md[vis_md], :2] - X[vis_md, :2]).mean(axis=0)
                    xy_shift = np.clip(.65 * vis_shift + .35 * (.5 * (nn_dm + nn_md)), -.012, .012)
                    J_xy = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
                    w_xy = np.full(2, .40 * int(close_dm.sum()))
                    J_list.append(J_xy)
                    r_list.append(xy_shift)
                    w_list.append(w_xy)

                J_all = np.vstack(J_list)
                r_all = np.concatenate(r_list)
                w_all = np.concatenate(w_list)

                step_so_far = np.r_[t_cur - t0_step, yaw_accum]
                JW = J_all.T * w_all
                H = JW @ J_all + np.diag(reg_diag)
                g = JW @ r_all - reg_diag * step_so_far
                delta = np.linalg.solve(H, g)
                dt = np.clip(delta[:3], -.009, .009)
                dyaw = float(np.clip(delta[3], -.08, .08))
                t_next = t_cur + dt
                tot_disp = t_next - t0_step
                tot_n = float(np.linalg.norm(tot_disp))
                if tot_n > .022:
                    t_next = t0_step + tot_disp * (.022 / tot_n)
                t_cur = t_next
                yaw_accum += dyaw
                R_cur = rotvec_to_mat(np.array([0., 0., yaw_accum])) @ R0_step
                if np.linalg.norm(dt) < 1e-4 and abs(dyaw) < 1e-4:
                    break

            predicted = self._free_surface(R_cur, t_cur)
            associated = cKDTree(predicted).query(local)[0] < .014
            self.observed = local[associated]
            self.visible = len(self.observed) >= 6
            if self.visible:
                self.translation = t_cur
                self.rotation = R_cur
            return

        # 1-DoF joint registration (prismatic slide or revolute hinge).
        center_val = float(self.value if angle_prior is None else angle_prior)
        expected = self._model_at([center_val])[0]
        local = _sample_surface(self._filter_foreground(points, expected, pad=.045, free=False), 1600, rng)
        if len(local) < 8:
            return
        if self.hinge and angle_prior is None and len(self._cross):
            obs_gain = np.abs(np.sum(self.normals0 * self._cross, axis=1))
            if float(obs_gain.max()) < .012:
                # Small or nearly axisymmetric rotary knob without active grip prior:
                # update visibility honestly so GripStateObserver engages, while holding last angle!
                associated = cKDTree(expected).query(local)[0] < .014
                self.observed = local[associated]
                self.visible = len(self.observed) >= 6
                return
        tree = cKDTree(local)
        span = (.12 if angle_prior is not None else .065) if self.hinge else .024
        values = center_val + np.linspace(-span, span, 21)
        models = self._model_at(values)
        distances = tree.query(models.reshape(-1, 3))[0].reshape(len(values), -1)
        costs = np.average(np.minimum(distances, .022), axis=1, weights=self.weights)
        costs += .00012 * np.abs(values - center_val) / span
        best = int(np.argmin(costs))
        if (distances[best] < .013).sum() < max(6, len(self.model) * .16):
            return
        value = values[best]
        if 0 < best < len(values) - 1:
            curvature = costs[best - 1] - 2 * costs[best] + costs[best + 1]
            if curvature > 1e-9:
                value += (values[1] - values[0]) * np.clip(.5 * (costs[best - 1] - costs[best + 1]) / curvature, -.5, .5)
        predicted = self._model_at([value])[0]
        associated = cKDTree(predicted).query(local)[0] < .012
        self.observed = local[associated]
        self.visible = len(self.observed) >= 6
        if self.visible:
            self.value = float(value)


def _unoccluded(points, views):
    """Project world points onto the measured depth buffers (positive camera z)."""
    if views is None:
        return np.ones(len(points), dtype=bool)
    seen = np.zeros(len(points), dtype=bool)
    for depth, K, pose in views:
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        use = np.flatnonzero(z > .02)
        xy = camera[use, :2] / z[use, None]
        pixels = np.floor(xy * (K[0, 0], K[1, 1]) + (K[0, 2], K[1, 2])).astype(int)
        u, v = pixels.T
        inside = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        measured = depth[v[inside], u[inside]]
        indices = use[inside]
        seen[indices] |= np.isfinite(measured) & (measured >= z[indices] - .008)
    return seen


def _joint_normals(points, normals, screw, axis, axial_slices=False):
    """Normals of a measured sheet or the normal plane of a thin observed edge."""
    if len(points) < 6:
        return normals
    # When two adjacent sheets are at different depths along a rotation axis,
    # their heights are invariant to motion. Avoid smoothing their edges together.
    coords = (np.column_stack((points, 5*(points @ axis))) if axial_slices else points)
    _, idx = cKDTree(coords).query(coords, k=min(16, len(points)))
    local = points[idx] - points[idx].mean(axis=1, keepdims=True)
    if axial_slices:
        local -= (local @ axis)[:, :, None]*axis
    eigen, axes = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
    tangent = axes[:, :, -1]
    projected = screw - tangent*(tangent*screw).sum(axis=1, keepdims=True)
    magnitude = np.linalg.norm(projected, axis=1)
    edge = (eigen[:, 1] < .12*eigen[:, 2]) & (magnitude > 1e-7)
    normals = normals.copy()
    normals[edge] = projected[edge] / magnitude[edge, None]
    return normals


def _mount_plane(target, background, normal):
    """A local static plane perpendicular to the axis, underneath a rotary part."""
    if len(target) < 6 or len(background) < 20:
        return None
    center = np.median(target, axis=0)
    rel = background - center
    axial = rel @ normal
    radii = np.linalg.norm(rel - axial[:, None] * normal, axis=1)
    edge = max(.035, np.max(np.linalg.norm(target - center, axis=1)))
    z = axial[(radii < edge + .07) & (np.abs(axial) < edge + .025)]
    if len(z) < 20:
        return None
    bins = np.arange(z.min() - .0015, z.max() + .0045, .003)
    counts, edges = np.histogram(z, bins=bins)
    k = int(counts.argmax())
    band = z[np.abs(z - .5 * (edges[k] + edges[k + 1])) < .004]
    if len(band) < 20:
        return None
    height = float(np.median(band) + center @ normal)
    # A plane running through the middle of the part is not its mounting surface.
    signed = target @ normal - height
    if np.quantile(signed, .2) < -.005 and np.quantile(signed, .8) > .005:
        return None
    return height


def _support_plane(points, center, height):
    """Fit the locally dominant support, ignoring the region occupied by the target."""
    points = np.asarray(points)
    rel = points - center
    radius = np.linalg.norm(rel[:, :2], axis=1)
    near = points[(radius < .35) & (radius > .09) & (np.abs(points[:, 2] - height) < .042)]
    if len(near) < 25:
        return None
    # Bin a slightly tilted plane, then iteratively reselect its inliers.
    hist, edges = np.histogram(near[:, 2], bins=np.arange(height - .045, height + .048, .003))
    k = int(hist.argmax())
    plane = np.array([0., 0., (edges[k] + edges[k+1]) / 2])
    xy1 = np.column_stack((near[:, :2], np.ones(len(near))))
    for _ in range(3):
        keep = np.abs(near[:, 2] - xy1 @ plane) < .006
        if keep.sum() < 20:
            return None
        plane = np.linalg.lstsq(xy1[keep], near[keep, 2], rcond=None)[0]
    return plane if np.linalg.norm(plane[:2]) < .15 else None


def _depth_pixels(points, view):
    depth, K, pose = view
    pc = (points-pose[:3,3])@pose[:3,:3]
    z = pc[:,2]
    uv = np.floor(pc[:,:2]/np.maximum(z[:,None],.001) *
                  np.array([K[0,0],K[1,1]]) + [K[0,2],K[1,2]]).astype(int)
    valid = ((z>.02) & (uv[:,0]>=0) & (uv[:,0]<depth.shape[1]) &
             (uv[:,1]>=0) & (uv[:,1]<depth.shape[0]))
    return uv, z, valid


def _static_depth(points, view):
    """Fixed camera background in pixel space retains samples omitted by voxelization."""
    from scipy.ndimage import distance_transform_edt
    depth = np.full_like(view[0], np.inf)
    uv, z, valid = _depth_pixels(points, view)
    u,v = uv[valid].T
    np.minimum.at(depth,(v,u),z[valid])
    if np.isfinite(depth).any():
        dist, idx = distance_transform_edt(~np.isfinite(depth), return_indices=True)
        missing = (dist > 0.) & (dist <= 1.5)
        depth[missing] = depth[idx[0][missing],idx[1][missing]]
    return depth


def _above_plane(cloud, plane):
    return cloud[:, 2] - cloud[:, :2] @ plane[:2] - plane[2]



class ReferenceDepthTracker(_LegacyDepthTracker):
    """Register the handoff-only template using visible geometry and known joint kinematics."""

    def __init__(self, target, background, origin, axis, hinge, rng, legacy=False,
                 camera_targets=None, camera_backgrounds=None, views=None):
        self.legacy = legacy
        if legacy:
            super().__init__(target, background, origin, axis, hinge, rng)
            return
        super().__init__(target, background, origin, axis, hinge, rng, count=900, scale=.0025)
        bg = np.asarray(background).reshape(-1, 3)
        self.mount_plane = None
        self.point_obs_gain = np.zeros(len(self.model))
        self.gain_threshold = .30
        if self.axis.any():
            self.mount_plane = _mount_plane(self.model, bg, self.axis)
            screw = self._cross if self.hinge else np.broadcast_to(self.axis, self.model.shape)
            self.normals0 = _joint_normals(self.model, self.normals0, screw, self.axis)
            if self.hinge:
                self.normals0 = _joint_normals(self.model, self.normals0, screw, self.axis, axial_slices=True)
            if self.hinge:
                self.gain_threshold = max(.008, .30 * np.linalg.norm(self._arm, axis=1).max(initial=0.))
                from .gcp_joint import visible_normals
                self.normals0, resolved = visible_normals(
                    self.model, self._cross, self.normals0, self.axis, self.origin,
                    camera_targets, views, self.gain_threshold)
                self.point_obs_gain = np.abs(np.einsum("ij,ij->i", self.normals0, self._cross))
                self.point_resolved = resolved
            else:
                self.point_obs_gain = np.abs(self.normals0 @ self.axis)
            self.visible = np.count_nonzero((self.point_obs_gain > self.gain_threshold)
                                           & getattr(self, "point_resolved", True)) >= 6
            return

        from .gcp_release import initial_surface
        if self.support_z is None:
            self.support_z = _mount_plane(self.model, bg, np.array([0., 0., 1.]))
        targets = camera_targets if camera_targets is not None else [target]
        backgrounds = camera_backgrounds if camera_backgrounds is not None else [bg]
        self.support_planes = []
        self.camera_background = []
        self.fixed_depth = []
        self.reference_surfaces = []
        self.anchor_rotation = np.eye(3)
        self.anchor_center = self.center0.copy()
        for i, (surface, background) in enumerate(zip(targets, backgrounds)):
            surface = np.asarray(surface, dtype=float).reshape(-1, 3)
            background = np.asarray(background, dtype=float).reshape(-1, 3)
            plane = None
            if len(surface) >= 3:
                plane = _support_plane(background, surface.mean(axis=0), np.quantile(surface[:, 2], .05))
                # A shelf cutting through the observed target is a neighboring
                # obstacle. Only remove a plane that supports the target from below.
                if plane is not None and np.quantile(_above_plane(surface, plane), .10) < -.005:
                    plane = None
            self.support_planes.append(plane)
            self.camera_background.append(cKDTree(background))
            view = views[i] if views else None
            self.fixed_depth.append((_static_depth(background, view), view[2].copy()) if view else None)
            self.reference_surfaces.append(initial_surface(surface, self.center0, plane, rng, view=view))
        self.reference_surfaces = tuple(self.reference_surfaces)

    @classmethod
    def from_older_snapshot(cls, old):
        """Recover a pre-z-buffer snapshot from its saved handoff geometry."""
        tracker = cls(old.model, old.background.data, old.origin, old.axis, old.hinge,
                      np.random.default_rng(0))
        tracker.value = tracker._progress_value = old.value
        tracker.rotation = old.rotation.copy()
        tracker.translation = old.translation.copy()
        if not old.axis.any():
            # Voxel sampling may move the centre about which rotation is represented.
            tracker.translation = old.transform(tracker.center0) - tracker.center0
        for name in ("observed", "visible", "center_in_grip", "rot_in_grip", "grip_R0", "grip_val0"):
            setattr(tracker, name, copy.deepcopy(getattr(old, name)))
        if tracker.center_in_grip is not None and tracker.rot_in_grip is not None:
            tracker.center_in_grip += tracker.rot_in_grip @ (tracker.center0 - old.center0)
        return tracker

    @property
    def progress_value(self):
        """Actor joint progress, including encoder information when depth is unobservable."""
        return getattr(self, "_progress_value", self.value)

    def transform(self, point):
        if not getattr(self, "legacy", True) and self.axis.any():
            p = np.asarray(point, dtype=float)
            if self.hinge:
                return self.origin + rotvec_to_mat(self.axis * self.progress_value) @ (p - self.origin)
            return p + self.axis * self.progress_value
        return super().transform(point)

    def record_grip(self, carrying, rotation, position, *, opening=False):
        newly_grasped = carrying and self.center_in_grip is None
        if not getattr(self, "legacy", True) and self.axis.any():
            if carrying:
                self.released_joint = False
            elif self.center_in_grip is not None:
                self.released_joint = bool(opening)
        super().record_grip(carrying, rotation, position)
        if not getattr(self, "legacy", True) and self.axis.any() and self.grip_R0 is not None:
            if newly_grasped or self.visible:
                self.grip_R0 = np.array(rotation, copy=True)
                self.grip_p0 = np.array(position, copy=True)
                self.grip_val0 = self.progress_value

    def carried_joint(self, rotation, position):
        """A pure rigid-grasp prediction in the reference joint coordinate."""
        if self.grip_R0 is None:
            return None
        if self.hinge:
            return self.carried_angle(rotation)
        if self.axis.any() and getattr(self, "grip_p0", None) is not None:
            return float(self.grip_val0 + self.axis @ (position - self.grip_p0))
        return None

    def _foreground(self, points, expected):
        local = self._filter_foreground(points, expected, pad=.045, free=not self.axis.any())
        if self.mount_plane is not None and len(local):
            on_mount = np.abs(local @ self.axis - self.mount_plane) < .0045
            # A translating front can become flush with its surround.
            # Keep the portion of the plane inside the predicted moving face.
            adjacent = expected[np.abs(expected @ self.axis - self.mount_plane) < .011]
            if len(adjacent) and on_mount.any():
                on_mount[on_mount] = cKDTree(adjacent).query(local[on_mount])[0] > .018
            local = local[~on_mount]
        return local

    def update(self, points, rng, free=False, grasp_prediction=None, angle_prior=None,
               views=None, camera_points=None):
        if getattr(self, "legacy", True):
            return super().update(points, rng, free, grasp_prediction, angle_prior)
        if self.center_in_grip is None:
            # A detached or stale caller prediction cannot move a free body
            # or a joint. Only the encoder transform of a registered grip
            # provides a motion constraint while a surface is hidden.
            grasp_prediction = angle_prior = None
        self.visible = False
        self.observed = np.empty((0, 3))
        if self.axis.any():
            center = self.progress_value if angle_prior is None else angle_prior
            expected = self._model_at(np.unique([center, self.progress_value])).reshape(-1, 3)
            local = (self._foreground(points, expected) if len(expected) else np.empty((0, 3)))
            local = _sample_surface(local, 2000, rng, scale=.0025)
            self._register_joint(local, views, angle_prior, camera_points)
            return
        if grasp_prediction is not None:
            if isinstance(grasp_prediction, tuple):
                self.rotation = np.asarray(grasp_prediction[0]).copy()
                self.translation = np.asarray(grasp_prediction[1]).copy()
            else:
                self.translation = np.asarray(grasp_prediction).copy()
        if not len(self.model):
            return
        expected = self._free_surface()
        clouds = camera_points if camera_points is not None else [points]
        camera_views = [[v] for v in views] if camera_points is not None and views else [views] * len(clouds)
        # Snapshots created before per-camera handoff surfaces were recorded
        # can still register the aggregate template against both cameras.
        if len(clouds) != len(self.reference_surfaces):
            clouds, camera_views = [points], [views]
        local = [self._free_foreground(cloud, expected, i, view, rng)
                 for i, (cloud, view) in enumerate(zip(clouds, camera_views))]
        self._register_free(local, camera_views, grasp_prediction is not None, rng)

    def _free_foreground(self, cloud, expected, i, views, rng):
        pad = .050
        local = cloud[np.all((cloud >= expected.min(axis=0) - pad) &
                             (cloud <= expected.max(axis=0) + pad), axis=1)]
        plane = self.support_planes[i]
        if plane is not None:
            local = local[_above_plane(local, plane) > .005]
        elif self.support_z is not None:
            local = local[np.abs(local[:, 2] - self.support_z) > .005]
        fixed = self.fixed_depth[i]
        if fixed is not None and views and np.allclose(views[0][2], fixed[1]):
            uv, z, valid = _depth_pixels(local, views[0])
            u, v = uv[valid].T
            keep = valid.copy()
            keep[valid] = z[valid] < fixed[0][v, u] - .006
            local = local[keep]
        if len(local):
            local = local[self.camera_background[i].query(local)[0] > .0045]
        return _sample_surface(local, 1200, rng, scale=.002)

    def _associate(self, local, predicted, views):
        seen = _unoccluded(predicted, views)
        if seen.sum() >= 6:
            self.observed = local[cKDTree(predicted[seen]).query(local)[0] < .012]
            self.visible = len(self.observed) >= 6

    def _register_free(self, clouds, camera_views, carrying, rng):
        from .gcp_release import register_surfaces, refresh_surfaces
        if carrying:
            refresh_surfaces(self, clouds, camera_views, rng)
        else:
            register_surfaces(self, clouds, camera_views)

    def _register_joint(self, local, views, angle_prior, camera_points=None):
        """Register the measured surfaces along their known unit screw."""
        previous_innovation = getattr(self, "pending_innovation", None)
        self.pending_innovation = None
        previous = self.progress_value
        center = previous if angle_prior is None else float(angle_prior)
        self._progress_value = center
        self.visible = False
        if len(local) < 6 or len(self.model) < 6:
            return
        predicted = self._model_at([center])[0]
        self._associate(local, predicted, views)
        self.visible = False  # Observing the contact patch need not reveal joint travel.
        high = (self.point_obs_gain > self.gain_threshold) & getattr(self, "point_resolved", True)
        previous_model = self._model_at([previous])[0]
        seen = high & (_unoccluded(predicted, views) | _unoccluded(previous_model, views))
        if seen.sum() < 6:
            return
        # Search around the measured wrist prediction and include the previous
        # pose, so a moving wrist cannot drag a stationary visible surface.
        span = .12 if self.hinge else .020
        values = np.unique(np.r_[center + np.linspace(-span, span, 25), previous])
        models = self._model_at(values)[:, seen]
        normals = self.normals0[seen]
        if self.hinge:
            normals = np.stack([normals @ rotvec_to_mat(v*self.axis).T for v in values])
        def coordinates(cloud):
            if self.hinge:
                # Axial height and radius are invariant, including when a
                # shallow raised surface moves along an adjacent lower one.
                arm = cloud - self.origin
                axial = arm @ self.axis
                radius = np.linalg.norm(arm - axial[:, None]*self.axis, axis=1)
                return np.column_stack((cloud, 2.5*axial, 1.2*radius))
            return cloud
        tree = cKDTree(coordinates(local))
        indices = tree.query(coordinates(models.reshape(-1, 3)))[1].reshape(models.shape[:2])
        distances = np.linalg.norm(local[indices]-models, axis=2)
        # Use the same supported template elements in every hypothesis to avoid
        # rewarding a candidate for losing correspondences behind the hand.
        centers = np.array([np.argmin(np.abs(values-v)) for v in (center, previous)])
        supported = distances[centers].min(axis=0) < .014
        if supported.sum() < 6:
            return
        offsets = local[indices] - models
        normal_residual = np.abs(np.sum(offsets * normals, axis=2))
        weights = self.point_obs_gain[seen][supported] ** 2
        residual = .75*normal_residual + .25*distances
        costs = np.average(np.minimum(residual[:, supported], .013), axis=1, weights=weights)
        # Depth quantization makes differences below a fraction of a
        # millimetre inconclusive. Prefer continuity within this noise band.
        costs += .00045 * np.minimum(np.abs(values-center) / span, 1.)
        minimum = int(costs.argmin())
        best = minimum
        if views:
            # Several depths from a single surface share quantization and
            # calibration errors. Six neighbouring returns can all prefer
            # the same spurious edge by a fraction of a millimetre.
            near = costs <= costs[best] + .00040
            if np.ptp(values[near]) > .85*span:
                return
            closest = np.flatnonzero(near)[np.argmin(np.abs(values[near]-center))]
            best = int(closest)
        value = values[best]
        if best == minimum and 0 < best < len(values)-1:
            left, right = values[best] - values[best-1], values[best+1] - values[best]
            curvature = costs[best-1] - 2*costs[best] + costs[best+1]
            if curvature > 1e-10 and np.isclose(left, right):
                value += left * np.clip(.5*(costs[best-1]-costs[best+1])/curvature, -.5, .5)
        predicted = self._model_at([value])[0]
        index = tree.query(coordinates(predicted))[1]
        distance = np.linalg.norm(local[index] - predicted, axis=1)
        matched = high & _unoccluded(predicted, views) & (distance < .012)
        if len(np.unique(index[matched])) < 6:
            return
        self._associate(local, predicted, views)
        if views:
            # Convert the innovation to normal motion of the *observed*
            # surface. A several-degree correction on a short moment arm can
            # be smaller than a single depth sample's metric uncertainty.
            # Give such a measurement fractional weight instead of storing
            # the entire pixel error as an encoder offset on the next tick.
            gain = np.sqrt(np.mean(self.point_obs_gain[matched]**2))
            resolved_motion = abs(value-center)*gain
            fraction = np.clip(resolved_motion/.003, .15, 1.)
            value = center + fraction*(value-center)
            if angle_prior is None and getattr(self, "released_joint", False):
                # Newly exposed surfaces can produce a single confident but
                # incorrect correspondence as the opening pads leave the view.
                # Confirm metric motion on two consecutive depth acquisitions.
                delta = value - previous
                if abs(delta)*gain > .00045:
                    self.pending_innovation = delta
                    if (previous_innovation is None or previous_innovation*delta <= 0.
                            or abs(delta-previous_innovation)*gain > .004):
                        self.visible = False
                        return
        self.value = self._progress_value = float(value)
        self.visible = True
