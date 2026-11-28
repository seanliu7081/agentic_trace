"""Sensorless general contact runtime. Poses map local column vectors to world.

The actor contract contains no simulator state or object identity. In Rev 3 the
listed scalar dimensions add up to 17 and 39. To retain the specified 16 / 36
wire format, the unit goal axis is octahedrally encoded in two coordinates and
the attempt-relative orientation is the principal three-coordinate rotation
vector (the usual axis ambiguity at pi). All distances below are metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from contact_contracts import CONTACT_GOALS

from .geometry import frame_from_axes, mat_to_rotvec, rotvec_to_mat


FAMILIES = tuple(CONTACT_GOALS)
OBS_SHAPES = {"scene_pts": (384, 5), "payload_pts": (128, 4), "goal": (16,), "history": (12, 36)}
ACTION_DIM = 8
CONTROL_DT = 0.05


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros_like(v)


def _cap(v, limit):
    return v * min(1.0, limit / max(float(np.linalg.norm(v)), 1e-12))


def encode_axis(axis):
    """Fold the two unit hemispheres onto a square. (2,2) means no axis."""
    v = np.asarray(axis, dtype=float)
    n = np.abs(v).sum()
    if n < 1e-10:
        return np.array([2.0, 2.0])
    x, y, z = v / n
    if z < 0:
        return np.copysign([1 - abs(y), 1 - abs(x)], [x, y])
    return np.array([x, y])


def decode_axis(uv):
    xy = np.asarray(uv, dtype=float)
    if np.array_equal(xy, [2, 2]):
        return np.zeros(3)
    z = 1.0 - np.abs(xy).sum()
    if z < 0:
        xy = np.copysign(1 - np.abs(xy[::-1]), xy)
    return _unit(np.r_[xy, z])


@dataclass
class ContactContract:
    """Signed travel and region geometry are defined in the static reference.

    Joint goal values and measured progress must use the same origin, e.g.
    both angles from closed, or both travel since *stage* entry. Changing grip
    or attempt never changes that origin. Hint and region are distinct points
    in the reference frame. An optional box models a destination with extent.
    """
    family: str
    target_entity: str
    reference_entity: str
    goal_dir_ref: tuple = (0., 0., 0.)
    goal_axis_ref: tuple = (0., 0., 0.)
    target_trans_m: float = 0.
    target_rot_rad: float = 0.
    goal_region_ref: tuple | None = None
    p_contact_hint: tuple | None = None
    want_held_end: int | None = None
    trans_tolerance: float = .01
    rot_tolerance: float = np.deg2rad(5.)
    region_half_extents: tuple = (0., 0., 0.)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"Unsupported contact family {self.family!r}")
        if not self.target_entity or not self.reference_entity or self.target_entity == self.reference_entity:
            raise ValueError("Expected distinct target and fixed reference entities")
        defaults = CONTACT_GOALS[self.family]
        if self.want_held_end is None:
            self.want_held_end = defaults.held_end
        if self.want_held_end not in (-1, 1):
            raise ValueError("want_held_end must be +1 (hold) or -1 (release)")
        self.goal_dir_ref = _unit(np.asarray(self.goal_dir_ref).reshape(3))
        self.goal_axis_ref = _unit(np.asarray(self.goal_axis_ref).reshape(3))
        if defaults.minimum_travel:
            if not self.goal_dir_ref.any():
                self.goal_dir_ref = np.array([0., 0., 1.])
            if self.target_trans_m == 0:
                self.target_trans_m = defaults.minimum_travel
        self.region_half_extents = np.asarray(self.region_half_extents, dtype=float).reshape(3)
        if min(self.trans_tolerance, self.rot_tolerance, self.region_half_extents.min()) < 0:
            raise ValueError("Goal tolerances and region extents must be nonnegative")
        if self.goal_region_ref is not None:
            self.goal_region_ref = np.asarray(self.goal_region_ref, dtype=float).reshape(3)
        if defaults.region_required and self.goal_region_ref is None:
            raise ValueError(f"{self.family} requires a goal_region_ref")
        if self.p_contact_hint is not None:
            self.p_contact_hint = np.asarray(self.p_contact_hint, dtype=float).reshape(3)

    @property
    def carried_at_entry(self):
        return CONTACT_GOALS[self.family].carried_at_entry

    @property
    def minimum_travel(self):
        return CONTACT_GOALS[self.family].minimum_travel

    def _region_offset(self, target_pos_ref):
        if self.goal_region_ref is None:
            return np.zeros(3)
        center_offset = self.goal_region_ref - np.asarray(target_pos_ref, dtype=float).reshape(3)
        return center_offset - np.clip(center_offset, -self.region_half_extents, self.region_half_extents)

    def goal_header(self, R_world_eef, R_world_ref, progress_trans_m, progress_rot_rad, target_pos_ref,
                    holding_now, time_left, min_joint_margin, manip_pos, manip_ori):
        ref_to_eef = np.asarray(R_world_eef).T @ R_world_ref
        offset = self._region_offset(target_pos_ref)
        remain_m = (np.linalg.norm(offset) if self.goal_region_ref is not None
                    else self.target_trans_m - progress_trans_m)
        return np.asarray(np.r_[ref_to_eef @ self.goal_dir_ref, encode_axis(ref_to_eef @ self.goal_axis_ref),
                                remain_m, self.target_rot_rad - progress_rot_rad, ref_to_eef @ offset,
                                self.want_held_end, float(holding_now), np.clip(time_left, 0., 1.),
                                min_joint_margin, manip_pos, manip_ori], dtype=np.float32)

    def goal_reached(self, progress_trans_m, progress_rot_rad, target_pos_ref, holding_now):
        if bool(holding_now) != (self.want_held_end > 0):
            return False
        if self.goal_region_ref is not None:
            return bool(np.linalg.norm(self._region_offset(target_pos_ref)) <= self.trans_tolerance)
        if self.minimum_travel:
            return bool(progress_trans_m >= self.target_trans_m)
        return bool(abs(self.target_trans_m - progress_trans_m) <= self.trans_tolerance
                    and abs(self.target_rot_rad - progress_rot_rad) <= self.rot_tolerance)


def proprio_row(R0, p0, R, p, v_world, w_world, R_des, p_des, grip_open, grip_rate,
               prev_action, torque_features):
    """36 raw SI features. Tracking is at 12:18, torque features are at 28:36."""
    R0t = np.asarray(R0).T
    return np.r_[R0t @ (np.asarray(p) - p0), mat_to_rotvec(R0t @ R),
                 R0t @ v_world, R0t @ w_world, R0t @ (np.asarray(p_des) - p),
                 R0t @ mat_to_rotvec(R_des @ np.asarray(R).T), grip_open, grip_rate,
                 np.asarray(prev_action).reshape(8), np.asarray(torque_features).reshape(8)].astype(np.float32)


class JointTorqueObserver:
    """Current-derived effort residual at the TCP (robot acting on environment).

    J is the linear-then-angular world Jacobian at the same TCP. Transforming
    the resulting wrench into TCP axes thus needs rotation only. The last
    scalar is the DLS fit residual. Even upstream contact can fit a full-rank
    square Jacobian exactly; this scalar alone cannot locate collisions.
    """
    def __init__(self, torque_limits, damping=.04):
        self.torque_limits = np.asarray(torque_limits, dtype=float).reshape(-1)
        if np.any(self.torque_limits <= 0) or damping <= 0:
            raise ValueError("Expected positive torque limits and damping")
        self.damping = damping
        self.delta_tau0 = np.zeros_like(self.torque_limits)

    def tare(self, measured, free):
        self.delta_tau0[:] = np.asarray(measured) - free

    def observe(self, measured, free, J_world, R_world_eef):
        r = np.asarray(measured) - free - self.delta_tau0
        J = np.asarray(J_world, dtype=float).reshape(6, self.torque_limits.size)
        w = np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), J @ r)
        Re = np.asarray(R_world_eef).T
        return np.r_[Re @ w[:3], Re @ w[3:], np.max(np.abs(r) / self.torque_limits),
                     np.linalg.norm(r - J.T @ w)].astype(np.float32)


class GripStateObserver:
    """Estimate a grasp from gripper commands, encoders and the visual grasp site.

    At 20 Hz the two pads must obstruct commanded closure for two stable samples.
    Opening while closure is commanded clears confidence immediately. Remember
    the previous supported width until the jaws reclose or the caller commands
    release, so the wide-open plateau following a disturbance cannot look held.
    The visual site and pad extents use the tool frame (x across the jaws).
    """
    def __init__(self, pad_z, pad_y, carried_at_handoff=False, *, legacy=False):
        self.pad_z = tuple(float(z) for z in pad_z)
        self.pad_y = float(pad_y)
        self.carried_at_handoff = bool(carried_at_handoff)
        self.legacy = legacy
        self.previous_gap = None
        self.last_supported_width = None
        self.widening = False
        self.stable_ticks = 0
        self.blind_ticks = 0
        self.held = False

    def retains_grip(self, requested_width, gap):
        """Check this tick's encoders before transporting the previous grasp."""
        threshold = .004 if getattr(self, "legacy", True) else .0025
        return (self.held and gap > requested_width + threshold and
                self.previous_gap is not None and abs(gap - self.previous_gap) <= .0045
                and not self.widening)

    def update(self, requested_width, gap, site_eef, visible):
        legacy = getattr(self, "legacy", True)  # snapshots from earlier smoke runs
        change = 0. if self.previous_gap is None else gap - self.previous_gap
        if self.previous_gap is None and self.carried_at_handoff and (visible or legacy):
            self.stable_ticks, self.held = 2, True
            self.last_supported_width = gap
        self.previous_gap = gap
        self.blind_ticks = 0 if visible else self.blind_ticks + 1
        # Calibrated on a disjoint development set. These tolerances describe
        # sensor/hand geometry and never trigger an action or end an attempt.
        width_change = .0045
        x, y, z = site_eef
        at_pads = (abs(x) <= max(0., gap/2) + .006 and abs(y) <= self.pad_y + .014
                   and self.pad_z[0] - .014 <= z <= self.pad_z[1] + .014)
        obstruction = gap - requested_width
        obstructed = obstruction > (.004 if legacy else .0025)
        if not obstructed:
            self.last_supported_width = None
            self.widening = False
        elif self.last_supported_width is not None and change > width_change:
            self.widening = True
        if self.widening and gap <= self.last_supported_width + width_change:
            self.widening = False
        stable = abs(change) <= width_change and not self.widening
        if not legacy and not self.held and obstruction < .005:
            # Slow encoder settling is still free closure. Resolve a narrow
            # section only after the two pad faces actually stop approaching.
            stable = stable and abs(change) <= .00015
        if legacy:
            supported_site = at_pads and (visible or (self.held and self.blind_ticks <= 3))
        else:
            supported_site = (visible and at_pads) or (self.held and not visible)
        if obstructed and stable and supported_site:
            self.stable_ticks = min(2, self.stable_ticks + 1)
        else:
            self.stable_ticks = 0
        self.held = self.stable_ticks == 2
        if self.held:
            self.last_supported_width = gap
        return bool(self.held)


class LeakyPLAI:
    """20 Hz pose integrator with exponential leak of the stored tracking lead.

    Retention .975 per 50 ms gives a 1.97 s time constant. Each action adds a
    fresh increment after the retained translation and rotation are measured
    relative to the *current* TCP. The alpha-dependent clamps bound both.
    """
    def __init__(self, R, p, retention=.975):
        if not 0. <= retention <= 1.:
            raise ValueError("PLAI retention must be in [0,1]")
        self.retention = float(retention)
        self.R_des = np.array(R, dtype=float, copy=True)
        self.p_des = np.array(p, dtype=float, copy=True)

    @staticmethod
    def pos_cap(alpha):
        return .004 + .010 * (alpha - .30) / .70

    def command(self, action, R, p):
        a = np.clip(np.asarray(action, dtype=float).reshape(ACTION_DIM), -1., 1.)
        R, p = np.asarray(R), np.asarray(p)
        alpha = .30 + .35 * (a[6] + 1.)
        grip_width = .04 * (a[7] + 1.)
        pos_step = .008 * _cap(a[:3], 1.)
        rot_step = np.deg2rad(4.5) * _cap(a[3:6], 1.)
        lead = self.retention * (self.p_des - p) + R @ pos_step
        self.p_des = p + _cap(lead, self.pos_cap(alpha))
        retained_rot = self.retention * mat_to_rotvec(self.R_des @ R.T)
        rot_lead = mat_to_rotvec(rotvec_to_mat(R @ rot_step) @ rotvec_to_mat(retained_rot))
        self.R_des = rotvec_to_mat(_cap(rot_lead, np.deg2rad(6.) * alpha)) @ R
        return self.R_des.copy(), self.p_des.copy(), float(alpha), float(grip_width)

    @staticmethod
    def apply_gains(robot, alpha):
        # Low gains plus the 14 mm lead limit starved 75-degree pulls within
        # 130 ticks. Keep a compliant low end and reach near-nominal Panda
        # OSC gains for full-scale actions; position lead is always capped.
        kp = np.repeat([180., 150.], 3) * np.clip(alpha, .3, 1.)
        kd = 2. * np.sqrt(kp)
        if hasattr(robot, "_kp"):
            robot._kp[:], robot._kd[:] = kp, kd
            robot = robot.base
        robot.ctrl.kp[:], robot.ctrl.kd[:] = kp, kd


@dataclass
class EntryCandidate:
    R_handoff: np.ndarray
    p_handoff: np.ndarray
    R_contact: np.ndarray
    p_contact: np.ndarray
    q_handoff: np.ndarray
    q_chain: np.ndarray
    approach: np.ndarray
    hypothesis: str
    arc_joint_margin_rad: float | None = None
    visibility_score: float | None = None


class MultiHypothesisEntrySampler:
    """Check sampled entry poses against a set of plausible motion corridors.

    collision_fn(q, finger, phase, fraction) must check the arm including a
    held payload. It owns the distinction between intentional contact and
    forbidden surfaces. 'approach' runs from -1 at handoff to 0 at contact,
    then the selected hypothesis runs from 0 to 1. The returned joint chain
    is a feasibility witness. Only its handoff is passed to the actor.
    """
    def __init__(self, robot_model, collision_fn, wrist_visible=None, candidate_score=None, min_direction_cosine=.4):
        if not callable(collision_fn):
            raise TypeError("A scene collision checker is required")
        self.model, self.collision_fn = robot_model, collision_fn
        self.wrist_visible = wrist_visible
        self.candidate_score = candidate_score
        self.min_direction_cosine = min_direction_cosine

    def _visible(self, R, p, contact, q, finger):
        if self.wrist_visible is not None:
            return self.wrist_visible(R, p, contact)
        R_cam, p_cam = self.model.wrist_camera_pose(self.model.full_q(q, finger))
        x, y, z = R_cam[0].T @ (contact - p_cam[0])
        return z < 0 and max(abs(x), abs(y)) < -z * np.tan(np.deg2rad(self.model.cam_fovy) / 2)

    @staticmethod
    def _motions(contract, R, p, R_ref, approach, radius, goal_world, *, legacy_order=False,
                 clearance_m=.04, prefer_orbit=False):
        tangent = _unit(R_ref @ contract.goal_dir_ref)
        axis = _unit(R_ref @ contract.goal_axis_ref)
        angle = contract.target_rot_rad
        span = max(.04, abs(contract.target_trans_m), radius * abs(angle))
        if contract.carried_at_entry:
            destination = np.asarray(goal_world(R) if callable(goal_world) else goal_world, dtype=float)
            up = R_ref[:, 2]
            direct = destination[-1] if destination.ndim == 2 else destination
            delta = direct - p
            yield "insertion", np.linalg.norm(delta), lambda s: (R, p + s*delta)
            above = direct + up*clearance_m
            points = np.vstack((p, above, direct)) if destination.ndim == 1 else np.vstack((p, destination))
            lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            def corridor(s):
                return R, np.array([np.interp(s*lengths[-1], lengths, col) for col in points.T])
            corridor.waypoint_fractions = lengths[1:] / max(lengths[-1], 1e-12)
            yield "insertion", lengths[-1], corridor
            return

        direction = tangent if tangent.any() else approach
        linear = ("prismatic", span, lambda s: (R, p + s*span*direction))
        if legacy_order or abs(angle) < .05:
            yield linear
        if not axis.any():
            return
        if abs(angle) >= .05 and not legacy_order:
            # A tool approaching parallel to a rotation axis can rotate about
            # its own grip site. A transverse tangent describes an orbit.
            spin = ("rotation", radius*abs(angle), lambda s: (rotvec_to_mat(axis*s*angle) @ R, p))
            axial_grip = not prefer_orbit and (abs(axis @ approach) > .7 or not tangent.any())
            if axial_grip:
                yield spin
            if tangent.any():
                radial = radius*np.cross(np.sign(angle)*tangent, axis)
                center = p - radial
                def orbit(s):
                    rotation = rotvec_to_mat(axis*s*angle)
                    return rotation @ R, center + rotation @ radial
                yield "arc", abs(angle)*radius, orbit
            if not axial_grip:
                yield spin
            yield linear
            return
        # Historical entry law used by the frozen smoke distribution.
        up = R_ref[:, 2]
        transverse = _unit(np.cross(up, direction))
        if not transverse.any():
            transverse = _unit(np.cross(R_ref[:, 0], direction))
        for name, axis_raw in (("hinge-L", up), ("hinge-R", -up), ("hinge-H", transverse)):
            candidate_axis = _unit(axis_raw - direction * (axis_raw @ direction))
            if not candidate_axis.any():
                continue
            radial = radius * np.cross(direction, candidate_axis)
            center = p - radial
            def arc(s, axis=candidate_axis, radial=radial, center=center):
                rotation = rotvec_to_mat(axis*(s*span/radius))
                return rotation @ R, center + rotation @ radial
            yield name, span, arc

    def _extend(self, chain, waypoints, finger, phase, start_fraction):
        q, last_fraction = chain[-1], start_fraction
        for R, p, fraction in waypoints:
            sol, ok, _, _ = self.model.ik(R, p, q[None], q_rest=q, finger=finger, iters=48)
            if not ok[0]:
                return False
            q_next = sol[0]
            # Check intermediate configurations as well as Cartesian waypoints.
            n = max(1, int(np.ceil(np.max(np.abs(q_next - q)) / .055)))
            for i in range(1, n + 1):
                f = i / n
                if not self.collision_fn(q + f * (q_next - q), finger, phase,
                                         last_fraction + f * (fraction - last_fraction)):
                    return False
            chain.append(q_next)
            q, last_fraction = q_next, fraction
        return True

    def sample(self, contract, q_ref, contact_world, approach_world, closing_world, *,
               R_world_ref=None, finger=.04, rng=None, max_found=1, n_directions=5,
               n_random=3, goal_world=None, hinge_radius=None, legacy_unsigned_arc=False,
               legacy_order=False, clearance_m=.04):
        """Verify multiple motion hypotheses, including endpoint-first rolling arc IK."""
        from .gcp_entry import rank_rolling_arc
        if max_found <= 0 or n_directions <= 0:
            return []
        ref = np.eye(3) if R_world_ref is None else np.asarray(R_world_ref)
        rng = np.random.default_rng(0) if rng is None else rng
        candidates = self._sample_baseline(contract, q_ref, contact_world, approach_world,
                       closing_world, R_world_ref=ref, finger=finger, rng=rng, max_found=max_found,
                       n_directions=n_directions, n_random=n_random, goal_world=goal_world,
                       hinge_radius=hinge_radius, legacy_order=legacy_order, clearance_m=clearance_m)
        if abs(contract.target_rot_rad) < .05:
            return candidates
        return rank_rolling_arc(self, contract, candidates, q_ref, contact_world, approach_world,
                                closing_world, ref, finger, rng, max_found, legacy_unsigned_arc=legacy_unsigned_arc,
                                radius=.17 if legacy_order else hinge_radius)

    def _sample_baseline(self, contract, q_ref, contact_world, approach_world, closing_world, *,
               R_world_ref=None, finger=.04, rng=None, max_found=1, n_directions=5,
               n_random=3, goal_world=None, hinge_radius=None, legacy_order=False, clearance_m=.04):
        """Sample a near hemisphere and both rolls, then verify IK and clearance.

        For insertion, goal_world is the destination TCP or a callable taking
        candidate tool orientation and returning the grasp-offset-corrected TCP.
        The reference-frame contract goal describes the payload. finger uses
        model IK units.
        """
        if max_found <= 0 or n_directions <= 0:
            return []
        if contract.carried_at_entry and goal_world is None:
            raise ValueError("place entry requires goal_world at the destination TCP")
        prefer_orbit = hinge_radius is not None and hinge_radius > .025 and not legacy_order
        hinge_radius = .20 if hinge_radius is None else hinge_radius
        if hinge_radius <= 0:
            raise ValueError("hinge_radius must be positive")
        rng = np.random.default_rng(0) if rng is None else rng
        ref = np.eye(3) if R_world_ref is None else np.asarray(R_world_ref)
        nominal = _unit(approach_world)
        if not nominal.any():
            raise ValueError("Expected a nonzero approach direction")
        frame = frame_from_axes(nominal, closing_world)
        candidates = []
        alternative = []
        ranked = []
        for k in range(n_directions):
            direction = nominal
            if k:
                cos_theta = rng.uniform(self.min_direction_cosine, 1.)
                phi = rng.uniform(-np.pi, np.pi)
                direction = frame @ [np.sqrt(1 - cos_theta**2) * np.cos(phi),
                                     np.sqrt(1 - cos_theta**2) * np.sin(phi), cos_theta]
            standoff = rng.uniform(.04, .08)
            for sign in (1., -1.):
                R = frame_from_axes(direction, sign * frame[:, 0])
                contact = np.asarray(contact_world(R) if callable(contact_world) else contact_world, dtype=float)
                entry = contact - direction * standoff
                seeds = self.model.solve_ik(R, entry, q_ref, n_random=n_random, rng=rng, finger=finger)
                for q_entry in seeds:
                    if not self._visible(R, entry, contact, q_entry, finger):
                        continue
                    if not self.collision_fn(q_entry, finger, "approach", -1.):
                        continue
                    approach = [(R, entry + s * (contact - entry), s - 1)
                                for s in np.linspace(0., 1., int(np.ceil(standoff / .015)) + 1)[1:]]
                    initial = [q_entry]
                    if not self._extend(initial, approach, finger, "approach", -1.):
                        continue
                    for name, span, pose in self._motions(contract, R, contact, ref, direction,
                                                          hinge_radius, goal_world, legacy_order=legacy_order,
                                                          clearance_m=clearance_m, prefer_orbit=prefer_orbit):
                        # Explicit 4 cm probe, then remaining stroke/corridor.
                        progress = np.unique(np.r_[min(1., .04 / max(span, 1e-8)),
                            np.linspace(0., 1., max(2, int(np.ceil(span / .025)) + 1))[1:]])
                        if hasattr(pose, "waypoint_fractions"):
                            progress = np.union1d(progress, pose.waypoint_fractions)
                        path = [(*pose(s), s) for s in progress if s > 0.]
                        chain = initial.copy()
                        if self._extend(chain, path, finger, name, 0.):
                            candidate = EntryCandidate(R, entry, R, contact, q_entry,
                                                       np.asarray(chain), direction, name)
                            if (not legacy_order and abs(contract.target_rot_rad) >= .05
                                    and (prefer_orbit or abs((ref @ contract.goal_axis_ref) @ nominal) < .5)
                                    and name != "arc"):
                                if not alternative:
                                    alternative.append(candidate)
                            elif self.candidate_score is None:
                                candidates.append(candidate)
                            else:
                                score = self.candidate_score(candidate, finger)
                                candidate.visibility_score = score
                                ranked.append((score, candidate))
                                if score >= 1. and max_found == 1:
                                    return [candidate]
                            break
                    if candidates and candidates[-1].q_handoff is q_entry:
                        break
                if len(candidates) >= max_found:
                    return candidates
        if ranked:
            return [c for _, c in sorted(ranked, key=lambda item: -item[0])[:max_found]]
        return candidates or alternative


class _PointEncoder(nn.Module):
    """PointNet with max and mean over valid points, including empty payload."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(),
                                 nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, out_dim), nn.ReLU())
        self.register_buffer("scale", torch.tensor([.22, .22, .22] + [1.] * (in_dim - 3)))

    def forward(self, points, valid):
        encoded = self.mlp(points / self.scale)
        keep = valid.unsqueeze(-1)
        # ReLU features are nonnegative; zero is neutral to both reductions.
        encoded = encoded * keep
        avg = encoded.sum(-2) / keep.sum(-2).clamp(min=1)
        return torch.cat((encoded.amax(-2), avg), dim=-1)


class UniversalContactPolicy(nn.Module):
    """Two local point groups plus goal and 0.6 s proprioception -> 8 actions.

    Input points and proprioception are in physical units. Fixed normalization
    buffers travel with the state_dict. forward() is deterministic. For PPO,
    action_head(encode(obs)) gives unsquashed distribution means.
    """
    feature_dim = 384

    def __init__(self):
        super().__init__()
        self.scene_encoder = _PointEncoder(5, 128)
        self.payload_encoder = _PointEncoder(4, 96)
        self.goal_encoder = nn.Sequential(nn.Linear(16, 64), nn.SiLU())
        self.history_encoder = nn.Sequential(nn.Linear(12 * 36, 384), nn.LayerNorm(384), nn.SiLU(),
                                             nn.Linear(384, 256), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512), nn.SiLU(),
                                    nn.Linear(512, self.feature_dim), nn.LayerNorm(self.feature_dim), nn.SiLU())
        self.action_head = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.SiLU(), nn.Linear(128, ACTION_DIM))
        nn.init.uniform_(self.action_head[-1].weight, -.003, .003)
        nn.init.zeros_(self.action_head[-1].bias)
        # Goal: direction, octahedral axis, metres/radians remaining, goal
        # offset, grasp intent/state, time fraction, margin, two manipulabilities.
        self.register_buffer("goal_scale", torch.tensor([1.] * 5 + [.22, np.pi] + [.22] * 3 + [1.] * 6,
                                                         dtype=torch.float32))
        # Proprioception: p, rotvec, v, omega, tracking pos/rot, width/rate,
        # previous bounded action, estimated force/moment and residual scalars.
        self.register_buffer("history_scale", torch.tensor([.22] * 3 + [np.pi] * 3 + [.2] * 3 + [2.] * 3 +
                             [.014] * 3 + [.1] * 3 + [.08, .12] + [1.] * 8 + [20.] * 3 +
                             [2.] * 3 + [1., 5.], dtype=torch.float32))

    def encode(self, obs):
        scene, payload = obs["scene_pts"], obs["payload_pts"]
        scene_z = self.scene_encoder(scene, scene.ne(0).any(-1))
        payload_z = self.payload_encoder(payload, payload[..., 3] > 0)
        goal_z = self.goal_encoder((obs["goal"] / self.goal_scale).clamp(-10., 10.))
        history = (obs["history"] / self.history_scale).clamp(-10., 10.)
        history_z = self.history_encoder(history.flatten(start_dim=-2))
        return self.fusion(torch.cat((scene_z, payload_z, goal_z, history_z), dim=-1))

    def forward(self, obs):
        return self.action_head(self.encode(obs)).tanh()
