"""Procedural, sensorless contact training on Panda MuJoCo and two EGL depth cameras.

The critic, reward, perturbation and final verdict may inspect simulated bodies.
The actor registers a depth template from the handoff against unlabelled live
depth in a fixed reference frame. Only the handoff uses target segmentation.
The robot mask is available from the measured robot kinematics at every step.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from collections import Counter

import mujoco
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize_scalar

from atrace.gcp import (ContactContract, GripStateObserver, JointTorqueObserver, LeakyPLAI,
                        MultiHypothesisEntrySampler, proprio_row)
from atrace.gcp_labels import contact_patch_evidence
from atrace.gcp_tracking import ReferenceDepthTracker
from atrace.geometry import frame_from_axes, mat_to_rotvec, rotvec_to_mat
from . import scenes, ucs_scenes
from .fast_sim import FastSimRobot
from .gcp_teacher import GripRollout
from .render import Cameras, HEAD, WRIST, backproject
from .sim import StageSim
from .student_obs import SensorModel

KINDS = {"door":"push_pull", "drawer":"push_pull", "turn":"turn", "place":"place",
         "pick":"pick", "planar_push":"planar_push"}
SMOKE = ("door", "drawer")
DT = .05
IDENTITY = np.eye(3)
UP = IDENTITY[:, 2]
_B, _G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM


def _unit(x):
    return np.asarray(x) / max(np.linalg.norm(x), 1e-10)


def _draw(rows, count, rng):
    if len(rows) == 0:
        return np.zeros((count, rows.shape[1]), np.float32)
    return rows[rng.choice(len(rows), count, replace=len(rows) < count)].astype(np.float32)


def forbidden_geom_pairs(robot, fingers, target, payload=None, support=None):
    """Symmetric geometry-pair table for the C++ substep scanner.

    Payload contact against any shell/wall is forbidden. Only its designated
    support and the finger-on-target interface are allowed.
    """
    robot, fingers, target = (np.asarray(x, dtype=bool) for x in (robot, fingers, target))
    payload = np.zeros_like(robot) if payload is None else np.asarray(payload, bool)
    support = np.zeros_like(robot) if support is None else np.asarray(support, bool)
    owners = robot | payload
    forbidden = (owners[:, None] ^ owners[None, :]) | (robot[:, None] & payload) | (payload[:, None] & robot)
    intended = fingers[:, None] & target[None, :] | payload[:, None] & support[None, :]
    return np.ascontiguousarray(forbidden & ~(intended | intended.T), np.uint8)


class ReversalNoise:
    """Per-joint hysteresis; a zero-velocity sample preserves the last sliding branch."""
    def __init__(self, rng, limits):
        self.coulomb = rng.uniform(.001, .012, len(limits)) * limits
        self.sigma = rng.uniform(.0004, .002, len(limits)) * limits
        self.sign = np.zeros(len(limits))
        self.transient = np.zeros(len(limits))

    def sample(self, rng, velocity):
        sign = np.where(np.abs(velocity) > .013, np.sign(velocity), self.sign)
        reversal = (self.sign * sign) < 0
        self.transient = .65 * self.transient + reversal * sign * self.coulomb
        self.sign = sign
        return sign * self.coulomb + self.transient + rng.normal(size=len(sign)) * self.sigma


ALL5_MECH_CFG = dict(scenes.E2E)
ALL5_MECH_CFG.update(
    drawers=(1, 3),
    plinth=(0.0, 0.10),
    handle=("bar", "low_bar", "low_bar", "knob", "recess"),
    slide_damping=(1.5, 30.0),
    slide_friction=(0.1, 4.0),
    travel=(0.09, 0.18),
    front_h=(0.080, 0.14),
    overhang=(0.0, 0.045),
    mount_yaw=np.deg2rad(42.0),
    door_size=(0.16, 0.30),
    heavy_door=0.35,
    heavy_size=(0.18, 0.30),
    heavy_density=(400.0, 1100.0),
    heavy_damping=(0.20, 3.8),
    heavy_friction=(0.04, 0.85),
    door_range=np.deg2rad((80.0, 120.0)),
    door_handle=("bar", "bar", "knob", "lip"),
    protrusion=(0.028, 0.042),
    ridge_h=0.014,
)


def _rigid_collision_surfaces(m, d, bid):
    """Return the component collision surfaces in one rigid body frame."""
    mask = scenes._subtree(m, bid)[m.geom_bodyid] & ((m.geom_contype | m.geom_conaffinity) != 0)
    world_to_body = d.xmat[bid].reshape(3, 3).T
    surfaces = []
    for g in np.flatnonzero(mask):
        rotation = world_to_body @ d.geom_xmat[g].reshape(3, 3)
        position = world_to_body @ (d.geom_xpos[g] - d.xpos[bid])
        center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        surfaces.append((position, rotation, m.geom_size[g], m.geom_type[g],
                         position + rotation @ center, np.abs(rotation) @ half))
    return surfaces


def _rear_surface(m, d, geoms, center, direction, height):
    """Rear support point of a horizontal cross-section of a compound solid.

    Parallel rays recover each convex component's first visible skin. Minimize
    its depth over the transverse coordinate, including endpoints and facets.
    Equal-depth faces are loaded as close to the centre line as possible.
    """
    across = np.cross(UP, direction)
    origins = d.geom_xpos[geoms]
    radius = np.max(np.linalg.norm(origins-center, axis=1) + m.geom_rbound[geoms])
    start = center - (radius+.01)*direction
    start[2] = height
    nearest = None
    for g in geoms:
        R = d.geom_xmat[g].reshape(3, 3)
        middle = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        half = m.geom_aabb[g, 3:]
        if abs(middle[2]-height) > np.abs(R[2]) @ half:
            continue
        y0 = (middle-center) @ across
        span = np.abs(across @ R) @ half
        def distance(y):
            t = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g],
                                   start + y*across, direction, m.geom_type[g])
            return t if t >= 0. else 2*radius+.02
        offsets = np.unique(np.r_[np.linspace(y0-span, y0+span, 25), np.clip(0., y0-span, y0+span)])
        depths = np.array([distance(y) for y in offsets])
        if depths.min() >= 2*radius+.02:
            continue
        i = min(np.flatnonzero(depths <= depths.min()+1.e-9), key=lambda j: abs(offsets[j]))
        depth, offset = depths[i], offsets[i]
        bounds = offsets[max(0, i-1)], offsets[min(len(offsets)-1, i+1)]
        if bounds[1] > bounds[0]:
            opt = minimize_scalar(distance, bounds=bounds, method="bounded", options={"xatol": 1e-7})
            if opt.fun < depth-1e-8:
                depth, offset = opt.fun, opt.x
        if (nearest is None or depth < nearest[0]-1e-8 or
                (abs(depth-nearest[0]) < 1.e-8 and abs(offset) < abs(nearest[1]))):
            nearest = depth, offset
    return None if nearest is None else start + nearest[0]*direction + nearest[1]*across


def _thin_sections(surfaces):
    """Trace opposing horizontal rays through collision surfaces in the body frame.

    A valid pinch has solid material between opposing contacts, free space
    outside both pads, and a locally flat contact patch on both sides.
    Candidates near the centre of mass are tried first to reduce load torque.
    """
    candidates = []
    for pos, rot, size, typ, center, half in surfaces:
        for basis in rot.T:
            axis = _unit(np.r_[basis[:2], 0.])
            if abs(basis[2]) > .1 or not axis.any():
                continue
            tangent = np.cross(UP, axis)
            span = np.abs(tangent) @ half
            for offset in (0., -.48 * span, .48 * span):
                # Keep the grasp close to the upper surface for table and wall clearance.
                site = center + offset * tangent + UP * (half[2] - min(.012, .45 * half[2]))
                def section(point):
                    hits = [mujoco.mju_rayGeom(pos, rot.ravel(), size, point + sign*.3*axis,
                                                -sign*axis, typ) for sign in (-1., 1.)]
                    if min(hits) < 0.:
                        return None
                    return np.array([hits[0] - .3, .3 - hits[1]])
                bound = section(site)
                if bound is None or not .003 <= np.diff(bound)[0] < .065:
                    continue
                # Non-antipodal contacts on an oblique edge roll out of the grip.
                patches = [section(site + t*tangent) for t in (-.004, .004)]
                if any(b is None or np.max(np.abs(b-bound)) > .0025 for b in patches):
                    continue
                site = site + np.mean(bound) * axis
                width = .5 * (bound[1] - bound[0])
                # Reject surfaces hidden inside another component of the rigid body.
                blocked = False
                for other, orientation, dimensions, geometry, *_ in surfaces:
                    if other is pos:
                        continue
                    for sign in (-1., 1.):
                        ray = sign * axis
                        for z in (-.010, 0., .010):
                            footprint = site + z*UP
                            a = mujoco.mju_rayGeom(other, orientation.ravel(), dimensions,
                                                   footprint + ray*(width+.025), -ray, geometry)
                            b = mujoco.mju_rayGeom(other, orientation.ravel(), dimensions,
                                                   footprint + ray*(width+.0002), ray, geometry)
                            blocked |= (0 <= a < .025 or 0 <= b < .025)
                if not blocked:
                    score = np.linalg.norm(site[:2]) + .20 * width
                    candidates.append((score, site, axis, width))
    return [(site, axis, width) for _, site, axis, width in sorted(candidates, key=lambda c: c[0])]


def _supported_face_point(m, d, bid, part, pad_half):
    """Find a contiguous face patch wide enough to load with the closed fingers."""
    normal, u = part.face_normal, part.face_u
    v = np.cross(normal, u)
    geoms = np.flatnonzero(scenes._subtree(m, bid)[m.geom_bodyid] &
                           ((m.geom_contype | m.geom_conaffinity) != 0))
    offsets = [(a*part.half_extents[0], b*part.half_extents[1])
               for a in np.linspace(-.8, .8, 9) for b in np.linspace(-.8, .8, 9)]
    for x, y in sorted(offsets, key=lambda xy: np.hypot(*xy)):
        point = part.face_point + x*u + y*v
        # A hole can have a back wall. Every footprint ray must meet the same
        # exposed front plane, including rays over the edge of either pad.
        for dx, dy in ((0., 0.), (-1., -1.), (-1., 1.), (1., -1.), (1., 1.)):
            origin = point + pad_half*(dx*u+dy*v) + .012*normal
            hits = [mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g],
                                      origin, -normal, m.geom_type[g]) for g in geoms]
            if not any(.010 <= distance <= .014 for distance in hits):
                break
        else:
            return point
    return part.face_point


def _exposed_grip_points(m, d, body, site, outward, closing, jaw):
    """Seat the finger pads on the exposed sides of a narrow mounted part."""
    geoms = np.flatnonzero(scenes._subtree(m, body)[m.geom_bodyid] &
                           ((m.geom_contype | m.geom_conaffinity) != 0))
    along = np.cross(outward, closing)

    def height(offset, longitudinal=0.):
        source = site + offset*closing + longitudinal*along + .15*outward
        hits = [mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g],
                                   source, -outward, m.geom_type[g]) for g in geoms]
        first = min((t for t in hits if t >= 0.), default=None)
        return None if first is None else .15 - first

    peak = height(0.)
    if peak is None:
        return site, site
    centered = None
    # Continuous crest material across the pad footprint and two lower
    # shoulders provide opposed sidewalls. Centre the rubber on those walls.
    shoulders = [height(s*.012) for s in (-1., 1.)]
    if all(h is not None for h in shoulders):
        relief = peak - max(shoulders)
        ends = [height(0., s*(jaw["pad_y"]+.003)) for s in (-1., 1.)]
        if relief >= .005 and any(h is not None and abs(h-peak) < .0015 for h in ends):
            depth = min(relief-.001, .018, max(.006, .57*relief))
            centered = site + (peak-depth+np.mean(jaw["pad_z"]))*outward
    shoulders = [height(s*.010) for s in (-1., 1.)]
    relief = peak - max(0. if h is None else h for h in shoulders)
    depth = max(-.002, min(.006, relief - jaw["tip_z"] - .004)) if relief > .005 else .012
    tip = site + (peak-depth)*outward
    return (tip if centered is None else centered), tip


def _door_push_close(layout_seed):
    # Independent of geometry draws. Only a closing layout may have a bare face.
    return np.random.default_rng([layout_seed, 915]).random() < .35


def _joint_target(part, start, delta):
    lo, hi = sorted(part.range)
    end = start + delta
    if not (np.isfinite(delta) and abs(delta) > 1e-9 and lo - 1e-7 <= end <= hi + 1e-7):
        raise ValueError(f"Requested travel {delta} from {start} exceeds joint range {part.range} or is zero")
    return float(end)


def _drawer_command(rng, part, initial_fraction, delta):
    travel = part.open - part.closed
    handled = part.handle_point_b is not None and part.protrusion > .012 and 2 * min(part.handle_half) < .065
    if initial_fraction is None and delta is not None:
        initial_fraction = 0. if delta * travel > 0. else 1.
    if initial_fraction is None:
        if handled and rng.random() < .55:
            initial_fraction = rng.uniform(.25, .70)
        else:
            # Outside the overlap retain both near-closed and near-open starts.
            initial_fraction = rng.uniform(0., .10) if handled and rng.random() < .5 else rng.uniform(.71, .98)
    start = part.closed + travel * initial_fraction
    if delta is None:
        if .25 <= initial_fraction <= .70:
            opening = handled and rng.random() < .5
        else:
            opening = handled and initial_fraction < .5
        if opening:
            end_fraction = rng.uniform(initial_fraction + .08, .99)
        elif initial_fraction <= 1e-8:
            raise ValueError("Cannot request closing from the closed stop")
        else:
            upper = min(.40, initial_fraction - .06)
            end_fraction = rng.uniform(min(.15, .5 * upper), upper) if upper > 0 and rng.random() < .8 else 0.
        delta = travel * (end_fraction - initial_fraction)
    if delta * travel > 0. and not handled:
        raise ValueError("Outward slide requires an accessible handle")
    return start, _joint_target(part, start, delta)


def _door_command(rng, part, initial_fraction, delta):
    travel = part.open - part.closed
    extent = np.rad2deg(abs(travel))
    handled = part.handle_point_b is not None
    if initial_fraction is None:
        if delta is not None:
            initial_fraction = max(0., -delta / travel)
        elif handled and rng.random() < .55:
            initial_fraction = rng.uniform(20., 45.) / extent
        else:
            initial_fraction = (0. if handled and rng.random() < .85
                                else rng.uniform(max(.50, 46. / extent), .85))
    start = part.closed + travel * initial_fraction
    if delta is None:
        angle = initial_fraction * extent
        if 20. - 1e-8 <= angle <= 45. + 1e-8:
            opening = handled and rng.random() < .5
        else:
            opening = handled and angle < 20.
        if opening:
            terminal = rng.uniform(60. if angle >= 20. - 1e-8 else 35., min(80., extent))
        elif angle > 0:
            terminal = rng.uniform(0., .45 * angle) if rng.random() < .5 else 0.
        else:
            raise ValueError("Cannot request closing from the closed stop")
        delta = np.copysign(1., travel) * np.deg2rad(terminal - angle)
    if delta * travel > 0. and not handled:
        raise ValueError("Opening a door requires an accessible handle")
    _joint_target(part, start, delta)
    return start, float(delta)


def _turn_command(rng, part, initial_fraction, delta):
    lo, hi = sorted(part.range)
    span = hi - lo
    start = None if initial_fraction is None else part.closed + initial_fraction * (part.open - part.closed)
    # Reserve the sign and start quantile even for controlled starts so that
    # swapping the prescribed command doesn't also move the sensor-noise stream.
    sign_index = int(rng.integers(2))
    start_quantile = rng.random()
    if delta is None:
        minimum = max(np.deg2rad(25.), .30 * span)
        maximum = min(np.deg2rad(85.), .80 * span)
        rooms = (span, span) if start is None else (start - lo, hi - start)
        signs = [sign for sign, room in zip((-1., 1.), rooms) if min(room, maximum) >= minimum]
        if not signs:
            raise ValueError("No commanded turn in the curriculum interval fits this joint/start")
        sign = signs[sign_index % len(signs)]
        room = rooms[int(sign > 0)]
        delta = sign * rng.uniform(minimum, min(maximum, room))
    if start is None:
        lower, upper = lo - min(0., delta), hi - max(0., delta)
        if lower > upper:
            raise ValueError("Requested turn exceeds the procedural joint range")
        start = lower + start_quantile * (upper - lower)
    _joint_target(part, start, delta)
    return start, float(delta)


def _layout(kind, seed, out_dir, preset="smoke_door_drawer"):
    rng = np.random.default_rng(seed)
    fixture = None
    if preset == "smoke_door_drawer" and kind in SMOKE:
        # Draw the ignition distribution before entry. Both commanded angles
        # lie inside every door's hard range; every drawer has a real neighbour.
        cfg = dict(scenes.E2E)
        cfg.update(heavy_door=1., heavy_size=(.16,.24), heavy_density=(350.,650.),
                   heavy_damping=(.035,.10), heavy_friction=(.008,.03),
                   door_range=np.deg2rad([95.,115.]), door_handle=("bar",),
                   protrusion=(.032,.042), mount_yaw=np.deg2rad(38.),
                   slide_friction=(.1,.6), slide_damping=(1.,4.), drawers=(2,2),
                   handle=("bar",), travel=(.09,.14), front_h=(.085,.12), plinth=(.055,.10))
        xml, parts = (scenes.door if kind == "door" else scenes.drawer_stack)(rng, cfg=cfg)
        xmls = [xml]
    elif kind == "door":
        mode, coordinate = np.random.default_rng([seed, 915]).random(2)
        cfg = dict(ALL5_MECH_CFG, hinge_coordinate_sign=(1 if coordinate < .5 else -1))
        if mode < .35:
            cfg = dict(cfg, door_handle=(*cfg["door_handle"], "none"))
        xml, parts = scenes.door(rng, cfg=cfg)
        xmls = [xml]
    elif kind == "drawer":
        aux = np.random.default_rng([seed, 911])
        if seed > 1 and aux.random() < 0.15:
            xml, parts = scenes.sliding_panel(rng)
        else:
            xml, parts = scenes.drawer_stack(rng, cfg=ALL5_MECH_CFG)
        xmls = [xml]
    elif kind == "turn":
        aux = np.random.default_rng([seed, 912])
        if seed > 1 and aux.random() < 0.35:
            xml, parts = scenes.knob(rng, cfg=ALL5_MECH_CFG)
        else:
            xml, parts = ucs_scenes.ucs_lever_knob(rng)
        xmls = [xml]
    elif kind == "place":
        aux = np.random.default_rng([seed, 913])
        if seed > 1 and aux.random() < 0.28:
            xmls, fixture = ucs_scenes.ucs_caddy_slot(rng)
        else:
            factory = rng.choice([ucs_scenes.ucs_cavity_box, ucs_scenes.ucs_overhang_drawer])
            if preset == "all_5_families" and factory is ucs_scenes.ucs_overhang_drawer:
                xmls, fixture = factory(rng, fit_payload=True)
            else:
                xmls, fixture = factory(rng)
        parts = []
    else:
        aux = np.random.default_rng([seed, 914])
        if preset == "all_5_families" and seed > 1:
            pool = ("box", "mug", "bowl", "thin_book") if kind == "pick" else ("plate", "plate", "box", "bowl", "mug")
            pay_kind = str(aux.choice(pool))
            yaw = float(aux.uniform(-np.pi, np.pi))
            sx = float(-0.10 + aux.uniform(-0.05, 0.05))
            sy = float(rng.uniform(-0.16, 0.16))
        else:
            pay_kind, yaw, sx, sy = "box", None, -0.1, float(rng.uniform(-0.16, 0.16))
        xml, _, half_h_spawn = ucs_scenes.make_payload_xml(
            rng, pay_kind, (sx, sy, scenes.CFG["table_z"] + 0.053), yaw=yaw
        )
        if preset == "all_5_families" and seed > 1:
            # Spawn just 4 mm above tabletop so objects settle upright at rest without bounce/roll!
            xml = xml.replace(f"{scenes.CFG['table_z'] + 0.053:.4f}", f"{scenes.CFG['table_z'] + half_h_spawn + 0.004:.4f}", 1)
        parts, xmls = [], [xml]
    return scenes.Layout(scenes.base_layout(out_dir), xmls, parts, seed, kind), fixture


class GCPEnv:
    """Single-worker environment. An explicit seed fixes layout, initial physics and sampling."""
    def __init__(self, seed=0, out_dir="/tmp/gcp_bddl", preset="smoke_door_drawer", gpu=0,
                 staged_reset_prob=.5, horizon=130, evaluation=False, reward_mode="baseline", smoke_sampling=None,
                 place_grasp_jitter=None):
        if preset not in ("smoke_door_drawer", "all_5_families"):
            raise ValueError(preset)
        if reward_mode not in ("baseline","centered"):
            raise ValueError("Unknown GCP reward mode")
        self.reward_mode = reward_mode
        self.smoke_sampling = None if smoke_sampling is None else tuple(map(float,smoke_sampling))
        if self.smoke_sampling is not None and (
                len(self.smoke_sampling)!=3 or not all(0<=x<=1 for x in self.smoke_sampling)):
            raise ValueError("smoke_sampling expects (p_door,p_door75,p_slip75) in [0,1]")
        self.seed = int(seed)
        self.evaluation = bool(evaluation)
        # None enables perturbations on unseeded training resets. Explicit
        # evaluation seeds can opt in to the same distribution reproducibly.
        self.place_grasp_jitter = place_grasp_jitter
        self.place_jitter_status = "off"
        self.place_grasp_transform = None
        self.selector = np.random.default_rng([int(seed), 1963])
        self.rng = np.random.default_rng(seed)
        self.preset = preset
        self.kinds = SMOKE if preset == "smoke_door_drawer" else tuple(KINDS)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.horizon = int(horizon)
        self.staged_reset_prob = float(staged_reset_prob)
        self.gpu = gpu
        self.ss = self.cameras = None
        self.render_count = self.starts = 0
        self.priming_steps = self.priming_dual_renders = self.priming_attempts = self.priming_successes = 0
        self.priming_seconds = 0.
        self._priming = False
        self.priming_failed = False
        self.staged_origin = "fresh"
        self.buffer = []
        self.key = None

    @property
    def geom_names(self):
        """Diagnostic labels for replay reports."""
        return [mujoco.mj_id2name(self.m, _G, g) or "" for g in range(self.m.ngeom)]

    def _load(self, kind, seed):
        if self.key == (kind,seed):
            return
        if self.ss is not None:
            if self.cameras is not None:
                self.cameras.close()
                self.cameras = None
            self.ss.close()
        layout, self.fixture = _layout(kind, seed, self.out_dir, preset=self.preset)
        if self.preset == "all_5_families" and (layout.fixtures or
                Path(layout.bddl).resolve() != (Path(self.out_dir) / "table_only.bddl").resolve()):
            raise ValueError("The procedural pilot requires the generated empty-table carrier")
        self.ss = ss = StageSim(layout)
        self.kind = kind
        self.key = kind, seed
        ss.robot.sensorless = True
        self.fast_rb = rb = FastSimRobot(ss.robot, split_step=True)
        self.m, self.d = rb.m, rb.d
        self.jaw_geometry = rb.model.gripper_geometry(rb.q()[:7])
        self.cameras = Cameras(rb.m, rb.d, size=96)
        self.home = ss.snapshot()
        self.starts = 0
        self.prime_tried = set()
        self.buffer.clear()
        bn = np.array(ss._body)[rb.m.geom_bodyid]
        self.robot_g = np.array([x.startswith(("robot0_link","robot0_right_hand","gripper0_")) for x in bn])
        self.left_g = scenes._subtree(self.m, mujoco.mj_name2id(self.m,_B,"gripper0_leftfinger"))[self.m.geom_bodyid]
        self.right_g = scenes._subtree(self.m, mujoco.mj_name2id(self.m,_B,"gripper0_rightfinger"))[self.m.geom_bodyid]
        self.fingers = self.left_g | self.right_g
        self.scratch = mujoco.MjData(rb.m)
        self.pose_solutions = {}

    def _setup_contract(self, target_deg, use_tip=False, initial_fraction=None, target_trans_m=None):
        self.has_sidewall_alternative = False
        ss, m, d, kind = self.ss, self.m, self.d, self.kind
        self.part = ss.parts[0] if ss.parts else None
        self.hinge = self.part is not None and self.part.kind == "hinge"
        self.free = self.part is None
        self.reference = np.zeros(3)
        self.Rref = IDENTITY.copy()
        self.direction = UP.copy()
        self.axis = np.zeros(3)
        self.anchor = np.zeros(3)
        self.initial_joint = 0.
        self.intended_initial_joint = 0.
        goal_region, extent = None, np.zeros(3)
        goal_lin, goal_rot = 0., 0.
        if self.part is not None:
            drawer_target_joint = None
            door_push_close = False
            if not self.hinge:
                if self.preset == "smoke_door_drawer":
                    self.part = min(ss.parts, key=lambda p:p.face_point[2])
                    for pi in ss.parts:
                        ss.set_joint(pi.joint, (.85 if pi is not self.part else .98) * pi.open)
                    drawer_target_joint = self.part.closed
                else:
                    self.part = ss.parts[int(self.rng.integers(len(ss.parts)))]
                    q_init, drawer_target_joint = _drawer_command(
                        self.rng, self.part, initial_fraction, target_trans_m)
                    for pi in ss.parts:
                        if pi is self.part:
                            val = q_init
                        else:
                            fraction = self.rng.uniform(.25, .90) if self.rng.random() < .4 else 0.
                            val = pi.closed + fraction * (pi.open - pi.closed)
                        ss.set_joint(pi.joint, val)
            elif KINDS[kind] != "push_pull":
                delta = None if target_deg is None else np.deg2rad(target_deg)
                if self.preset == "all_5_families":
                    start, requested = _turn_command(self.rng, self.part, initial_fraction, delta)
                else:
                    lo, hi = sorted(self.part.range)
                    requested = (float(delta) if delta is not None else
                                 float(self.rng.choice([-1., 1.])*min(.65, .65*(hi-lo))))
                    lower, upper = lo-min(0., requested), hi-max(0., requested)
                    if lower > upper:
                        raise ValueError("Requested turn exceeds the procedural joint range")
                    start = self.rng.uniform(lower, upper)
                ss.set_joint(self.part.joint, start)
            else:
                start = self.part.closed
                if self.preset == "all_5_families":
                    delta = None if target_deg is None else np.deg2rad(target_deg)
                    start, requested = _door_command(self.rng, self.part, initial_fraction, delta)
                    door_push_close = requested * (self.part.open - self.part.closed) < 0.
                ss.set_joint(self.part.joint, start)
            p = self.part
            self.bid = mujoco.mj_name2id(m, _B, p.body)
            self.axis = p.axis.copy()
            self.anchor = p.axis_point.copy()
            self.reference = d.xpos[m.body_parentid[self.bid]].copy()
            self.initial_joint = ss.joint(p.joint)
            self.intended_initial_joint = float(self.initial_joint)
            push_without_grip = (not self.hinge and self.preset != "smoke_door_drawer" and
                                 p.protrusion < .012 and (drawer_target_joint-self.initial_joint)*(p.open-p.closed) < 0.)
            if push_without_grip:
                point = _supported_face_point(m, d, self.bid, p, self.jaw_geometry["finger_y"]+.003)
                self.hint_local = d.xmat[self.bid].reshape(3, 3).T @ (point - self.target_pos())
            elif door_push_close:
                self.hint_local = p.face_point_b + p.face_normal_b * (p.protrusion + .003)
            else:
                self.hint_local = p.handle_point_b.copy() if p.handle_point_b is not None else p.face_point_b.copy()
            contact = self.p_contact()
            self.approach = -p.face_normal.copy()
            grip_axis = p.face_u if p.handle_axis is None else p.handle_axis
            self.closing = _unit(np.cross(p.face_normal, grip_axis))
            if not self.hinge and np.isclose(*p.handle_half) and p.handle_point_b is not None:
                horizontal = _unit(np.cross(UP, self.approach))
                if horizontal.any():
                    self.closing = horizontal
            if (self.hinge and self.preset != "smoke_door_drawer" and
                    np.linalg.norm(np.cross(self.axis, contact - self.anchor)) < .025):
                pad, tip = _exposed_grip_points(m, d, self.bid, contact, p.face_normal,
                                                self.closing, self.jaw_geometry)
                self.has_sidewall_alternative = not use_tip and np.linalg.norm(pad-tip) > 1e-6
                contact = tip if use_tip else pad
                self.hint_local = d.xmat[self.bid].reshape(3,3).T @ (contact - self.target_pos())
            if self.hinge:
                if self.preset == "all_5_families":
                    goal_rot = requested
                elif target_deg is not None:
                    goal_rot = np.deg2rad(target_deg)
                elif KINDS[kind] == "push_pull":
                    goal_rot = np.deg2rad(self.rng.choice([40., 75.]))
                else:
                    goal_rot = requested
                rot_sgn = 1.0 if goal_rot >= 0. else -1.0
                tangent = np.cross(self.axis, contact-self.anchor) * rot_sgn
                self.direction = _unit(tangent) if np.linalg.norm(tangent) > 1e-8 else np.zeros(3)
                if self.direction @ self.approach > .5:
                    self.closing = _unit(np.cross(UP, self.approach))
                    if not self.closing.any():
                        self.closing = IDENTITY[:, 1].copy()
            else:
                target_j = p.closed if drawer_target_joint is None else drawer_target_joint
                goal_lin = target_j - self.initial_joint
                self.direction = _unit(self.axis * np.sign(goal_lin))
                if self.direction @ self.approach > .5:
                    # Align the broad axis of the closed hand with the support
                    # plane; a roll chosen to pinch a bar needlessly spans levels.
                    self.closing = _unit(np.cross(UP,self.approach))
            refname = ss._body[m.body_parentid[self.bid]]
        else:
            self.bid = mujoco.mj_name2id(m, _B, "ucs_payload")
            self.body_target = scenes._subtree(m, self.bid)
            self.hint_local = np.zeros(3)
            n_settle = 36 if self.preset == "smoke_door_drawer" else 80
            for _ in range(n_settle):
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m,d)
            center = self.target_pos()
            refname = "table"
            self.reference = np.array([0.,0.,scenes.CFG["table_z"]])
            self.approach = -UP.copy()
            self.closing = IDENTITY[:,1].copy()
            contact = center.copy()
            self.surfaces = _rigid_collision_surfaces(m, d, self.bid)
            object_R = d.xmat[self.bid].reshape(3, 3)
            self.sections = _thin_sections(self.surfaces)
            if kind == "pick":
                if self.sections:
                    self.hint_local, closing, _ = self.sections[0]
                    self.closing = object_R @ closing
                contact = self.p_contact()
                goal_lin = .06
            elif kind == "planar_push":
                self.direction = _unit(np.r_[self.rng.uniform(-1, 1, 2), 0.])
                self.approach = _unit(.28*self.direction - UP)
                self.closing = _unit(np.cross(self.direction, UP))
                centerline = self._push_edge(center, self.direction)
                geoms = np.flatnonzero(self.body_target[m.geom_bodyid] & ((m.geom_contype | m.geom_conaffinity) != 0))
                contact = _rear_surface(m, d, geoms, center, self.direction, centerline[2])
                if contact is None:
                    # A thin or disconnected footprint can miss the nominal
                    # height. Its component centres provide occupied slices.
                    for g in geoms:
                        contact = _rear_surface(m, d, geoms, center, self.direction, d.geom_xpos[g,2])
                        if contact is not None:
                            break
                if contact is None:
                    raise ValueError("The target has no ray-intersectable collision surface")
                self.hint_local = object_R.T @ (contact - center)
                push_dist = .09 if self.preset == "smoke_door_drawer" else float(self.rng.uniform(.08, .14))
                goal_region = center + push_dist*self.direction - self.reference
            else:
                f = ucs_scenes.bind_container_world(self.fixture,m,d)
                refname = f.fixture_body
                self.reference = f.floor_center_world.copy()
                self.Rref = f.R_world.copy()
                goal_region = np.array([0.,0.,f.payload_half_h+.002])
                extent[:2] = .55*np.array(f.floor_half_xy)
                self.direction = -f.open_normal_world.copy()
                lateral = _unit(np.cross(self.direction, UP))
                self.closing = lateral if lateral.any() else f.R_world[:, 1].copy()
                self.approach = _unit(self.direction - .32*UP)
                goal = self.reference + self.Rref @ goal_region
                # A waypoint outside the mouth with clearance for the lower lip.
                contact = goal - .055*self.direction + (.022 + f.sill_height)*UP
        self.hint_world = contact.copy()
        tol_m = .007 if kind == "drawer" else (.015 if kind == "planar_push" else .012)
        self.contract = ContactContract(KINDS[kind],ss._body[self.bid],refname,
            self.Rref.T@self.direction, self.Rref.T@self.axis, goal_lin,goal_rot,
            goal_region,self.Rref.T@(contact-self.reference),trans_tolerance=tol_m,
            region_half_extents=extent)
        self.part_joint_name = self.part.joint if self.part is not None else None
        self._set_target_masks(contact)
        self._pairs(False)
        self.q_initial_world = self.target_pos().copy()

    def _set_target_masks(self, contact):
        m, d = self.m, self.d
        self.body_target = scenes._subtree(m,self.bid)
        self.target_g = self.body_target[m.geom_bodyid]
        # Contact patch on the designated body. Adjacent panel/frame geoms may
        # share the same rigid body; use local distance to the declared patch.
        in_patch = np.linalg.norm(d.geom_xpos-contact,axis=1) - m.geom_rbound < .035
        self.grasp_g = self.target_g & in_patch
        if self.contract.family=="push_pull" and self.part is not None and self.part.handle_point_b is not None and not self.surface_push:
            # Only outward-facing material can be pinched against this panel.
            outward = (d.geom_xpos-self.part.face_point) @ self.part.face_normal > .001
            if np.any(self.grasp_g & outward):
                self.grasp_g &= outward
        if self.surface_push or self.free:
            self.grasp_g = self.target_g.copy()
        # Cast a short ray onto the declared support plane. More than one
        # collision surface may coincide there (e.g. visual and collision table).
        floor = self.reference if self.fixture is not None else np.r_[0., 0., scenes.CFG["table_z"]]
        self.support_g = np.zeros(m.ngeom, bool)
        solid = ((m.geom_contype | m.geom_conaffinity) != 0) & ~self.target_g & ~self.robot_g
        for g in np.flatnonzero(solid):
            hit = mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g],
                                     floor + .003*UP, -UP, m.geom_type[g])
            self.support_g[g] = 0. <= hit <= .005

    @property
    def surface_push(self):
        return self.contract.family=="planar_push" or (
            self.contract.family=="push_pull" and self.direction@self.approach>.5)

    def _pairs(self, carrying):
        self.payload_owned = bool(carrying)
        self.pairs = forbidden_geom_pairs(self.robot_g,self.fingers,self.grasp_g,
                         self.target_g if carrying else None,self.support_g)
        self.fast_rb.set_penetration_pairs(self.pairs)

    def target_pos(self):
        return self.d.xpos[self.bid].copy()

    def p_contact(self):
        return self.target_pos() + self.d.xmat[self.bid].reshape(3,3)@self.hint_local

    def _push_edge(self, center, direction):
        """Intersect the rear skin on the line of action through the body."""
        z_floor = self.reference[2]
        axes = self.d.xmat[self.bid].reshape(3, 3)
        half_height = max((axes @ c)[2] + np.abs(axes[2]) @ h
                          for _, _, _, _, c, h in self.surfaces)
        height = z_floor + min(.012, max(.004, .70*half_height))
        start = center - .25*direction
        start[2] = height
        distances = [mujoco.mju_rayGeom(self.d.geom_xpos[g], self.d.geom_xmat[g],
                                         self.m.geom_size[g], start, direction, self.m.geom_type[g])
                     for g in np.flatnonzero(self.body_target[self.m.geom_bodyid])]
        return start + min((t for t in distances if t >= 0.), default=.25)*direction

    def progress(self):
        if self.part is not None:
            delta = self.ss.joint(self.part.joint) - self.initial_joint
            return (0.,delta) if self.hinge else (delta,0.)
        return float((self.target_pos()-self.q_initial_world)@self.direction),0.

    def _stroke_visibility(self, candidate, finger):
        """Minimum depth coverage along the proposed translation, before handoff.

        Render the two calibrated 96-pixel depth buffers against
        the arm, object, and static scene at three certified configurations.
        The endpoint target is a geometric hypothesis confined to entry search.
        """
        m, scratch = self.m, self.scratch
        rb = self.fast_rb
        joint = m.jnt_qposadr[m.body_jntadr[self.bid]]
        q_chain = candidate.q_chain
        center = self.target_pos()
        score = np.inf
        for q in (q_chain[0], q_chain[len(q_chain)//2], q_chain[-1]):
            scratch.qpos[rb.model.qpos_adr] = rb.model.full_q(q, finger)[0]
            _, tcp = rb.model.site_pose(q[None])
            displacement = max(0., (tcp[0] - candidate.p_contact) @ self.direction)
            proposed = center + displacement*self.direction
            scratch.qpos[joint:joint+3] = proposed
            mujoco.mj_fwdPosition(m, scratch)
            counts = []
            for view in self.cameras.render(data=scratch):
                mask = (view.body >= 0) & self.body_target[view.body]
                # Boundary pixels are the first lost to edge holes and self masking.
                counts.append(np.count_nonzero(binary_erosion(mask)))
            score = min(score, .45*min(1., counts[0]/50.) + min(1.25, counts[1]/430.))
        scratch.qpos[joint:joint+3] = self.d.qpos[joint:joint+3]
        return float(score)

    def _supports_shallow_push(self, candidate):
        """Probe the real finger/target contact without advancing the simulator."""
        rb, m, scratch = self.fast_rb, self.m, self.scratch
        scratch.qpos[:] = self.d.qpos
        center = self.d.xipos[self.bid]
        target = np.flatnonzero(self.target_g)
        if not len(target) or not np.any(self.support_g):
            return False
        radius = np.max(np.linalg.norm((self.d.geom_xpos[target]-center)[:, :2], axis=1)
                        + m.geom_rbound[target])
        robot = np.flatnonzero(self.robot_g & ((m.geom_contype | m.geom_conaffinity) != 0))
        support = np.flatnonzero(self.support_g)
        _, sites = rb.model.site_pose(candidate.q_chain)
        q = candidate.q_chain[np.linalg.norm(sites-candidate.p_contact, axis=1).argmin()]
        # The operating aperture is 13.6 mm between the two sliding fingers.
        finger = .0068
        for advance in (0., .002, .004, .006, .009):
            if advance:
                point = candidate.p_contact + advance*self.direction
                solution, valid, _, _ = rb.model.ik(candidate.R_contact, point, q[None],
                                                     q_rest=q, finger=finger, iters=35)
                if not valid[0]:
                    continue
                q = solution[0]
            scratch.qpos[rb.model.qpos_adr] = rb.model.full_q(q, finger)[0]
            mujoco.mj_fwdPosition(m, scratch)
            contacts = scratch.contact
            a, b = contacts.geom[:scratch.ncon].T
            ab = self.fingers[a] & self.target_g[b]
            ba = self.fingers[b] & self.target_g[a]
            loaded = (ab | ba) & (contacts.dist[:scratch.ncon] <= 0.)
            if not np.any(loaded):
                continue
            # Overlap weights approximate the resultant normal of a closed
            # pair of pads. Its yaw moment must stay small about the body's
            # mass center and it must push the sidewall below that center.
            normals = contacts.frame[:scratch.ncon][loaded, :3] * np.where(ab[loaded], 1., -1.)[:, None]
            arms = contacts.pos[:scratch.ncon][loaded] - center
            weights = np.maximum(.00001, -contacts.dist[:scratch.ncon][loaded])
            normal = np.average(normals, axis=0, weights=weights)
            pitch_height = np.average(arms[:, 2], weights=weights)
            yaw_moment = np.average(np.cross(arms, normals)[:, 2], weights=weights)
            if (normal @ self.direction <= .72 or not -.40 <= normal[2] <= .25
                    or pitch_height >= .001 or abs(yaw_moment) >= .32*radius):
                continue
            if all(mujoco.mj_geomDistance(m, scratch, g, floor, .015, None) >= .001
                   for g in robot for floor in support):
                return True
        return False

    def _sample_shallow_push(self, sampler, contract, q_home, finger, n_random):
        """Prefer reachable side contacts with the hand pitched 20–35° below horizontal."""
        model = self.fast_rb.model
        fingers, hand = [], []
        for aperture in (finger, .0068):
            q = model.full_q(q_home, aperture)
            rotation, tcp = model.site_pose(q)
            local = (model.points_world(q)[0]-tcp[0]) @ rotation[0]
            fingers.append(local[model.points.group == 2])
            hand.append(local[model.points.group >= 1])
        fingers, hand = np.concatenate(fingers), np.concatenate(hand)
        floor = self.reference[2]
        def contact(rotation):
            height = floor + max(.0025, self.hint_world[2]-floor-.005) - np.min(hand @ rotation[2])
            pads = fingers @ rotation.T
            at_hint = np.abs(height+pads[:, 2]-self.hint_world[2]) < .010
            side = pads[at_hint] if at_hint.any() else pads
            along = side @ self.direction
            front = np.mean(side[along >= along.max()-.0015], axis=0)
            point = self.hint_world - front
            point[2] = height
            return point
        def score(candidate, opening):
            if not self._supports_shallow_push(candidate):
                return -np.inf
            return self._stroke_visibility(candidate, opening)
        shallow = MultiHypothesisEntrySampler(model, sampler.collision_fn, sampler.wrist_visible, score)
        proposal_rng = copy.deepcopy(self.rng)
        for pitch, yaw in ((30., 0.), (25., 90.), (25., -90.), (35., 60.), (35., -60.),
                           (20., 30.), (20., -30.)):
            rad = np.deg2rad(pitch)
            direction = rotvec_to_mat(UP*np.deg2rad(yaw)) @ self.direction
            approach = np.cos(rad)*direction - np.sin(rad)*UP
            closing = _unit(np.cross(UP, approach))
            found = shallow.sample(contract, q_home, contact, approach, closing, R_world_ref=self.Rref,
                                   finger=finger, rng=proposal_rng, n_directions=1, n_random=n_random)
            if found and found[0].visibility_score >= 1.:
                self.rng.bit_generator.state = proposal_rng.bit_generator.state
                return found
        return []

    def _joint_view_score(self, candidate, finger):
        """Measure the observable screw at handoff and at the closed-finger contact."""
        from atrace.gcp_joint import visible_normals
        rb = self.fast_rb
        chain = candidate.q_chain
        _, positions = rb.model.site_pose(chain)
        contact = np.linalg.norm(positions - candidate.p_contact, axis=1).argmin()
        score = np.inf
        for q, opening in ((chain[0], finger), (chain[contact], .009)):
            self.scratch.qpos[rb.model.qpos_adr] = rb.model.full_q(q, opening)[0]
            mujoco.mj_fwdPosition(self.m, self.scratch)
            pixels = interior_count = resolved_count = 0
            for view in self.cameras.render(data=self.scratch):
                mask = (view.body >= 0) & self.body_target[np.maximum(0, view.body)]
                points, _ = backproject(view, mask=mask)
                if not len(points):
                    continue
                pixels += len(points)
                field = np.zeros((*mask.shape, 3))
                field[mask] = points
                # Estimate a surface normal only if both difference stencils
                # remain on the measured part. The tracker also admits
                # silhouettes that pass its pixel-scale and depth tests.
                valid = binary_erosion(mask)[1:-1, 1:-1]
                dx = (field[1:-1, 2:] - field[1:-1, :-2])[valid]
                dy = (field[2:, 1:-1] - field[:-2, 1:-1])[valid]
                n = np.cross(dx, dy)
                n /= np.maximum(1e-12, np.linalg.norm(n, axis=1, keepdims=True))
                normals = np.zeros_like(field)
                normals[1:-1, 1:-1][valid] = n
                normals = normals[mask]
                screw = np.cross(self.axis, points - self.anchor)
                threshold = max(.008, .30*np.linalg.norm(screw, axis=1).max())
                interior_count += np.count_nonzero(np.abs(np.sum(normals*screw, axis=1)) > threshold)
                _, resolved = visible_normals(points, screw, normals, self.axis, self.anchor,
                                              [points], [(view.depth, view.K, view.T_wc)], threshold)
                resolved_count += np.count_nonzero(resolved)
            score = min(score, pixels/150., interior_count/5., resolved_count/6.)
        return float(score)

    def _joint_handoff_supported(self, candidate):
        """Check whether six *measured handoff* features will survive pad occlusion."""
        from atrace.gcp_tracking import _depth_pixels
        tracker = self.tracker
        high = (tracker.point_obs_gain > tracker.gain_threshold) & tracker.point_resolved
        if np.count_nonzero(high) < 6:
            return False
        rb, scratch, m = self.fast_rb, self.scratch, self.m
        scratch.qpos[:] = self.d.qpos
        chain = candidate.q_chain
        _, positions = rb.model.site_pose(chain)
        contact = np.linalg.norm(positions - candidate.p_contact, axis=1).argmin()
        # Find the aperture at the first physical pad touch against this part.
        opening = .008
        for width in np.linspace(.036, 0., 13):
            scratch.qpos[rb.model.qpos_adr] = rb.model.full_q(chain[contact], width)[0]
            mujoco.mj_fwdPosition(m, scratch)
            a, b = scratch.contact.geom[:scratch.ncon].T
            if np.any((self.fingers[a] & self.grasp_g[b]) | (self.fingers[b] & self.grasp_g[a])):
                opening = width
                break
        joint = m.jnt_qposadr[m.body_jntadr[self.bid]]
        # Reproject the very same measured template at acquisition and at the
        # requested endpoint. A point in front of an empty pixel is insufficient.
        for q, angle in ((chain[contact], 0.), (chain[-1], self.contract.target_rot_rad)):
            scratch.qpos[rb.model.qpos_adr] = rb.model.full_q(q, opening)[0]
            scratch.qpos[joint] = self.d.qpos[joint] + angle
            mujoco.mj_fwdPosition(m, scratch)
            points = tracker._model_at([angle])[0, high]
            supported = np.zeros(len(points), bool)
            for view in self.cameras.render(data=scratch):
                uv, z, inside = _depth_pixels(points, (view.depth, view.K, self.sensor.pose(view)))
                idx = np.flatnonzero(inside)
                x, y = uv[idx].T
                observed = view.depth[y, x]
                # This is an entry preflight render. A nearby cabinet surface
                # at the same depth is not part of the rotating template.
                bodies = view.body[y, x]
                hit = ((bodies >= 0) & self.body_target[np.maximum(bodies, 0)]
                       & (np.abs(observed-z[idx]) < .005+.0025*z[idx]**2))
                if np.any(hit):
                    pixels = y[hit]*view.depth.shape[1] + x[hit]
                    first = np.unique(pixels, return_index=True)[1]
                    supported[idx[hit][first]] = True
            if np.count_nonzero(supported) >= 6:
                return True
        return False

    def _sample_place_grasp_delta(self):
        """Offset the pinching site in the pad plane, plus independent tilt and yaw."""
        shift = np.r_[0., self.rng.uniform(-.008, .008, 2)]
        azimuth = self.rng.uniform(-np.pi, np.pi)
        tilt, yaw = self.rng.uniform(-np.deg2rad(9.), np.deg2rad(9.), 2)
        tilt_axis = np.array([np.cos(azimuth), np.sin(azimuth), 0.])
        return shift, rotvec_to_mat(tilt_axis*tilt) @ rotvec_to_mat(UP*yaw)

    def _entry(self, refine_view=False, place_delta=None):
        ss, rb, scratch = self.ss,self.fast_rb,self.scratch
        m,d = self.m,self.d
        q0 = ss.q_home
        scratch.qpos[:] = d.qpos
        # Approach is collision checked against the stationary scene. For the
        # hypothetical contact probe, the moving target voxels leave the guard.
        carrying = self.kind == "place"
        pinching = self.free and not carrying and not self.surface_push
        if carrying:
            qa = m.jnt_qposadr[m.body_jntadr[self.bid]]
            solid = (m.geom_contype | m.geom_conaffinity) != 0
            load_geoms = np.flatnonzero(self.target_g & solid)
            obstacles = np.flatnonzero(~self.robot_g & ~self.target_g & ~self.support_g & solid)
            # The compliant 20 Hz stroke can deviate from its IK witness.
            # Reserve half the controller's maximum Cartesian lead at walls.
            clearance = .5 * LeakyPLAI.pos_cap(1.)
            # Rotate about the selected grasp site, then translate it within
            # the pads. IK, the complete swept payload and initial physics all
            # use this same attachment, including its moment arm.
            def attachment(R, section):
                site, normal, _ = section
                world_axis = _unit(np.r_[R[:2, 0], 0.])
                body_R = np.column_stack((world_axis, np.cross(UP, world_axis), UP)) @ np.column_stack(
                    (normal, np.cross(UP, normal), UP)).T
                relative_R = R.T @ body_R
                if place_delta is None:
                    return relative_R, pad_center - relative_R @ site
                shift, rotation = place_delta
                relative_R = relative_R @ rotation
                return relative_R, pad_center + shift - relative_R @ site
            pad_center = np.array([0., 0., np.mean(self.jaw_geometry["pad_z"])])
            carry_rotation = carry_offset = None
            approach_pairs = forbidden_geom_pairs(self.robot_g,self.fingers,self.target_g,self.target_g,self.support_g)
        else:
            approach_pairs = self.pairs.copy()
            if self.kind == "pick" and self.preset == "all_5_families":
                # The fingers are open on this path. A swept intersection can
                # eject a light object before the observation template is taken.
                approach_pairs[np.ix_(self.fingers, self.target_g)] = 1
                approach_pairs[np.ix_(self.target_g, self.fingers)] = 1
        stroke_pairs = approach_pairs.copy()
        if not carrying:
            stroke_pairs[:,self.target_g] = stroke_pairs[self.target_g,:] = 0
        self.entry_collisions = Counter()
        def clear(q, finger, phase, fraction):
            scratch.qpos[rb.model.qpos_adr] = rb.model.full_q(q,finger)[0]
            mujoco.mj_kinematics(m,scratch)
            if carrying:
                nonlocal carry_rotation, carry_offset
                R = scratch.site_xmat[rb._site_id].reshape(3,3)
                p_site = scratch.site_xpos[rb._site_id]
                if phase == "approach" and fraction == -1.:
                    carry_rotation, carry_offset = attachment(R, section)
                R_pay = R @ carry_rotation
                scratch.qpos[qa:qa+3] = p_site + R @ carry_offset
                mujoco.mju_mat2Quat(scratch.qpos[qa+3:qa+7], R_pay.ravel())
                mujoco.mj_kinematics(m,scratch)
                for load in load_geoms:
                    reach = m.geom_rbound[load] + m.geom_rbound[obstacles] + clearance
                    near = np.linalg.norm(scratch.geom_xpos[obstacles] - scratch.geom_xpos[load], axis=1) < reach
                    for obstacle in obstacles[near]:
                        gap = mujoco.mj_geomDistance(m, scratch, load, obstacle, clearance+.0001, None)
                        if gap < clearance:
                            return False
            mujoco.mj_collision(m,scratch)
            pairs = approach_pairs if phase == "approach" else stroke_pairs
            c = scratch.contact
            if not scratch.ncon:
                return True
            g = c.geom[:scratch.ncon]
            bad = np.flatnonzero(pairs[g[:,0],g[:,1]] & (c.dist[:scratch.ncon]<-.00005))
            if len(bad):
                i,k = g[bad[0]]
                self.entry_collisions[(phase, int(i), int(k))] += 1
            return not len(bad)
        def visible(R,p,hint):
            if carrying and abs(R[2,0]) > .2:
                # The selected upright section has a horizontal opposing normal.
                # Large elevation of the jaw axis would wedge the load during closure.
                return False
            # Wrist camera is rigid on link 7; calibrated optical direction and
            # offset are expressed relative to the gripper site.
            rw,pw = rb.eef_pose()
            cid = self.cameras.ids[1]
            Rcam = R@rw.T@d.cam_xmat[cid].reshape(3,3)
            pcam = p+R@rw.T@(d.cam_xpos[cid]-pw)
            v = Rcam.T@(hint-pcam)
            return -v[2]>.02 and np.max(np.abs(v[:2])) < -v[2]*np.tan(np.deg2rad(m.cam_fovy[cid]/2))
        view_score = self._stroke_visibility if self.free and self.surface_push else None
        radial_distance = np.linalg.norm(np.cross(self.axis, self.hint_world-self.anchor))
        axial_view = (self.preset != "smoke_door_drawer" and self.hinge and radial_distance >= .025
                      and abs(self.axis @ self.approach) > .7)
        if axial_view and refine_view:
            # An axial view of an offset rotating contact can hide the moving
            # edges behind the hand. Retain the best reachable corridor even
            # if no permitted camera angle resolves the rotation.
            view_score = self._joint_view_score
        sampler = MultiHypothesisEntrySampler(rb.model, clear, wrist_visible=visible,
                                              candidate_score=view_score,
                                              min_direction_cosine=.77 if axial_view and refine_view else .4)
        finger = .015 if self.surface_push else .038
        preflight = copy.copy(self.contract)
        entry_contact = self.hint_world
        if self.free and self.surface_push:
            # The lower pad should intersect the measured sidewall while its tip
            # remains above the support. This offset is shared by every footprint.
            entry_contact = self.hint_world - self.direction * self.jaw_geometry["finger_y"]
            entry_contact = entry_contact - self.approach * np.mean(self.jaw_geometry["pad_z"])
            entry_contact[2] = max(entry_contact[2], self.reference[2] + self.jaw_geometry["tip_z"] + .0025)
            goal = self.reference + self.Rref @ preflight.goal_region_ref
            preflight.target_trans_m = np.linalg.norm((goal - self.target_pos())[:2])
        if self.kind == "door" and self.preset == "smoke_door_drawer":
            # The paired-command curriculum uses one physical entry law before
            # choosing a goal, including on separate fresh starts for each goal.
            preflight.target_rot_rad = np.copysign(max(np.deg2rad(75.),abs(preflight.target_rot_rad)),
                                                   preflight.target_rot_rad)
        key = (self.kind,tuple(self.hint_world.round(4)),tuple(self.direction.round(4)),
               preflight.target_trans_m,preflight.target_rot_rad)
        base_key = key
        if refine_view:
            key = (*key, "visibility")
        # A continuous perturbation changes both IK endpoints and swept
        # geometry. Its result cannot be reused for another attachment.
        solutions = self.pose_solutions if place_delta is None else {}
        if key not in solutions:
            n_dir = 5 if self.preset == "smoke_door_drawer" else 10
            n_rnd = 2 if self.preset == "smoke_door_drawer" else 3
            options = [(section, self.approach, 0.) for section in self.sections] if carrying or pinching else [
                (None, self.approach, 0.)]
            if self.free and self.surface_push:
                # Move within the low sidewall cone if the arm, palm or table
                # occludes the nominal approach. The support clearance and
                # entry IK are certified for each candidate.
                options = [(None, _unit(tilt*self.direction - UP), lift)
                           for tilt, lift in ((.28, .004), (.85, .004), (.85, .009), (.50, .009))]
            centered_grip = (self.hinge and self.preset != "smoke_door_drawer" and
                             np.linalg.norm(np.cross(self.axis, entry_contact-self.anchor)) < .025)
            if centered_grip:
                options = [(roll, self.approach, 0.) for roll in (0., np.pi/4, -np.pi/4, np.pi/2, None)]
            candidates = (self._sample_shallow_push(sampler, preflight, q0, finger, n_rnd)
                          if self.free and self.surface_push else [])
            shallow = bool(candidates)
            selected = None
            selected_finger = finger
            if shallow:
                options = []
            for section, approach, lift in options:
                closing = self.closing
                directions = n_dir
                if centered_grip and section is not None:
                    closing = rotvec_to_mat(self.axis*section) @ closing
                    directions = 1
                contact = entry_contact + lift*UP
                destination = None
                if carrying:
                    finger = min(.04, section[2] + .0005)
                    goal = self.reference + self.Rref @ self.contract.goal_region_ref
                    def destination(R):
                        _, offset = attachment(R, section)
                        return goal - R @ offset
                    def contact(R):
                        _, offset = attachment(R, section)
                        return entry_contact - R @ offset
                elif pinching:
                    site, normal, _ = section
                    body_rotation = d.xmat[self.bid].reshape(3, 3)
                    closing = body_rotation @ normal
                    horizontal = body_rotation @ np.r_[site[:2], 0.]
                    # Keep the jaws parallel to the two measured faces. A tilt
                    # across their normal rotates the payload as the pads seat.
                    tangent = horizontal - (horizontal @ closing) * closing
                    approach = _unit(-UP - .6 * tangent/max(.035, np.linalg.norm(horizontal)))
                    # Contact denotes the TCP; the physical site lies between the pads.
                    site_world = self.target_pos() + body_rotation @ site
                    def contact(R):
                        return site_world - R[:, 2] * np.mean(self.jaw_geometry["pad_z"])
                found = sampler.sample(
                    preflight, q0, contact, approach, closing, R_world_ref=self.Rref,
                    finger=finger, rng=self.rng, n_directions=directions, n_random=n_rnd,
                    goal_world=destination, legacy_order=self.preset == "smoke_door_drawer",
                    hinge_radius=(np.linalg.norm(np.cross(self.axis, entry_contact-self.anchor))
                                  if self.hinge and self.preset != "smoke_door_drawer" and not centered_grip
                                  else None),
                    clearance_m=(self.fixture.sill_height + .012 if carrying else .04))
                if found:
                    if centered_grip and found[0].hypothesis != "rotation" and section is not None:
                        if not candidates:
                            candidates, selected = found, section
                        continue
                    score = found[0].visibility_score
                    if not candidates or score is None or score > candidates[0].visibility_score:
                        candidates = found
                        selected, selected_finger = section, finger
                    if score is None or score >= 1.:
                        break
            if refine_view and not candidates:
                solutions[key] = solutions[base_key]
            else:
                solutions[key] = candidates[:1], selected, selected_finger, shallow
        candidates, section, finger, self.shallow_push_entry = solutions[key]
        self._view_candidate = candidates[0] if axial_view and candidates else None
        self.entry_hypothesis = candidates[0].hypothesis if candidates else None
        self.entry_arc_margin = candidates[0].arc_joint_margin_rad if candidates else None
        if not candidates:
            return False
        e = candidates[0]
        if self.free and self.surface_push:
            self.push_tcp_height = float(e.p_contact[2])
            self.push_contact_offset = (self.hint_world - e.p_contact if self.shallow_push_entry
                                        else np.zeros(3))
        if pinching:
            self.hint_local = section[0].copy()
            self.hint_world = self.p_contact()
            self.contract.p_contact_hint = self.Rref.T @ (self.hint_world - self.reference)
        elif carrying:
            self.hint_world = e.p_contact.copy()
            self.contract.p_contact_hint = self.Rref.T @ (self.hint_world - self.reference)
        self.approach = e.approach.copy()
        self.contact_R = e.R_contact.copy()
        d.qpos[rb.model.qpos_adr] = rb.model.full_q(e.q_handoff,finger)[0]
        d.qvel[ss._robot_dofs()] = 0
        ss.robot.ctrl.update_initial_joints(e.q_handoff.copy())
        ss.robot.posture = e.q_handoff.copy()
        rb._q_init[:] = e.q_handoff
        opening = (finger-.02)/.02
        rb._grip_cur[:] = [opening,-opening]
        mujoco.mj_forward(m,d)
        rot,pos = rb.eef_pose()
        self.entry_payload_pen = 0.
        if self.contract.family == "place":
            j = m.body_jntadr[self.bid]
            q,v = m.jnt_qposadr[j],m.jnt_dofadr[j]
            relative_R, offset = attachment(rot, section)
            rotations, tcp = rb.model.site_pose(rb.model.full_q(e.q_chain, finger))
            self.place_path = tcp + rotations @ offset
            d.qpos[q:q+3] = pos + rot @ offset
            mujoco.mju_mat2Quat(d.qpos[q+3:q+7], (rot @ relative_R).ravel())
            d.qpos[rb.model.qpos_adr[7:]] = [finger, -finger]
            d.qvel[rb._qvel_idx] = 0.
            d.qvel[v:v+6] = 0.
            mujoco.mj_forward(m,d)
            rb._grip_cur[:] = [-1.,1.]
            self._pairs(True)
        for _ in range(5):
            rb.step(rot,pos,0.)
            if self.payload_owned:
                self.entry_payload_pen = max(self.entry_payload_pen,rb.last_substep_max_pen_m)
        rb.sync_to_robosuite()
        if self.kind == "place":
            if self.preset == "all_5_families":
                grip_R, grip_p = rb.eef_pose()
                self.place_grasp_transform = (grip_R.T @ d.xmat[self.bid].reshape(3, 3),
                                              grip_R.T @ (d.xpos[self.bid] - grip_p))
            if place_delta is not None:
                held_R, held_offset = self.place_grasp_transform
                nominal_R = relative_R @ place_delta[1].T
                settled_delta = nominal_R.T @ held_R
                tilt = np.arccos(np.clip(settled_delta[2, 2], -1., 1.))
                yaw = np.arctan2(settled_delta[1, 0] - settled_delta[0, 1],
                                 settled_delta[0, 0] + settled_delta[1, 1])
                site_shift = held_offset + held_R @ section[0] - pad_center
                # A light asymmetric load can rotate or slide across the pads
                # during the settle. Keep the actual handoff in distribution.
                if (self.entry_payload_pen >= .001 or max(tilt, abs(yaw)) > np.deg2rad(10.)
                        or np.max(np.abs(site_shift[1:])) > .008):
                    return False
            return self._contacts()["firm"]
        return np.linalg.norm(rb.eef_pose()[1]-pos) < .006

    def _contacts(self):
        facts = contact_patch_evidence(self.d.contact[:self.d.ncon], self.left_g, self.right_g,
                    self.robot_g, self.grasp_g, self.direction)
        facts.update(width_m=float(self.fast_rb.finger_opening()),
                    command_width_m=float(getattr(self,"command_width",.08)),
                    target_delta=float(sum(self.progress())),
                    separation_m=float(np.linalg.norm(self.p_contact()-self.fast_rb.eef_pose()[1])))
        return facts

    def _clouds(self, initialize=False):
        views = self.cameras.render()
        self.render_count += 1
        R,p = self.fast_rb.eef_pose()
        scene, world, target, background = [], [], [], []
        legacy = self.preset == "smoke_door_drawer"
        depth_views = []
        for view in views:
            z = view.depth
            valid = np.isfinite(z) & (z>.02) & (z<3.) & ~self.sensor.holes(z,self.rng)
            b = view.body
            K = view.K
            T = self.sensor.pose(view)
            if not legacy:
                # The z-buffer includes the robot as an occluder. The registration
                # cloud below excludes it using the deployable kinematic self-mask.
                measured = np.full(z.shape, np.nan)
                measured[valid] = self.sensor.noisy(z[valid], view.name, self.rng)
                depth_views.append((measured, K, T))
            valid &= ~self.ss._is_robot[np.maximum(b,0)] | (b<0)
            ri,ci = np.nonzero(valid)
            depth = self.sensor.noisy(z[ri,ci],view.name,self.rng) if legacy else measured[ri,ci]
            pc = np.column_stack(((ci+.5-K[0,2])*depth/K[0,0],(ri+.5-K[1,2])*depth/K[1,1],depth))
            pts = pc@T[:3,:3].T + T[:3,3]
            if initialize:
                which = b[ri,ci]
                is_target = (which>=0) & self.body_target[np.maximum(which,0)]
                target.append(pts[is_target])
                background.append(pts[~is_target])
            else:
                world.append(pts)
            local = (pts-p)@R
            crop = np.max(np.abs(local),axis=1)<.22
            scene.append((local[crop], view.name==WRIST))
        if initialize:
            observed = np.concatenate(target)
            self.tracker = ReferenceDepthTracker(observed,np.concatenate(background),
                                                 self.anchor,self.axis,self.hinge,self.rng,legacy=legacy,
                                                 camera_targets=None if legacy else target,
                                                 camera_backgrounds=None if legacy else background,
                                                 views=depth_views)
        else:
            retaining = (self.sensed_hold if legacy else
                         self.grip_observer.retains_grip(self.requested_width, self.fast_rb.finger_opening()))
            if not legacy and self.sensed_hold and not retaining:
                opening = self.requested_width > self.fast_rb.finger_opening() + .0025
                self.tracker.record_grip(False, R, p, opening=opening)
            joint_prior = (self.tracker.carried_joint(R, p)
                           if retaining and self.axis.any() and not legacy else None)
            self.tracker.update(np.concatenate(world),self.rng,self.free,
                                grasp_prediction=self.tracker.carried_pose(R,p) if retaining or legacy else None,
                                angle_prior=joint_prior, views=depth_views,
                                camera_points=None if legacy else world)
            observed = self.tracker.observed
        hint = self.tracker.transform(self.hint_world)
        if initialize:
            carried = self.contract.carried_at_entry and self.entry_ok
            pad_z = self.jaw_geometry["pad_z"]
            self.grip_observer = GripStateObserver(pad_z,self.jaw_geometry["pad_y"],carried,legacy=legacy)
            self.grip_site0_world = p+R[:,2]*np.mean(pad_z) if carried else self.hint_world.copy()
        site = self.tracker.transform(self.grip_site0_world)
        # A smooth knob can be seen between the pads even when depth cannot
        # resolve its rotation about the symmetry axis.
        site_visible = self.tracker.visible if legacy else len(self.tracker.observed) >= 6
        self.sensed_hold = self.grip_observer.update(self.requested_width,self.fast_rb.finger_opening(),
                                                      R.T@(site-p),site_visible)
        self.tracker.record_grip(self.sensed_hold and (self.free or not legacy),R,p)
        payload = np.zeros((128,4),np.float32)
        if self.free and self.sensed_hold:
            if self.frozen_payload is None and len(observed)>=3:
                self.frozen_payload = _draw((observed-p)@R,128,self.rng)
            if self.frozen_payload is not None:
                payload[:,:3],payload[:,3] = self.frozen_payload,1.
        else:
            self.frozen_payload = None
        out = []
        for xyz,wrist in scene:
            heat = (np.exp(-np.sum((xyz-(hint-p)@R)**2,axis=1)/(.03**2*2))
                    if not self.sensed_hold and not self.contract.carried_at_entry else np.zeros(len(xyz)))
            out.append(np.column_stack((xyz,heat,np.full(len(xyz),float(wrist)))))
        nonempty = [x for x in out if len(x)]
        if len(nonempty)==2:
            cloud = np.concatenate([_draw(x,192,self.rng) for x in out])
        else:
            cloud = _draw(np.concatenate(out),384,self.rng)
        return cloud,payload

    def _proprio(self, initialize=False):
        rb, d = self.fast_rb, self.d
        R,p = rb.eef_pose()
        q,qd = rb.q()[:7],rb.joint_vel()
        jp,jr = np.zeros((3,self.m.nv)),np.zeros((3,self.m.nv))
        mujoco.mj_jacSite(self.m,d,jp,jr,rb._site_id)
        J = np.row_stack((jp[:,rb._qvel_idx],jr[:,rb._qvel_idx]))
        v = J @ qd
        tau_free = d.qfrc_bias[rb._qvel_idx] - d.qfrc_passive[rb._qvel_idx]
        measured = d.qfrc_actuator[rb._qvel_idx] + self.noise.sample(self.rng,qd)
        if initialize:
            self.observer.tare(measured,tau_free)
        features = self.observer.observe(measured,tau_free,J,R)
        width = rb.finger_opening()
        reference = self.core if self.policy_reference is None else self.policy_reference
        row = proprio_row(self.R0,self.p0,R,p,v[:3],v[3:],reference.R_des,reference.p_des,
                          width,(width-self.last_width)/DT,self.action,features)
        self.last_width = width
        if initialize:
            self.history = np.repeat(row[None],12,axis=0)
        else:
            self.history = np.concatenate((self.history[1:],row[None]))
        limits = rb.model.limits[:7]
        self.health = [float(np.minimum(q-limits[:,0], limits[:,1]-q).min()),
                       float(np.linalg.svd(J[:3],compute_uv=False).min()),
                       float(np.linalg.svd(J[3:],compute_uv=False).min())]

    def _header(self):
        R,_ = self.fast_rb.eef_pose()
        tracker = self.tracker
        if self.free:
            if self.preset == "smoke_door_drawer":
                estpos = self.q_initial_world + tracker.translation
                linear,angle = tracker.translation @ self.direction,0.
            else:
                estpos = tracker.transform(self.q_initial_world)
                linear,angle = (estpos - self.q_initial_world) @ self.direction,0.
        else:
            estpos = tracker.transform(self.q_initial_world)
            linear,angle = (0.,tracker.progress_value) if self.hinge else (tracker.progress_value,0.)
        header = self.contract.goal_header(R,self.Rref,linear,angle,
                    self.Rref.T@(estpos-self.reference),self.sensed_hold,1.-self.steps/self.horizon,*self.health)
        if self.hinge:   # transport initial tangent by measured target rotation
            header[:3] = R.T @ rotvec_to_mat(self.axis*angle) @ self.direction
        return header

    def _observe(self, initialize=False):
        scene,payload = self._clouds(initialize)
        self._proprio(initialize)
        self.obs = {"scene_pts":scene,"payload_pts":payload,"goal":self._header(),"history":self.history.copy()}

    def scan_penetration(self):
        c = self.d.contact
        g = c.geom[:self.d.ncon]
        if not len(g):
            return 0.
        mask = self.pairs[g[:,0],g[:,1]].astype(bool)
        return max(0.,float(np.max(-c.dist[:self.d.ncon][mask],initial=0.)))

    def _target_distance(self):
        linear,angle = self.progress()
        c = self.contract
        if c.goal_region_ref is not None:
            return np.linalg.norm(c._region_offset(self.Rref.T@(self.target_pos()-self.reference)))
        if c.family=="pick":
            return max(0.,c.target_trans_m-linear)
        if self.hinge:
            return abs(c.target_rot_rad-angle) * max(.04,np.linalg.norm(
                np.cross(self.axis,self.hint_world-self.anchor)))
        return abs(c.target_trans_m-linear)

    def _goal(self):
        c = self.contract
        ok = c.goal_reached(*self.progress(),self.Rref.T@(self.target_pos()-self.reference),self.facts["firm"])
        if self.kind == "place":
            ok = ok and self.d.xmat[self.bid].reshape(3,3)[2,2]>.90 and self.facts["separation_m"]>.035
            # The object must rest on the intended support.
            geoms = self.d.contact.geom[:self.d.ncon]
            ok = ok and np.any((self.target_g[geoms[:,0]] & self.support_g[geoms[:,1]]) |
                               (self.target_g[geoms[:,1]] & self.support_g[geoms[:,0]]))
        return bool(ok and self.entry_ok)

    @staticmethod
    def _new_reward_counts():
        return dict(goal_entries=0,goal_dwell_steps=0,slow_goal_steps=0,slips_observed=0,
                    reacquisitions=0,causal_recoveries=0,recovery_bonuses=0)

    def _centered_state(self):
        """Dimensionless target error and speed in the fixed reference."""
        c = self.contract
        linear,angular = self.progress()
        if self.hinge:
            error = abs(c.target_rot_rad-angular)
            tolerance = c.rot_tolerance
        else:
            error = self._target_distance()
            if c.family=="pick":
                error = max(0.,c.target_trans_m-linear)
            tolerance = c.trans_tolerance
        if self.free:
            dof = self.m.jnt_dofadr[self.m.body_jntadr[self.bid]]
            speed = np.linalg.norm(self.d.qvel[dof:dof+3])
        else:
            speed = abs(self.d.qvel[self.ss._adr(self.part.joint)[1]])
        tolerance = max(tolerance,1e-5)
        center = float(np.exp(-.5*(error/tolerance)**2))
        speed_ratio = float(speed/(2.*tolerance))  # one tolerance per 0.5 s
        return center,speed_ratio

    def _info(self):
        lin,ang = self.progress()
        pos = self.target_pos()
        c = self.contract
        critic = np.asarray(np.r_[lin,ang,c.target_trans_m,c.target_rot_rad,
            self.Rref.T@(pos-self.reference),self.facts["firm"],self.facts["left_target"],self.facts["right_target"],
            self.facts["separation_m"],self.max_pen_m,self.progress_error,self.slip_injected,
            self.entry_ok,self.steps/self.horizon],np.float32)
        info = dict(kind=self.kind,layout_seed=self.key[1],family=c.family,steps=self.steps,critic=critic,
                    reward_mode=self.reward_mode,reward_counts=self.reward_counts.copy(),
                    entry_ok=self.entry_ok,entry_failed=not self.entry_ok,render_count=self.render_count,
                    entry_hypothesis=self.entry_hypothesis,entry_arc_joint_margin_rad=self.entry_arc_margin,
                    staged=self.staged,staged_origin=self.staged_origin,priming_steps=self.priming_steps,
                    priming_dual_renders=self.priming_dual_renders,priming_seconds=self.priming_seconds,
                    priming_attempts=self.priming_attempts,priming_successes=self.priming_successes,
                    priming_failed=self.priming_failed,goal_deg=float(np.rad2deg(c.target_rot_rad)),angle_deg=float(np.rad2deg(ang)),
                    target_delta=lin,target_pos_ref=self.Rref.T@(pos-self.reference),slip_injected=self.slip_injected,
                    slip_lost=self.slip_lost,regrasped=self.regrasped,held=self.facts["firm"],
                    contact_facts=self.facts.copy(),max_pen_m=self.max_pen_m,goal_satisfied=self._goal(),
                    success=bool(self.good_steps>=10 and self.entry_ok),depth_tracking_visible=self.tracker.visible)
        if self.preset == "all_5_families" and self.kind == "place":
            info["place_grasp_jitter"] = self.place_jitter_status
        return info

    def reset(self, seed=None, kind=None, target_deg=None, staged=None, slip_at_deg=None, *,
              initial_fraction=None, target_trans_m=None):
        """Optionally set a fresh joint's closed-to-open fraction and signed travel."""
        if initial_fraction is not None and not 0. <= initial_fraction <= 1.:
            raise ValueError("initial_fraction must lie in [0, 1]")
        if (initial_fraction is not None or target_trans_m is not None) and self.preset != "all_5_families":
            raise ValueError("Controlled joint starts require all_5_families")
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        training = seed is None and not self.evaluation
        request = training and self.selector.random()<self.staged_reset_prob if staged is None else bool(staged)
        if self.evaluation and request:
            raise ValueError("Evaluation requires a fresh, unconditional start")
        # Keep expensive scene meshes only for a finite block of training starts.
        if training and kind is None and self.key is not None and self.starts<12 and self.entry_ok:
            kind,layout_seed = self.key
        else:
            mix = self.smoke_sampling if training and self.preset=="smoke_door_drawer" else None
            if kind is None and mix is not None:
                kind = "door" if self.rng.random()<mix[0] else "drawer"
            kind = str(kind or self.rng.choice(self.kinds))
            layout_seed = int(seed if seed is not None else self.rng.integers(0,100_000))
        if kind not in KINDS:
            raise ValueError(kind)
        if initial_fraction is not None and (kind not in ("drawer", "door", "turn") or request):
            raise ValueError("initial_fraction requires a fresh procedural mechanism start")
        if target_trans_m is not None and kind != "drawer":
            raise ValueError("A translational override requires a drawer")
        if self.preset == "all_5_families" and target_deg is not None and kind not in ("door", "turn"):
            raise ValueError("An angular override requires a rotational mechanism")
        self._load(kind,layout_seed)
        self.starts += 1
        self.priming_failed = False
        self.staged_origin = "fresh"
        if request and self.buffer:
            return self._draw_stage(target_deg, slip_at_deg, target_trans_m)
        self.ss.restore(self.home)
        self.fast_rb.sync_from_robosuite()
        if (training and self.preset=="smoke_door_drawer" and self.smoke_sampling is not None
                and kind=="door" and target_deg is None):
            target_deg = 75. if self.rng.random()<self.smoke_sampling[1] else 40.
        start_rng = copy.deepcopy(self.rng.bit_generator.state)
        selector_rng = copy.deepcopy(self.selector.bit_generator.state)
        jitter_enabled = training if self.place_grasp_jitter is None else self.place_grasp_jitter
        try_jitter = self.preset == "all_5_families" and kind == "place" and jitter_enabled
        self.place_jitter_status = "requested" if try_jitter else "off"
        self._initialize_handoff(target_deg, slip_at_deg, training, refine_view=False,
                                 initial_fraction=initial_fraction, target_trans_m=target_trans_m)
        retry_place = try_jitter and not self.entry_ok
        if try_jitter:
            self.place_jitter_status = "nominal_fallback" if retry_place else "applied"
        retry_tip = not self.entry_ok and self.has_sidewall_alternative
        retry_view = (self.entry_ok and self._view_candidate is not None
                      and not self._joint_handoff_supported(self._view_candidate))
        if retry_tip or retry_view or retry_place:
            # Restore the same contract draw and physical state before testing
            # a different entry geometry or the nominal grasp.
            self.ss.restore(self.home)
            self.fast_rb.sync_from_robosuite()
            self.rng.bit_generator.state = start_rng
            self.selector.bit_generator.state = selector_rng
            self._initialize_handoff(target_deg, slip_at_deg, training, refine_view=retry_view, use_tip=retry_tip,
                                     initial_fraction=initial_fraction, target_trans_m=target_trans_m)
        if request and self.entry_ok and not self.evaluation and self.horizon>18:
            key = (self.kind,round(self.contract.target_rot_rad,2),round(self.contract.target_trans_m,2))
            if key not in self.prime_tried:
                self.prime_tried.add(key)
                self._prime_stages()
            if self.buffer:
                return self._draw_stage(target_deg, slip_at_deg, target_trans_m)
            self.priming_failed = True
        return self.obs,self._info()

    def _initialize_handoff(self, target_deg, slip_at_deg, training, refine_view, use_tip=False,
                            initial_fraction=None, target_trans_m=None):
        self.staged = False
        self.teacher_release_pending = False
        self.teacher_contact_acquired = False
        self.teacher_release_width = 0.
        self.teacher_place_release = None
        self._setup_contract(target_deg, use_tip=use_tip, initial_fraction=initial_fraction,
                             target_trans_m=target_trans_m)
        self.command_width = .08
        self.contact_R = self.fast_rb.eef_pose()[0]
        self.entry_payload_pen = 0.
        if self.preset == "all_5_families" and self.kind == "place":
            self.place_path = None
            self.place_grasp_transform = None
        if self.place_jitter_status == "requested":
            self.entry_ok = bool(self._entry(refine_view, place_delta=self._sample_place_grasp_delta()))
            if not self.entry_ok:
                # Only an admitted handoff initializes cameras, tracker and history.
                # reset() rewinds physics and both RNGs before the nominal trial.
                return
        else:
            self.entry_ok = bool(self._entry(refine_view))
        self.R0,self.p0 = self.fast_rb.eef_pose()
        self.core = LeakyPLAI(self.R0,self.p0)
        self.q_initial_world = self.target_pos().copy()
        self.initial_joint = self.ss.joint(self.part.joint) if self.part is not None else 0.
        # Keep the exact intended pre-entry joint reference even if entry settle moved the door slightly.
        if self.hinge and self.contract.family=="push_pull":
            self.initial_joint = self.intended_initial_joint
        if self.kind=="place":
            self.command_width = 0.
        self.requested_width = self.command_width
        self.policy_reference = None
        self.last_width = self.fast_rb.finger_opening()
        self.facts = self._contacts()
        self._pairs(self.free and self.facts["firm"])
        self.sensor = SensorModel(self.rng,96)
        self.frozen_payload = None
        limits = np.max(np.abs(self.m.actuator_ctrlrange[self.fast_rb._arm_act]),axis=1)
        self.observer,self.noise = JointTorqueObserver(limits),ReversalNoise(self.rng,limits)
        self.action = np.zeros(8)
        self.steps = self.good_steps = self.release_left = self.grasp_streak = 0
        self.turn_rollout = None
        self.teacher_load, self.teacher_load_step = None, -100
        self.slip_injected = self.slip_lost = self.regrasped = self.before_slip_held = False
        self.grasp_latched = self.facts["firm"]
        self.last_secure_grasp = -1000
        self.slip_at_deg = (15. if training and abs(self.contract.target_rot_rad)>.9 and not self.surface_push
                            and slip_at_deg is None and self.selector.random() < (
                            .30 if self.smoke_sampling is None else self.smoke_sampling[2]) else slip_at_deg)
        self.max_pen_m = max(self.scan_penetration(),self.entry_payload_pen)
        self.progress_error = self._target_distance()
        self.center_potential = self._centered_state()[0] if self.reward_mode=="centered" else 0.
        self.slip_peak_gap = 0.
        self.slip_policy_opened = self.causal_recovered = False
        self.reward_counts = self._new_reward_counts()
        self.max_travel = 0.
        self.episode_contact_saved = False
        self._observe(initialize=True)

    def _draw_stage(self, target_deg, slip_at_deg, target_trans_m=None):
        snapshot = self.buffer[int(self.selector.integers(len(self.buffer)))]
        self.restore(snapshot,target_deg=target_deg,slip_at_deg=slip_at_deg,target_trans_m=target_trans_m)
        self.staged = True
        self.reward_counts = self._new_reward_counts()  # count learner events since the staged handoff
        self.staged_origin = snapshot.get("bank_source","on_policy")
        return self.obs,self._info()

    def _prime_stages(self):
        """Physically execute demonstrations solely to seed this layout's reset bank.

        The learner never counts these steps as its own transitions. Store only
        acquired contacts and witnessed recovery states. Restore the complete
        original start including sensor noise/registration if no bank was built.
        """
        original = self.snapshot()
        self.priming_attempts += 1
        before_render = self.render_count
        started = time.monotonic()
        saved = []
        categories = set()
        self._priming = True
        try:
            if abs(self.contract.target_rot_rad) > (.5 if self.preset=="smoke_door_drawer" else .9) and not self.surface_push:
                self.slip_at_deg = 15. if self.selector.random()<.5 else None
            full = max(self.progress_error,.015)
            for _ in range(min(105,self.horizon-15)):
                _,_,done,_ = self.step(self.expert_action())
                self.priming_steps += 1
                if done:
                    break
                progress = float(np.clip(1.-self.progress_error/full,0.,.99))
                phase = ("reacquired" if self.regrasped else
                         "slipped" if self.slip_lost and self.release_left==0 else "engaged")
                key = (phase,int(progress*5))
                contact = self.grasp_streak>=3 or self.facts["push_face_loaded"] or (
                    phase=="slipped" and self.before_slip_held)
                if contact and key not in categories and self.max_pen_m<.001:
                    shot = self.snapshot()
                    shot["bank_source"] = "primed"
                    saved.append(shot)
                    categories.add(key)
            if saved:
                self.buffer = (self.buffer+saved)[-16:]
                self.priming_successes += 1
        finally:
            self._priming = False
            self.priming_dual_renders += self.render_count-before_render
            self.priming_seconds += time.monotonic()-started
            self.restore(original)

    def step(self, action):
        a = np.asarray(action,float)
        if a.shape!=(8,) or not np.isfinite(a).all():
            raise ValueError("Expected a finite 8-D action")
        requested = np.clip(a,-1.,1.)
        a = requested
        self.requested_width = .04*(requested[7]+1.)
        rb = self.fast_rb
        R,p = rb.eef_pose()
        old_gap = self.facts["separation_m"]
        forced = False
        had_loss, had_regrasp = self.slip_lost,self.regrasped
        self.steps += 1
        if self.entry_ok:
            if (self.slip_at_deg is not None and not self.slip_injected and self.hinge and
                    np.rad2deg(self.progress()[1])*np.sign(self.contract.target_rot_rad)>=self.slip_at_deg):
                self.slip_injected = True
                self.before_slip_held = (self.grasp_latched and self.last_secure_grasp>=0 and
                    (self.facts["left_target"] or self.facts["right_target"]))
                self.release_left = 7
                # The exogenous release modifies the plant command only. The
                # policy retains its own requested reference for tracking error.
                self.policy_reference = copy.deepcopy(self.core)
            if self.policy_reference is not None:
                self.policy_reference.command(requested,R,p)
            if self.release_left:
                forced = True
                a = a.copy()
                a[:6] = np.r_[R.T@(-self.approach*.8),np.zeros(3)]
                a[7] = 1.
                self.release_left -= 1
            rd,pd,alpha,self.command_width = self.core.command(a,R,p)
            self.core.apply_gains(rb,alpha)
            # PandaGripper integrates sign(action) in the fast stepper. Place
            # the continuous position command directly in its target register.
            width = np.clip((self.command_width-2*rb._pad_off)/.04-1.,-1.,1.)
            rb._grip_cur[:] = (width,-width)
            rb.step(rd,pd,0.)
        self.action = requested.copy()
        self.facts = self._contacts()
        held = self.facts["firm"]
        if held:
            self.grasp_latched = True
        elif (not self.facts["left_target"] and not self.facts["right_target"]) or self.facts["separation_m"]>.028:
            self.grasp_latched = False
        self.grasp_streak = self.grasp_streak + 1 if held else 0
        if self.grasp_streak >= 3:
            self.last_secure_grasp = self.steps
        elif not self.grasp_latched and not self.slip_injected:
            self.last_secure_grasp = -1000
        if self.slip_injected and self.before_slip_held and not self.facts["left_target"] and not self.facts["right_target"]:
            self.slip_lost = True
        if self.slip_lost and self.release_left==0 and self.grasp_streak>=3 and self.command_width<.025:
            self.regrasped = True
        if self.free:
            # A sidewall can peel one pad off a carried body. Keep ownership
            # across that frame, and across short contact-solver dropouts while
            # the hand still commands a closed grasp around the payload.
            touches_pad = self.facts["left_target"] or self.facts["right_target"]
            closed_near = (self.command_width+.003 < self.facts["width_m"] and
                           self.facts["separation_m"]<.08)
            carrying = held or (self.payload_owned and (touches_pad or closed_near))
            if carrying != self.payload_owned:
                self._pairs(carrying)
        step_pen = max(self.scan_penetration(),rb.last_substep_max_pen_m if self.entry_ok else 0.)
        self.max_pen_m = max(self.max_pen_m,step_pen)
        error = self._target_distance()
        reward = 45.*(self.progress_error-error)+3.*np.clip(old_gap-self.facts["separation_m"],-.015,.015)
        reward -= .004 + .08*min(5.,max(0.,step_pen-.001)*1000.)
        self.progress_error = error
        self.good_steps = self.good_steps+1 if self._goal() else 0
        counts = self.reward_counts
        counts["goal_entries"] += int(self.good_steps==1)
        counts["goal_dwell_steps"] += int(self.good_steps>0)
        counts["slips_observed"] += int(self.slip_lost and not had_loss)
        counts["reacquisitions"] += int(self.regrasped and not had_regrasp)
        if self.good_steps:
            reward += .055
        if forced:
            self.slip_peak_gap = max(self.slip_peak_gap,self.facts["separation_m"])
        if self.slip_lost and not forced and not had_regrasp and self.requested_width>.03:
            self.slip_policy_opened = True
        recovered_now = (not self.causal_recovered and self.regrasped and self.slip_policy_opened and held
                         and not forced and self.requested_width<.025
                         and self.slip_peak_gap-self.facts["separation_m"]>=.005)
        if recovered_now:
            self.causal_recovered = True
            counts["causal_recoveries"] += 1
        if self.reward_mode=="centered" and not self.evaluation:
            center,speed = self._centered_state()
            reward += .55*(center-self.center_potential)
            self.center_potential = center
            if self.good_steps:
                counts["slow_goal_steps"] += int(speed<1.)
                reward += .025*center*np.exp(-.5*speed**2)*min(1.,self.good_steps/10.)
            if recovered_now:
                reward += .5
                counts["recovery_bonuses"] += 1
        self._observe()
        done = self.steps>=self.horizon or not self.entry_ok
        if done:
            reward += 3.0 if self.good_steps>=10 else -.5
        if (not self.evaluation and not self._priming and not done and not self.episode_contact_saved
                and self.steps<self.horizon-15 and (self.grasp_streak>=3 or self.facts["push_face_loaded"])):
            self.episode_contact_saved = True
            shot = self.snapshot()
            shot["bank_source"] = "on_policy"
            self.buffer = (self.buffer+[shot])[-16:]
        return self.obs,float(reward),bool(done),self._info()

    def expert_action(self):
        """Privileged geometric proposal for collecting training demonstrations.

        No task-specific thresholds enter deployed actor inference or evaluation.
        """
        if not self.entry_ok:
            return np.zeros(8,np.float32)
        R,p = self.fast_rb.eef_pose()
        c = self.contract
        if c.family == "planar_push":
            return self._planar_teacher_action(R,p)
        if c.family == "place" and self.preset == "all_5_families":
            return self._place_teacher_action(R,p)
        if (self.preset == "all_5_families" and self.hinge
                and abs(self.axis @ self.part.face_normal) < .5):
            return self._swing_teacher_action(R, p)
        h = self.p_contact()
        error = self._target_distance()
        angular = self.progress()[1]
        Rtarget = rotvec_to_mat(self.axis*angular)@self.contact_R if self.hinge else self.contact_R
        direction = rotvec_to_mat(self.axis*angular)@self.direction if self.hinge else self.direction
        linear, angular = self.progress()
        if self.hinge:
            direction *= np.sign((c.target_rot_rad-angular)*c.target_rot_rad)
        elif self.part is not None:
            direction *= np.sign((c.target_trans_m-linear)*c.target_trans_m)
        release = self.contract.want_held_end < 0 and (
            abs(c.target_rot_rad-angular)<.75*c.rot_tolerance if self.hinge else error<.005)
        push = self.surface_push
        engaged = self.grasp_latched
        if self.hinge and self.preset != "smoke_door_drawer":
            touch = self.facts["left_target"] or self.facts["right_target"]
            # A pad against an eccentric projection can drive its arc even
            # while the two contact normals do not oppose each other.
            engaged |= touch and self.facts["separation_m"] < .020 and self.command_width < .025
        if release:
            target = h-self.approach*.055
            width = .08
        elif c.family=="pick" and self.grasp_latched:
            # A small grasp-load reserve above the one-sided minimum offsets
            # the static tracking error of the compliant lifting hand.
            target = p+self.direction*min(.028,max(0.,c.target_trans_m+.003-linear))
            if self.preset == "all_5_families" and linear < .004 and self.grasp_streak < 6:
                # Let both pads seat before loading the grip with the payload's
                # moment. A single first contact can still be sliding into place.
                target = p.copy()
            width = 0.
        elif c.family == "place" and self.facts["firm"]:
            goal_pos = self.reference + self.Rref @ c.goal_region_ref
            rem_vec = goal_pos - self.target_pos()
            horiz_rem = np.linalg.norm(rem_vec[:2])
            if abs(self.direction[2]) < 0.8 and horiz_rem > 0.024:
                # Slide horizontally clear of upper overhang / sill before lowering to floor.
                rem_vec = rem_vec.copy()
                rem_vec[2] = max(rem_vec[2], 0.012)
            elif abs(self.direction[2]) >= 0.8 and horiz_rem > 0.018:
                # Align over open-top caddy slot before descending between narrow walls.
                rem_vec = np.r_[rem_vec[:2], max(rem_vec[2], 0.0)]
            target = p + rem_vec
            width = 0.
        elif not push and not engaged:
            target = (h - self.contact_R[:, 2]*np.mean(self.jaw_geometry["pad_z"])
                      if self.free else h)
            gap = np.linalg.norm(target-p)
            width = .08 if gap>.012 else 0.
        else:
            # Contact tracking with look-ahead small enough to stay on a hinge.
            if self.hinge:
                # Move both the gripper and its orientation along the same
                # reference arc. A lagging wrist otherwise peels the inside pad
                # off the rail, even while the target keeps moving.
                lookahead = np.clip(c.target_rot_rad-angular, -.19, .19)
                future = rotvec_to_mat(self.axis*lookahead)
                target = self.anchor + future@(h-self.anchor)
                if push:
                    if self.preset == "smoke_door_drawer":
                        inward = -rotvec_to_mat(self.axis*angular)@self.part.face_normal
                    else:
                        inward = -self.d.xmat[self.bid].reshape(3, 3) @ self.part.face_normal_b
                    target = target + .015 * inward
                Rtarget = future@Rtarget
            else:
                target = h+direction*min(.028,error+.003)
            width = .009 if push else 0.
        # A pinch on a rail permits relative roll about its long axis (local
        # hand y). Transport the contact point on the entire arc while yielding
        # half the wrist rotation about that axis. Turning a part instead
        # requires transmitting the full angular motion through the grasp.
        if (self.hinge and c.family=="push_pull" and self.grasp_latched
                and abs(self.axis@self.contact_R[:,1])>.85):
            ahead = angular + np.clip(c.target_rot_rad-angular,-.19,.19)
            Rtarget = rotvec_to_mat(self.axis*(.5*ahead))@self.contact_R
        rv = R.T@mat_to_rotvec(Rtarget@R.T)/np.deg2rad(4.5)
        action = np.clip(np.r_[R.T@(target-p)/.008,rv,1.,width/.04-1.],-1.,1.).astype(np.float32)
        if c.family=="pick" and self.grasp_latched and error<.022:
            # Cancel any lead left in the compliant position integrator once
            # the one-sided requested lift has been attained.
            action[:3] = np.clip(R.T@(target-self.core.p_des)/.008,-1.,1.)
        if self.hinge:
            self._brake_teacher_rotation(action,R,p,h,angular)
            if self.preset != "smoke_door_drawer":
                self._regulate_turn_rate(action, R, h, angular)
        return action

    def _swing_teacher_action(self, rotation, tcp):
        """Transport an offset contact on the measured arc, then unload the pads."""
        angle = self.progress()[1]
        error = self.contract.target_rot_rad - angle
        rate = self.d.qvel[self.ss._adr(self.part.joint)[1]]
        body = self.d.xmat[self.bid].reshape(3, 3)
        pose = rotvec_to_mat(self.axis * angle) @ self.contact_R
        closing = self.contract.target_rot_rad * (self.part.open-self.part.closed) < 0.
        if closing:
            # The hint may stand in front of a panel. Apply pressure to the
            # actual face through closure, even when the goal is close.
            point = self.target_pos() + body @ (self.part.face_point_b - .012*self.part.face_normal_b)
            ahead = np.clip(error-.14*rate, -.21, .21)
            target = self.anchor + rotvec_to_mat(self.axis*ahead) @ (point-self.anchor)
            width = .011
        else:
            point = self.p_contact()
            if self.facts["firm"]:
                self.teacher_contact_acquired = True
            if not (self.facts["left_target"] or self.facts["right_target"]) and np.linalg.norm(tcp-point) > .017:
                self.teacher_contact_acquired = False
            ahead = np.clip(error-.13*rate, -.26, .26)
            if self.teacher_contact_acquired:
                # Carry the load-induced pad offset around a certified arc.
                # A straight or in-place entry only guarantees the nominal site.
                contact = point
                if self.facts["firm"] and self.entry_hypothesis in ("arc", "axis_arc_roll"):
                    load_fraction = np.linalg.norm(np.cross(self.axis, UP))
                    contact = point + .7*load_fraction*(tcp-point)
                target = self.anchor + rotvec_to_mat(self.axis*ahead) @ (contact-self.anchor)
                width = 0.
            else:
                target = point + .003*pose[:, 2]
                width = .08 if np.linalg.norm(tcp-target) > .015 else 0.
            if abs(error) < .54*self.contract.rot_tolerance and abs(rate) < .27:
                if not self.teacher_release_pending:
                    self.teacher_release_width = self.fast_rb.finger_opening()
                self.teacher_release_pending = True
            elif abs(error) > .8*self.contract.rot_tolerance:
                self.teacher_release_pending = False
            # Match wrist transport to the motion admitted by entry. A
            # straight passage keeps the forearm level; an unobstructed arc
            # carries both opposed pads with the part. A rail with rolling
            # clearance lets the fingers yield some pitch under load.
            if self.entry_hypothesis == "arc":
                follow = 1.
            elif self.entry_hypothesis == "axis_arc_roll":
                follow = .75
            elif self.entry_hypothesis == "prismatic":
                follow = 0.
            else:
                follow = float(self.axis @ UP)**2
            lead = ahead if self.teacher_contact_acquired and not self.teacher_release_pending else 0.
            pose = rotvec_to_mat(self.axis*follow*(angle+lead)) @ self.contact_R
            if self.teacher_release_pending:
                width = .08
                target = point - .04*pose[:, 2]
                # A withdrawing finger still loads a vertical hinge until the
                # measured aperture clears the section at the current pose.
                if (abs(self.axis @ UP) > .5 and
                        self.fast_rb.finger_opening() < self.teacher_release_width + .012):
                    target = point
                if abs(self.axis @ UP) < .5:
                    # Leave clearance around the actual loaded grasp, including
                    # oblique contacts that span more than the nominal section.
                    width = min(.08, max(.008 + 2*min(self.part.handle_half),
                                         self.teacher_release_width + .006))
                    target = point
        translation = rotation.T @ (target-tcp) / .008
        if (not closing and self.entry_hypothesis != "arc" and self.teacher_contact_acquired
                and not self.teacher_release_pending and abs(error) > .20):
            # A wrist allowed to roll cannot transmit the full rigid-arc moment.
            # Aim its available Cartesian lead along the requested pull, accounting
            # for the lead that the compliant controller already stores.
            requested = target-tcp
            cap = self.core.pos_cap(1.)
            requested *= min(1., cap/max(1e-10, np.linalg.norm(requested)))
            step = requested - self.core.retention*(self.core.p_des-tcp)
            translation = rotation.T @ step / max(.008, np.linalg.norm(step))
        if closing:
            # Saturate the Cartesian step along the requested pressure direction.
            translation /= max(1., np.linalg.norm(translation))
        turn = rotation.T @ mat_to_rotvec(pose @ rotation.T) / np.deg2rad(4.5)
        return np.clip(np.r_[translation, turn, 1., width/.04-1.], -1., 1.).astype(np.float32)

    def _place_teacher_action(self, R, p):
        """Carry to supported contact, open at rest, then clear the object."""
        goal = self.reference + self.Rref @ self.contract.goal_region_ref
        delta = goal - self.target_pos()
        horizontal = np.linalg.norm(delta[:2])
        contacts = self.d.contact.geom[:self.d.ncon]
        on_floor = np.any((self.target_g[contacts[:, 0]] & self.support_g[contacts[:, 1]]) |
                          (self.target_g[contacts[:, 1]] & self.support_g[contacts[:, 0]]))
        vertical = self.d.xmat[self.bid].reshape(3, 3)[:, 2]
        # Keep the payload level throughout transport so its high side does
        # not catch the ceiling while the pads advance into the opening.
        level = self.teacher_place_release is None
        correction = rotvec_to_mat(np.cross(vertical, UP)) if level else IDENTITY
        upright = vertical[2] > np.cos(.29)
        if self.teacher_place_release is None and on_floor and horizontal < .016 and upright:
            # Latch the measured release TCP and supported grasp width. Later
            # slips or estimates outside the region must not reclose the hand.
            self.teacher_place_release = (p.copy(), self.fast_rb.finger_opening(), self.steps)
        if self.teacher_place_release is None:
            # Follow the same collision-certified payload corridor used at
            # entry. Project onto its nearest segment and advance without
            # cutting across a corner under a roof or above a support lip.
            path = getattr(self, "place_path", None)
            if path is not None and np.linalg.norm(delta) > .012:
                segments = np.diff(path, axis=0)
                lengths = np.linalg.norm(segments, axis=1)
                unit = segments / np.maximum(lengths[:, None], 1e-8)
                along = np.clip(np.sum((self.target_pos() - path[:-1])*unit, axis=1), 0., lengths)
                dist = np.linalg.norm(self.target_pos() - path[:-1] - along[:, None]*unit, axis=1)
                i = np.argmin(dist)
                arc = np.r_[0., np.cumsum(lengths)]
                ahead = min(arc[-1], arc[i] + along[i] + .025)
                waypoint = np.array([np.interp(ahead, arc, col) for col in path.T])
                delta = waypoint - self.target_pos()
            else:
                delta[2] -= .003
            target = p + delta + (correction - IDENTITY) @ (p - self.target_pos())
            width = 0.
        else:
            release_pos, held_width, release_step = self.teacher_place_release
            ticks = self.steps - release_step
            width = min(.080, held_width + .026)
            # The inside fingertip needs longer to clear a rim than a simple
            # two-sided grasp. Let the measured opening settle before lifting.
            if ticks < 8:
                target = release_pos
            elif -self.approach[2] < .65 and self.fixture.sill_height < .005:
                # Slide out of the roofed cavity along the entry orientation.
                target = np.asarray(release_pos) - .10 * self.approach
            else:
                # Withdraw through the open top or lift the inside fingertip
                # above the rim before travelling outward over a drawer sill.
                height = goal[2] + self.fixture.payload_half_h + .025
                if p[2] < height:
                    target = np.r_[release_pos[:2], height + .005]
                else:
                    target = self.p0 + .020 * UP
                    width = .080
        translation = R.T @ (target - self.core.p_des) / .008
        if self.teacher_place_release is not None:
            # Returning to the entry roll here would hook the freshly levelled rim.
            target_R = self.core.R_des
        else:
            target_R = correction @ R
        rotation = R.T @ mat_to_rotvec(target_R @ self.core.R_des.T) / np.deg2rad(4.5)
        return np.clip(np.r_[translation, rotation, 1., width/.04 - 1.], -1., 1.).astype(np.float32)

    def _planar_teacher_action(self,R,p):
        """Push below the payload center, facing the remaining in-plane error."""
        up = self.Rref[:,2]
        center = self.target_pos()
        delta = self.reference+self.Rref@self.contract.goal_region_ref-center
        delta -= up*(up@delta)
        distance = np.linalg.norm(delta)
        lateral = delta - self.direction*(self.direction @ delta)
        direction = self.direction
        rear = self._push_edge(center, direction)
        height = max(rear[2] - self.contact_R[2, 2] * np.mean(self.jaw_geometry["pad_z"]),
                     self.reference[2] + self.jaw_geometry["tip_z"] + .0025)
        # Shift the load across the rear face to cancel lateral drift. Keep the
        # stroke tangent fixed so the jaw never sweeps around a convex corner.
        # A shallow exposed sidewall offers little purchase for steering.
        # Bound the cross-track motion so it stays inside the loaded footprint.
        axes = self.d.xmat[self.bid].reshape(3, 3)
        top = center[2] + max((axes @ c)[2] + np.abs(axes[2]) @ h
                              for _, _, _, _, c, h in self.surfaces)
        cross_step = 3.*lateral
        cross_limit = np.clip(top - self.reference[2], .012, .040)
        target = rear + cross_step * min(1., cross_limit/max(np.linalg.norm(cross_step), 1e-9))
        # Initial clearance is for the open approach; during contact the lower
        # pad must engage the skin without rolling over the upper edge.
        height = max(self.reference[2] + self.jaw_geometry["tip_z"] + .001,
                     getattr(self, "push_tcp_height", height) - .005)
        shallow = getattr(self, "shallow_push_entry", False)
        if shallow:
            height = self.push_tcp_height
            target[:2] -= self.push_contact_offset[:2]
        target[2] = height
        if distance < self.contract.trans_tolerance - .002:
            target = p + .040*up - .018*direction
        elif p[2] > height + .009:
            target -= (.010 + self.jaw_geometry["finger_y"])*direction
        else:
            target += np.clip(delta @ direction - .008, -.020, .032)*direction
        if (shallow and distance >= self.contract.trans_tolerance - .002 and
                (self.facts["left_target"] or self.facts["right_target"])):
            # The projected contact face can move relative to the TCP as the
            # load rotates. Maintain pressure from the measured tool position
            # and steer by the remaining cross-track displacement.
            normal_step = np.clip(delta @ direction, .006, .017)
            target += max(0., normal_step - (target-p) @ direction)*direction
            side = np.cross(up, direction)
            side_step = np.clip(.32*(delta @ side), -.009, .009)
            target += (side_step - (target-p) @ side)*side
        step = target - self.core.p_des
        translation = R.T @ step / max(.008, np.linalg.norm(step))
        rotation = R.T @ mat_to_rotvec(self.contact_R @ self.core.R_des.T) / np.deg2rad(4.5)
        return np.clip(np.r_[translation, rotation, 1., -.66], -1., 1.).astype(np.float32)

    def counterfactual_expert(self):
        """An optional second imitation label for the exact same training observation.

        Only the commanded angular header changes. Its original live depth
        registration error is preserved in the counterfactual remaining angle.
        The shallow view shares read-only simulation/sensor data; the new
        contract and the teacher's release latch are privately owned.
        """
        if self.preset == "all_5_families":
            return self._joint_counterfactual()
        if (self.evaluation or not self.entry_ok or not self.hinge or self.contract.family!="push_pull"
                or self.horizon-self.steps<20):
            return None
        actual = np.rad2deg(self.progress()[1])
        original = self.contract.target_rot_rad
        sign = np.sign(original)
        if not 25. <= actual * sign <= 55.:
            return None
        goals = sign * np.deg2rad([40., 75.])
        if self.preset == "smoke_door_drawer" and np.min(np.abs(goals-original)) > 1e-6:
            return None
        lo, hi = sorted(self.part.range)
        eligible = [goal for goal in goals if abs(goal-original) > 1e-6
                    and lo <= self.initial_joint+goal <= hi]
        if not eligible:
            return None
        alternate = max(eligible, key=lambda goal: abs(goal-original))
        header = self.obs["goal"].copy()
        header[6] += alternate-original
        view = copy.copy(self)
        view.contract = copy.copy(self.contract)
        view.contract.target_rot_rad = float(alternate)
        view.teacher_release_pending = False
        return header,view.expert_action()

    def _joint_counterfactual(self):
        if self.evaluation or not self.entry_ok or self.part is None or self.horizon-self.steps < 15:
            return None
        contract = self.contract
        field = "target_rot_rad" if self.hinge else "target_trans_m"
        original = getattr(contract, field)
        sign = np.sign(original)
        if sign == 0. or (not self.hinge and self.part.handle_point_b is None):
            return None
        actual = self.progress()[int(self.hinge)] * sign
        tolerance = contract.rot_tolerance if self.hinge else contract.trans_tolerance
        if actual < 2. * tolerance:
            return None
        limit = max((q - self.initial_joint) * sign for q in self.part.range)
        # At a meaningful decision state, pair reaching a nearby stop with
        # continuing toward the far end. Keep both directives physically legal.
        far = min(limit, np.deg2rad(85.)) if self.hinge else limit
        if actual > far or far-actual < 2. * tolerance:
            return None
        short = actual
        if self.hinge and self.contract.family == "turn" and short < np.deg2rad(25.):
            return None
        alternate = sign * (short if abs(abs(original)-short) >= 2.*tolerance else far)
        if abs(alternate-original) < 2.*tolerance:
            return None
        view = copy.copy(self)
        view.contract = copy.copy(contract)
        setattr(view.contract, field, float(alternate))
        view.teacher_release_pending = False
        view.teacher_load, view.teacher_load_step = None, -100
        # The actor's real tracker error remains present; only the supplied
        # command delta changes. No alternative rollout is stepped or rendered.
        header = self.obs["goal"].copy()
        header[6 if self.hinge else 5] += alternate - original
        return header, view.expert_action()

    def _regulate_turn_rate(self, action, rotation, contact, angle):
        """Recover a slowing grip without accumulating unsupported wrist rotation."""
        pad_loaded = (self.facts["left_target"] or self.facts["right_target"]) and self.command_width < .025
        if self.teacher_release_pending or action[7] > .8 or not (self.grasp_latched or self.sensed_hold or pad_loaded):
            self.teacher_load, self.teacher_load_step = None, -100
            return
        error = self.contract.target_rot_rad - angle
        distance = abs(error)
        tolerance = self.contract.rot_tolerance
        direction = np.sign(error)
        speed = direction * self.d.qvel[self.ss._adr(self.part.joint)[1]]
        frames_left = self.horizon - self.steps
        # Reserve two intervals to open the fingers and ten to settle at the goal.
        drive_time = max(DT, (frames_left - 12)*DT)
        if distance < .98*tolerance:
            coast = max(0., speed)*min(.12, DT*frames_left)
            if distance > coast and (distance - .6*tolerance > speed*drive_time or frames_left < 20):
                self.teacher_release_pending = True
                action[:3] = np.clip(rotation.T @ (contact-self.core.p_des)/.008, -1., 1.)
                action[3:6] = 0.
                action[7] = 1.
            return
        arm = np.linalg.norm(np.cross(self.axis, contact - self.anchor))
        if arm > .5*self.fast_rb.finger_opening() + .008:
            # An eccentric contact already has a usable tangential moment arm.
            # Keep following its arc. Any extra angular command must fit inside
            # the available wrist-to-shaft displacement, including stored lead.
            required = max(0., (distance-.75*tolerance)/drive_time)
            deficit = required-max(speed, 0.)
            if deficit > .01:
                twist = self.axis @ mat_to_rotvec(rotation @ self.contact_R.T) - angle
                lead = self.core.retention * (self.axis @ mat_to_rotvec(self.core.R_des @ rotation.T))
                scale = np.deg2rad(4.5)
                omega = rotation @ (action[3:6]/max(1., np.linalg.norm(action[3:6])))
                budget = .07 - direction*(twist + lead + scale*(self.axis @ omega))
                extra = min(.24, 2.*deficit, max(0., budget))
                action[3:6] = np.clip(action[3:6]+rotation.T @ (self.axis*direction*extra)/scale, -1., 1.)
            # A single pad can push against the wrong side of an offset lever.
            # Keep the current arc while measured travel can still reach the goal.
            if speed > .02 and speed*drive_time > .85*max(0., distance-.055):
                return
        remaining = (frames_left - 15)*DT
        if remaining > .01 and max(0., speed)*remaining > 1.15*max(0., distance-.055):
            self.teacher_load = None
            return
        if self.turn_rollout is None:
            self.turn_rollout = GripRollout(self)
        if self.teacher_load is None or self.steps-self.teacher_load_step >= 4:
            self.teacher_load = self.turn_rollout.select(self, action)
            self.teacher_load_step = self.steps
        if self.teacher_load is not None:
            action[:] = self.turn_rollout.drive(self, self.fast_rb, self.core, self.teacher_load)

    def _brake_teacher_rotation(self,action,R,p,contact,angle):
        """Damp angular motion near the requested goal before opening the pads."""
        c = self.contract
        error = c.target_rot_rad-angle
        velocity = float(self.d.qvel[self.ss._adr(self.part.joint)[1]])
        if (self.preset != "smoke_door_drawer" and self.teacher_release_pending
                and abs(error) > .9*c.rot_tolerance and error*velocity < 0.):
            self.teacher_release_pending = False
        # When the contact is close to the axis its arc supplies little
        # translational leverage. Servo the wrist by measured angular error
        # while keeping the pinch centred, including the contact settling time.
        arm = np.linalg.norm(np.cross(self.axis, contact - self.anchor))
        if self.preset != "smoke_door_drawer" and arm < .5*self.fast_rb.finger_opening() + .008:
            if self.teacher_release_pending:
                action[7] = 1.
                action[3:6] = 0.
                target = contact - .055*self.contact_R[:, 2]
            elif self.grasp_latched or self.sensed_hold:
                # With a short moment arm, bias the normal load towards one
                # pad and apply tangential force there. Keeping the opposite
                # pad in contact transmits torque without excessive wrist lead.
                jaw = _unit(R[:, 0] - self.axis*(self.axis @ R[:, 0]))
                tangent = np.cross(self.axis, -jaw)
                drive = np.clip(error/.09, -1., 1.)
                # A stuck joint needs more tangential load, while a joint that
                # turns ahead of the wrist already has enough. Regulate contact
                # forces with the teacher's measured relative angle.
                hand_angle = self.axis @ mat_to_rotvec(R @ self.contact_R.T)
                twist = (hand_angle-angle)*np.sign(error)
                load = np.clip((twist + .004)/.047, 0., 1.)
                settling = abs(error) < .09 and twist < .035
                target = contact if settling else contact + load*(.012*jaw + drive*.015*tangent)
                if self.sensed_hold or self.facts["firm"]:
                    # At a stall excessive squeeze balances both pad loads,
                    # cancelling the applied tangential couple. Keep a measured
                    # closure obstruction and increase grip reserve at speed.
                    speed = max(0., velocity*np.sign(error))
                    compression = np.clip(.009 + .08*speed, .009, .026)
                    width = max(0., self.fast_rb.finger_opening() - compression)
                    action[7] = np.clip(width/.04 - 1., -1., 1.)
                else:
                    action[7] = -1.
                reference = c.target_rot_rad if settling else angle + np.clip(error, -.075, .075)
                Rtarget = rotvec_to_mat(self.axis*reference) @ self.contact_R
                rv = mat_to_rotvec(Rtarget @ self.core.R_des.T)
                action[3:6] = np.clip(R.T@rv/np.deg2rad(4.5), -1., 1.)
                # Opening takes a control interval. Release as the measured
                # travel enters its stopping window, including a fast crossing.
                if abs(error - DT*velocity) < .65*c.rot_tolerance:
                    self.teacher_release_pending = True
                    action[7] = 1.
                    action[3:6] = 0.
            else:
                return
            action[:3] = np.clip(R.T@(target-self.core.p_des)/.008, -1., 1.)
            return
        if abs(error)>.15:
            self.teacher_release_pending = False
        if abs(error)>=.23 or not (self.grasp_latched or self.teacher_release_pending):
            return
        if abs(error)<np.deg2rad(3.4) and abs(velocity)<np.deg2rad(13.):
            self.teacher_release_pending = True
        if self.teacher_release_pending:
            touching = self.facts["left_target"] or self.facts["right_target"]
            target = p if touching else contact-.048*R[:,2]
            Rtarget = R
            action[7] = 1.
        else:
            lead = np.clip(error-.12*velocity,-.15,.15)
            target = self.anchor+rotvec_to_mat(self.axis*lead)@(contact-self.anchor)
            can_roll = c.family=="push_pull" and abs(self.axis@self.contact_R[:,1])>.85
            Rtarget = rotvec_to_mat(self.axis*((.5 if can_roll else 1.)*(angle+lead)))@self.contact_R
            action[7] = -1.
        action[:3] = np.clip(R.T@(target-self.core.p_des)/.008,-1.,1.)
        action[3:6] = np.clip(R.T@mat_to_rotvec(Rtarget@self.core.R_des.T)/np.deg2rad(4.5),-1.,1.)
        # The pads take a control interval to unload. A fast motion can
        # traverse the entire settling band before the next position sample.
        if (self.preset != "smoke_door_drawer" and self.grasp_latched and
                abs(error - DT*velocity) < .53*c.rot_tolerance):
            self.teacher_release_pending = True
            action[7] = 1.
            action[3:6] = 0.

    _state_fields = ("contract","reference","Rref","direction","axis","anchor","initial_joint","intended_initial_joint","hint_local",
        "hint_world","q_initial_world","approach","contact_R","entry_ok","command_width","requested_width",
        "policy_reference","R0","p0","core","entry_hypothesis","entry_arc_margin",
        "observer","noise","sensor","frozen_payload","history","health","obs","tracker","sensed_hold","grip_observer","grip_site0_world",
        "action","last_width","facts","steps","good_steps","release_left","grasp_streak","slip_injected",
        "slip_lost","regrasped","last_secure_grasp","grasp_latched","before_slip_held","slip_at_deg","max_pen_m","progress_error","max_travel",
        "episode_contact_saved","payload_owned","teacher_release_pending","staged","staged_origin",
        "reward_counts","center_potential","slip_peak_gap","slip_policy_opened","causal_recovered")

    _binding_fields = ("part_joint_name", "bid", "hinge", "free", "closing", "body_target",
                       "target_g", "grasp_g", "support_g")

    def snapshot(self):
        self.fast_rb.sync_to_robosuite()
        fields = self._state_fields
        if self.preset == "all_5_families":
            fields += self._binding_fields
            fields += ("teacher_contact_acquired", "teacher_release_width")
            fields += ("teacher_load", "teacher_load_step")
            if self.free and self.surface_push and hasattr(self, "push_tcp_height"):
                fields += ("push_tcp_height", "push_contact_offset", "shallow_push_entry")
            if self.kind == "place":
                fields += ("teacher_place_release", "place_jitter_status", "place_grasp_transform")
                if hasattr(self, "place_path"):
                    fields += ("place_path",)
        return dict(layout=self.key,sim=self.ss.snapshot(),rng=copy.deepcopy(self.rng.bit_generator.state),
                    state=copy.deepcopy({k:getattr(self,k) for k in fields}),
                    gains=(self.fast_rb._kp.copy(), self.fast_rb._kd.copy()),
                    penetration_pairs=self.pairs.copy(),last_substep_pen=self.fast_rb.last_substep_max_pen_m)

    def restore(self, snap, target_deg=None, slip_at_deg=None, *, target_trans_m=None):
        if target_trans_m is not None and (self.preset != "all_5_families" or snap["layout"][0] != "drawer"):
            raise ValueError("A translational override requires an all-five drawer")
        if target_deg is not None and snap["layout"][0] not in ("door", "turn"):
            raise ValueError("An angular override needs a rotational target")
        if self.key!=tuple(snap["layout"]):
            self._load(*snap["layout"])
            self._setup_contract(None)
        self.ss.restore(snap["sim"])
        self.teacher_contact_acquired = False
        self.teacher_release_width = 0.
        self.teacher_place_release = None
        self.turn_rollout = None
        self.teacher_load, self.teacher_load_step = None, -100
        self.place_path = None
        self.place_jitter_status = "off"
        self.place_grasp_transform = None
        self.__dict__.pop("push_tcp_height", None)
        self.shallow_push_entry = False
        self.push_contact_offset = np.zeros(3)
        for k,v in copy.deepcopy(snap["state"]).items():
            setattr(self,k,v)
        self.ss.robot.ctrl.kp[:], self.ss.robot.ctrl.kd[:] = snap["gains"]
        self.fast_rb.sync_from_robosuite()
        if self.preset == "all_5_families":
            # Part instances belong to the loaded simulator, even for an identical layout.
            if (not hasattr(self.tracker, "legacy")
                    or (not self.tracker.axis.any() and not hasattr(self.tracker, "reference_surfaces"))):
                self.tracker = ReferenceDepthTracker.from_older_snapshot(self.tracker)
                self.grip_observer.legacy = False
            target = self.contract.target_entity
            self.part = next((p for p in self.ss.parts if p.body == target), None)
            if self.ss.parts and self.part is None:
                raise ValueError(f"Snapshot target {target!r} is missing in layout {self.key}")
            self.part_joint_name = self.part.joint if self.part is not None else None
            if snap["state"].get("part_joint_name", self.part_joint_name) != self.part_joint_name:
                raise ValueError(f"Snapshot joint disagrees with target body {target!r}")
            if "body_target" not in snap["state"]:  # Snapshots from before binding masks were stored
                self.bid = mujoco.mj_name2id(self.m, _B, target)
                self.free = self.part is None
                self.hinge = self.part is not None and self.part.kind == "hinge"
                self._set_target_masks(self.p_contact())
        pairs = snap["penetration_pairs"]
        if pairs.shape != (self.m.ngeom, self.m.ngeom):
            # Older layouts have the same bodies and DOFs but fewer collision
            # geoms on the support face. Rebind IDs using the restored contract.
            self._set_target_masks(self.p_contact())
            self._pairs(self.payload_owned)
        else:
            self.pairs = pairs.copy()
            self.fast_rb.set_penetration_pairs(self.pairs)
        self.fast_rb.last_substep_max_pen_m = snap["last_substep_pen"]
        self.rng.bit_generator.state = copy.deepcopy(snap["rng"])
        if target_deg is not None or target_trans_m is not None:
            angle = target_deg is not None
            delta = float(np.deg2rad(target_deg) if angle else target_trans_m)
            field = "target_rot_rad" if angle else "target_trans_m"
            previous = getattr(self.contract, field)
            if self.preset == "all_5_families":
                _joint_target(self.part, self.initial_joint, delta)
                if (self.contract.family == "push_pull" and self.part.handle_point_b is None
                        and delta * (self.part.open-self.part.closed) > 0):
                    raise ValueError("Outward travel requires a handle")
                if np.sign(delta) != np.sign(previous):
                    self._reverse_joint_goal(delta)
                if delta != previous:
                    self.good_steps = 0
                    self.teacher_release_pending = False
                    self.teacher_load, self.teacher_load_step = None, -100
            setattr(self.contract, field, delta)
            self.progress_error = self._target_distance()
            self.center_potential = self._centered_state()[0] if self.reward_mode=="centered" else 0.
            self.obs["goal"] = self._header()
        if slip_at_deg is not None:
            self.slip_at_deg = slip_at_deg
        return self.obs,self._info()

    def _reverse_joint_goal(self, delta):
        # The observed, reachable contact patch is part of this handoff. Swap
        # direction about that same patch, without repainting the point cloud.
        part = self.part
        if self.hinge:
            tangent = np.cross(self.axis, self.hint_world-self.anchor)
        else:
            tangent = self.axis
        self.direction = np.sign(delta) * _unit(tangent)
        self.contract.goal_dir_ref = self.Rref.T @ self.direction
        if self.contract.family == "push_pull":
            if self.surface_push:
                self.closing = _unit(np.cross(UP, self.approach))
            else:
                rotation = rotvec_to_mat(-self.axis*self.progress()[1]) if self.hinge else IDENTITY
                grip_axis = part.face_u if part.handle_axis is None else part.handle_axis
                self.closing = rotation @ _unit(np.cross(part.face_normal, grip_axis))
            if not self.closing.any():
                self.closing = IDENTITY[:, 1].copy()
        self._set_target_masks(self.p_contact())
        self._pairs(self.payload_owned)

    def render_rgb(self):
        """Same physics state and camera as the depth view, in literal RGB."""
        cam = self.cameras
        cam.gl.make_current()
        cam.cam.fixedcamid = cam.ids[0]
        mujoco.mjv_updateScene(self.m,self.d,cam.opt,cam.pert,cam.cam,mujoco.mjtCatBit.mjCAT_ALL,cam.scn)
        seg,idcolor = mujoco.mjtRndFlag.mjRND_SEGMENT,mujoco.mjtRndFlag.mjRND_IDCOLOR
        cam.scn.flags[seg] = cam.scn.flags[idcolor] = 0
        try:
            mujoco.mjr_render(cam.rect,cam.scn,cam.con)
            mujoco.mjr_readPixels(cam.rgb,None,cam.rect,cam.con)
            return np.ascontiguousarray(cam.rgb[::-1])
        finally:
            cam.scn.flags[seg] = cam.scn.flags[idcolor] = 1

    def close(self):
        if self.cameras is not None:
            self.cameras.close()
            self.cameras = None
        if self.ss is not None:
            self.ss.close()
            self.ss = None
