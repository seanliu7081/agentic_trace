"""Training scenes of LRL_PLAN.md 6.2, the parts that move: generated mechanisms (MJCF from primitives, added to a
table-only LIBERO layout through the model XML) and LIBERO fixtures placed at random through generated BDDL files.

Frames. A generated mechanism is one static body whose origin lies on the table top under the centre of its front;
its +x is the outward normal of that front, +z is up. A Part keeps its face in the frame of its own body, so the face
moves with the part; place_parts writes the world values at the current joint values.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

OUT = Path("/workspace/agentic_trace/runs/lrl_m0_20261125/sim")
ROBOT = ("robot0_", "gripper0_", "mount0_")
_B, _J = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT
_SIGNS = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)

CFG = {
    # arena and robot, read from the loaded kitchen-table model (runs/lrl_m0_20261125/sim/measure_fixtures.py)
    "table_z": 0.90,                   # M  top of the kitchen table
    "base_xy": (-0.66, 0.0),           # M  Panda base axis
    "base_keepout": 0.25,              # T  m: no generated footprint corner nearer than this to the base axis
    "home_strip": ((-0.66, -0.10), 0.13),   # M  m: x range and half width of the table area under the arm at home
    "home_clear": 0.17,                # T  m: a mechanism taller than this stays out of that area (home hand: 0.26)
    # placement, LRL_PLAN.md 6.2: on the table within reach, yaw +-60 deg of facing the robot, random height
    "mount_r": (0.42, 0.65),           # T  m: face centre, distance from the base axis
    "mount_bearing": np.deg2rad(35),   # T  rad: face centre, bearing from the base's forward direction
    "mount_yaw": np.deg2rad(60),       # T  rad: face normal against the direction back to the base
    "plinth": (0.0, 0.12),             # T  m: height of the solid block under a mechanism
    "fixture_r": (0.50, 0.75),         # T  m: origin of a LIBERO fixture from the base axis (its front stands nearer)
    "fixture_bearing": np.deg2rad(30),  # T rad
    "distractors": (0, 3),             # T  loose objects in a fixture layout
    "distractor_gap": (0.30, 0.12),    # T  m: from the fixture origin, from one another
    "distractor_xy": ((-0.25, 0.20), (-0.38, 0.38)),   # T  m: table area they are dropped in
    # shared build settings of the generated mechanisms
    "panel_t": 0.016,                  # T  m: thickness of fronts and door panels
    "wall_t": 0.012,                   # T  m: carcass walls
    "gap": 0.003,                      # T  m: clearance between a moving part and its carcass (they never touch)
    "density": (300.0, 900.0),         # T  kg/m3: mass randomization
    "handle": ("bar", "knob", "lip", "none"),
    "protrusion": (0.012, 0.04),       # T  m: handles ("none" protrudes 0)
    "bar_len": (0.3, 0.7),             # T  a bar's length as a share of the front's extent along it
    "bar_t": 0.005,                    # T  m: half thickness of bars and posts
    "pull_r": (0.008, 0.018),          # T  m: radius of a knob-shaped handle
    "box_t": 0.008,                    # T  m: walls of a drawer's box
    "stay_in": 0.03,                   # T  m: of a drawer's box inside the carcass at full travel
    "door_inset": 0.006,               # T  m: the body behind a door is this much smaller than the panel on each side
    # drawer stacks
    "drawers": (1, 4), "front_w": (0.15, 0.45), "front_h": (0.06, 0.20),          # T
    "stack_h": 0.50,                   # T  m: fronts shrink so that a stack stays within the arm's height
    "travel": (0.08, 0.35),            # T  m
    "overhang": (0.0, 0.03),           # T  m: carcass top (and, half the time, sides) beyond the fronts
    "slide_damping": (2.0, 60.0),      # T  N s/m (LIBERO's cabinets: 50)
    "slide_friction": (0.0, 8.0),      # T  N
    # doors
    "door_size": (0.15, 0.45),         # T  m, each side
    "door_range": np.deg2rad((60.0, 150.0)),   # T
    "door_depth": (0.15, 0.30),        # T  m: body behind the door
    "hinge_damping": (0.2, 2.0),       # T  N m s/rad
    "hinge_friction": (0.0, 1.0),      # T  N m
    # heavy, stiff doors (a fridge door: tall, dense, a stiff hinge); the LIBERO short_fridge itself stays unseen
    "heavy_door": 0.35,                # T  share of generated doors that are heavy
    "heavy_size": (0.30, 0.55),        # T  m, each side
    "heavy_density": (900.0, 2500.0),  # T  kg/m3
    "heavy_damping": (2.0, 15.0),      # T  N m s/rad
    "heavy_friction": (0.3, 3.0),      # T  N m
    "flap_hold": (1.1, 1.6),           # T  friction of a horizontal hinge as a multiple of the panel's gravity torque
    # knobs
    "knob_dia": (0.02, 0.06), "knob_h": (0.015, 0.04),                            # T  m
    "knob_range": np.deg2rad((60.0, 300.0)),   # T
    "knob_damping": (0.02, 0.5),       # T  N m s/rad
    "knob_friction": (0.0, 0.15),      # T  N m
    "ridge_h": 0.012,                  # T  m: grip bar across half of the knobs
    "knob_block_front": ((0.10, 0.08, 0.08), (0.30, 0.20, 0.20)),   # T  m: block behind a knob, width height depth
    "knob_block_top": ((0.15, 0.03, 0.15), (0.30, 0.12, 0.30)),     # T  m: block under a knob
    # sliding panels
    "panel_w": (0.15, 0.35), "panel_h": (0.15, 0.40), "panel_travel": (0.08, 0.30),   # T  m
    # a sliding panel is pushed on the side of its handle; every generated panel has one (the edge of a 16 mm panel
    # flush in front of its body had no reachable pose in 4 of 5 trials and closed 8-14 % with every method:
    # runs/lrl_m1_20261125/train/REPORT.md, M0 REFERENCE.md failure table)
    "slider_t": (0.016, 0.035),        # T  m: panel thickness
    "slider_handle": ("bar", "knob", "lip"),   # T
    # handles of LIBERO fixtures, read from the ray grid: cells standing this far out of the face level
    "handle_tol": 0.004,               # T  m
    "grip_depth": 0.006,               # T  m: the pads' centre this far below the handle's outer surface
    # reading a LIBERO fixture's parts from the model
    "ray_grid": (11, 41),              # T  rays per side: choosing the exposed side, then measuring the face
    "ray_start": 0.10,                 # T  m: rays start this far outside the fixture, the room a hand needs
    "level_bin": 0.002, "level_tol": 0.003,    # T  m: the face is the fullest height bin of the exposed surface
    # handle styles per generator (the M1 sets; E2E below widens them)
    "door_handle": ("bar", "knob", "none"),    # T
    # M1E handle kinds (E2E below): a low bar near the bottom edge of a front, a recess (a slot through the front
    # near its top edge, nothing proud; the fingers reach into the drawer behind it)
    "low_bar_z": (0.012, 0.035),       # T  m: bar centre above the front's bottom edge (KITCHEN_SCENE1's wooden
                                       #    cabinet: bottom bar 4-5 cm above the table, LRL_PLAN.md M1E)
    "low_bar_p": (0.012, 0.032),       # T  m: how far a low bar stands out: from nearly flush (a 1 cm rail with
                                       #    no room behind it, held only by pinching it) to LIBERO's 3.1 cm
                                       #    (read_parts on the wooden and white cabinets; room to hook behind)
    "slot_w": (0.06, 0.14),            # T  m: width of a recess
    "slot_h": (0.02, 0.035),           # T  m: its height, at most slot_share of the front's
    "slot_top": (0.01, 0.025),         # T  m: the strip of front above it
    "slot_share": 0.4,                 # T
    "low_handle": 0.07,                # T  m: a LIBERO fixture's bar whose centre is lower than this above the
                                       #    fixture's lowest point is a low bar (handle_kind, for evaluation)
}

# LIBERO fixtures: joint value when closed (off) and open (on), from envs/objects/articulated_objects.py and the
# asset XMLs (T); "front": yaw of the fixture's outward side in its own frame (M, measure_fixtures.py).
FIXTURES = {
    "wooden_cabinet": {"closed": 0.0, "open": -0.16, "front": -1.571},
    "white_cabinet": {"closed": 0.0, "open": -0.16, "front": -1.571},
    "microwave": {"closed": 0.0, "open": -2.094, "front": -1.571},
    "flat_stove": {"closed": 0.0, "open": 2.1, "front": 3.14},
}
# No LIBERO task uses these: kept out of training, loaded for the gates only (LRL_PLAN.md 6.2). "seat": LIBERO puts
# the fixture 5 to 13 cm deep into the table (M, measure_fixtures.py); StageSim lifts it to stand on the table.
UNSEEN = {
    "short_cabinet": {"closed": 0.0, "open": 0.16, "front": -1.571},
    "short_fridge": {"closed": 0.0, "open": 2.7, "front": -1.571},
    "slide_cabinet": {"closed": 0.0, "open": 0.44, "front": -1.571, "seat": True},
    "window": {"closed": 0.0, "open": 0.2, "front": 1.571, "seat": True},
    "faucet": {"closed": 0.0, "open": 1.57, "front": 1.571, "seat": True},
    "basin_faucet": {"closed": 0.0, "open": 1.5, "front": 0.0, "seat": True},
}
FIXTURES.update(UNSEEN)
DISTRACTORS = ("akita_black_bowl", "plate", "butter", "cream_cheese", "ketchup", "chocolate_pudding",
               "alphabet_soup", "tomato_sauce", "milk", "orange_juice")
GENERATED = ("drawer_stack", "door", "knob", "sliding_panel")
# M1E: the generated mechanisms with the handle kinds of LRL_PLAN.md M1E (low bars, recesses, lips; a countertop over
# the top drawer; knobs one grasp can turn end to end). New kinds, appended, so M1's kinds keep their draws and seeds.
E2E = CFG | {
    "handle": ("bar", "low_bar", "low_bar", "recess", "knob", "lip", "none"),  # T  low bars twice: G1E asks for them
    "overhang": (0.0, 0.08),                   # T  m: up to a countertop standing out over the top drawer
    "door_handle": ("bar", "knob", "lip", "none"),     # T
    "knob_range": np.deg2rad((45.0, 150.0)),   # T  a wrist turn of about 330 deg covers it from any start posture
    # distances from the robot: the faces of M1's kinds stand 0.38-0.65 m from the base axis, where the start region
    # (10-20 cm out) folds the elbow to 0.1-0.3 rad of its limit and an opening drawer runs into the robot
    # (e1/NOTES.md); KITCHEN_SCENE1's wooden cabinet stands 0.73 m from the base (its bddl)
    "mount_r": (0.52, 0.75),                   # T  m: face centre of a generated mechanism
    "fixture_r": (0.62, 0.82),                 # T  m: origin of a LIBERO fixture
    "open_reach": 0.42,                        # T  m: a drawer's handle at its open end, a door's free edge half
                                               #    open, stay this far from the base axis
    "travel": (0.08, 0.30),                    # T  m: drawers (M1: 8-35 cm)
    # fronts turned up to 90 deg from facing the robot (e2 request 5: the head camera stands opposite the robot, so
    # a front that faces the robot is seen from behind; 55 % of e2's loop episodes had >= 20 part points in the head
    # view, drawers 28 %); the LIBERO dev cabinets face sideways. A generated front is kept only where the point
    # 15 cm out from it lies in reach_start of the base axis (where the start region of a sideways front can be)
    "mount_yaw": np.deg2rad(90),               # T
    "reach_start": (0.30, 0.80),               # T  m
}
E2E_GENERATED = {"drawer_e2e": lambda rng: drawer_stack(rng, cfg=E2E), "door_e2e": lambda rng: door(rng, cfg=E2E),
                 "knob_e2e": lambda rng: knob(rng, cfg=E2E)}
KINDS = GENERATED + tuple(FIXTURES) + tuple(E2E_GENERATED)


@dataclass
class Part:
    name: str
    joint: str
    kind: str                       # "slide" | "hinge"
    body: str
    fixture_bodies: tuple           # the other bodies of the same fixture
    range: tuple                    # joint range, m or rad
    closed: float
    open: float
    half_extents: tuple             # of the face, along face_u and along face_normal x face_u
    protrusion: float               # how far a handle stands out beyond the face
    face_point_b: np.ndarray        # the face in the part body's frame: centre, outward normal, in-plane axis
    face_normal_b: np.ndarray
    face_u_b: np.ndarray
    goal: float = None              # joint value an episode drives towards (StageSim.part_travel); default: closed
    # the handle in the part body's frame: the centre the pads close on, its long axis (in the face plane), half
    # length along that axis and half width across it; None: no handle to hold. For a knob that turns about its
    # face normal the handle is the knob itself, centred on the axis.
    handle_point_b: np.ndarray = None
    handle_axis_b: np.ndarray = None
    handle_half: tuple = (0.0, 0.0)
    handle_kind: str = ""           # bar, low_bar, recess, knob, lip, none, rotary (a turning knob): evaluation only
    # world values at the current joint values, written by place_parts
    axis: np.ndarray = None         # joint axis, unit; the joint value grows along (slide) or about (hinge) it
    axis_point: np.ndarray = None   # a point on a hinge's axis
    face_point: np.ndarray = None
    face_normal: np.ndarray = None
    face_u: np.ndarray = None
    handle_point: np.ndarray = None
    handle_axis: np.ndarray = None

    def __post_init__(self):
        if self.goal is None:
            self.goal = self.closed


@dataclass
class Layout:
    bddl: str
    mjcf: list                      # generated <body> fragments, appended to the worldbody
    parts: list                     # Parts of the generated mechanisms
    seed: int
    kind: str = ""
    fixtures: dict = field(default_factory=dict)   # LIBERO fixture instance -> category; StageSim reads their Parts


def _name(m, t, i):
    return mujoco.mj_id2name(m, t, i) or ""


def place_parts(parts, m, d):
    """Write the world fields of the Parts from the model at its current joint values."""
    for p in parts:
        b, j = mujoco.mj_name2id(m, _B, p.body), mujoco.mj_name2id(m, _J, p.joint)
        R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
        p.axis, p.axis_point = d.xaxis[j].copy(), d.xanchor[j].copy()
        p.face_point, p.face_normal, p.face_u = x + R @ p.face_point_b, R @ p.face_normal_b, R @ p.face_u_b
        if p.handle_point_b is not None:
            p.handle_point, p.handle_axis = x + R @ p.handle_point_b, R @ p.handle_axis_b
    return parts


def turns(part):
    """A part that turns about the normal of its own face: a knob (turn mode), not a door or a drawer."""
    return part.kind == "hinge" and abs(part.axis @ part.face_normal) > 0.7


# ------------------------------------------------------------------------------------------ generated mechanisms
def _v(a):
    return " ".join(f"{x:.5f}" for x in np.atleast_1d(a))


def _box(name, lo, hi, rgba, density):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (f'<geom name="{name}" type="box" pos="{_v((lo + hi) / 2)}" size="{_v((hi - lo) / 2)}" '
            f'rgba="{rgba}" density="{density:.0f}"/>')


def _handle(rng, name, style, p, at, along, span, density):
    """Handle geoms on a face whose outward normal is +x, standing out by p: `at` its centre on the face, `along` the
    in-plane axis of a bar or lip (1 = y, 2 = z), `span` the length available along it. Returns the XML and the
    handle as Part keeps it (point, axis, half sizes; point None without a handle)."""
    at, x, e = np.asarray(at, float), np.eye(3)[0], np.eye(3)[along]
    w, dark = np.cross(x, e), "0.2 0.2 0.2 1"
    grip = at + (p - min(CFG["grip_depth"], p / 2)) * x

    def box(tag, centre, half):
        return _box(f"{name}_{tag}", centre - np.abs(half), centre + np.abs(half), dark, density)

    if style in ("bar", "low_bar"):     # a rail on two posts
        L, t = span * rng.uniform(*CFG["bar_len"]), CFG["bar_t"]
        post = (p / 2 - t) * x + t * (e + w)
        return (box("bar", at + (p - t) * x, t * (x + w) + L / 2 * e)
                + "".join(box(f"post{k}", at + (p / 2 - t) * x + s * (L / 2 - 2 * t) * e, post)
                          for k, s in enumerate((-1, 1)))), (at + (p - t) * x, e, (L / 2, t))
    if style == "knob":
        r = rng.uniform(*CFG["pull_r"])
        return (f'<geom name="{name}_knob" type="cylinder" fromto="{_v(at)} {_v(at + p * x)}" '
                f'size="{r:.4f}" rgba="{dark}" density="{density:.0f}"/>'), (grip, e, (r, r))
    if style == "lip":              # a thin plate standing out along the whole grip length
        return (box("lip", at + p / 2 * x, p / 2 * x + 0.4 * span * e + CFG["box_t"] / 2 * w),
                (grip, e, (0.4 * span, CFG["box_t"] / 2)))
    return "", (None, e, (0.0, 0.0))


def _mount(rng, x_back, y_half, height, cfg=CFG, r_lo=None):
    """Pose of a mechanism's root: face centre within reach, normal within mount_yaw of facing the robot, the footprint
    (x from x_back to 0, |y| <= y_half in the root frame) clear of the robot's base and, when the mechanism is tall,
    of the space the arm takes at home, where every episode of a real scene starts. r_lo: a nearer bound of the face
    centre's distance from the base (a part that opens towards the robot needs room to open)."""
    base, ((sx0, sx1), sy) = np.array(CFG["base_xy"]), CFG["home_strip"]
    foot = np.stack(np.meshgrid(np.linspace(x_back, 0, 9), np.linspace(-y_half, y_half, 9)), -1).reshape(-1, 2)
    lo, hi = cfg["mount_r"]
    r_range = (lo, hi) if r_lo is None else (min(max(r_lo, lo), hi), hi)
    while True:
        b = rng.uniform(-1, 1) * CFG["mount_bearing"]
        xy = base + rng.uniform(*r_range) * np.array([np.cos(b), np.sin(b)])
        yaw = b + np.pi + rng.uniform(-1, 1) * cfg["mount_yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        pts = xy + foot @ np.array([[c, s], [-s, c]])
        under_arm = (pts[:, 0] > sx0) & (pts[:, 0] < sx1) & (np.abs(pts[:, 1]) < sy)
        out = np.linalg.norm(xy + 0.15 * np.array([c, s]) - base)
        if np.linalg.norm(pts - base, axis=1).min() > CFG["base_keepout"] and not (
                height > CFG["home_clear"] and under_arm.any()) and (
                cfg is CFG or cfg["reach_start"][0] <= out <= cfg["reach_start"][1]):
            return f'pos="{_v([*xy, CFG["table_z"]])}" quat="{_v([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])}"'


def _carcass(name, lo, hi, cavity, density, z0):
    """Body behind a moving front: a solid block, or five walls around a cavity open towards +x; on its plinth."""
    lo, hi, t = np.asarray(lo, float), np.asarray(hi, float), CFG["wall_t"]
    rgba = "0.75 0.7 0.6 1"
    # no plinth when _v would write its half height as zero: MuJoCo refuses a box of zero size, and one such draw
    # in about 10,000 layouts ended a training run
    out = _box(f"{name}_plinth", [lo[0], lo[1], 0], [hi[0], hi[1], z0], rgba, density) if round(z0 / 2, 5) > 0 else ""
    if not cavity:
        return out + _box(f"{name}_solid", lo, hi, rgba, density)
    walls = {"back": (lo, [lo[0] + t, hi[1], hi[2]]), "bottom": (lo, [hi[0], hi[1], lo[2] + t]),
             "top": ([lo[0], lo[1], hi[2] - t], hi), "side0": (lo, [hi[0], lo[1] + t, hi[2]]),
             "side1": ([lo[0], hi[1] - t, lo[2]], hi)}
    return out + "".join(_box(f"{name}_{k}", a, b, rgba, density) for k, (a, b) in walls.items())


def _finish(xml, parts):
    """Compile the fragment alone (which also checks it) and place the Parts in the world."""
    m = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{xml}</worldbody></mujoco>")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return xml, place_parts(parts, m, d)


def _part(name, i, kind, others, rng_, closed_open, he, p, face, normal=(1, 0, 0), u=(0, 1, 0), handle=None,
          handle_kind=""):
    point, axis, half = handle if handle is not None else (None, None, (0.0, 0.0))
    return Part(f"{name}_p{i}", f"{name}_j{i}", kind, f"{name}_p{i}", tuple(others), tuple(rng_), *closed_open,
                tuple(he), float(p), np.asarray(face, float), np.asarray(normal, float), np.asarray(u, float),
                handle_point_b=None if point is None else np.asarray(point, float),
                handle_axis_b=None if point is None else np.asarray(axis, float), handle_half=tuple(half),
                handle_kind=handle_kind)


def _slotted(name, lo, hi, slot, rgba, density):
    """A front panel (x from lo[0] to hi[0], face at x = hi[0]) with a rectangular slot through it: slot is
    ((y0, y1), (z0, z1)). Four boxes around the slot."""
    (y0, y1), (z0, z1) = slot
    (xa, ya, za), (xb, yb, zb) = lo, hi
    pieces = {"below": ([xa, ya, za], [xb, yb, z0]), "above": ([xa, ya, z1], [xb, yb, zb]),
              "left": ([xa, ya, z0], [xb, y0, z1]), "right": ([xa, y1, z0], [xb, yb, z1])}
    return "".join(_box(f"{name}_{k}", a, b, rgba, density) for k, (a, b) in pieces.items())


def drawer_stack(rng, name="lrl_drawers", cfg=CFG):
    """1 to 4 drawers in a carcass, each on a slide joint along the outward normal; value 0 closed, travel open.
    cfg: the generator's settings (CFG for M1; E2E adds low bars and recesses). With CFG the draws are M1's."""
    tp, tw, c = cfg["panel_t"], cfg["wall_t"], cfg["gap"]
    n = int(rng.integers(cfg["drawers"][0], cfg["drawers"][1] + 1))
    W = rng.uniform(*cfg["front_w"])
    H = rng.uniform(cfg["front_h"][0], min(cfg["front_h"][1], cfg["stack_h"] / n))
    T, o = round(rng.uniform(*cfg["travel"]), 4), rng.uniform(*cfg["overhang"])   # T as the XML will state it
    z0, rho = rng.uniform(*cfg["plinth"]), rng.uniform(*cfg["density"])
    style = str(rng.choice(cfg["handle"]))
    p = rng.uniform(*cfg["low_bar_p" if style == "low_bar" else "protrusion"]) if style not in ("none", "recess") \
        else 0.0
    side_o = o if rng.random() < 0.5 else 0.0
    L, bt = T + cfg["stay_in"], cfg["box_t"]        # the box behind the front
    xb, yh, pitch = -(tp + L + c + tw), W / 2 + c + tw, H + 2 * c + tw
    top = z0 + n * pitch + tw
    rgba = "0.75 0.7 0.6 1"
    g = [_box(f"{name}_plinth", [xb, -yh, 0], [0, yh, z0], rgba, rho) if z0 > 0 else "",
         _box(f"{name}_back", [xb, -yh, z0], [xb + tw, yh, top], rgba, rho),
         _box(f"{name}_top", [xb, -yh, top - tw], [o, yh, top], rgba, rho)]
    g += [_box(f"{name}_side{k}", [xb, s * yh - (s > 0) * tw, z0], [side_o, s * yh + (s < 0) * tw, top], rgba, rho)
          for k, s in enumerate((-1, 1))]
    g += [_box(f"{name}_shelf{k}", [xb, -yh, z0 + k * pitch], [0, yh, z0 + k * pitch + tw], rgba, rho)
          for k in range(n)]
    parts, kids = [], ""
    yb, hs = W / 2 - 1.5 * bt, 0.35 * H             # box narrower than the front, its walls lower
    for i in range(n):
        dn = f"{name}_p{i}"
        if style == "low_bar":
            dz = -H / 2 + rng.uniform(*cfg["low_bar_z"])
        else:
            dz = rng.uniform(-0.2, 0.2) * H if style in ("bar", "knob") else (H - bt) / 2 if style == "lip" else 0.0
        zb = -H / 2 + bt / 2                        # the bottom plate, clear of the shelf under it
        lo, hi, wood = [-tp, -W / 2, -H / 2], [0, W / 2, H / 2], "0.6 0.45 0.3 1"
        if style == "recess":
            # a slot through the front under its top strip; the drawer's box walls stay lower, so fingers reach in
            sw, sh = rng.uniform(*cfg["slot_w"]), min(rng.uniform(*cfg["slot_h"]), cfg["slot_share"] * H)
            st = min(rng.uniform(*cfg["slot_top"]), cfg["slot_share"] * H)
            front = _slotted(f"{dn}_front", lo, hi, ((-sw / 2, sw / 2), (H / 2 - st - sh, H / 2 - st)), wood, rho)
            hx, handle = "", (np.array([-tp / 2, 0.0, H / 2 - st / 2]), np.array([0.0, 1.0, 0.0]), (sw / 2, st / 2))
        else:
            front = _box(f"{dn}_front", lo, hi, wood, rho)
            hx, handle = None, None
        box = [front, _box(f"{dn}_bottom", [-tp - L, -yb, zb], [-tp, yb, zb + bt], rgba, rho),
               _box(f"{dn}_rear", [-tp - L, -yb, zb + bt], [-tp - L + bt, yb, hs], rgba, rho)]
        box += [_box(f"{dn}_wall{k}", [-tp - L, s * yb - (s > 0) * bt, zb + bt], [-tp, s * yb + (s < 0) * bt, hs],
                     rgba, rho) for k, s in enumerate((-1, 1))]
        if hx is None:
            hx, handle = _handle(rng, dn, style, p, [0, 0, dz], 1, W, rho)
        kids += (f'<body name="{dn}" pos="0 0 {z0 + tw + c + H / 2 + i * pitch:.5f}">'
                 f'<joint name="{name}_j{i}" type="slide" axis="1 0 0" limited="true" range="0 {T:.4f}" '
                 f'damping="{rng.uniform(*cfg["slide_damping"]):.2f}" '
                 f'frictionloss="{rng.uniform(*cfg["slide_friction"]):.2f}"/>'
                 + "".join(box) + hx + "</body>")
        others = [name] + [f"{name}_p{k}" for k in range(n) if k != i]
        parts.append(_part(name, i, "slide", others, (0.0, T), (0.0, T), (W / 2, H / 2), p, [0, 0, 0],
                           handle=handle, handle_kind=style))
    r_lo = None if cfg is CFG else cfg["open_reach"] + T        # the open drawer's handle stays within reach
    return _finish(f'<body name="{name}" {_mount(rng, xb, yh, top, cfg, r_lo)}>{"".join(g)}{kids}</body>', parts)


def door(rng, name="lrl_door", cfg=CFG):
    """A panel on a hinge along one of its edges (vertical: left or right; horizontal: top or bottom) in front of a
    cavity or a solid body; value 0 closed, the free edge swings outward as the value grows. With cfg = CFG the
    draws are M1's."""
    tp, c = cfg["panel_t"], cfg["gap"]
    heavy = rng.random() < cfg["heavy_door"]
    W, H = rng.uniform(*cfg["heavy_size" if heavy else "door_size"], 2)
    D, th = rng.uniform(*cfg["door_depth"]), round(rng.uniform(*cfg["door_range"]), 4)
    z0, rho = rng.uniform(*cfg["plinth"]), rng.uniform(*cfg["heavy_density" if heavy else "density"])
    edge = "vertical" if heavy else str(rng.choice(("vertical", "horizontal")))
    s = float(rng.choice((-1, 1)))
    style = str(rng.choice(cfg["door_handle"]))
    p = rng.uniform(*cfg["protrusion"]) if style != "none" else 0.0
    zc, i = z0 + H / 2, cfg["door_inset"]
    # the body is smaller than the panel, so a door swung past 90 deg does not meet its edge
    body = _carcass(name, [-tp - c - D, -W / 2 + i, z0], [-tp - c, W / 2 - i, z0 + H - i], rng.random() < 0.5, rho, z0)
    if edge == "vertical":          # hinge at y = s W/2; the panel reaches to y = -s W/2
        pos, axis, off, along, span = [-tp / 2, s * W / 2, zc], [0, 0, s], np.array([0, -s * W / 2, 0]), 2, H
        damp = rng.uniform(*cfg["heavy_damping" if heavy else "hinge_damping"])
        fric = rng.uniform(*cfg["heavy_friction" if heavy else "hinge_friction"])
    else:                           # hinge at z = zc + s H/2; a friction hinge, or the panel would fall by itself
        pos, axis, off, along, span = [-tp / 2, 0, zc + s * H / 2], [0, -s, 0], np.array([0, 0, -s * H / 2]), 1, W
        damp = rng.uniform(*cfg["hinge_damping"])
        fric = rng.uniform(*cfg["flap_hold"]) * rho * tp * W * H * 9.81 * H / 2
    # The same physical hinge can use either signed joint coordinate.
    convention = cfg.get("hinge_coordinate_sign", 1)
    if convention < 0:
        axis = -np.asarray(axis)
    travel = convention * th
    limits = (min(0., travel), max(0., travel))
    range_text = f"{travel:.4f} 0" if convention < 0 else f"0 {travel:.4f}"
    h = np.array([tp / 2, W / 2, H / 2])
    hx, handle = _handle(rng, f"{name}_p0", style, p, off * 1.6 + [tp / 2, 0, 0], along, span, rho)
    r_lo = None if cfg is CFG else cfg["open_reach"] + 0.5 * W
    xml = (f'<body name="{name}" {_mount(rng, -tp - c - D, W / 2, z0 + H, cfg, r_lo)}>{body}'
           f'<body name="{name}_p0" pos="{_v(pos)}">'
           f'<joint name="{name}_j0" type="hinge" axis="{_v(axis)}" limited="true" range="{range_text}" '
           f'damping="{damp:.3f}" frictionloss="{fric:.3f}"/>'
           + _box(f"{name}_panel", off - h, off + h, "0.6 0.45 0.3 1", rho) + hx + "</body></body>")
    return _finish(xml, [_part(name, 0, "hinge", [name], limits, (0.0, travel), (W / 2, H / 2), p,
                               off + [tp / 2, 0, 0], handle=handle, handle_kind=style)])


def knob(rng, name="lrl_knob", cfg=CFG):
    """A knob turning about the normal of the surface it is mounted on: the front of a block, or its top. With
    cfg = CFG the draws are M1's."""
    r, h = rng.uniform(*cfg["knob_dia"]) / 2, rng.uniform(*cfg["knob_h"])
    th, rho = round(rng.uniform(*cfg["knob_range"]), 4), rng.uniform(*cfg["density"])
    ridge = cfg["ridge_h"] if rng.random() < 0.5 else 0.0
    if rng.random() < 0.5:          # front of a block on a plinth
        z0, (W, H, D) = rng.uniform(*cfg["plinth"]), rng.uniform(*cfg["knob_block_front"])
        lo, hi, n, u = [-D, -W / 2, z0], [0, W / 2, z0 + H], np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        pos = [0, rng.uniform(-1, 1) * (W / 2 - r - 0.01), z0 + H / 2]
    else:                           # top of a low block, near its front edge
        z0, (W, H, D) = 0.0, rng.uniform(*cfg["knob_block_top"])
        lo, hi, n, u = [-D, -W / 2, 0], [0, W / 2, H], np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
        pos = [-r - 0.02 - rng.uniform(0, 0.05), rng.uniform(-1, 1) * (W / 2 - r - 0.01), H]
    v, pos = np.cross(n, u), pos + n * cfg["gap"]
    geoms = (f'<geom name="{name}_cap" type="cylinder" fromto="0 0 0 {_v(n * h)}" size="{r:.4f}" '
             f'rgba="0.2 0.2 0.2 1" density="{rho:.0f}"/>')
    if ridge:
        hb = 0.9 * r * u + 0.005 * v
        geoms += _box(f"{name}_ridge", n * h - hb, n * (h + ridge) + hb, "0.8 0.8 0.8 1", rho)
    xml = (f'<body name="{name}" {_mount(rng, -D, W / 2, z0 + H + h, cfg)}>' + _carcass(name, lo, hi, False, rho, z0)
           + f'<body name="{name}_p0" pos="{_v(pos)}">'
           f'<joint name="{name}_j0" type="hinge" axis="{_v(n)}" limited="true" range="0 {th:.4f}" '
           f'damping="{rng.uniform(*cfg["knob_damping"]):.3f}" '
           f'frictionloss="{rng.uniform(*cfg["knob_friction"]):.3f}"/>{geoms}</body></body>')
    return _finish(xml, [_part(name, 0, "hinge", [name], (0.0, th), (0.0, th), (r, r), ridge, n * h, n, u,
                               handle=(n * h / 2, u, (r, r)), handle_kind="rotary")])


def sliding_panel(rng, name="lrl_slider"):
    """A panel that travels parallel to its own face, in front of a body wide enough for the whole travel."""
    c = CFG["gap"]
    W, H = rng.uniform(*CFG["panel_w"]), rng.uniform(*CFG["panel_h"])
    T = round(rng.uniform(CFG["panel_travel"][0], min(CFG["panel_travel"][1], W)), 4)
    D, s = rng.uniform(*CFG["door_depth"]), float(rng.choice((-1, 1)))
    z0, rho = rng.uniform(*CFG["plinth"]), rng.uniform(*CFG["density"])
    tp = round(rng.uniform(*CFG["slider_t"]), 4)
    style = str(rng.choice(CFG["slider_handle"]))
    p = rng.uniform(*CFG["protrusion"])
    yh, h = (W + T) / 2, np.array([tp / 2, W / 2, H / 2])
    body = _carcass(name, [-tp - c - D, -yh, z0], [-tp - c, yh, z0 + H + 0.01], rng.random() < 0.5, rho, z0)
    hx, handle = _handle(rng, f"{name}_p0", style, p, [tp / 2, rng.uniform(-0.35, 0.35) * W, 0], 2, H, rho)
    xml = (f'<body name="{name}" {_mount(rng, -tp - c - D, yh, z0 + H + 0.01)}>{body}'
           f'<body name="{name}_p0" pos="{_v([-tp / 2, -s * T / 2, z0 + 0.005 + H / 2])}">'
           f'<joint name="{name}_j0" type="slide" axis="0 {s:.0f} 0" limited="true" range="0 {T:.4f}" '
           f'damping="{rng.uniform(*CFG["slide_damping"]):.2f}" '
           f'frictionloss="{rng.uniform(*CFG["slide_friction"]):.2f}"/>'
           + _box(f"{name}_panel", -h, h, "0.6 0.45 0.3 1", rho) + hx + "</body></body>")
    return _finish(xml, [_part(name, 0, "slide", [name], (0.0, T), (0.0, T), (W / 2, H / 2), p, [tp / 2, 0, 0],
                               handle=handle, handle_kind=style)])


# ---------------------------------------------------------------------------------------- LIBERO layouts by BDDL
def write_bddl(path, fixtures=(), objects=()):
    """A kitchen-table layout: fixtures and objects as (category, x, y, yaw), table coordinates. Instances are named
    <category>_<k>. With neither, the table alone, which LIBERO accepts: the carrier of generated mechanisms."""
    regions, names, init, count = "", {False: {}, True: {}}, "", {}
    for is_obj, items in ((False, fixtures), (True, objects)):
        for cat, x, y, yaw in items:
            count[cat] = count.get(cat, 0) + 1
            inst = f"{cat}_{count[cat]}"
            names[is_obj].setdefault(cat, []).append(inst)
            regions += (f"      ({inst}_init_region\n          (:target kitchen_table)\n          (:ranges (\n"
                        f"              ({x - 0.005:.4f} {y - 0.005:.4f} {x + 0.005:.4f} {y + 0.005:.4f})\n"
                        f"            )\n          )\n          (:yaw_rotation (\n              ({yaw:.5f} {yaw:.5f})\n"
                        f"            )\n          )\n      )\n")
            init += f"    (On {inst} kitchen_table_{inst}_init_region)\n"
    decl = {k: "".join(f"    {' '.join(v)} - {c}\n" for c, v in d.items()) for k, d in names.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}")     # written whole, then renamed: other processes share it
    tmp.write_text("(define (problem LIBERO_Kitchen_Tabletop_Manipulation)\n  (:domain robosuite)\n"
                    f"  (:language lrl stage layout)\n    (:regions\n{regions}    )\n\n"
                    f"  (:fixtures\n    kitchen_table - kitchen_table\n{decl[False]}  )\n\n"
                    f"  (:objects\n{decl[True]}  )\n\n  (:obj_of_interest\n  )\n\n  (:init\n{init}  )\n\n"
                    "  (:goal\n    (And )\n  )\n\n)\n")
    tmp.replace(path)
    return str(path)


def base_layout(out_dir=OUT / "bddl"):
    return write_bddl(Path(out_dir) / "table_only.bddl")


def fixture_layout(rng, category, path, cfg=CFG):
    """One LIBERO fixture at a random pose on the kitchen table, its front within mount_yaw of facing the robot, and a
    few loose objects around it."""
    base = np.array(CFG["base_xy"])
    b = rng.uniform(-1, 1) * CFG["fixture_bearing"]
    xy = base + rng.uniform(*cfg["fixture_r"]) * np.array([np.cos(b), np.sin(b)])
    yaw = b + np.pi + rng.uniform(-1, 1) * cfg["mount_yaw"] - FIXTURES[category]["front"]
    spots = [xy]
    (x0, x1), (y0, y1) = CFG["distractor_xy"]
    objects = []
    for cat in rng.choice(DISTRACTORS, int(rng.integers(CFG["distractors"][0], CFG["distractors"][1] + 1)), False):
        while True:
            q = rng.uniform([x0, y0], [x1, y1])
            gaps = np.linalg.norm(np.array(spots) - q, axis=1)
            if gaps[0] > CFG["distractor_gap"][0] and (gaps[1:] > CFG["distractor_gap"][1]).all():
                break
        spots.append(q)
        objects.append((str(cat), q[0], q[1], 0.0))
    return write_bddl(path, [(category, xy[0], xy[1], yaw % (2 * np.pi))], objects)


def seat_fixture(m, d, instance):
    """Lift a loaded fixture so that its lowest colliding geometry stands on the table (LIBERO places fixtures the
    same way: through the model's body position)."""
    names = [_name(m, _B, b) for b in range(m.nbody)]
    fixture = np.array([nm.startswith(instance + "_") for nm in names])
    solid = ((m.geom_contype > 0) | (m.geom_conaffinity > 0)) & fixture[m.geom_bodyid]
    m.body_pos[int(np.flatnonzero(fixture)[0]), 2] += CFG["table_z"] - _corners(m, d, np.flatnonzero(solid))[:, 2].min()
    mujoco.mj_forward(m, d)


def _subtree(m, root):
    par, inside = m.body_parentid, np.zeros(m.nbody, bool)
    inside[root] = True
    for b in range(root + 1, m.nbody):              # MuJoCo numbers a child after its parent
        inside[b] = inside[par[b]]
    return inside


def _corners(m, d, geoms):
    """World corners (G*8, 3) of the geoms' bounding boxes."""
    loc = m.geom_aabb[geoms, None, :3] + m.geom_aabb[geoms, None, 3:] * _SIGNS[None]
    R = d.geom_xmat[geoms].reshape(-1, 3, 3)
    return (d.geom_xpos[geoms, None] + np.einsum("gij,gkj->gki", R, loc)).reshape(-1, 3)


def _cast(m, d, own, pts, h0, n, u, v, G):
    """A G x G grid of rays along -n over the points' extent, started at height h0 along n. Returns the grid
    coordinates along u and v, whether each ray's first hit is on a body flagged in `own`, and the hit heights."""
    a, b = pts @ u, pts @ v
    gu = a.min() + (np.arange(G) + 0.5) / G * np.ptp(a)
    gv = b.min() + (np.arange(G) + 0.5) / G * np.ptp(b)
    gid, group = np.zeros(1, np.int32), np.array([1, 0, 0, 0, 0, 0], np.uint8)
    hit, h = np.zeros((G, G), bool), np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            dist = mujoco.mj_ray(m, d, gu[i] * u + gv[j] * v + h0 * n, -n, group, 1, -1, gid)
            hit[i, j] = dist >= 0 and own[m.geom_bodyid[gid[0]]]
            h[i, j] = h0 - dist
    return gu, gv, hit, h


def _read_handle(gu, gv, hit, h, level, u, v, n):
    """The handle on a measured face: the cells that stand out of the face level; centre, long axis (principal
    direction of the cells in the face plane) and half sizes. (None, None, (0, 0)) when fewer than two cells do."""
    cells = hit & (h > level + CFG["handle_tol"])
    if cells.sum() < 2:
        return None, None, (0.0, 0.0)
    i, k = np.nonzero(cells)
    xy = np.stack([gu[i], gv[k]], 1)
    c = xy.mean(0)
    w, vec = np.linalg.eigh(np.cov((xy - c).T) + 1e-12 * np.eye(2))
    e2 = vec[:, 1]
    along, across = (xy - c) @ e2, (xy - c) @ vec[:, 0]
    cell = 0.5 * (gu[1] - gu[0])
    top = h[cells].max()
    point = c[0] * u + c[1] * v + (top - min(CFG["grip_depth"], (top - level) / 2)) * n
    axis = e2[0] * u + e2[1] * v
    return point, axis / np.linalg.norm(axis), (float(np.ptp(along) / 2 + cell), float(np.ptp(across) / 2 + cell))


def read_parts(m, d, instance, category):
    """Parts of a loaded LIBERO fixture. The face of a part is defined by geometry alone: with every part of the
    fixture closed, the side (of the six of the fixture's frame) from which most of the part is seen from outside,
    and on it the main level of the seen surface. What stands out beyond that level is the handle."""
    spec = FIXTURES[category]
    names = [_name(m, _B, b) for b in range(m.nbody)]
    fixture = np.array([nm.startswith(instance + "_") for nm in names])
    joints = [j for j in range(m.njnt) if fixture[m.jnt_bodyid[j]]
              and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)]
    # rays see what a hand can touch: colliding geoms, the robot left out, on a copy so the live model keeps its groups
    ms = copy.copy(m)
    robot = np.array([nm.startswith(ROBOT) for nm in names])[m.geom_bodyid]
    solid = ((m.geom_contype > 0) | (m.geom_conaffinity > 0)) & ~robot
    ms.geom_group[:] = np.where(solid, 0, 5)
    ds = mujoco.MjData(ms)
    ds.qpos[:] = d.qpos
    ds.qpos[m.jnt_qposadr[joints]] = spec["closed"]
    mujoco.mj_forward(ms, ds)
    F = ds.xmat[int(np.flatnonzero(fixture)[0])].reshape(3, 3)
    all_pts = _corners(ms, ds, np.flatnonzero(solid & fixture[m.geom_bodyid]))
    parts = []
    for j in joints:
        b = int(m.jnt_bodyid[j])
        own = _subtree(m, b)
        pts = _corners(ms, ds, np.flatnonzero(solid & own[m.geom_bodyid]))
        best = None
        for k in range(3):
            for s in (1.0, -1.0):
                n, u, v = s * F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
                h0 = (all_pts @ n).max() + CFG["ray_start"]
                gu, gv, hit, _ = _cast(ms, ds, own, pts, h0, n, u, v, CFG["ray_grid"][0])
                area = hit.sum() * (gu[1] - gu[0]) * (gv[1] - gv[0])
                if best is None or area > best[0]:
                    best = (area, n, u, v, h0)
        _, n, u, v, h0 = best
        if abs(n[2]) < 0.7:                         # an upright face: u level, n x u up, as on generated parts
            u = np.cross([0.0, 0.0, 1.0], n) / np.linalg.norm(np.cross([0.0, 0.0, 1.0], n))
            v = np.cross(n, u)
        gu, gv, hit, h = _cast(ms, ds, own, pts, h0, n, u, v, CFG["ray_grid"][1])
        edges = np.arange(h[hit].min(), h[hit].max() + 2 * CFG["level_bin"], CFG["level_bin"])
        cnt = np.histogram(h[hit], edges)[0]
        mid = edges[np.argmax(cnt)] + CFG["level_bin"] / 2
        level = float(np.median(h[hit][np.abs(h[hit] - mid) <= CFG["level_tol"]]))
        face = hit & (np.abs(h - level) <= CFG["level_tol"])
        iu, iv = np.flatnonzero(face.any(1)), np.flatnonzero(face.any(0))
        du, dv = gu[1] - gu[0], gv[1] - gv[0]
        centre = 0.5 * (gu[iu[0]] + gu[iu[-1]]) * u + 0.5 * (gv[iv[0]] + gv[iv[-1]]) * v + level * n
        R, x = ds.xmat[b].reshape(3, 3), ds.xpos[b]
        kind = "slide" if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge"
        hp, ha, hh = _read_handle(gu, gv, hit, h, level, u, v, n)
        hk = "none" if hp is None else "low_bar" if hp[2] - all_pts[:, 2].min() < CFG["low_handle"] else "bar"
        if hp is not None and kind == "hinge" and abs(ds.xaxis[j] @ n) > 0.7:
            a = ds.xaxis[j]                         # a knob: held centred on its axis
            hp, hk = ds.xanchor[j] + ((hp - ds.xanchor[j]) @ a) * a, "rotary"
        parts.append(Part(_name(m, _J, j), _name(m, _J, j), kind, names[b],
                          tuple(nm for i, nm in enumerate(names) if fixture[i] and not own[i]),
                          tuple(m.jnt_range[j]), spec["closed"], spec["open"],
                          ((gu[iu[-1]] - gu[iu[0]] + du) / 2, (gv[iv[-1]] - gv[iv[0]] + dv) / 2),
                          float(h[hit].max() - level), R.T @ (centre - x), R.T @ n, R.T @ u,
                          handle_point_b=None if hp is None else R.T @ (hp - x),
                          handle_axis_b=None if hp is None else R.T @ ha, handle_half=hh, handle_kind=hk))
    return place_parts(parts, m, d)


def sample_layout(rng, kind, out_dir=OUT / "bddl", cfg=CFG):
    """A random training layout holding one mechanism or LIBERO fixture of the given kind (one of KINDS)."""
    seed = int(rng.integers(2 ** 31))
    if kind in GENERATED or kind in E2E_GENERATED:
        xml, parts = globals()[kind](rng) if kind in GENERATED else E2E_GENERATED[kind](rng)
        return Layout(base_layout(out_dir), [xml], parts, seed, kind)
    return Layout(fixture_layout(rng, kind, Path(out_dir) / f"{kind}_{seed:010d}.bddl", cfg), [], [], seed, kind,
                  {f"{kind}_1": kind})
