"""One simulator process of the M1E teacher training (LRL_PLAN.md M1E step 2): TeacherEnv.

An episode is one contact through lrl_e2e.E2ESkill, the interface the student and the harness share. The hand starts
in the start region of a believed part (lrl_e2e.start_region), fingers open, the arm set by inverse kinematics and
touching nothing. The belief is the truth plus the hand-off error measured in M1 (harness2/HANDOFF_ERROR.md): the
centroid within a few cm, the normal either within a few degrees or, in about a third of the hand-offs, on a wrong
face (46-90 deg off: the start is above a drawer or beside a door). From there the policy chooses the contact.

Goal (lrl_e2e.Goal, general, no per-fixture constant): open = to the open end of the joint's range, close = to its
closed end, turn_on / turn_off = to that end of a knob's range; sometimes an amount instead. Reached means within
end_tol of the span at an open end, within close_tol of a closed end, within amount_tol of an amount.

The actor sees the proprio of lrl_e2e plus privileged geometry a depth camera could show, in frame A: the target's
face, handle (centre, axis, half sizes, protrusion, clearance under and over it) and box, its joint (type, axis,
value), and the K nearest other geoms as oriented boxes. Hidden physics, injected failures, the start error and the
contact flags go to the critic only: the student must be able to infer what the teacher acts on (LRL_PLAN.md M1E).

Rewards and labels are simulator truth. reset(arg) and step(action) follow StageEnv's protocol, so vec.VecEnv runs
either environment.
"""
from __future__ import annotations

import copy
import os
import pickle
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from atrace.geometry import rotvec_to_mat
from atrace.lrl_e2e import ACT_DIM, E2E_CFG, PROPRIO_DIM, Belief, E2ESkill, Goal, frame_a, start_region
from atrace.lrl_models import DETECTOR_CLASSES

from . import env as M1
from . import eval_stroke as E
from . import scenes
from .env import CFG as M1_ENV, Sensors, UNSEEN_KINDS, VAL_SEED, layout_rng
from .sim import CFG as SIM, StageSim

TEACHER_KINDS = ("drawer_e2e", "door_e2e", "knob_e2e", "sliding_panel", "wooden_cabinet", "white_cabinet",
                 "microwave", "flat_stove")
HANDLE_KINDS = ("bar", "low_bar", "recess", "knob", "lip", "none", "rotary")
INJECT = ("blocked", "slip")
K_GEOMS = 10
GEOM_DIM = 14
PRIV_DIM = 11 + 12 + 12 + 10 + K_GEOMS * GEOM_DIM
OBS_DIM = PROPRIO_DIM + PRIV_DIM
CRIT_DIM = 24

CFG = {
    "kinds": TEACHER_KINDS,
    "seeds": (0, VAL_SEED),         # T  layout seeds; validation layouts from VAL_SEED on
    "time": 15.0,                   # T  s: episode cap (LRL_PLAN.md M1E task)
    # start: distance from the believed centroid grows with the curriculum scale s to start_region's 10-20 cm
    "dist_lo": (0.04, 0.10),        # T  m at s = 0 and s = 1
    "dist_hi": (0.08, 0.20),        # T
    "start_draws": 4,               # T  belief draws per start before the start is given up
    "start_tries": 8,               # T  start poses (of a draw) tried by inverse kinematics
    # reverse curriculum: a share of the starts, falling with the scale, at the contact itself (pads open around
    # the true handle or knob as M1's pull and turn entries; closed fingertips 1.5 cm off the face for a close)
    "near_share": 0.5,              # T  at scale 0
    "near_end": 0.6,                # T  scale at which no start is near
    "near_standoff": 0.015,         # T  m: a close's fingertips off the face
    "near_grasped": 0.5,            # T  share of the near starts at a handle that begin with the pads already
                                    #    closed on it (s4: with open pads around it, no pull or turn in 0.5 M steps)
    "close_steps": 12,              # T  control steps the pads are closed for (lrl_skill close_steps)
    # hand-off error at s = 1 (HANDOFF_ERROR.md, widened by err_widen as LRL_PLAN.md 6.1 says)
    "err_widen": 1.5,               # T
    "err_pos": 0.03,                # M  m: centroid off in the face plane (36 of 40 within 3 cm)
    "err_depth": 0.01,              # M  m: along the normal (pull positions 1.1 cm median)
    "err_angle": 7.0,               # M  deg: the good mode (34 of 52 within 7.3 deg)
    "wrong_share": 1.0 / 3.0,       # M  the wrong-face mode (18 of 52 at 46-90 deg)
    "wrong_angle": (46.0, 90.0),    # M  deg
    "wrong_up": 0.6,                # T  share of wrong faces that are the part's upper face (7 of 8 wrong pushes)
    # what an episode asks
    "start_open": (0.0, 0.15),      # T  share of the span a part to be opened starts open
    "start_ajar": (0.25, 0.6),      # T  ... a part without a handle (its edge is what can be held)
    "start_close": (0.25, 1.0),     # T  ... a part to be closed (turned off: start_off)
    "start_off": (0.5, 1.0),        # T
    "amount": 0.2,                  # T  share of episodes with an amount instead of the end of the range
    "amount_share": (0.3, 0.8),     # T  of the travel left in the goal's direction
    "sense": 0.5,                   # T  share of knob episodes whose goal states the sense of the turn
    "others_open": 0.3,             # T  share of episodes whose other parts of the fixture stand open (an
                                    #    overhanging drawer above a lower one: the t03 block)
    "others_share": (0.3, 1.0),     # T  how far open they stand
    # reached (truth)
    "end_tol": 0.10,                # T  share of the span: an end of a range reached within this
    "end_tol_door": 0.30,           # T  ... for a door (a hinge that is not a knob): opened to 70 % of its range.
                                    #    A door's last third is no use to anyone, and no one grasp follows the
                                    #    arc that far (expert.py: no posture chain to 90 % of a 150 deg door);
                                    #    LIBERO's microwave predicate asks 74 of 120 deg (62 %)
    "close_tol": 0.005,             # T  m at the handle (slide, door): a closed end reached within this
    "rotary_tol": 0.05,             # T  rad: a knob's end reached within this
    "amount_tol": (0.01, 0.1, 0.15),    # T  m, rad, share of the amount: an amount reached within the larger
    # injected failures (no wrong-axis injection: the wrong-face starts replace it)
    "inject": 0.1,                  # T  share of episodes
    "slip_friction": M1_ENV["slip_friction"],   # T
    "block": M1_ENV["block"],                   # T
    # physics and sensors as M1 (env.CFG)
    **{k: M1_ENV[k] for k in ("damping", "mass", "finger_friction", "gain", "loss", "loss_add", "wrench_noise",
                              "torque_arm", "wrench_bias", "wrench_drift", "delay")},
    # layouts per process
    "pool": 3, "starts_per_layout": 8, "episodes_per_start": 3, "misses": 6, "max_loads": 20,    # T (env.CFG)
    # observation
    "clear_max": 0.25,              # T  m: clearance rays stop here
    "half_max": 0.5,                # T  m: half sizes of geoms (the table) are clipped here
    "knob_lever": 0.05,             # T  m per rad: a knob's turn in the units of progress
    # labels
    "effect": {"m": 0.01, "rad": 0.1, "share": 0.2},    # T  env.CFG
    "held_steps": 3,                # T
    "dump_share": 0.3, "dump_chunk": 64,                # T
}

# LRL_PLAN.md 6.3 plus approach shaping. Shaping is the change Phi' - Phi of Phi = - reach x distance of the grip
# site to the contact feature + align x pad alignment around a handle + hold x both fingers on the part, so it sums
# to Phi_end - Phi_start over an episode; its weights fall from 1 to shaping_end as the curriculum scale goes from 0
# to 1. (gamma Phi' - Phi with gamma 0.99 paid 0.01 x reach x distance per step for staying away, as much as the
# step cost at 8 cm: run s1 did not learn to approach in 1 M steps, NOTES.md.)
REWARD = {
    "progress": 2.0,        # T  per share of the travel left at the start
    "travel_floor": 0.02,   # T  m: that travel counts as at least this
    "reach": 6.0,           # T  per m
    "align": 0.3,           # T
    "hold": 0.6,            # T
    "shaping_end": 0.25,    # T
    "force_soft": 15.0,     # T  N: tared wrist force above which effort costs
    "force": 0.02,          # T  per step, per force_soft above the soft limit (0.05 with the noise of s2's
                            #    contacts taught to keep off the part)
    "fixture": 0.02,        # T  per step with a robot body on the part's fixture outside the part
    "outside": 0.05,        # T  per step with a robot body on anything outside the part and its fixture
    "contact_start": 0.25,  # T  share of the two contact costs at curriculum scale 0, all of them at scale 1 (at
                            #    full weight from the start, s6's hands kept off cabinets near the table)
    "step": 0.003,          # T  per step
    "smooth": 0.002,        # T  per step, times the mean square change of the six motion numbers (0.01 cost
                            #    the exploration noise itself 0.5 per episode at std 0.3)
    "reached": 1.0,         # T  once, when the goal is first reached (progress pays the last 5 mm of a close
                            #    only 0.08: s2 and s4 learnt to stop short of the closed end, NOTES.md)
    "done_reached": 1.0,    # T
    "done_short": -1.0,     # T
    "done_giveup": 0.3,     # T  done short of the goal in an episode with an injected failure
    "force_stop": -0.5,     # T
}

_SKILL_REFS = ("io", "frame", "goal", "guard", "cfg")      # what an E2ESkill refers to but does not own


def _unit(x):
    return np.asarray(x, float) / np.linalg.norm(x)


def _perp(n, rng):
    return _unit(np.cross(n, rng.normal(size=3)))


def effects(part):
    return ("turn_on", "turn_off") if scenes.turns(part) else ("open", "close")


def span_of(part):
    """(sign of opening, open end of the joint's range, span from closed to that end)."""
    s = float(np.sign(part.open - part.closed))
    lo, hi = sorted(part.range)
    end = hi if s > 0 else lo
    return s, end, abs(end - part.closed)


def lever(part):
    """m of travel at the hand per unit of joint value: 1 for a slide, the radius of the handle (else the face centre)
    for a door; for a knob the knob_lever setting."""
    if part.kind == "slide":
        return 1.0
    if scenes.turns(part):
        return CFG["knob_lever"]
    point = part.handle_point if part.handle_point is not None else part.face_point
    r = point - part.axis_point
    return float(np.linalg.norm(r - (r @ part.axis) * part.axis))


def _boxes(m, d, ids):
    """World oriented boxes of geoms (their local bounding boxes): centres, rotations (G,3,3), half sizes."""
    R = d.geom_xmat[ids].reshape(-1, 3, 3)
    return d.geom_xpos[ids] + np.einsum("gij,gj->gi", R, m.geom_aabb[ids, :3]), R, m.geom_aabb[ids, 3:].copy()


def _box_dist(p, c, R, h):
    q = np.einsum("gji,gj->gi", R, p - c)
    return np.linalg.norm(q - np.clip(q, -h, h), axis=1)


def _rect_dist(p, c, n, u, half):
    """Distance from p to a face rectangle (centre, normal, level axis, half extents)."""
    v = np.cross(n, u)
    r = p - c
    a, b = np.clip(r @ u, -half[0], half[0]), np.clip(r @ v, -half[1], half[1])
    return float(np.linalg.norm(r - a * u - b * v))


class TeacherEnv:
    """reset(arg) -> (obs, crit, info); step(action) -> (obs, crit, reward, ended, info). arg is the curriculum scale
    in [0, 1] (negative: drawn uniformly per start), or (scale, {kind: weight}). After an episode has ended the next
    call must be reset."""

    def __init__(self, seed, out_dir, cfg=CFG, reward=REWARD, randomize=True, dump_dir=None, source="teacher",
                 belief_fn=None):
        """belief_fn(sim, part) -> lrl_e2e.Belief, optional: called once per start with the episode's scene set and
        the arm at rest; its centroid (and points) replace the drawn centroid (the student's env derives them from
        a rendered, degraded mask, as the harness will). The normal's error and the wrong-face mode stay drawn."""
        self.belief_fn = belief_fn
        self.rng = np.random.default_rng(seed)
        self.out, self.cfg, self.rw, self.randomize = Path(out_dir), cfg, reward, randomize
        self.pool = deque()                 # [sim, home snapshot, misses in a row, kind, layout index, geometry]
        self.loads, self.since_load, self.replays = 0, 0, 0
        self.skill = E2ESkill()
        self.weights = None
        self.dump_dir, self.source, self.dumped, self.chunks = dump_dir and Path(dump_dir), source, [], 0
        self.tag = f"{os.getpid()}_{time.time_ns()}"

    # ------------------------------------------------------------------------------------------------ layouts
    def _load(self):
        kinds = self.cfg["kinds"]
        p = None
        if self.weights:
            p = np.array([self.weights.get(k, 0.0) for k in kinds], float)
            p = p / p.sum() if p.sum() > 0 else None
        kind = kinds[int(self.rng.choice(len(kinds), p=p))]
        index = int(self.rng.integers(*self.cfg["seeds"]))
        layout = scenes.sample_layout(layout_rng(kind, index), kind, self.out, scenes.E2E)
        sim = StageSim(layout)
        if layout.fixtures:
            Path(layout.bddl).unlink()
        self.pool.append([sim, sim.snapshot(), 0, kind, index, self._geometry(sim)])
        self.loads += 1
        self.since_load = 0

    def _geometry(self, sim):
        """What a layout's observation needs once: per part its own box in its body frame and its subtree; the
        solid geoms outside the robot; the bodies of each part's fixture."""
        m, d = sim.robot.m, sim.robot.d
        solid = ((m.geom_contype > 0) | (m.geom_conaffinity > 0)) & ~sim._is_robot[m.geom_bodyid]
        solid &= m.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
        geo = {"solid": np.flatnonzero(solid), "parts": {}, "bid": {n: i for i, n in enumerate(sim._body)},
               "grip": mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")}
        for part in sim.parts:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, part.body)
            own = scenes._subtree(m, b)
            ids = np.flatnonzero(solid & own[m.geom_bodyid])
            pts = (scenes._corners(m, d, ids) - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
            rest = set(part.fixture_bodies)
            fix = np.array([n in rest for n in sim._body])
            for k in np.flatnonzero(fix):               # a fixture body's children belong to the fixture too
                fix |= scenes._subtree(m, k)
            geo["parts"][part.name] = {"body": b, "own": own, "fixture": fix & ~own,
                                       "box": ((pts.min(0) + pts.max(0)) / 2, (pts.max(0) - pts.min(0)) / 2)}
        return geo

    def _drop(self, slot):
        slot[0].close()
        self.pool.remove(slot)

    def _slot(self):
        if not self.pool or self.since_load >= self.cfg["starts_per_layout"]:
            self._load()
            if len(self.pool) > self.cfg["pool"]:
                self._drop(self.pool[0])
        return self.pool[int(self.rng.integers(len(self.pool)))]

    # ------------------------------------------------------------------------------------------------ episode
    def reset(self, arg=1.0):
        scale, weights = arg if isinstance(arg, tuple) else (arg, None)
        if scale < 0:               # every scale at once: drawn per start (easy starts keep a learning signal while
            scale = float(self.rng.uniform())    # the full error stays in the mix; e1 NOTES.md, the long run)
        self.weights = weights or self.weights
        if self.replays:                    # the start of the last episode once more
            self.replays -= 1
            self._again()
            return self._obs(self.obs0), self._critic(), dict(self.info)
        while True:
            slot = self._slot()
            if self._start(slot, scale):
                slot[2] = 0
                self.since_load += 1
                self.replays = self.cfg["episodes_per_start"] - 1
                io = self.io if self.randomize else None
                self.start = (self.sim.snapshot(), self.obs0, io and (list(io.buf), io.bias),
                              copy.deepcopy({k: v for k, v in vars(self.skill).items() if k not in _SKILL_REFS}))
                return self._obs(self.obs0), self._critic(), dict(self.info)
            slot[2] += 1
            if slot[2] >= self.cfg["misses"]:
                self._drop(slot)

    def _belief(self, part, scale, seen=None):
        """Truth plus the hand-off error at curriculum scale `scale`: (Belief, error record). seen: belief_fn's
        Belief, whose centroid and points are taken as they are."""
        c, rng = self.cfg, self.rng
        n = part.face_normal
        w = c["err_widen"] * scale
        wrong = rng.random() < c["wrong_share"] * scale
        if wrong:
            up = np.array([0.0, 0.0, 1.0])
            if rng.random() < c["wrong_up"]:
                # the part's upper face, or for a part facing up a side facing the robot's half of the table
                n_b = up if abs(n[2]) < 0.7 else _unit(np.r_[_perp(up, rng)[:2] - [0.5, 0.0], 0.0])
            else:
                n_b = rotvec_to_mat(_perp(n, rng) * np.radians(rng.uniform(*c["wrong_angle"]))) @ n
        else:
            n_b = rotvec_to_mat(_perp(n, rng) * np.radians(rng.uniform(0, c["err_angle"] * w))) @ n
        t = _perp(n, rng)
        dc = t * rng.uniform() * c["err_pos"] * w + n * rng.uniform(-1, 1) * c["err_depth"] * w
        if seen is not None:
            dc = seen.centroid - part.face_point
        return Belief(part.face_point + dc, n_b, None if seen is None else seen.points), {
            "err_pos": float(np.linalg.norm(dc)), "err_angle": float(np.degrees(np.arccos(np.clip(n_b @ n, -1, 1)))),
            "wrong": bool(wrong)}

    def _start(self, slot, scale):
        cfg, rng = self.cfg, self.rng
        sim, home, _, kind, index, geo = slot
        part = sim.parts[int(rng.integers(len(sim.parts)))]
        effect = str(rng.choice(effects(part)))
        rotary = scenes.turns(part)
        s_open, end, span = span_of(part)
        inject = str(rng.choice(INJECT)) if rng.random() < cfg["inject"] else None
        sim.restore(home)
        slip = rng.uniform(*cfg["slip_friction"]) if inject == "slip" else None
        block = cfg["block"] if inject == "blocked" else None
        phys = sim.randomize(rng, part, cfg, slip, block) if self.randomize else None
        # only a close meets its neighbours standing open (the t03 block): a drawer to be opened between two open
        # ones leaves no room for a hand that closes upright across its bar (expert.py, e1 NOTES.md)
        others = rng.random() < cfg["others_open"] and effect == "close"
        for o in sim.parts:
            if o is not part:
                so, _, sp = span_of(o)
                sim.set_joint(o.joint, o.closed + so * sp * (rng.uniform(*cfg["others_share"]) if others else 0.0))
        opening = effect in ("open", "turn_on")
        share0 = rng.uniform(*cfg["start_open" if part.handle_point is not None or rotary else "start_ajar"]) \
            if opening else rng.uniform(*cfg["start_off" if rotary else "start_close"])
        q0 = part.closed + s_open * share0 * span
        sim.set_joint(part.joint, q0)
        base = sim.snapshot()
        # what is seen of the scene is read with the arm at rest, before it is placed at the start
        self.sim, self.part, self.geo, self.scale = sim, part, geo["parts"][part.name], scale
        self.bid, self.grip_site = geo["bid"], geo["grip"]
        self._near(geo)
        self._clearance()
        seen = self.belief_fn(sim, part) if self.belief_fn is not None else None
        # the goal, in the opening coordinate x = (q - closed) s_open: 0 closed, span at the open end
        dirn = 1.0 if opening else -1.0
        left_j = span * (1.0 - share0) if opening else span * share0
        unit = "m" if part.kind == "slide" else "rad"
        amount = None
        tol_amount = cfg["amount_tol"][0 if unit == "m" else 1]
        if rng.random() < cfg["amount"] and left_j > 3 * tol_amount:
            amount = float(rng.uniform(*cfg["amount_share"]) * left_j)
            tol_amount = max(tol_amount, cfg["amount_tol"][2] * amount)
        sense = 0
        if rotary and rng.random() < cfg["sense"]:
            sense = int(np.sign((part.axis * s_open * dirn) @ part.face_normal))
        lev = lever(part)
        close_tol = cfg["rotary_tol"] if rotary else cfg["close_tol"] / lev
        # the start: belief, start region, inverse kinematics, nothing touched
        base_pos = sim.robot.d.xpos[mujoco.mj_name2id(sim.robot.m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")].copy()
        lo, hi = (a + (b - a) * scale for a, b in (cfg["dist_lo"], cfg["dist_hi"]))
        near = rng.random() < cfg["near_share"] * max(0.0, 1.0 - scale / cfg["near_end"])
        if near:
            belief, err = self._belief(part, scale, seen)
            cands, finger = self._near_poses(part, effect)
            dist, pose = 0.0, None
            for k in rng.permutation(len(cands)):
                sim.restore(base)
                if sim.start_at(*cands[k], finger):
                    pose = int(k)
                    break
            near = pose is not None
            if near and effect != "close" and rng.random() < cfg["near_grasped"]:
                sim.robot.hold(cfg["close_steps"], grip=1.0)
                sim.robot.grip_cmd = 0.0            # the fingers stay where they closed until the policy acts
                near = "grasped"
        for _ in range(0 if near else cfg["start_draws"]):
            belief, err = self._belief(part, scale, seen)
            dist = float(rng.uniform(lo, hi))
            cands = start_region(belief, dists=(dist,))
            pose = None
            for k in rng.permutation(len(cands))[:cfg["start_tries"]]:
                R, p = cands[k]
                sim.restore(base)
                if sim.start_at(R, p, SIM["finger_open"]):
                    pose = int(k)
                    break
            if pose is not None:
                break
        else:
            if not near:
                return False
        R0, p0 = sim.robot.eef_pose()
        frame = frame_a(p0, belief, base_pos)
        goal = Goal(effect, sense, amount, unit)
        io = Sensors(sim.robot, rng, cfg) if self.randomize else sim.robot
        self.frame, self.goal = frame, goal
        self.obs0 = self.skill.start(io, frame, goal, max_steps=round(cfg["time"] / E2E_CFG["dt"]))
        self.io, self.phys, self.inject, self.effect, self.rotary = io, phys, inject, effect, rotary
        self.s_open, self.span, self.end, self.lev, self.dirn = s_open, span, end, lev, dirn
        self.x0, self.amount, self.tol_amount, self.close_tol = share0 * span, amount, tol_amount, close_tol
        self.sense, self.err, self.q0 = sense, err, q0
        self.F0 = sim.robot.wrench()[0]
        self._begin()
        floor = self.rw["travel_floor"]
        self.left0 = max(self.left, floor)
        hk = part.handle_kind or ("none" if part.handle_point is None else "bar")
        self.info = {"kind": kind, "layout": index, "part": part.name, "effect": effect, "handle": hk,
                     "scale": float(scale), "inject": inject or "", "amount": amount is not None, "sense": sense,
                     "dist": dist, "pose": pose, "near": near, "left0": float(self.left)} | err
        return True

    def _near_poses(self, part, effect):
        """Start poses at the contact itself, from the truth (reverse curriculum): (poses, finger joint value)."""
        if effect == "close":                   # on the face, or on the outside of a handle standing in front of it
            c, n, _ = E.contact_face(part, self.sim.joint(part.joint))
            return [(R, p - a * self.cfg["near_standoff"]) for c_ in (c, c + n * part.protrusion)
                    for _, a, R, p in E.press_poses(c_, n)["tip"]], 0.0
        if part.handle_point is None:          # an ajar part without a handle: no grip pose is known
            return [], 0.0
        hp, e, round_, stand, across = M1.handle(part)
        return M1.grip_poses(hp, part.face_normal, e, round_, stand), M1.finger_for(across)

    def _near(self, geo):
        """The K solid geoms nearest to the target's face, outside the target itself (fixed for the episode)."""
        m, d = self.sim.robot.m, self.sim.robot.d
        ids = geo["solid"][~self.geo["own"][m.geom_bodyid[geo["solid"]]]]
        c, R, h = _boxes(m, d, ids)
        order = np.argsort(_box_dist(self.part.face_point, c, R, h))[:K_GEOMS]
        self.near = ids[order]
        self.near_fix = self.geo["fixture"][m.geom_bodyid[self.near]].astype(float)

    def _clearance(self):
        """Free distance under and over the handle (rays from its centre, the part itself excluded), what a depth
        camera shows of the room a hand has there; clear_max where nothing is hit."""
        m, d, p = self.sim.robot.m, self.sim.robot.d, self.part.handle_point
        self.clear = np.zeros(2)
        if p is None:
            return
        gid = np.zeros(1, np.int32)
        for i, z in enumerate((-1.0, 1.0)):
            dist = mujoco.mj_ray(m, d, p, np.array([0.0, 0.0, z]), None, 1, self.geo["body"], gid)
            self.clear[i] = self.cfg["clear_max"] if dist < 0 else min(dist, self.cfg["clear_max"])

    def _again(self):
        snap, self.obs0, sensors, skill = self.start
        self.sim.restore(snap)          # restore places the parts
        vars(self.skill).update(copy.deepcopy(skill))
        if sensors:
            self.io.buf, self.io.bias = deque(sensors[0], maxlen=self.io.delay + 1), sensors[1]
        self._begin()

    def _begin(self):
        self.left = self._left()
        self.ok = self.ever = self._reached()
        self.ret, self.count = 0.0, np.zeros(4)     # steps with: fixture contact, outside contact, > soft force, held
        self.t, self.t_done = 0, -1
        self.touched_part = self.outside_any = self.fixture_any = False
        self.held = 0
        self.a_prev = np.zeros(ACT_DIM)
        self.shape_w = 1.0 - (1.0 - self.rw["shaping_end"]) * self.scale
        self.contact_w = self.rw["contact_start"] + (1.0 - self.rw["contact_start"]) * self.scale
        scenes.place_parts([self.part], self.sim.robot.m, self.sim.robot.d)
        self._flags()
        self.phi = self._phi()
        self.dump_this = self.dump_dir is not None and self.rng.random() < self.cfg["dump_share"]

    # ------------------------------------------------------------------------------------------------ truth
    def _x(self):
        return (self.sim.joint(self.part.joint) - self.part.closed) * self.s_open

    def _left(self):
        """Joint distance left to the goal, in m at the hand (knob: knob_lever per rad)."""
        x = self._x()
        if self.amount is not None:
            return abs(self.x0 + self.dirn * self.amount - x) * self.lev
        return max(0.0, (self.span - x) if self.dirn > 0 else x) * self.lev

    def _reached(self):
        x = self._x()
        if self.amount is not None:
            return bool(abs(x - self.x0 - self.dirn * self.amount) <= self.tol_amount)
        if self.dirn > 0:
            tol = self.cfg["end_tol_door" if self.part.kind == "hinge" and not self.rotary else "end_tol"]
            return bool(x >= (1.0 - tol) * self.span)
        return bool(x <= self.close_tol)

    def _flags(self):
        """Contact flags of the present step: fingers on the part, any robot body on the part's fixture, any on
        something outside; _grip = how many fingers touch the part."""
        fingers, f = set(), np.zeros(3, bool)
        own, fix = self.geo["own"], self.geo["fixture"]
        for a, b, _ in self.sim.robot_contacts():
            i, r = self.bid[b], E._robot(a)
            if r < 0:                       # the base and mount stand where they stand
                continue
            t = 0 if own[i] else 1 if fix[i] else 2
            if t == 0:
                self.touched_part = True
                if r == 0:
                    f[0] = True
                    fingers.add("left" if ("left" in a or "joint1" in a) else "right")
            else:
                f[t] = True
        self._grip = len(fingers)
        self.flags = f
        return f

    def _feature(self):
        """Distance of the grip site to the contact feature, and the pad alignment (0-1) around a handle."""
        part, d = self.part, self.sim.robot.d
        R, p = d.site_xmat[self.grip_site].reshape(3, 3), d.site_xpos[self.grip_site]
        if self.effect != "close" and part.handle_point is not None:
            dist = float(np.linalg.norm(p - part.handle_point))
            h = part.handle_half
            across = 1.0 if abs(h[0] - h[1]) < 0.3 * max(h) else 1.0 - abs(R[:, 0] @ part.handle_axis)
            into = max(0.0, -R[:, 2] @ part.face_normal)
            return dist, 0.5 * (across + into) * np.exp(-dist / 0.05)
        return _rect_dist(p, part.face_point, part.face_normal, part.face_u, part.half_extents), 0.0

    def _phi(self):
        rw = self.rw
        dist, align = self._feature()
        hold = float(self._grip == 2 and self.effect != "close")
        return self.shape_w * (-rw["reach"] * dist + rw["align"] * align + rw["hold"] * hold)

    def step(self, action):
        rw = self.rw
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        obs = self.skill.step(a)
        stop = self.skill.stop
        scenes.place_parts([self.part], self.sim.robot.m, self.sim.robot.d)
        left = self._left()
        self.ok = self._reached()
        first = self.ok and not self.ever
        self.ever |= self.ok
        f = self._flags()
        force = np.linalg.norm(self.sim.robot.wrench()[0] - self.F0)
        over = max(0.0, force / rw["force_soft"] - 1.0)
        phi = self._phi()
        r = (rw["progress"] * (self.left - left) / self.left0 + phi - self.phi
             - rw["force"] * over - self.contact_w * (rw["fixture"] * f[1] + rw["outside"] * f[2]) - rw["step"]
             - rw["smooth"] * np.mean((a[:6] - self.a_prev[:6]) ** 2) + rw["reached"] * first)
        if stop == "done":
            r += rw["done_reached"] if self.ok else rw["done_giveup"] if self.inject else rw["done_short"]
            self.t_done = self.t
        elif stop == "force":
            r += rw["force_stop"]
        self.left, self.phi, self.a_prev = left, phi, a
        self.ret += r
        self.count += (f[1], f[2], over > 0, self._grip == 2)
        self.held += self._grip == 2
        self.outside_any |= bool(f[2])
        self.fixture_any |= bool(f[1])
        self.t += 1
        info = {}
        if stop is not None:
            label, travel = self._label()
            info = self.info | {
                "stop": stop, "steps": self.t, "t_done": self.t_done, "return": float(self.ret),
                "success": self.ok, "ever": self.ever, "libero": self._libero(), "left": float(left),
                "label": label, "travel": travel, "fixture_any": self.fixture_any, "outside_any": self.outside_any,
                "fixture_steps": int(self.count[0]), "outside_steps": int(self.count[1]),
                "force_steps": int(self.count[2]), "held_steps": int(self.count[3]),
                "retire": self.loads >= self.cfg["max_loads"]}
            if self.dump_this:
                self._dump(info)
        return self._obs(obs), self._critic(), float(r), stop is not None, info

    def _libero(self):
        """LIBERO's own predicate for the goal's effect on a LIBERO fixture (reported beside the general rule);
        None for generated mechanisms."""
        if not self.sim.layout.fixtures:
            return None
        obj = self.sim.env.env.get_object(next(iter(self.sim.layout.fixtures)))
        q = self.sim.joint(self.part.joint)
        test = {"open": "is_open", "close": "is_close", "turn_on": "turn_on", "turn_off": "turn_off"}[self.effect]
        # the base class's predicates raise NotImplementedError: only a fixture's own count (some unseen fixtures)
        own = getattr(type(obj), test, None)
        return bool(getattr(obj, test)(q)) if own is not None and own is not getattr(M1._articulated(), test, None) \
            else None

    def _label(self):
        """Detector class from truth (DETECTOR_CLASSES) and the travel towards the goal (m at the hand)."""
        cfg = self.cfg
        travel = float((self._x() - self.x0) * self.dirn * self.lev)
        want = (self.amount if self.amount is not None else
                (self.span - self.x0 if self.dirn > 0 else self.x0)) * self.lev
        unit = cfg["effect"]["rad"] * self.lev if self.part.kind == "hinge" else cfg["effect"]["m"]
        if self.ok:
            label = "reached"
        elif self.outside_any:
            label = "outside"
        elif not self.touched_part:
            label = "no_contact"
        elif self.effect != "close" and self.held >= cfg["held_steps"] and self._grip == 0:
            label = "lost_grip"
        elif travel < min(unit, cfg["effect"]["share"] * want):
            label = "no_effect"
        else:
            label = "short"
        assert label in DETECTOR_CLASSES
        return label, travel

    # ------------------------------------------------------------------------------------------- observation
    def _obs(self, proprio):
        return np.hstack([proprio, self._priv()])

    def _priv(self):
        """Privileged geometry in frame A, at the present joint values (PRIV_DIM)."""
        sim, part, A, c = self.sim, self.part, self.frame, self.cfg
        m, d = sim.robot.m, sim.robot.d
        ps = E2E_CFG["p_scale"]
        b = self.geo["body"]
        Rb, xb = d.xmat[b].reshape(3, 3), d.xpos[b]
        face = np.hstack([ps * A.point(part.face_point), A.vec(part.face_normal), A.vec(part.face_u),
                          ps * np.asarray(part.half_extents)])
        if part.handle_point is not None:
            handle = np.hstack([1.0, ps * A.point(part.handle_point), A.vec(part.handle_axis),
                                ps * np.asarray(part.handle_half), 4 * ps * part.protrusion, ps * self.clear])
        else:
            handle = np.zeros(12)
        bc, bh = self.geo["box"]
        Rbox = A.R.T @ Rb
        box = np.hstack([ps * A.point(xb + Rb @ bc), Rbox[:, 0], Rbox[:, 1], ps * bh])
        if self.rotary and not self.sense:
            axis = part.face_normal             # a knob's sense of turn is not seen: only its axis line
        else:
            axis = part.axis * self.s_open      # + opens: a drawer or door shows which way it opens
        x = self._x()
        joint = np.hstack([part.kind == "slide", part.kind == "hinge", A.vec(axis),
                           ps * A.point(part.axis_point) if part.kind == "hinge" else np.zeros(3),
                           x * (ps if part.kind == "slide" else 1.0), x / self.span])
        gc, gR, gh = _boxes(m, d, self.near)
        Rg = np.einsum("ji,gjk->gik", A.R, gR)
        geoms = np.zeros((K_GEOMS, GEOM_DIM))      # a scene with fewer geoms leaves rows empty (presence 0)
        geoms[:len(self.near)] = np.hstack([ps * A.point(gc), Rg[:, :, 0], Rg[:, :, 1],
                                            ps * np.minimum(gh, c["half_max"]), self.near_fix[:, None],
                                            np.ones((len(self.near), 1))])
        return np.hstack([face, handle, box, joint, geoms.ravel()])

    def _critic(self):
        """Simulator truth for the critic only (CRIT_DIM)."""
        f = self.phys or {"damping": 1.0, "mass": 1.0, "finger_friction": 1.0, "gain": 1.0, "loss": 0.0}
        io = self.io if self.randomize else None
        force = np.linalg.norm(self.sim.robot.wrench()[0] - self.F0)
        return np.hstack([
            np.log([f["damping"], f["mass"], f["finger_friction"], f["gain"]]),
            min(f["loss"] / self.cfg["loss_add"][self.part.kind], 10.0),
            np.array(INJECT) == self.inject, self.err["err_pos"] * 20, self.err["err_angle"] / 45, self.err["wrong"],
            self.flags, self._grip / 2.0, self.left / self.left0, self.ok, force / self.rw["force_soft"],
            (io.delay / 2.0, io.sigma[0]) if io else (0.0, 0.0), self.frame.vec(self.part.axis * self.s_open),
            self._x() / self.span, self.shape_w])

    # ------------------------------------------------------------------------------------------------ dumps
    def _dump(self, info):
        tr = self.skill.trace.arrays()
        g = self.skill.goal
        self.dumped.append({
            "trace": {k: v.astype(np.float32) for k, v in tr.items()},
            "frame": (self.frame.R.astype(np.float32), self.frame.o.astype(np.float32)),
            "goal": {"effect": g.effect, "sense": g.sense, "amount": g.amount, "unit": g.unit},
            "source": self.source} | {k: info[k] for k in (
                "kind", "layout", "part", "effect", "handle", "inject", "stop", "steps", "label", "travel", "success",
                "scale", "wrong", "err_pos", "err_angle")})
        if len(self.dumped) >= self.cfg["dump_chunk"]:
            self._flush()

    def _flush(self):
        if not self.dumped:
            return
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        path = self.dump_dir / f"{self.source}_{self.tag}_{self.chunks:04d}.pkl"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.dumped, f)
        tmp.replace(path)
        self.dumped, self.chunks = [], self.chunks + 1

    def close(self):
        if self.dump_dir is not None:
            self._flush()
        for slot in list(self.pool):
            self._drop(slot)


__all__ = ["TeacherEnv", "CFG", "REWARD", "TEACHER_KINDS", "UNSEEN_KINDS", "VAL_SEED", "HANDLE_KINDS", "INJECT",
           "OBS_DIM", "CRIT_DIM", "PRIV_DIM", "ACT_DIM", "effects", "span_of", "lever"]
