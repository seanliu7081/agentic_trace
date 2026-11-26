"""One simulator process of the move_part training (LRL_PLAN.md 6.3, M1 second half): StageEnv.

An episode is one contact of the lrl skill on a training layout, in one of three modes:
  push  an open part is pushed closed with closed fingertips; the hand pose is one of the press search's
        (eval_stroke.press_poses) at the believed face, the arm starts at the standoff and the skill makes its own
        way in;
  pull  a closed part is opened by its handle: the arm starts with the pads open around the believed handle, the
        skill closes them and leads along the believed opening direction (a door's initial one: the steer residual
        is how the arc is followed);
  turn  a knob is turned: pads open around the believed knob, the skill closes them and turns about the believed
        axis.
The believed frame is the true one plus a sampled hand-off error. In a share of the episodes a failure is injected
(a wrong face or axis, a blocked part, an entry off the part, slipping pads) so that the detector sees them and the
policy learns that giving up beats pushing on. The actor sees what the skill sees (OBS_DIM); rewards, the critic's
extra vector and the labels are simulator truth and never reach the actor.

CFG "base" False runs the skill with its base law off (M1E step 1, the control run E0): the policy's numbers are the
whole motion, and the reward charges their change from step to step instead of their size.

Loading a layout costs 0.6 to 1.6 s and robosuite keeps about 15 MB of every environment it built (both measured,
train/NOTES.md), so a process keeps a small pool of layouts, replaces the oldest every few episodes, and asks to be
retired after a number of loads. A share of the finished episodes is written to disk for the detector (dump).
"""
from __future__ import annotations

import copy
import os
import pickle
import time
from collections import deque
from pathlib import Path

import numpy as np

from atrace import manip
from atrace.geometry import frame_from_axes, rotvec_to_mat
from atrace.lrl_contract import Contract, Entry, Goal
from atrace.lrl_models import DETECTOR_CLASSES
from atrace.lrl_skill import ACT_DIM, CFG as SKILL, MODES, OBS_DIM, Skill

from . import eval_stroke as E
from . import scenes
from .sim import StageSim

TRAIN_KINDS = ("drawer_stack", "door", "sliding_panel", "knob", "wooden_cabinet", "white_cabinet", "microwave",
               "flat_stove")
UNSEEN_KINDS = ("short_cabinet", "short_fridge", "slide_cabinet", "window", "faucet", "basin_faucet")   # never trained on
VAL_SEED = 100000               # layout seeds below are training layouts, from here on validation layouts
INJECT = ("wrong_axis", "blocked", "no_contact", "slip")
CRIT_DIM = 30

CFG = {
    "kinds": TRAIN_KINDS,
    "seeds": (0, VAL_SEED),         # T  layout seeds of this environment
    "time": {"push": 8.0, "pull": 10.0, "turn": 10.0},     # T  s: longest episode, a push's way in included
    "err_widen": 1.5,               # T  hand-off error up to this multiple of the prior range (LRL_PLAN.md 6.1; the
                                    #    measured good mode lies inside the prior: harness2/HANDOFF_ERROR.md)
    "err_depth": 0.01,              # T  m: believed handle depth along the face normal (pull positions measured
                                    #    1.1 cm median, 2.3 cm 90th percentile: HANDOFF_ERROR.md), times err_widen
    "err_turn": 0.2,                # T  rad: believed turn travel, the turn's err_travel
    "open_slide": (0.03, 0.22),     # T  m: opening of a slide at the start of a push (its full range if shorter)
    "open_hinge": (20.0, 120.0),    # T  deg: opening of a hinge at the start of a push (its full range if shorter)
    "others_open": 0.2,             # T  share of episodes in which the other parts of the fixture stand open too
    "standoff": (0.04, 0.08),       # T  m: where the hand-off leaves the hand of a push, back along the approach
    "shuffle": 0.5,                 # T  share of episodes that try a family's poses in random order
    "pull_share": 0.5,              # T  share of the episodes on a part with a handle that pull it open
    # goal of a pull or turn on a generated part (a LIBERO fixture's goal is its own predicate's threshold + over)
    "goal_slide": (0.08, 0.25),     # T  m of opening
    "goal_hinge": (50.0, 110.0),    # T  deg of opening
    "goal_turn": (45.0, 150.0),     # T  deg of turn
    "goal_over": {"slide": 0.01, "hinge": 0.1},     # T  m, rad beyond a LIBERO predicate's threshold
    "reach_tol": {"m": 0.005, "rad": 0.05},         # T  a generated part within this of its goal has reached it
    "grip_clear": 0.003,            # T  m: open fingertips this far off the surface behind the handle
    "grip_draws": 4,                # T  error draws tried for a grip entry before the start is given up
    "grip_margin": 0.015,           # T  m: pads open this far beyond the believed handle on each side (the harness
                                    #    opens them as wide as the grasp needs, not fully: lrl.py, width)
    # injected failures: share of episodes, then what each one is
    "inject": 0.15,                 # T
    "wrong_angle": (45.0, 90.0),    # T  deg: believed direction or axis off the true one (as measured in the
                                    #    harness: 46-90 deg, HANDOFF_ERROR.md)
    "off_part": (0.06, 0.10),       # T  m: entry this far beyond the part's extent
    "slip_friction": (0.02, 0.1),   # T  sliding friction of pads and part
    "block": (1000.0, 1e5),         # T  friction loss (N or N m) and damping (N s/m or N m s/rad): the joint does
                                    #    not move
    "pool": 3,                      # T  layouts a process keeps loaded
    "starts_per_layout": 8,         # T  starts between two loads
    "episodes_per_start": 2,        # T  episodes run from one start (posture search and way in cost as much
                                    #    simulator time as the episode they lead to; measured, train/NOTES.md)
    "misses": 4,                    # T  starts in a row without a pose after which a layout is dropped
    "max_loads": 20,                # T  loads after which the process asks to be retired (memory, see above)
    # physics of an episode (StageSim.randomize): factors on the loaded values, drawn log-uniformly
    "damping": (0.25, 6.0),         # T  (0.5-2 in runs 1-3 left the unseen fridge door outside: train/REPORT.md)
    "mass": (0.25, 6.0),            # T
    "finger_friction": (0.5, 1.5),  # T
    "gain": (0.8, 1.2),             # T  controller kp (LRL_PLAN.md 6.2: +-20 %)
    "loss": (1.0, 1.3),             # T  factor on the loaded friction loss, on top of the mass factor
    "loss_add": {"slide": 4.0, "hinge": 0.3},   # T  N, N m: friction loss added, uniform from 0
    # sensors of an episode
    "wrench_noise": (0.3, 1.5),     # T  N per axis and step (LRL_PLAN.md 6.2)
    "torque_arm": 0.05,             # T  m: torque noise = force noise times this lever
    "wrench_bias": 2.0,             # T  N per axis, constant (the skill's tare removes it)
    "wrench_drift": 0.05,           # T  N per axis and step: largest random-walk rate of the bias
    "delay": (0, 2),                # T  steps: age of what the skill reads (LRL_PLAN.md 6.2)
    # labels
    "effect": {"m": 0.01, "rad": 0.1, "share": 0.2},    # T  less travel than min(unit, share of the goal): no effect
    "held_steps": 3,                # T  steps with both fingers on the part that make a grip
    # episodes written for the detector
    "dump_share": 0.3,              # T
    "dump_chunk": 64,               # T  episodes per file
    "base": True,                   # T  the skill's base law (lrl_skill CFG "base"); False for the control run E0
                                    #    of M1E step 1, which charges REWARD "change" in place of "residual"
}

# Reward of LRL_PLAN.md 6.3. Progress sums to `progress` over the whole goal; everything else is small against it,
# except a done flag raised short of the goal, which must cost more than waiting for the time limit, unless the
# episode carries an injected failure: there giving up is what is wanted.
REWARD = {
    "progress": 2.0,        # T  per share of the travel left when the policy takes over
    "travel_floor": 0.02,   # T  m (rad: x10): that travel counts as at least this (a part the way in all but closed)
    "force_soft": 15.0,     # T  N: tared wrist force above which effort costs (effort follows the felt resistance)
    "force": 0.05,          # T  per step, per force_soft above the soft limit
    "body": 0.01,           # T  per step with the hand body or an arm link on the rest of the fixture
    "outside": 0.02,        # T  per step with any robot body on something that is neither the part nor its fixture
    "step": 0.003,          # T  per step
    "residual": 0.002,      # T  per step, times the mean square of the nine residual numbers
    "change": 0.002,        # T  with the base law off, in place of residual (whose size would charge the motion
                            #    itself): per step, times the mean square of the nine numbers' change since the last
    "done_closed": 1.0,     # T  the done flag raised with the goal reached
    "done_open": -1.0,      # T  the done flag raised short of the goal
    "done_giveup": 0.3,     # T  the done flag raised short of the goal in an episode with an injected failure
}


_SKILL_REFS = ("io", "goal", "lim", "guard", "policy", "cfg")     # what a Skill refers to but does not own


def layout_rng(kind, index):
    """The layout is a function of (kind, index) alone; kinds of eval_stroke keep its draws (comparable with M0
    and the first M1 evaluation)."""
    k = E.KINDS.index(kind) if kind in E.KINDS else 100 + scenes.KINDS.index(kind)
    return np.random.default_rng([k, index])


def _unit(x):
    return x / np.linalg.norm(x)


class Sensors:
    """What the skill reads from the robot in training: hand pose, wrench and finger gap `delay` steps old, the
    wrench with white noise and a bias that drifts. Commands go through unchanged."""

    def __init__(self, robot, rng, cfg=CFG):
        self.rb, self.rng = robot, rng
        self.sigma = rng.uniform(*cfg["wrench_noise"]) * np.r_[np.ones(3), cfg["torque_arm"] * np.ones(3)]
        self.bias = rng.uniform(-1, 1, 6) * cfg["wrench_bias"] * self.sigma / self.sigma[0]
        self.drift = rng.uniform(0, cfg["wrench_drift"]) * self.sigma / self.sigma[0]
        self.delay = int(rng.integers(cfg["delay"][0], cfg["delay"][1] + 1))
        self.buf = deque(maxlen=self.delay + 1)
        self._read()

    def _read(self):
        self.bias = self.bias + self.drift * self.rng.normal(size=6)
        w = np.hstack(self.rb.wrench()) + self.bias + self.sigma * self.rng.normal(size=6)
        self.buf.append((self.rb.eef_pose(), (w[:3], w[3:]), self.rb.finger_opening()))

    def step(self, R, p, grip):
        self.rb.step(R, p, grip)
        self._read()

    def eef_pose(self):
        return self.buf[0][0]

    def wrench(self):
        return self.buf[0][1]

    def finger_opening(self):
        return self.buf[0][2]


def enter(sim, hand, base, cands, standoff):
    """eval_stroke.find_entry with the standoff as an argument: index of the first pose whose standoff has an arm
    posture that touches nothing and a Cartesian line to the touch pose; the arm is left there."""
    for k, (mode, a, R, p) in enumerate(cands):
        sim.restore(base)
        if (sim.entry(R, p - a * standoff) and not any(E._robot(b) >= 0 for b, _, _ in sim.robot_contacts())
                and hand.line_configs(R, E._aim(mode, a, p), sim.robot.q()[:7], finger=0.0) is not None):
            return k
    return None


def enter_grip(sim, base, cands, finger):
    """Index of the first grip pose (R, p) the arm can be set to with the fingers open at `finger` (joint value),
    touching nothing; the arm is left there. The harness reaches it by a certified transit and a guarded line; here
    the arm is placed."""
    for k, (R, p) in enumerate(cands):
        sim.restore(base)
        if sim.entry(R, p, finger=finger) and not any(E._robot(b) >= 0 for b, _, _ in sim.robot_contacts()):
            return k
    return None


# ------------------------------------------------------------------------------------- pull and turn geometry
def handle(part):
    """Where the pads close on a part: centre, long axis (the pads close across it), whether it is round (any roll
    holds it), how far the centre stands out of the surface behind it, and the width across. A part without a handle
    that turns (a faucet lever) is held by the top of its face, across its shorter side."""
    n = part.face_normal
    if part.handle_point is not None:
        hp, e = part.handle_point, part.handle_axis
        round_ = abs(part.handle_half[0] - part.handle_half[1]) < 0.3 * max(part.handle_half)
        across = 2.0 * part.handle_half[1]
    else:
        hp = part.face_point - n * scenes.CFG["grip_depth"]
        u, v = part.face_u, np.cross(n, part.face_u)
        e, round_ = (u if part.half_extents[0] >= part.half_extents[1] else v), False
        across = 2.0 * min(part.half_extents)
    # a knob's centre stands out of the surface it is mounted on, where its joint is anchored
    behind = part.axis_point if scenes.turns(part) else part.face_point
    return hp, e, round_, float((hp - behind) @ n), across


def finger_for(across, cfg=CFG):
    """Finger joint value whose pad gap is the believed width plus the margin on each side (manip: pad inner face =
    finger joint + PAD_OFFSET)."""
    return float(np.clip(across / 2 + cfg["grip_margin"] - manip.PAD_OFFSET, 0.0, 0.04))


def opening_direction(part, q):
    """The true direction a held handle moves in when the part opens from q (a hinge: the tangent of its arc)."""
    s = np.sign(part.open - part.closed)
    if part.kind == "slide":
        return part.axis * s
    hp = handle(part)[0]
    return _unit(np.cross(part.axis * s, hp - part.axis_point))


def lever_at(part, point):
    """m of travel at `point` per unit of joint value."""
    if part.kind == "slide":
        return 1.0
    r = point - part.axis_point
    return float(np.linalg.norm(r - (r @ part.axis) * part.axis))


def grip_poses(hb, n_b, e_b, round_, stand):
    """Hand poses with the pads open around a believed handle centre hb on a face with believed outward normal n_b:
    approach along -n_b, closing axis across the handle's long axis e_b, the grip site at the centre unless the
    fingertips would then reach the surface behind (stand: how far the centre stands out of it)."""
    a = -n_b
    x = np.cross(n_b, e_b)
    x = _unit(x) if np.linalg.norm(x) > 1e-6 else _unit(np.cross(n_b, [0.0, 0.0, 1.0] if abs(n_b[2]) < 0.9
                                                                   else [1.0, 0.0, 0.0]))
    p = hb + n_b * max(0.0, manip.TIP_Z + CFG["grip_clear"] - stand)
    rolls = (0.0, np.pi, np.pi / 2, -np.pi / 2) if round_ else (0.0, np.pi)
    return [(frame_from_axes(a, rotvec_to_mat(a * r) @ x), p) for r in rolls]


def goal_value(part, mode, rng, sim):
    """Joint value a pull or turn drives to (truth), and the predicate of having reached it: the fixture's own where
    LIBERO has one, else within reach_tol at the handle."""
    s = np.sign(part.open - part.closed)
    lo, hi = sorted(part.range)
    fix = _libero(sim)
    if fix is not None and mode == "pull" and hasattr(fix, "is_open") and \
            type(fix).is_open is not _articulated().is_open:
        thr = max(fix.object_properties["articulation"]["default_open_ranges"]) if s < 0 else \
            min(fix.object_properties["articulation"]["default_open_ranges"])
        q = float(np.clip(thr + s * CFG["goal_over"][part.kind], lo, hi))
        return q, lambda x: bool(fix.is_open(x))
    if fix is not None and mode == "turn" and hasattr(fix, "turn_on"):
        thr = min(fix.object_properties["articulation"]["default_turnon_ranges"])
        q = float(np.clip(thr + CFG["goal_over"]["hinge"], lo, hi))
        return q, lambda x: bool(fix.turn_on(x))
    span = abs(part.open - part.closed)
    if mode == "turn":
        want = np.radians(rng.uniform(*CFG["goal_turn"]))
    elif part.kind == "slide":
        want = rng.uniform(*CFG["goal_slide"])
    else:
        want = np.radians(rng.uniform(*CFG["goal_hinge"]))
    q = float(part.closed + s * min(want, span))
    tol = CFG["reach_tol"]["rad"] if mode == "turn" else CFG["reach_tol"]["m"] / lever_at(part, handle(part)[0])
    return q, lambda x: bool((x - q) * s >= -tol)


def _articulated():
    from libero.libero.envs.objects.articulated_objects import ArticulatedObject
    return ArticulatedObject


def _libero(sim):
    """The LIBERO object of a fixture layout (its predicates), or None for generated mechanisms."""
    if not sim.layout.fixtures:
        return None
    return sim.env.env.get_object(next(iter(sim.layout.fixtures)))


def part_modes(part):
    if scenes.turns(part):
        return ("turn",)
    return ("push", "pull") if part.handle_point is not None else ("push",)


def believe_grip(part, mode, rng, scale, cfg=CFG, inject=None):
    """The believed frame of a pull or turn: (grip candidates, believed goal direction or axis, believed axis
    point, the errors drawn). Truth plus the hand-off error of eval_stroke.sample_error (lateral position, tilt),
    a depth error along the face normal; an injection moves the entry off the part or turns the goal's direction."""
    n = part.face_normal
    hp, e, round_, stand, _ = handle(part)
    w = cfg["err_widen"] * scale
    dc, dn, _ = E.sample_error(rng, n, w)
    dc = dc + n * rng.uniform(-1, 1) * cfg["err_depth"] * w
    if inject == "no_contact":
        t = _unit(np.cross(n, rng.normal(size=3)))
        u, v = part.face_u, np.cross(n, part.face_u)
        extent = abs(t @ u) * part.half_extents[0] + abs(t @ v) * part.half_extents[1]
        dc = (extent + rng.uniform(*cfg["off_part"])) * t
    R_err = rotvec_to_mat(dn)
    n_b, e_b = R_err @ n, R_err @ e
    if mode == "turn":
        d = part.axis * np.sign(part.open - part.closed)
        # the pads close across the knob, approaching along its axis: the face normal of a turning part is its axis
        cands = grip_poses(hp + dc, n_b, e_b, round_, stand)
    else:
        d = opening_direction(part, part.closed)
        cands = grip_poses(hp + dc, n_b, e_b, round_, stand)
    d_b = R_err @ d
    if inject == "wrong_axis":
        k = _unit(np.cross(d, rng.normal(size=3)))
        d_b = rotvec_to_mat(k * np.radians(rng.uniform(*cfg["wrong_angle"]))) @ d
    return cands, d_b, part.axis_point + dc, {"err_pos": float(np.linalg.norm(dc)),
                                               "err_angle": float(np.degrees(np.linalg.norm(dn)))}


# ----------------------------------------------------------------------------------------------- the env
class StageEnv:
    """reset(arg) -> (obs, crit, info); step(action) -> (obs, crit, reward, ended, info). `arg` is the curriculum
    share of the hand-off error in [0, 1], or (that share, {kind: weight}) to sample the kinds of new layouts by
    weight (the trainer favours kinds with low success). After an episode has ended the next call must be reset.
    `source` names what drives the episodes in the dumps (the trainer's policy with noise, by default)."""

    def __init__(self, seed, out_dir, cfg=CFG, reward=REWARD, randomize=True, dump_dir=None, source="train"):
        self.rng = np.random.default_rng(seed)
        self.out, self.cfg, self.rw, self.randomize = Path(out_dir), cfg, reward, randomize
        self.pool = deque()                 # [sim, hand, home snapshot, misses in a row, kind, layout index]
        self.loads, self.since_load, self.replays = 0, 0, 0
        self.skill = Skill(cfg=SKILL | {"base": cfg["base"]})
        self.weights = None
        self.dump_dir, self.source, self.dumped, self.chunks = dump_dir and Path(dump_dir), source, [], 0
        self.tag = f"{os.getpid()}_{time.time_ns()}"       # unique per environment, also for two in one process

    # --------------------------------------------------------------------------------------------- layouts
    def _load(self):
        kinds = self.cfg["kinds"]
        p = None
        if self.weights:
            p = np.array([self.weights.get(k, 0.0) for k in kinds], float)
            p = p / p.sum() if p.sum() > 0 else None
        kind = kinds[int(self.rng.choice(len(kinds), p=p))]
        index = int(self.rng.integers(*self.cfg["seeds"]))
        layout = scenes.sample_layout(layout_rng(kind, index), kind, self.out)
        sim = StageSim(layout)
        if layout.fixtures:
            Path(layout.bddl).unlink()      # one file per fixture layout, read once
        self.pool.append([sim, E.Hand(sim.robot), sim.snapshot(), 0, kind, index])
        self.loads += 1
        self.since_load = 0

    def _drop(self, slot):
        slot[0].close()
        self.pool.remove(slot)

    def _slot(self):
        if not self.pool or self.since_load >= self.cfg["starts_per_layout"]:
            self._load()
            if len(self.pool) > self.cfg["pool"]:
                self._drop(self.pool[0])
        return self.pool[int(self.rng.integers(len(self.pool)))]

    # --------------------------------------------------------------------------------------------- episode
    def reset(self, arg=1.0):
        scale, weights = arg if isinstance(arg, tuple) else (arg, None)
        self.weights = weights or self.weights
        if self.replays:                    # the start of the last episode once more (`scale` is that start's)
            self.replays -= 1
            self._again()
            return self.obs, self._critic(), dict(self.info)
        while True:
            slot = self._slot()
            if self._start(slot, scale):
                slot[3] = 0
                self.since_load += 1
                self.replays = self.cfg["episodes_per_start"] - 1
                io = self.io if self.randomize else None
                self.start = (self.sim.snapshot(), self.obs, io and (list(io.buf), io.bias),
                              copy.deepcopy({k: v for k, v in vars(self.skill).items() if k not in _SKILL_REFS}))
                return self.obs, self._critic(), dict(self.info)
            slot[3] += 1
            if slot[3] >= self.cfg["misses"]:
                self._drop(slot)

    def _start(self, slot, scale):
        cfg, rng = self.cfg, self.rng
        sim, hand, home, _, kind, index = slot
        part = sim.parts[int(rng.integers(len(sim.parts)))]
        modes = part_modes(part)
        mode = "turn" if modes == ("turn",) else "pull" if len(modes) > 1 and rng.random() < cfg["pull_share"] \
            else "push"
        inject = None
        if rng.random() < cfg["inject"]:
            inject = str(rng.choice([k for k in INJECT if mode != "push" or k != "slip"]))
        sim.restore(home)
        slip = rng.uniform(*cfg["slip_friction"]) if inject == "slip" else None
        block = cfg["block"] if inject == "blocked" else None
        phys = sim.randomize(rng, part, cfg, slip, block) if self.randomize else None

        def opening(p):
            span = cfg["open_slide"] if p.kind == "slide" else np.radians(cfg["open_hinge"])
            return p.closed + np.sign(p.open - p.closed) * min(rng.uniform(*span), abs(p.open - p.closed))

        others = rng.random() < cfg["others_open"]
        for other in sim.parts:
            sim.set_joint(other.joint, opening(other) if others and other is not part else other.closed)
        q0 = opening(part) if mode == "push" else part.closed
        sim.set_joint(part.joint, q0)
        base = sim.snapshot()
        if mode == "push":
            c, n, left = E.contact_face(part, q0)
            lever = left / abs(q0 - part.closed)
            dc, dn, dt = E.sample_error(rng, n, cfg["err_widen"] * scale)
            if inject == "no_contact":
                t = _unit(dc) if np.linalg.norm(dc) > 1e-9 else _unit(np.cross(n, rng.normal(size=3)))
                u, v = part.face_u, np.cross(part.face_normal, part.face_u)
                dc = (abs(t @ u) * part.half_extents[0] + abs(t @ v) * part.half_extents[1]
                      + rng.uniform(*cfg["off_part"])) * t
            n_b = rotvec_to_mat(dn) @ n
            if inject == "wrong_axis":
                k = _unit(np.cross(n, rng.normal(size=3)))
                n_b = rotvec_to_mat(k * np.radians(rng.uniform(*cfg["wrong_angle"]))) @ n
            poses = E.press_poses(c + dc, n_b)
            standoff = rng.uniform(*cfg["standoff"])
            for family in rng.permutation(list(poses)):
                cands = poses[family]
                if rng.random() < cfg["shuffle"]:
                    cands = [cands[i] for i in rng.permutation(len(cands))]
                k = enter(sim, hand, base, cands, standoff)
                if k is not None:
                    break
            else:
                return False
            a = cands[k][1]
            io = Sensors(sim.robot, rng, cfg) if self.randomize else sim.robot
            R0, p0 = sim.robot.eef_pose()
            goal = Goal("push", -n_b, travel=left + dt)
            contract = Contract("move_part", "tips", (part.name,), [Entry(R0, p0, a, (0.0, standoff))], goal)
            way_in, goal_q, reached = standoff, part.closed, E.closed_test(sim, part, lever)
            reached1 = lambda x: bool(reached(np.array([x]))[0])        # noqa: E731
            err = {}
        else:
            # A grip is placed with the pads a few cm around the handle, so a large error often leaves no pose; a
            # few draws are tried, or pulls would fade from the data as the curriculum widens the error.
            for _ in range(cfg["grip_draws"]):
                cands, d_b, ap_b, err = believe_grip(part, mode, rng, scale, cfg, inject)
                order = rng.permutation(len(cands))
                k = enter_grip(sim, base, [cands[i] for i in order], finger_for(handle(part)[4], cfg))
                if k is not None:
                    break
            else:
                return False
            family = f"roll{order[k]}"
            goal_q, reached1 = goal_value(part, mode, rng, sim)
            hp = handle(part)[0]
            lever = 1.0 if mode == "turn" else lever_at(part, hp)
            want = abs(goal_q - q0) * lever
            dt = rng.uniform(-1, 1) * cfg["err_widen"] * scale * (cfg["err_turn"] if mode == "turn"
                                                                    else E.CFG["err_travel"])
            io = Sensors(sim.robot, rng, cfg) if self.randomize else sim.robot
            R0, p0 = sim.robot.eef_pose()
            goal = Goal(mode, d_b, axis_point=ap_b if mode == "turn" else None, travel=max(0.0, want + dt))
            contract = Contract("move_part", "pads", (part.name,), [Entry(R0, p0, R0[:, 2], (0.0, 0.0))], goal)
            way_in = 0.0
        rec = E.Recorder(sim, part)             # for its contact classes and the force reference at the start
        self.sim, self.part, self.touched = sim, part, rec.touched
        # what the way in (a push) or the closing pads touch belongs to the contact: watched step by step
        self.way_touch = [False, False]         # the part, anything outside the part and its fixture

        def watch(rb, obs):
            f = self._flags()
            self.way_touch[0] |= self._part_touch
            self.way_touch[1] |= bool(f[3])
        sim.robot.on_step = watch
        self.obs = self.skill.start(io, contract, max_steps=round(cfg["time"][mode] / SKILL["dt"]), way_in=way_in)
        sim.robot.on_step = None
        if self.skill.stop is not None:         # the way in ended on a limit: no contact for the policy to run
            return False
        self.lever, self.mode, self.inject = lever, mode, inject
        self.goal_q, self.reached, self.q_start, self.contract = goal_q, reached1, q0, contract
        self.sgn = float(np.sign(goal_q - q0)) or float(np.sign(part.closed - part.open))
        self.F0 = rec.F0
        self.phys, self.io = phys, io
        self._begin()
        # progress is counted from where the way in left the part (a light part is pushed along on the way)
        floor = self.rw["travel_floor"] * (10.0 if mode == "turn" else 1.0)
        self.left0 = max(self.left, floor)
        self.info = {"kind": kind, "layout": index, "part": part.name, "family": str(family), "scale": float(scale),
                     "mode": mode, "inject": inject or "", "left0": float(self._left_at(q0)),
                     "way_in": int(self.skill.k)} | err
        return True

    def _again(self):
        """Back to the moment the policy took over: simulator, skill and sensor buffer as they were. The sensor
        noise and the policy's own noise go on from where they are, so the episode is a new one."""
        snap, self.obs, sensors, skill = self.start
        self.sim.restore(snap)
        vars(self.skill).update(copy.deepcopy(skill))
        if sensors:
            self.io.buf, self.io.bias = deque(sensors[0], maxlen=self.io.delay + 1), sensors[1]
        self._begin()

    def _begin(self):
        self.left = self._left()
        self.ok = self.ever = self.reached(self.sim.joint(self.part.joint))
        self.ret, self.count = 0.0, np.zeros(3)             # steps with: body on fixture, outside contact, > soft force
        self.t = 0                                          # policy steps (a force stop holds three control steps more)
        self.held = 0                                       # steps with both fingers on the part
        self.prev = np.zeros(9)                             # the last motion numbers (the change cost of E0)
        self.touched_part, self.outside_any = self.way_touch
        self.dump_this = self.dump_dir is not None and self.rng.random() < self.cfg["dump_share"]

    def step(self, action):
        rw = self.rw
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.obs = self.skill.step(a)
        stop = self.skill.stop
        left = self._left()
        q = self.sim.joint(self.part.joint)
        self.ok = self.reached(q)
        self.ever |= self.ok
        flags, force = self._flags(), np.linalg.norm(self.sim.robot.wrench()[0] - self.F0)
        over = max(0.0, force / rw["force_soft"] - 1.0)
        cost = (rw["residual"] * np.mean(a[:9] ** 2) if self.cfg["base"]
                else rw["change"] * np.mean((a[:9] - self.prev) ** 2))
        self.prev = a[:9]
        r = (rw["progress"] * (self.left - left) / self.left0 - rw["force"] * over - rw["body"] * flags[2]
             - rw["outside"] * flags[3] - rw["step"] - cost)
        if stop == "done":
            r += rw["done_closed"] if self.ok else rw["done_giveup"] if self.inject else rw["done_open"]
        self.left = left
        self.ret += r
        self.count += (flags[2], flags[3], over > 0)
        self.held += self._grip == 2
        self.touched_part |= bool(flags[0] or self._part_touch)
        self.outside_any |= bool(flags[3])
        self.t += 1
        info = {}
        if stop is not None:
            label, travel = self._label(q)
            info = self.info | {
                "stop": stop, "steps": self.t, "return": float(self.ret),
                "success": bool(self.ever), "closed_at_end": self.ok, "left": float(left), "label": label,
                "travel": travel, "body_steps": int(self.count[0]), "outside_steps": int(self.count[1]),
                "force_steps": int(self.count[2]), "retire": self.loads >= self.cfg["max_loads"]}
            if self.dump_this:
                self._dump(info)
        return self.obs, self._critic(), float(r), stop is not None, info

    # ----------------------------------------------------------------------------------------------- truth
    def _left_at(self, q):
        # one-sided: a LIBERO drawer can be pushed 1 cm past its closed value, which is closed, not travel left
        return max(0.0, (self.goal_q - q) * self.sgn) * self.lever

    def _left(self):
        return self._left_at(self.sim.joint(self.part.joint))

    def _flags(self):
        """fingers on the part, fingers on the rest of the fixture, hand body or arm link on the rest of the fixture,
        any robot body on anything else. Also sets _grip (how many fingers touch the part) and _part_touch (any
        robot body on the part)."""
        c = np.zeros((3, 3), bool)
        fingers = set()
        for a, b, _ in self.sim.robot_contacts():
            r = E._robot(a)
            if r >= 0:
                t = self.touched(b)
                c[r, t] = True
                if r == 0 and t == 0:
                    fingers.add("left" if ("left" in a or "joint1" in a) else "right")
        self._grip, self._part_touch = len(fingers), bool(c[:, 0].any())
        return np.array([c[0, 0], c[0, 1], c[1:, 1].any(), c[:, 2].any()], float)

    def _label(self, q):
        """Detector class of the contact from simulator truth (DETECTOR_CLASSES), and the part's travel toward the
        goal since the episode started (m at the contact point or handle; rad for a turn)."""
        cfg = self.cfg
        travel = float((q - self.q_start) * self.sgn * self.lever)
        want = abs(self.goal_q - self.q_start) * self.lever
        unit = cfg["effect"]["rad"] if self.mode == "turn" else cfg["effect"]["m"]
        if self.ok:
            label = "reached"
        elif self.outside_any:
            label = "outside"
        elif not self.touched_part:
            label = "no_contact"
        elif self.mode != "push" and self.held >= cfg["held_steps"] and not self._part_touch:
            label = "lost_grip"
        elif travel < min(unit, cfg["effect"]["share"] * want):
            label = "no_effect"
        else:
            label = "short"
        assert label in DETECTOR_CLASSES
        return label, travel

    def _dump(self, info):
        g = self.contract.goal
        tr = self.skill.trace.arrays()
        self.dumped.append({
            "trace": {k: v.astype(np.float32) for k, v in tr.items()}, "touch": int(self.skill.trace.touch),
            "goal": {"mode": g.mode, "direction": g.direction, "travel": float(g.travel),
                     "axis_point": None if g.axis_point is None else g.axis_point},
            "label": info["label"], "travel": info["travel"], "wanted": float(abs(self.goal_q - self.q_start) * self.lever),
            "source": self.source} | {k: info[k] for k in ("kind", "layout", "mode", "inject", "stop", "steps", "family",
                                                          "scale", "part")})
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

    def _critic(self):
        """Simulator truth for the critic only: travel left, the true direction of motion (a turn: its axis) and face
        normal in the contact frame, contacts, the episode's physics and sensor settings, mode and injection."""
        sim, part, sk = self.sim, self.part, self.skill
        scenes.place_parts([part], sim.robot.m, sim.robot.d)
        if self.mode == "turn":
            way = part.axis * self.sgn
        elif self.mode == "pull":
            way = opening_direction(part, sim.joint(part.joint))
        else:
            s = np.sign(part.closed - part.open)
            way = (part.axis * s if part.kind == "slide"
                   else _unit(np.cross(part.axis * s, part.face_point - part.axis_point)))
        f = self.phys or {"damping": 1.0, "mass": 1.0, "finger_friction": 1.0, "gain": 1.0, "loss": 0.0}
        io = self.io if self.randomize else None
        flags = self._flags()
        return np.hstack([
            self.left / self.left0, 10.0 * self.left, sk.R_c.T @ way, sk.R_c.T @ part.face_normal, flags,
            np.log([f["damping"], f["mass"], f["finger_friction"], f["gain"]]),
            min(f["loss"] / self.cfg["loss_add"][part.kind], 10.0),
            (io.delay / 2.0, io.sigma[0]) if io else (0.0, 0.0), part.kind == "hinge",
            np.linalg.norm(sim.robot.wrench()[0] - self.F0) / self.rw["force_soft"], self.ok,
            np.array(MODES) == self.mode, np.array(INJECT) == self.inject, self._grip / 2.0])

    def close(self):
        if self.dump_dir is not None:
            self._flush()
        for slot in list(self.pool):
            self._drop(slot)


__all__ = ["StageEnv", "Sensors", "CFG", "REWARD", "TRAIN_KINDS", "UNSEEN_KINDS", "VAL_SEED", "OBS_DIM", "ACT_DIM",
           "CRIT_DIM", "INJECT", "layout_rng", "handle", "grip_poses", "believe_grip", "goal_value", "enter_grip",
           "part_modes", "lever_at", "opening_direction", "finger_for"]
