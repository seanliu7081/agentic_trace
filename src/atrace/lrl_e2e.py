"""lrl end-to-end contact for moving parts (LRL_PLAN.md M1E): the run-time part that the teacher's training, the
student's distillation and, later, the harness share. One action space, one frame, one safety wrapper.

The hand is handed over 10-20 cm from the part S1 named (start_region). From there a policy chooses the contact
itself (which face, push or hold, how to hold) and moves the part until it raises done. No base law: every step the
policy sets a bounded change of the hand pose, the gripper and the done flag; the wrapper here enforces the limits.

Frame A (Frame): origin at the believed part's centroid, x horizontal from the part towards the hand's start, z up.
Everything a policy sees from the robot (proprio) and every action is expressed in A, so a policy is the same
function wherever the part stands and however it is turned about the vertical.

Action (ACT_DIM = 8), each number in [-1, 1] (the wrapper caps the lead along the felt force, f_cap):
  0-2  position lead in A: the commanded hand position is the measured one plus E2E_CFG["dp"] * a * |a| per axis.
       A target that leads the measured hand is what moves it and what presses (impedance: about 9 N per mm of lead
       on a stop, 5 cm/s of free motion per cm of lead, probe_lead.py); the signed square gives millimetres of lead
       near 0 (force) and centimetres near 1 (transit).
  3-5  orientation lead in A: rotation vector E2E_CFG["dr"] * a * |a| (deg per axis), applied to the measured hand
       orientation on the left (about A's axes).
  6    gripper: above +grip_band close, below -grip_band open, in between hold the fingers where they are (the
       Panda's finger target moves by a quarter of its range per control step while closing or opening).
  7    done: positive ends the contact (the policy's claim that the goal is reached, or its giving up).

Goal (Goal): effect open / close / turn_on / turn_off, with no per-fixture constant: open = towards the open end of
the joint's range, close = to its closed end, turn_on / turn_off = towards that end of a knob's range; the sense of a
turn when known (+1 counter-clockwise seen from outside the believed face, -1 clockwise, 0 unknown); an optional
amount (m at the part, or rad) when S1 gives one. The end of a range is felt (the part stops).

The skill (E2ESkill) talks to the robot only through `io` (SimRobot's step, eef_pose, wrench, finger_opening), as
lrl_skill.Skill does, so it runs unchanged in the stage simulator and in the harness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import frame_from_axes, mat_to_rotvec, rotvec_to_mat

E2E_CFG = {
    "dt": 0.05,              # R  s, control period (robosuite OSC_POSE at 20 Hz, motion.DT)
    # per-step bounds of the hand-pose change, at |a| = 1 per axis of A (signed square map, see the module text)
    "dp": 0.025,             # M  m of lead: 13 cm/s of free motion per axis (5.2 cm/s per cm of lead, probe_lead.py)
    "dr": 10.0,              # M  deg of lead: 42 deg/s of free rotation (probe_lead.py)
    "lead_max": 0.03,        # T  m: longest position lead (norm), so a transit cannot outrun the speed limit by much
    "grip_band": 0.33,       # T  |a| below which the gripper number holds the fingers where they are
    # limits of the safety wrapper
    "force": 40.0,           # T  N: tared, low-passed wrist force that stops the contact (lrl_contract.LIMITS)
    "force_steps": 4,        # T  consecutive steps above it that stop the contact. 4, not lrl_skill's 2: the
                             #    3-step mean keeps a single step's spike in its window for 3 steps, so 2 steps
                             #    over the limit can be one transient the cap has already undone; the stop is for
                             #    a force that lasts (35 of 112 contacts of a teacher ended on it, none an impact:
                             #    e1 NOTES.md, scratch/diag_force_stops.py)
    "backoff_steps": 3,      # T  steps the measured pose is held after a force stop: the lead and so the force go
    "f_touch": 5.0,          # T  N: tared force above which the hand is taken to press on something
    "f_cap": 30.0,           # T  N: the lead along the felt force is cut to what the impedance turns into at most
                             #    this force at the hand's present speed along it (see b_osc)
    "k_contact": 9000.0,     # M  N/m: force per lead on a stop (probe_lead.py: 9.3 N per mm; 3 N per mm on a
                             #    drawer front, probe_force_sign.py): a lead noise of a few mm otherwise spikes the
                             #    force by 50-100 N in one step (scratch/diag_force.py) and ends the contact
    "b_osc": 1730.0,         # M  N s/m: the controller's damping, k_contact / 5.2 per s (probe_lead.py: free motion
                             #    at 5.2 x the lead per s). The force of a lead L at speed v along it is about
                             #    k L - b v: a part that is moving takes the lead as speed, a stop as force. (The
                             #    first cap, (f_cap - |F|) / k, counted the force a lead makes as added to it: it
                             #    held a steady push or pull under f_cap / 2 and a resisting part to ~1 cm/s, e1
                             #    NOTES.md 2026-11-26.)
    "rot_contact": 4.0,      # T  deg: longest rotation lead while the hand presses on something (felt force over
                             #    f_touch): a turn of the pressed hand makes contact force the position cap does
                             #    not see (16 of those 35 stops followed a rotation number over 0.6, i.e. > 3.6 deg);
                             #    a held door turns with the hand at well under 1 deg per step, a knob is turned by
                             #    torque with little force
    "speed": 0.15,           # T  m/s: hand speed above which the position lead is scaled down
    "box_lo": (-0.40, -0.45, -0.30),    # T  m in A: the workspace box the commanded position is clipped to; it holds
    "box_hi": (0.60, 0.45, 0.45),       # T  the part (35 cm of drawer travel, a 55 cm door's arc) and the start region
    "time": 15.0,            # T  s: longest contact
    # observation
    "stack": 4,              # T  steps of proprio history
    "lowpass": 3,            # T  steps, moving average of the tared wrench (lrl_skill)
    "f_scale": 10.0,         # T  N, force unit of the observation
    "t_scale": 1.0,          # T  N m, torque unit of the observation
    "gap_scale": 10.0,       # T  1/m, finger gap unit of the observation
    "p_scale": 5.0,          # T  1/m, position unit of the observation (20 cm -> 1)
    "v_scale": 5.0,          # T  s/m, velocity unit of the observation
    "w_scale": 0.5,          # T  s/rad, angular velocity unit of the observation
    "amount_scale": 5.0,     # T  1/m of a wanted amount in metres (rad: 1)
    # frame A
    "min_horizontal": 0.05,  # T  m: a start nearer than this to the vertical through the part takes x from the
                             #    believed normal, then from the robot's base (a start above or below the part)
    "min_level": 0.3,        # T  level share of the believed normal below which it gives no x either
    # start region (start_region): candidate start poses, in the order TRACE is asked for them
    "start_dist": (0.15, 0.12, 0.18, 0.10, 0.20),   # T  m from the believed centroid along the believed normal
    "start_elev": (0.0, 40.0),                       # T  deg: the way out raised above the normal (a part low over
                                                     #    the table has no level start with open fingers)
    "start_lateral": (0.0, 0.04, -0.04),             # T  m along the believed face's level axis
    "start_rolls": (0.0, 90.0),                      # T  deg: closing axis level, then upright
}

EFFECTS = ("open", "close", "turn_on", "turn_off")
ACT_DIM = 8
ACT_NAMES = ("dx", "dy", "dz", "rx", "ry", "rz", "grip", "done")
GOAL_DIM = 8
STEP_DIM = 31            # per stacked step: p 3, R 6, v 3, w 3, force 3, torque 3, gap 2, previous action 8
PROPRIO_DIM = STEP_DIM * E2E_CFG["stack"] + 1 + GOAL_DIM      # stack, time left, goal

_FIELDS = (("p", 3), ("rv", 3), ("v", 3), ("w", 3), ("f", 3), ("t", 3), ("gap", 1),
           ("p_cmd", 3), ("rv_cmd", 3), ("action", ACT_DIM))
_WIDTH = sum(n for _, n in _FIELDS)


def _unit(x):
    return np.asarray(x, float) / np.linalg.norm(x)


# ---------------------------------------------------------------------------------------------------- belief, frame
@dataclass(eq=False)
class Belief:
    """The part as the harness believes it: S1's pointer segmented into surface points, their centroid and an
    estimated outward normal (unit, world). Either may be off: the normal by a wrong face (HANDOFF_ERROR.md)."""
    centroid: np.ndarray
    normal: np.ndarray
    points: np.ndarray | None = None

    def __post_init__(self):
        self.centroid, self.normal = np.asarray(self.centroid, float), _unit(self.normal)


@dataclass(eq=False)
class Frame:
    """Frame A: world from A is (R, o); R's columns are A's axes in the world."""
    R: np.ndarray
    o: np.ndarray

    def point(self, p):
        """World point(s) -> A."""
        return (np.asarray(p, float) - self.o) @ self.R

    def vec(self, v):
        """World vector(s) -> A."""
        return np.asarray(v, float) @ self.R

    def to_world(self, p):
        return np.asarray(p, float) @ self.R.T + self.o


def _level(v):
    v = np.asarray(v, float)
    return np.array([v[0], v[1], 0.0])


def frame_a(p_start, belief, base=None, cfg=E2E_CFG):
    """Frame A of a contact: origin at the believed centroid, z up, x horizontal from the part towards the hand's
    start position. A start (nearly) above or below the part takes x from the believed normal's level part, and
    when that is (nearly) vertical too, from the direction to the robot's base (base: its position, world; without
    it, world -x, the robot's side of a LIBERO table)."""
    c = belief.centroid
    h = _level(np.asarray(p_start, float) - c)
    if np.linalg.norm(h) < cfg["min_horizontal"]:
        h = _level(belief.normal)
        if np.linalg.norm(h) < cfg["min_level"]:
            h = _level(np.asarray(base, float) - c) if base is not None else np.array([-1.0, 0.0, 0.0])
    x, z = _unit(h), np.array([0.0, 0.0, 1.0])
    return Frame(np.stack([x, np.cross(z, x), z], 1), c.copy())


def start_region(belief, cfg=E2E_CFG, dists=None):
    """Candidate start poses (R, p), world, in the order TRACE is asked for them: the hand 10-20 cm out from the
    believed centroid along the believed normal, or along it raised by start_elev towards up; fingers pointing at
    the part (hand z = minus that way out), closing axis level or across; on the way out and a few cm to either side.
    A (nearly) upward normal has no raised way out. dists: distances instead of start_dist (a curriculum)."""
    n, c = belief.normal, belief.centroid
    up = np.array([0.0, 0.0, 1.0])
    level = abs(n[2]) < 0.9
    u = _unit(np.cross(n, up) if level else np.cross(n, [1.0, 0.0, 0.0]))
    k = _unit(up - (up @ n) * n) if level else None         # up, square to the normal
    out = []
    for d in (cfg["start_dist"] if dists is None else dists):
        for e in np.radians(cfg["start_elev"] if level else cfg["start_elev"][:1]):
            w = n if e == 0 else np.cos(e) * n + np.sin(e) * k
            v = np.cross(w, u)
            for lat in cfg["start_lateral"]:
                for roll in np.radians(cfg["start_rolls"]):
                    out.append((frame_from_axes(-w, np.cos(roll) * u + np.sin(roll) * v), c + d * w + lat * u))
    return out


# ------------------------------------------------------------------------------------------------------------ goal
@dataclass(eq=False)
class Goal:
    effect: str                     # one of EFFECTS
    sense: int = 0                  # turn only: +1 counter-clockwise about the believed outward normal, -1 clockwise
    amount: float | None = None     # wanted travel, m at the part or rad; None: to the end of the range
    unit: str = "m"                 # "m" or "rad"

    def __post_init__(self):
        assert self.effect in EFFECTS and self.sense in (-1, 0, 1) and self.unit in ("m", "rad")

    def vector(self, cfg=E2E_CFG):
        """(GOAL_DIM,): effect one-hot, sense, amount given, amount in m (scaled), amount in rad."""
        has = self.amount is not None
        return np.hstack([np.array(EFFECTS) == self.effect, self.sense, has,
                          cfg["amount_scale"] * self.amount if has and self.unit == "m" else 0.0,
                          self.amount if has and self.unit == "rad" else 0.0]).astype(float)


# ----------------------------------------------------------------------------------------------------------- skill
def action_to_command(a, R, p, frame, cfg=E2E_CFG):
    """The pose command of an action at the measured hand pose (R, p), before the speed and box limits:
    (R_cmd, lead) with lead the world position change."""
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    lead = frame.R @ (cfg["dp"] * a[:3] * np.abs(a[:3]))
    lead *= cfg["lead_max"] / max(np.linalg.norm(lead), cfg["lead_max"])
    R_cmd = rotvec_to_mat(frame.R @ (np.radians(cfg["dr"]) * a[3:6] * np.abs(a[3:6]))) @ R
    return R_cmd, lead


class Trace:
    """One contact, one row per control step: the action, the command it produced, and the hand as measured after
    that step (world frame; f and t tared, not low-passed). `frame` and `goal` say what the contact was for."""

    def __init__(self, frame, goal):
        self.frame, self.goal = frame, goal
        self.rows = []
        self.stop = None
        self.steps = 0

    def arrays(self):
        a = np.array(self.rows).reshape(-1, _WIDTH)
        edges = np.cumsum([0] + [n for _, n in _FIELDS])
        out = {name: a[:, i:j] for (name, _), i, j in zip(_FIELDS, edges[:-1], edges[1:])}
        out["gap"] = out["gap"][:, 0]
        return out


class E2ESkill:
    """start(io, frame, goal) and step(action) -> proprio observation (PROPRIO_DIM,); `stop` stays None while the
    contact runs, then names what ended it: done, force, guard, time. run(io, frame, goal, policy) is the loop."""

    def __init__(self, cfg=E2E_CFG):
        self.cfg = cfg

    def run(self, io, frame, goal, policy, guard=None, max_steps=None, tare=None):
        obs = self.start(io, frame, goal, guard, max_steps, tare)
        while self.stop is None:
            obs = self.step(policy(obs))
        return self.trace

    def start(self, io, frame, goal, guard=None, max_steps=None, tare=None):
        """tare: a wrench reading (force, torque), world, to tare against instead of the first reading."""
        c = self.cfg
        self.io, self.frame, self.goal, self.guard = io, frame, goal, guard
        n = int(round(c["time"] / c["dt"]))
        self.n_max = n if max_steps is None else min(n, int(max_steps))
        self.R, self.p = io.eef_pose()
        self.gap = io.finger_opening()
        self.w0 = np.hstack(io.wrench() if tare is None else tare).astype(float)
        self.wbuf = np.zeros((c["lowpass"], 6))
        self.goal_vec = goal.vector(c)
        self.box = (np.array(c["box_lo"]), np.array(c["box_hi"]))
        self.grip = 0.0
        self.a_prev = np.zeros(ACT_DIM)
        self.k, self.over, self.stop = 0, 0, None
        self.hist = np.zeros((c["stack"], STEP_DIM))
        self.trace = Trace(frame, goal)
        self.v = self.w = np.zeros(3)
        self.raw = np.zeros(6)
        self._sense()
        return self.obs()

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        if a[7] > 0:
            self.a_prev = a
            return self._end("done")
        R_cmd, lead = action_to_command(a, self.R, self.p, self.frame, c)
        lead *= c["speed"] / max(np.linalg.norm(self.v), c["speed"])
        f = np.linalg.norm(self.raw[:3])
        if f > c["f_touch"]:
            # the wrist reads the force the hand applies: the lead along it makes k L - b v of force (v the hand's
            # speed along it), so it is cut to (f_cap + b v) / k; the rest of the lead is free
            u = self.raw[:3] / f
            lead -= max(lead @ u - (c["f_cap"] + c["b_osc"] * max(self.v @ u, 0.0)) / c["k_contact"], 0.0) * u
            w = mat_to_rotvec(R_cmd @ self.R.T)
            R_cmd = rotvec_to_mat(w * min(1.0, np.radians(c["rot_contact"]) / max(np.linalg.norm(w), 1e-9))) @ self.R
        p_cmd = self.frame.to_world(np.clip(self.frame.point(self.p + lead), *self.box))
        band = c["grip_band"]
        self.grip = 1.0 if a[6] > band else -1.0 if a[6] < -band else 0.0
        self.a_prev = a
        self._advance(R_cmd, p_cmd, a)
        return self.obs()

    def obs(self):
        """The proprio observation: the stacked steps (oldest first), the share of time left, the goal."""
        return np.hstack([self.hist.ravel(), 1.0 - self.k / self.n_max, self.goal_vec])

    # ------------------------------------------------------------------------------------------------- internals
    def _advance(self, R_cmd, p_cmd, a):
        c = self.cfg
        self._tick(R_cmd, p_cmd, a)
        self.over = self.over + 1 if np.linalg.norm(self.lp[:3]) > c["force"] else 0
        if self.over >= c["force_steps"]:
            # a target at the measured pose carries no lead, so the impedance lets go of what it pressed
            for _ in range(c["backoff_steps"]):
                self._tick(self.R, self.p, np.zeros(ACT_DIM))
            self._end("force")
        elif self.guard is not None and self.guard(self.io):
            self._end("guard")
        elif self.k >= self.n_max:
            self._end("time")

    def _tick(self, R_cmd, p_cmd, a):
        self.io.step(R_cmd, p_cmd, self.grip)
        self.k += 1
        self._sense()
        self.trace.rows.append(np.hstack([self.p, mat_to_rotvec(self.R), self.v, self.w, self.raw, self.gap,
                                          p_cmd, mat_to_rotvec(R_cmd), a]))

    def _sense(self):
        c, A = self.cfg, self.frame
        R, p = self.io.eef_pose()
        gap = self.io.finger_opening()
        if self.k:
            self.v = (p - self.p) / c["dt"]
            self.w = mat_to_rotvec(R @ self.R.T) / c["dt"]
        gap_rate = (gap - self.gap) / c["dt"] if self.k else 0.0
        self.R, self.p, self.gap = R, p, gap
        self.raw = np.hstack(self.io.wrench()) - self.w0
        self.wbuf[self.k % c["lowpass"]] = self.raw
        self.lp = self.wbuf.mean(0)
        Ra = A.R.T @ R
        row = np.hstack([c["p_scale"] * A.point(p), Ra[:, 0], Ra[:, 1], c["v_scale"] * A.vec(self.v),
                         c["w_scale"] * A.vec(self.w), A.vec(self.lp[:3]) / c["f_scale"],
                         A.vec(self.lp[3:]) / c["t_scale"], c["gap_scale"] * gap, c["v_scale"] * gap_rate,
                         self.a_prev])
        self.hist = np.vstack([self.hist[1:], row])

    def _end(self, stop):
        self.stop = self.trace.stop = stop
        self.trace.steps = len(self.trace.rows)
        return self.obs()


__all__ = ["E2E_CFG", "EFFECTS", "ACT_DIM", "ACT_NAMES", "GOAL_DIM", "STEP_DIM", "PROPRIO_DIM", "Belief", "Frame",
           "frame_a", "start_region", "Goal", "action_to_command", "Trace", "E2ESkill"]
