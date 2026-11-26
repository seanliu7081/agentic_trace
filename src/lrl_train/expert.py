"""Privileged scripted expert of the M1E teacher env (LRL_PLAN.md M1E; e1 step 1, interface acceptance, and the
behaviour-cloning warm start of step 2).

Training-time only: it reads simulator truth (the part's geometry, its joint's axis, value and range, contacts) and
nothing under src/atrace may import it. It drives one contact through lrl_e2e.E2ESkill as a policy does, by the
action numbers of each step, from the teacher env's own starts (near, grasped, or the start region 10-20 cm out with
a right or a wrong face believed):

  plan     contact candidates by the part's handle kind and the goal's effect (pads across a bar or a knob handle,
           pads around a turning knob, closed fingertips on the face or the handle's outside for a close, pads on the
           edge of an ajar part without a handle), each a hand pose; the first whose pre-pose, contact pose and end of
           travel have an arm posture inside the joint limits that touches nothing (inverse kinematics, checked on
           a copy of the state) is taken
  route    out in front of the true face, across to the pre-pose, in along the approach (slow near the part)
  grasp    pads closed on the handle or edge; or, for a close, fingertips brought to the face until it is felt
  move     the hand kept where it is on the part while the part is carried along its own joint: a line for slides,
           the arc for hinges with the hand turning with it, a rotation about a knob's axis
  release  pads opened, the hand backed off; then done
  give up  done when the part has not moved for a while (an injected block) or the hold is lost too often

  python -m lrl_train.expert --episodes 32 --procs 80 --out <dir>       # acceptance: per kind, effect, start
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from atrace import manip
from atrace.geometry import frame_from_axes, mat_to_rotvec, rotvec_to_mat
from atrace.lrl_e2e import ACT_DIM, E2E_CFG, EFFECTS

from . import scenes
from .sim import CFG as SIM

CFG = {
    # contact poses
    "clear": 0.004,             # T  m: fingertips this far off a surface at a contact pose
    "margins": (0.012, 0.006),  # T  m: pads open this far beyond the handle or edge on each side, tried in order
    "tilts": (0.0, 30.0, 55.0, 70.0),  # T  deg: approach tilted down from level (upright faces), tried in order
                                       #    (steep ones keep a hand that closes upright off the table under a
                                       #    low bar)
    "edge_in": 0.012,           # T  m: pads' centre this far in from the free edge of a part without a handle
    "press_at": 0.65,           # T  share of the way from a door's hinge to its free edge a close presses at
    # route
    "depth_min": 0.03,          # T  m: the route crosses at least this much further out than the pre-pose
    "rises": (None, 0.0),       # T  routes tried: straight in to the pre-pose (a start in front of the face), a
                                #    box path crossing at the start's height (one 12 cm higher found nothing more:
                                #    accept/v5)
    "max_cands": 16,            # T  candidates tried in priority order (a start with no contact cost 16-18 s of
                                #    checks over 60: scratch/prof_plan.py)
    "pre": 0.05,                # T  m: pre-pose out along the approach
    "arrive": 0.008,            # T  m: a waypoint is reached within this
    "arrive_rot": 6.0,          # T  deg
    "step": 0.02,               # T  m of lead per step in free space (about 10 cm/s, probe_lead.py)
    "slow": 0.006,              # T  m of lead per step on the last segment into the contact
    "rot_step": 8.0,            # T  deg of rotation lead per step
    # contact and motion
    "close_steps": 9,           # T  steps the pads close for (the finger target crosses its range in 8)
    "press_lead": 0.003,        # T  m: a press keeps this much lead into the face
    "touch_force": 4.0,         # T  N: a press has met the face
    "lead": 0.012,              # T  m: lead along the part's path per step (about 6 cm/s free)
    "lead_min": 0.003,          # T  m: near an amount's end
    "slow_gain": 0.5,           # T  lead per m of travel left to an amount's end
    "turn": 8.0,                # T  deg per step of a knob's turn lead
    "release_steps": 4,         # T
    "back": 0.02,               # T  m: backed off after the release
    "stall_steps": 40,          # T  steps of the move with under stall_travel of progress: give up (a block)
    "stall_travel": 0.002,      # T  m at the hand (knob: 0.02 rad)
    "lost_steps": 6,            # T  steps without both fingers on the part while carrying it: the hold is lost
    "retries": 2,               # T  new plans after a lost hold or a missed grasp
    "ik_random": 6,             # T  random IK seeds per pose
    "arc_checks": 4,            # T  postures checked along the part's path (its end and points before)
    "route_step": 0.04,         # T  m (or 20 deg) between the postures a route is checked at: the controller moves
                                #    the hand in Cartesian space, so a route whose posture chain (each one seeded
                                #    with the last, as the arm will move) leaves the joint limits gets stuck there
    "route_rot": 20.0,          # T  deg
}


def _unit(x):
    x = np.asarray(x, float)
    return x / np.linalg.norm(x)


def _level(v):
    return np.array([v[0], v[1], 0.0])


@dataclass
class Cand:
    """A contact: hand pose at it (R, p: grip site), the finger joint value on the way in, the approach (hand z), the
    mode (pinch: pads closed on it; press: fingertips pushed on it) and a name for the tables."""
    name: str
    R: np.ndarray
    p: np.ndarray
    finger: float
    a: np.ndarray
    mode: str
    group: tuple = ()                                   # priority tier; within a tier the least turn comes first
    q: np.ndarray = field(default=None, repr=False)     # the arm posture found for it
    way: list = field(default=None, repr=False)         # the route found to it


class Expert:
    """expert = Expert(env); after every env.reset: expert.start(); then a = expert() each step until the episode
    ends. Reads env.sim, env.part, env.skill, env.frame and the goal fields of TeacherEnv."""

    def __init__(self, env, cfg=CFG):
        self.env, self.c = env, cfg
        self.d2 = None
        self.memo = {}          # the contact chosen at the last start (env replays a start episodes_per_start times)

    # ---------------------------------------------------------------------------------------------- planning
    def start(self):
        env = self.env
        rb = env.sim.robot
        if self.d2 is None or self.d2_model is not rb.m:
            self.d2, self.d2_model = mujoco.MjData(rb.m), rb.m
            self.touchy = np.array([n.startswith(("robot0_link", "robot0_right_hand", "gripper0_"))
                                    for n in env.sim._body])
            self.finger_body = np.array(["finger" in n for n in env.sim._body])
        self.retries, self.why, self.cand, self.k = 0, "", None, 0
        self.fails = Counter()          # why candidates were passed over: route or path, waypoint, ik or touch
        self.x_best, self.stall = None, 0
        near = env.info["near"]
        R, p = rb.eef_pose()
        if not near:
            self._plan()
            return
        mode = "press" if env.effect == "close" else "pinch"
        self.cand = Cand(f"near_{mode}" if near is True else "grasped", R, p, rb.finger_opening(), R[:, 2], mode)
        if not self._path_ok(R, p, rb.q()[:7], 0.0 if mode == "pinch" else rb.q()[7]):
            # the start's own contact cannot follow the part to the goal (a wrist at its limit on a door's arc):
            # let go, back off, and take a contact that can
            self.phase, self.k, self.why = "letgo", 0, "regrasp"
        elif near == "grasped":
            self._carry()
        else:
            self.phase, self.k = mode if mode == "press" else "grasp", 0

    def _plan(self):
        """Choose a contact and the route to it from where the hand is."""
        env, c = self.env, self.c
        rb = env.sim.robot
        q0 = rb.q()[:7]
        R0 = rb.eef_pose()[0]
        self.cand = None
        cands = sorted(self._candidates(), key=lambda x: (x.group, self._angle(x.R, R0)))[:c["max_cands"]]
        memo = self.memo.get(id(env.start)) if self.retries == 0 else None
        if memo is not None:            # a replay of the same start: the same contact, or none
            cands = [x for x in cands if x.name == memo]
        for cand in cands:
            if self._feasible(cand, q0):
                self.cand = cand
                break
        if self.retries == 0:
            self.memo = {id(env.start): self.cand.name if self.cand else ""}
        if self.cand is None:
            self.phase, self.why = "giveup", "no_candidate"
            return
        self.way, self.phase, self.k = self.cand.way, "route", 0

    def _route(self, cand, rise=0.0):
        """Waypoints (position, orientation, lead per step) to a contact: a box path out from the face (its outward
        level direction, or up) to the depth of the start, or a little beyond the pre-pose's, whichever is further,
        across at that depth, then in to the pre-pose and the contact; turned to the contact's orientation on the
        way across. (A point further out, 10 cm along the approach, had no posture for most low parts near the
        robot: accept/v3.)"""
        c, env = self.c, self.env
        R, p = env.sim.robot.eef_pose()
        pre = cand.p - cand.a * c["pre"]
        n = env.part.face_normal
        h = _unit(_level(n)) if np.linalg.norm(_level(n)) > 0.3 else np.array([0.0, 0.0, 1.0])
        if rise is None:            # straight in, turning on the way: only from in front of the face
            if p @ h < pre @ h + c["depth_min"]:
                return None
            return [(pre, cand.R, c["step"]), (cand.p, cand.R, c["slow"])]
        depth = max(p @ h, pre @ h + c["depth_min"])
        up = np.array([0.0, 0.0, rise])
        way = [(p + h * (depth - p @ h) + up, R, c["step"]), (pre + h * (depth - pre @ h) + up, cand.R, c["step"]),
               (pre, cand.R, c["step"]), (cand.p, cand.R, c["slow"])]
        keep = [way[0]]
        for w in way[1:]:
            if np.linalg.norm(w[0] - keep[-1][0]) > 0.01 or w is way[-1]:
                keep.append(w)
        return keep

    def _candidates(self):
        env = self.env
        part, effect = env.part, env.effect
        if env.rotary:
            return self._pinch_cands(knob=True)
        if effect == "close":
            if part.kind == "slide" and abs(part.axis @ part.face_normal) < 0.5:     # a panel sliding along its face
                return self._pinch_cands() + self._edge_cands()
            return self._press_cands() + (self._pinch_cands() if part.handle_point is not None else []) \
                + self._edge_cands()
        if part.handle_point is not None:
            return self._pinch_cands() + self._edge_cands()
        return self._edge_cands()

    def _approaches(self, n):
        """Approach directions (hand z) onto a face with outward normal n: level first, then tilted down."""
        if abs(n[2]) > 0.7:
            return [(0.0, -n)]
        h = _unit(_level(n))
        return [(t, _unit(-np.cos(np.radians(t)) * h - np.sin(np.radians(t)) * np.array([0.0, 0.0, 1.0])))
                for t in self.c["tilts"]]

    def _pinch_cands(self, knob=False):
        """Pads around the handle (a bar, low bar, lip or knob handle; or a turning knob), closing across its long
        axis; the grip site at the handle's centre unless the fingertips would then reach the surface behind."""
        part, c = self.env.part, self.c
        n, hp, e = part.face_normal, part.handle_point, part.handle_axis
        hl, hw = part.handle_half
        round_ = abs(hl - hw) < 0.3 * max(hl, hw, 1e-6)
        behind = part.axis_point if knob else part.face_point
        stand = float((hp - behind) @ n)
        across = 2.0 * hw
        out = []
        for mi, margin in enumerate(c["margins"]):
            finger = float(np.clip(across / 2 + margin - manip.PAD_OFFSET, 0.0, SIM["finger_open"]))
            for ti, (tilt, a) in enumerate(self._approaches(n)):
                # back off along the approach until the fingertips stay `clear` off the surface behind
                cos = max(-(a @ n), 0.2)
                p = hp - a * max(0.0, manip.TIP_Z - (stand - c["clear"]) / cos)
                x0 = np.cross(n, e)
                xs = [x0, -x0]
                if round_:
                    x1 = np.cross(a, x0)
                    xs += [x1, -x1]
                for k, x in enumerate(xs):
                    if np.linalg.norm(x - (x @ a) * a) < 1e-3:
                        continue
                    out.append(Cand(f"pinch_t{tilt:.0f}_m{margin * 1000:.0f}_r{k}", frame_from_axes(a, x), p, finger,
                                    a, "pinch", (1, mi, ti)))
        return out

    def _press_cands(self):
        """Closed fingertips on the face for a close: at the face centre (a door: on its outer face towards the free
        edge), or on the outside of a handle in front of it; closing axis level, then upright."""
        part, c = self.env.part, self.c
        n, fp, u = part.face_normal, part.face_point, part.face_u
        v = np.cross(n, u)
        pts = []
        if part.kind == "hinge":
            r = fp - part.axis_point
            r = r - (r @ part.axis) * part.axis
            k = _unit(r)
            reach = abs(k @ u) * part.half_extents[0] + abs(k @ v) * part.half_extents[1]
            pts.append(fp + k * ((c["press_at"] - 0.5) * 2.0 * reach))
        pts.append(fp)
        if part.handle_point is not None and part.handle_kind != "low_bar":
            pts.append(part.handle_point + n * (part.protrusion - (part.handle_point - fp) @ n))
        pts += [fp + 0.4 * part.half_extents[1] * v, fp + 0.4 * part.half_extents[0] * u,
                fp - 0.4 * part.half_extents[0] * u]
        out = []
        up = np.array([0.0, 0.0, 1.0])
        for i, pt in enumerate(pts):
            for ti, (tilt, a) in enumerate(self._approaches(n)):
                # the press phase moves in until the face is felt: start a little out
                p = pt - a * (manip.TIP_Z + c["clear"] + 0.008)
                level = np.cross(up, a) if abs(a[2]) < 0.95 else u
                for k, x in enumerate((level, np.cross(a, level))):     # hand's width level, then upright
                    out.append(Cand(f"press_p{i}_t{tilt:.0f}_r{k}", frame_from_axes(a, x), p, 0.0, a, "press",
                                    (0, i, ti)))
        return out

    def _edge_cands(self):
        """Pads on the free edge of the part's front, closing across its thickness: a door's edge away from the
        hinge, a drawer front's top edge, a sliding panel's trailing edge."""
        part, c = self.env.part, self.c
        n, fp, u = part.face_normal, part.face_point, part.face_u
        v = np.cross(n, u)
        if part.kind == "hinge":
            r = fp - part.axis_point
            r = r - (r @ part.axis) * part.axis
            k = _unit(r)
        elif abs(part.axis @ n) < 0.5:
            k = -part.axis * self.env.s_open * self.env.dirn          # trailing edge of a slider
        else:
            k = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.7 else v   # a drawer front's top edge
        reach = abs(k @ u) * part.half_extents[0] + abs(k @ v) * part.half_extents[1]
        edge = fp + k * reach
        thick = scenes.CFG["panel_t"]
        mid = edge - n * thick / 2
        out = []
        for mi, margin in enumerate(c["margins"]):
            finger = float(np.clip(thick / 2 + margin - manip.PAD_OFFSET, 0.0, SIM["finger_open"]))
            for ti, t in enumerate((0.0, 35.0)):
                # approach into the edge (along -k), tilted to come from in front of the face by t; the pads' centre
                # edge_in inside the edge, the closing axis across the panel's thickness
                a = _unit(-np.cos(np.radians(t)) * k - np.sin(np.radians(t)) * n)
                p = mid + a * c["edge_in"]
                for s in (1, -1):
                    out.append(Cand(f"edge_t{t:.0f}_m{margin * 1000:.0f}_r{s}", frame_from_axes(a, s * n), p, finger,
                                    a, "pinch", (2, mi, ti)))
        return out

    # -------------------------------------------------------------------------------------- feasibility checks
    def _clear(self, q, finger, allow_part, joint):
        env = self.env
        rb, m, d2 = env.sim.robot, env.sim.robot.m, self.d2
        d2.qpos[:] = rb.d.qpos
        d2.qpos[rb.model.qpos_adr] = rb.model.full_q(q, finger)[0]
        if joint is not None:
            d2.qpos[env.sim._adr(env.part.joint)[0]] = joint
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_collision(m, d2)
        b = m.geom_bodyid[d2.contact.geom[:d2.ncon]].reshape(-1, 2)
        robot = env.sim._is_robot[b]
        sel = robot[:, 0] ^ robot[:, 1]
        mine = np.where(robot[:, 0], b[:, 0], b[:, 1])[sel]
        other = np.where(robot[:, 0], b[:, 1], b[:, 0])[sel]
        bad = self.touchy[mine]
        if allow_part:
            bad &= ~(env.geo["own"][other] & self.finger_body[mine])
        return not bad.any()

    def _feasible(self, cand, q0):
        """A route to the contact (of those tried) and the part's travel to the goal have a chain of postures (each
        found from the last, inside the joint limits), touching nothing at the waypoints (the contact: only the
        part). The route found is kept in cand.way."""
        for rise in self.c["rises"]:
            cand.way = self._route(cand, rise)
            q = cand.way and self._chain(cand, q0)
            if q is not None:
                cand.q = q
                return self._path_ok(cand.R, cand.p, q, 0.0 if cand.mode == "pinch" else cand.finger)
        return False

    def _chain(self, cand, q0):
        c = self.c
        q, (R0, p0) = q0, self.env.sim.robot.eef_pose()
        way = cand.way
        for i, (p1, R1, _) in enumerate(way):
            k = int(np.ceil(max(np.linalg.norm(p1 - p0) / c["route_step"], self._angle(R1, R0) / c["route_rot"], 1)))
            w = mat_to_rotvec(R1 @ R0.T)
            for j in range(1, k + 1):
                q = self._ik1(rotvec_to_mat(w * j / k) @ R0, p0 + (p1 - p0) * j / k, cand.finger, q)
                if q is None:
                    self.fails[f"route{i - len(way) + 1}_ik"] += 1
                    return None
            if not self._clear(q, cand.finger, i == len(way) - 1, None):
                self.fails[f"route{i - len(way) + 1}_touch"] += 1
                return None
            R0, p0 = R1, p1
        return q

    def _ik1(self, R, p, finger, q):
        """The posture for hand pose (R, p) found from q alone (the arm moving there continuously), inside the
        joint limits with limit_slack; None if it does not converge."""
        rb = self.env.sim.robot
        qs, ok, _, _ = rb.model.ik(R, p, q[None], q_rest=rb.posture, limit_margin=SIM["limit_slack"], finger=finger)
        return qs[0] if ok[0] else None

    def _path_ok(self, R, p, q, finger):
        """A hand at (R, p) on the part (posture q) can be carried with it to the goal: a chain of postures, the
        part set to its joint value at each, nothing but the part touched."""
        env = self.env
        x_end, x_now = self._x_goal(), env._x()
        for f in np.linspace(0.0, 1.0, self.c["arc_checks"] + 1)[1:]:
            x = x_now + f * (x_end - x_now)
            Rx, px = self._carried(R, p, (x - x_now) * env.s_open)
            q = self._ik1(Rx, px, finger, q)
            if q is None or not self._clear(q, finger, True, env.part.closed + env.s_open * x):
                self.fails[f"path{f:.2f}_{'ik' if q is None else 'touch'}"] += 1
                return False
        return True

    def _x_goal(self):
        """Where the part is taken: an amount's end, the closed end, or just past where an open end counts as
        reached (a door need not, and often cannot, be carried to the end of its range)."""
        env = self.env
        if env.amount is not None:
            return env.x0 + env.dirn * env.amount
        if env.dirn < 0:
            return 0.0
        tol = env.cfg["end_tol_door" if env.part.kind == "hinge" and not env.rotary else "end_tol"]
        return min(1.0, 1.05 - tol) * env.span

    # ------------------------------------------------------------------------------------------ part motion
    def _body(self):
        d = self.env.sim.robot.d
        b = self.env.geo["body"]
        return d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy()

    def _carried(self, R, p, dq):
        """Hand pose (R, p) carried along with the part when its joint value grows by dq."""
        part = self.env.part
        if part.kind == "slide":
            return R, p + part.axis * dq
        Rq = rotvec_to_mat(part.axis * dq)
        return Rq @ R, Rq @ (p - part.axis_point) + part.axis_point

    # ------------------------------------------------------------------------------------------------ acting
    def __call__(self):
        """The action of the present step. A phase that ends hands over to the next one within the same step."""
        env, c = self.env, self.c
        rb = env.sim.robot
        R, p = rb.eef_pose()
        self.k += 1
        ph = self.phase
        if ph == "giveup":
            return self._action(p, R, 0.0, done=True)
        if ph == "letgo":                          # open, back off along the approach, then plan
            if self.k <= c["release_steps"]:
                return self._toward(p, R, c["slow"], -1.0)
            if self.k <= c["release_steps"] + 4:
                return self._toward(p - R[:, 2] * c["back"], R, c["step"], 0.0)
            self._plan()
            return self()
        if ph == "route":
            target, R_t, step = self.way[0]
            if np.linalg.norm(target - p) < c["arrive"] and self._angle(R_t, R) < c["arrive_rot"]:
                self.way.pop(0)
                self.k = 1
                if not self.way:
                    self.phase, self.k = ("press" if self.cand.mode == "press" else "grasp"), 0
                    return self()
                target, R_t, step = self.way[0]
            if self.k > 60:                         # stuck on the way to one waypoint
                return self._replan("route_stuck")
            grip = 1.0 if self.cand.mode == "press" else self._finger_cmd(self.cand.finger)
            return self._toward(target, R_t, step, grip)
        if ph == "grasp":
            if self.k > c["close_steps"]:
                if env._grip == 2:
                    self._carry()
                    return self()
                return self._replan("missed")
            return self._toward(self.cand.p, self.cand.R, c["slow"], 1.0)
        if ph == "press":
            touched = env.flags[0] or np.linalg.norm(rb.wrench()[0] - env.F0) > c["touch_force"]
            if touched or self.k > 60:
                self._carry()
                return self()
            return self._toward(p + self.cand.a * c["slow"], self.cand.R, c["slow"], 1.0)
        if ph == "move":
            return self._move(R, p)
        # release: pads open where the hand is, then back off along the approach, then done
        if self.k <= c["release_steps"]:
            return self._toward(p, R, c["slow"], -1.0 if self.cand.mode == "pinch" else 0.0)
        if self.k <= c["release_steps"] + 3:
            return self._toward(p - self.cand.a * c["back"], R, c["slow"], 0.0)
        return self._action(p, R, 0.0, done=True)

    def _carry(self):
        """Start carrying the part: the hand's pose in the part body's frame is kept from here on."""
        R, p = self.env.sim.robot.eef_pose()
        Rb, xb = self._body()
        self.rel = (Rb.T @ R, Rb.T @ (p - xb))
        self.phase, self.k, self.lost, self.stall = "move", 0, 0, 0
        self.x_best = self.env._x()

    def _replan(self, why):
        self.why = why
        self.retries += 1
        if self.retries > self.c["retries"]:
            self.phase = "giveup"
        else:
            self._plan()
        return self()

    def _move(self, R, p):
        env, c = self.env, self.c
        part = env.part
        if env._reached():
            self.phase, self.k = "release", 0
            return self()
        x = env._x()
        # m at the hand per unit of joint travel (a knob: 0.1 m per rad, for the stall test only)
        lever = 1.0 if part.kind == "slide" else 0.1 if env.rotary else \
            max(np.linalg.norm(np.cross(part.axis, p - part.axis_point)), 0.05)
        if (x - self.x_best) * env.dirn * lever > c["stall_travel"]:
            self.x_best, self.stall = x, 0
        else:
            self.stall += 1
        if self.stall > c["stall_steps"]:
            self.why = "stalled"
            self.phase, self.k = "release", 0
            return self()
        if self.cand.mode == "pinch":
            self.lost = self.lost + 1 if env._grip < 2 else 0
            if self.lost > c["lost_steps"]:
                return self._replan("lost_grip")
        left = abs(self._x_goal() - x)
        sgn = env.s_open * env.dirn
        if env.rotary:
            dq = sgn * min(np.radians(c["turn"]), left + 0.02)
        else:
            L = c["lead"] if env.amount is None else float(np.clip(c["slow_gain"] * left * lever, c["lead_min"],
                                                                   c["lead"]))
            dq = sgn * L / lever
        Rb, xb = self._body()
        R_t, p_t = self._carried(Rb @ self.rel[0], xb + Rb @ self.rel[1], dq)
        if env.rotary:          # the turn leads the hand itself: pads that twist on the knob do not eat the lead
            R_t = rotvec_to_mat(part.axis * dq) @ R
        if self.cand.mode == "press":
            p_t = p_t + self.cand.a * c["press_lead"]
        return self._action(p_t, R_t, 1.0)

    # ------------------------------------------------------------------------------------------- actions
    @staticmethod
    def _angle(Ra, Rb):
        return float(np.degrees(np.linalg.norm(mat_to_rotvec(Ra @ Rb.T))))

    def _finger_cmd(self, finger):
        """Gripper number that brings the fingers' target to joint value `finger` (open 0.04, closed 0). The target
        moves 1/8 of its range per step while closing or opening (robosuite: 0.01 per physics step); the measured
        joint lags it, so the target is what is steered."""
        now = 0.02 + 0.02 * self.env.sim.robot.robot.gripper.current_action[0]
        return 1.0 if now > finger + 0.003 else -1.0 if now < finger - 0.003 else 0.0

    def _toward(self, target, R_t, step, grip):
        """A step towards (R_t, target): position and orientation by the same share of what is left, so the hand
        follows the straight, evenly turning path whose postures _chain checked."""
        rb = self.env.sim.robot
        R, p = rb.eef_pose()
        dp, w = target - p, mat_to_rotvec(R_t @ R.T)
        f = min(1.0, step / max(np.linalg.norm(dp), 1e-9), np.radians(self.c["rot_step"]) / max(np.linalg.norm(w), 1e-9))
        return self._action(p + f * dp, rotvec_to_mat(f * w) @ R, grip)

    def _action(self, p_cmd, R_cmd, grip, done=False):
        """The action numbers whose command, from the pose the skill measured, is (R_cmd, p_cmd) (the inverse of
        lrl_e2e.action_to_command, within its bounds)."""
        sk, A = self.env.skill, self.env.frame
        x = A.vec(p_cmd - sk.p) / E2E_CFG["dp"]
        r = A.vec(np.degrees(mat_to_rotvec(R_cmd @ sk.R.T))) / E2E_CFG["dr"]
        a = np.zeros(ACT_DIM)
        a[:3] = np.clip(np.sign(x) * np.sqrt(np.abs(x)), -1.0, 1.0)
        a[3:6] = np.clip(np.sign(r) * np.sqrt(np.abs(r)), -1.0, 1.0)
        a[6] = grip
        a[7] = 1.0 if done else -1.0
        return a


# ------------------------------------------------------------------------------------------ acceptance runs
def run_job(job):
    """Expert episodes of one kind from one start type: list of episode summaries plus the expert's choice."""
    from .teacher_env import CFG as ENV, TeacherEnv, VAL_SEED
    kind, start, k, n, out, extra = job
    over = extra | {"kinds": (kind,), "episodes_per_start": 1, "inject": 0.0,
            "seeds": (VAL_SEED + 1000 * k, VAL_SEED + 1000 * (k + 1))}
    if start == "region":
        over |= {"near_share": 0.0}
        scale = 1.0
    else:
        over |= {"near_share": 1.0, "near_grasped": 1.0 if start == "grasped" else 0.0}
        scale = 0.0
    env = TeacherEnv([11, k, len(start)], Path(out) / "bddl" / f"{kind}_{start}_{k}", ENV | over)
    ex = Expert(env)
    rows = []
    for _ in range(n):
        env.reset(scale)
        ex.start()
        while True:
            _, _, _, ended, info = env.step(ex())
            if ended:
                break
        rows.append(info | {"start": start, "cand": ex.cand.name if ex.cand else "", "why": ex.why,
                            "phase": ex.phase, "retries": ex.retries, "fails": dict(ex.fails)})
    env.close()
    return rows


def tables(rows):
    out = ["| kind | effect | " + " | ".join(("grasped", "near", "region right", "region wrong")) + " |",
           "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        s = r["start"] if r["start"] != "region" else f'region {"wrong" if r["wrong"] else "right"}'
        if r["start"] != "region" and r["near"] is False:
            s = "region right"                  # no near pose (an ajar part without a handle): a region start
        by[(r["kind"], r["effect"], s)].append(r)
    kinds = sorted({r["kind"] for r in rows})
    for kind in kinds:
        for e in EFFECTS:
            cells = []
            for s in ("grasped", "near", "region right", "region wrong"):
                v = by.get((kind, e, s), [])
                cells.append(f'{sum(x["success"] for x in v)}/{len(v)}' if v else "–")
            if any(c != "–" for c in cells):
                out.append(f"| {kind} | {e} | " + " | ".join(cells) + " |")
    fails = Counter((r["kind"], r["effect"], r["start"], r["handle"], r["label"], r["stop"], r["why"] or r["phase"])
                    for r in rows if not r["success"])
    out += ["", "Failures (kind, effect, start, handle, label, stop, expert's reason): count"]
    out += [f"- {k}: {v}" for k, v in fails.most_common(60)]
    stages = defaultdict(Counter)
    for r in rows:
        if r["why"] == "no_candidate":
            stages[(r["kind"], r["effect"], r["handle"])].update(r["fails"])
    out += ["", "No candidate: where the candidates failed (route-k: k-th waypoint from the contact; path f: share of "
            "the part's travel)"]
    out += [f"- {k}: {dict(v.most_common(4))}" for k, v in sorted(stages.items())]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=32, help="per kind and start type")
    ap.add_argument("--per-job", type=int, default=8)
    ap.add_argument("--procs", type=int, default=64)
    ap.add_argument("--kinds", nargs="+")
    ap.add_argument("--starts", nargs="+", default=["grasped", "near", "region"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--over", default="{}", help="JSON: teacher env settings to override")
    a = ap.parse_args()
    from .teacher_env import TEACHER_KINDS
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(kind, start, k, min(a.per_job, a.episodes - k * a.per_job), str(a.out), json.loads(a.over))
            for kind in (a.kinds or TEACHER_KINDS) for start in a.starts
            for k in range(-(-a.episodes // a.per_job))]
    rows = []
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=1) as pool, open(a.out / "episodes.jsonl", "w") as f:
        for i, part in enumerate(pool.imap_unordered(run_job, jobs)):
            rows += part
            f.writelines(json.dumps(r, default=float) + "\n" for r in part)
            f.flush()
            print(f"job {i + 1}/{len(jobs)}  episodes {len(rows)}", flush=True)
    (a.out / "tables.md").write_text(tables(rows) + "\n")
    print(tables(rows))


if __name__ == "__main__":
    main()
