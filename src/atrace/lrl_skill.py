"""lrl skill loop: one loop for every contact, at the control rate (LRL_PLAN.md 5.2).

Per step: read the hand, compute the base command of the mode, add the policy's bounded residual, clip to the
contract's tube and speed, step, record. The loop talks to the robot only through `io` (SimRobot's methods: step,
eef_pose, wrench, finger_opening), so it runs unchanged on a simulated hand.

The contact frame C is the hand pose at the first reading (origin p_c, axes R_c). Everything the policy sees or
commands is expressed in C, except the wrench, which is in the hand frame as a wrist sensor gives it.

No stall rule and no travel rule: ending a contact is the policy's done flag and the detector's job. The loop
itself stops only on a limit of the contract (force, guard, time).

CFG["base"] False switches the base law off (M1E step 1, the control run E0): no lead along the direction and no
base turn; the policy's position and orientation numbers, with the wider bounds free_p and free_r, are the whole
motion. The way in, the closing steps, the tube, the speed clip, the force stop and done stay as they are.
"""
from __future__ import annotations

import numpy as np

from .geometry import mat_to_rotvec, rotvec_to_mat

CFG = {
    "dt": 0.05,             # s, control period (motion.DT).                                                R
    "lead": 0.015,          # m, how far the command leads the hand (as articulate.translate).              T
    "turn_rate": 30.0,      # deg/s, base turn about the believed axis.                                     T
    "res_p": 0.01,          # m, position residual at |a| = 1, per axis.                                    T
    "res_r": 3.0,           # deg per step, orientation residual at |a| = 1, per axis.                      T
    "effort": (0.3, 2.0),   # lead scale at a = -1 and a = +1; 1 at a = 0.                                  T
    "steer": 5.0,           # deg per step, largest turn of the direction.                                  T
    "stack": 8,             # steps of observation the policy gets.                                         T
    "f_scale": 10.0,        # N, force unit of the observation.                                             T
    "t_scale": 1.0,         # N m, torque unit of the observation.                                          T
    "gap_scale": 10.0,      # 1/m, finger gap unit of the observation.                                      T
    "lowpass": 3,           # steps, moving average of the tared wrench.                                    T
    "close_steps": 12,      # steps the entry pose is held while the fingers close (pull, turn).            T
    "force_steps": 2,       # consecutive steps above the force limit that stop the contact.                T
    "backoff": 0.01,        # m, retreat along -d after a force stop.                                       T
    "backoff_steps": 3,     # steps the retreat is held (articulate.translate ends with 3 hold steps).      T
    "grip_band": 0.33,      # |a| below which the gripper action means "keep".                              T
    "approach_speed": 0.04,  # m/s of the way in (the guarded line of harness._press_stroke: 0.04-0.05).    T
    "contact_force": 8.0,   # N, tared force that ends the way in (as harness._press_stroke).               T
    "approach_over": 0.03,  # m, the way in goes at most this far past the believed contact.                T
    "base": True,           # the base law of the mode; False: the policy's numbers are the whole motion.   T
    "free_p": 0.03,         # m, position number at |a| = 1 with the base off: the largest lead
                            # (lead x effort[1]).                                                           T
    "free_r": 6.0,          # deg per step, orientation number at |a| = 1 with the base off: the largest
                            # base turn (turn_rate x dt x effort[1]) plus res_r.                            T
}

MODES = ("push", "pull", "turn")
OBS_STEP = 38
OBS_DIM = OBS_STEP * CFG["stack"]
ACT_DIM = 11

_FIELDS = (("p", 3), ("rv", 3), ("v", 3), ("w", 3), ("f", 3), ("t", 3), ("gap", 1),
           ("p_cmd", 3), ("rv_cmd", 3), ("action", ACT_DIM))
_WIDTH = sum(n for _, n in _FIELDS)


def neutral_action():
    """The action that leaves the base law alone."""
    return np.zeros(ACT_DIM)


class Trace:
    """Record of one contact. One row per control step: the action, the command it produced, and the hand as
    measured after that step (world frame; f and t are tared with the entry reading, not low-passed). The entry
    state itself is the contact frame (R_c, p_c) with zero velocity and zero tared wrench."""

    def __init__(self, R_c, p_c):
        self.R_c, self.p_c = R_c, p_c
        self.rows = []
        self.stop = None
        self.steps = 0
        self.touch = 0          # rows before this index are the way in; (R_c, p_c) is the pose at the touch

    def arrays(self):
        a = np.array(self.rows).reshape(-1, _WIDTH)
        edges = np.cumsum([0] + [n for _, n in _FIELDS])
        out = {name: a[:, i:j] for (name, _), i, j in zip(_FIELDS, edges[:-1], edges[1:])}
        out["gap"] = out["gap"][:, 0]
        return out


class Skill:
    """run(io, contract) executes one contact and returns its Trace. start() and step(action) are the same loop
    opened up for a trainer that supplies the actions itself: `stop` stays None while the contact runs."""

    def __init__(self, policy=None, cfg=CFG):
        self.policy, self.cfg = policy, cfg

    def run(self, io, contract, guard=None, max_steps=None, way_in=0.0, tare=None):
        obs = self.start(io, contract, guard, max_steps, way_in, tare)
        while self.stop is None:
            obs = self.step(neutral_action() if self.policy is None else self.policy(obs))
        return self.trace

    def start(self, io, contract, guard=None, max_steps=None, way_in=0.0, tare=None):
        """way_in: distance (m) from the hand to the believed contact along the entry's approach axis, where the
        hand-off left the hand; 0 when the hand already touches. tare: a wrench reading (force, torque), world frame,
        to tare against instead of the first reading: a contact that starts on an already loaded hand (a grip
        closed on the handle, a part leaning on the fingers) would otherwise read that load as zero."""
        c, g = self.cfg, contract.goal
        if g.mode not in MODES:
            raise NotImplementedError(f"lrl skill for mode {g.mode!r} comes with a later milestone")
        self.io, self.goal, self.lim, self.guard = io, g, contract.limits, guard
        n = int(round(self.lim["time"] / c["dt"]))
        self.n_max = n if max_steps is None else min(n, int(max_steps))
        R, p = io.eef_pose()
        self.R_c, self.p_c = R.copy(), p.copy()
        self.R, self.p, self.gap = self.R_c, self.p_c, io.finger_opening()
        self.w0 = np.hstack(io.wrench() if tare is None else tare).astype(float)
        self.wbuf = np.zeros((c["lowpass"], 6))
        self.d0 = g.direction.copy()
        self.d = self.d0.copy()
        # the believed path of a push or pull: the hand's progress along the steered direction, summed; with no
        # steer it is the line through p_c along d0
        self.p_path, self.p_seen = self.p_c.copy(), self.p_c.copy()
        self.R_s = np.eye(3)                  # the steer summed (a pull turns the hand with it)
        # Steer axes: fixed, perpendicular to d0, and a function of d0 as seen in C, so that the two steer
        # numbers keep one meaning for the policy whatever the world orientation of the contact.
        e1 = np.cross(self.d0, R[:, np.argmin(np.abs(R.T @ self.d0))])
        e1 /= np.linalg.norm(e1)
        self.e = np.stack([e1, np.cross(self.d0, e1)])
        self.R_t, self.p_t = self.R_c, self.p_c
        self.theta = 0.0                      # turn commanded so far
        self.angle = 0.0                      # turn measured so far
        self.grip = 1.0
        self.cmd = (self.R_c, self.p_c)
        self.k, self.over, self.stop = 0, 0, None
        self.hist = np.zeros((c["stack"], OBS_STEP))
        self.trace = Trace(self.R_c, self.p_c)
        self._sense()
        if way_in > 0:
            self._approach(contract.entries[0].approach, way_in)
        if g.mode != "push":
            # the policy has no say while the fingers close on the part, so it is not asked
            for _ in range(c["close_steps"]):
                if self.stop is not None:
                    break
                self._advance(self.R_c, self.p_c, neutral_action())
        return self.hist.ravel().copy()

    def _approach(self, axis, way_in):
        """The way in: along the entry's approach axis until the wrist feels the part, then the contact frame is
        re-anchored at the touch pose. The motion law and the tube of a contact are about the touch, not about
        where the hand-off happened to stop; the policy is not asked on the way."""
        c = self.cfg
        a = axis / np.linalg.norm(axis)
        step = c["approach_speed"] * c["dt"]
        p_ref = self.p_c.copy()
        for _ in range(int(np.ceil((way_in + c["approach_over"]) / step))):
            if self.stop is not None or np.linalg.norm(self.lp[:3]) > c["contact_force"]:
                break
            p_ref = p_ref + step * a
            self._advance(self.R_c, p_ref, neutral_action())
        if self.stop is None:
            self.R_c, self.p_c = self.R.copy(), self.p.copy()
            self.R_t, self.p_t, self.cmd = self.R_c, self.p_c, (self.R_c, self.p_c)
            self.p_path, self.p_seen = self.p_c.copy(), self.p_c.copy()
            self.trace.R_c, self.trace.p_c, self.trace.touch = self.R_c, self.p_c, len(self.trace.rows)
            self.angle = 0.0
            self.hist[:] = 0.0
            self._sense()

    def step(self, action):
        c, g, lim = self.cfg, self.goal, self.lim
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        if a[10] > 0:
            return self._end("done")
        lo, hi = c["effort"]
        slow = lim["speed"] / max(np.linalg.norm(self.v), lim["speed"])         # the speed clip
        effort = (1.0 + a[6] * (hi - 1.0 if a[6] >= 0 else 1.0 - lo)) * slow
        self.p_path = self.p_path + ((self.p - self.p_seen) @ self.d) * self.d
        self.p_seen = self.p
        s = a[7:9] / max(1.0, np.linalg.norm(a[7:9]))
        Rs = rotvec_to_mat(np.radians(c["steer"]) * (s @ self.e))
        self.d = Rs @ self.d
        if g.mode == "pull":
            # a held handle turns with the path it is pulled along (a door's handle on its arc): the steer turns the
            # hand with the direction, and the orientation tube follows (without, the pads slipped off a microwave
            # handle at ~40 deg even when steered along the true tangent: train2/NOTES.md)
            self.R_t, self.R_s = Rs @ self.R_t, Rs @ self.R_s
        if c["base"]:
            self.R_t = rotvec_to_mat(self.R_c @ (np.radians(c["res_r"]) * a[3:6])) @ self.R_t
            dp = self.R_c @ (c["res_p"] * a[:3])
        else:
            # the speed clip scales the policy's numbers as it scales the base law's lead and turn
            w = self.R_c @ (np.radians(c["free_r"]) * slow * a[3:6])
            self.R_t = rotvec_to_mat(w) @ self.R_t
            dp = self.R_c @ (c["free_p"] * slow * a[:3])
        if g.mode == "turn":
            # without the base law the turn about the believed axis is the policy's own: the believed path (where
            # the tube is, the point on the arc) follows that part of its rotation
            dth = np.radians(c["turn_rate"]) * c["dt"] * effort if c["base"] else w @ self.d
            Rk = rotvec_to_mat(dth * self.d)
            if c["base"]:
                self.R_t = Rk @ self.R_t
            self.p_t = g.axis_point + Rk @ (self.p_t - g.axis_point)
            self.theta += dth
            p_cmd = self.p_t + dp
            # the believed path of a turn is the rotation about the believed axis: the tube follows it
            R_ref = rotvec_to_mat(self.theta * self.d0) @ self.R_c
        else:
            p_cmd = self.p + (effort * c["lead"] * self.d if c["base"] else 0.0) + dp
            # The tube is about the steered path, so a door's arc can be followed by steering (the plan's "the steer
            # residual is how the policy follows the arc"); without steer this is the line along d0 through p_c.
            r = p_cmd - self.p_path
            lat = r - (r @ self.d) * self.d
            p_cmd = p_cmd - lat * (1.0 - lim["tube"] / max(np.linalg.norm(lat), lim["tube"]))
            R_ref = self.R_s @ self.R_c
        if g.mode != "push":                  # a push keeps the fingers closed
            band = c["grip_band"]
            self.grip = -1.0 if a[9] < -band else 1.0 if a[9] > band else self.grip
        # the clipped orientation is written back, so the target cannot wind up outside the tube
        e = mat_to_rotvec(self.R_t @ R_ref.T)
        cap = np.radians(lim["tube_angle"])
        self.R_t = rotvec_to_mat(e * (cap / max(np.linalg.norm(e), cap))) @ R_ref
        self._advance(self.R_t, p_cmd, a)
        return self.hist.ravel().copy()

    # ------------------------------------------------------------------ internals
    def _advance(self, R_cmd, p_cmd, a):
        c = self.cfg
        self._tick(R_cmd, p_cmd, a)
        self.over = self.over + 1 if np.linalg.norm(self.lp[:3]) > self.lim["force"] else 0
        if self.over >= c["force_steps"]:
            # for a turn d is an axis, not a way out: holding the measured pose is what unloads the part
            back = self.p - c["backoff"] * self.d * (self.goal.mode != "turn")
            R = self.R
            for _ in range(c["backoff_steps"]):
                self._tick(R, back, neutral_action())
            self._end("force")
        elif self.guard is not None and self.guard(self.io):
            self._end("guard")
        elif self.k >= self.n_max:
            self._end("time")

    def _tick(self, R_cmd, p_cmd, a):
        self.io.step(R_cmd, p_cmd, self.grip)
        self.cmd = (R_cmd, p_cmd)
        self.k += 1
        self._sense()
        self.trace.rows.append(np.hstack([self.p, mat_to_rotvec(self.R), self.v, self.w, self.raw, self.gap,
                                          p_cmd, mat_to_rotvec(R_cmd), a]))

    def _sense(self):
        c, g, Rc = self.cfg, self.goal, self.R_c
        R, p = self.io.eef_pose()
        gap = self.io.finger_opening()
        self.v = (p - self.p) / c["dt"]
        self.w = mat_to_rotvec(R @ self.R.T) / c["dt"]
        gap_rate = (gap - self.gap) / c["dt"]
        self.R, self.p, self.gap = R, p, gap
        self.raw = np.hstack(self.io.wrench()) - self.w0
        self.wbuf[self.k % c["lowpass"]] = self.raw
        self.lp = self.wbuf.mean(0)
        self.angle += self.w @ self.d0 * c["dt"]          # summed, so a turn past 180 deg does not wrap
        Rr = Rc.T @ R
        R_q, p_q = self.cmd
        row = np.hstack([
            Rc.T @ (p - self.p_c), Rr[:, 0], Rr[:, 1],
            Rc.T @ self.v, Rc.T @ self.w,
            R.T @ self.lp[:3] / c["f_scale"], R.T @ self.lp[3:] / c["t_scale"],
            c["gap_scale"] * gap, c["gap_scale"] * gap_rate,
            Rc.T @ (p_q - p), Rc.T @ mat_to_rotvec(R_q @ R.T),
            np.array(MODES) == g.mode, Rc.T @ self.d,
            g.travel, self.angle if g.mode == "turn" else (p - self.p_c) @ self.d0,
            1.0 - self.k / self.n_max])
        self.hist = np.vstack([self.hist[1:], row])

    def _end(self, stop):
        self.stop = self.trace.stop = stop
        self.trace.steps = len(self.trace.rows)
        return self.hist.ravel().copy()
