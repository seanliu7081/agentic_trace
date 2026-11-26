"""lrl contract: the one record that hands a contact to a skill (LRL_PLAN.md 5.1).

Data only. The mapping of an Entry to TRACE's EndCandidate is M1 and is not here; nothing in this module touches
the simulator or the planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KINDS = ("move_part", "put", "grasp")
TOUCHERS = ("tips", "pads", "held")                 # closed fingertips, open pads, the held object
MODES = ("push", "pull", "turn", "put", "grasp")

# Default limits of a contract. They are the envelope the policies are trained inside, so a change here is a
# change of training setting, not a run-time tuning knob.
LIMITS = {
    "force": 40.0,        # N, wrist force change at which the wrapper stops (as articulate.FORCE_LIMIT).   T
    "speed": 0.10,        # m/s, hand speed above which the lead is scaled down.                           T
    "tube": 0.06,         # m, lateral distance of a command from the believed line (push, pull).          T
    "tube_angle": 40.0,   # deg, angle of a commanded orientation from the believed path.                  T
    "time": 12.0,         # s, longest contact.                                                            T
}


@dataclass(eq=False)
class Entry:
    """One entry pose of the hand: where the transit ends and the skill starts."""
    R: np.ndarray                 # (3, 3) hand orientation, world
    p: np.ndarray                 # (3,) hand position, world
    approach: np.ndarray          # (3,) unit, world: the hand slides back along -approach to find a certified end
    standoff: tuple               # (s_min, s_max) m back along the approach; s_max is the skill's capture range
    twin: bool = False            # the 180-degree twin about the approach is an equal end
    cand: int = 0                 # index of the candidate this entry came from

    def __post_init__(self):
        self.R, self.p, self.approach = (np.asarray(x, float) for x in (self.R, self.p, self.approach))


@dataclass(eq=False)
class Goal:
    """What the contact is for. `direction` is a unit world vector: the believed motion of the hand for push and
    pull, the rotation axis for turn (right-hand rule gives the sense). `travel` is in m, or rad for turn."""
    mode: str
    direction: np.ndarray
    axis_point: np.ndarray | None = None      # (3,) a point on the axis, turn only
    travel: float = 0.0

    def __post_init__(self):
        self.direction = np.asarray(self.direction, float)
        if self.axis_point is not None:
            self.axis_point = np.asarray(self.axis_point, float)


@dataclass(eq=False)
class Contract:
    kind: str                     # one of KINDS
    toucher: str                  # one of TOUCHERS
    allowed: tuple                # ids of what may be touched
    entries: list                 # ordered Entry list
    goal: Goal
    limits: dict = field(default_factory=LIMITS.copy)
    done: tuple = ()              # detector class names that end the stage
