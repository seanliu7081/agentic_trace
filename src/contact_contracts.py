"""Wire labels and default goal semantics shared by contact planners and actors."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ContactGoal:
    held_end: int = -1
    carried_at_entry: bool = False
    region_required: bool = False
    minimum_travel: float = 0.


CONTACT_GOALS = {
    "pick": ContactGoal(held_end=1, minimum_travel=.06),
    "place": ContactGoal(carried_at_entry=True, region_required=True),
    "push_pull": ContactGoal(),
    "turn": ContactGoal(),
    "planar_push": ContactGoal(region_required=True),
}
