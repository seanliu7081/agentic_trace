"""Offline contact labels from time-ordered, evaluator-only contact evidence."""
from __future__ import annotations

import numpy as np


LABELS = ("firm_handle_grasp", "single_finger_graze", "cabinet_frame_pinch",
          "push_face_contact", "mid_pull_slip", "air_close")


def contact_patch_evidence(contacts, left_fingers, right_fingers, robot_geoms, grasp_geoms, push_dir):
    """Convert MuJoCo contact pairs/normals to evaluator evidence. Distances are signed metres."""
    normals = ([], [])
    frame = [False, False]
    pen, push = 0., False
    for con in contacts:
        if con.dist > 0:
            continue
        for g, other, sign in ((con.geom1, con.geom2, 1.), (con.geom2, con.geom1, -1.)):
            if robot_geoms[other] or not (left_fingers[g] or right_fingers[g]):
                continue
            side = int(right_fingers[g])
            if not grasp_geoms[other]:
                frame[side] = True
                continue
            normal = sign * con.frame[:3]
            normals[side].append(normal)
            push |= float(normal @ push_dir) > .6
            pen = max(pen, -float(con.dist))
    opposed = any(float(a @ b) < -.4 for a in normals[0] for b in normals[1])
    return dict(left_target=bool(normals[0]), right_target=bool(normals[1]),
                left_frame=frame[0], right_frame=frame[1], target_opposition=opposed,
                firm=opposed, push_face_loaded=push, target_contact_pen_m=pen)


def _sustained(mask, steps=3):
    if len(mask) < steps:
        return np.empty(0, dtype=int)
    return np.flatnonzero(np.convolve(mask.astype(int), np.ones(steps, int), "valid") == steps)


def classify_contacts(trace):
    """Classify a 20 Hz contact trace. Target normals point from pads into target.

    Opposing pad loads distinguish a pinch from a two-pad face push. A slip
    requires witnessed transport and subsequent loss under a continuing grasp
    command. A deliberate release should stop the manipulation trace.
    """
    if not trace:
        return "uncertain"
    def field(key):
        return np.asarray([row[key] for row in trace])

    left, right = field("left_target"), field("right_target")
    opposed = left & right & field("target_opposition")
    grip_starts = _sustained(opposed)
    if grip_starts.size:
        start = int(grip_starts[0])
        travel = field("target_delta")
        carried = opposed & (np.abs(travel - travel[start]) >= 0.005)
        lost = _sustained(~left & ~right & (field("command_width_m") < 0.025))
        if carried.any() and np.any(lost > np.flatnonzero(carried)[0]):
            return "mid_pull_slip"
        if _sustained(opposed[-5:]).size:
            return "firm_handle_grasp"
        return "uncertain"
    if _sustained(field("left_frame") & field("right_frame")).size:
        return "cabinet_frame_pinch"
    if _sustained((left | right) & field("push_face_loaded")).size:
        return "push_face_contact"
    if (left ^ right).any():
        return "single_finger_graze"
    if not (left | right | field("left_frame") | field("right_frame")).any():
        if np.any((field("width_m") < 0.012) & (field("command_width_m") < 0.012)):
            return "air_close"
    return "uncertain"
