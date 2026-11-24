"""S1 contract: metric-free constraint program (CPL) + VERIFY / REPAIR exchanges."""
from __future__ import annotations

import copy
from collections import Counter

from .levels import LEVELS, MODEL_LEVELS

IMG = 512

PROGRAM_INSTRUCTIONS = f"""You are System 1 (the planner) of a robot manipulation system: one Franka Panda arm with a
parallel gripper at a table, a fixed HEAD camera (image 1) and a WRIST camera on the gripper (image 2).
You decide WHAT must happen. A local System 2 decides HOW: it measures all geometry from depth images,
segments the objects you name, chooses grasps and placements, plans collision-free whole-body motion and
executes it with force and finger feedback. Never output metric numbers, coordinates in metres, heights or sizes.

Write a constraint program:

objects: every object the task involves.
  - role "target": an object that must be moved.
  - role "destination": a container, support, or reference object for a relation.
  - role "part": a movable part that must be operated (knob, dial, handle, button, drawer, door). Give
    part_of = the id of the fixture it belongs to, and also list that fixture (role "destination" or "context").
  - role "context": any other object you want System 2 to know by name (optional).
  For each object:
  - id: short id like "o1".
  - name: a description that distinguishes it in this scene (colour, label, position).
  - prompts: 1-3 short noun phrases for an open-vocabulary segmenter, most specific first, and always include
    one plain generic shape noun (e.g. ["tomato sauce can", "can"], ["moka pot", "coffee pot", "kettle"],
    ["stove knob", "knob"]). Brand names alone are not recognised.
  - x, y: integer pixel in IMAGE 1 ({IMG}x{IMG}, x to the right, y downward, origin top-left) on the object's
    visible surface, near the centre of its visible part. For a part, point at the part itself.
  - image: 1 (x, y are in IMAGE 1). Only when the object cannot be seen in IMAGE 1 (hidden behind something) but can
    be seen in IMAGE 2 (the wrist camera): 2, and x, y are a pixel of IMAGE 2 ({IMG}x{IMG}, same convention).

stages: steps in execution order.
  - op "transfer": pick up `object` (a target) and place it so that relation(object, reference) holds.
    relation "in" = inside a container; "on" = resting on top of a support; "left_of" / "right_of" /
    "front_of" / "behind" = next to the reference as seen in image 1 (left/right = image left/right,
    front = toward the head camera = lower in the image).
    place_x, place_y: optional pixel in image 1 where the object must end up on/in the reference when only
    part of the reference is valid (e.g. the burner / cooking area of a stove, one compartment, one plate);
    otherwise null.
  - op "actuate": operate a part. motion "rotate" (knobs, dials), "push" (close a drawer or door, press a
    button), "pull" (open a drawer or door); goal "on" / "off" / "open" / "closed".
    turn_direction (rotate only): "clockwise" or "counterclockwise" as seen looking straight at the part's
    face, if you know which way achieves the goal; otherwise null.
    effect (actuate only): what a camera will see change in the scene when the goal is reached, in a few
    words (e.g. "the burner glows red", "the lamp lights up", "the drawer stands open"). For rotate, System 2
    turns the part in small steps and stops as soon as this change appears (it reverses if it never appears);
    for push / pull it is the check System 2 falls back on when depth does not show the part's state. Null if
    reaching the goal leaves nothing visible.
    effect_x, effect_y (actuate only): integer pixel in IMAGE 1 at the centre of where that change will appear
    (e.g. the middle of the burner, the lamp shade); System 2 watches a small area around it. Null when effect
    is null.
  - tolerance: "tight" when precise placement matters (narrow slot, compartment), "normal" otherwise,
    "loose" when anywhere in/on the reference is fine.
  - release (transfer only): copy the verb the task instruction uses for this object. "drop" when it says drop:
    System 2 lets go over the reference and any resulting pose counts. "place" when it says place: System 2 sets
    the object down upright and centred, never dropping it. Null when the instruction uses neither verb (put,
    move, ...): System 2 then drops into containers ("in") and places on supports. Null for actuate stages.
  - note: one short sentence of intent or caution for this stage.
  - terms: motion requirements for this stage as named levels (System 2 turns levels into numbers). Use terms
    only when the task or the objects call for them; usually terms is []. Collision avoidance is automatic:
    never add a term for it.
    prim "tilt_limit", of = the object this stage carries: keep it "upright" (e.g. an open container with
    contents), "tilted" or "loose".
    prim "clearance", of = an object to stay away from while moving (fragile, precarious): "close", "normal",
    "wide".
    prim "approach", of = the object grasped or the destination: arrive "vertical" or "steep" (e.g. over a
    narrow opening).
    role "require" = must hold, "prefer" = try to. relax_to: a looser level of the same prim that System 2 may
    switch to by itself when the stricter one makes the motion impossible, or null.
  Order stages so that they do not interfere (e.g. operate a fixture before putting an object on it if the
  object could block the part; place objects into a container one at a time).
  Set every field that does not apply to null.

System 2 avoids collisions by itself: every object not being manipulated is an obstacle, and objects may not
touch each other or anything other than their own destination support. You do not need to list obstacles.
finish_condition: one sentence describing what the scene must look like when the task is complete.
"""

_STR = {"type": "string"}
_NSTR = {"type": ["string", "null"]}
_NINT = {"type": ["integer", "null"]}

TERM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["prim", "of", "level", "role", "relax_to"],
    "properties": {
        "prim": {"type": "string", "enum": list(LEVELS)},
        "of": _STR,
        "level": {"type": "string", "enum": MODEL_LEVELS},
        "role": {"type": "string", "enum": ["require", "prefer"]},
        "relax_to": {"type": ["string", "null"], "enum": MODEL_LEVELS + [None]},
    },
}

OBJECT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "name", "prompts", "role", "x", "y", "image", "part_of"],
    "properties": {
        "id": _STR, "name": _STR,
        "prompts": {"type": "array", "items": _STR},
        "role": {"type": "string", "enum": ["target", "destination", "part", "context"]},
        "x": {"type": "integer"}, "y": {"type": "integer"},
        "image": {"type": "integer", "enum": [1, 2]},
        "part_of": _NSTR,
    },
}

STAGE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "op", "object", "relation", "reference", "place_x", "place_y", "motion", "goal",
                 "turn_direction", "effect", "effect_x", "effect_y", "tolerance", "release", "note", "terms"],
    "properties": {
        "id": _STR,
        "op": {"type": "string", "enum": ["transfer", "actuate"]},
        "object": _STR,
        "relation": {"type": ["string", "null"],
                     "enum": ["in", "on", "left_of", "right_of", "front_of", "behind", None]},
        "reference": _NSTR,
        "place_x": _NINT, "place_y": _NINT,
        "motion": {"type": ["string", "null"], "enum": ["rotate", "push", "pull", None]},
        "goal": {"type": ["string", "null"], "enum": ["on", "off", "open", "closed", None]},
        "turn_direction": {"type": ["string", "null"], "enum": ["clockwise", "counterclockwise", None]},
        "effect": _NSTR,
        "effect_x": _NINT, "effect_y": _NINT,
        "tolerance": {"type": "string", "enum": ["tight", "normal", "loose"]},
        "release": {"type": ["string", "null"], "enum": ["drop", "place", None]},
        "note": _STR,
        "terms": {"type": "array", "items": TERM_SCHEMA},
    },
}

PROGRAM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["scene", "objects", "stages", "finish_condition"],
    "properties": {
        "scene": _STR,
        "objects": {"type": "array", "items": OBJECT_SCHEMA},
        "stages": {"type": "array", "items": STAGE_SCHEMA},
        "finish_condition": _STR,
    },
}

REPAIR_SCHEMA = copy.deepcopy(PROGRAM_SCHEMA)
REPAIR_SCHEMA["required"] = ["diagnosis"] + REPAIR_SCHEMA["required"]
REPAIR_SCHEMA["properties"]["diagnosis"] = _STR

REPAIR_INSTRUCTIONS = PROGRAM_INSTRUCTIONS + """
REPAIR MODE. System 2 could not complete the program it was given. You receive the task, the previous
program, the ledger of stages already completed, the failure report (what failed and why, as measured
locally), and current images. Write `diagnosis` (one or two sentences), then a NEW program covering ONLY the
remaining work from the current scene state. You may re-point objects (new pixels in the CURRENT image 1, or with
image 2 in the current image 2 when the object cannot be seen in image 1), change segmenter prompts, change the
reference location (place_x/place_y), the relation, or the order.
Do not repeat stages the ledger reports as done unless the images show they are undone.
Rules for the new program:
- Only the objects the task instruction asks to move may be moved. Never add a stage that moves, pushes, parks
  or re-orients any other object, even one that is in the way or has fallen over: touching it is a failure.
  Work around it (another place_x/place_y on the reference, another order) or keep the stage as it was.
- Keep the ids of the previous program for the same physical objects; the object in the gripper keeps its id.
- The failure report holds System 2's measurements (what a move touched, why placements or grasps were
  rejected, the nearest obstacle). Base the diagnosis on them and on the images.
- To set the object in the gripper down somewhere neutral, use relation "on" with an object named "table"
  whose pixel is on a free part of the table top.
- After a no_visible_effect failure keep `effect` set; if the report shows the watched area was mostly hidden
  or in the wrong place, re-point effect_x / effect_y (or the part).
- After a push / pull failure the report lists `faces`: the faces System 2 measured around the part as the face it
  is mounted on (which way each looks in image 1, which way the motion went, and whether it was rejected, found no
  grip, or was used and with what result), and `part_state`: the part's state measured from depth (closed = its
  front flush with the surrounding face; otherwise how far it stands out, or how far it is turned). This measurement
  is finer than the images (a drawer 2 cm out can look closed): when it says the goal state is not reached, keep a
  stage for it. When no face worked, the segmented part may not be the piece that moves: re-point it.
- Grasp and placement failure reports carry `cause` (what System 2 measured to fail: e.g. no grasp on the model,
  no room for the open hand, no arm posture, the pads missed the object, the way there is blocked by a named thing)
  and `levers` (what would change it). Base the fix on them, not on the object's size: a wrong or partial model is
  fixed by re-pointing the object (in image 2 if the head camera cannot see it), not by pointing at a smaller part.
- After a trace_conflict failure the report's `quote` lists the stage terms that cannot all hold for that motion,
  each with its level and the smallest relaxation, in physical units, that would make the motion feasible
  (`value` now, `needed` for feasibility). Relax them (a looser level), rescope or drop them, or change the plan.
- `dropped_terms` lists terms System 2 had to leave out because they could not be applied (e.g. an unknown object
  id or a term the motion cannot affect); fix or drop them.
"""

VERIFY_INSTRUCTIONS = """You verify task completion for a robot. You get the task instruction, the program
that was executed, the ledger of stage outcomes measured locally, and current images (image 1: head camera;
image 2: wrist camera looking at the work area). Judge only from what is visible plus the ledger.
Be strict about where things are: a moved object must be on its destination (upright when that matters) or
inside its destination container. For an object that must be "in" a container its pose does not matter:
standing, lying on its side, tilted, leaning on the container's wall, or resting on or against another object
already inside all count as in. But it must have gone down into the container (into the named compartment, if
one is named): an object held up on top of the rim or of an internal divider, lying across the opening instead of
inside it, is not in; nor is one that rests outside the container, sits on the rim with most of it outside, or is
in a different compartment than the named one. A device
that had to be operated must be in its goal state. If its state is visible in the current images (a light or
burner glowing or not, a door or drawer open or closed), judge from the images: not in the goal state means
not done, whatever the ledger says. Only when its state is hidden now (e.g. covered by an object placed
later) may the ledger stand in: effect_observed=true with confirmed_by_s1=true (the change was seen in the
camera images while operating it and confirmed then). A rotation or travel in the ledger is only the hand's
motion and never evidence by itself. For a drawer or door the ledger may give System 2's measured state (`state`,
from depth: closed = its front flush with the surrounding face, else how far it stands out or is turned); it is
finer than the images, so a measured "not closed" means not done even if the images look closed. Answer
complete=true only if the instruction is satisfied now. If not,
say what is wrong in `problem` and give a one-sentence hint for System 2 in `hint`."""

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["complete", "stages", "problem", "hint"],
    "properties": {
        "complete": {"type": "boolean"},
        "stages": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "status", "evidence"],
            "properties": {"id": _STR, "status": {"type": "string", "enum": ["done", "not_done", "unclear"]},
                           "evidence": _STR}}},
        "problem": _STR,
        "hint": _STR,
    },
}


EFFECT_INSTRUCTIONS = """You check one thing for a robot that is operating a device (turning a knob, pressing a switch,
moving a handle). You get the task, the stage being executed with the visible effect expected when its goal is
reached, and two head-camera images: IMAGE 1 before the robot started operating the part, IMAGE 2 now.
Decide whether the expected effect is visible in IMAGE 2 and was not there in IMAGE 1 (a real change of the
device's state, not a shadow, a reflection or the robot arm covering something). Answer effect_visible and one
sentence of evidence."""

EFFECT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["effect_visible", "evidence"],
    "properties": {"effect_visible": {"type": "boolean"}, "evidence": _STR},
}


# ---------------------------------------------------------------- placement "v2" (r1's mechanism): System 1 owns the
# placement. It points at where the held object goes; System 2 carries it there, lowers it until contact, lets go, and
# System 1 judges the result from fresh images and points again when it is not right.
_LONG_SIDE = ["any", "left_right", "near_far", "diagonal_rising", "diagonal_falling"]

_POINT_RULES = f"""x, y: integer pixel in IMAGE 1 ({IMG}x{IMG}, x to the right, y downward, origin top-left) where the CENTRE of the
held object must come down. System 2 lowers the object straight down from above, and it reads your pixel at the
height of the destination's top, so point at where the place appears in the image at that height:
  - relation "in": the middle of the container's opening (of the named compartment) at the level of its rim, not the
    floor seen through the opening.
  - relation "on": the middle of the free part of the support's top surface (of the area the task names).
  - "left_of" / "right_of" / "front_of" / "behind": a free spot on the table next to the reference on that side
    (left/right = image left/right, front = lower in the image), about one object width away from it.
  Point at the destination, never at the robot or at the object in the gripper, and keep the object clear of the
  other objects in the scene.
long_side: how the held object's longest horizontal side must lie when it comes down, as a direction in IMAGE 1:
  "left_right", "near_far" (from the camera away into the scene), "diagonal_rising" (lower left to upper right),
  "diagonal_falling" (upper left to lower right), or "any" when its turn about the vertical does not matter. Set it
  when the object's shape has to line up with the destination (a long object into a slot, a box onto a narrow
  shelf); otherwise "any"."""

PLACE_POSE_INSTRUCTIONS = f"""You are System 1 (the planner) of a robot manipulation system: one Franka Panda arm with a parallel
gripper at a table and a fixed HEAD camera. The gripper is holding an object, and you decide where it is put down.
You get the task instruction, the stage being executed (the held object, the reference, the relation to reach), what
happened in earlier attempts of this stage (if any), and IMAGE 1: the head camera now.
System 2 carries the object above the point you give and lowers it straight down until it touches something, then
opens the gripper (when the stage's release is "drop" it lets go just above the destination instead). System 2 does
not check whether the object fits there and it never refuses: the place is yours to choose. After the release you
are shown the result and can have it picked up and put again.
{_POINT_RULES}
why: one short sentence.
Never output metric numbers."""

_POINT_RULES_UNIFIED = f"""x, y: integer pixel in IMAGE 1 ({IMG}x{IMG}, x to the right, y downward, origin top-left) on the surface
where the CENTRE of the held object should come to rest, as seen in the image:
  - relation "in": the floor of the container's opening (of the named compartment) where the object should land.
    For a container that opens at the side (a cavity behind an opening or a door, such as a microwave or an oven),
    the floor inside it as seen through the opening, never its front panel, its top or its door.
  - relation "on": the middle of the free part of the support's top surface (of the area the task names).
  - "left_of" / "right_of" / "front_of" / "behind": a free spot on the table next to the reference on that side
    (left/right = image left/right, front = lower in the image), about one object width away from it.
  Point at the destination, never at the robot or at the object in the gripper. System 2 searches for the exact rest
  pose around your point (clear of other things, room for the hand).
long_side: how the held object's longest horizontal side must lie when it comes down, as a direction in IMAGE 1:
  "left_right", "near_far" (from the camera away into the scene), "diagonal_rising" (lower left to upper right),
  "diagonal_falling" (upper left to lower right), or "any" when its turn about the vertical does not matter. Set it
  when the object's shape has to line up with the destination (a long object into a slot, a box onto a narrow
  shelf); otherwise "any".
entry: the side from which the held object can come in, when the destination is open on one side only (a cavity with
  a roof and a front opening, a drawer under a shelf or a handle): "from_image_left", "from_image_right",
  "from_camera_side" (from the camera toward the destination), "from_far_side"; "from_above" when it is open from
  above, "any" when you cannot tell. System 2 checks every way in (straight down, slanted, sliding in from a side) and
  takes the first one that is free for the hand and the object; yours is tried first."""

PLACE_POSE_INSTRUCTIONS_UNIFIED = f"""You are System 1 (the planner) of a robot manipulation system: one Franka Panda arm with a
parallel gripper at a table and a fixed HEAD camera. The gripper is holding an object, and you decide where it is put
down. You get the task instruction, the stage being executed (the held object, the reference, the relation to reach),
what happened in earlier attempts of this stage (if any), and IMAGE 1: the head camera now.
System 2 looks near the point you give for a place where the object can rest on the destination (nothing else standing
there, a free way in for the hand and the object, room for the open hand), brings it in (straight down, slanted, or
sliding in under what is above) and opens the gripper only when the object meets its support where expected (when the stage's release is "drop" it may let go a little above the floor of the
opening). When no place near your point works, or the object touched something else first, the earlier attempts say
why in words: point again where that reason does not hold. After the release you are shown the result and can have
it picked up and put again.
{_POINT_RULES_UNIFIED}
why: one short sentence.
Never output metric numbers."""

PLACE_POSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["x", "y", "long_side", "why"],
    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                   "long_side": {"type": "string", "enum": _LONG_SIDE}, "why": _STR},
}

ENTRY_WORDS = ["from_above", "from_image_left", "from_image_right", "from_camera_side", "from_far_side", "any"]
PLACE_POSE_SCHEMA_UNIFIED = {
    "type": "object", "additionalProperties": False,
    "required": ["x", "y", "long_side", "entry", "why"],
    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"},
                   "long_side": {"type": "string", "enum": _LONG_SIDE},
                   "entry": {"type": "string", "enum": ENTRY_WORDS}, "why": _STR},
}

PLACE_LOOK_INSTRUCTIONS = f"""You are System 1 (the planner) of a robot manipulation system. The robot has just put an object down
and moved its hand away. You get the task instruction, the stage (the object, the reference, the relation to reach),
what System 2 measured while putting it down (where it aimed, whether the descent ended on contact), the earlier
attempts of this stage, and current images (IMAGE 1: head camera; IMAGE 2: wrist camera looking down at the work
area).
done: true only if the stage's relation holds now for this object, judged from the images.
  - "in": the object went down into the container (into the named compartment, if one is named). Any pose counts
    (standing, lying, leaning on a wall, resting on another object inside); resting on top of the rim or of a
    divider, lying across the opening, or outside the container does not.
  - "on": the object rests on the support. When the stage's release is "place" it must also be upright.
  - "left_of" / "right_of" / "front_of" / "behind": the object rests on the table on that side of the reference,
    near it.
If it is not done, System 2 picks the object up again and puts it where you point now. Say in `evidence` what went
wrong and correct for it (for example "it landed on the rim toward the left: aim further right").
{_POINT_RULES}
When done is true, x, y and long_side are null.
evidence: one or two short sentences.
Never output metric numbers."""

PLACE_LOOK_INSTRUCTIONS_UNIFIED = PLACE_LOOK_INSTRUCTIONS.replace(_POINT_RULES, _POINT_RULES_UNIFIED)

PLACE_LOOK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["done", "evidence", "x", "y", "long_side"],
    "properties": {"done": {"type": "boolean"}, "evidence": _STR, "x": _NINT, "y": _NINT,
                   "long_side": {"type": ["string", "null"], "enum": _LONG_SIDE + [None]}},
}


class ProgramError(ValueError):
    pass


def validate_program(prog, img_size=IMG):
    """Check references / required fields; fix small issues in place. Returns list of warnings."""
    warn = []
    objs = prog.get("objects") or []
    ids = [o["id"] for o in objs]
    if len(set(ids)) != len(ids):
        raise ProgramError("duplicate object ids")
    by_id = {o["id"]: o for o in objs}
    for o in objs:
        o["x"] = int(min(max(o["x"], 0), img_size - 1))
        o["y"] = int(min(max(o["y"], 0), img_size - 1))
        o["image"] = 2 if o.get("image") == 2 else 1
        prompts = [p.strip() for p in o.get("prompts") or [] if p and p.strip()]
        if not prompts:
            prompts = [o["name"]]
        o["prompts"] = prompts[:4]
        if o.get("part_of") and o["part_of"] not in by_id:
            warn.append(f"{o['id']}: part_of {o['part_of']} unknown; ignored")
            o["part_of"] = None
    stages = prog.get("stages") or []
    if not stages:
        raise ProgramError("program has no stages")
    sids = set()
    for s in stages:
        if s["id"] in sids:
            s["id"] = s["id"] + "_" + str(len(sids))
        sids.add(s["id"])
        if s["object"] not in by_id:
            raise ProgramError(f"stage {s['id']}: unknown object {s['object']}")
        if s["op"] == "transfer":
            if not s.get("relation"):
                raise ProgramError(f"stage {s['id']}: transfer needs a relation")
            if s.get("reference") not in by_id:
                raise ProgramError(f"stage {s['id']}: unknown reference {s.get('reference')}")
        else:
            if not s.get("motion"):
                s["motion"] = "rotate" if s.get("goal") in ("on", "off") else "push"
                warn.append(f"{s['id']}: motion defaulted to {s['motion']}")
        for k in ("place_x", "place_y", "effect_x", "effect_y"):
            if s.get(k) is not None:
                s[k] = int(min(max(s[k], 0), img_size - 1))
        for kx, ky in (("place_x", "place_y"), ("effect_x", "effect_y")):
            if (s.get(kx) is None) != (s.get(ky) is None):
                s[kx] = s[ky] = None
        if s.get("effect") and s.get("effect_x") is None:
            warn.append(f"{s['id']}: effect without effect_x/effect_y; System 2 watches the whole device")
        terms = []
        for t in s.get("terms") or []:
            table = LEVELS[t["prim"]]
            if t["level"] not in table:
                warn.append(f"{s['id']}: {t['prim']} has no level {t['level']}; term dropped")
                continue
            if t.get("relax_to") is not None and t["relax_to"] not in table:
                warn.append(f"{s['id']}: {t['prim']} has no level {t['relax_to']}; relax_to ignored")
                t["relax_to"] = None
            terms.append(t)
        s["terms"] = terms
    return warn


def merge_compiles(programs, demote_below=0.4):
    """Self-consistency over N compiles of one instruction (TRACE_method §5.5, DESIGN §3.8). Stages and everything
    else come from the first program. Per stage index a term (prim, of) gets kappa = the fraction of compiles that
    contain it and the most common level among them; a require term with kappa < demote_below becomes prefer."""
    main, n = programs[0], len(programs)
    for i, st in enumerate(main["stages"]):
        found = {}
        for prog in programs:
            stages = prog.get("stages") or []
            if i < len(stages):
                for t in {(t["prim"], t["of"]): t for t in stages[i].get("terms") or []}.values():
                    found.setdefault((t["prim"], t["of"]), []).append(t)
        terms = []
        for ts in found.values():
            kappa = len(ts) / n
            level = Counter(t["level"] for t in ts).most_common(1)[0][0]
            t = dict(next(t for t in ts if t["level"] == level), kappa=round(kappa, 3))
            if t["role"] == "require" and kappa < demote_below:
                t["role"] = "prefer"
            terms.append(t)
        st["terms"] = terms
    return main
