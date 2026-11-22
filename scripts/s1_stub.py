"""DEV / TEST FIXTURE ONLY: a stand-in for the S1 PROGRAM call, built from simulator truth.

Not part of the policy and never a run-time input of the system. For the current state of a non-test
LIBERO scene (after the settle steps) it writes the program an ideal S1 would write, so S2 changes can be
developed without model calls and without S1 variance:
  * stages follow the BDDL goal (or the dev --goal override), in its order:
      in / on        -> transfer with relation in / on;
      open / close   -> actuate pull / push on the region's moving part (its handle when the asset has one);
      turnon / off   -> actuate rotate on the fixture's moving part (e.g. a knob), with no visible effect,
                        so S2 judges the turn by the measured motion;
  * every pixel is the most interior point of the object's visible part in the head-camera segmentation
    render, the same image S1 gets; names and segmenter prompts mimic the ones S1 writes;
  * a sub-region of a fixture (a burner, the top of a cabinet) becomes place_x / place_y; a container's
    contain region does not; a region with its own joint (a drawer) becomes a part of its fixture;
  * every transfer stage carries the term {tilt_limit of the object, upright, require, relax_to tilted}, so
    offline runs exercise the TRACE term path (level table, conflicts, allowed relaxation); other stages have none.
Pixels therefore have no pointing error: dev results are somewhat better than with the real S1.

StubAstra hands this program to the harness in place of the PROGRAM call. It forwards every other call
(REPAIR, VERIFY, effect checks) to a real AstraClient, or refuses them when offline (real=None).
"""
from __future__ import annotations

import concurrent.futures as cf
import copy
import json
import re
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy import ndimage

from atrace.llm import AstraError
from atrace.sim_iface import HEAD_CAMERA

# S1-style names and segmenter prompts (most specific first, a plain generic shape noun last)
CATEGORY = {
    "alphabet_soup": ("alphabet soup can", ["alphabet soup can", "soup can", "can"]),
    "tomato_sauce": ("tomato sauce can", ["tomato sauce can", "can"]),
    "cherries": ("cherries can", ["cherries can", "can"]),
    "corn": ("corn can", ["corn can", "can"]),
    "cream_cheese": ("cream cheese box", ["cream cheese box", "small box", "box"]),
    "butter": ("butter box", ["butter box", "small box", "box"]),
    "chocolate_pudding": ("chocolate pudding box", ["chocolate pudding box", "small box", "box"]),
    "macaroni_and_cheese": ("macaroni and cheese box", ["macaroni and cheese box", "box"]),
    "cookies": ("cookies box", ["cookies box", "box"]),
    "popcorn": ("popcorn box", ["popcorn box", "box"]),
    "milk": ("milk carton", ["milk carton", "carton"]),
    "orange_juice": ("orange juice carton", ["orange juice carton", "carton"]),
    "ketchup": ("ketchup bottle", ["ketchup bottle", "bottle"]),
    "bbq_sauce": ("bbq sauce bottle", ["bbq sauce bottle", "bottle"]),
    "salad_dressing": ("salad dressing bottle", ["salad dressing bottle", "bottle"]),
    "new_salad_dressing": ("salad dressing bottle", ["salad dressing bottle", "bottle"]),
    "mayo": ("mayonnaise jar", ["mayonnaise jar", "jar"]),
    "basket": ("white-lined woven basket", ["white lined woven basket", "basket"]),
    "plate": ("plate", ["plate"]),
    "akita_black_bowl": ("black bowl", ["black bowl", "bowl"]),
    "white_bowl": ("white bowl", ["white bowl", "bowl"]),
    "glazed_rim_porcelain_ramekin": ("small ramekin", ["ramekin", "small bowl", "bowl"]),
    "chefmate_8_frypan": ("frying pan", ["frying pan", "pan"]),
    "moka_pot": ("silver moka pot", ["silver moka pot", "coffee pot", "kettle"]),
    "flat_stove": ("single-burner stove", ["single-burner stove", "hot plate", "stove"]),
    "wine_bottle": ("wine bottle", ["wine bottle", "bottle"]),
    "wine_rack": ("wine rack", ["wine rack", "rack"]),
    "wooden_cabinet": ("wooden cabinet", ["wooden cabinet", "cabinet"]),
    "white_cabinet": ("white cabinet", ["white cabinet", "cabinet"]),
    "microwave": ("microwave", ["microwave"]),
    "wooden_tray": ("wooden tray", ["wooden tray", "tray"]),
    "black_book": ("black book", ["black book", "book"]),
    "yellow_book": ("yellow book", ["yellow book", "book"]),
    "red_coffee_mug": ("red coffee mug", ["red coffee mug", "mug"]),
    "porcelain_mug": ("white porcelain mug", ["white porcelain mug", "mug"]),
    "white_yellow_mug": ("white and yellow mug", ["white and yellow mug", "mug"]),
    "desk_caddy": ("desk caddy", ["desk caddy", "organizer"]),
}
RANK = {"target": 3, "destination": 2, "part": 1, "context": 0}


def describe(instance):
    cat = re.sub(r"_\d+$", "", instance)
    if cat in CATEGORY:
        return CATEGORY[cat]
    words = cat.replace("_", " ")
    return words, list(dict.fromkeys([words, words.split()[-1]]))


class _Scene:
    """Head-camera segmentation render plus the geoms owned by each LIBERO object / moving part."""

    def __init__(self, env, size=512):
        self.inner = env.env
        self.sim = env.sim
        self.m, self.d = env.sim.model._model, env.sim.data._data
        self.size = size
        seg = env.sim.render(width=size, height=size, camera_name=HEAD_CAMERA, segmentation=True)[::-1]
        self.geom_px = np.where(seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 1], -1)
        self.cam = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA)

    def project(self, p):
        """World point -> (u, v) pixel of the head image (row 0 = top), same camera model as SimRobot."""
        f = 0.5 * self.size / np.tan(np.deg2rad(float(self.m.cam_fovy[self.cam])) / 2)
        R = self.d.cam_xmat[self.cam].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        pc = (np.asarray(p, float) - self.d.cam_xpos[self.cam]) @ R
        u, v = f * pc[0] / pc[2] + self.size / 2, f * pc[1] / pc[2] + self.size / 2
        return int(np.clip(u, 0, self.size - 1)), int(np.clip(v, 0, self.size - 1))

    def subtree_geoms(self, body):
        out = set()
        for g in range(self.m.ngeom):
            b = int(self.m.geom_bodyid[g])
            while b > 0 and b != body:
                b = int(self.m.body_parentid[b])
            if b == body:
                out.add(g)
        return out

    def body(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def object_body(self, instance):
        return self.body(self.inner.get_object(instance).root_body)

    def joint_bodies(self, name):
        """Bodies moved by the joints of a region (drawer, door) or an object (knob)."""
        joints = (self.inner.object_sites_dict[name].joints if name in self.inner.object_sites_dict
                  else self.inner.get_object(name).joints)
        return [int(self.m.jnt_bodyid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in joints]

    def handle_geoms(self, geoms):
        named = set()
        for g in geoms:
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                n += " " + (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_MESH, self.m.geom_dataid[g]) or "")
            if "handle" in n:
                named.add(g)
        return named or geoms

    def pixel(self, geoms, body):
        """Most interior point of the visible part; the projected body origin when nothing is visible."""
        mask = np.isin(self.geom_px, list(geoms))
        if mask.sum() < 4:
            return self.project(self.d.xpos[body])
        dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        v, u = np.unravel_index(np.argmax(dt), dt.shape)
        return int(u), int(v)

    def site_pixel(self, site):
        return self.project(self.d.site_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, site)])


def stub_program(env, goal=None, scene="dev stub program (simulator truth)"):
    sc = _Scene(env)
    inner = sc.inner
    goal = goal or inner.parsed_problem["goal_state"]
    objects, stages, said = {}, [], []

    def add(key, geoms, body, role, name, prompts, part_of=None):
        if key in objects:
            if RANK[role] > RANK[objects[key]["role"]]:
                objects[key]["role"] = role
            return objects[key]["id"]
        u, v = sc.pixel(geoms, body)
        objects[key] = {"id": f"o{len(objects) + 1}", "name": name, "prompts": prompts, "role": role,
                        "x": u, "y": v, "part_of": part_of}
        return objects[key]["id"]

    def add_object(instance, role):
        b = sc.object_body(instance)
        name, prompts = describe(instance)
        return add(instance, sc.subtree_geoms(b), b, role, name, prompts)

    def level(region):
        m = re.search(r"_(top|middle|bottom|upper|lower|left|right)_region$", region)
        return m.group(1) + " " if m else ""

    def moving(region):
        """The moving part of an articulated region: a drawer (slide joint) or a door (hinge)."""
        fixture = inner.object_sites_dict[region].parent_name if region in inner.object_sites_dict else region
        bodies = sc.joint_bodies(region)
        geoms = set().union(*(sc.subtree_geoms(b) for b in bodies))
        slide = any(sc.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE for j in range(sc.m.njnt)
                    if sc.m.jnt_bodyid[j] in bodies)
        kind = "drawer" if slide else "door"
        return fixture, geoms, bodies[0], kind, f"{level(region)}{kind} of the {describe(fixture)[0]}"

    for pred, *args in goal:
        pred = pred.lower()
        sid = f"s{len(stages) + 1}"
        stage = {"id": sid, "op": None, "object": None, "relation": None, "reference": None, "place_x": None,
                 "place_y": None, "motion": None, "goal": None, "turn_direction": None, "effect": None,
                 "effect_x": None, "effect_y": None, "tolerance": "normal", "note": "dev stub stage", "terms": []}
        if pred in ("in", "on"):
            obj, ref = args
            stage.update(op="transfer", object=add_object(obj, "target"), relation=pred)
            stage["terms"] = [{"prim": "tilt_limit", "of": stage["object"], "level": "upright", "role": "require",
                               "relax_to": "tilted"}]
            site = inner.object_sites_dict.get(ref)
            if site is None:
                stage["reference"] = add_object(ref, "destination")
            elif site.joints:
                fixture, geoms, body, kind, name = moving(ref)
                stage["reference"] = add(ref, geoms, body, "destination", name, [f"{level(ref)}{kind}".strip(), kind],
                                         part_of=add_object(fixture, "context"))
            else:
                stage["reference"] = add_object(site.parent_name, "destination")
                if not ref.endswith("contain_region"):
                    stage["place_x"], stage["place_y"] = sc.site_pixel(ref)
                    stage["tolerance"] = "tight"
            ref_obj = next(o for o in objects.values() if o["id"] == stage["reference"])
            said.append(f"the {describe(obj)[0]} is {pred} the {ref_obj['name']}")
        elif pred in ("open", "close"):
            region = args[0]
            fixture, geoms, body, kind, name = moving(region)
            fid = add_object(fixture, "context")
            parent = objects[region]["id"] if region in objects else fid      # the drawer, when it is listed
            hid = add(region + "#handle", sc.handle_geoms(geoms), body, "part", f"handle of the {name}",
                      [f"{level(region)}{kind} handle".strip(), "handle"], part_of=parent)
            goal_word = "open" if pred == "open" else "closed"
            stage.update(op="actuate", object=hid, motion="pull" if pred == "open" else "push", goal=goal_word)
            said.append(f"the {name} is {goal_word}")
        elif pred in ("turnon", "turnoff"):
            fixture = args[0]
            fid = add_object(fixture, "context")
            bodies = sc.joint_bodies(fixture)
            fname, fprompts = describe(fixture)
            kid = add(fixture + "#knob", set().union(*(sc.subtree_geoms(b) for b in bodies)), bodies[0], "part",
                      f"{fname} control knob", [f"{fprompts[-1]} control knob", "knob"], part_of=fid)
            goal_word = "on" if pred == "turnon" else "off"
            stage.update(op="actuate", object=kid, motion="rotate", goal=goal_word)
            said.append(f"the {fname} is turned {goal_word}")
        else:
            continue
        stages.append(stage)
    # same-category objects: tell them apart by image position, as S1 would
    names = [o["name"] for o in objects.values()]
    for o in objects.values():
        if names.count(o["name"]) > 1:
            twins = sorted(p["x"] for p in objects.values() if p["name"] == o["name"])
            o["name"] = ("left " if o["x"] == twins[0] else "right " if o["x"] == twins[-1] else "middle ") + o["name"]
    return {"scene": scene, "objects": list(objects.values()), "stages": stages,
            "finish_condition": "; ".join(said) + "."}


def annotate(env, prog, size=512):
    """Head RGB with the program's pixels drawn on it (for checking a stub program by eye)."""
    img = np.ascontiguousarray(env.sim.render(width=size, height=size, camera_name=HEAD_CAMERA)[::-1])
    for o in prog["objects"]:
        cv2.circle(img, (o["x"], o["y"]), 5, (255, 0, 0) if o["role"] == "target" else (0, 160, 255), -1)
        cv2.putText(img, f"{o['id']} {o['prompts'][0]}", (o["x"] + 7, o["y"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1, cv2.LINE_AA)
    for s in prog["stages"]:
        if s["place_x"] is not None:
            cv2.drawMarker(img, (s["place_x"], s["place_y"]), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
    return img


class StubAstra:
    """AstraClient stand-in: PROGRAM comes from the stub; every other call goes to `real` (None = offline)."""

    def __init__(self, program, log_dir, real=None):
        self.program = program
        self.real = real
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "00_program.stub.json").write_text(json.dumps(program, indent=1))

    @property
    def calls(self):
        return self.real.calls if self.real is not None else []

    @property
    def max_calls(self):
        return self.real.max_calls if self.real is not None else 0

    def submit(self, kind, instructions, content, schema, effort=None):
        fut = cf.Future()
        fut.set_result((copy.deepcopy(self.program), {"kind": kind, "stub": True}))
        return fut

    def call(self, kind, instructions, content, schema, effort=None):
        if self.real is None:
            raise AstraError(f"offline dev run: no {kind} call", code="offline")
        return self.real.call(kind, instructions, content, schema, effort)

    def stats(self):
        s = self.real.stats() if self.real is not None else {
            "calls": 0, "ok": 0, "latency_s": [], "kinds": [], "output_tokens": 0, "reasoning_tokens": 0}
        return {**s, "program": "stub"}
