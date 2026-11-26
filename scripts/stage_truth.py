"""Evaluator-side stage truth for development tasks (LRL_PLAN.md 7.4, run_episode.py --stage-truth).

attach() wraps the harness log: after each of the stage events below it appends one record to <run_dir>/truth.jsonl
with the simulator's state at that moment. The policy never reads it; the harness is only called through.

  header   once: every articulated joint (kind, body, range, the fixture's open / closed / on ranges), the goal
           predicates of the task and the joints they read
  record   event, step, t; joints {name: {q, axis, anchor}} (world frame); bodies {name: [x, y, z, qw, qx, qy, qz]}
           for every object and fixture body; goal [[predicate, argument..., holds]]; and, when the event carries a
           believed contact point (lrl_entry.point, press_plan.point, part_grasp.p), probe {joint: {near_m, front,
           back, lo, hi}}: where the moving part really is around that point (see _probe)

Development tasks only: simulator truth of a test or held-out scene is never read (LRL_PLAN.md section 3).
"""
import json

import mujoco
import numpy as np

EVENTS = ("stage_start", "handoff", "lrl_arrival", "part_state_before", "part_state", "press_plan", "part_grasp",
          "lrl_entry", "lrl_skill", "stage_done", "stage_failed")
POINT = {"lrl_entry": "point", "press_plan": "point", "part_grasp": "p"}
DEV_SCENES = ("KITCHEN_SCENE1_", "KITCHEN_SCENE10_", "LIVING_ROOM_SCENE3_", "STUDY_SCENE2_")
ROBOT = ("world", "table", "robot0_", "gripper0_", "mount0_")


def is_dev(suite, task):
    """The development tasks of LRL_PLAN.md 7.1 (libero_goal and libero_object are development suites as wholes)."""
    return suite in ("libero_goal", "libero_object") or (suite == "libero_90" and task.startswith(DEV_SCENES))


def _name(m, kind, i):
    return mujoco.mj_id2name(m, kind, i) or ""


def _subtree(m, body):
    """Geoms of a body and of the bodies attached below it (a drawer and its handle)."""
    under = [b for b in range(m.nbody) if b == body or _descends(m, b, body)]
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] in under]


def _descends(m, b, root):
    while b != 0:
        b = m.body_parentid[b]
        if b == root:
            return True
    return False


def _probe(m, d, geoms, axis, point):
    """Where a moving part is around a believed contact point, along its true joint axis. near_m: distance from
    the point to the nearest of the part's geoms (their bounding boxes); front / back: the part's first surface
    on the line through the point along the axis, seen from the +axis and from the -axis side, as a signed
    coordinate along the axis relative to the point (None: the line misses the part); lo / hi: the part's extent
    along the axis relative to the point (bounding-box corners)."""
    near, lo, hi, front, back = np.inf, np.inf, -np.inf, None, None
    gid = np.zeros(1, np.int32)
    for g in geoms:
        R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        centre, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        local = (point - c) @ R - centre
        near = min(near, float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0))))
        corners = (centre + half * np.array(np.meshgrid(*[(-1, 1)] * 3)).reshape(3, -1).T) @ R.T + c
        s = (corners - point) @ axis
        lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
        for sign in (1.0, -1.0):
            start = point + sign * axis
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                dist = mujoco.mj_rayMesh(m, d, g, start, -sign * axis)
            else:
                dist = mujoco.mju_rayGeom(c, d.geom_xmat[g], m.geom_size[g], start, -sign * axis, m.geom_type[g])
            if dist >= 0:
                at = sign * (1.0 - dist)
                if sign > 0:
                    front = at if front is None else max(front, at)
                else:
                    back = at if back is None else min(back, at)
    r4 = lambda x: None if x is None else round(float(x), 4)
    return {"near_m": r4(near), "front": r4(front), "back": r4(back), "lo": r4(lo), "hi": r4(hi)}


def attach(h, env, path):
    m, d = env.sim.model._model, env.sim.data._data
    joints = [j for j in range(m.njnt) if m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
              and not _name(m, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(ROBOT)]
    jname = {j: _name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints}
    geoms = {j: _subtree(m, m.jnt_bodyid[j]) for j in joints}
    bodies = [b for b in range(m.nbody) if not _name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith(ROBOT)]
    problem = env.env.parsed_problem
    reads = {}                    # goal argument -> the joints its predicate reads
    for g in problem["goal_state"]:
        for arg in g[1:]:
            site = env.env.object_sites_dict.get(arg)
            fix = env.env.fixtures_dict.get(arg)
            reads[arg] = list(getattr(site, "joints", None) or getattr(fix, "joints", None) or [])
    out = open(path, "w")

    def write(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    write({"event": "header", "language": env.language_instruction, "goal": problem["goal_state"], "goal_joints": reads,
           "joints": {jname[j]: {"kind": "slide" if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else "hinge",
                                 "body": _name(m, mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]),
                                 "range": m.jnt_range[j].tolist()} for j in joints},
           "fixtures": {k: v.object_properties.get("articulation") for k, v in env.env.fixtures_dict.items()
                        if hasattr(v, "object_properties")}})
    log = h.log

    def logged(event, **kw):
        log(event, **kw)
        if event not in EVENTS:
            return
        rec = {"event": event, "step": int(h.rb.steps), "t": round(h.elapsed(), 3),
               "joints": {jname[j]: {"q": float(d.qpos[m.jnt_qposadr[j]]),
                                     "axis": (d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j]).round(5).tolist(),
                                     "anchor": d.xanchor[j].round(5).tolist()} for j in joints},
               "bodies": {_name(m, mujoco.mjtObj.mjOBJ_BODY, b): np.r_[d.xpos[b], d.xquat[b]].round(5).tolist()
                          for b in bodies},
               "goal": [[*g, bool(env.env._eval_predicate(g))] for g in problem["goal_state"]]}
        point = kw.get(POINT.get(event))
        if point is not None:
            p = np.asarray(point, float)
            rec["probe"] = {jname[j]: _probe(m, d, geoms[j], d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_axis[j], p)
                            for j in joints}
        write(rec)
    h.log = logged
