"""CP-5 executive: S1 writes one metric-free program; S2 grounds, plans, executes, monitors, repairs.

Timeline: t=0 capture head+wrist RGB-D -> PROGRAM request (async) -> wrist survey while
Astra thinks -> ground objects (SAM 3.1 + depth, multi-view) -> per stage: look, solve,
move, contact template, measure relation -> local retry ladder -> REPAIR only when the
ladder is exhausted -> VERIFY once at the end.
"""
from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from traceplan.contract import EndCandidate

from . import articulate
from .body import ControllerModel
from .contact import ContactMixin
from .guard import ContactDetector
from .snapshot import entity_of
from .tracemotion import TraceMotion
from .levels import LADDER, tube as _tube
from .lrl import CFG as LRL, LrlMixin
from .geometry import frame_from_axes, rot_z, rotvec_to_mat, top_down_rotation, mat_to_rotvec
from .llm import AstraError, image_part, text_part
from . import manip
from .manip import (container_region, grasp_candidates, outline, placement_candidates, preshape_finger,
                    reserve_space, soft_candidates, support_region, _wrap_pi, set_gripper_geometry)
from .motion import Attached, Checker, Executor, Planner, densify
from .perception import (FREE, OCC, UNKNOWN, ObjectModel, SceneMap, estimate_table_z, grow_region,
                         hull_surface_samples, mask_points, project_mask, voxel_down)
from scipy.spatial import cKDTree
from .program import (EFFECT_INSTRUCTIONS, EFFECT_SCHEMA, IMG, PLACE_LOOK_INSTRUCTIONS, PLACE_LOOK_SCHEMA,
                      PLACE_LOOK_INSTRUCTIONS_UNIFIED, PLACE_POSE_INSTRUCTIONS, PLACE_POSE_INSTRUCTIONS_UNIFIED,
                      PLACE_POSE_SCHEMA, PLACE_POSE_SCHEMA_UNIFIED,
                      PROGRAM_INSTRUCTIONS, PROGRAM_SCHEMA,
                      REPAIR_INSTRUCTIONS, REPAIR_SCHEMA, VERIFY_INSTRUCTIONS, VERIFY_SCHEMA, ProgramError,
                      merge_compiles, validate_program)
from .sim_iface import HEAD_CAMERA, WRIST_CAMERA

TRANSIT = {"arm": 0.02, "palm": 0.02, "finger": 0.015, "payload": 0.015,
           "plane_arm": 0.03, "plane_palm": 0.02, "plane_finger": 0.015, "plane_payload": 0.012}
NEAR = {"arm": 0.008, "palm": 0.008, "finger": 0.004, "payload": 0.004,
        "plane_arm": 0.015, "plane_palm": 0.008, "plane_finger": 0.002, "plane_payload": 0.001}
# placement margin ladder (user 2026-11-22): tier 0 = NEAR; tier 1 = half of it; tier 2 = half of it, and what the
# scoring rule lets be touched may be touched (the destination by hand and payload; what is in/on it and the other
# task objects by the payload). The arm keeps NEAR at every tier.
# Tier 3 exists only with placement "v1" (rank, don't reject): hand and payload need no margin, only no predicted
# contact, and only against what the scoring rule counts; the destination and the fixture it belongs to rank a
# candidate (fewest predicted contacts first) and never rule it out
TIER_SCALE = (1.0, 0.5, 0.5, 0.0)
# placement "v2": at most this many put-downs per stage call before the stage fails over to REPAIR
V2_TRIES = 4
# a grasp's fingertips may touch the support under the object (fingertip clearance is closed by contact), the rest of
# the hand keeps NEAR: one ordered tolerance chain from the generator (TIP_CLEAR) through IK to the configuration check
GRASP_NEAR = {**NEAR, "plane_finger": 0.0}
FLIP = np.diag([-1.0, -1.0, 1.0])     # the gripper's 180-degree twin about its approach axis (pads are symmetric)
DEPART_MAX = 0.20                     # m: a departure goes at most this far before TRACE takes over
GRASP_JOINT_SLACK = 0.08              # rad: a grasp posture keeps this far from every joint limit (the lift starts there)
SLIDE = (0.03, 0.30)                  # m: hand-off search range above a put-down's contact pose (TRACE slide)
# lateral window (m) of a placement for TRACE's standoff check: CP-5's region radius of the stage's tolerance class
PLACE_WINDOW = {"in": {"tight": 0.03, "normal": 0.06, "loose": 0.2}, "on": {"tight": 0.03, "normal": 0.05, "loose": 0.09},
                "beside": {"tight": 0.05, "normal": 0.05, "loose": 0.05}}


class StageFailure(Exception):
    def __init__(self, reason, **info):
        super().__init__(reason)
        self.reason = reason
        self.info = info


class Harness(ContactMixin, LrlMixin):
    def __init__(self, robot, instruction, run_dir, astra, segmenter, max_wall_s=900.0, max_repairs=2,
                 verify=True, log_images=True, prompt_note=None, motion="trace", trace_device="cuda:0",
                 trace_budget_ms=2000.0, compiles=1, placement="ladder", in_release=None, lrl_parts=(),
                 lrl_weights=None):
        # in_release: None = each "in" transfer releases as the program says ("place" | "drop"); "drop" / "place":
        #   every "in" transfer uses that release, whatever the program says (a run-wide setting, all destinations).
        # placement: how a held object is put down.
        #   "ladder": the careful planner with the margin ladder (tiers 0-2); it returns nothing when no candidate
        #             passes every test.
        #   "v1":     the same planner, ranking instead of rejecting on the destination side (tier 3).
        #   "v2":     no placement planner (r1's mechanism): System 1 points, the descent ends on contact, System 1
        #             judges the result and points again.
        #   "unified": the contact stage of CONTACT_STAGE_PLAN.md: System 1 points, geometry finds the rest pose near
        #             the spot, the way down carries its expected contact and the gripper opens only when it holds
        #             (contact.py); System 1 judges the result.
        #   "lrl":    v2, with the learned contact layer of LRL_PLAN.md (lrl.py): each part named in lrl_parts takes
        #             the place of v2's code for its step; with no part it is v2.
        assert placement in ("ladder", "v1", "v2", "unified", "lrl"), placement
        assert in_release in (None, "drop", "place"), in_release
        self.placement = placement
        self.lrl_setup(placement, lrl_parts, lrl_weights)
        self.in_release = in_release
        self.rb = robot
        self.rm = robot.model
        self.instruction = instruction
        self.prompt_note = prompt_note          # optional task-specific text for the S1 prompts (operator-given)
        self.run_dir = Path(run_dir)
        (self.run_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.astra = astra
        self.seg = segmenter
        self.max_wall_s = max_wall_s
        self.max_repairs = max_repairs
        self.do_verify = verify
        self.log_images = log_images
        self.t0 = time.time()
        self._events = open(self.run_dir / "events.jsonl", "w")
        self.objects = {}
        self.id_map = {}
        self.frames = []
        self.ledger = []
        self.program = None
        self.repairs = 0
        self.timing = {}
        self.ex = Executor(robot, self.rm, log=self.log)
        self.gripper_geom = self.rm.gripper_geometry(robot.q()[:7])
        set_gripper_geometry(self.gripper_geom)
        self.log("gripper_geometry", **self.gripper_geom)
        self.held = None
        self.held_width = None
        self.finished = False
        self._escapes = 0
        self.swept = []                # configurations the arm has physically occupied (re-carved on rebuilds)
        self.recorder = None           # evaluator-side recorder (videos); only ever called, never read
        # TRACE integration: motion "trace" | "v0" route every contact-free segment through the TRACE contract,
        # "cp5" is CP-5's planner unchanged (A/B)
        self.motion = motion
        self.compiles = compiles
        self.cm = ControllerModel(self.rm)
        self.detector = ContactDetector(self.cm)
        self.cur_stage = None          # the stage being executed: its terms go into every TRACE request
        self.dropped_terms = []        # language terms TRACE's program tests rejected (next REPAIR report)
        self.tm = None
        if motion != "cp5":
            (self.run_dir / "trace").mkdir(exist_ok=True)
            self.tm = TraceMotion(self, motion, trace_device, trace_budget_ms, self.run_dir / "trace")

    # ================================================================ utilities
    def log(self, event, **kw):
        rec = {"t": round(time.time() - self.t0, 3), "step": self.rb.steps, "event": event}
        rec.update(kw)
        self._events.write(json.dumps(rec, default=_jsonable) + "\n")
        self._events.flush()

    def elapsed(self):
        return time.time() - self.t0

    def _rec(self, kind, *args):
        """Hand read-only state to the evaluator-side recorder (no effect on the policy)."""
        if self.recorder is None:
            return
        try:
            getattr(self.recorder, kind)(self, *args)
        except Exception as e:
            self.log("recorder_error", kind=kind, error=repr(e))

    def _tick(self, name, t):
        self.timing[name] = round(self.timing.get(name, 0.0) + (time.time() - t), 3)

    def save_img(self, name, rgb):
        if self.log_images:
            cv2.imwrite(str(self.run_dir / "frames" / f"{self.rb.steps:05d}_{name}.png"), rgb[:, :, ::-1])

    # ================================================================ perception
    def capture(self, cam):
        fr = self.rb.capture(cam)
        if self.held is not None and self.checker.attached is not None:
            # the held object moves with the hand: none of its pixels may enter the map as an obstacle
            _, att = self.checker.world_points(self.rb.q()[None])
            fr.meta["held_mask"] = ndimage.binary_fill_holes(project_mask(fr, att[0], radius_px=5))
            # ... nor any depth point inside its box in the hand frame (its model can fall short of it)
            S = self.checker.attached.pts_site
            Rs, ps = self.rb.eef_pose()
            fr.meta["held_box"] = (Rs.copy(), ps.copy(), S.min(0) - 0.01, S.max(0) + 0.01)
        return fr

    def shadow_rule(self, fr):
        if fr.name != HEAD_CAMERA:
            return None
        return (self.table_z + 0.12, self.rm.root_p[0] + 0.25, self.table_z + 0.02)

    def integrate(self, frames, labels=None):
        for i, fr in enumerate(frames):
            lab = None if labels is None else labels[i]
            self.scene.integrate(fr, lab, extra_mask=fr.meta.get("held_mask"), robot_shadow_free=self.shadow_rule(fr),
                                 held_box=fr.meta.get("held_box"))
        self._remember_free()
        self.carve_robot()
        self._clear_held_ghost()
        self._clear_vacated()
        self._recall_free()
        self.scene.finalize()
        self._rec("on_map", "integrate")

    # ---------------------------------------------------------------- free-space memory (scene rule 1)
    def _remember_free(self):
        """Free space a camera (or the robot's body) has shown stays known free across map rebuilds."""
        mem = getattr(self, "free_mem", None)
        if mem is None or mem.shape != self.scene.shape:
            self.free_mem = mem = np.zeros(self.scene.shape, bool)
        mem |= self.scene.state == FREE

    def _recall_free(self):
        """Unseen voxels of the current map that were seen free before (and nothing was put there since) are free."""
        mem = getattr(self, "free_mem", None)
        if mem is None or mem.shape != self.scene.shape:
            return
        sel = (self.scene.state == UNKNOWN) & mem
        if sel.any():
            self.scene.state[sel] = FREE
            self.scene._touch()

    def _forget_free(self, pts, pad=0.015):
        """Something is (now) there: free space remembered around these points no longer holds."""
        mem = getattr(self, "free_mem", None)
        pts = np.asarray(pts, float).reshape(-1, 3)
        if mem is None or len(pts) == 0:
            return
        a = np.clip(self.scene.index(pts.min(0) - pad), 0, np.array(mem.shape) - 1)
        b = np.clip(self.scene.index(pts.max(0) + pad), 0, np.array(mem.shape) - 1)
        mem[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = False

    def _clear_vacated(self):
        """Volumes objects were lifted out of: their unlabelled leftover (pixels outside the object's mask in frames
        taken before the lift) is cleared again after every map rebuild; labelled objects are kept."""
        for oid, lo, hi in getattr(self, "_vacated", []):
            n = self.scene.clear_box(np.eye(3), np.zeros(3), lo, hi, 0.0)
            if n:
                self.log("vacated_cleared", object=self.objects[oid].name if oid in self.objects else oid, points=n)

    def _clear_held_ghost(self, margin=0.01):
        """The held object fills its own volume: what the cameras show inside its box in the hand frame (grown by
        margin) is the object itself where its model falls short of it (the held mask lets those pixels through),
        never an obstacle. Unlabeled occupancy and cloud points there are cleared; labelled objects are kept."""
        if self.held is None or self.checker.attached is None:
            return
        S = self.checker.attached.pts_site
        if S is None or len(S) < 8:
            return
        Rs, ps = self.rb.eef_pose()
        n = self.scene.clear_box(Rs, ps, S.min(0), S.max(0), margin)
        if n:
            self.log("held_ghost_cleared", points=int(n))

    def carve_robot(self, Q=None, hulls=True):
        """Space the robot body occupies (now, or along a path it just executed) holds nothing else:
        mark it free - around the surface samples and through each link's interior (convex hull), so
        never-observed space the arm moved into does not linger inside its own links."""
        Q = self.rb.q()[None, :7] if Q is None else np.atleast_2d(Q)
        for q in Q[:, :7]:
            if not self.swept or np.abs(np.asarray(self.swept) - q).max(axis=1).min() > 0.08:
                self.swept.append(np.array(q, float))
        W, att = self.checker.world_points(Q, finger=0.04)
        pts = W.reshape(-1, 3)
        if att is not None:
            pts = np.concatenate([pts, att.reshape(-1, 3)])
        self.scene.carve_points(pts, radius=0.012)
        if not hulls:
            return
        if not hasattr(self, "_link_tri"):
            from scipy.spatial import Delaunay
            self._link_tri = []
            for bi, loc in enumerate(self.rm.points.local):
                if len(loc) >= 8:
                    try:
                        tri = Delaunay(loc)
                        self._link_tri.append((bi, tri, loc[tri.convex_hull.ravel()]))
                    except Exception:
                        pass
        Qh = Q[::max(1, int(np.ceil(len(Q) / 8)))]
        R, P = self.rm.fk(self.rm.full_q(Qh[:, :7], 0.04))
        posed = []
        for n in range(len(Qh)):
            for bi, tri, verts in self._link_tri:
                posed.append((tri, R[n, bi], P[n, bi], P[n, bi] + verts @ R[n, bi].T))
        self.scene.carve_hulls(posed)

    def build_scene(self, head):
        base = self.rm.root_p
        self.table_z = estimate_table_z(head, base[:2])
        self.table_z_visual = self.table_z          # the table surface as depth sees it (table_z may rise to where objects rest)
        self.visual_table_z = self.table_z
        lo = np.array([base[0] - 0.25, -0.75, self.table_z - 0.06])
        hi = np.array([base[0] + 1.15, 0.75, self.table_z + 0.74])
        self.scene = SceneMap(lo, hi, res=0.01, table_z=self.table_z)
        self.scene.visual_table_z = self.table_z
        self.scene.robot_zone = (base[0], base[1], 0.13, self.table_z - 0.1, base[2] + 0.42)
        self.scene.out_of_view_free_z = self.table_z + 0.12      # same height rule as the robot-shadow rule
        self.checker = Checker(self.rm, self.scene, self.objects)
        self.planner = Planner(self.checker, rng=np.random.default_rng(7), log=self.log)
        self.log("scene", table_z=self.table_z, lo=lo, hi=hi, shape=self.scene.shape)

    def detect(self, frame, prompts):
        dets = []
        for p in prompts:
            try:
                for m, s in self.seg.detect(frame.rgb, p):
                    dets.append((m & ~frame.robot_mask, s, p))
            except Exception as e:  # segmenter failure must not kill the episode
                self.log("segment_error", prompt=p, error=repr(e))
        return [d for d in dets if d[0].sum() >= 15]

    def mask_for_point(self, frame, prompts, uv):
        u, v = int(uv[0]), int(uv[1])
        dets = self.detect(frame, prompts)
        H, W = frame.depth.shape
        best, best_key = None, None
        for m, s, p in dets:
            if m.sum() > 0.35 * H * W:
                continue
            if m[min(max(v, 0), H - 1), min(max(u, 0), W - 1)]:
                key = (0, -s * (1.0 - 0.05 * prompts.index(p)))
            else:
                dist = ndimage.distance_transform_edt(~m)[min(max(v, 0), H - 1), min(max(u, 0), W - 1)]
                if dist > 14:
                    continue
                key = (1, dist)
            if best_key is None or key < best_key:
                best, best_key = (m, s, p), key
        return best

    def mask_matching(self, frame, prompts, model, min_iou=0.2):
        proj = project_mask(frame, model.points, radius_px=3)
        if proj.sum() < 15:
            return None
        best, best_iou = None, min_iou
        for m, s, p in self.detect(frame, prompts):
            inter = (m & proj).sum()
            iou = inter / max((m | proj).sum(), 1)
            if iou > best_iou:
                best, best_iou = (m, s, p, iou), iou
        return best

    def filter_points(self, pts, model, pad=0.05):
        if len(pts) == 0 or len(model.points) == 0:
            return pts
        lo, hi = model.bbox
        keep = np.all(pts[:, :2] >= lo[:2] - pad, axis=1) & np.all(pts[:, :2] <= hi[:2] + pad, axis=1)
        keep &= pts[:, 2] > self.table_z + 0.004
        return pts[keep]

    def survey(self, head):
        """Wrist views above the clusters of objects visible from the head camera (runs during PROGRAM)."""
        t = time.time()
        pts, _ = head.backproject(mask=~head.robot_mask, stride=2)
        base = self.rm.root_p
        sel = (pts[:, 2] > self.table_z + 0.012) & (pts[:, 0] > base[0] + 0.2) & (pts[:, 0] < base[0] + 0.95)
        sel &= np.abs(pts[:, 1]) < 0.6
        obj_pts = pts[sel]
        if len(obj_pts) < 50:
            return []
        xy = obj_pts[:, :2]
        lo, hi = xy.min(0), xy.max(0)
        span = hi - lo
        nx = 1 if span[0] < 0.32 else 2
        ny = 1 if span[1] < 0.32 else 2
        targets = []
        for i in range(nx):
            for j in range(ny):
                cx = lo[0] + span[0] * (i + 0.5) / nx
                cy = lo[1] + span[1] * (j + 0.5) / ny
                targets.append(np.array([cx, cy]))
        # visit in a nearest-neighbour order from the current camera position
        Rc, pc = self.rm.wrist_camera_pose(self.rb.q()[None])
        order, cur = [], pc[0][:2]
        rest = list(range(len(targets)))
        while rest:
            k = min(rest, key=lambda i: np.linalg.norm(targets[i] - cur))
            order.append(k)
            cur = targets[k]
            rest.remove(k)
        frames = []
        for k in order:
            if self.elapsed() > 120:
                break
            fr = self.look_from_above(np.r_[targets[k], self.table_z], height=0.42, settle=4, rrt=False)
            if fr is not None:
                frames.append(fr)
        self._tick("survey", t)
        self.log("survey", views=len(frames), targets=[t_.tolist() for t_ in targets])
        return frames

    def camera_offset(self):
        q = self.rb.q()[None]
        Rs, ps = self.rm.site_pose(q)
        Rc, pc = self.rm.wrist_camera_pose(q)
        return Rs[0].T @ (pc[0] - ps[0]), Rs[0].T @ Rc[0]

    def look_from_above(self, point, height=0.3, yaw=None, settle=3, margins=None, rrt=True):
        """Move so the wrist camera looks down at `point` from `height` above it; capture a frame."""
        off, Rcs = self.camera_offset()
        R0, _ = self.rb.eef_pose()
        yaw0 = np.arctan2(R0[1, 0], R0[0, 0]) if yaw is None else yaw
        tries = [(d, False) for d in (0.0, np.pi / 2, -np.pi / 2, np.pi)] + ([(0.0, True)] if rrt else [])
        for dyaw, use_rrt in tries:
            # holding something: keep the current hand orientation (only yaw about vertical changes)
            Rs = rot_z(dyaw) @ R0 if self.held is not None else top_down_rotation(yaw0 + dyaw)
            d = Rs @ Rcs @ np.array([0, 0, -1.0])
            h_view = height
            if self.held is not None and self.motion != "cp5":
                # a held payload hangs below the hand: raise the view by its extent below the grip site, so the
                # payload ends where an empty hand's fingers would be (else every look pose puts it into the scene)
                h_view += max(0.0, -float((self.checker.attached.pts_site @ Rs.T)[:, 2].min()))
            z_cam = point[2] + h_view
            cam_xy = point[:2] - d[:2] * (h_view / max(-d[2], 0.3))
            p_cam = np.r_[cam_xy, z_cam]
            p_site = p_cam - Rs @ off
            sols = self.rm.solve_ik(Rs, p_site, self.rb.q()[:7], n_random=12)
            if len(sols) == 0:
                continue
            ok = self.go_configs(list(sols[:4]), margins or TRANSIT, grip=self.rb.grip_cmd,
                                 finger=self._finger(), eef_v=0.35, tag="look", allow_rrt=use_rrt, rrt_budget=2.0)
            if ok:
                self.ex.run_path(np.array([self.rb.q()[:7]] * 2), settle=True, max_extra=settle)
                fr = self.capture(WRIST_CAMERA)
                self.frames.append(fr)
                self.save_img("look", fr.rgb)
                return fr
        self.log("look_failed", point=point)
        return None

    def look_toward(self, point, view_dir, dist=0.25, settle=3, margins=None, allow_rrt=True, escape=True,
                    min_limit_margin=None):
        """Wrist camera at `point - dist * view_dir`, optical axis along view_dir; capture a frame."""
        off, Rcs = self.camera_offset()
        v = np.asarray(view_dir, float)
        v /= np.linalg.norm(v)
        z_c = -v                                         # MuJoCo cameras look along their -z axis
        x_c = np.cross([0.0, 0.0, 1.0], z_c)
        x_c = x_c / np.linalg.norm(x_c) if np.linalg.norm(x_c) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y_c = np.cross(z_c, x_c)
        Rc0 = np.stack([x_c, y_c, z_c], 1)
        p_cam = np.asarray(point, float) - v * dist
        lim = self.rm.limits[:7]
        for roll in (0.0, np.pi, np.pi / 2, -np.pi / 2):
            Rc = Rc0 @ rot_z(roll)
            Rs = Rc @ Rcs.T
            sols = self.rm.solve_ik(Rs, p_cam - Rs @ off, self.rb.q()[:7], n_random=12)
            if len(sols) and min_limit_margin is not None:
                # a viewpoint the arm only reaches at a joint limit leaves it where the next moves cannot be tracked
                sols = sols[np.minimum(sols - lim[:, 0], lim[:, 1] - sols).min(axis=1) >= min_limit_margin]
            if len(sols) == 0:
                continue
            ok = self.go_configs(list(sols[:4]), margins or TRANSIT, grip=self.rb.grip_cmd, finger=self._finger(),
                                 eef_v=0.35, tag="look_toward", rrt_budget=2.0, allow_rrt=allow_rrt,
                                 escaped=not escape)
            if ok:
                self.ex.run_path(np.array([self.rb.q()[:7]] * 2), settle=True, max_extra=settle)
                fr = self.capture(WRIST_CAMERA)
                self.frames.append(fr)
                self.save_img("look_toward", fr.rgb)
                return fr
        self.log("look_failed", point=point, view=v)
        return None

    def side_coverage(self, obj, bins=12):
        """How much of an object's side has been observed: the largest azimuth gap (degrees) around its
        footprint centre with no observed side-surface points, and the gap's centre azimuth (rad).
        Short objects (< 4 cm) count as covered (their top view bounds them)."""
        P = obj.points
        if len(P) < 30:
            return 360.0, 0.0
        lo, hi = float(P[:, 2].min()), obj.top_z
        h = hi - lo
        if h < 0.04:
            return 0.0, 0.0
        band = P[(P[:, 2] > lo + 0.2 * h) & (P[:, 2] < hi - 0.2 * h)]
        if len(band) < 10:
            return 360.0, 0.0
        c = obj.footprint_center()
        az = np.arctan2(band[:, 1] - c[1], band[:, 0] - c[0])
        k = np.floor((az + np.pi) / (2 * np.pi / bins)).astype(int) % bins
        cov = np.bincount(k, minlength=bins) >= 3
        if cov.all():
            return 0.0, 0.0
        if not cov.any():
            return 360.0, 0.0
        best, start = 0, 0
        for s in range(bins):
            if cov[s - 1] and not cov[s]:
                n = 0
                while not cov[(s + n) % bins]:
                    n += 1
                if n > best:
                    best, start = n, s
        return best * 360.0 / bins, float(-np.pi + (start + best / 2.0) * 2 * np.pi / bins)

    def match_view(self, frame, prompts, model):
        """The segment of `model` in a new view: projected-mask IoU, else the detection that contains most of
        the projected model (a new viewpoint sees surfaces the model lacks), else the nearest 3D centre."""
        mt = self.mask_matching(frame, prompts, model, min_iou=0.15)
        if mt is not None:
            return mt[0], float(mt[3])
        proj = project_mask(frame, model.points, radius_px=3)
        dets = self.detect(frame, prompts)
        H, W = frame.depth.shape
        best, best_s = None, 0.5
        if proj.sum() >= 15:
            for m, s, p in dets:
                if m.sum() > 0.35 * H * W:
                    continue
                cont = (m & proj).sum() / max(proj.sum(), 1)
                if cont > best_s:
                    best, best_s = m, cont
        if best is not None:
            return best, float(best_s)
        c0 = model.points.mean(0)
        best_d = 0.06
        for m, s, p in dets:
            pts = mask_points(frame, m)
            if len(pts) < 20 or m.sum() > 0.35 * H * W:
                continue
            d = float(np.linalg.norm(pts.mean(0) - c0))
            if d < best_d:
                best, best_d = m, d
        return (best, 0.0) if best is not None else None

    def look_unseen_side(self, obj, min_gap=120.0):
        """Before lifting an object: if a large part of its side has never been seen (the head camera sees
        one side only), look at that side with the wrist camera and add what it shows to the model."""
        gap, az = self.side_coverage(obj)
        if gap < min_gap:
            return False
        c = obj.footprint_center()
        zc = float(obj.points[:, 2].min() + 0.5 * (obj.top_z - obj.points[:, 2].min()))
        q_before = self.rb.q()[:7].copy()
        added, looked = 0, None
        for tilt_deg, rot in ((35.0, 0.0), (50.0, 0.0), (35.0, 0.5), (35.0, -0.5)):
            a, tilt = az + rot, np.deg2rad(tilt_deg)
            d = np.array([np.cos(a) * np.cos(tilt), np.sin(a) * np.cos(tilt), np.sin(tilt)])   # object -> camera
            fr = self.look_toward(np.r_[c, zc], -d, dist=0.30, allow_rrt=False, escape=False, min_limit_margin=0.25)
            if fr is None:
                continue
            looked = (tilt_deg, rot)
            mt = self.match_view(fr, [obj.category] + ([obj.name] if obj.name != obj.category else []), obj)
            if mt is not None:
                n0 = len(obj.points)
                obj.add_points(self.filter_points(mask_points(fr, mt[0]), obj, pad=0.05))
                obj.views.append((fr.name, "side", mt[1]))
                added = len(obj.points) - n0
            break
        back = None
        if looked is not None:
            # return to the configuration the arm came from: the next moves then start exactly as they would
            # have without the detour (IK seeded at the side view leaves the arm in a twisted posture)
            back = self.go_configs([q_before], TRANSIT, grip=self.rb.grip_cmd, finger=self._finger(), eef_v=0.35,
                                   tag="side_look_back", allow_rrt=False, escaped=True)
        gap2, _ = self.side_coverage(obj)
        self.log("side_look", object=obj.name, gap_deg=gap, azimuth=az, looked=looked, added=added,
                 gap_after_deg=gap2, returned=back)
        return looked is not None

    def payload_model(self, obj):
        """Closed collision model of the object about to be lifted (world frame).

        Observed surface points never close a solid: the side facing away from every camera and the base
        resting on the support are missing, so an obstacle could slip into the hole. The model is the convex
        hull of the observed points and of their projection onto the level of the lowest observed point
        (the unseen base and lower sides lie inside it). Last resort, when a large part of the side was never
        seen and no wrist view of the object exists (its top outline may be incomplete too), the observed
        points mirrored through the footprint centre are added as an extent bound.
        Returns (hull surface + observed points, info)."""
        P = obj.points
        if len(P) > 50:
            # isolated stray points (segment bleeding) must not inflate a hull
            P = P[cKDTree(P).query_ball_point(P, 0.012, return_length=True) >= 6]
        base_z = float(P[:, 2].min())
        S = [P, np.c_[P[:, :2], np.full(len(P), base_z)]]
        gap, _ = self.side_coverage(obj)
        mirrored = gap >= 120.0 and not any(v[0] == WRIST_CAMERA for v in obj.views)
        if mirrored:
            M = P.copy()
            M[:, :2] = 2 * obj.footprint_center() - P[:, :2]
            S += [M, np.c_[M[:, :2], np.full(len(M), base_z)]]
        S = voxel_down(np.concatenate(S), 0.004)
        surf = hull_surface_samples(S, spacing=0.006)
        full = voxel_down(np.concatenate([P, surf]), 0.003) if len(surf) else P
        info = {"observed": int(len(P)), "points": int(len(full)), "mirrored": mirrored, "side_gap_deg": gap,
                "size_observed": np.round(np.ptp(P, axis=0), 3).tolist(),
                "size_model": np.round(np.ptp(full, axis=0), 3).tolist()}
        return full, info

    def nearest_obstacle(self, P, exclude=()):
        """What a set of world points is closest to (for failure reports): the object whose observed surface is
        nearest, 'unsegmented surface', 'unobserved space' or 'table', with the distance."""
        P = np.asarray(P, float).reshape(-1, 3)
        if len(P) == 0:
            return None
        near = self.checker.near
        near._labels(self.scene)
        excl = set(int(e) for e in exclude) | set(near.held) | ({self.held} if self.held is not None else set())
        keep = ~np.isin(near.lab, list(excl)) if excl else np.ones(len(near.lab), bool)
        best = ("table", float(P[:, 2].min() - self.table_z))
        if keep.any():
            d, i = cKDTree(near.cloud[keep]).query(P)
            k = int(np.argmin(d))
            if d[k] < best[1]:
                lab = int(near.lab[keep][i[k]])
                best = (self.objects[lab].name if lab in self.objects else "unsegmented surface", float(d[k]))
        c = self.scene.clearance(P, tuple(excl))
        if float(c.min()) + 0.01 < best[1] - 0.01:
            best = ("unobserved space", max(0.0, float(c.min())))
        return {"object": best[0], "distance_m": round(best[1], 4)}

    def reobserve(self):
        """Fresh head + wrist frames into the map (after an unexpected contact or a stale-map failure)."""
        frames = [self.capture(HEAD_CAMERA), self.capture(WRIST_CAMERA)]
        self.frames += frames
        self.rebuild_map(self.frames[-8:])

    def _lift_heights(self, q0, goals, finger):
        """Grip-site heights for the up-over-down route: the lowest point of the hand and of a hanging payload
        must pass above the tallest observed thing under the corridor from start to goal."""
        if not goals:
            return [self.table_z + 0.30]
        Q = np.array([np.asarray(q0, float)[:7], np.asarray(goals[0], float)[:7]])
        W, att = self.checker.world_points(Q, finger)
        _, ps = self.rm.site_pose(self.rm.full_q(Q, finger))
        hang = W[0][self.checker.pg >= 1]
        if att is not None:
            hang = np.concatenate([hang, att[0]])
        below = float(ps[0][2] - hang[:, 2].min())
        reach = float(np.linalg.norm(hang[:, :2] - ps[0][:2], axis=1).max()) + 0.03
        a, b = ps[0][:2], ps[1][:2]
        lo, hi = np.minimum(a, b) - reach, np.maximum(a, b) + reach
        excl = (self.held,) if self.held is not None else ()
        hmap, org = self.scene.heightmap((lo[0], hi[0]), (lo[1], hi[1]), exclude=excl)
        ij = np.argwhere(np.isfinite(hmap))
        top = self.table_z
        if len(ij):
            xy = np.array(org) + (ij + 0.5) * self.scene.res
            ab = b - a
            tt = np.clip(((xy - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)
            dseg = np.linalg.norm(xy - (a + tt[:, None] * ab), axis=1)
            sel = dseg <= reach
            if sel.any():
                top = max(top, float(hmap[ij[sel, 0], ij[sel, 1]].max()))
        need = top + below + 0.035
        cap = self.table_z + 0.65
        base = self.table_z + 0.30                 # the default route height (enough for an empty hand)
        if need <= base:
            return [base] + ([need] if self.held is not None and base - need > 0.03 else [])
        out = [min(need, cap)]
        if need + 0.08 <= cap:
            out.append(need + 0.08)
        if self.held is None:
            out.append(base)
        return out

    # ================================================================ motion helpers
    def _finger(self):
        return 0.04 if self.held is None else max(0.0, (self.rb.q()[7] - self.rb.q()[8]) / 2)

    def go_configs(self, goals, margins, grip=None, finger=0.04, exclude=(), eef_v=0.3, tag="move",
                   lift_z=None, allow_rrt=True, rrt_budget=6.0, goal_near=False, escaped=False):
        if self.motion != "cp5":
            return self.trace_move([EndCandidate(*self._site_pose(g, finger), q=np.asarray(g, float)[:7]) for g in goals],
                                   tag=tag, grip=grip, finger=finger)
        t = time.time()
        q0 = self.rb.q()[:7]
        self.carve_robot()
        self.planner.set_relax(q0, exclude=exclude, finger=finger)
        self.planner.goal_relax = ([np.asarray(g, float)[:7] for g in goals], NEAR) if goal_near else None
        if lift_z is None:
            lift_z = self._lift_heights(q0, goals, finger)
        self.last_move = {"tag": tag}
        path, how = self.planner.plan(q0, goals, margins, exclude=exclude, finger=finger, lift_z=lift_z,
                                      time_budget=rrt_budget, allow_rrt=allow_rrt)
        self.planner.set_relax(None)
        self.planner.goal_relax = None
        self._tick("plan", t)
        if path is None and allow_rrt and not escaped and not getattr(self, "_escaping", False) \
                and self._escapes < 4:
            # stuck: back out (straight up, then toward the start posture) and plan once more
            self.log("plan_fail", tag=tag, how=how)
            self._escapes += 1
            if self.escape():
                return self.go_configs(goals, margins, grip, finger, exclude, eef_v, tag + "_retry", lift_z,
                                       allow_rrt, rrt_budget, goal_near, escaped=True)
            return False
        if path is None:
            self.log("plan_fail", tag=tag, how=how)
            self.last_move = {"tag": tag, "status": "no_path", "planner": how}
            return False
        t = time.time()
        self._rec("on_plan", path, finger, tag, how)
        if self.held is not None:
            eef_v = min(eef_v, 0.15)
        res = self.ex.run_path(path, grip=grip, finger=finger, eef_v=eef_v, vmax=0.6 if self.held else 1.2,
                               drop_check=self._drop_check if self.held is not None else None)
        self._tick("execute", t)
        self.last_move = {"tag": tag, "status": res.status, "planner": how, "lift_z": lift_z if how == "lift" else None}
        if res.status == "done":
            self.carve_robot(densify(path, 0.15))
        else:
            # stopped part-way (the rest of the path was never occupied: do not carve it)
            self.carve_robot()
        if res.status == "contact":
            # what the hand / payload ran into (measured contact force + nearest mapped thing)
            W, att = self.checker.world_points(self.rb.q()[None, :7], finger)
            pts = W[0][self.checker.pg >= 1] if att is None else np.concatenate([W[0][self.checker.pg >= 1], att[0]])
            self.last_move.update(force_n=np.round(res.info.get("force", []), 1).tolist(),
                                  contact_near=self.nearest_obstacle(pts, exclude))
        self.log("move", tag=tag, how=how, waypoints=len(path), status=res.status, steps=res.steps, info=res.info,
                 contact_near=self.last_move.get("contact_near"))
        self._rec("on_map", "move:" + tag)
        if res.status == "dropped":
            raise StageFailure("dropped_in_transit")
        return res.status == "done"

    # ---------------------------------------------------------------- TRACE contract (DESIGN §3.5, §3.6)
    def _site_pose(self, q, finger=0.04):
        Rs, ps = self.rm.site_pose(self.rm.full_q(np.asarray(q, float)[None, :7], finger))
        return Rs[0], ps[0]

    def trace_move(self, ends, tag="move", grip=None, finger=0.04):
        """A contact-free segment through TRACE to the first reachable end candidate (go_configs' contract: True
        when it got there; a drop in transit raises as in CP-5; no_path returns False, last_move says why)."""
        res = self.tm.move(ends, tag, grip, finger)
        if res is None:
            return False
        if res.status == "dropped":
            raise StageFailure("dropped_in_transit")
        return res.status == "done"

    def reach(self, contact_R, contact_p, q_contact, approach, entity_oid, window, tag, grip, finger, q_pre,
              alts=(), twin=True, standoff=(0.0, 0.10)):
        """Approach a contact pose (grasp, place, part, press). TRACE ends the segment where certification ends: from
        its perception standoff ("auto") back along the approach until its end test passes and the arm has a posture
        (standoff (s_min, s_max)); the caller's guarded move covers the rest.
        The hand-off is a set, not a point: the contact pose, the gripper's 180-degree twin about the approach axis
        (pads and closed fingertips are symmetric), and alternative contact poses `alts` (dicts R, p, q, approach,
        window), each with its twin, ordered so the hand turns least; TRACE certifies the first it can.
        Returns the end the hand went to (its R, p: the contact pose to go on to; .k: 0 = the given pose, i = alts[i-1];
        .twin: True for the turned twin) or None. CP-5: its pre-pose."""
        if self.motion == "cp5":
            if not self.go_configs([q_pre], TRANSIT, grip=grip, finger=finger, tag=tag, goal_near=True):
                return None
            e = EndCandidate(np.asarray(contact_R, float), np.asarray(contact_p, float), q=q_contact)
            e.k, e.twin = 0, False
            return e
        ent = entity_of(self, entity_oid)
        R0, _ = self.rb.eef_pose()
        main = {"R": contact_R, "p": contact_p, "q": q_contact, "approach": approach, "window": window}
        ends = []
        for k, c in enumerate([main] + list(alts)):
            a = np.asarray(c.get("approach") if c.get("approach") is not None else np.asarray(c["R"])[:, 2], float)
            a = a / np.linalg.norm(a)
            R = np.asarray(c["R"], float)
            q = None if c.get("q") is None else np.asarray(c["q"], float)[:7]
            w = c.get("window", window)
            grp = [(R, q, False)] + ([(R @ FLIP, None, True)] if twin else [])
            grp.sort(key=lambda g: _turn_angle(R0, g[0]))
            for Rg, qg, tw in grp:
                e = EndCandidate(Rg, np.asarray(c["p"], float), q=qg, approach=a, standoff=standoff,
                                 window=None if w is None else float(w), entity=ent)
                e.k, e.twin = k, tw
                ends.append(e)
        a0 = ends[0].approach
        res = self.tm.move(ends, tag, grip, finger, entity=ent, approach=a0, seed_goal=q_pre, goal_near=True,
                           keep_ends=len(ends))
        if res is None:
            return None
        if res.status == "dropped":
            raise StageFailure("dropped_in_transit")
        if res.status != "done" or self.tm.last_end is None:
            return None
        e = self.tm.last_end
        self.log("handoff", tag=tag, end=int(getattr(e, "k", 0)), twin=bool(getattr(e, "twin", False)),
                 n_ends=len(ends))
        return e

    def retreat(self, R_goal, p_goal, grip=None, speed=0.10, finger=None, tag="retreat", until_clear=True):
        """Leave a contact (lift, back-off, lift after a release): a guarded straight line toward p_goal (the reverse
        of the approach), continued until the hand and the held object clear the observed scene by TRACE's margins
        (at most DEPART_MAX), so the next certified segment does not start inside them. When the line has no posture
        or ends on contact, a certified TRACE segment to the goal; never an unguarded line. Returns an ExecResult or
        None (nothing moved the hand there)."""
        finger = self._finger() if finger is None else finger
        R_goal = np.asarray(R_goal, float)
        Rs, ps = self.rb.eef_pose()
        d = np.asarray(p_goal, float) - ps
        dist = float(np.linalg.norm(d))
        if dist < 1e-4:
            return None
        u = d / dist
        if until_clear and self.motion != "cp5":
            dist = self._depart_distance(R_goal, ps, u, dist, finger)
        goal = ps + u * dist
        # leaving a contact releases force (a press, a set-down): no force rule, only the contact detector, which
        # checks that the hand falls behind its predicted step (a released preload does not)
        res = self.move_line(R_goal, goal, grip=grip, speed=speed, finger=finger, guard=self._guard("contact", None),
                             force_limit=1e9)
        self.log("depart", tag=tag, status=None if res is None else res.status, dist=round(dist, 3))
        if res is not None and res.status in ("done", "dropped"):
            return res
        if self.motion != "cp5":
            res2 = self.tm.move([EndCandidate(R_goal, goal)], tag, grip, finger, retreat=True)
            if res2 is not None and res2.status in ("done", "dropped"):
                return res2
        return res

    def _depart_distance(self, R, p0, u, d0, finger):
        """How far a departure goes: at least d0, then on in 1 cm steps (at most DEPART_MAX) until the hand and the
        held object clear the observed scene and the table by TRACE's transit margins; else where they clear it
        most."""
        L, grp, rad = self._hand_local(finger)
        att = self.checker.attached
        excl = (self.held,) if self.held is not None else ()
        tv = _tube(LADDER, LADDER["v_min"])
        need = lambda g: LADDER["guard"][g] + tv + LADDER["perception"]
        need_pl = lambda g: LADDER["guard_plane"][g] + tv
        best, best_def = d0, np.inf
        for s in np.arange(d0, max(d0, DEPART_MAX) + 1e-9, 0.01):
            p = p0 + u * s
            Wh = p + L @ R.T
            c = self.checker.point_clearance(Wh, excl, unknown=False) - rad
            zc = Wh[:, 2] - self.table_z - rad
            deficit = max(need("palm") - c[grp == 1].min(), need("finger") - c[grp == 2].min(),
                          need_pl("palm") - zc[grp == 1].min(), need_pl("finger") - zc[grp == 2].min())
            if att is not None:
                Wp = p + att.pts_site @ R.T
                cp = self.checker.point_clearance(Wp, excl, unknown=False) - 0.004
                deficit = max(deficit, need("payload") - cp.min(), need_pl("payload") - (Wp[:, 2] - self.table_z).min())
            if deficit <= 0:
                return float(s)
            if deficit < best_def:
                best, best_def = float(s), deficit
        return best

    def look_at_point(self, at, approach=None):
        """A wrist view of a point a plan needs (unseen space next to a hand-off): from above, else along the
        approach. Frame or None."""
        at = np.asarray(at, float)
        fr = self.look_from_above(np.r_[at[:2], max(float(at[2]), self.table_z)], height=0.28, rrt=False)
        if fr is None and approach is not None and abs(float(np.asarray(approach)[2])) < 0.9:
            fr = self.look_toward(at, np.asarray(approach, float), dist=0.28, allow_rrt=False)
        return fr

    def _guard(self, reason, rule):
        """Guarded descents end on contact (ContactDetector with CP-5's force rule and reason kept); CP-5 alone in
        cp5 mode."""
        return rule if self.motion == "cp5" else self.detector.guard(reason, rule)

    def escape(self):
        """Leave a cramped configuration: lift the hand, then move toward the start posture."""
        self._escaping = True
        try:
            Rs, ps = self.rb.eef_pose()
            moved = False
            qs = self.line_configs(Rs, ps + np.array([0, 0, 0.08]), self.rb.q()[:7], finger=self._finger())
            if qs is not None:
                self.carve_robot()
                self.planner.set_relax(self.rb.q()[:7], finger=self._finger())
                ok, _, _ = self.planner.path_ok(qs, NEAR, finger=self._finger())
                self.planner.set_relax(None)
                if ok:
                    r = self.ex.run_path(qs, finger=self._finger(), eef_v=0.10,
                                         drop_check=self._drop_check if self.held is not None else None)
                    moved = r.status in ("done", "contact")
            q_home = self.q_home
            if np.abs(self.rb.q()[:7] - q_home).max() > 0.2:
                self.carve_robot()
                self.planner.set_relax(self.rb.q()[:7], finger=self._finger())
                path, how = self.planner.plan(self.rb.q()[:7], [q_home], NEAR, finger=self._finger(),
                                              lift_z=self.table_z + 0.30, time_budget=4.0)
                self.planner.set_relax(None)
                if path is not None:
                    r = self.ex.run_path(path, finger=self._finger(), eef_v=0.2,
                                         drop_check=self._drop_check if self.held is not None else None)
                    self.carve_robot(densify(path, 0.15))
                    moved = moved or r.status == "done"
            self.log("escape", moved=moved)
            return moved
        finally:
            self._escaping = False

    def _drop_check(self, robot):
        """A lost object lets the fingers close well past the settled width they held (the finger target sits
        inside contact). Debounced: two consecutive control steps; a grip that merely settles (the object
        seating between the pads) moves a few millimetres and must not count - a false drop opens the hand."""
        low = robot.finger_opening() < max(0.002, (self.held_width or 0.0) - 0.006)
        self._drop_hits = (getattr(self, "_drop_hits", 0) + 1) if low else 0
        return self._drop_hits >= 2

    def line_configs(self, R_goal, p_goal, q_start, n=8, finger=0.04):
        """IK along a straight Cartesian segment from the current pose (continuity seeded)."""
        R0, p0 = self.rm.site_pose(self.rm.full_q(np.asarray(q_start)[None, :7], finger))
        R0, p0 = R0[0], p0[0]
        dist = np.linalg.norm(p_goal - p0)
        n = max(2, int(np.ceil(dist / 0.012)))
        qs = [np.asarray(q_start, float)[:7]]
        from scipy.spatial.transform import Rotation, Slerp
        sl = Slerp([0, 1], Rotation.from_matrix(np.stack([R0, R_goal])))
        for k in range(1, n + 1):
            a = k / n
            Rk = sl([a]).as_matrix()[0]
            pk = p0 + a * (p_goal - p0)
            q, ok, ep, er = self.rm.ik(Rk, pk, qs[-1][None], q_rest=qs[-1], iters=60, finger=finger)
            if not ok[0] or np.abs(q[0] - qs[-1]).max() > 0.35:
                return None
            qs.append(q[0])
        return np.array(qs)

    def move_line(self, R_goal, p_goal, grip=None, speed=0.06, guard=None, finger=None, force_limit=30.0):
        finger = self._finger() if finger is None else finger
        qs = self.line_configs(R_goal, p_goal, self.rb.q()[:7], finger=finger)
        if qs is None:
            return None
        res = self.ex.run_path(qs, grip=grip, finger=finger, eef_v=speed, guard=guard, force_limit=force_limit,
                               drop_check=self._drop_check if self.held is not None else None)
        if res.status == "done":
            self.carve_robot(qs[::3])
        else:                       # stopped part-way: only what the arm went through holds nothing else
            k = int(np.argmin(np.abs(qs - self.rb.q()[:7]).max(axis=1)))
            self.carve_robot(np.r_[qs[:k + 1:3], self.rb.q()[None, :7]])
        return res

    # ================================================================ S1 calls
    def head_and_wrist_content(self, head, wrist):
        return [text_part("IMAGE 1 (head camera, 512x512):"), image_part(head.rgb),
                text_part("IMAGE 2 (wrist camera):"), image_part(wrist.rgb)]

    def task_parts(self):
        parts = [text_part(f"Task instruction: {self.instruction}")]
        if self.prompt_note:
            parts.append(text_part(f"Operator note for this task: {self.prompt_note}"))
        return parts

    def call_program(self, head, wrist):
        content = self.task_parts() + self.head_and_wrist_content(head, wrist)
        return self.astra.submit("program", PROGRAM_INSTRUCTIONS, content, PROGRAM_SCHEMA)

    def call_program_extra(self, head, wrist, n):
        """--compiles: n extra low-effort PROGRAM calls in parallel (term confidence by self-consistency)."""
        content = self.task_parts() + self.head_and_wrist_content(head, wrist)
        return [self.astra.submit("program_extra", PROGRAM_INSTRUCTIONS, content, PROGRAM_SCHEMA, effort="low")
                for _ in range(n)]

    def ledger_text(self):
        return json.dumps(self.ledger, default=_jsonable)[:6000]

    def call_repair(self, report):
        head = self.capture(HEAD_CAMERA)
        wrist = self.capture(WRIST_CAMERA)
        self.save_img("repair_head", head.rgb)
        content = self.task_parts() + [
                   text_part("Previous program: " + json.dumps(self.program)[:6000]),
                   text_part("Ledger (completed / attempted stages with local measurements): " + self.ledger_text()),
                   text_part("Failure report: " + json.dumps(report, default=_jsonable)[:5000])]
        content += self.head_and_wrist_content(head, wrist)
        t = time.time()
        data, rec = self.astra.call("repair", REPAIR_INSTRUCTIONS, content, REPAIR_SCHEMA, effort="high")
        self._tick("model_wait", t)
        self.log("repair", diagnosis=data.get("diagnosis"), record=rec, program=data)
        self._repair_wrist = wrist
        return data, head

    def call_verify(self):
        head = self.capture(HEAD_CAMERA)
        wrist = self.capture(WRIST_CAMERA)
        self.save_img("verify_head", head.rgb)
        self.save_img("verify_wrist", wrist.rgb)
        content = self.task_parts() + [
                   text_part("Executed program: " + json.dumps(self.program)[:6000]),
                   text_part("Ledger: " + self.ledger_text())]
        content += self.head_and_wrist_content(head, wrist)
        t = time.time()
        try:
            data, rec = self.astra.call("verify", VERIFY_INSTRUCTIONS, content, VERIFY_SCHEMA)
        except AstraError as e:
            self._tick("model_wait", t)
            self.log("verify_error", error=repr(e))
            return {"complete": None, "problem": "VERIFY unavailable: " + repr(e)[:120], "hint": ""}
        self._tick("model_wait", t)
        self.log("verify", result=data, record=rec)
        return data

    # ================================================================ grounding
    def ground(self, prog, head, frames, keep_ids=(), s1_wrist=None):
        """Create / refresh object models for the program's objects from head + wrist frames. An object System 1
        pointed at in its wrist image (image 2: hidden from the head camera) is segmented there (s1_wrist: the wrist
        frame System 1 saw)."""
        t = time.time()
        objs = prog["objects"]
        order = sorted(objs, key=lambda o: 1 if o.get("part_of") else 0)
        failures = []
        bound = set()           # tracked objects already claimed by an id of this program
        for o in order:
            sid = o["id"]
            uv = (o["x"], o["y"])
            src = s1_wrist if (o.get("image") == 2 and s1_wrist is not None) else head
            others = [f for f in [head] + list(frames) if f is not src]     # views to add to the model
            oid = self.id_map.get(sid)
            if oid is None:
                h_oid = self._held_at_pixel(src, uv)
                if h_oid is not None and h_oid not in bound:
                    # a new name for the object in the hand: same physical object, same model and payload
                    self.id_map[sid] = h_oid
                    bound.add(h_oid)
                    self.log("ground_alias", object=sid, oid=h_oid, why="the held object is under the pixel")
                    continue
            if oid is not None and oid in self.objects and self.objects[oid].held:
                bound.add(oid)
                continue
            hit = self.mask_for_point(src, o["prompts"], uv)
            how = "sam"
            if hit is None and self._is_table_pixel(src, uv):
                # no object segment there and the pixel lies on the table plane: the named thing is the table top
                # (region growing from a table pixel would snap to the edge of some neighbouring object)
                oid = oid if oid is not None else self._new_oid()
                self.id_map[sid] = oid
                bound.add(oid)
                self.objects[oid] = self._table_model(oid, src, uv, o)
                self.log("grounded", object=sid, oid=oid, how="table_plane", summary=self.objects[oid].summary())
                continue
            if hit is None:
                m = grow_region(src, (uv[1], uv[0]), self.table_z)
                how = "grow"
                if m.sum() < 15:
                    failures.append({"object": sid, "name": o["name"], "reason": "no segment at the given pixel"})
                    self.log("ground_fail", object=sid)
                    continue
                hit = (m, 0.0, None)
            mask = hit[0]
            # one object, one model: a segment's points on another tracked object are not this object's - except its
            # own parent fixture (whose model still holds the part's surface until the parts are cut out below)
            parent_oid = self.id_map.get(o.get("part_of")) if o.get("part_of") else None
            pts = mask_points(src, mask)
            if parent_oid is None:
                pts = self._without_others(pts, oid)
            old = self.objects.get(oid) if oid is not None else None
            if old is not None and not old.held and not o.get("part_of") and len(old.points) >= 30 and \
                    len(pts) >= 10 and float(np.mean(old.surface_distance(pts) < 0.015)) >= 0.7 and \
                    float(np.ptp(pts, axis=0).max()) < 0.8 * float(np.ptp(old.points, axis=0).max()):
                # System 1 named a part of an object already tracked (its handle, its rim): the object stays whole and
                # the part is where its grasp is preferred
                old.add_points(pts)
                old.prefer = (pts.mean(0), max(0.02, 0.5 * float(np.ptp(pts, axis=0).max())))
                self.id_map[sid] = oid
                bound.add(oid)
                self.log("ground_part_of_tracked", object=sid, oid=oid, prefer_center=old.prefer[0],
                         prefer_radius=old.prefer[1], summary=old.summary())
                continue
            if oid is None:
                oid = self._same_as_tracked(pts, exclude=bound)
                if oid is not None:
                    self.log("ground_alias", object=sid, oid=oid, why="same surface as a tracked object")
                else:
                    oid = self._new_oid()
            self.id_map[sid] = oid
            bound.add(oid)
            if oid in self.objects and self.objects[oid].held:
                continue
            support = self._support_under(pts)
            model = ObjectModel(oid, o["name"], o["prompts"][0], o["role"], support, pixel_hint=uv)
            model.prompts = list(o["prompts"])
            model.add_points(pts)
            model.views.append(("head" if src is head else src.name, how, float(hit[1])))
            if len(model.points) < 10:
                failures.append({"object": sid, "name": o["name"], "reason": "segment has no valid depth"})
                continue
            for fr in others:
                mt = self.mask_matching(fr, o["prompts"], model)
                if mt is None:
                    continue
                extra = self.filter_points(mask_points(fr, mt[0]), model)
                if parent_oid is None:
                    extra = self._without_others(extra, oid)
                model.add_points(extra)
                model.views.append((fr.name, "sam", float(mt[3])))
            if o.get("part_of"):
                model.part_of = self.id_map.get(o["part_of"])
            self.objects[oid] = model
            self._forget_free(model.points)
            self.log("grounded", object=sid, oid=oid, how=how, prompt=hit[2], score=hit[1], summary=model.summary(),
                     views=model.views, image=2 if src is not head else 1)
        # parts: remove their points from the parent model
        for m in self.objects.values():
            if m.part_of and m.part_of in self.objects:
                parent = self.objects[m.part_of]
                d = m.surface_distance(parent.points)
                parent.set_points(parent.points[d > 0.006])
        self._tick("ground", t)
        return failures

    def _new_oid(self):
        return max(list(self.id_map.values()) + list(self.objects.keys()) + [0]) + 1

    def _held_at_pixel(self, frame, uv):
        """Id of the held object if the image pixel shows it (projected payload model), else None."""
        if self.held is None or self.checker.attached is None:
            return None
        _, att = self.checker.world_points(self.rb.q()[None, :7])
        m = ndimage.binary_fill_holes(project_mask(frame, att[0], radius_px=6))
        H, W = m.shape
        u, v = int(np.clip(uv[0], 0, W - 1)), int(np.clip(uv[1], 0, H - 1))
        return self.held if m[v, u] else None

    def _same_as_tracked(self, pts, exclude=()):
        """A tracked object whose model this new segment re-observes (identity by 3D overlap): most of the
        segment lies on the model and the segment covers a good part of the model."""
        if len(pts) < 20:
            return None
        tree = cKDTree(pts)
        best, best_s = None, 0.0
        for oid, m in self.objects.items():
            if oid in exclude or m.held or len(m.points) == 0 or getattr(m, "virtual", False):
                continue
            a = float(np.mean(m.surface_distance(pts) < 0.01))
            b = float(np.mean(tree.query(m.points)[0] < 0.01))
            if a >= 0.6 and b >= 0.3 and a + b > best_s:
                best, best_s = oid, a + b
        return best

    def _is_table_pixel(self, frame, uv):
        """The pixel and its surroundings (3 x 3 samples, 8 px apart) show the bare table plane."""
        zs = np.array([frame.pixel_ray_point(uv[0] + du, uv[1] + dv)[2] for du in (-8, 0, 8) for dv in (-8, 0, 8)])
        zs = zs[np.isfinite(zs)]
        return bool(len(zs) >= 7 and np.mean(np.abs(zs - self.visual_table_z) < 0.006) >= 0.8)

    def _table_model(self, oid, frame, uv, o):
        """The table top itself as a named support (e.g. to set an object down): a flat 30 x 30 cm patch."""
        p = frame.pixel_ray_point(uv[0], uv[1])
        g = np.stack(np.meshgrid(np.arange(-0.15, 0.1501, 0.01), np.arange(-0.15, 0.1501, 0.01)), -1).reshape(-1, 2)
        m = ObjectModel(oid, o["name"], o["prompts"][0], o["role"], self.table_z - 0.01, pixel_hint=tuple(uv))
        m.prompts = list(o["prompts"])
        m.set_points(np.c_[g + p[:2], np.full(len(g), self.table_z)])
        m.virtual = True
        m.views.append(("head", "table_plane", 0.0))
        return m

    def update_support_plane(self, frames=None):
        """Support height = the lowest base among everything standing on the table (lowest observed point of
        each connected cluster of above-table surface points, all views), never below the visual table.

        Nothing rests below its own lowest point, so each base bounds the contact surface from above
        (render and collision tops may differ); a hidden base (e.g. a large fixture seen only from
        above) can only make one cluster's base high, so the minimum over all clusters is used.
        """
        tz = self.visual_table_z
        base = self.rm.root_p
        pts = []
        for fr in (frames if frames is not None else self.frames):
            P, _ = fr.backproject(mask=~ndimage.binary_dilation(fr.robot_mask, iterations=2), stride=2)
            sel = (P[:, 2] > tz + 0.0015) & (P[:, 2] < tz + 0.5) & (P[:, 0] > base[0] + 0.15) & \
                (P[:, 0] < base[0] + 1.15) & (np.abs(P[:, 1] - base[1]) < 0.75)
            pts.append(P[sel])
        pts = voxel_down(np.concatenate(pts), 0.003) if pts else np.zeros((0, 3))
        if len(pts) < 50:
            return
        ij = np.floor((pts[:, :2] - pts[:, :2].min(0)) / 0.01).astype(int)
        grid = np.zeros(ij.max(0) + 1, bool)
        grid[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        pl = lab[ij[:, 0], ij[:, 1]]
        bases = []
        for k in range(1, n + 1):
            z = pts[pl == k, 2]
            if len(z) < 40 or np.ptp(z) < 0.008:
                continue
            b = float(np.percentile(z, 0.5))
            if b <= tz + 0.03:
                bases.append(round(b, 4))
        if not bases:
            return
        # the lowest base that a second cluster corroborates (within 3 mm): one spurious low cluster - relief of
        # the table itself, a thin mat - must not pull the plane below where objects actually rest
        b = np.sort(np.array(bases))
        ok = [i for i in range(len(b)) if np.sum(np.abs(b - b[i]) <= 0.003) >= 2]
        plane = max(tz, float(b[ok[0]] if ok else b[0] if len(b) == 1 else b[1]))
        self.log("support_plane", visual=tz, resting=plane, n=len(bases), bases=sorted(bases)[:6])
        self.table_z_visual = tz
        if abs(plane - self.table_z) > 0.0005:
            self.table_z = plane
            self.scene.table_z = plane
            for m in self.objects.values():
                if abs(m.support_z - tz) < 0.002:
                    m.support_z = plane
                    m._finalize()

    def _support_under(self, pts):
        if len(pts) == 0:
            return self.table_z
        lo = pts[:, :2].min(0) - 0.03
        hi = pts[:, :2].max(0) + 0.03
        return float(self.table_z)

    def rebuild_map(self, frames):
        """Re-integrate all frames with object labels (masks from projected object models)."""
        t = time.time()
        base = self.scene
        self.scene = SceneMap(base.lo, base.hi, base.res, table_z=self.table_z)
        self.scene.visual_table_z = getattr(base, "visual_table_z", None)
        self.scene.free_above = self.visual_table_z + 0.30
        self.scene.robot_zone = base.robot_zone
        self.scene.out_of_view_free_z = base.out_of_view_free_z
        self.checker.scene = self.scene
        for fr in frames:
            lab = np.zeros(fr.depth.shape, np.int16)
            for oid, m in sorted(self.objects.items(), key=lambda kv: 1 if kv[1].part_of else 0):
                if m.held or len(m.points) == 0 or getattr(m, "virtual", False):
                    continue
                pm = project_mask(fr, m.points, radius_px=2)
                lab[pm] = oid
            self.scene.integrate(fr, lab, extra_mask=fr.meta.get("held_mask"), robot_shadow_free=self.shadow_rule(fr),
                                 held_box=fr.meta.get("held_box"))
        self._remember_free()
        hollow = self._hollow_oids()
        for m in self.objects.values():
            if not m.held and not getattr(m, "virtual", False):
                # inferred fill never overwrites seen-free; a container's unseen interior is not inferred solid
                self.scene.insert_object(m, seen_free=self.free_mem, hollow=m.oid in hollow)
        if self.swept:
            self.carve_robot(np.array(self.swept), hulls=False)   # everywhere the arm has been holds nothing else
        self.carve_robot()
        self._clear_held_ghost()
        self._recall_free()
        self.scene.finalize()
        for m in self.objects.values():
            if not m.held and not getattr(m, "virtual", False):
                self.scene.propagate_label(m)
        self._tick("map", t)
        self._rec("on_map", "rebuild")

    def _hollow_oids(self):
        """Objects something is put "in" by the program (containers): their unseen interior is unknown, not solid."""
        out = set()
        for st in (self.program or {}).get("stages", []) or []:
            if st.get("op") == "transfer" and st.get("relation") == "in":
                oid = self.id_map.get(st.get("reference"))
                if oid is not None:
                    out.add(oid)
        return out

    def refresh_object(self, model, frames, prompts):
        """Replace an object's points with fresh observations matched to its current model
        (projected-mask IoU first; else the detection whose 3D centre is nearest the model). Points on another tracked
        object are not this object's (a destination never absorbs what stands on it)."""
        new, pts = [], np.zeros((0, 3))
        c0 = model.points.mean(0) if len(model.points) else None
        for fr in frames:
            mt = self.mask_matching(fr, prompts, model, min_iou=0.1)
            if mt is not None:
                new.append(self.filter_points(mask_points(fr, mt[0]), model, pad=0.06))
                continue
            if c0 is None:
                continue
            best, best_d = None, 0.06
            for m, s, p in self.detect(fr, prompts):
                pts = mask_points(fr, m)
                if len(pts) < 20:
                    continue
                d = float(np.linalg.norm(pts.mean(0) - c0))
                if d < best_d:
                    best, best_d = pts, d
            if best is not None:
                new.append(best)
        if new and sum(len(n) for n in new) > 30:
            pts = self._without_others(np.concatenate(new), model.oid)     # one object, one model
        if new and len(pts) > 30:
            bottom = float(np.percentile(pts[:, 2], 2))
            if bottom < model.support_z - 0.005:
                # the object stands lower than recorded (a support height planned too high): measured wins, else the
                # support filter would throw the object's own points away
                self.log("support_measured", object=model.name, planned=model.support_z, measured=bottom, by="view")
                model.support_z = bottom
            old = model.points
            model.points = np.zeros((0, 3))
            model.add_points(pts)
            if len(model.points) >= 20:
                self._forget_free(model.points)
            if len(model.points) < 20:
                # the fresh points fell below the model's support height (planned too high): keep the old model
                model.points = old
                model._finalize()
                self.log("refresh_rejected", object=model.name, fresh=int(len(pts)), support_z=model.support_z)
                return False
            return True
        return False

    # ================================================================ stages
    def run_stage_guarded(self, st):
        """run_stage, with any unexpected error turned into StageFailure('internal_error') after re-syncing the state from
        the sensors (holding or not from the finger opening; fresh views into the map), so REPAIR can carry on.
        StageFailure and AstraError pass unchanged; KeyboardInterrupt and the evaluator's stop are not Exceptions."""
        self.cur_stage = st
        if self.in_release and st.get("op") == "transfer" and st.get("relation") == "in" and \
                st.get("release") != self.in_release:
            self.log("release_override", stage=st.get("id"), program=st.get("release"), used=self.in_release)
            st["release"] = self.in_release
        try:
            return self.run_stage(st)
        except (StageFailure, AstraError):
            raise
        except Exception as e:
            self.log("internal_error", stage=st.get("id"), error=repr(e), tb=traceback.format_exc()[-2500:])
            try:
                if self.held is not None and self.rb.finger_opening() < 0.002:
                    self._release_attachment(self.objects[self.held], dropped=True)
                self.reobserve()
            except Exception as e2:
                self.log("resync_error", error=repr(e2))
            raise StageFailure("internal_error", error=repr(e)[:300])

    def run_stage(self, st):
        if st["op"] == "transfer":
            return self.do_transfer(st)
        if st["op"] == "actuate":
            return self.do_actuate(st)
        raise StageFailure("unsupported_op", op=st["op"])

    def obj(self, sid):
        oid = self.id_map.get(sid)
        if oid is None or oid not in self.objects:
            raise StageFailure("object_not_grounded", object=sid)
        return self.objects[oid]

    # ---------------------------------------------------------------- transfer
    def do_transfer(self, st):
        obj = self.obj(st["object"])
        dest = self.obj(st["reference"])
        place_pt = None
        if st.get("place_x") is not None and self.head0 is not None:
            place_pt = self.head0.pixel_ray_point(st["place_x"], st["place_y"])
        result = {"stage": st["id"], "op": "transfer", "object": obj.name, "reference": dest.name,
                  "relation": st["relation"], "attempts": []}
        if self.held is not None and self.held != obj.oid:
            self.park_held()
        for attempt in range(3):
            try:
                if self.held is None or self.held != obj.oid:
                    want = self.lookahead_grasp(obj, dest, st, place_pt) \
                        if self.placement == "unified" and self.motion != "cp5" and attempt == 0 else None
                    self.grasp_object(obj, st, result, want=want)
                self.place_object(obj, dest, st, place_pt, result)
                return result
            except StageFailure as f:
                if f.reason != "dropped_in_transit" or attempt == 2:
                    raise
                _, p_drop = self.rb.eef_pose()
                self._release_attachment(obj, dropped=True)
                result["attempts"].append({"phase": "transit", "status": "dropped", "where": p_drop})
                self.log("dropped", object=obj.name, where=p_drop)
                if not self.relocate_object(obj, p_drop):
                    raise StageFailure("object_lost", object=obj.name)
        raise StageFailure("transfer_failed", object=obj.name)

    def relocate_object(self, obj, hint):
        """Find a dropped / displaced object again: head view instances nearest to a 3D hint."""
        Rs, ps = self.rb.eef_pose()
        self.move_line(Rs, ps + np.array([0, 0, 0.08]), grip=-1.0, speed=0.15, finger=0.04)
        head = self.capture(HEAD_CAMERA)
        self.frames.append(head)
        others = [project_mask(head, m.points, 2) for m in self.objects.values() if m.oid != obj.oid and not m.held]
        best, best_d = None, 0.35
        for m, s, p in self.detect(head, [obj.category, obj.name]):
            if any((m & o).sum() > 0.3 * m.sum() for o in others):
                continue
            pts = mask_points(head, m)
            if len(pts) < 20:
                continue
            d = np.linalg.norm(pts.mean(0)[:2] - np.asarray(hint)[:2])
            if d < best_d:
                best, best_d = pts, d
        if best is None:
            return False
        obj.points = np.zeros((0, 3))
        obj.support_z = self.table_z
        obj.add_points(best)
        self._forget_free(obj.points)
        fr = self.look_from_above(np.r_[obj.footprint_center(), obj.top_z], height=0.26)
        if fr is not None:
            self.refresh_object(obj, [fr, head], [obj.category, obj.name])
        self.rebuild_map(self.frames[-8:])
        self.log("relocated", object=obj.name, center=obj.footprint_center(), top=obj.top_z)
        return True

    # closed width minus predicted contact width (m) that still counts as the pads on the object where planned: the
    # full test held 151 of 158 closes within -6..+4 mm and 14 of 16 within +4..+10 mm, 4 of 15 when more than 6 mm
    # narrower and 3 of 7 when more than 10 mm wider
    CLOSE_WINDOW = (-0.006, 0.010)

    def grasp_object(self, obj, st, result, want=None):
        """Grasp the object (top-down first, from the side when no top-down grasp fits or has an arm posture);
        want: a grasp planned earlier to take (a regrasp at a staging spot), see plan_grasp."""
        failed = []            # (point, radius): contact regions that failed, so the next attempt differs there
        for attempt in range(3):
            # a side no camera has seen yet is looked at first; the top look below then brings the hand back
            # above the object (top-down) and integrates both views
            side = self.look_unseen_side(obj) if attempt == 0 else False
            # look closely from above (fills unseen sides, refreshes the map)
            fr = self.look_from_above(np.r_[obj.footprint_center(), obj.top_z], height=0.26)
            if fr is None and side:
                self.rebuild_map(self.frames[-8:])
            if fr is not None:
                if attempt > 0:
                    head = self.capture(HEAD_CAMERA)
                    self.frames.append(head)
                    self.refresh_object(obj, [fr, head], [obj.category, obj.name])
                for m in self.objects.values():
                    if not m.held and not getattr(m, "virtual", False) and (m.oid != obj.oid or attempt == 0):
                        mt = self.mask_matching(fr, [m.category] + ([m.name] if m.name != m.category else []), m)
                        if mt is not None:
                            m.add_points(self.filter_points(mask_points(fr, mt[0]), m))
                            if m.oid == obj.oid:
                                m.views.append((fr.name, "top", float(mt[3])))
                self.rebuild_map(self.frames[-8:])
            # a wanted grasp from above needs only the top-down family (no side generation)
            fam = "top" if want is None or float(np.asarray(want[0])[2, 2]) < -0.8 else "all"
            plans = self.plan_grasp(obj, failed, families=fam, want=want)
            if not plans and want is None and self.motion != "cp5":
                # no top-down grasp fits or has a posture (an object near the base, a handle on the side): any
                # other approach direction on the same model (CONTACT_STAGE_PLAN 4.5)
                plans = self.plan_grasp(obj, failed, families="side")
                self.log("side_grasps", found=len(plans))
            if not plans and want is not None:
                want = None                       # the planned regrasp is not there any more: any grasp
                plans = self.plan_grasp(obj, failed, families="all")
            if not plans:
                why = self._grasp_why(obj, result)
                result["attempts"].append({"phase": "grasp", "status": "no_feasible_grasp", "cause": why["cause"]})
                if attempt >= 1:
                    raise StageFailure("no_feasible_grasp", object=obj.name, size=obj.summary()["size"], **why)
                continue
            g, q_pre, q_grasp, info = plans[0]
            f_open = g.info["finger"]
            win = lambda h: (2 * (f_open + manip.PAD_OFFSET) - h.width) / 2
            used = [plans[0]] + [pl for pl in plans[1:] if pl[0].info.get("gap") == g.info.get("gap")]
            alts = [{"R": h.R, "p": h.p, "q": qg, "approach": h.R[:, 2], "window": win(h)} for h, _, qg, _ in used[1:]]

            def open_hand():
                if g.info.get("gap") == "full":
                    # r1: binary open until the measured opening is stable, no width-targeted pre-shape
                    return self.ex.gripper(-1.0, min_steps=8)
                return self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))   # pre-shape where full open does not fit
            self.log("open", opening=open_hand(), gap=g.info.get("gap"))
            # the hand-off: the planned grasps (and their 180-degree twins) as one set; TRACE takes the first it
            # certifies; lateral window = finger clearance per side at this opening
            end = self.reach(g.R, g.p, q_grasp, g.R[:, 2], obj.oid, win(g), "to_pregrasp", 0.0, f_open, q_pre, alts=alts)
            if end is None:
                for h, _, _, _ in used:
                    failed.append((h.p, 0.01))
                result["attempts"].append({"phase": "grasp", "status": "transit_failed", "move": self.last_move})
                if self.last_move.get("status") == "contact":
                    self.reobserve()
                continue
            g = used[int(getattr(end, "k", 0))][0]
            Rg, pg = np.asarray(end.R, float), np.asarray(end.p, float)    # the twin's orientation when it was taken
            # descend with a force guard
            def guard(robot, F, F0, k):
                return "early_contact" if np.linalg.norm(F - F0) > 12.0 and k > 2 else None
            res = self.move_line(Rg, pg, grip=0.0, speed=0.06, guard=self._guard("early_contact", guard),
                                 finger=f_open)
            if res is None or res.status not in ("done",):
                self.log("descend", status=None if res is None else res.status)
            short = float((pg - self.rb.eef_pose()[1]) @ Rg[:, 2])
            tip_below_top = g.info.get("depth", 0.0) + manip.TIP_Z - short     # fingertips below the object's top
            if short > 0.006 and tip_below_top < 0.008:
                # stopped with the fingertips at the top: a finger is on the object, not beside it - back off and
                # look again (as r1: the spot is not excluded; the next look corrects the object model)
                self.retreat(Rg, pg - Rg[:, 2] * 0.10, grip=0.0, speed=0.12, finger=f_open, tag="back_off")
                result["attempts"].append({"phase": "grasp", "status": "blocked_descent", "short_m": round(short, 4)})
                self.log("blocked_descent", short=short, tip_below_top=tip_below_top)
                continue
            if short > 0.006:
                # as r1: the descent ended on contact with the pads already down the sides - close where it stopped
                self.log("close_at_contact", short=short, tip_below_top=tip_below_top)
            opening = self.ex.gripper(1.0, min_steps=12)               # r1: binary close until the opening is stable
            self.log("close", opening=opening, expect=g.width)
            if opening < 0.002:
                open_hand()
                self.retreat(Rg, pg - Rg[:, 2] * 0.10, grip=0.0, speed=0.15, finger=f_open, tag="back_off")
                failed.append((g.p, 0.015, g.info.get("yaw", 0.0)))   # nothing between the pads there: differ next
                result["attempts"].append({"phase": "grasp", "status": "empty_close", "opening": opening})
                continue
            dw = opening - g.width
            if not (self.CLOSE_WINDOW[0] <= dw <= self.CLOSE_WINDOW[1]):
                # the pads did not meet the object where planned (slid off, pushed it, or closed on something else):
                # no lift; open, back off, and the next attempt grasps another region
                open_hand()
                self.retreat(Rg, pg - Rg[:, 2] * 0.10, grip=0.0, speed=0.15, finger=f_open, tag="back_off")
                failed.append((g.p, 0.015, g.info.get("yaw", 0.0)))
                result["attempts"].append({"phase": "grasp", "status": "unexpected_width", "opening": round(opening, 4),
                                           "expected": round(g.width, 4)})
                self.log("grasp_width_mismatch", opening=opening, expected=g.width, window=self.CLOSE_WINDOW)
                continue
            # attach payload (re-centred between the pads: closing fingers centre the object) and lift;
            # the collision model is closed (hull of the observed surface down to its base), the object model
            # after release stays the observed points
            Rs, ps = self.rb.eef_pose()
            full, pinfo = self.payload_model(obj)
            local = (obj.points - ps) @ Rs
            local_full = (full - ps) @ Rs
            band = (np.abs(local[:, 1]) <= manip.FINGER_Y) & (local[:, 2] >= manip.PAD_Z[0] - 0.01) & \
                (local[:, 2] <= manip.TIP_Z) & (np.abs(local[:, 0]) <= g.width / 2 + manip.APPROACH_GAP + 0.004)
            if band.sum() > 5:
                shift = 0.5 * (local[band, 0].min() + local[band, 0].max())
                local[:, 0] -= shift
                local_full[:, 0] -= shift
                self.log("payload_centering", shift=shift)
            self.checker.attached = Attached(obj.oid, local_full, 0.0, model_site=local,
                                             up_site=Rs.T @ np.array([0.0, 0.0, 1.0]))
            self.log("payload", object=obj.name, **pinfo)
            self.held, self.held_width = obj.oid, opening
            obj.held = True
            self.scene.clear_label(obj.oid)
            lift_to = ps + np.array([0, 0, 0.10])
            res = self.retreat(Rs, lift_to, grip=0.0, speed=0.12, tag="lift")
            opening2 = self.rb.finger_opening()
            self.held_width = opening2                       # settled grip after the lift = drop reference
            if res is not None and res.status == "dropped" or opening2 < 0.002:
                self._release_attachment(obj, dropped=True)
                failed.append((g.p, 0.015, g.info.get("yaw", 0.0)))   # it slipped from this grasp: differ next
                result["attempts"].append({"phase": "lift", "status": "dropped", "opening": opening2})
                continue
            if (res is None or res.status not in ("done", "dropped")) and \
                    np.linalg.norm(self.rb.eef_pose()[1] - ps) < 0.01:
                # the arm cannot leave the grasp pose (no posture along the lift, no certified segment): let go where
                # the object still stands, so nothing is carried from a pose the next moves cannot start from; the
                # next attempt grasps elsewhere
                self.log("cannot_leave_pose", status=None if res is None else res.status, tag="lift")
                self.ex.gripper(-1.0, min_steps=8)
                self._release_attachment(obj)
                self.scene.insert_object(obj)
                failed.append((g.p, 0.02))
                result["attempts"].append({"phase": "lift", "status": "cannot_leave_pose"})
                continue
            if not hasattr(self, "_grasp_origin"):
                self._grasp_origin = {}
            self._grasp_origin[obj.oid] = np.asarray(obj.footprint_center(), float)   # free and reachable now
            # the volume it stood in is empty now: what the cameras showed there outside its mask (a handle, a rim
            # edge) is the object's leftover, not an obstacle; kept clear across map rebuilds (_clear_vacated)
            lo, hi = obj.points.min(0) - 0.015, obj.points.max(0) + 0.015
            lo[2] = max(float(lo[2]), self.table_z + 0.005)
            self._vacated = getattr(self, "_vacated", []) + [(obj.oid, lo, hi)]
            self._clear_vacated()
            result["attempts"].append({"phase": "grasp", "status": "held", "opening": round(opening2, 4),
                                       "grasp_width_pred": round(g.width, 4), "yaw": g.info.get("yaw")})
            self.log("held", object=obj.name, opening=opening2)
            return True
        raise StageFailure("grasp_failed", object=obj.name, attempts=result["attempts"][-3:],
                           **self._grasp_why(obj, result))

    def _grasp_why(self, obj, result):
        """Typed cause of a failed grasp stage, with the levers that match it (System 1 acts on these, not on the
        object's size)."""
        d = getattr(self, "_grasp_diag", None) or {}
        tried = [a for a in result.get("attempts", []) if a.get("phase") in ("grasp", "lift")]
        last = tried[-1] if tried else {}
        model = {"points": int(len(obj.points)), "size_m": obj.summary()["size"], "views": [v[0] for v in obj.views]}
        if last.get("status") == "transit_failed":
            why = (last.get("move") or {}).get("why") or {}
            return {"cause": "hand_could_not_get_to_the_grasp", "detail": why, "model": model,
                    "levers": "the way to the grasp is blocked or the arm is at its limits there: another object order, "
                              "or re-point the object if its model includes a neighbour"}
        if last.get("status") in ("dropped", "unexpected_width", "empty_close"):
            return {"cause": {"dropped": "slipped_at_lift", "unexpected_width": "pads_missed_the_object",
                              "empty_close": "closed_on_nothing"}[last["status"]], "model": model,
                    "levers": "the object's model may be wrong (another object, or only part of it): re-point it, in "
                              "image 2 if the head camera does not show it"}
        if d.get("candidates", 0) == 0:
            cause, lever = "no_grasp_on_the_model", ("the model may be partial or not the object: re-point it (image 2 "
                                                     "if the head camera does not show it)")
        elif d.get("hand_fit", 0) == 0:
            cause, lever = "no_room_for_the_open_hand", (f"the open hand meets {d.get('blocked_by')}: another order, or "
                                                         "re-point if the model includes a neighbour")
        else:
            cause, lever = "no_arm_posture", (f"grasps exist on the model, none with an arm posture "
                                              f"({d.get('reach_m')} m from the shoulder)")
        return {"cause": cause, "levers": lever, "model": model, "grasps_on_model": d.get("candidates"),
                "hand_fits": d.get("hand_fit")}

    def _release_attachment(self, obj, dropped=False):
        self.checker.attached = None
        self.held = None
        obj.held = False
        if dropped:
            self.ex.gripper(-1.0)

    def hand_height(self, cands, top_z):
        """How far each grasp's open hand rises above top_z (m): a put-down under something (a roof, a shelf) needs
        the hand low beside the object, not above it."""
        if not cands:
            return np.zeros(0)
        L, _, _ = self._hand_local(manip.FULL_JOINT)
        return np.array([float((g.p + L @ g.R.T)[:, 2].max()) - float(top_z) for g in cands])

    def plan_grasp(self, obj, failed, k_max=3, budget_s=12.0, families="top", want=None, level_first=False):
        """See _plan_grasp. Unsegmented surface hugging the object (a handle or a rim edge outside its mask) is the
        object itself during the checks, not an obstacle next to it."""
        fn, self.checker.free_near = self.checker.free_near, [(np.asarray(obj.points, float), 0.01, True)]
        try:
            return self._plan_grasp(obj, failed, k_max, budget_s, families, want, level_first)
        finally:
            self.checker.free_near = fn

    def _plan_grasp(self, obj, failed, k_max=3, budget_s=12.0, families="top", want=None, level_first=False):
        """Up to k_max distinct feasible grasps [(g, q_pre, q_grasp, info)], best first (the hand-off is a set);
        [] when none. failed: (point, radius) regions excluded by earlier attempts. self._grasp_diag says why when
        nothing is found.
        families: "top" (approach within 35 degrees of vertical), "side" (any other direction, CONTACT_STAGE_PLAN 4.5)
        or "all". want: (R, p) of a grasp planned earlier (a regrasp): only grasps near it, nearest first.
        level_first: the grasps whose hand rises least above the object first (the hand low beside it, for a
        put-down under something)."""
        t = time.time()
        R0, _ = self.rb.eef_pose()
        cur_yaw = np.arctan2(R0[1, 0], R0[0, 0])
        cands = grasp_candidates(obj, obj.support_z, cur_yaw=cur_yaw) if families in ("top", "all") else []
        if families in ("side", "all"):
            cands = sorted(cands + manip.side_grasp_candidates(obj.points, obj.name, manip.side_directions(), cur_R=R0),
                           key=lambda c: -c.score)
        if level_first and cands:
            cands = [c for c in cands if float(c.R[2, 2]) <= 0.25]       # not from below (for a put-down)
            hh = self.hand_height(cands, obj.top_z)
            order_ = np.lexsort((-np.array([c.score for c in cands]), np.round(hh, 2)))
            cands = [cands[i] for i in order_]
        if want is not None:
            Rw, pw = np.asarray(want[0], float), np.asarray(want[1], float)
            near = lambda c: (float(np.degrees(np.arccos(np.clip(c.R[:, 2] @ Rw[:, 2], -1, 1)))),
                              float(np.linalg.norm(c.p - pw)))
            cands = [c for c in cands if near(c)[0] <= 30.0 and near(c)[1] <= 0.03]
            cands.sort(key=lambda c: near(c)[0] / 30.0 + near(c)[1] / 0.03)
        pref = getattr(obj, "prefer", None)
        if pref is not None and cands and want is None:
            # a part System 1 named on this object (handle, rim): its region is preferred, the object stays whole
            for c in cands:
                if np.linalg.norm(c.p - pref[0]) <= pref[1]:
                    c.score += 2.0
            cands.sort(key=lambda c: -c.score)
        cands = [c for c in cands if all(np.linalg.norm(c.p - f[0]) > f[1] or
                                         (len(f) > 2 and abs(_wrap_pi(c.info.get("yaw", 0.0) - f[2])) > 0.5)
                                         for f in failed)]
        self.log("grasp_cands", n=len(cands), top=[{"p": c.p, "w": c.width, "s": c.score, **c.info}
                                                  for c in cands[:5]])
        q_now = self.rb.q()[:7]
        tried = 0
        # rigid-hand prefilter (no IK): palm/fingers (pre-shaped) at the grasp pose and 4 cm before it
        hands = {}
        blocked = {}
        def hand_fits(g, f):
            f = round(f, 3)
            if f not in hands:
                hands[f] = self._hand_local(f)
            L, grp, rad = hands[f]
            for back in (0.0, 0.04):
                Wh = g.p - g.R[:, 2] * back + L @ g.R.T
                # observed geometry (user decision D2: unseen space is looked at by the transit, then crossed slowly)
                c = self.checker.point_clearance(Wh, (obj.oid,), unknown=False) - rad
                if c[grp == 1].min() < NEAR["palm"] or c[grp == 2].min() < NEAR["finger"]:
                    if not blocked:
                        blocked["pts"] = Wh[grp >= 1]
                    return False
                zc = Wh[:, 2] - self.table_z - 0.5 * rad
                if zc[grp == 1].min() < GRASP_NEAR["plane_palm"] or zc[grp == 2].min() < GRASP_NEAR["plane_finger"]:
                    return False
            # the object as a solid seen along the grasp's own direction (from above for the top-down family)
            d = self.object_clearance(obj.points, g.p + L @ g.R.T, g.info.get("side"),
                                      self.__dict__.setdefault("_side_model_cache", {})) - rad \
                if g.info.get("side") is not None else obj.signed_clearance(g.p + L @ g.R.T) - rad
            return d[grp == 1].min() >= 0.006 and d[grp == 2].min() >= 0.0015
        def hand_ok(g):
            # fully open hand (binary open, as r1) wherever it fits; otherwise the widest pre-shape the
            # surroundings allow (observed slabs under-estimate flared / bevelled sides)
            if hand_fits(g, manip.FULL_JOINT):
                g.info["finger"], g.info["gap"] = manip.FULL_JOINT, "full"
                return True
            for gap in manip.APPROACH_GAPS:
                f = preshape_finger(g.width, gap)
                if hand_fits(g, f):
                    g.info["finger"], g.info["gap"] = f, gap
                    return True
            return False
        order, seen = [], {}
        for g in cands:
            k = (round(g.info.get("tilt", 0.0), 2), tuple(g.info.get("side") or ()))
            if seen.get(k, 0) < 10 and hand_ok(g):
                order.append(g)
                seen[k] = seen.get(k, 0) + 1
        self.log("grasp_prefilter", kept=len(order), of=len(cands))
        found = []
        for g in order[:40]:
            if len(found) >= k_max or (found and time.time() - t > budget_s):
                break
            if any(np.linalg.norm(g.p - h[0].p) < 0.015 and abs(_wrap_pi(g.info["yaw"] - h[0].info["yaw"])) < 0.5
                   for h in found):
                continue                                   # a near-duplicate of a grasp already in the set
            f = g.info["finger"]
            # a wanted grasp (look-ahead, regrasp) is often a pitched one near the arm's limits: more starts
            sols = self.ik_clear(g.R, g.p, q_now, exclude=(obj.oid,), finger=f, n_random=48 if want is not None else 16)
            if len(sols) == 0:
                continue
            tried += 1
            lim = self.rm.limits[:7]
            jm = lambda Q: float(np.min(np.minimum(np.asarray(Q)[..., :7] - lim[:, 0], lim[:, 1] - np.asarray(Q)[..., :7])))
            for q_g in sols[:3]:
                if not self._grasp_config_ok(q_g, obj, f):
                    continue
                if jm(q_g) < GRASP_JOINT_SLACK:
                    continue                         # the grasp posture sits at a joint limit: nothing can start from it
                p_pre = g.p - g.R[:, 2] * 0.10
                line = self.line_configs(g.R, g.p, self._ik_near(g.R, p_pre, q_g), finger=f)
                if line is None:
                    continue
                good = all(self._grasp_config_ok(q, obj, f) for q in line[::2])
                if not good:
                    continue
                # a grasp is a whole motion: the lift that follows must have a posture all the way up too
                lift = self.line_configs(g.R, g.p + np.array([0.0, 0.0, 0.10]), line[-1], finger=f)
                if lift is None or jm(lift) < 0.04:
                    continue
                self.log("grasp_plan", p=g.p, yaw=g.info["yaw"], tilt=g.info.get("tilt"), side=g.info.get("side"),
                         approach=g.R[:, 2], width=g.width, tried=tried,
                         finger=f, local=g.info.get("local"), align=g.info.get("align"), depth=g.info.get("depth"),
                         rank=len(found))
                found.append((g, line[0], line[-1], g.info))
                break
            if tried >= 30:
                break
        self._tick("grasp_plan", t)
        sh = self.rm.root_p + np.array([0.0, 0.0, 0.333])
        self._grasp_diag = {"candidates": len(cands), "hand_fit": len(order), "arm_tried": tried,
                            "reach_m": round(float(np.linalg.norm(np.r_[obj.footprint_center(), obj.top_z] - sh)), 2)
                            if len(obj.points) else None,
                            "blocked_by": (self.nearest_obstacle(blocked["pts"], (obj.oid,)) or {}).get("object")
                            if blocked else None}
        return found

    def ik_clear(self, R, p, q_ref, exclude=(), finger=0.04, n_random=16):
        """IK solutions incl. null-space (elbow) variants, sorted by whole-arm clearance."""
        sols = self.rm.solve_ik(R, p, q_ref, n_random=n_random, finger=finger)
        if len(sols) == 0:
            return sols
        c0 = self.checker.clearance(sols, exclude=exclude, finger=finger)
        if ((c0["arm"] >= NEAR["arm"] + 0.004) & (c0["palm"] >= NEAR["palm"]) & (c0["finger"] >= NEAR["finger"])).any():
            return sols[np.argsort(-np.minimum(c0["arm"], c0["palm"]))]
        variants = [sols]
        offs = []
        for d3 in (-0.9, -0.5, 0.5, 0.9):
            o = np.zeros(7)
            o[2] = d3
            o[0] = -0.5 * d3
            offs.append(o)
        for o in ([0, 0.3, 0, 0.3, 0, 0, 0], [0, -0.3, 0, -0.3, 0, 0, 0]):
            offs.append(np.array(o, float))
        base = sols[0]
        seeds = np.array([base + o for o in offs])
        rests = seeds.copy()
        for sd, rs in zip(seeds, rests):
            q, ok, _, _ = self.rm.ik(R, p, sd[None], q_rest=rs, iters=120, null_gain=0.5, finger=finger)
            if ok[0]:
                variants.append(q)
        allq = np.concatenate(variants)
        keep = []
        for q in allq:
            if all(np.abs(q - k).max() > 0.04 for k in keep):
                keep.append(q)
        keep = np.array(keep)
        c = self.checker.clearance(keep, exclude=exclude, finger=finger)
        score = np.minimum(np.minimum(c["arm"], c["palm"]), c["finger"] + 0.01)
        keep = keep[np.argsort(-score)]
        if score.max() < NEAR["arm"]:
            q_opt, c_opt = self.null_clearance(keep[0], R, p, exclude, finger)
            if c_opt > score.max():
                keep = np.concatenate([q_opt[None], keep])
        return keep

    def _soft_clear(self, q, exclude, finger):
        W, _ = self.checker.world_points(q[None], finger)
        c = self.scene.clearance(W, tuple(exclude))[0] - self.checker.pr
        sel = (self.checker.pg <= 1) & (c < 0.08)
        if not sel.any():
            return 0.0, float(c.min())
        return float(-np.exp(-np.clip(c[sel], -0.05, None) / 0.012).sum()), float(c[self.checker.pg <= 1].min())

    def null_clearance(self, q, R, p, exclude=(), finger=0.04, iters=30, step=0.1):
        """Redundancy resolution: climb whole-arm clearance in the Jacobian null space at a fixed pose."""
        q = np.asarray(q, float)[:7].copy()
        f0, cmin = self._soft_clear(q, exclude, finger)
        best = (cmin, q.copy())
        for it in range(iters):
            grad = np.zeros(7)
            for j in range(7):
                dq = np.zeros(7)
                dq[j] = 0.02
                grad[j] = (self._soft_clear(q + dq, exclude, finger)[0] - f0) / 0.02
            J = self.rm.jacobian(self.rm.full_q(q[None], finger))[0]
            Nm = np.eye(7) - np.linalg.pinv(J) @ J
            d = Nm @ grad
            n = np.linalg.norm(d)
            if n < 1e-9:
                break
            qn, ok, _, _ = self.rm.ik(R, p, (q + step * d / n)[None], q_rest=q + step * d / n, iters=40,
                                      null_gain=0.0, finger=finger)
            if not ok[0]:
                step *= 0.5
                if step < 0.01:
                    break
                continue
            q = qn[0]
            f0, cmin = self._soft_clear(q, exclude, finger)
            if cmin > best[0]:
                best = (cmin, q.copy())
            if cmin > NEAR["arm"] + 0.01:
                break
        return best[1], best[0]

    def _ik_near(self, R, p, q_ref):
        q, ok, _, _ = self.rm.ik(R, p, np.asarray(q_ref)[None], q_rest=q_ref, iters=80)
        return q[0]

    def _grasp_config_ok(self, q, obj, finger=0.04):
        # observed geometry (user decision D2: unseen space is looked at by the transit, then crossed slowly)
        good, c = self.checker.ok(q[None], GRASP_NEAR, exclude=(obj.oid,), finger=finger, unknown=False)
        if not good[0]:
            return False
        f = self.checker.fine(q[None], [obj.oid], finger=finger)
        return f["palm"][0] >= 0.006 and f["finger"][0] >= 0.0015 and f["arm"][0] >= 0.01

    # ---------------------------------------------------------------- place
    def place_object(self, obj, dest, st, place_pt, result, region_override=None):
        if self.placement in ("v2", "lrl") and region_override is None:
            return self.place_object_v2(obj, dest, st, place_pt, result)
        if self.placement == "unified" and region_override is None:
            return self.place_object_unified(obj, dest, st, place_pt, result)
        rel = st["relation"]
        tol = st.get("tolerance", "normal")
        # release (S1 copies the instruction's verb): "drop" lets go over the reference, any final pose; the careful
        # placement only where no drop exists (placing also satisfies it). "place" sets it down upright and centred,
        # never a drop. Null: drop into containers ("in"), place on supports
        release = st.get("release") or ("drop" if rel == "in" else "place")
        drop = rel in ("in", "on") and region_override is None and release == "drop"
        # a drop beside something (left of / right of ...): the careful planner's spot, let go from higher, any pose
        beside_drop = rel not in ("in", "on") and release == "drop"
        failed = []
        drop_first = drop
        looked = False

        def plan_once():
            plan = self.plan_drop_in(obj, dest, place_pt, failed, rel=rel) if drop_first else None
            if plan is None:
                plan = self.plan_place(obj, dest, rel, place_pt, tol, failed, region_override=region_override,
                                       drop=beside_drop)
            if plan is None and drop and not drop_first:
                plan = self.plan_drop_in(obj, dest, place_pt, failed, rel=rel)    # no careful placement: a drop again
            return plan
        for attempt in range(3):
            plan = plan_once()
            if plan is not None and plan[0].get("unknown_near") and not looked:
                # the planned spot borders unobserved space (planned as free): look at it first, then plan again
                looked = True
                fr = self.look_from_above(np.r_[plan[5][:2], plan[3]["floor_z"]], height=0.30)
                self.log("look_unknown", at=plan[5][:2], ok=fr is not None)
                if fr is not None:
                    self.rebuild_map(self.frames[-8:])
                plan = plan_once()
            if plan is None:
                result["attempts"].append({"phase": "place", "status": "no_feasible_placement"})
                # re-observe the destination from above once, then retry
                if attempt == 0:
                    fr = self.look_from_above(np.r_[dest.footprint_center(), dest.top_z], height=0.30)
                    if fr is not None:
                        mt = self.mask_matching(fr, [dest.category, dest.name], dest)
                        if mt is not None:
                            dest.add_points(self.filter_points(mask_points(fr, mt[0]), dest))
                        self.rebuild_map(self.frames[-8:])
                    continue
                raise StageFailure("no_feasible_placement", destination=dest.name, relation=rel,
                                   held_size=self._held_size(), region=self._region_info(dest, rel, place_pt),
                                   measured=getattr(self, "_place_diag", None))
            cand, q_pre, q_place, region, R_place, p_place = plan
            if rel == "in" and region_override is None:
                # before going there: look down from higher at the planned landing spot; is it on the container?
                land = self.landing_check(dest, region, R_place, p_place)
                if land["verdict"] == "outside" and not (self.placement == "v1" and attempt == 2):
                    # (v1: on the last attempt the look no longer rules the spot out; contact and the check after
                    # the release decide)
                    failed.append((cand["xy"], 0.03))
                    result["attempts"].append({"phase": "place", "status": "landing_outside", "check": land})
                    continue
            finger = self._finger()
            ok = self.reach(R_place, p_place, q_place, p_place - self._site_pose(q_pre, finger)[1], dest.oid,
                            PLACE_WINDOW["beside" if rel not in PLACE_WINDOW else rel].get(tol, 0.01), "to_preplace",
                            0.0, finger, q_pre, twin=False) is not None
            if not ok and self.placement == "v1" and self.motion != "cp5" \
                    and (getattr(self, "last_move", None) or {}).get("status") == "no_path":
                # v1: TRACE certifies a contact approach only up to a standoff where nothing touches, and a ranked
                # candidate may be predicted to touch the destination there. Then go to the pose above the
                # destination (contact-free; higher when that one is not free either); the guarded descent below
                # covers the whole way down from it and ends on contact
                p_pre = self._site_pose(q_pre, finger)[1]
                for up in (0.0, 0.05, 0.10):
                    q_up = q_pre if up == 0.0 else self._ik_near(R_place, p_pre + np.array([0, 0, up]), q_pre)
                    ok = self.go_configs([q_up], TRANSIT, grip=0.0, finger=finger, tag="to_above_place")
                    if ok or (getattr(self, "last_move", None) or {}).get("status") != "no_path":
                        break
            if not ok:
                failed.append((cand["xy"], 0.02))
                if cand.get("drop_in"):
                    drop_first = False     # a drop that cannot be reached: the careful placement next (it satisfies it too)
                result["attempts"].append({"phase": "place", "status": "transit_failed", "move": self.last_move})
                if self.last_move.get("status") == "contact":
                    self.reobserve()           # something was hit: the map is stale where it matters
                continue
            def guard(robot, F, F0, k):
                return "support_contact" if (F - F0)[2] > 4.0 and k > 2 else None
            res = self.move_line(R_place, p_place, grip=0.0, speed=0.05, guard=self._guard("support_contact", guard))
            self.log("place_descend", status=None if res is None else res.status, drop=cand.get("drop"))
            # release: open only as far as checked (just clear of the payload), then lift straight up;
            # a drop into a container opens fully (binary, as r1)
            if cand.get("drop_in"):
                opening = self.ex.gripper(-1.0, min_steps=8)
            else:
                opening = self.ex.gripper_to(2 * (cand["f_release"] + manip.PAD_OFFSET))
            Rs, ps = self.rb.eef_pose()
            self._finish_release(obj, cand.get("rest_z", region["floor_z"]),
                                 touched=res is not None and res.status == "support_contact")
            res = self.retreat(Rs, ps + np.array([0, 0, 0.10]), grip=0.0, speed=0.08, finger=cand["f_release"],
                               tag="release_lift")
            self.ex.gripper(-1.0, max_steps=12)
            if res is None:
                self.go_configs([self._ik_near(Rs, ps + np.array([0, 0, 0.10]), self.rb.q()[:7])], NEAR,
                                grip=-1.0, finger=0.04, tag="retreat")
            if region_override is not None:
                head = self.capture(HEAD_CAMERA)
                self.frames.append(head)
                self.refresh_object(obj, [head], [obj.category, obj.name])
                self.rebuild_map(self.frames[-8:])
                return True
            # measure the result
            check = self.check_relation(obj, dest, rel, region, place_pt, tol,
                                        any_pose=bool(cand.get("drop_in")) or beside_drop,
                                        compartment_xy=cand.get("compartment_xy"),
                                        rest_z=cand["rest_z"] if cand.get("stacked") or (cand.get("drop_in") and rel == "on")
                                        else None)
            if check.get("unverified") and cand.get("drop_in"):
                # a drop whose result was not seen is not accepted as it is: look once more, from another side and
                # with all of S1's prompts for the object
                again = self.check_relation(obj, dest, rel, region, place_pt, tol, any_pose=True,
                                            compartment_xy=cand.get("compartment_xy"), second_view=True)
                self.log("placed_recheck", object=obj.name, check=again)
                if not again.get("unverified"):
                    check = again
            result["attempts"].append({"phase": "place", "status": "released", "check": check,
                                       "xy": cand["xy"], "clearance": cand["clearance"]})
            result["relation_check"] = check
            self.log("placed", object=obj.name, check=check)
            if check["satisfied"] or check.get("unverified"):
                return True
            if check.get("object_lost"):
                raise StageFailure("object_lost_after_release", check=check)
            # not satisfied: re-grasp and try again
            self.grasp_object(obj, st, result)
            failed.append((cand["xy"], 0.006))
        raise StageFailure("placement_failed", attempts=result["attempts"][-3:])

    # ---------------------------------------------------------------- place, "v2" (r1's mechanism)
    def place_object_v2(self, obj, dest, st, place_pt, result):
        """Placement "v2" (r1's mechanism, user 2026-11-22): no placement planner. System 1 points at where the held
        object goes; System 2 carries it above that point (a collision-checked transit), lowers it straight down until
        it touches something, opens the gripper, and System 1 judges the result from fresh images and points again
        when it is not right. Nothing is refused on geometry; a motion that cannot be executed is reported back to
        System 1, which then points elsewhere."""
        rel = st["relation"]
        release = st.get("release") or ("drop" if rel == "in" else "place")
        notes, aim = [], None                    # notes: what happened in earlier attempts, for System 1
        cleared = False
        for attempt in range(V2_TRIES):
            self._v2_half = self._v2_held_half()
            if aim is None:
                aim = self._v2_pose(obj, dest, st, release, notes, place_pt)
            put = self._v2_put(obj, dest, rel, release == "drop", aim)
            if put["status"] != "released" and not cleared and self._start_side(put.get("detail")):
                # the way is blocked where it starts (the carried object still inside the margins): a pixel cannot
                # change that; lift clear and try the same spot again, no model call
                cleared = True
                self._lift_clear()
                put = self._v2_put(obj, dest, rel, release == "drop", aim)
            note = {"attempt": attempt + 1, "aimed_at_pixel": aim.get("pixel"), "long_side": aim.get("long_side"),
                    "result": put}
            if put["status"] != "released":
                result["attempts"].append({"phase": "place", "status": put["status"], "xy": aim["xy"]})
                notes.append(note)
                aim = None
                continue
            look, nxt = self._v2_look(obj, dest, st, release, put, notes)
            check = {"satisfied": bool(look.get("done")), "by": "system1_look", "evidence": look.get("evidence"),
                     "ended_on_contact": put.get("ended_on_contact")}
            if look.get("done") is None:
                check["unverified"] = True           # no look (model budget / error): VERIFY judges at the end
            result["attempts"].append({"phase": "place", "status": "released", "check": check, "xy": aim["xy"]})
            result["relation_check"] = check
            self.log("placed", object=obj.name, check=check)
            if check["satisfied"] or check.get("unverified"):
                return True
            notes.append({**note, "seen_after_release": look.get("evidence")})
            if attempt == V2_TRIES - 1:
                break
            if not look.get("refreshed") and not self.relocate_object(obj, self._v2_release_p):
                # it is not where it was let go and the head camera does not find it: System 1 re-points it (REPAIR)
                raise StageFailure("object_lost_after_release", check=check, notes=notes[-2:])
            self.grasp_object(obj, st, result)       # pick it up again and put it where System 1 points now
            aim = nxt
        raise StageFailure("placement_failed", attempts=result["attempts"][-3:], notes=notes[-3:])

    def _why_words(self, w):
        """A typed refusal (TraceMotion.why) in a sentence for System 1."""
        k = w.get("kind")
        if k == "no_arm_posture":
            return "no arm posture reaches the pose above that point"
        if k == "end_blocked":
            return (f"the pose above that point is blocked: the {w.get('body')} would overlap {w.get('thing')} by "
                    f"{w.get('overlap_mm')} mm")
        if k == "path_blocked":
            return (f"no collision-free way there: the path meets {w.get('thing')} "
                    f"({w.get('short_mm')} mm short of the margin)")
        if k == "arm":
            d = w.get("detail") or {}
            return f"the arm cannot follow the way there ({w.get('reason')}" + \
                (f", joint {d.get('joint')})" if d.get("joint") else ")")
        return "no collision-free path to above that point (search found none)"

    @staticmethod
    def _start_side(detail):
        """A typed refusal that binds where the move starts (path or arm within the first 15 % of the way)."""
        d = detail or {}
        return d.get("kind") in ("path_blocked", "arm") and d.get("at_t") is not None and float(d["at_t"]) <= 0.15

    def _lift_clear(self):
        """Lift the hand (and what it holds) straight up until it clears the observed scene by the transit margins."""
        Rs, ps = self.rb.eef_pose()
        res = self.retreat(Rs, ps + np.array([0.0, 0.0, 0.06]), grip=0.0 if self.held is not None else None,
                           speed=0.08, tag="lift_clear")
        self.log("lift_clear", status=None if res is None else res.status)

    def _v2_held_half(self):
        """The held object's narrowest horizontal half-extent from its footprint centre (at any turn about the
        vertical): the room an opening must leave around the aim."""
        fp = self._payload_world(*self.rb.eef_pose())[:, :2]
        fp = fp - 0.5 * (fp.min(0) + fp.max(0))
        a = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        return float((fp @ np.stack([np.cos(a), np.sin(a)])).max(axis=0).min())

    def _v2_stage(self, obj, dest, st, release):
        return {"object": obj.name, "reference": dest.name, "relation": st["relation"], "release": release,
                "note": st.get("note")}

    def _v2_pose(self, obj, dest, st, release, notes, place_pt, before_grasp=False):
        """System 1 points at where the held object goes (one model call). Without an answer (call budget, error):
        the program's place point, else the middle of the destination. before_grasp: asked before the object is
        picked up (the grasp is chosen for that put-down, CONTACT_STAGE_PLAN 4.5 look-ahead)."""
        head = self.capture(HEAD_CAMERA)
        self.frames.append(head)
        self.save_img("v2_pose_head", head.rgb)
        content = self.task_parts() + ([text_part(
            "The object is not held yet: it still stands where it is now and is picked up next, with a grasp chosen "
            "for this put-down. Point where it should be put down.")] if before_grasp else []) + [
            text_part("Stage: " + json.dumps(self._v2_stage(obj, dest, st, release))),
            text_part("Earlier attempts of this stage: "
                      + (json.dumps(notes, default=_jsonable)[:3000] if notes else "none")),
            text_part(f"IMAGE 1 (head camera, {IMG}x{IMG}):"), image_part(head.rgb)]
        t = time.time()
        instr = PLACE_POSE_INSTRUCTIONS_UNIFIED if self.placement == "unified" else PLACE_POSE_INSTRUCTIONS
        try:
            data, rec = self.astra.call("place_pose", instr, content,
                                        PLACE_POSE_SCHEMA_UNIFIED if self.placement == "unified" else PLACE_POSE_SCHEMA)
        except AstraError as e:
            self._tick("model_wait", t)
            xy = np.asarray(place_pt[:2] if place_pt is not None else dest.footprint_center(), float)
            self.log("v2_pose_error", error=repr(e), fallback_xy=xy)
            return {"xy": xy, "long_side": "any", "pixel": None}
        self._tick("model_wait", t)
        aim = self._v2_aim(head, data["x"], data["y"], data["long_side"], dest, st["relation"])
        aim["entry"] = data.get("entry")
        self.log("v2_pose", answer=data, record=rec, **{k: v for k, v in aim.items() if k not in ("long_side", "pixel")})
        return aim

    def _v2_aim(self, fr, x, y, long_side, dest, rel):
        """System 1's pixel as a place. The object comes straight down, so the pixel is read on the horizontal plane at
        the destination's top near the ray (the table for a spot beside the reference), not where the ray ends on an
        inner wall. For a container whose opening the model shows, the pixel selects the opening (compartment) its ray
        passes through, and the aim is the spot of that opening nearest to the pointed one where the object has room
        (its middle when the opening is narrower than the object): a pixel cannot resolve a few millimetres along the
        viewing direction."""
        H, W = fr.depth.shape
        x, y = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
        if self.placement == "unified":
            # the surface under the pixel is where the object's centre should rest (System 1 is told so); the rest
            # poses are searched around it. "in" on a vertical face (a container's front panel): the interior support
            # behind it instead
            p = fr.pixel_ray_point(x, y)
            if np.all(np.isfinite(p)):
                aim = {"xy": p[:2].copy(), "rest_xy": p[:2].copy(), "z_top": float(p[2]), "long_side": long_side,
                       "pixel": [x, y]}
                if rel == "in" and not getattr(dest, "virtual", False):
                    q = self._inward_support(fr, x, y, p, dest)
                    if q is not None:
                        self.log("aim_moved_inward", pointed=p, to=q)
                        aim.update(xy=q[:2].copy(), rest_xy=q[:2].copy(), z_top=float(q[2]), pointed=p.copy())
                return aim
        o = fr.T_wc[:3, 3]
        d = fr.T_wc[:3, :3] @ np.array([(x + 0.5 - fr.K[0, 2]) / fr.K[0, 0], (y + 0.5 - fr.K[1, 2]) / fr.K[1, 1], 1.0])
        if abs(d[2]) < 1e-6:
            p = fr.pixel_ray_point(x, y)
            return {"xy": p[:2], "z_top": float(p[2]), "long_side": long_side, "pixel": [x, y]}
        at = lambda z: o + d * ((z - o[2]) / d[2])
        z = self.table_z
        if rel in ("in", "on") and len(dest.points):
            # z -> the top of the destination's points within 5 cm of the ray at z. A level top (flat surface, even
            # rim) converges. Where the top steps along the ray (a low near wall, a tall far wall: a slanted opening)
            # the heights alternate, and the opening lies between them: their mean
            z, tops = dest.top_z, []
            for _ in range(4):
                near = dest.points[np.linalg.norm(dest.points[:, :2] - at(z)[:2], axis=1) < 0.05]
                if len(near) == 0:
                    break
                tops.append(float(np.percentile(near[:, 2], 99.5)))
                if abs(tops[-1] - z) < 0.003:
                    break
                z = tops[-1]
            else:
                z = 0.5 * (tops[-1] + tops[-2])
        p = at(z)
        out = {"xy": p[:2].copy(), "z_top": float(z), "long_side": long_side, "pixel": [x, y]}
        if rel == "in" and len(dest.points) and self.placement != "unified":
            # (unified: no snap; the rest poses come from all observed geometry around the pointed spot)
            reg = container_region(dest)
            comp, lo_xy, cell, gxy = reg["compartments"], reg["lo_xy"], reg["cell"], reg["grid_xy"]

            def label(q):
                ij = np.floor((np.atleast_2d(q) - lo_xy) / cell).astype(int)
                ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < comp.shape[0]) & (ij[:, 1] < comp.shape[1])
                lab = np.zeros(len(ij), int)
                lab[ok] = comp[ij[ok, 0], ij[ok, 1]]
                return lab
            glab = label(gxy) if len(gxy) else np.zeros(0, int)
            if (glab > 0).any():
                # the opening the ray passes through between the destination's top and its floor (else the nearest)
                along = label(np.array([at(v)[:2] for v in np.linspace(max(dest.top_z, z), reg["floor_z"], 24)]))
                along = along[along > 0]
                live = np.flatnonzero(glab > 0)
                k = int(np.bincount(along).argmax()) if len(along) else \
                    int(glab[live[np.argmin(np.linalg.norm(gxy[live] - p[:2], axis=1))]])
                cells = gxy[glab == k]
                ij = np.floor((cells - lo_xy) / cell).astype(int)
                room = (ndimage.distance_transform_edt(comp == k) * cell)[ij[:, 0], ij[:, 1]]   # distance to its walls
                need = min(float(room.max()) - 0.5 * cell, getattr(self, "_v2_half", 0.0) + 0.005)
                ok = cells[room >= need]
                out.update(xy=ok[np.argmin(np.linalg.norm(ok - p[:2], axis=1))].copy(), pointed_xy=p[:2].copy(),
                           opening=k, room=round(float(room.max()), 3))
        return out

    def _inward_support(self, fr, x, y, p0, dest, depth=(0.03, 0.25), lateral=0.12):
        """System 1 pointed an "in" destination on a vertical face (the surface normal at the pixel is horizontal):
        the interior support behind it, i.e. the lowest surface of the destination (its model and the unlabelled
        surface connected to it) behind the face along the inward normal (`depth`), within `lateral` of that line and
        not above the pointed spot; the point of it nearest that line. None when the face is not vertical or nothing is
        found."""
        Hh, Ww = fr.depth.shape
        P = np.array([fr.pixel_ray_point(u, v) for u in range(max(0, x - 6), min(Ww, x + 7), 3)
                      for v in range(max(0, y - 6), min(Hh, y + 7), 3)])
        P = P[np.all(np.isfinite(P), axis=1)]
        if len(P) < 8:
            return None
        w, V = np.linalg.eigh(np.cov((P - P.mean(0)).T))
        n = V[:, 0]
        if float(n @ (fr.T_wc[:3, 3] - p0)) < 0.0:
            n = -n
        if abs(float(n[2])) >= 0.5 or np.linalg.norm(n[:2]) < 1e-6:
            return None
        inward = -np.r_[n[:2], 0.0] / np.linalg.norm(n[:2])
        side = np.array([-inward[1], inward[0], 0.0])
        _, fx = self._fixture(dest, dest)
        S = dest.points if fx is None or not len(fx) else np.concatenate([dest.points, fx])
        d_in, lat = (S - p0) @ inward, (S - p0) @ side
        sel = (d_in >= depth[0]) & (d_in <= depth[1]) & (np.abs(lat) <= lateral) & (S[:, 2] <= p0[2] + 0.02) & \
            (S[:, 2] > self.table_z + 0.01)
        if sel.sum() < 5:
            return None
        Q, dq, lq = S[sel], d_in[sel], lat[sel]
        low = Q[:, 2] <= np.percentile(Q[:, 2], 5) + 0.01          # the lowest level: the floor inside
        k = int(np.argmin(np.abs(lq[low]) + 0.3 * dq[low]))
        return Q[low][k].copy()

    def _v2_yaws(self, fp, long_side):
        """Turns of the hand about the vertical to try: those that lay the payload's longest horizontal side along
        the direction System 1 named in the head image (the smaller of the two equivalent turns first), then the hand
        as it is. When the long side does not matter ("any"), quarter turns too (smallest first): the set of hand-offs
        the arm can choose from."""
        if long_side in (None, "any") and self.motion != "cp5":
            return [0.0, np.pi / 2, -np.pi / 2, np.pi]
        yaws = []
        if long_side not in (None, "any") and len(fp) >= 10:
            w, v = np.linalg.eigh(np.cov(fp.T))
            if w[1] > 1.3 * max(w[0], 1e-9):                       # elongated: its long side has a direction
                R = self.head0.T_wc[:3, :3]
                right = R[:2, 0] / np.linalg.norm(R[:2, 0])
                front = -R[:2, 2] / np.linalg.norm(R[:2, 2])
                want = {"left_right": right, "near_far": front, "diagonal_rising": right - front,
                        "diagonal_falling": right + front}[long_side]
                u = v[:, 1]
                a = _wrap_pi(np.arctan2(want[1], want[0]) - np.arctan2(u[1], u[0]))
                b = _wrap_pi(a + np.pi)
                yaws = sorted([float(a), float(b)], key=abs)
        return yaws + [y for y in (0.0,) if all(abs(y - k) > 0.05 for k in yaws)]

    def _v2_put(self, obj, dest, rel, drop, aim):
        """Carry the held object above the aimed point, lower it straight down until contact (a drop: let go 3 cm
        above what lies under it), open the gripper and lift off. The place itself is not tested."""
        Rs, ps = self.rb.eef_pose()
        finger = self._finger()
        pay = self._payload_world(Rs, ps)
        pay_c = 0.5 * (pay[:, :2].min(0) + pay[:, :2].max(0))
        pay_r = float(np.linalg.norm(pay[:, :2] - pay_c, axis=1).max())
        bottom = float(pay[:, 2].min())
        xy = np.asarray(aim["xy"], float)
        # the top of what lies under the footprint there (start height of the descent)
        under = [m.points for m in self.objects.values()
                 if not m.held and len(m.points) and not getattr(m, "virtual", False)]
        under = np.concatenate(under) if under else np.zeros((0, 3))
        under = under[np.linalg.norm(under[:, :2] - xy, axis=1) < pay_r + 0.01]
        z_top = max(self.table_z, float(under[:, 2].max()) if len(under) else self.table_z)
        # the object cannot go lower than the base of what it is put in / on
        z_low = float(dest.points[:, 2].min()) if rel in ("in", "on") and len(dest.points) else self.table_z
        q_now = self.rb.q()[:7]
        yaw, why = None, "no arm posture reaches above that point"
        yaws = self._v2_yaws(pay[:, :2] - pay_c, aim.get("long_side"))
        if self.motion != "cp5":
            # one hand-off set (CONTACT_STAGE_PLAN 4.2): the contact pose (payload just above what is under it) for
            # every allowed turn about the vertical; TRACE slides each back up until its end test passes and the arm
            # has a posture there, and takes the first it certifies; the guarded descent covers the rest
            ends = []
            for yaw_ in yaws:
                Rz = rot_z(yaw_)
                site_xy = xy + Rz[:2, :2] @ (ps[:2] - pay_c)
                e = EndCandidate(Rz @ Rs, np.r_[site_xy, z_top + 0.005 + (ps[2] - bottom)],
                                 approach=np.array([0.0, 0.0, -1.0]), standoff=SLIDE, entity=entity_of(self, dest.oid))
                e.yaw = yaw_
                ends.append(e)
            res = self.tm.move(ends, "v2_above", grip=0.0, finger=finger, entity=entity_of(self, dest.oid),
                               approach=np.array([0.0, 0.0, -1.0]), keep_ends=len(ends))
            if res is not None and res.status == "dropped":
                raise StageFailure("dropped_in_transit")
            if res is not None and res.status == "done" and self.tm.last_end is not None:
                yaw = float(self.tm.last_end.yaw)
            else:
                lm = getattr(self, "last_move", None) or {}
                w = lm.get("why") or {}
                why = self._why_words(w) if w else f"no collision-free path to above that point ({lm.get('status')})"
                if lm.get("status") == "contact":
                    why = f"the move there ran into {lm.get('contact_near')}"
                    self.reobserve()
        else:
            for yaw_ in yaws:
                Rz = rot_z(yaw_)
                R_place = Rz @ Rs
                site_xy = xy + Rz[:2, :2] @ (ps[:2] - pay_c)
                for up in (0.04, 0.10, 0.16):
                    p_pre = np.r_[site_xy, z_top + up + (ps[2] - bottom)]
                    sols = self.ik_clear(R_place, p_pre, q_now, exclude=(dest.oid,), finger=finger, n_random=8)
                    if len(sols) == 0:
                        continue
                    if self.go_configs(list(sols[:3]), TRANSIT, grip=0.0, finger=finger, tag="v2_above"):
                        yaw = yaw_
                        break
                    lm = getattr(self, "last_move", None) or {}
                    why = f"no collision-free path to above that point ({lm.get('status') or lm.get('result')})"
                    if lm.get("status") == "contact":
                        why = f"the move there ran into {lm.get('contact_near')}"
                        self.reobserve()               # something was hit: the map is stale where it matters
                        q_now = self.rb.q()[:7]
                        break
                if yaw is not None:
                    break
        if yaw is None:
            detail = (getattr(self, "last_move", None) or {}).get("why")
            self.log("v2_put", status="could_not_reach_above_point", why=why, xy=xy, z_top=z_top, detail=detail)
            return {"status": "could_not_reach_above_point", "why": why, "detail": detail}
        Rc, pc = self.rb.eef_pose()
        bottom = float(self._payload_world(Rc, pc)[:, 2].min())
        Wr, _ = self.checker.world_points(self.rb.q()[None, :7], finger)
        hand_low = float(Wr[0][self.checker.pg >= 1][:, 2].min())
        down = bottom - (z_top + 0.03) if drop else bottom - max(z_low - 0.02, self.table_z + 0.002)
        down = max(0.0, min(down, hand_low - (self.table_z + 0.004)))

        def guard(robot, F, F0, k):
            return "support_contact" if (F - F0)[2] > 4.0 and k > 2 else None
        res = None
        for cut in (0.0, 0.03, 0.06, 0.10, 0.15):                  # as far down as the arm can follow the line
            if down - cut < 0.004:
                break
            res = self.move_line(Rc, pc - np.array([0, 0, down - cut]), grip=0.0, speed=0.05,
                                 guard=self._guard("support_contact", guard))
            if res is not None:
                break
        touched = res is not None and res.status == "support_contact"
        self.ex.gripper(-1.0, min_steps=8)                       # binary release, as r1
        Rr, pr = self.rb.eef_pose()
        self._v2_release_p = pr.copy()
        end_bottom = float(self._payload_world(Rr, pr)[:, 2].min())
        self._finish_release(obj, end_bottom if touched else z_low, touched=touched)
        res2 = self.retreat(Rr, pr + np.array([0, 0, 0.10]), grip=0.0, speed=0.08, finger=manip.FULL_JOINT,
                            tag="release_lift")
        self.ex.gripper(-1.0, max_steps=12)
        if res2 is None:
            self.go_configs([self._ik_near(Rr, pr + np.array([0, 0, 0.10]), self.rb.q()[:7])], NEAR,
                            grip=-1.0, finger=0.04, tag="retreat")
        out = {"status": "released", "ended_on_contact": bool(touched),
               "let_go": "on contact" if touched else ("above the destination (a drop)" if drop
                                                         else "at the end of the descent, nothing touched"),
               "turned_deg": int(round(np.degrees(yaw))), "lowered_cm": round(100 * float(pc[2] - pr[2]), 1)}
        self.log("v2_put", **out, xy=xy, z_top=z_top, z_low=z_low, descend=None if res is None else res.status,
                 release_bottom_above_table=round(end_bottom - self.table_z, 4))
        return out

    def _v2_look(self, obj, dest, st, release, put, notes, relocalized=False):
        """After the release: lift clear, take fresh head and wrist images, and let System 1 judge the result (one
        model call). Returns (its answer, the next aim when it is not done and System 1 pointed again). Without an
        answer, done is None: accepted as put, VERIFY judges at the end. relocalized: the object's known shape was
        already registered where it rests (one object, one model): no new segmentation."""
        self.retreat_home()
        head = self.capture(HEAD_CAMERA)
        wrist = self.capture(WRIST_CAMERA)
        self.frames += [head, wrist]
        self.save_img("v2_look_head", head.rgb)
        self.save_img("v2_look_wrist", wrist.rgb)
        refreshed = relocalized
        if not relocalized:
            fit = self.relocalize(obj, [wrist, head])
            refreshed = fit is not None or self.refresh_object(
                obj, [wrist, head], list(dict.fromkeys([obj.category, obj.name] + list(obj.prompts or []))))
        self.rebuild_map(self.frames[-8:])
        content = self.task_parts() + [
            text_part("Stage: " + json.dumps(self._v2_stage(obj, dest, st, release))),
            text_part("This attempt, as measured by System 2: " + json.dumps(put, default=_jsonable)),
            text_part("Earlier attempts of this stage: "
                      + (json.dumps(notes, default=_jsonable)[:3000] if notes else "none"))]
        content += self.head_and_wrist_content(head, wrist)
        t = time.time()
        try:
            data, rec = self.astra.call("place_look", PLACE_LOOK_INSTRUCTIONS_UNIFIED if self.placement == "unified"
                                        else PLACE_LOOK_INSTRUCTIONS, content, PLACE_LOOK_SCHEMA)
        except AstraError as e:
            self._tick("model_wait", t)
            self.log("v2_look_error", error=repr(e))
            return {"done": None, "evidence": "look unavailable: " + repr(e)[:120]}, None
        self._tick("model_wait", t)
        nxt = None
        if not data["done"] and data.get("x") is not None and data.get("y") is not None:
            nxt = self._v2_aim(head, data["x"], data["y"], data.get("long_side") or "any", dest, st["relation"])
        self.log("v2_look", answer=data, record=rec, refreshed=refreshed, next_xy=None if nxt is None else nxt["xy"],
                 next_opening=None if nxt is None else nxt.get("opening"))
        return {**data, "refreshed": bool(refreshed)}, nxt

    def _finish_release(self, obj, support_z, touched=False):
        """Object left the hand: its model moves with the hand's last pose, then re-enters the map.
        Its support height is measured, not planned: where its bottom is when the descent ended on contact with the
        support (touched), else at most that (it cannot rest above where it was let go). A planned rest height comes from
        the container's estimated floor, which can be several cm too high."""
        Rs, ps = self.rb.eef_pose()
        moved = self._payload_world(Rs, ps)
        att = self.checker.attached
        model = ps + att.model_site @ Rs.T if att.model_site is not None else moved
        bottom = float(moved[:, 2].min())
        planned = support_z
        support_z = bottom if touched else min(support_z, bottom)
        if abs(support_z - planned) > 0.003:
            self.log("support_measured", object=obj.name, planned=planned, measured=support_z, touched=touched)
        self._release_attachment(obj)
        obj.held = False
        obj.support_z = support_z
        drop = max(0.0, float(moved[:, 2].min()) - (support_z + 0.001))
        obj.set_points(model - np.array([0, 0, drop]))
        self._forget_free(obj.points)
        # something is there again: a vacated volume it lands in is no longer cleared
        lo_o, hi_o = obj.points.min(0), obj.points.max(0)
        self._vacated = [(o, lo, hi) for o, lo, hi in getattr(self, "_vacated", [])
                         if np.any(hi < lo_o) or np.any(lo > hi_o)]
        self.scene.insert_object(obj, force=True)

    def _payload_world(self, Rs, ps):
        return ps + self.checker.attached.pts_site @ Rs.T

    def _held_size(self):
        if self.checker.attached is None:
            return None
        p = self.checker.attached.pts_site
        return np.round(p.max(0) - p.min(0), 3).tolist()

    def _region_info(self, dest, rel, place_pt):
        try:
            reg = self.region_for(dest, rel, place_pt, "normal")
            return {"kind": reg["kind"], "cells": int(len(reg["grid_xy"])),
                    "max_clearance": float(reg["clearance"].max()) if len(reg["clearance"]) else 0.0}
        except Exception as e:
            return {"error": repr(e)}

    def region_for(self, dest, rel, place_pt, tol):
        if rel == "in":
            reg = container_region(dest)
            if place_pt is not None:
                d = np.linalg.norm(reg["grid_xy"] - place_pt[:2], axis=1)
                r = {"tight": 0.03, "normal": 0.06, "loose": 0.2}[tol]
                keep = d <= max(r, d.min() + 0.01)
                reg = {**reg, "grid_xy": reg["grid_xy"][keep], "clearance": reg["clearance"][keep]}
            return reg
        if rel == "on":
            r = {"tight": 0.03, "normal": 0.05, "loose": 0.09}[tol]
            return support_region(dest, point=place_pt, radius=r)
        # beside relations: a table region next to the reference, direction from the head camera
        R = self.head0.T_wc[:3, :3]
        right = R[:, 0].copy()
        right[2] = 0
        right /= np.linalg.norm(right)
        front = -R[:, 2].copy()
        front[2] = 0
        front /= np.linalg.norm(front)
        dirs = {"right_of": right, "left_of": -right, "front_of": front, "behind": -front}
        u = dirs[rel][:2]                  # top-view direction (the footprint quantities below are 2D)
        lo, hi = dest.bbox
        half = 0.5 * np.abs((hi - lo)[:2] @ u) + 0.5 * np.abs((hi - lo)[:2] @ np.array([-u[1], u[0]])) * 0.0
        c = dest.footprint_center() + u * (half + 0.07)
        g = np.stack(np.meshgrid(np.arange(-0.05, 0.051, 0.005), np.arange(-0.05, 0.051, 0.005)), -1).reshape(-1, 2)
        gxy = c + g
        return {"floor_z": self.table_z, "rim_z": self.table_z, "grid_xy": gxy,
                "clearance": np.full(len(gxy), 0.05), "kind": "on", "beside": u, "ref_center": dest.footprint_center()}

    def plan_place(self, obj, dest, rel, place_pt, tol, failed, region_override=None, drop=False):
        """Careful placement. Plans with the user's scoring rule: unlabeled surface and unobserved space hugging the
        destination count as the destination; unobserved space elsewhere is no obstacle (the returned candidate says
        'unknown_near' when it relies on that, and the caller looks first); margins follow the ladder TIER_SCALE."""
        prev = self.checker.free_near
        base = list(prev or []) + ([(dest.points, 0.01, True)] if len(dest.points) else [])
        # what the checker treats as part of the destination, per ladder tier (v1's tier 3 adds its fixture)
        self._free_tiers = [base, base, base]
        self._fix_oids = ()
        if self.placement == "v1":
            self._fix_oids, fix_pts = self._fixture(dest, obj)
            self._free_tiers.append(base + ([(fix_pts, 0.01, True)] if len(fix_pts) else []))
        self.checker.free_near = base
        self._place_unknown, self._tier = False, 0
        try:
            return self._plan_place(obj, dest, rel, place_pt, tol, failed, region_override, drop)
        finally:
            self.checker.free_near = prev
            self._place_unknown, self._tier = True, 0

    def _set_tier(self, tier, need=None):
        self._tier = tier
        if need is not None:
            self._need = need
        self.checker.free_near = self._free_tiers[tier]

    def _fixture(self, dest, obj):
        """The fixture a destination belongs to (placement "v1"): the objects System 1 tied to it (part_of in either
        direction, and the other parts of the same parent), and the unlabeled observed surface connected to them
        above the table (1 cm cells, 26-connected). Under the scoring rule touching it does not count, so it ranks
        a placement instead of ruling it out. Returns (object ids, surface points)."""
        none = np.zeros((0, 3))
        if dest.oid not in self.objects or getattr(dest, "virtual", False):
            return (), none                          # a table spot (parking): no fixture
        par = dest.part_of
        oids = [m.oid for m in self.objects.values()
                if m.oid not in (dest.oid, obj.oid) and not m.held and not getattr(m, "virtual", False)
                and (m.oid == par or m.part_of == dest.oid or (par is not None and m.part_of == par))]
        near = self.checker.near
        near._labels(self.scene)
        seed = [p for p in [dest.points] + [self.objects[o].points for o in oids] if len(p)]
        if not seed or len(near.cloud) == 0:
            return tuple(oids), none
        seed = np.concatenate(seed)
        res = self.scene.res
        lo = np.r_[seed[:, :2].min(0) - 0.30, getattr(self, "table_z_visual", self.table_z) + 0.012]
        hi = np.r_[seed[:, :2].max(0) + 0.30, seed[:, 2].max() + 0.30]
        free = (near.lab == 0) & np.all((near.cloud >= lo) & (near.cloud <= hi), axis=1)
        cloud = near.cloud[free]
        seed = seed[np.all((seed >= lo) & (seed <= hi), axis=1)]
        if len(cloud) == 0 or len(seed) == 0:
            return tuple(oids), none
        shape = tuple(np.ceil((hi - lo) / res).astype(int) + 1)
        ci = np.floor((cloud - lo) / res).astype(int)
        si = np.floor((seed - lo) / res).astype(int)
        grid = np.zeros(shape, bool)
        grid[ci[:, 0], ci[:, 1], ci[:, 2]] = True
        grid[si[:, 0], si[:, 1], si[:, 2]] = True
        comp, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
        mine = np.unique(comp[si[:, 0], si[:, 1], si[:, 2]])
        pts = cloud[np.isin(comp[ci[:, 0], ci[:, 1], ci[:, 2]], mine[mine > 0])]
        self.log("fixture", destination=dest.name, objects=[self.objects[o].name for o in oids], points=int(len(pts)),
                 size=np.round(np.ptp(pts, axis=0), 3).tolist() if len(pts) else None)
        return tuple(oids), pts

    def _soft_touch(self, R, p, hand, pts_site, reg):
        """Placement "v1" ranking: the share of the hand's and the payload's points in predicted contact with the
        destination's walls, its fixture or what the payload may touch, at the placing pose (0 = touches nothing)."""
        L, _, rad = hand
        Wh = p + L @ R.T
        prev, self.checker.free_near = self.checker.free_near, self._free_tiers[0]
        try:
            n = int(((self.checker.point_clearance(Wh, self._place_excl, self._place_ignore, unknown=False) - rad) < 0).sum())
            tot = len(Wh)
            walls = reg.get("wall_pts")
            tree = None
            if walls is not None and len(walls):
                tree = reg.get("_tree3")
                if tree is None:
                    tree = reg["_tree3"] = cKDTree(walls)
                n += int(((tree.query(Wh)[0] - rad) < 0).sum())
            if pts_site is not None:
                Wp = p + pts_site @ R.T
                n += int(((self.checker.point_clearance(Wp, self._place_excl, self._place_ignore, unknown=False)
                           - 0.004) < 0).sum())
                if tree is not None:
                    n += int((tree.query(Wp)[0] < 0.002).sum())
                tot += len(Wp)
        finally:
            self.checker.free_near = prev
        return n / max(tot, 1)

    def _task_object_ids(self, obj):
        """The program's other transfer objects (task objects: the payload may touch them under the scoring rule)."""
        out = []
        for st in (self.program or {}).get("stages", []):
            oid = self.id_map.get(st.get("object")) if st.get("op") == "transfer" else None
            if oid is not None and oid != obj.oid and oid in self.objects and oid not in out:
                out.append(oid)
        return tuple(out)

    def _plan_place(self, obj, dest, rel, place_pt, tol, failed, region_override=None, drop=False):
        t = time.time()
        reg = region_override if region_override is not None else self.region_for(dest, rel, place_pt, tol)
        v1 = self.placement == "v1"
        n_tiers = 4 if v1 else 3
        # 'on' may stack: the destination and whatever already rests on it form the support, tested in 2.5D
        stack = rel == "on" and region_override is None
        contents = []
        if stack:
            contents = self._on_contents(dest, obj)
            sup = np.concatenate([dest.points] + [self.objects[o].points for o in contents])
            centres = manip.support_centres(sup)
            if place_pt is not None:
                r = {"tight": 0.03, "normal": 0.05, "loose": 0.09}[tol]
                d_pt = np.linalg.norm(centres - place_pt[:2], axis=1)
                keep = d_pt <= r
                if v1 and len(d_pt) and not keep.any():
                    keep = d_pt <= d_pt.min() + 0.02       # v1: none within the window: the support cells nearest to it
                centres = centres[keep]
            reg = {**reg, "grid_xy": centres}
        if len(reg["grid_xy"]) == 0:
            if not v1 or len(dest.points) == 0:
                return None
            # v1: no opening / flat top was found on the destination: aim at System 1's point, else at its middle
            aim = np.asarray(place_pt[:2] if place_pt is not None else dest.footprint_center(), float)
            reg = {**reg, "grid_xy": aim[None], "clearance": np.zeros(1)}
        # contact classes: the payload may touch the destination, what is in/on it and the other task objects
        task_ids = self._task_object_ids(obj)
        inside = list(contents)
        if reg["kind"] == "in" and "interior" in reg:
            for m in self.objects.values():
                if m.oid in (dest.oid, obj.oid) or m.held or len(m.points) == 0 or getattr(m, "virtual", False):
                    continue
                ij = np.floor((m.footprint_center() - reg["lo_xy"]) / reg["cell"]).astype(int)
                if (ij >= 0).all() and ij[0] < reg["interior"].shape[0] and ij[1] < reg["interior"].shape[1] \
                        and reg["interior"][ij[0], ij[1]]:
                    inside.append(m.oid)
        self._place_excl_pay = tuple(dict.fromkeys((dest.oid, obj.oid) + tuple(inside) + task_ids))
        Rs, ps = self.rb.eef_pose()
        pay = self._payload_world(Rs, ps)
        pay_c = 0.5 * (pay[:, :2].min(0) + pay[:, :2].max(0))
        fp_local = voxel_down(np.c_[pay[:, :2] - pay_c, np.zeros(len(pay))], 0.002)[:, :2]
        pay_r = float(np.linalg.norm(fp_local, axis=1).max())
        bottom = float(pay[:, 2].min())
        gx = reg["grid_xy"]
        lo, hi = gx.min(0) - 0.08, gx.max(0) + 0.08
        if stack:
            prof_xy, prof_b = manip.bottom_profile(np.c_[pay[:, :2] - pay_c, pay[:, 2]])
            g_xy = pay[:, :2].mean(0) - pay_c
            obst = self._on_obstacles(dest, obj, contents + list(task_ids), sup, lo, hi)
            touch = [self.objects[o].points for o in task_ids if o not in contents and len(self.objects[o].points)]
            touch = np.concatenate(touch) if touch else np.zeros((0, 3))
            touch = touch[np.all((touch[:, :2] >= lo) & (touch[:, :2] <= hi), axis=1)]
        # things the payload footprint must avoid: other known objects (+ unknown clutter for open supports)
        block, block_hard = [], []            # block_hard (v1): only what the payload may not touch
        for m in ([] if stack else self.objects.values()):
            if m.oid in (dest.oid, obj.oid) or m.held or len(m.points) == 0:
                continue
            mp = m.points[m.points[:, 2] > reg["floor_z"] + 0.006][:, :2]
            inb = np.all((mp >= lo) & (mp <= hi), axis=1)
            if inb.any():
                block.append(voxel_down(np.c_[mp[inb], np.zeros(inb.sum())], 0.004)[:, :2])
                if m.oid not in self._place_excl_pay:
                    block_hard.append(block[-1])
        if reg["kind"] != "in" and not stack:
            def clutter(exclude):
                hmap, org = self.scene.heightmap((lo[0], hi[0]), (lo[1], hi[1]), exclude=exclude)
                ij = np.argwhere(hmap > reg["floor_z"] + 0.006)
                if len(ij) == 0:
                    return None
                bl = np.array(org) + (ij + 0.5) * self.scene.res
                if len(dest.points):
                    dxy, _ = cKDTree(dest.points[:, :2]).query(bl)
                    bl = bl[dxy > 0.015]
                return bl
            bl = clutter((dest.oid, obj.oid))
            if bl is not None:
                block.append(bl)
            bl = clutter(self._place_excl_pay) if v1 else None
            if bl is not None:
                block_hard.append(bl)
        block = np.concatenate(block) if block else None
        block_hard = np.concatenate(block_hard) if block_hard else None
        target_xy = place_pt[:2] if place_pt is not None else None
        if target_xy is None and reg["kind"] == "on" and "beside" not in reg and dest.oid >= 0:
            target_xy = dest.footprint_center()
        yaws = [0.0, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4, np.pi]
        future = self.future_footprints(dest, obj) if reg["kind"] == "in" else []
        # clearance floor = resolution of the object point models (3 mm); never planned below it
        levels = {"loose": [0.008, 0.005, 0.003], "normal": [0.005, 0.003], "tight": [0.003]}[tol]
        # margin ladder: the tolerance's levels (tier 0); half the last one (tier 1); touching the destination's walls
        # and support (and, stacked, the other task objects), half the last level from everything else (tier 2)
        half = levels[-1] / 2
        ladder = [(0, lv, lv) for lv in levels] + [(1, half, half), (2, 0.0, half)]
        fp_eval = outline(fp_local) if len(fp_local) > 40 else fp_local
        full_reg = container_region(dest) if (future and reg["kind"] == "in") else reg
        cands = []
        for tier, lv, lv_other in ladder:
            if stack:
                if tier < 2:
                    cs_ = manip.rest_candidates(gx, prof_xy, prof_b, g_xy, sup, np.concatenate([obst, touch]), yaws,
                                                lv, target_xy=target_xy, base_z=reg["floor_z"])
                else:
                    cs_ = manip.rest_candidates(gx, prof_xy, prof_b, g_xy, sup, obst, yaws, lv_other,
                                                target_xy=target_xy, base_z=reg["floor_z"], need_support=0.0,
                                                touch_pts=touch)
            else:
                cs_ = placement_candidates(reg, fp_eval, pay_r, yaws, scene_block_xy=block, target_xy=target_xy,
                                           need_clear=lv, edge_ok=reg["kind"] == "on", max_cands=400 if future else 40,
                                           block_clear=lv_other)
            if future:
                cs_ = reserve_space(full_reg, cs_, fp_local, future, yaws, block, max(lv, lv_other))
                good = [c for c in cs_ if c.get("future_clearance", 1.0) >= max(lv, lv_other)]
                cs_ = good if good else (cs_ if tier == 2 else [])
            seen = [(c["xy"], c["yaw"]) for c in cands]
            for c in cs_:
                c["level"], c["tier"] = lv, tier
            cands += [c for c in cs_ if all(np.linalg.norm(c["xy"] - xy) > 0.004 or abs(c["yaw"] - y) > 0.2
                                            for xy, y in seen)]
            if len(cands) >= 12:
                break
        cands.sort(key=lambda c: (c["tier"], -c["score"]))
        cands = [c for c in cands if all(np.linalg.norm(c["xy"] - f) > r for f, r in failed)]
        self.log("place_cands", n=len(cands), top=cands[:3], region_cells=len(gx), floor=reg["floor_z"],
                 rim=reg["rim_z"], payload_r=pay_r, n_block=len(obst) if stack else 0 if block is None else len(block),
                 stack=stack, contents=[self.objects[o].name for o in contents],
                 future=len(future), max_region_clear=float(reg["clearance"].max()) if len(reg["clearance"]) else 0.0,
                 tiers=dict(Counter(c["tier"] for c in cands)))
        q_now = self.rb.q()[:7]
        finger = self._finger()
        f_release = min(0.04, finger + 0.009)
        # the support surface under the region is the intended contact: ignored by the clearance field
        box0 = (lo - 0.02, hi + 0.02, reg["floor_z"] + 0.008)
        self._place_ignore = box0
        self._place_excl = (dest.oid, obj.oid)

        def ignore_for(cand):
            # a stacked payload rests on what lies under its own footprint, up to its rest height
            if not cand.get("stacked"):
                return box0
            return (cand["xy"] - pay_r - 0.01, cand["xy"] + pay_r + 0.01, cand["rest_z"] + 0.008)
        if reg["kind"] == "in":
            top_level = max(0.02, reg["rim_z"] - reg["floor_z"] + 0.03)
            drop_levels = list(np.arange(0.006, top_level + 1e-6, 0.015))
        else:
            drop_levels = [0.006, 0.015, 0.025]
        if drop:
            # a drop (the instruction says drop): let go 3 cm above the surface, lower only where the hand cannot be
            drop_levels = [0.03] + [d for d in drop_levels[::-1] if d < 0.03]
        hand_hold = self._hand_local(finger)
        hand_rel = self._hand_local(f_release)
        pts_site = self.checker.attached.pts_site
        def solve(pool):
            """Hand / payload stage, then arm stage, over a pool of footprint candidates: (plan, None), or
            (None, why nothing in the pool worked)."""
            ranked, pre_why, pre_blame = [], {}, {}
            for cand in pool:
                self._place_ignore = ignore_for(cand)
                Rz = rot_z(cand["yaw"])
                R_place = Rz @ Rs
                site_xy = cand["xy"] + (Rz[:2, :2] @ (ps[:2] - pay_c))
                why = None
                # the margin ladder per candidate: from its footprint tier down to touching what may be touched
                for tier in range(cand["tier"], n_tiers):
                    self._set_tier(tier, cand["level"])
                    best = None
                    for dz in drop_levels:
                        z_b = cand.get("rest_z", reg["floor_z"]) + dz
                        p_place = np.r_[site_xy, z_b + (ps[2] - bottom)]
                        lift = max(0.05, reg["rim_z"] + 0.025 - z_b)
                        sweep = [p_place + np.array([0, 0, h_]) for h_ in np.linspace(0, lift, 4)]
                        why = None
                        for pz in sweep:
                            why = self._hand_block(R_place, pz, hand_hold, pts_site, reg, tol)
                            if why:
                                break
                        if why is None:
                            why = self._hand_block(R_place, p_place, hand_rel, None, reg, tol, release=True)
                            why = None if why is None else "release_" + why
                        if why is None:
                            # tier 3: how much of the destination's side hand and payload are predicted to touch there
                            # orders the candidates. The release height stays the lowest one that touches nothing that
                            # counts: the object is aimed at the floor and contact ends the descent
                            touch_ = round(self._soft_touch(R_place, p_place, hand_hold, pts_site, reg), 2) if tier == 3 else 0.0
                            best = (tier, touch_, dz, -cand["score"], len(ranked), cand, R_place, p_place, lift, sweep)
                            break
                    if best is not None:
                        ranked.append(best)
                        why = None
                        break
                if why is not None:
                    pre_why[why] = pre_why.get(why, 0) + 1
                    if why not in pre_blame and len(pre_blame) < 3 and getattr(self, "_block_pts", None) is not None:
                        pre_blame[why] = self.nearest_obstacle(self._block_pts, self._place_excl)
            ranked.sort(key=lambda r: (r[0], r[1], -r[2] if drop else r[2], r[3]))
            self.log("place_precheck", n=len(ranked), drops=[r[2] for r in ranked[:6]], rejected=pre_why,
                     blame=pre_blame, tiers=dict(Counter(r[0] for r in ranked)),
                     soft_touch=[r[1] for r in ranked if r[0] == 3][:6])
            ik_why, ik_blame, tried, t_ik = {}, {}, [], time.time()
            for tier, touch_, dz, _, _, cand, R_place, p_place, lift, sweep in ranked:
                if len(tried) >= 16 or time.time() - t_ik > 15.0:
                    break
                # a near-duplicate of a candidate that already failed would fail the same way
                if any(abs(cand["yaw"] - y) < 1e-3 and np.linalg.norm(cand["xy"] - xy) < 0.008 for y, xy in tried):
                    continue
                tried.append((cand["yaw"], cand["xy"]))
                self._place_ignore = ignore_for(cand)
                self._set_tier(tier, cand["level"])
                p_pre = p_place + np.array([0, 0, lift])
                sols = self.ik_clear(R_place, p_pre, q_now, exclude=(dest.oid,), finger=finger, n_random=8)
                why = "pre_pose_no_ik" if len(sols) == 0 else None
                for q_pre in sols[:3]:
                    line = self.line_configs(R_place, p_place, q_pre, finger=finger)
                    if line is None:
                        why = why or "descent_no_ik"
                        continue
                    bad = None
                    for q in list(line[::3]) + [line[-1]]:
                        bad = self._place_config_bad(q, reg, finger, tol)
                        if bad:
                            break
                    if bad:
                        why = "descent_" + bad
                        if why not in ik_blame and len(ik_blame) < 3:
                            ik_blame[why] = self._blame_config(q, bad, finger)
                        continue
                    # does it pass only because unobserved space was no obstacle? Then the caller looks first
                    self._place_unknown = True
                    unknown_near = bool(any(self._hand_block(R_place, pz, hand_hold, pts_site, reg, tol) for pz in sweep)
                                        or self._hand_block(R_place, p_place, hand_rel, None, reg, tol, release=True)
                                        or any(self._place_config_bad(q, reg, finger, tol)
                                               for q in list(line[::3]) + [line[-1]]))
                    self._place_unknown = False
                    self._tick("place_plan", t)
                    self.log("place_plan", xy=cand["xy"], yaw=cand["yaw"], clearance=cand["clearance"],
                             drop=dz, future_clearance=cand.get("future_clearance"), tried=len(tried), rejected=ik_why,
                             tier=tier, footprint_tier=cand["tier"], unknown_near=unknown_near,
                             soft_touch=touch_ if tier == 3 else None, inside=cand.get("inside"), lean=cand.get("lean"))
                    cand = dict(cand, drop=dz, f_release=f_release, tier=tier, unknown_near=unknown_near)
                    return (cand, line[0], line[-1], reg, R_place, p_place), None
                ik_why[why] = ik_why.get(why, 0) + 1
            return None, {"candidates": len(pool), "passed_hand_payload_check": len(ranked),
                          "hand_payload_rejections": pre_why, "hand_payload_nearest": pre_blame,
                          "arm_plans_tried": len(tried), "arm_plan_rejections": ik_why, "arm_nearest": ik_blame}
        plan, diag = solve(cands[:24])
        if plan is None and v1:
            # rank, don't reject (tier 3), only when nothing above worked: every cell of the region is a candidate;
            # how well the payload fits the destination (inside the opening, off its walls, a stable rest) orders
            # them and rules none out. Only what the payload may not touch does
            if stack:
                soft = manip.rest_candidates(gx, prof_xy, prof_b, g_xy, sup, obst, yaws, 0.0, target_xy=target_xy,
                                             base_z=reg["floor_z"], need_support=0.0, touch_pts=touch, soft=True,
                                             max_cands=24)
            else:
                soft = soft_candidates(reg, fp_eval, yaws, block_hard, target_xy=target_xy, max_cands=24,
                                       step_m=0.005 if len(gx) <= 400 else 0.01)
            for c in soft:
                c["level"], c["tier"] = 0.0, 3
            soft = [c for c in soft if all(np.linalg.norm(c["xy"] - f) > r for f, r in failed)]
            self.log("place_cands_soft", n=len(soft), top=soft[:3])
            if soft:
                plan, more = solve(soft[:16])
                if plan is None:
                    for k in ("candidates", "passed_hand_payload_check", "arm_plans_tried"):
                        diag[k] += more[k]
                    for k in ("hand_payload_rejections", "arm_plan_rejections"):
                        for w, n in more[k].items():
                            diag[k][w] = diag[k].get(w, 0) + n
                    for k in ("hand_payload_nearest", "arm_nearest"):
                        diag[k] = {**more[k], **diag[k]}
        if plan is not None:
            return plan
        self._place_diag = diag
        self.log("place_ik_fail", **self._place_diag)
        self._tick("place_plan", t)
        return None

    def _on_contents(self, dest, obj):
        """Objects resting on a destination: footprint centre over its footprint, bottom above its bottom."""
        if dest.hm is None or len(dest.points) == 0:
            return []
        fp = ndimage.binary_dilation(np.isfinite(dest.hm), iterations=2)
        z0 = float(dest.points[:, 2].min())
        out = []
        for m in self.objects.values():
            if m.oid in (dest.oid, obj.oid) or m.held or len(m.points) == 0 or getattr(m, "virtual", False):
                continue
            ij = np.floor((m.footprint_center() - dest.lo_xy) / dest.cell).astype(int)
            if (0 <= ij[0] < fp.shape[0] and 0 <= ij[1] < fp.shape[1] and fp[ij[0], ij[1]]
                    and float(m.points[:, 2].min()) > z0 + 0.005):
                out.append(m.oid)
        return out

    def _on_obstacles(self, dest, obj, contents, sup, lo, hi):
        """Everything else near an 'on' placement as surface points: other objects, and mapped columns (voxel centre
        height) farther than 1.5 cm from the support (unsegmented skin of the support itself)."""
        skip = (dest.oid, obj.oid) + tuple(contents)
        pts = [m.points for m in self.objects.values()
               if m.oid not in skip and not m.held and len(m.points) and not getattr(m, "virtual", False)]
        P = np.concatenate(pts) if pts else np.zeros((0, 3))
        P = P[np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), axis=1)]
        hmap, org = self.scene.heightmap((lo[0], hi[0]), (lo[1], hi[1]), exclude=skip)
        ij = np.argwhere(np.isfinite(hmap))
        if len(ij):
            col = np.c_[np.array(org) + (ij + 0.5) * self.scene.res, hmap[ij[:, 0], ij[:, 1]] - 0.5 * self.scene.res]
            d, _ = cKDTree(sup[:, :2]).query(col[:, :2])
            P = np.concatenate([P, col[d > 0.015]])
        return P

    def landing_check(self, dest, region, R_place, p_place):
        """'in': before going to the planned placement, the wrist camera looks down from 30 cm above the planned
        landing spot and measures, with its depth, the surfaces just around the payload's planned footprint: a ring
        1.5 cm outside it at the container's floor height (pixels hidden by the robot or the held payload are skipped).
        A container's floor lies above the table it stands on, so ring pixels that see the table (within 1.2 cm of its
        plane) mean part of the payload would hang outside the container ('outside': this spot is dropped and the
        placement planned again). The table plane is used, not the estimated floor, which can be several cm too high;
        containers whose floor is not clearly above the table are not tested. Too little visible ring or no view:
        'unclear' (placed as planned; VERIFY judges at the end, never a REPAIR from here). Colour segmentation is not
        used: seen from above it often misses the container."""
        out = {"verdict": "unclear"}
        table = getattr(self, "table_z_visual", self.table_z)          # the surface depth sees, not the resting plane
        if region["floor_z"] - table < 0.02:
            out["why"] = "container floor not clearly above the table"
            self.log("landing_check", **out)
            return out
        pay = p_place + self.checker.attached.pts_site @ R_place.T
        c = 0.5 * (pay[:, :2].min(0) + pay[:, :2].max(0))
        r = float(np.linalg.norm(pay[:, :2] - c, axis=1).max()) + 0.015
        fr = self.look_from_above(np.r_[c, region["floor_z"]], height=0.30)
        if fr is None:
            out["why"] = "no view from above"
            self.log("landing_check", **out)
            return out
        H, W = fr.depth.shape
        a = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
        ring = np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(len(a), region["floor_z"])]
        uv, z = fr.project(ring)
        ok = (z > 0.03) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        u, v = uv[:, 0].astype(int).clip(0, W - 1), uv[:, 1].astype(int).clip(0, H - 1)
        hidden = fr.robot_mask | project_mask(fr, self._payload_world(*self.rb.eef_pose()), radius_px=4)
        d = fr.depth[v, u]
        vis = ok & ~hidden[v, u] & np.isfinite(d) & (d > 0.02)
        n = int(vis.sum())
        frac = 0.0
        if n:
            dz, uu, vv = d[vis], u[vis], v[vis]
            pc = np.stack([(uu + 0.5 - fr.K[0, 2]) / fr.K[0, 0] * dz, (vv + 0.5 - fr.K[1, 2]) / fr.K[1, 1] * dz, dz], 1)
            seen_z = (pc @ fr.T_wc[:3, :3].T + fr.T_wc[:3, 3])[:, 2]
            frac = float(np.mean(seen_z < table + 0.012))                  # the table: outside the container
        out.update(visible=n, on_table=round(frac, 2), radius=round(r, 3))
        if n >= 10 and frac >= 0.25:
            out["verdict"] = "outside"
        elif n >= 10 and frac < 0.1:
            out["verdict"] = "inside"
        self.log("landing_check", **out)
        return out

    def _top_region(self, dest):
        """An 'on' destination as a drop opening: its top-view footprint as one compartment, floor at its flat top."""
        top = support_region(dest)["floor_z"]
        interior = np.isfinite(dest.hm)
        gxy = dest.lo_xy + (np.argwhere(interior) + 0.5) * dest.cell
        clear = (ndimage.distance_transform_edt(interior) * dest.cell)[interior]
        return {"floor_z": top, "rim_z": top, "grid_xy": gxy, "clearance": clear, "kind": "on", "interior": interior,
                "compartments": interior.astype(int), "lo_xy": dest.lo_xy, "cell": dest.cell}

    def plan_drop_in(self, obj, dest, place_pt, failed=(), rel="in"):
        """A drop: put the object over the opening and let it go just above the rim, the hand in free space (lower,
        just above whatever is under it, only where the hand cannot be there). Any resulting pose counts (user 2026-11-20: "we just don't
        care about the can's pose, just put it over there"). Only where the container is open from above.
        The payload may touch the container and what is inside it, the hand may touch the container (neither counts
        as a collision); the hand stays clear of the objects inside, the arm and payload of everything else."""
        t = time.time()
        if len(dest.points) == 0:
            return None
        reg = self._top_region(dest) if rel == "on" else container_region(dest)
        if len(reg["grid_xy"]) == 0:
            return None
        interior, lo_xy, cell = reg["interior"], reg["lo_xy"], reg["cell"]

        def inside(xy):
            ij = np.floor((np.atleast_2d(xy) - lo_xy) / cell).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < interior.shape[0]) & (ij[:, 1] < interior.shape[1])
            out = np.zeros(len(ij), bool)
            out[ok] = interior[ij[ok, 0], ij[ok, 1]]
            return out
        Rs, _ = self.rb.eef_pose()
        contents = [m.oid for m in self.objects.values()
                    if m.oid not in (dest.oid, obj.oid) and not m.held and len(m.points)
                    and not getattr(m, "virtual", False) and inside(m.footprint_center())[0]]
        # compartments: the parts of the opening that the container's walls and internal dividers separate
        comp_map = reg["compartments"]

        def comp(xy):
            ij = np.floor((np.atleast_2d(xy) - lo_xy) / cell).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < interior.shape[0]) & (ij[:, 1] < interior.shape[1])
            out = np.zeros(len(ij), int)
            out[ok] = comp_map[ij[ok, 0], ij[ok, 1]]
            return out
        under = np.concatenate([dest.points] + [self.objects[o].points for o in contents])
        under_comp = comp(under[:, :2])
        # where a dropped object comes to rest: on the highest surface under its footprint. Height map of what lies
        # under (the container and its contents) on a 2.5 mm grid, each cell holding the highest surface within one
        # cell of it (the footprint grows by about the 3 mm minimum clearance)
        HG = 0.0025
        hg_lo = under[:, :2].min(0) - 0.1
        hg = np.full(np.ceil((under[:, :2].max(0) + 0.1 - hg_lo) / HG).astype(int) + 1, -np.inf)
        gi = np.floor((under[:, :2] - hg_lo) / HG).astype(int)
        np.maximum.at(hg, (gi[:, 0], gi[:, 1]), under[:, 2])
        hg = ndimage.maximum_filter(hg, size=3)

        def landing_height(xy):
            q = np.floor((xy - hg_lo) / HG).astype(int)
            q = q[(q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < hg.shape[0]) & (q[:, 1] < hg.shape[1])]
            return float(hg[q[:, 0], q[:, 1]].max()) if len(q) else -np.inf
        under_tree = cKDTree(under[:, :2])
        dest_tree = cKDTree(dest.points)
        grid_comp = comp(reg["grid_xy"])
        n_comp = len(np.unique(grid_comp[grid_comp > 0]))
        target = None
        if place_pt is not None and n_comp:
            # the compartment S1 pointed at (or the nearest one): the object goes there and nowhere else
            pp = np.asarray(place_pt[:2], float)
            target = int(comp(pp)[0])
            if target == 0:
                # S1's pixel often lands on the wall behind a deep, narrow compartment: the nearest compartment, and
                # S1's point moved into it
                live = np.flatnonzero(grid_comp > 0)
                j = live[int(np.argmin(np.linalg.norm(reg["grid_xy"][live] - pp, axis=1)))]
                target, pp = int(grid_comp[j]), reg["grid_xy"][j].copy()
        gsel = reg["grid_xy"][grid_comp > 0] if n_comp else reg["grid_xy"]
        # one aim per 1.5 cm cell of the opening: its point farthest from the walls (in a slot only a few cells wide
        # that is its middle line, which a corner pick misses)
        depth = ndimage.distance_transform_edt(comp_map > 0 if n_comp else interior)
        gij = np.floor((gsel - lo_xy) / cell).astype(int)
        keys = np.floor(gsel / 0.015).astype(int)
        order = np.lexsort((-depth[gij[:, 0], gij[:, 1]], keys[:, 1], keys[:, 0]))
        _, first = np.unique(keys[order], axis=0, return_index=True)
        grid = gsel[order[first]]
        if target is not None:
            # aim: S1's point and small offsets around it first, then the rest of that compartment
            grid = grid[comp(grid) == target]
            grid = grid[np.argsort(np.linalg.norm(grid - pp, axis=1))]
            offs = [np.array([dx, dy]) for dx in (-0.01, -0.005, 0.0, 0.005, 0.01) for dy in (-0.01, -0.005, 0.0, 0.005, 0.01)]
            aims = [pp + o for o in sorted(offs, key=np.linalg.norm)] + list(grid)
        else:
            # aim: the middle of the opening, the tops of the objects already inside, then the rest of the opening on
            # a 1.5 cm grid from the middle outward (tall neighbours can block the wide palm)
            mid = reg["grid_xy"][int(np.argmax(reg["clearance"]))]
            grid = grid[np.argsort(np.linalg.norm(grid - mid, axis=1))]
            aims = ([] if place_pt is None else [np.asarray(place_pt[:2], float)]) + [mid] \
                + [self.objects[o].footprint_center() for o in contents] + list(grid)
        finger = self._finger()
        hand_hold, hand_open = self._hand_local(finger), self._hand_local(manip.FULL_JOINT)
        pts_site = self.checker.attached.pts_site
        excl, excl_pay = (dest.oid, obj.oid), (dest.oid, obj.oid) + tuple(contents)
        # what lies inside the container up to just above its rim (its liner, the objects in it, also where they are
        # unlabeled surface) may be touched by the payload; the hand (above the rim) and the arm are not exempted
        box = (dest.points[:, :2].min(0), dest.points[:, :2].max(0), reg["rim_z"] + 0.01)
        q_now = self.rb.q()[:7]

        def hand_bad(R, p, hand, margin):
            L, grp, rad = hand
            Wh = p + L @ R.T
            c = self.checker.point_clearance(Wh, excl) - rad              # objects inside count (and all else)
            lim = np.where(grp == 1, margin * NEAR["palm"], margin * NEAR["finger"])
            if (c < lim).any():
                self._block_pts = Wh[c < lim]
                return "hand_object"
            if (Wh[:, 2] - self.table_z - 0.5 * rad).min() < NEAR["plane_finger"]:
                return "hand_table"
            if (dest_tree.query(Wh)[0] - rad).min() < 0.0:                 # may touch the container, not pass into it
                return "hand_into_container"
            return None
        why_n, blame, tried = {}, {}, {"upright": 0, "tilted": 0}

        def note(why, pts=None, extra=None):
            why_n[why] = why_n.get(why, 0) + 1
            if why not in blame and len(blame) < 6 and why != "footprint_outside_opening":
                b = dict(extra or {})
                if pts is not None and len(pts):
                    b["nearest"] = self.nearest_obstacle(pts, excl)
                blame[why] = b
        # hand orientations: upright (the held hand turned about the vertical; the 20 cm palm's direction matters),
        # then tilted: leaned 20 / 35 deg toward 8 directions at 4 turns. Leaning moves the wrist and forearm sideways,
        # away from what stands beside the container; the payload leans with the hand (its final pose does not matter)
        # A hand turned by 180 deg is the same hand only when the payload hangs centred on the grip axis; a payload held
        # off the axis (a bowl by its rim) swings round the hand, so the turns cover the full circle then: the usual
        # half circle first (same order as for a centred payload), the other half (a half-turned hand, a long wrist
        # swing in transit) only after it
        Wn = pts_site @ Rs.T
        off_axis = float(np.linalg.norm(0.5 * (Wn[:, :2].min(0) + Wn[:, :2].max(0))))
        full_turn = off_axis > 0.01
        arc = np.arange(-90, 90, 15)
        # smallest wrist turn first: in free space above the rim every turn passes the checks, so the first one listed
        # is the one used, and a large turn (-90 deg) is the one TRACE most often finds no path to
        turn = lambda y: abs((y + np.pi) % (2 * np.pi) - np.pi)
        upright = sorted([(rot_z(y) @ Rs, float(y), 0.0, None)
                          for y in np.deg2rad(np.r_[arc, arc + 180] if full_turn else arc)], key=lambda o: turn(o[1]))
        tilted = sorted([(rotvec_to_mat(tl * np.array([-np.sin(az), np.cos(az), 0.0])) @ rot_z(y) @ Rs, float(y),
                          float(tl), float(az))
                         for tl in np.deg2rad((20.0, 35.0)) for az in np.deg2rad(np.arange(0, 360, 45))
                         for y in np.deg2rad(np.arange(0, 360, 45) if full_turn else (0, 45, 90, 135))],
                        key=lambda o: (o[2], turn(o[1])))
        for label, orients, aim_set, max_tried, t_max in (("upright", upright, aims, 16, 25.0),
                                                          ("tilted", tilted, aims[:12], 24, 60.0)):
            for aim in aim_set:
                if tried[label] >= max_tried or time.time() - t > t_max:
                    break
                if any(np.linalg.norm(aim - f) <= min(r, 0.008) for f, r in failed):
                    continue                                  # its transit or landing already failed: shift a little
                for R_place, yaw, tilt, az in orients:
                    tag = {"aim": np.round(aim, 3).tolist(), "yaw": round(yaw, 2), "tilt": round(tilt, 2),
                           "toward": None if az is None else round(az, 2)}
                    W0 = pts_site @ R_place.T                                     # payload about the grip site
                    c0 = 0.5 * (W0[:, :2].min(0) + W0[:, :2].max(0))
                    fpw = W0[:, :2] - c0
                    # it falls in when its centre and most of its footprint are over the opening (a tilted payload,
                    # or an over-estimated, mirrored payload model, may reach over the rim)
                    k = int(comp(aim)[0])
                    fc = comp(aim + fpw)
                    if k == 0 or (target is not None and k != target) or (fc == k).mean() < 0.75 \
                            or ((fc > 0) & (fc != k)).mean() > 0.02:
                        # centre over the (requested) compartment, most of the footprint in it, none over another one
                        note("footprint_outside_opening")
                        continue
                    ball = under_tree.query_ball_point(aim, float(np.linalg.norm(fpw, axis=1).max()))
                    near = [i for i in ball if under_comp[i] == k]               # what lies in this compartment
                    rest_lo = max(reg["floor_z"], float(under[near, 2].max()) if near else reg["floor_z"])
                    rest_hi = max(reg["floor_z"], float(under[ball, 2].max()) if ball else reg["floor_z"])
                    # predicted landing: the highest surface under the whole footprint must be what lies in this
                    # compartment (its floor or the objects in it), never a wall, rim or divider top
                    land = landing_height(aim + fpw)
                    if rel == "on":
                        # no walls on a support: let go over the highest thing under the footprint, whatever it is
                        rest_lo = rest_hi = max(reg["floor_z"], land)
                    elif land > rest_lo + manip.STEP_OVER:
                        note("landing_on_wall", None, {**tag, "above_support_m": round(land - rest_lo, 3)})
                        continue
                    # let go with the payload's bottom 3 cm above the rim: the hand stays in free space and the object
                    # falls onto what lies under it in this compartment. Where the hand cannot be there (an overhang
                    # above the opening), 2 cm above what lies under it, then 2 cm above everything within the
                    # payload's reach (the rim, divider tops)
                    heights = []
                    for z_b, rest_z in [(reg["rim_z"] + 0.03, rest_lo), (rest_lo + 0.02, rest_lo)] \
                            + ([(rest_hi + 0.02, rest_hi)] if rest_hi > rest_lo + 0.01 else []):
                        if z_b >= rest_z + 0.02 - 1e-6 and all(abs(z_b - z) > 0.01 for z, _ in heights):
                            heights.append((z_b, rest_z))
                    for z_b, rest_z in heights:
                        p_place = np.r_[aim - c0, z_b - float(W0[:, 2].min())]
                        lift = max(0.05, reg["rim_z"] + 0.03 - z_b)
                        why = None
                        for h_ in np.linspace(0.0, lift, 5):
                            pz = p_place + np.array([0, 0, h_])
                            why = hand_bad(R_place, pz, hand_hold, 1.0)
                            if why is None:
                                cp = self.checker.point_clearance(pz + W0, excl_pay, box) - 0.004
                                why = "payload" if cp.min() < NEAR["payload"] else None
                            if why:
                                break
                        why = why or hand_bad(R_place, p_place, hand_open, 0.0)    # the binary release must not touch them
                        if why:
                            note(why, self._block_pts if why == "hand_object" else None, tag)
                            continue
                        tried[label] += 1
                        p_pre = p_place + np.array([0, 0, lift])
                        sols = self.ik_clear(R_place, p_pre, q_now, exclude=(dest.oid,), finger=finger, n_random=8)
                        for q_pre in sols[:3]:
                            line = self.line_configs(R_place, p_place, q_pre, finger=finger)
                            if line is None:
                                continue
                            bad = pts_bad = None
                            for q in list(line[::3]) + [line[-1]]:
                                good, c = self.checker.ok(q[None], NEAR, exclude=excl_pay, finger=finger, ignore_box=box)
                                W, att = self.checker.world_points(q[None], finger)
                                if not good[0]:
                                    bad = next((k for k, v in c.items() if v[0] < NEAR.get(k, 0.02)), "clearance")
                                    key = bad.replace("plane_", "")
                                    pts_bad = att[0] if key == "payload" and att is not None else \
                                        W[0][self.checker.pg == {"arm": 0, "palm": 1, "finger": 2}.get(key, 0)]
                                    break
                                arm = W[0][self.checker.pg == 0]
                                d_arm = dest_tree.query(arm)[0] - self.checker.pr[self.checker.pg == 0]
                                if d_arm.min() < 0.01:                             # the arm stays off the container
                                    bad, pts_bad = "arm_container", arm[d_arm < 0.01]
                                    break
                            if bad:
                                note(f"{label}_descent_{bad}", pts_bad, tag)
                                continue
                            self._tick("place_plan", t)
                            cand = {"xy": aim, "yaw": yaw, "tilt": tilt, "clearance": 0.0, "drop": round(z_b - rest_z, 3),
                                    "rest_z": rest_z,
                                    "f_release": manip.FULL_JOINT, "drop_in": True,
                                    "compartment_xy": None if target is None else reg["grid_xy"][grid_comp == target]}
                            self.log("drop_in_plan", xy=aim, yaw=yaw, tilt=tilt, toward=tag["toward"], rest_z=rest_z,
                                     release_z=z_b, above_rim=bool(z_b > reg["rim_z"]),
                                     floor=reg["floor_z"], rim=reg["rim_z"], contents=contents, tried=tried,
                                     rejected=why_n, blame=blame, compartments=n_comp, compartment=target,
                                     compartment_aim=None if target is None else pp, landing_z=land,
                                     full_turn=full_turn, off_axis=off_axis)
                            return cand, line[0], line[-1], reg, R_place, p_place
                        if len(sols) == 0:
                            note(f"{label}_pre_pose_no_ik", None, tag)
                        if tried[label] >= max_tried or time.time() - t > t_max:
                            break
                    if tried[label] >= max_tried or time.time() - t > t_max:
                        break
        self._place_diag = {"drop_in": True, "aims": len(aims), "tried": tried, "contents": contents,
                            "rejected": why_n, "blame": blame, "compartments": n_comp, "compartment": target,
                            "full_turn": full_turn, "off_axis": round(off_axis, 3)}
        self.log("drop_in_fail", **self._place_diag)
        self._tick("place_plan", t)
        return None

    def _hand_local(self, finger):
        """Palm + finger collision points in the grip-site frame for a given finger opening."""
        q = self.rb.q()[:7]
        qf = self.rm.full_q(q[None], finger)
        W = self.rm.points_world(qf)[0]
        Rs, ps = self.rm.site_pose(qf)
        sel = self.checker.pg >= 1
        return ((W[sel] - ps[0]) @ Rs[0], self.checker.pg[sel], self.checker.pr[sel])

    def _hand_block(self, R, p, hand, pts_site, reg, tol, release=False):
        """Why the hand (+ payload) posed at (R, p) is too close to something, or None when clear.
        The offending world points are kept in self._block_pts (for failure reports). Margins scale with the ladder
        tier self._tier; at tier 2 the hand may touch the destination, the payload also what is in/on it and the other
        task objects (self._place_excl_pay). self._place_unknown False: unobserved space is no obstacle."""
        tier = getattr(self, "_tier", 0)
        k = TIER_SCALE[tier]
        unk = getattr(self, "_place_unknown", True)
        # tier 3 (v1): no margin, only no predicted contact, and the destination's fixture is no obstacle to the hand
        # or the payload (its surface is taken out through checker.free_near, its objects here)
        fix = tuple(getattr(self, "_fix_oids", ())) if tier == 3 else ()
        L, grp, rad = hand
        Wh = p + L @ R.T
        self._block_pts = None
        c = self.checker.point_clearance(Wh, self._place_excl + fix, self._place_ignore, unknown=unk) - rad
        if c[grp == 1].min() < k * NEAR["palm"]:
            self._block_pts = Wh[grp == 1]
            return "palm"
        if c[grp == 2].min() < (0.0 if release else k * NEAR["finger"]):
            self._block_pts = Wh[grp == 2]
            return "finger"
        zc = Wh[:, 2] - self.table_z - 0.5 * rad
        if zc[grp == 1].min() < k * NEAR["plane_palm"] or zc[grp == 2].min() < k * NEAR["plane_finger"]:
            return "hand_table"
        tree = reg.get("_tree3")
        walls = reg.get("wall_pts")
        if walls is not None and len(walls):
            if tree is None:
                tree = reg["_tree3"] = cKDTree(walls)
            if tier < 2:
                d, _ = tree.query(Wh)
                d -= rad
                if d[grp == 1].min() < k * 0.006 or d[grp == 2].min() < k * 0.002:
                    return "hand_destination_wall"
        if pts_site is not None:
            Wp = p + pts_site @ R.T
            excl = (getattr(self, "_place_excl_pay", self._place_excl) if tier >= 2 else self._place_excl) + fix
            cp = self.checker.point_clearance(Wp, excl, self._place_ignore, unknown=unk) - 0.004
            if cp.min() < k * NEAR["payload"]:
                self._block_pts = Wp[cp < k * NEAR["payload"]]
                return "payload"
            if tree is not None and tier < 2:
                dp, _ = tree.query(Wp)
                if dp.min() < getattr(self, "_need", 0.004) - 0.001:
                    return "payload_destination_wall"
        return None

    def future_footprints(self, dest, obj):
        """Footprints of objects that later stages still have to put into the same destination."""
        out = []
        if not self.program:
            return out
        started = False
        for st in self.program.get("stages", []):
            if st.get("op") != "transfer":
                continue
            o = self.objects.get(self.id_map.get(st["object"]))
            if o is None:
                continue
            if o.oid == obj.oid:
                started = True
                continue
            if started and self.id_map.get(st.get("reference")) == dest.oid and len(o.points):
                fp = o.points[o.points[:, 2] > o.support_z + 0.004][:, :2]
                c = 0.5 * (fp.min(0) + fp.max(0))
                out.append(outline(fp - c))
        return out

    def _place_config_bad(self, q, reg, finger, tol, release=False):
        """Why a placing configuration is not clear (clearance group / destination wall), or None. Margins follow
        the ladder tier (the arm keeps NEAR); at tier 2 hand and payload may touch what the scoring rule allows."""
        tier = getattr(self, "_tier", 0)
        k = TIER_SCALE[tier]
        unk = getattr(self, "_place_unknown", True)
        fix = tuple(getattr(self, "_fix_oids", ())) if tier == 3 else ()
        margins = {g: (v if g.endswith("arm") else k * v) for g, v in NEAR.items()}
        if tier >= 2:
            margins["payload"] = -np.inf             # checked below, against all but the allowed contacts
        if tier == 3:
            # the arm keeps NEAR from everything but the destination, its fixture included (an arm contact is not
            # sensed); only hand and payload are measured against what counts alone
            arm_m = {g: (v if g.endswith("arm") else -np.inf) for g, v in NEAR.items()}
            prev, self.checker.free_near = self.checker.free_near, self._free_tiers[0]
            try:
                good, c = self.checker.ok(q[None], arm_m, exclude=self._place_excl[:1], finger=finger,
                                          ignore_box=self._place_ignore, unknown=unk)
            finally:
                self.checker.free_near = prev
            if not good[0]:
                return "arm" if c["arm"][0] < NEAR["arm"] else "plane_arm"
            margins["arm"] = margins["plane_arm"] = -np.inf
        good, c = self.checker.ok(q[None], margins, exclude=self._place_excl[:1] + fix, finger=finger,
                                  ignore_box=self._place_ignore, unknown=unk)
        if not good[0]:
            fails = [g for g, v in c.items() if v[0] < margins.get(g, 0.02)]
            return fails[0] if fails else "clearance"
        W, att = self.checker.world_points(q[None], finger)
        if tier >= 2 and att is not None and not release:
            ca = self.checker.point_clearance(att[0], getattr(self, "_place_excl_pay", self._place_excl) + fix,
                                              self._place_ignore, unknown=unk) - 0.004
            if ca.min() < k * NEAR["payload"]:
                return "payload"
        walls = reg.get("wall_pts")
        if walls is None or len(walls) == 0:
            return None
        tree = reg.get("_tree3")
        if tree is None:
            tree = reg["_tree3"] = cKDTree(walls)
        dW, _ = tree.query(W[0])
        dW -= self.checker.pr
        if dW[self.checker.pg == 0].min() < 0.01 or (tier < 2 and (dW[self.checker.pg == 2].min() < k * 0.002
                                                                   or dW[self.checker.pg == 1].min() < k * 0.006)):
            return "destination_wall"
        if att is not None and not release and tier < 2:
            dA, _ = tree.query(att[0])
            if dA.min() < getattr(self, "_need", 0.004) - 0.001:
                return "payload_destination_wall"
        return None

    def _place_config_ok(self, q, reg, finger, tol, release=False):
        return self._place_config_bad(q, reg, finger, tol, release) is None

    def _blame_config(self, q, why, finger):
        """Nearest mapped thing to the robot part (or payload) that failed a clearance check."""
        W, att = self.checker.world_points(np.asarray(q)[None, :7], finger)
        key = why.replace("plane_", "")
        grp = {"arm": 0, "palm": 1, "finger": 2}
        if key.startswith("payload") and att is not None:
            pts = att[0]
        elif key in grp:
            pts = W[0][self.checker.pg == grp[key]]
        else:
            pts = W[0]
        return self.nearest_obstacle(pts, self._place_excl)

    def check_relation(self, obj, dest, rel, region, place_pt, tol, any_pose=False, compartment_xy=None,
                       second_view=False, rest_z=None):
        """Look at the result and measure the relation from fresh depth (local VERIFY).
        any_pose: 'in' holds whenever the object is over the opening (stacked, leaning or lying all count).
        compartment_xy: cells of the compartment a drop-in aimed at; the object must have ended up in it.
        second_view: look from the head camera's side instead of from above (close, so the arm carrying the wrist
        camera hides nothing), and segment with all of S1's prompts for the object."""
        prompts = [obj.category, obj.name]
        if second_view:
            c3 = np.r_[dest.footprint_center(), max(dest.top_z, obj.top_z)]
            fr = self.look_toward(c3, c3 - self.head0.T_wc[:3, 3], dist=0.30)
            if fr is None:
                self.retreat_home()
            prompts = list(dict.fromkeys(prompts + list(obj.prompts or [])))
        else:
            fr = self.look_from_above(np.r_[dest.footprint_center(), max(dest.top_z, obj.top_z)], height=0.32)
        frames = [fr] if fr is not None else []
        head = self.capture(HEAD_CAMERA)
        frames.append(head)
        self.frames.append(head)
        pred = obj.points.copy()                      # where the released model says the object is
        h_before = float(np.ptp(pred[:, 2])) if len(pred) else None
        refreshed = self.refresh_object(obj, frames, prompts)
        partial = disagree = False
        view = {}
        if refreshed and len(pred):
            fresh = obj.points
            d, _ = cKDTree(pred).query(fresh)
            agree = float(np.mean(d < 0.012))
            cover = float(np.ptp(fresh[:, 2]) / max(h_before, 1e-3))
            view = {"view_agreement": round(agree, 2), "view_height_coverage": round(cover, 2)}
            if cover < 0.6:
                # the fresh views show only part of the object (e.g. its top): no evidence about its height
                partial = True
                if agree >= 0.7:
                    # ... but what they show lies where the object was put: it is there, as put
                    obj.set_points(np.concatenate([pred, fresh]))
                else:
                    disagree = True
        self.rebuild_map(self.frames[-8:])
        c = obj.footprint_center()
        bottom = float(np.percentile(obj.points[:, 2], 2))
        out = {"refreshed": refreshed, "center": c, "bottom": bottom, "floor": region["floor_z"], **view}
        if partial:
            out["partial_view"] = True
        if refreshed and h_before and not disagree:
            h_after = float(np.ptp(obj.points[:, 2]))
            out["height_ratio"] = round(h_after / max(h_before, 1e-3), 2)
            out["upright"] = bool(abs(h_after - h_before) <= max(0.3 * h_before, 0.01))
        if rel == "in":
            reg = container_region(dest)
            d = np.linalg.norm(reg["grid_xy"] - c, axis=1)
            out["inside_interior"] = bool(len(d) and d.min() < 0.01)
            out["height_ok"] = bool(bottom < region["floor_z"] + 0.04)
            # inside the opening and below the rim: contained, also when it settled on its side
            out["below_rim"] = bool(np.percentile(obj.points[:, 2], 90) < reg["rim_z"] + 0.01)
            # any pose counts only once it has gone down into it: its bottom below the rim (not held up across the
            # top of the rim or of a divider)
            out["went_in"] = bool(bottom < reg["rim_z"] - 0.01)
            out["satisfied"] = out["inside_interior"] and (out["height_ok"] or out["below_rim"] or (any_pose and out["went_in"]))
            if compartment_xy is not None and len(compartment_xy):
                out["in_compartment"] = bool(np.linalg.norm(compartment_xy - c, axis=1).min() < 0.015)
                out["satisfied"] = out["satisfied"] and out["in_compartment"]
        elif rel == "on" and rest_z is not None:
            # stacked or dropped: over the destination's footprint; stacked at the planned rest height, dropped at or
            # below the surface it was let go over (any pose)
            fpx = dest.footprint_xy()
            out["on_support"] = bool(len(fpx) and np.linalg.norm(fpx - c, axis=1).min() < 0.02)
            out["height_ok"] = bool(bottom < rest_z + 0.03) if any_pose else bool(abs(bottom - rest_z) < 0.03)
            out["stacked"] = not any_pose
            if place_pt is not None:
                lim = {"tight": 0.03, "normal": 0.05, "loose": 0.09}[tol]
                out["dist_to_target"] = float(np.linalg.norm(c - place_pt[:2]))
                out["near_target"] = out["dist_to_target"] < lim
            out["satisfied"] = out["on_support"] and out["height_ok"] and out.get("near_target", True)
        elif rel == "on":
            reg = support_region(dest, point=place_pt, radius=1.0)
            d = np.linalg.norm(reg["grid_xy"] - c, axis=1) if len(reg["grid_xy"]) else np.array([1.0])
            out["on_support"] = bool(d.min() < 0.012)
            out["height_ok"] = bool(abs(bottom - region["floor_z"]) < 0.03)
            if place_pt is not None:
                lim = {"tight": 0.03, "normal": 0.05, "loose": 0.09}[tol]
                out["dist_to_target"] = float(np.linalg.norm(c - place_pt[:2]))
                out["near_target"] = out["dist_to_target"] < lim
            out["satisfied"] = out["on_support"] and out["height_ok"] and out.get("near_target", True)
        else:
            u = region["beside"]
            off = (c - region["ref_center"]) @ u
            out["offset_along_dir"] = float(off)
            out["satisfied"] = bool(off > 0.04 and abs(bottom - self.table_z) < 0.03)
        if "upright" in out and not out["upright"] and rel != "in" and not any_pose:
            out["satisfied"] = False
        if not refreshed or disagree:
            # fail closed: no usable fresh measurement of the object -> not established locally (VERIFY
            # decides from the images); never a reason to grab an object that may be resting correctly
            out["satisfied"] = False
            out["unverified"] = True
        return out

    # ---------------------------------------------------------------- actuate
    def do_actuate(self, st):
        """General articulation (no knowledge of what the part is): grasp the part System 1 named,
        move it the way System 1 asked about / along the measured mounting geometry, report the motion."""
        part = self.obj(st["object"])
        parent = self.objects.get(part.part_of) if part.part_of else None
        motion = st.get("motion") or "rotate"
        if motion in ("push", "pull"):
            # a part of a part (a handle on a drawer, on a door) moves with it: the body that slides or swings is
            # the outermost moving part, and its state is measured against the fixture it is mounted in
            while parent is not None and parent.part_of is not None and parent.part_of in self.objects \
                    and len(parent.points) >= 20:
                self.log("actuate_part_promoted", named=part.name, moving=parent.name)
                part, parent = parent, self.objects[parent.part_of]
        direction = st.get("turn_direction")
        result = {"stage": st["id"], "op": "actuate", "object": part.name, "motion": motion,
                  "goal": st.get("goal"), "attempts": []}
        try:
            return self._actuate(st, part, parent, motion, direction, result)
        finally:
            self.checker.free_near = None

    def _actuate(self, st, part, parent, motion, direction, result):
        if motion in ("push", "pull"):
            return self._actuate_linear(st, part, parent, motion, result)
        if "skill" in self.lrl_parts:
            return self.lrl_turn(st, part, parent, direction, result)      # axes and senses as candidates (lrl.py)
        failed = []
        for attempt in range(3):
            mf = self._mount_frame(part, parent)
            if attempt == 0 or failed:
                # look at the part along its mounting normal (sees the approach corridor in front of the face)
                if mf["n"][2] > 0.7:
                    fr = self.look_from_above(np.r_[mf["axis_point"][:2], part.top_z], height=0.24)
                else:
                    v = -mf["n"] + np.array([0.0, 0.0, -0.6])
                    fr = self.look_toward(mf["axis_point"], v, dist=0.24)
                if fr is not None:
                    # nothing has moved yet: merge the new view into the part model (a partial mask must not
                    # shrink it, or the unsegmented rest of the part would become an obstacle)
                    mt = self.mask_matching(fr, [part.category, part.name], part, min_iou=0.1)
                    if mt is not None:
                        part.add_points(self.filter_points(mask_points(fr, mt[0]), part, pad=0.02))
                    if parent is not None:
                        mt = self.mask_matching(fr, [parent.category, parent.name], parent)
                        if mt is not None:
                            parent.add_points(self.filter_points(mask_points(fr, mt[0]), parent))
                    self.rebuild_map(self.frames[-8:])
                    mf = self._mount_frame(part, parent)
            self.checker.free_near = [(part.points, 0.015, True)] + \
                ([(parent.points, 0.010, False)] if parent is not None else [])
            self.log("mount_frame", normal=mf["n"], axis_point=mf["axis_point"], protrusion=mf["protrusion"],
                     inliers=mf["plane_inliers"], from_scene=bool(mf.get("from_scene")))
            before = part.points.copy()
            plan = self.plan_part_grasp(part, parent, mf, failed, motion)
            if plan is None:
                result["attempts"].append({"status": "no_feasible_grip"})
                continue
            g, q_pre, q_g = plan
            f_open = g.info["finger"]
            self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))
            end = self.reach(g.R, g.p, q_g, g.R[:, 2], part.oid, (2 * (f_open + manip.PAD_OFFSET) - g.width) / 2,
                             "to_part", 0.0, f_open, q_pre)
            if end is None:
                failed.append(g.p)
                result["attempts"].append({"status": "transit_failed", "why": (self.last_move or {}).get("why")})
                continue
            g, q_g = self._taken(g, end, q_g)
            def guard(robot, F, F0, k):
                return "early_contact" if np.linalg.norm(F - F0) > 10.0 and k > 2 else None
            self.move_line(g.R, g.p, grip=0.0, speed=0.05, guard=self._guard("early_contact", guard), finger=f_open)
            short = float((g.p - self.rb.eef_pose()[1]) @ g.R[:, 2])
            if short > 0.006:
                failed.append(g.p)
                Rs, ps = self.rb.eef_pose()
                self.retreat(Rs, ps - Rs[:, 2] * 0.08, grip=0.0, speed=0.12, finger=f_open, tag="back_off")
                result["attempts"].append({"status": "blocked_descent", "short_m": round(short, 4)})
                continue
            opening = self.ex.gripper(1.0, max_steps=25)          # fixed parts: full grip for turning / pulling
            if opening < 0.002:
                self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))
                Rs, ps = self.rb.eef_pose()
                self.retreat(Rs, ps - Rs[:, 2] * 0.08, grip=0.0, speed=0.12, finger=f_open, tag="back_off")
                failed.append(g.p)
                result["attempts"].append({"status": "missed_part", "opening": opening})
                continue
            moved, effect = None, None
            if motion == "rotate" and st.get("effect"):
                moved, effect = self._turn_until_effect(st, g, q_g, mf, direction, part, parent)
                for t_ in effect["turns"]:
                    result["attempts"].append({"status": "turn", "sense": t_["sense"], "achieved_deg": t_.get("turned_deg"),
                                               "effect": t_.get("effect")})
            else:
                for s in articulate.turn_order(self, g, q_g, mf["axis_point"], mf["n"], direction):
                    ach = articulate.twist_about(self, mf["axis_point"], mf["n"], s * articulate.MAX_TURN)
                    result["attempts"].append({"status": "turn", "sign": s, "achieved_deg": round(np.rad2deg(ach), 1)})
                    if abs(ach) >= articulate.MIN_FOLLOW:
                        moved = {"rotation_deg": round(float(np.rad2deg(ach)), 1),
                                 "sense": "counterclockwise" if ach > 0 else "clockwise"}
                        break
            self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))            # release only as wide as approached
            Rs, ps = self.rb.eef_pose()
            self.retreat(Rs, ps - Rs[:, 2] * 0.08, grip=0.0, speed=0.12, finger=f_open, tag="release_back")
            # measure the part from fresh views (reported to the ledger; System 1 judges the state)
            head = self.capture(HEAD_CAMERA)
            self.frames.append(head)
            if self.refresh_object(part, [head], [part.category, part.name]):
                shift = float(np.linalg.norm(part.points.mean(0) - before.mean(0)))
                result["measured_part_shift_m"] = round(shift, 4)
            self.rebuild_map(self.frames[-8:])
            if effect is not None:
                # success is the device's visible change, never the hand's rotation
                result["effect"] = effect
                result["measured"] = moved
                ctx = getattr(self, "_effect_ctx", None)
                if effect["observed"] and ctx is not None:
                    # the change must still be there once the hand has let go
                    n_ch, n_vis, seen, hit = ctx["measure"](head, ctx["ref0"])
                    effect["after_release"] = {"changed_px": n_ch, "visible_px": n_vis, "seen": seen, "present": hit}
                    if seen and not hit:
                        effect["observed"] = None
                        effect["note"] = "the change was gone after releasing the part"
                rc = {"effect_observed": effect["observed"], "confirmed_by_s1": effect.get("confirmed_by_s1"), **moved}
                if effect["observed"]:
                    result["relation_check"] = {"satisfied": True, **rc}
                    self.log("actuated", result=result)
                    return result
                if effect["observed"] is None:
                    # hidden, unconfirmed or undone: not established locally, VERIFY judges the images
                    result["relation_check"] = {"satisfied": False, "unverified": True, **rc}
                    self.log("actuated", result=result)
                    return result
                if effect.get("lost_grip"):
                    failed.append(g.p)                     # the part slipped out of this grasp: grasp it again
                    result["attempts"].append({"status": "lost_grip", "turned_deg": moved["rotation_deg"]})
                    self.log("actuate_lost_grip", result=result)
                    continue
                self.log("actuated", result=result)
                raise StageFailure("no_visible_effect", expected_effect=effect["expected"], effect_pixel=effect.get("pixel"),
                                   watched_px=effect.get("watched_px"), returned_to_start=effect.get("returned"),
                                   residual_changed_px=effect.get("residual_changed_px"),
                                   turns=[{k: t_.get(k) for k in ("sense", "turned_deg", "checks", "max_changed_px",
                                                                  "min_visible_px", "rejected_changes", "stop")}
                                          for t_ in effect["turns"]],
                                   note="no turning sense produced the expected change in the watched area of the head "
                                        "camera (S1 confirms candidate changes); see returned_to_start for the part's state")
            if moved is not None:
                result["measured"] = moved
                result["relation_check"] = {"satisfied": True, **moved}
                self.log("actuated", result=result)
                return result
        raise StageFailure("actuation_failed", attempts=result["attempts"][-4:])

    def _mount_frame(self, part, parent):
        """Mounting face of a part: fitted to its parent's surface around it; when the parent's model does not
        surround the part (e.g. a drawer front segmented with the handle or with the cabinet), to every observed
        surface around it."""
        mf = articulate.mount_frame(part, parent, self.table_z)
        if mf["plane_inliers"] == 0 and len(part.points) and len(self.scene.cloud):
            c = part.points.mean(0)
            ext = float(np.linalg.norm(part.points[:, :2] - c[:2], axis=1).max())
            cl = self.scene.cloud
            near = cl[np.linalg.norm(cl - c, axis=1) < ext + 0.06]
            if len(near) > 30:
                mf2 = articulate.mount_frame(part, SimpleNamespace(points=near), self.table_z)
                if mf2["plane_inliers"] > 0:
                    mf2["from_scene"] = True
                    return mf2
        return mf

    # ---------------------------------------------------------------- push / pull (drawers, doors, sliders ...)
    MAX_LINEAR_ATTEMPTS = 4

    def _actuate_linear(self, st, part, parent, motion, result):
        """push / pull. Nothing here knows what the part is.
        Mounting face as a hypothesis: the candidate faces around the part (articulate.mount_faces: the parent's, then
        every observed surface's) are checked for plausibility (_face_implausible); the best usable one is looked at
        along its normal and refitted, then used for a grip (or, for a push, a press with the closed hand). A face that
        yields no grip / press, or along which the part does not move, gives way to the next one.
        Completion by the measured state: after every stroke the part is found again and its state measured from fresh
        depth views (articulate.articulation_state): goal 'closed' = its front flush with the surrounding fixture face,
        with the push ended against the end stop; goal 'open' = it stands out of that face (or is turned away from it)
        and the pull ended against the end stop. Otherwise another stroke from where the part now is. When the views do
        not give the state, the stroke's own evidence and System 1's effect pixel (a change there, confirmed by
        System 1) decide. Other goals (e.g. pressing a button): the part's face must have moved along the push."""
        goal = st.get("goal")
        want = goal if goal in ("open", "closed") else None
        lrl = "skill" in self.lrl_parts           # the skill pushes and pulls: one candidate per attempt (lrl.py)
        if self._part_not_separated(part, parent):
            size = lambda P: np.round(np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0), 3)
            raise StageFailure("actuation_failed", part_size_m=size(part.points), fixture_size_m=size(parent.points),
                               note="the segmented part is as large as its whole fixture (the segmenter returned the "
                                    "fixture for the part), so no face of the fixture can be measured around it")
        has_effect = bool(st.get("effect")) and st.get("effect_x") is not None and st.get("effect_y") is not None
        ref0 = self.capture(HEAD_CAMERA) if has_effect else None
        faces, bad, looked, failed = [], [], [], []
        last_state = None
        confirmed = None          # the face along which a stroke moved the part: kept for the next strokes
        remaining, fixture_pt = None, None      # closing: how far the part still stands out, a point on the fixture face
        front_pt = None           # a point on the part's measured moving front (world), for the next stroke

        def usable(hyps):
            use = [f for f in hyps if (lrl or f["why_not"] is None) and not any(self._same_face(f, b_) for b_ in bad)]
            if lrl:                              # v2's vetoes order the faces, they do not drop them (stable sort)
                use.sort(key=lambda f: f["why_not"] is not None)
            if confirmed is not None and not any(self._same_face(confirmed, b_) for b_ in bad):
                same = [f for f in use if float(np.dot(f["n"], confirmed["n"])) > np.cos(np.deg2rad(10.0))]
                use = same + [f for f in use if not any(f is g_ for g_ in same)] if same else [confirmed] + use
            return use

        prev_state, no_progress = None, 0
        for attempt in range(LRL["iterations"] if lrl else self.MAX_LINEAR_ATTEMPTS):
            if lrl and self.lrl_entered(result) >= LRL["candidates"]:
                break
            if self.elapsed() > self.max_wall_s:
                result["attempts"].append({"status": "wall_clock_limit"})
                break
            hyps = self._face_hypotheses(part, parent)
            self._note_faces(faces, hyps, motion)
            use = usable(hyps)
            if use and not any(self._same_face(use[0], l_) for l_ in looked):
                looked.append(use[0])
                fr = self._look_at_face(part, parent, use[0])
                if fr is not None:
                    hyps = self._face_hypotheses(part, parent)          # refit with the new view
                    self._note_faces(faces, hyps, motion)
                    use = usable(hyps)
                    if front_pt is None and use:
                        # the part's state before the first stroke: where its front is (the next stroke measures its
                        # displacement from there) and how far it stands out (sizes the end-of-stroke check)
                        pre = self._measure_state(part, use[0], [fr, self.capture(HEAD_CAMERA)])
                        self.log("part_state_before", **pre)
                        if pre.get("measured"):
                            prev_state = pre
                        if pre.get("front_m") is not None:
                            front_pt = use[0]["p0"] + np.asarray(use[0]["n"]) * float(pre["front_m"])
                        if pre.get("measured") and pre.get("closed") is False and pre.get("offset_m") is not None:
                            remaining = float(pre["offset_m"])
                            fixture_pt = use[0]["p0"] + np.asarray(use[0]["n"]) * float(pre["face_m"])
            if not use:
                break
            mf = use[0]
            self.checker.free_near = [(part.points, 0.015, True)] + \
                ([(parent.points, 0.010, False)] if parent is not None else [])
            self.log("mount_frame", normal=mf["n"], axis_point=mf["axis_point"], protrusion=mf["protrusion"],
                     inliers=mf["plane_inliers"], source=mf["source"], faces=self._dir_words(mf["n"]),
                     hypotheses=[{"n": np.round(f["n"], 3), "source": f["source"], "score": round(f["score"], 2),
                                  "lift": round(f["lift"], 3), "why_not": f["why_not"]} for f in hyps])
            before = part.points.copy()
            contents = self._contents_of(part)
            end_ok = self._closing_end_check(part, mf, remaining, fixture_pt) if want == "closed" else None
            # how far the hand has to push for the part to close (its measured offset, plus the end-stop margin)
            expected = remaining + self.CLOSE_PAST if (want == "closed" and remaining is not None) else None
            plan = None if lrl else self.plan_part_grasp(part, parent, mf, failed, motion, end_check=end_ok)
            if lrl:
                stroke = self.lrl_move(motion, part, parent, mf, failed, result, want, remaining, front_pt,
                                       fixture_pt)
            elif plan is not None:
                stroke = self._grip_stroke(mf, plan, motion, failed, result, part)
                if stroke is None and motion == "push" and result["attempts"] and \
                        result["attempts"][-1].get("status") == "missed_part":
                    # the grip closed on nothing (a thin edge): a push does not need a grip, press the part instead
                    stroke = self._press_stroke(part, parent, mf, failed, result, end_check=end_ok, expected=expected)
            elif motion == "push":
                stroke = self._press_stroke(part, parent, mf, failed, result, end_check=end_ok, expected=expected)
            else:
                result["attempts"].append({"status": "no_feasible_grip"})
                stroke = "no_plan"
            if stroke == "no_plan":
                bad.append(mf)
                self._face_status(faces, mf, "used: no grip" + (" and no press" if motion == "push" else "") +
                                  " found on the part from this face")
                continue
            if stroke is None:                   # the approach failed: same face, another grasp / press point
                continue
            if lrl and stroke["label"] in LRL["marking"]:
                # marked in lrl_move; no_effect: the part did not move along this face's axis, the face is done
                if stroke["label"] == "no_effect":
                    bad.append(mf)
                self._face_status(faces, mf, f"used: {stroke['mode']} {motion}, hand travel {stroke['travel']:.3f} m, "
                                             f"stop {stroke['stop']}: {stroke['label']}")
                continue
            fb = None if front_pt is None else float((front_pt - mf["p0"]) @ np.asarray(mf["n"], float))
            fl = None if fixture_pt is None else float((fixture_pt - mf["p0"]) @ np.asarray(mf["n"], float))
            state = self._observe_after_stroke(part, mf, motion, stroke, before, front_before=fb, fixture_level=fl,
                                               contents=contents)
            shift = float(state.get("face_shift_m") or 0.0) * (1.0 if motion == "push" else -1.0)
            hand_ax = max(0.0, float(np.asarray(stroke["disp"]) @ (-np.asarray(mf["n"]) if motion == "push"
                                                                    else np.asarray(mf["n"]))))
            if prev_state is not None and state.get("closed") is not None and \
                    shift > hand_ax + max(0.02, 0.15 * hand_ax):
                # the part cannot have moved further than the hand pushed or pulled it: the reading contradicts the
                # travel (a part pressed on the wrong face is not closed by being looked at); the earlier measurement
                # moved by the hand's travel stands, unmeasured
                self.log("state_inconsistent", hand_travel=round(hand_ax, 4), measured_shift=round(shift, 4),
                         now=self._state_words(state), before=self._state_words(prev_state))
                off = prev_state.get("offset_m")
                if hand_ax < 0.01:
                    # the hand did not move: the part did not either, the earlier measurement stands
                    state = dict(prev_state, face_shift_m=0.0, why="the hand did not move: the earlier measurement "
                                                                   "stands")
                else:
                    state = dict(state, measured=False, closed=None, face_shift_m=round(
                        hand_ax * (1.0 if motion == "push" else -1.0), 4),
                        offset_m=None if off is None else round(max(0.0, float(off) - hand_ax), 4),
                        why="the measured shift exceeds the hand's travel")
            prev_state = state if state.get("measured") else prev_state
            last_state = state
            front_pt = None if state.get("front_m") is None else \
                mf["p0"] + np.asarray(mf["n"]) * float(state["front_m"])
            fs = state.get("face_shift_m")
            if fs is not None and (fs if motion == "push" else -fs) >= 0.01:
                confirmed = mf                   # the part moved along this face's axis
            hand_axis = float(np.asarray(stroke["disp"]) @ (-np.asarray(mf["n"]) if motion == "push" else np.asarray(mf["n"])))
            if fs is not None and hand_axis >= 0.03 and (fs if motion == "push" else -fs) < 0.3 * hand_axis:
                failed.append(np.asarray(stroke["point"]))    # the hand slid on / off the part: press / grasp elsewhere
                result["attempts"][-1]["slipped"] = True
            if state.get("measured") and state.get("face_m") is not None:
                remaining = max(0.0, float(state.get("offset_m") or 0.0))
                fixture_pt = mf["p0"] + np.asarray(mf["n"]) * float(state["face_m"])
            verdict = self._linear_verdict(want, state, stroke)
            eff = None
            if verdict == "unmeasured" and has_effect:
                eff = self._effect_after_stroke(st, ref0, stroke["head_after"], motion, stroke["travel"])
                if eff.get("confirmed"):
                    verdict = "done_effect"
            if verdict == "unmeasured":
                verdict = "again"
            result["attempts"][-1].update(state=self._state_words(state), verdict=verdict,
                                          **({"effect": eff} if eff else {}))
            if lrl and (verdict == "face_suspect" or (stroke["label"] == "reached" and not verdict.startswith("done"))):
                if stroke["label"] == "reached":
                    self.log("lrl_disagree", stage=result["stage"], label="reached", v2=verdict,
                             state=self._state_words(state))
                self.lrl_mark(result, stroke, f"{stroke['label']}, v2's verdict {verdict}")
            self._face_status(faces, mf, f"used: {stroke['mode']} {motion}, hand travel {stroke['travel']:.3f} m, "
                                         f"stop {stroke['stop']}; then {self._state_words(state)}")
            if verdict.startswith("done"):
                moved = {"travel_m": round(float(stroke["travel"]), 3), "stop": stroke["stop"], "mode": stroke["mode"],
                         "state": self._state_words(state), "closed": state.get("closed"),
                         "stands_out_m": state.get("offset_m"), "tilt_deg": state.get("angle_deg")}
                result["measured"] = moved
                if verdict == "done":
                    result["relation_check"] = {"satisfied": True, **moved}
                elif verdict == "done_effect":
                    result["relation_check"] = {"satisfied": True, "effect_observed": True, "confirmed_by_s1": True,
                                                **moved}
                else:
                    result["relation_check"] = {"satisfied": False, "unverified": True, **moved}
                self.log("actuated", result=result)
                return result
            if verdict == "again":
                prog = abs(float(state.get("face_shift_m") or 0.0))
                no_progress = no_progress + 1 if prog < 0.005 else 0
                if no_progress >= 2 and not lrl:
                    # two strokes without the part moving: another stroke from the same contact will not do it
                    result["attempts"][-1]["no_progress"] = True
                    bad.append(mf)
                    no_progress = 0
            if verdict == "face_suspect":
                if not lrl:                      # lrl: the candidate is marked, the next one on this face first
                    bad.append(mf)
                self._face_status(faces, mf, f"used: the part did not move along this face's axis "
                                             f"(hand travel {stroke['travel']:.3f} m, stop {stroke['stop']}); "
                                             f"{self._state_words(state)}")
        faces.sort(key=lambda fs_: 0 if fs_[1]["status"].startswith("used") else 1)      # stable: used faces first
        info = {"attempts": result["attempts"][-4:], "faces": [s_ for _, s_ in faces][:5],
                "part_state": self._state_words(last_state) if last_state is not None else "no stroke was made"}
        if want is not None and last_state is not None:
            info["note"] = (f"goal '{want}' was not reached: the measured state is the part_state above "
                            f"(from depth; finer than the images)")
        raise StageFailure("actuation_failed", **info)

    def _linear_verdict(self, want, state, stroke):
        """done / done_unverified / again (another stroke on the same face) / face_suspect (the part did not move along
        this face's axis) / unmeasured (no state from the views)."""
        moved = stroke["travel"] >= 0.01
        resisted = stroke["stop"] in ("stalled", "force_limit")
        if want is None:
            fs = state.get("face_shift_m")
            if moved and (stroke["mode"] == "grip" or (fs is not None and fs > 0.005)):
                return "done"
            return "again" if moved else "face_suspect"
        if state.get("closed") is not None:
            if want == "closed":
                if state["closed"]:
                    return "done" if (resisted or not moved) else "again"     # flush and against the end stop
                return "again" if moved else "face_suspect"
            opened = (not state["closed"]) and ((state.get("offset_m") or 0.0) >= 0.02
                                                 or (state.get("angle_deg") or 0.0) >= 15.0)
            if opened:
                return "done" if (resisted or not moved) else "again"         # out of the face, at the end stop
            return "again" if moved else "face_suspect"
        if not moved and resisted:
            return "done_unverified"         # it does not go any further: at its end stop (state not seen)
        if want == "open" and resisted and stroke["travel"] >= 0.03:
            return "done_unverified"         # pulled until it stopped
        return "unmeasured" if moved else "face_suspect"

    def _state_words(self, state):
        if state is None:
            return "not measured"
        if state.get("closed") is True:
            return "closed: its front is flush with the surrounding face"
        if state.get("closed") is False:
            if state.get("offset_m") is not None:
                return f"open: its front stands {100 * state['offset_m']:.1f} cm out of the surrounding face"
            return f"open: it is turned {state.get('angle_deg')} deg away from the face"
        return "not measured (" + str(state.get("why", "")) + ")"

    def _dir_words(self, v):
        """A world direction in the words of image 1: up / down, image left / right, toward / away from the head camera."""
        v = np.asarray(v, float)
        if abs(v[2]) >= 0.7 * np.linalg.norm(v):
            return "up" if v[2] > 0 else "down"
        R = self.head0.T_wc[:3, :3]
        right, fwd = R[:, 0].copy(), R[:, 2].copy()              # image right; optical axis (away from the camera)
        right[2] = fwd[2] = 0.0
        right /= max(np.linalg.norm(right), 1e-9)
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        a, b = float(v @ right), float(v @ fwd)
        if abs(a) >= abs(b):
            return "image right" if a > 0 else "image left"
        return "away from the head camera" if b > 0 else "toward the head camera"

    @staticmethod
    def _same_face(f, g):
        return float(np.dot(f["n"], g["n"])) > np.cos(np.deg2rad(10.0)) and \
            abs(float(np.dot(np.asarray(f["p0"]) - np.asarray(g["p0"]), g["n"]))) < 0.015

    def _note_faces(self, faces, hyps, motion):
        for f in hyps:
            if not any(self._same_face(f, g) for g, _ in faces):
                a = -np.asarray(f["n"]) if motion == "push" else np.asarray(f["n"])
                faces.append((f, {"faces": self._dir_words(f["n"]), f"{motion}_goes": self._dir_words(a),
                                  "normal": np.round(f["n"], 2).tolist(), "from": f["source"],
                                  "status": ("rejected: " + f["why_not"]) if f["why_not"] else "not tried"}))

    def _face_status(self, faces, f, status):
        for g, s_ in faces:
            if self._same_face(f, g):
                s_["status"] = status
                return

    def _part_not_separated(self, part, parent):
        """The segmenter returned the whole fixture for the part: the part's extent and centre match its parent's."""
        if parent is None or len(parent.points) < 30 or len(part.points) < 30:
            return False
        ext = lambda P: np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0)
        return bool(np.all(ext(part.points) >= 0.85 * ext(parent.points))
                    and np.linalg.norm(part.points.mean(0) - parent.points.mean(0)) < 0.04)

    def _face_hypotheses(self, part, parent):
        """Mounting-face hypotheses for a push / pull, best first: the parent's faces around the part (5 planes), then
        those of every observed surface around it (the parent's model may not surround the part). why_not is None for a
        usable face, else why it cannot be the one."""
        if len(part.points) < 20:
            return []
        faces = [dict(f, source="parent") for f in articulate.mount_faces(part, parent, self.table_z, k=5)]
        c = part.points.mean(0)
        ext = float(np.linalg.norm(part.points[:, :2] - c[:2], axis=1).max())
        cl = self.scene.cloud
        if len(cl):
            near = cl[np.linalg.norm(cl - c, axis=1) < ext + 0.06]
            if len(near) > 30:
                for f in articulate.mount_faces(part, SimpleNamespace(points=near), self.table_z, k=5):
                    if not any(self._same_face(f, g) for g in faces):
                        faces.append(dict(f, source="scene"))
        for f in faces:
            f["why_not"] = self._face_implausible(part, f)
        return faces

    def _face_implausible(self, part, f):
        """Why a mounting-face hypothesis cannot be the face a push / pull works along (None if it can). The hand works
        on the part from the face's outer side and moves it along the face normal, so: the face must not look down (the
        hand would work from under the part); the part must sit on it (its nearest points within 4.5 cm of it: a
        fixture side that merely runs past the part does not hold it); and the space just in front of the part's outer
        face must not be solid (observed surfaces there, or below the table)."""
        n = np.asarray(f["n"], float)
        if n[2] < -0.5:
            return "faces_down"
        if self._seen_from_behind(f):
            return "seen_from_behind"
        if f.get("lift", 0.0) > 0.045:
            return "part_not_on_face"
        Q = articulate.mount_rotation(n)
        L = (part.points - f["p0"]) @ Q.T
        lo, hi = np.percentile(L[:, :2], 20, axis=0), np.percentile(L[:, :2], 80, axis=0)
        gu, gv = np.meshgrid(np.linspace(lo[0], hi[0], 5), np.linspace(lo[1], hi[1], 5))
        outer = float(np.percentile(L[:, 2], 95))
        S = np.concatenate([np.c_[gu.ravel(), gv.ravel(), np.full(gu.size, outer + dz)] for dz in (0.03, 0.06)]) @ Q \
            + f["p0"]
        occ = S[:, 2] < self.table_z
        idx = self.scene.index(S)
        ins = self.scene.inside(idx)
        if ins.any():
            ii = tuple(idx[ins].T)
            occ[ins] |= (self.scene.state[ii] == OCC) & (self.scene.label[ii] != part.oid)
        return "approach_side_solid" if occ.mean() >= 0.5 else None

    def _seen_from_behind(self, f):
        """The cameras that saw this face saw it from the other side of the hypothesis: its outer side would be the
        inside of the fixture (a camera only sees a surface from the side it faces)."""
        S = f.get("inl_sample")
        if S is None or len(S) == 0:
            return False
        n = np.asarray(f["n"], float)
        front = back = 0
        for fr in self.frames[-10:] + [self.head0]:
            uv, z = fr.project(S)
            H, W = fr.depth.shape
            ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if not ok.any():
                continue
            u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
            vis = np.abs(fr.depth[v, u] - z[ok]) < 0.01
            if not vis.any():
                continue
            s_ = (fr.T_wc[:3, 3] - S[ok][vis]) @ n
            front += int((s_ > 0).sum())
            back += int((s_ < 0).sum())
        return back >= 20 and back > 2 * front

    def _look_at_face(self, part, parent, mf):
        """Look at the part's outer face along the face normal (sees the face and the corridor in front of it) and merge
        the view into the part and parent models (nothing has moved since they were measured). Frame or None."""
        n = np.asarray(mf["n"], float)
        h = (part.points - mf["p0"]) @ n
        tgt = part.points[h >= np.percentile(h, 90)].mean(0)
        if n[2] > 0.7:
            fr = self.look_from_above(np.r_[tgt[:2], part.top_z], height=0.24)
        else:
            fr = self.look_toward(tgt, -n + np.array([0.0, 0.0, -0.6]), dist=0.24)
        if fr is None:
            return None
        mt = self.mask_matching(fr, [part.category, part.name], part, min_iou=0.1)
        if mt is not None:
            part.add_points(self.filter_points(mask_points(fr, mt[0]), part, pad=0.02))
        if parent is not None:
            mt = self.mask_matching(fr, [parent.category, parent.name], parent)
            if mt is not None:
                parent.add_points(self.filter_points(mask_points(fr, mt[0]), parent))
        self.rebuild_map(self.frames[-8:])
        return fr

    def _grip_stroke(self, mf, plan, motion, failed, result, part):
        """Grasp the part (plan from plan_part_grasp), push / pull it along the face normal (articulate.translate:
        compliant, bounded by force, travel, stall and grip), release and back off. The stroke, or None when the
        approach failed (the grasp point joins `failed`)."""
        g, q_pre, q_g = plan
        f_open = g.info["finger"]
        self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))
        end = self.reach(g.R, g.p, q_g, g.R[:, 2], part.oid, (2 * (f_open + manip.PAD_OFFSET) - g.width) / 2,
                         "to_part", 0.0, f_open, q_pre)
        if end is None:
            failed.append(g.p)
            result["attempts"].append({"status": "transit_failed", "why": (self.last_move or {}).get("why")})
            return None
        g, q_g = self._taken(g, end, q_g)
        def guard(robot, F, F0, k):
            return "early_contact" if np.linalg.norm(F - F0) > 10.0 and k > 2 else None
        self.move_line(g.R, g.p, grip=0.0, speed=0.05, guard=self._guard("early_contact", guard), finger=f_open)
        short = float((g.p - self.rb.eef_pose()[1]) @ g.R[:, 2])
        if short > 0.006:
            failed.append(g.p)
            Rs, ps = self.rb.eef_pose()
            self.retreat(Rs, ps - Rs[:, 2] * 0.08, grip=0.0, speed=0.12, finger=f_open, tag="back_off")
            result["attempts"].append({"status": "blocked_descent", "short_m": round(short, 4)})
            return None
        opening = self.ex.gripper(1.0, max_steps=25)          # fixed parts: full grip for pulling / pushing
        if opening < 0.002:
            self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))
            Rs, ps = self.rb.eef_pose()
            self.retreat(Rs, ps - Rs[:, 2] * 0.08, grip=0.0, speed=0.12, finger=f_open, tag="back_off")
            failed.append(g.p)
            result["attempts"].append({"status": "missed_part", "opening": opening})
            return None
        p_start = self.rb.eef_pose()[1].copy()
        travel, why, d = articulate.translate(self, mf["n"], motion)
        self.log("leg", tag="grip_stroke", cause=why, travel=round(travel, 4), motion=motion)
        disp = self.rb.eef_pose()[1] - p_start
        result["attempts"].append({"status": motion, "mode": "grip", "travel_m": round(travel, 3), "stop": why})
        self.ex.gripper_to(2 * (f_open + manip.PAD_OFFSET))            # release only as wide as approached
        Rs, ps = self.rb.eef_pose()
        self.retreat(Rs, ps - Rs[:, 2] * 0.08, grip=0.0, speed=0.12, finger=f_open, tag="release_back")
        return {"mode": "grip", "travel": float(travel), "stop": why, "d": d, "disp": disp, "point": g.p}

    PRESS_SEARCH_S = (15.0, 30.0)   # s: press-pose search: stop at the first nearest-to-clear pose after the first, always
    # after the second (a refused search must not cost a minute: CONTACT_STAGE_PLAN 4.2.6)
    RESEATS = 6                     # continuations in contact per stroke, while each one still moves the part

    def _press_stroke(self, part, parent, mf, failed, result, end_check=None, expected=None):
        """'push' when the part cannot be gripped: press it with the closed hand. The hand's approach axis points along
        the push direction (-n, into the face the part is mounted on); the closed fingertips meet the part's outermost
        face at its centre, coming in from 6 cm in front; then the same compliant push as with a grip
        (articulate.translate: re-aims along the displacement the part allows; force, travel and stall bounds); back
        off. Nothing here knows what the part is: S1 names the part and the motion, the mount face gives the direction.
        The stroke; 'no_plan' when no press pose is clear; None when the approach failed (the point joins `failed`)."""
        n = np.asarray(mf["n"], float)
        P = part.points
        if len(P) < 20:
            result["attempts"].append({"status": "no_feasible_grip_or_press", "why": "no part model"})
            return "no_plan"
        proj = P @ n
        # centre of the part's outer face: the middle of the extent (in the face plane) of the slab within 1 cm of its
        # outer level; the mean of the outermost points leans toward the edge the cameras see most of
        slab = P[proj >= np.percentile(proj, 90) - 0.01]
        Qm = articulate.mount_rotation(n)
        Ls = (slab - P.mean(0)) @ Qm.T
        mid = np.r_[0.5 * (np.percentile(Ls[:, :2], 5, axis=0) + np.percentile(Ls[:, :2], 95, axis=0)),
                    np.median(Ls[:, 2])]
        # lateral window of a press: half the outer face's smaller extent (the fingertips must land on it)
        press_win = 0.5 * float(np.min(np.percentile(Ls[:, :2], 95, axis=0) - np.percentile(Ls[:, :2], 5, axis=0)))
        c0 = P.mean(0) + mid @ Qm
        u = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        excl = (part.oid,) + ((parent.oid,) if parent is not None else ())
        q_now = self.rb.q()[:7]
        plan = None
        fallback = None             # clear on the way in, but would meet the fixture before the part closes
        why_n, blame = {}, {}
        # Ways to press: the closed fingertips on the face, coming straight along the push or leaning down onto it from
        # above; or (a part on a face that is not level) the side of the closed fingers, the hand pointing down just in
        # front of the face (a part low on its fixture: a level hand would put the forearm through the table). The push
        # itself always goes along -n.
        down = np.array([0.0, 0.0, -1.0]) - n * float(np.dot([0.0, 0.0, -1.0], n))
        side_ok = np.linalg.norm(down) > 0.7
        down = down / max(np.linalg.norm(down), 1e-9)
        poses = []                                                            # (mode, approach, R, site at the face, c)
        for du, dv in ((0, 0), (0.015, 0), (-0.015, 0), (0, 0.015), (0, -0.015), (0, 0.03), (0, -0.03)):
            c = c0 + du * u + dv * v
            if any(np.linalg.norm(c - f) < 0.008 for f in failed):
                continue
            tilts = [0.0] + ([65.0, 80.0] if side_ok else [])
            sides = [0.01, -0.01, -0.025] if side_ok and dv == 0 else []
            lean = [30.0, 50.0] if side_ok else []
            # order: level fingertips; steep fingertips (the hand stays behind its fingertips all the way to the end
            # stop); the finger sides; moderate leans
            for pt in tilts:
                a_ = np.cos(np.deg2rad(pt)) * -n + np.sin(np.deg2rad(pt)) * down
                for roll in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4):
                    poses.append(("tip", a_, frame_from_axes(a_, np.cos(roll) * u + np.sin(roll) * v),
                                  c - a_ * (manip.TIP_Z + 0.002), c))
            # finger sides: fingertips 1 cm below the contact point, or 1 / 2.5 cm above it (the finger sides then meet
            # the face above the part: same rigid body for a drawer or door), 1.8 cm out from it
            for dz in sides:
                tip = c + n * 0.018 + down * dz
                for xh in (u, n):
                    poses.append(("side", down, frame_from_axes(down, xh), tip - down * manip.TIP_Z, c))
            for pt in lean:
                a_ = np.cos(np.deg2rad(pt)) * -n + np.sin(np.deg2rad(pt)) * down
                for roll in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4):
                    poses.append(("tip", a_, frame_from_axes(a_, np.cos(roll) * u + np.sin(roll) * v),
                                  c - a_ * (manip.TIP_Z + 0.002), c))
        t_search = time.time()
        for mode, a_dir, R, p_touch, c in poses:
            el = time.time() - t_search
            if el > self.PRESS_SEARCH_S[1] or (el > self.PRESS_SEARCH_S[0] and fallback is not None):
                why_n["search_time_limit"] = 1
                break
            sols = self.ik_clear(R, p_touch - a_dir * 0.06, q_now, exclude=excl, finger=0.0, n_random=8)
            if len(sols) == 0:
                why_n[f"{mode}_no_clear_ik"] = why_n.get(f"{mode}_no_clear_ik", 0) + 1
            for q_pre in sols[:3]:
                line = self.line_configs(R, p_touch, q_pre, finger=0.0)
                if line is None:
                    why_n[f"{mode}_no_line"] = why_n.get(f"{mode}_no_line", 0) + 1
                    continue
                bad = None
                for q in list(line[:-1:2]) + ([line[-1]] if mode == "side" else []):
                    good, cl = self.checker.ok(q[None], NEAR, exclude=excl, finger=0.0)
                    if not good[0]:
                        bad = next((k for k, v in cl.items() if v[0] < NEAR.get(k, 0.02)), "clearance")
                        if f"{mode}_{bad}" not in blame:
                            W, _ = self.checker.world_points(q[None], 0.0)
                            grp = {"arm": 0, "palm": 1, "finger": 2}.get(bad.replace("plane_", ""), 0)
                            blame[f"{mode}_{bad}"] = {**self.nearest_obstacle(W[0][self.checker.pg == grp], excl),
                                                      "clearance": {k_: round(float(v_[0]), 4) for k_, v_ in cl.items()}}
                        break
                if bad:
                    why_n[f"{mode}_line_{bad}"] = why_n.get(f"{mode}_line_{bad}", 0) + 1
                    continue
                m_end = None if end_check is None else end_check(R, p_touch, line[-1], 0.0)
                if m_end is not None and m_end < 0:
                    why_n[f"{mode}_end_blocked"] = why_n.get(f"{mode}_end_blocked", 0) + 1
                    if fallback is None or m_end > fallback[0]:
                        fallback = (m_end, (mode, R, p_touch, q_pre, c, a_dir))
                    continue
                plan = (mode, R, p_touch, q_pre, c, a_dir)
                break
            if plan:
                break
        if plan is None and fallback is not None:
            plan = fallback[1]          # no press reaches the end stop clear of the fixture: the nearest-to-clear one
            self.log("press_end_check_relaxed", rejected=why_n, end_margin=round(fallback[0], 4))
        if plan is None:
            result["attempts"].append({"status": "no_feasible_grip_or_press", "press_rejected": why_n,
                                       "press_blame": blame})
            self.log("press_fail", rejected=why_n, blame=blame, normal=n, face_point=c0)
            return "no_plan"
        mode, R, p_touch, q_pre, c, a_dir = plan
        self.log("press_plan", point=c, normal=n, approach=a_dir, mode=mode)
        self.ex.gripper(1.0, max_steps=25)                                  # closed hand
        end = self.reach(R, p_touch, self._ik_near(R, p_touch, q_pre), a_dir, part.oid, press_win, "to_press", 1.0,
                         0.0, q_pre)
        if end is None:
            failed.append(c)
            result["attempts"].append({"status": "transit_failed", "pressing": True,
                                       "why": (self.last_move or {}).get("why")})
            return None
        R = np.asarray(end.R, float)                 # closed fingertips are symmetric: the twin presses the same
        def guard(robot, F, F0, k):
            return "contact" if np.linalg.norm(F - F0) > 8.0 and k > 2 else None
        if mode == "tip":
            self.move_line(R, p_touch + a_dir * 0.01, grip=1.0, speed=0.04, guard=self._guard("contact", guard),
                           finger=0.0)                                                    # in until contact
        else:
            self.move_line(R, p_touch, grip=1.0, speed=0.05, guard=self._guard("contact", guard),
                           finger=0.0)                                                    # down in front of the face
        p_start = self.rb.eef_pose()[1].copy()
        travel, why, d = articulate.translate(self, n, "push", grip_check=False)
        reseats = 0
        # continuation in contact (CONTACT_STAGE_PLAN 4.3): the hand stopped (force, stall) before the part can have
        # reached its goal, which the measured offset says is `expected` metres of travel away: the hand is still at
        # the part, so re-seat a little away from what blocked it and push on, with no new transit and no new search
        while expected is not None and why in ("force_limit", "stalled") and reseats < self.RESEATS:
            done_so_far = float((self.rb.eef_pose()[1] - p_start) @ -n)
            if done_so_far >= expected - 0.005:
                break
            if not self._reseat(part, parent, n, a_dir):
                break
            reseats += 1
            t2, why, d = articulate.translate(self, n, "push", max_travel=max(0.02, expected - done_so_far + 0.02),
                                              grip_check=False)
            if t2 < 0.005:
                break                                # this re-seat no longer moves the part: the stroke ends here
        disp = self.rb.eef_pose()[1] - p_start
        travel = float(np.linalg.norm(disp)) if reseats else travel
        self.log("leg", tag="press_stroke", cause=why, travel=round(travel, 4), expected=expected, reseats=reseats)
        result["attempts"].append({"status": "press_push", "mode": "press", "pose": mode, "travel_m": round(travel, 3),
                                   "stop": why, "reseats": reseats})
        Rs, ps = self.rb.eef_pose()
        back = ps - a_dir * 0.06 if mode == "tip" else ps + n * 0.02 - a_dir * 0.06
        self.retreat(Rs, back, grip=1.0, speed=0.12, finger=0.0, tag="press_back")              # back off
        self.ex.gripper(-1.0, max_steps=12)
        return {"mode": "press", "travel": float(travel), "stop": why, "d": d, "disp": disp, "point": c}

    def _reseat(self, part, parent, n, a_dir=None):
        """Continue a push in contact: back off a few mm the way the hand came in (the reverse of its approach a_dir,
        else along the face normal), move the contact point 1.5 cm (in the face plane) away from the nearest observed
        thing that is not the part (what stopped the hand), and come back in guarded until the hand meets the part
        again. True when it is on the part again."""
        n = np.asarray(n, float)
        out_dir = n if a_dir is None else -np.asarray(a_dir, float) / max(np.linalg.norm(a_dir), 1e-9)
        Rs, ps = self.rb.eef_pose()
        P = self._touch_points()
        near = self.checker.near
        near._labels(self.scene)
        keep = near.lab != part.oid
        down = np.array([0.0, 0.0, -1.0]) - n * float(n[2] * -1.0)
        v = down
        if keep.any():
            d, i = cKDTree(near.cloud[keep]).query(P)
            k = int(np.argmin(d))
            if d[k] < 0.03:
                v = P[k] - near.cloud[keep][i[k]]
                v = v - n * float(v @ n)
                if np.linalg.norm(v) < 1e-3:
                    v = down
        if np.linalg.norm(v) < 1e-6:
            return False
        v = v / np.linalg.norm(v)
        back = ps + out_dir * 0.006
        shifted = back + v * 0.015
        for goal in (back, shifted):
            # off the pressed part: the released preload is no contact; the detector alone guards (see retreat)
            r = self.move_line(Rs, goal, grip=1.0, speed=0.03, finger=0.0, guard=self._guard("contact", None),
                               force_limit=1e9)
            if r is None or r.status != "done":
                self.log("reseat", ok=False, at=goal, status=None if r is None else r.status)
                return False
        out = self.guarded_leg(Rs, shifted - n * 0.025, speed=0.03, window=(0.0, 0.03), grip=1.0, finger=0.0,
                               tag="reseat_in")
        self.log("reseat", ok=out.verdict == "as_expected", shift=v, verdict=out.verdict)
        return out.verdict == "as_expected"

    def _contents_of(self, part):
        """Objects resting in or on a part: they move with it (a bowl in a drawer)."""
        if len(part.points) < 20:
            return []
        lo, hi = part.points.min(0), part.points.max(0)
        out = []
        for oid, m in self.objects.items():
            if oid in (part.oid, part.part_of) or m.held or getattr(m, "virtual", False) or len(m.points) < 10:
                continue
            c, b = m.footprint_center(), float(m.points[:, 2].min())
            if np.all(c >= lo[:2] - 0.01) and np.all(c <= hi[:2] + 0.01) and lo[2] - 0.02 <= b <= hi[2] + 0.02:
                out.append(m)
        return out

    def _observe_after_stroke(self, part, mf, motion, stroke, before, front_before=None, fixture_level=None,
                              contents=()):
        """Measure the part's state after a stroke (articulate.articulation_state) from the head view and a wrist view
        along the face normal, and find the part again. A sliding part keeps its footprint on the face, so its moving
        front is searched over that footprint anywhere along the stroke axis (from deep inside to well past where it
        was); the pre-stroke model shifted by the measured displacement of the front is the new model, refined by the
        segmenter around it. Without a measured front: the segmenter around the pre-stroke model moved by the hand's
        travel along the axis, then around where it was; if neither view shows it, that estimate (never the stale
        pre-stroke model). front_before: height of the front along n before the stroke (measured), if known."""
        n = np.asarray(mf["n"], float)
        p0 = np.asarray(mf["p0"], float)
        a = -n if motion == "push" else n
        disp = max(0.0, float(np.asarray(stroke["disp"], float) @ a))
        prompts = [part.category, part.name]
        head = self.capture(HEAD_CAMERA)
        self.frames.append(head)
        stroke["head_after"] = head
        hb = (before - p0) @ n
        fb = float(front_before) if front_before is not None else float(np.percentile(hb, 90))
        outer = before[hb >= np.percentile(hb, 90) - 0.01]
        tgt = np.median(outer, axis=0) + a * min(disp, 0.25)
        fr = self.look_toward(tgt, -n + np.array([0.0, 0.0, -0.6]), dist=0.32)
        if fr is None:
            fr = self.look_toward(tgt, tgt - self.head0.T_wc[:3, 3], dist=0.34)       # from the head camera's side
        views = [head] + ([fr] if fr is not None else [])
        band = (fb - 0.40, fb + 0.03) if motion == "push" else (fb - 0.03, fb + 0.40)
        state = self._measure_state(part, mf, views, part_pts=before, front_band=band, allow_turned=False)
        if state.get("front_m") is not None and not state.get("measured") and fixture_level is not None \
                and (state.get("angle_deg") or 0.0) <= articulate.CLOSED_TILT:
            # the surroundings were not seen well enough now, but the fixture face does not move: use where it was
            # measured before
            off = float(state["front_m"]) - float(fixture_level)
            state.update(measured=True, face_m=round(float(fixture_level), 4), offset_m=round(max(off, 0.0), 4),
                         closed=bool(off <= articulate.FACE_TOL), why="fixture face from an earlier view")
        found = False
        if state.get("front_m") is not None:
            delta = float(state["front_m"]) - fb                    # the front's measured displacement along n
            part.set_points(before + n * delta)
            self._forget_free(part.points)
            for m in contents:                                      # what lies in the part moved with it
                m.set_points(m.points + n * delta)
                self._forget_free(m.points)
            if contents:
                self.log("contents_moved", objects=[m.name for m in contents], shift_m=round(float(delta), 4))
            state["face_shift_m"] = round(-delta, 4)
            state["model_from"] = "measured front"
            keep = part.points.copy()
            if self.refresh_object(part, views, prompts):
                o_new = float(np.percentile((part.points - p0) @ n, 90))
                o_keep = float(np.percentile((keep - p0) @ n, 90))
                if abs(o_new - o_keep) > 0.02:       # the segmenter's mask is not the moved front: keep the shift
                    part.set_points(keep)
            found = True
        else:
            for g_ in (before + a * disp, before):
                part.set_points(g_)
                if self.refresh_object(part, views, prompts):
                    found = True
                    break
            if not found:
                part.set_points(before + a * disp)
                state["model_from"] = "hand travel (not seen)"
            else:
                state["model_from"] = "segmenter"
                state["face_shift_m"] = round(float(np.percentile(hb, 90) - np.percentile((part.points - p0) @ n, 90)), 4)
                turned = self._measure_state(part, mf, views, allow_turned=True)
                if turned.get("closed") is False and turned.get("offset_m") is None:
                    state.update({k: turned[k] for k in ("measured", "closed", "angle_deg", "why")})
        self.rebuild_map(self.frames[-8:])
        state["part_found"] = bool(found)
        self.log("part_state", stroke={k: stroke[k] for k in ("mode", "travel", "stop")}, view=fr is not None,
                 front_before=round(fb, 4), **state)
        return state

    def _measure_state(self, part, mf, views, part_pts=None, front_band=None, allow_turned=True):
        """articulate.articulation_state of the part from the given frames (part pixels from the projected model;
        part_pts: the footprint to use instead of the current model)."""
        P = part.points if part_pts is None else part_pts
        obs = []
        for v in views:
            if v is None:
                continue
            Pv, Nv, rc = articulate.depth_points_normals(v)
            onp = project_mask(v, P, radius_px=2)[rc[:, 0], rc[:, 1]]
            obs.append((Pv, Nv, onp))
        return articulate.articulation_state(obs, P, np.asarray(mf["n"], float), mf["p0"], front_band=front_band,
                                             allow_turned=allow_turned)

    CLOSE_PAST = 0.01       # m: a closing stroke is planned to reach this far past flush (end stops sit behind the face)

    def _closing_end_check(self, part, mf, remaining, fixture_pt):
        """For a closing push whose remaining travel is measured: a check(R, p, q_ref, finger) that the hand pose (R, p)
        moved along the push until the part is CLOSE_PAST behind its fixture face stays clear of the fixture's static
        surfaces (observed points at or behind the fixture face: the fixture's own edges around the part included). A
        press or grip that would jam on the fixture before the part closes is not preferred. The check returns the
        clearance margin (>= 0: clear); None when the remaining travel is not known."""
        if remaining is None or fixture_pt is None:
            return None
        n = np.asarray(mf["n"], float)
        cl = self.scene.cloud
        if len(cl) == 0:
            return None
        c = part.points.mean(0)
        S = cl[np.linalg.norm(cl - c, axis=1) < 0.45]
        S = S[(S - fixture_pt) @ n <= 0.004]
        if len(S) < 10:
            return None
        tree = cKDTree(S)
        shift = -n * (float(remaining) + self.CLOSE_PAST)
        g_ = self.checker.pg
        r_ = self.checker.pr

        def check(R, p, q_ref, finger):
            q_ref = np.asarray(q_ref, float)[:7]
            q, ok, _, _ = self.rm.ik(R, np.asarray(p) + shift, q_ref[None], q_rest=q_ref, iters=80)
            if not ok[0]:
                return -1.0
            W, _ = self.checker.world_points(q[:1], finger)
            d, _ = tree.query(W[0], distance_upper_bound=0.06)
            d = np.minimum(d - r_, 0.06)
            return float(min(d[g_ == 2].min() - 0.002, d[g_ == 1].min() - 0.005, d[g_ == 0].min() - 0.01))
        return check

    def _effect_after_stroke(self, st, ref0, cur, motion, travel):
        """System 1's effect pixel for a push / pull: a visible change in a disc around it (head camera, before the
        first stroke vs now; the part itself is watched too, its motion is the effect), confirmed by System 1 while
        the budget allows (REPAIR and VERIFY keep theirs)."""
        H, W = ref0.depth.shape
        P = ref0.pixel_ray_point(st["effect_x"], st["effect_y"])
        if not np.all(np.isfinite(P)):
            return {"change_seen": False, "why": "no depth at the effect pixel"}
        pc = np.linalg.inv(ref0.T_wc) @ np.r_[P, 1.0]
        r_px = max(6.0, float(ref0.K[0, 0]) * self.EFFECT_RADIUS / max(float(pc[2]), 0.1))
        vv, uu = np.mgrid[0:H, 0:W]
        region = (uu - st["effect_x"]) ** 2 + (vv - st["effect_y"]) ** 2 <= r_px ** 2
        n_ch, n_vis = self._effect_change(ref0, cur, region)
        hit = n_vis >= max(self.EFFECT_MIN_VISIBLE, 0.5 * int(region.sum())) and n_ch >= max(40, self.EFFECT_FRAC * n_vis)
        out = {"changed_px": n_ch, "visible_px": n_vis, "change_seen": bool(hit)}
        if hit and self._effect_budget_ok():
            ok, why = self._confirm_effect(st, ref0, cur, 0.0,
                                           progress=f"So far the hand has {'pushed' if motion == 'push' else 'pulled'} "
                                                    f"the part {100 * travel:.0f} cm (the hand's travel, not the part's).")
            out.update(confirmed=ok, s1_evidence=why)
        self.log("effect_after_stroke", **out)
        return out

    # ---------------------------------------------------------------- visible effect of operating a part
    EFFECT_MIN_VISIBLE = 400          # px of the watched area that must be visible in the head view to judge a change
    EFFECT_DE = 30.0                  # colour distance (8-bit Lab) that counts a pixel as changed
    EFFECT_RADIUS = 0.05              # m: watched area around the point where System 1 says the effect appears
    EFFECT_FRAC = 0.10                # changed share of the visible watched area that counts as a change

    def _effect_region(self, ref, part, parent, st):
        """Head-image pixels to watch for the effect. With System 1's effect pixel: a disc of EFFECT_RADIUS around
        the surface point under it (a shiny device mirrors the moving arm elsewhere; the effect itself is local).
        Otherwise the parent device (projected model, filled, grown a little), or the whole image. The part itself
        is never watched (it moves when operated)."""
        H, W = ref.depth.shape
        dev = None
        if st.get("effect_x") is not None and st.get("effect_y") is not None:
            P = ref.pixel_ray_point(st["effect_x"], st["effect_y"])
            if np.all(np.isfinite(P)):
                pc = np.linalg.inv(ref.T_wc) @ np.r_[P, 1.0]
                r_px = max(6.0, float(ref.K[0, 0]) * self.EFFECT_RADIUS / max(float(pc[2]), 0.1))
                vv, uu = np.mgrid[0:H, 0:W]
                dev = (uu - st["effect_x"]) ** 2 + (vv - st["effect_y"]) ** 2 <= r_px ** 2
        if dev is None and parent is not None and len(parent.points):
            dev = ndimage.binary_fill_holes(project_mask(ref, parent.points, radius_px=4))
            dev = ndimage.binary_dilation(dev, iterations=6)
        if dev is None:
            dev = np.ones((H, W), bool)
        if len(part.points):
            dev &= ~ndimage.binary_dilation(project_mask(ref, part.points, radius_px=4), iterations=8)
        return dev

    def _true_robot(self, fr):
        """Robot-labelled pixels that really show the robot: their depth lies on the robot's body (within 4 cm of
        its collision samples at the frame's joint state). Segmentation also labels the robot's reflections in
        shiny surfaces (a metal stove top) with the robot's ids; those pixels are surface, not robot."""
        if "robot_true" in fr.meta:
            return fr.meta["robot_true"]
        m = fr.robot_mask
        out = np.zeros_like(m)
        if m.any():
            rc = np.argwhere(m)
            z = fr.depth[rc[:, 0], rc[:, 1]].astype(float)
            x = (rc[:, 1] + 0.5 - fr.K[0, 2]) / fr.K[0, 0] * z
            y = (rc[:, 0] + 0.5 - fr.K[1, 2]) / fr.K[1, 1] * z
            P = np.stack([x, y, z], 1) @ fr.T_wc[:3, :3].T + fr.T_wc[:3, 3]
            q = np.asarray(fr.q, float)
            W = self.rm.points_world(self.rm.full_q(q[None, :7], max(0.0, float(q[7] - q[8]) / 2)))[0]
            d, _ = cKDTree(W).query(P, distance_upper_bound=0.04)
            on = np.isfinite(d)
            out[rc[on, 0], rc[on, 1]] = True
        fr.meta["robot_true"] = out
        return out

    def _effect_change(self, ref, cur, region):
        """(changed px, visible px) in the watched region between two head frames. The robot's body in either
        frame is excluded with a margin; its reflections are skipped pixel by pixel (no margin: speckles of a
        reflection on a burner must not blank the burner). Single isolated changed pixels are ignored, thin
        structures (glowing rings 2-3 px wide) are kept."""
        body = self._true_robot(ref) | self._true_robot(cur)
        refl = (ref.robot_mask | cur.robot_mask) & ~body
        valid = region & ~ndimage.binary_dilation(body, iterations=6) & ~refl
        a = cv2.cvtColor(np.ascontiguousarray(ref.rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
        b = cv2.cvtColor(np.ascontiguousarray(cur.rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
        changed = valid & (np.linalg.norm(a - b, axis=2) > self.EFFECT_DE)
        nb = ndimage.convolve(changed.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - changed
        changed &= nb >= 2
        return int(changed.sum()), int(valid.sum())

    def _confirm_effect(self, st, ref, cur, turned, progress=None):
        """System 1 judges a before / now head-image pair: is the expected effect really there?"""
        stage = {k: st.get(k) for k in ("id", "op", "object", "motion", "goal", "turn_direction", "effect", "effect_x",
                                        "effect_y")}
        content = self.task_parts() + [
            text_part("Stage being executed: " + json.dumps(stage)),
            text_part(progress or f"So far the part has been turned {abs(np.rad2deg(turned)):.0f} degrees "
                                  f"{'counterclockwise' if turned > 0 else 'clockwise'} (as seen looking at its face)."),
            text_part("IMAGE 1 (head camera, before the part was operated at all):"), image_part(ref.rgb),
            text_part("IMAGE 2 (head camera, now):"), image_part(cur.rgb)]
        t = time.time()
        try:
            data, rec = self.astra.call("effect_check", EFFECT_INSTRUCTIONS, content, EFFECT_SCHEMA)
        except Exception as e:                          # no answer: the change is NOT accepted (caller decides)
            calls = getattr(self.astra, "calls", [])
            self.log("effect_check", result=None, error=repr(e), record=calls[-1] if calls else None)
            return None, "System 1 unavailable: " + repr(e)[:120]
        finally:
            self._tick("model_wait", t)
        self.log("effect_check", result=data, record=rec)
        return bool(data.get("effect_visible")), data.get("evidence")

    def _effect_budget_ok(self):
        """An effect check may call System 1 only while a REPAIR and a VERIFY call remain in the budget."""
        return self.astra.max_calls - len(getattr(self.astra, "calls", [])) >= 3

    def _sense_room(self, g, q_g, ap, n, s, angle=np.deg2rad(45.0)):
        """Joint-limit margin of the grasp turned by s * angle about the axis (< 0: unreachable)."""
        Rr = articulate.rot_about(n, s * angle)
        q, ok, _, _ = self.rm.ik(Rr @ g.R, ap + Rr @ (g.p - ap), np.asarray(q_g)[None], q_rest=q_g, iters=80)
        lim = self.rm.limits[:7]
        return float(np.min(np.minimum(q[0] - lim[:, 0], lim[:, 1] - q[0]))) if ok[0] else -1.0

    def _hold(self, k=4):
        R, p = self.rb.eef_pose()
        for _ in range(k):
            self.rb.step(R, p, grip=1.0)

    def _turn_until_effect(self, st, g, q_g, mf, direction, part, parent):
        """Turn the grasped part in small increments (articulate.TURN_STEP) until the visible effect System 1
        expects shows in the head camera; a quarter turn without it -> back to the start, other sense.

        The effect is measured, never assumed from the hand's rotation: a colour change in the watched area that
        persists over one more increment (also the margin past the switching point), confirmed by System 1 on
        the pre-turn image and the current one. A change System 1 rejects becomes the pixel baseline (S1 always
        sees the true pre-turn image); 2 checks per sense, never past the budget REPAIR / VERIFY need. Without an
        answer from System 1 a change is never accepted: the part stays where it is and the stage is unverified.
        Every increment and turn-back is guarded (grip lost, part pressed against a stop); turn-backs are checked
        with the camera. A hidden watched area (2 blind checks) -> the first sense is completed and kept
        (unverified). Returns (moved, report); report['observed'] is True / None (unverified) / False."""
        n, ap = mf["n"], mf["axis_point"]
        ref0 = self.capture(HEAD_CAMERA)                 # before operating: S1's IMAGE 1 and the undo reference
        self.save_img("effect_ref", ref0.rgb)
        region = self._effect_region(ref0, part, parent, st)
        watched = int(region.sum())
        has_px = st.get("effect_x") is not None and st.get("effect_y") is not None
        F0 = self.rb.wrench()[0]

        def measure(cur, ref):
            n_ch, n_vis = self._effect_change(ref, cur, region)
            seen = n_vis >= max(self.EFFECT_MIN_VISIBLE, 0.5 * watched) if has_px else n_vis >= self.EFFECT_MIN_VISIBLE
            need = max(40, self.EFFECT_FRAC * n_vis) if has_px else 250        # fallback region: fixed effect size
            return n_ch, n_vis, seen, bool(seen and n_ch >= need)

        self._effect_ctx = {"ref0": ref0, "measure": measure}
        self.log("effect_watch", effect=st.get("effect"), pixel=[st.get("effect_x"), st.get("effect_y")],
                 watched_px=watched)
        senses = list(articulate.turn_order(self, g, q_g, ap, n, direction))
        if len(senses) == 1:
            room = self._sense_room(g, q_g, ap, n, -senses[0])
            if room > 0:
                senses.append(-senses[0])
            else:
                self.log("reverse_skipped", sense=-senses[0], joint_margin=room)
        net, turns = 0.0, []
        report = {"expected": st.get("effect"), "pixel": [st.get("effect_x"), st.get("effect_y")], "watched_px": watched}

        def sense_of(a):
            return "counterclockwise" if a > 0 else "clockwise"

        def guard():
            if self.rb.finger_opening() < 0.002:
                return "lost_grip"
            if np.linalg.norm(self.rb.wrench()[0] - F0) > articulate.FORCE_LIMIT:
                return "force"
            return None

        def done(observed, cur=None, **kw):
            moved = {"rotation_deg": round(float(np.rad2deg(net)), 1), "sense": sense_of(net)}
            report.update(observed=observed, turns=turns, **kw)
            if observed and cur is not None:
                self.save_img("effect_seen", cur.rgb)
            return moved, report

        def confirm(cur):
            if not self._effect_budget_ok():
                return None, "no model budget left for an effect check (REPAIR / VERIFY keep theirs)"
            return self._confirm_effect(st, ref0, cur, net)

        def effect_now(note):
            """At a stop (stall, lost grip, end of a sense, after a turn-back): is the change there vs. the start?"""
            self._hold(4)
            cur = self.capture(HEAD_CAMERA)
            n_ch, n_vis, seen, hit = measure(cur, ref0)
            self.log("effect_recheck", note=note, changed_px=n_ch, visible_px=n_vis, hit=hit)
            if not hit:
                return None, cur, n_ch
            ok, why = confirm(cur)
            return (ok, why), cur, n_ch

        for si, s in enumerate(senses):
            if si > 0:
                if abs(net) > np.deg2rad(2.0):
                    back = articulate.twist_about(self, ap, n, -net, settle=6)       # back to where the part started
                    net += back
                    self.log("turn_back", turned_deg=round(float(np.rad2deg(back)), 1),
                             net_deg=round(float(np.rad2deg(net)), 1), stop=guard())
                res, cur, n_ch = effect_now("after turning back")
                if res is not None:
                    ok, why = res
                    if ok:
                        return done(True, cur, confirmed_by_s1=True, s1_evidence=why,
                                    note="the change was still there after turning back")
                    if ok is None:
                        return done(None, cur, unconfirmed_change=True, s1_evidence=why)
                if self.rb.finger_opening() < 0.002:
                    return done(False, None, lost_grip=True, note="grip lost while turning back")
            ref_px, cmd, ach, hits, confirms, blind = ref0, 0.0, 0.0, 0, 0, 0
            info = {"sense": sense_of(s), "checks": 0, "max_changed_px": 0, "min_visible_px": watched,
                    "rejected_changes": 0, "stop": None}
            turns.append(info)
            while abs(cmd) < articulate.MAX_TURN - 1e-6:
                inc = s * min(articulate.TURN_STEP, articulate.MAX_TURN - abs(cmd))
                a = articulate.twist_about(self, ap, n, inc, settle=4)
                cmd, ach, net = cmd + inc, ach + a, net + a
                stop = guard() or ("stall" if abs(cmd) - abs(ach) > articulate.STALL_LAG else None)
                cur = self.capture(HEAD_CAMERA)
                self.save_img("turn_check", cur.rgb)
                n_ch, n_vis, seen, hit = measure(cur, ref_px)
                blind += 0 if seen else 1
                info["checks"] += 1
                info["max_changed_px"] = max(info["max_changed_px"], n_ch)
                info["min_visible_px"] = min(info["min_visible_px"], n_vis)
                info["turned_deg"] = round(float(np.rad2deg(ach)), 1)
                self.log("turn_check", sense=info["sense"], turned_deg=info["turned_deg"], changed_px=n_ch,
                         visible_px=n_vis, hit=hit, stop=stop)
                if si == 0 and info["checks"] == 2 and blind == 2:
                    # the watched area is hidden: cannot judge by sight -> finish this sense and keep it
                    while abs(cmd) < articulate.MAX_TURN - 1e-6 and stop is None:
                        inc = s * min(articulate.TURN_STEP, articulate.MAX_TURN - abs(cmd))
                        a = articulate.twist_about(self, ap, n, inc, settle=4)
                        cmd, ach, net = cmd + inc, ach + a, net + a
                        stop = guard() or ("stall" if abs(cmd) - abs(ach) > articulate.STALL_LAG else None)
                    info.update(turned_deg=round(float(np.rad2deg(ach)), 1), stop=stop, blind=True)
                    return done(None, None, occluded=True)
                hits = hits + 1 if hit else 0
                if stop is not None:
                    info["stop"] = stop
                    if stop == "stall":
                        self.log("twist_stall", commanded=np.rad2deg(cmd), achieved=np.rad2deg(ach))
                    break
                if hits < 2 or confirms >= 2:
                    continue
                confirms += 1
                ok, why = confirm(cur)
                if ok:
                    return done(True, cur, confirmed_by_s1=True, s1_evidence=why, changed_px=n_ch, visible_px=n_vis)
                if ok is None:                               # no answer: never accept, keep the part where it is
                    return done(None, cur, unconfirmed_change=True, s1_evidence=why, changed_px=n_ch)
                ref_px, hits = cur, 0                       # S1: not the effect -> only later changes can count
                info["rejected_changes"] += 1
            # the sense ended (quarter turn, stall, force, lost grip): a change seen at the very end gets its check
            if info["stop"] != "lost_grip" and hits >= 1:
                res, cur, n_ch = effect_now("end of sense")
                if res is not None:
                    ok, why = res
                    if ok:
                        return done(True, cur, confirmed_by_s1=True, s1_evidence=why, changed_px=n_ch)
                    if ok is None:
                        return done(None, cur, unconfirmed_change=True, s1_evidence=why)
            if info["stop"] == "lost_grip":
                res, cur, n_ch = effect_now("grip lost")
                if res is not None and res[0]:
                    return done(True, cur, confirmed_by_s1=True, s1_evidence=res[1], note="the part slipped after the change")
                return done(False, None, lost_grip=True)
        # no sense produced the effect: leave the part as it was found, and check that with the camera
        if abs(net) > np.deg2rad(2.0) and self.rb.finger_opening() >= 0.002:
            back = articulate.twist_about(self, ap, n, -net, settle=6)
            net += back
            self.log("turn_back", turned_deg=round(float(np.rad2deg(back)), 1), net_deg=round(float(np.rad2deg(net)), 1))
        self._hold(4)
        cur = self.capture(HEAD_CAMERA)
        n_ch, n_vis, seen, hit = measure(cur, ref0)
        return done(False, None, returned=bool(seen and not hit) if seen else None, residual_changed_px=n_ch,
                    residual_net_deg=round(float(np.rad2deg(net)), 1))

    def plan_part_grasp(self, part, parent, mf, failed, motion, end_check=None, time_limit=None):
        """time_limit: seconds after which the search over arm postures stops (lrl; None: no bound, as v2)."""
        t_end = np.inf if time_limit is None else time.time() + time_limit
        R0, _ = self.rb.eef_pose()
        cands = articulate.part_grasps(part, mf, R0, max_depth=0.07 if motion in ("push", "pull") else None)
        cands = [c for c in cands if all(np.linalg.norm(c.p - f) > 0.006 for f in failed)]
        self.log("part_grasp_cands", n=len(cands))
        q_now = self.rb.q()[:7]
        excl = [part.oid]          # the fixture stays an obstacle (exact points); only the part may be touched
        why = {}
        # rigid-hand prefilter (no IK), then visit approach families round-robin (best first within each)
        hands = {}
        def hand_bad(g):
            bad = None
            for gap in manip.APPROACH_GAPS:            # widest pre-shape the surroundings allow
                f = preshape_finger(g.width, gap)
                bad = hand_bad_at(g, f)
                if bad is None:
                    g.info["finger"], g.info["gap"] = f, gap
                    return None
            return bad
        def hand_bad_at(g, f):
            f = round(f, 3)
            if f not in hands:
                hands[f] = self._hand_local(f)
            L, grp, rad = hands[f]
            for back in (0.0, 0.04):
                Wh = g.p - g.R[:, 2] * back + L @ g.R.T
                c = self.checker.point_clearance(Wh, tuple(excl)) - rad
                if c[grp == 1].min() < NEAR["palm"] or c[grp == 2].min() < NEAR["finger"]:
                    return "hand_scene"
                zc = Wh[:, 2] - self.table_z - 0.5 * rad
                if zc[grp == 1].min() < NEAR["plane_palm"] or zc[grp == 2].min() < NEAR["plane_finger"]:
                    return "hand_table"
            W0 = g.p + L @ g.R.T
            dp = part.surface_distance(W0) - rad
            if dp[grp == 1].min() < 0.006:
                return "hand_part"
            if parent is not None:
                d2 = parent.surface_distance(W0) - rad
                if d2[grp == 1].min() < 0.008 or d2[grp == 2].min() < 0.003:
                    return "hand_parent"
            return None
        fams = {}
        end_fallback = None
        for g in cands:
            if time.time() > t_end:
                why["search_time_limit"] = 1
                break
            b = hand_bad(g)
            if b:
                why[b] = why.get(b, 0) + 1
                continue
            fams.setdefault((round(g.info["tilt"], 2), round(g.info["pitch"], 2)), []).append(g)
        order = []
        while any(fams.values()) and len(order) < 30:
            for k in list(fams):
                if fams[k]:
                    order.append(fams[k].pop(0))
        self.log("part_grasp_prefilter", kept=len(order), rejected=dict(why))
        for g in order:
            if time.time() > t_end:
                why["search_time_limit"] = 1
                break
            f = g.info["finger"]
            sols = self.ik_clear(g.R, g.p, q_now, exclude=tuple(excl), finger=f)
            if len(sols) == 0:
                why["no_ik"] = why.get("no_ik", 0) + 1
            for q_g in sols[:3]:
                bad = self._part_config_bad(q_g, part, parent, excl, f)
                if bad:
                    why[bad] = why.get(bad, 0) + 1
                    continue
                q_pre = self._ik_near(g.R, g.p - g.R[:, 2] * 0.08, q_g)
                line = self.line_configs(g.R, g.p, q_pre, finger=f)
                if line is None or any(self._part_config_bad(q, part, parent, excl, f) for q in line[::2]):
                    why["approach"] = why.get("approach", 0) + 1
                    continue
                if motion == "rotate" and not self._turn_reachable(q_g, g, mf, excl):
                    why["turn"] = why.get("turn", 0) + 1
                    continue
                m_end = None if end_check is None else end_check(g.R, g.p, line[-1], 0.5 * g.width)
                if m_end is not None and m_end < 0:
                    why["end_blocked"] = why.get("end_blocked", 0) + 1
                    if end_fallback is None or m_end > end_fallback[0]:
                        end_fallback = (m_end, (g, line[0], line[-1]))
                    continue
                self.log("part_grasp", p=g.p, yaw=g.info["yaw"], tilt=g.info.get("tilt"), pitch=g.info.get("pitch"),
                         width=g.width, finger=f, align=g.info.get("align"))
                return g, line[0], line[-1]
        if end_fallback is not None:
            # every grasp would meet the fixture before the part closes: the nearest-to-clear one (the state decides)
            g = end_fallback[1][0]
            self.log("part_grasp", p=g.p, yaw=g.info["yaw"], tilt=g.info.get("tilt"), pitch=g.info.get("pitch"),
                     width=g.width, finger=g.info["finger"], align=g.info.get("align"), end_check_relaxed=True,
                     end_margin=round(end_fallback[0], 4))
            return end_fallback[1]
        self.log("part_grasp_fail", why=why)
        return None

    def _taken(self, g, end, q_g):
        """The grasp the hand went to: g itself, or its 180-degree twin (same pads, hand turned about the approach)
        with the arm posture there."""
        if not getattr(end, "twin", False):
            return g, q_g
        R = np.asarray(end.R, float)
        info = dict(g.info, yaw=float(_wrap_pi(g.info.get("yaw", 0.0) + np.pi)), twin=True)
        return manip.GraspCand(R, g.p, g.width, g.score, info), self._ik_near(R, g.p, self.rb.q()[:7])

    def _turn_reachable(self, q_g, g, mf, excl):
        """At least one turning sense is reachable with the arm clear of the scene along the way."""
        for s in (1.0, -1.0):
            q_prev, good = q_g, True
            for frac in (0.33, 0.67, 1.0):
                Rr = articulate.rot_about(mf["n"], s * frac * articulate.MAX_TURN)
                p = mf["axis_point"] + Rr @ (g.p - mf["axis_point"])
                q, ok, _, _ = self.rm.ik(Rr @ g.R, p, q_prev[None], q_rest=q_prev, iters=80)
                if not ok[0]:
                    good = False
                    break
                c = self.checker.clearance(q, exclude=tuple(excl), finger=0.0)
                if c["arm"][0] < NEAR["arm"]:
                    good = False
                    break
                q_prev = q[0]
            if good:
                return True
        return False

    def _surface_clear(self, q, obj, finger=0.04):
        """Exact distance from each robot group to an object's observed surface points (no 2.5D fill:
        parts may sit on vertical faces and fixtures may overhang)."""
        W, _ = self.checker.world_points(q[None], finger)
        d = obj.surface_distance(W[0]) - self.checker.pr
        return {name: float(d[self.checker.pg == g].min()) for g, name in ((0, "arm"), (1, "palm"), (2, "finger"))}

    def _part_config_bad(self, q, part, parent, excl, finger):
        """None if the configuration is clear (scene exact, part and parent against their own points),
        else the name of the first violated check."""
        good, c = self.checker.ok(q[None], NEAR, exclude=tuple(excl), finger=finger)
        if not good[0]:
            return "scene_" + min((k for k in c if c[k][0] < NEAR.get(k, 0.02)), key=lambda k: c[k][0])
        f = self._surface_clear(q, part, finger)
        if f["palm"] < 0.006 or f["finger"] < 0.001:
            return "part"
        if parent is not None:
            f2 = self._surface_clear(q, parent, finger)
            if f2["palm"] < 0.008 or f2["finger"] < 0.003 or f2["arm"] < 0.01:
                return "parent"
        return None

    # ================================================================ main loop
    def run(self):
        status = "unknown"
        try:
            status = self._run()
        except Exception as e:
            self.log("crash", error=repr(e), tb=traceback.format_exc())
            status = "crash"
        self.log("end", status=status, timing=self.timing, astra=self.astra.stats())
        self._events.close()
        return status

    def _run(self):
        t = time.time()
        head = self.capture(HEAD_CAMERA)
        wrist = self.capture(WRIST_CAMERA)
        self.head0 = head
        self.q_home = self.rb.q()[:7].copy()
        self.save_img("head0", head.rgb)
        self.build_scene(head)
        self.frames = [head, wrist]
        self.integrate([head, wrist])
        self._tick("perceive0", t)
        fut = self.call_program(head, wrist)
        extra = self.call_program_extra(head, wrist, self.compiles - 1)
        t_call = time.time()
        survey = self.survey(head)
        t_wait = time.time()
        prog, rec = fut.result()
        self._tick("model_wait", t_wait)
        self.log("program", program=prog, record=rec, overlap_s=round(t_wait - t_call, 2))
        warn = validate_program(prog)
        if warn:
            self.log("program_warnings", warnings=warn)
        if extra:
            progs = [prog]
            for f in extra:
                try:                                   # an extra compile is optional: a model error only drops it
                    p_, _ = f.result()
                    validate_program(p_)
                    progs.append(p_)
                except (AstraError, ProgramError) as e:
                    self.log("compile_error", error=repr(e))
            prog = merge_compiles(progs)
            self.log("compiles", n=len(progs), terms={s["id"]: s["terms"] for s in prog["stages"]})
        self.program = prog
        failures = self.ground(prog, head, [wrist] + survey, s1_wrist=wrist)
        self.update_support_plane(self.frames)
        self.rebuild_map(self.frames)
        if failures:
            prog = self.repair_loop({"failure": "grounding", "details": failures})
            if prog is None:
                return "grounding_failed"
        stages = list(prog["stages"])
        i = 0
        while i < len(stages):
            if self.elapsed() > self.max_wall_s:
                self.log("timeout")
                return "timeout"
            st = stages[i]
            self.log("stage_start", stage=st)
            try:
                res = self.run_stage_guarded(st)
                self.ledger.append({"stage": st["id"], "status": "done", **_short(res)})
                self.log("stage_done", stage=st["id"], result=res)
                i += 1
            except StageFailure as f:
                self.ledger.append({"stage": st["id"], "status": "failed", "reason": f.reason,
                                    "info": f.info})
                self.log("stage_failed", stage=st["id"], reason=f.reason, info=f.info)
                if self.elapsed() > self.max_wall_s:
                    self.log("timeout")
                    return "timeout"
                new = self.repair_loop({"failed_stage": st, "reason": f.reason, "info": f.info,
                                        "robot_state": self.robot_state()})
                if new is None:
                    return "failed"
                stages = list(new["stages"])
                i = 0
        self.cur_stage = None                      # moves outside a stage carry no stage terms
        # VERIFY
        if self.do_verify:
            self.retreat_home()
            v = self.call_verify()
            if v.get("complete") is False and self.repairs < self.max_repairs:
                new = self.repair_loop({"failure": "verify", "problem": v.get("problem"), "hint": v.get("hint")})
                if new is not None:
                    for st in new["stages"]:
                        try:
                            res = self.run_stage_guarded(st)
                            self.ledger.append({"stage": st["id"], "status": "done", **_short(res)})
                        except StageFailure as f:
                            self.ledger.append({"stage": st["id"], "status": "failed", "reason": f.reason})
                            return "failed_after_verify"
                    self.retreat_home()
        else:
            self.retreat_home()
        self.finished = True
        return "finished"

    def robot_state(self):
        held_ids = [s for s, o in self.id_map.items() if o == self.held] if self.held is not None else []
        return {"gripper": ("holding " + self.objects[self.held].name + f" (id {', '.join(held_ids)})")
                if self.held is not None else "empty",
                "objects_last_seen": {m.name: {"center_xy": np.round(m.footprint_center(), 3).tolist(),
                                               "top_z_above_table": round(m.top_z - self.table_z, 3)}
                                      for m in self.objects.values() if len(m.points)}}

    def repair_loop(self, report):
        if self.repairs >= self.max_repairs:
            return None
        if self.dropped_terms:
            report = {**report, "dropped_terms": self.dropped_terms}
            self.dropped_terms = []
        self.repairs += 1
        try:
            data, head = self.call_repair(report)
            if not data.get("stages"):
                # System 1 judges that nothing remains to be done
                self.log("repair_nothing_left", diagnosis=data.get("diagnosis"))
                self.program = data
                return data
            validate_program(data)
        except (AstraError, ProgramError) as e:
            self.log("repair_error", error=repr(e))
            return None
        self.program = data
        self.head0 = head
        survey = []
        failures = self.ground(data, head, [f for f in self.frames[-8:] if f.name == WRIST_CAMERA] + survey,
                               s1_wrist=getattr(self, "_repair_wrist", None))
        self.frames.append(head)
        self.rebuild_map(self.frames[-8:])
        if failures:
            self.log("repair_ground_fail", failures=failures)
        return data

    def park_held(self):
        """Put the held object down on a free table spot near its original place (collision-checked)."""
        obj = self.objects[self.held]
        R = self.head0.T_wc[:3, :3]
        lo = np.array([self.rm.root_p[0] + 0.30, -0.35])
        hi = np.array([self.rm.root_p[0] + 0.75, 0.35])
        g = np.stack(np.meshgrid(np.arange(lo[0], hi[0], 0.01), np.arange(lo[1], hi[1], 0.01)), -1).reshape(-1, 2)
        hmap, org = self.scene.heightmap((lo[0] - 0.1, hi[0] + 0.1), (lo[1] - 0.1, hi[1] + 0.1), exclude=(obj.oid,))
        ij = np.argwhere(hmap > self.table_z + 0.006)
        block = np.array(org) + (ij + 0.5) * self.scene.res if len(ij) else None
        reg = {"floor_z": self.table_z, "rim_z": self.table_z, "grid_xy": g, "clearance": np.full(len(g), 0.1),
               "kind": "on"}
        table = ObjectModel(-1, "table", "table", "context", self.table_z - 0.01)
        table.set_points(np.c_[g, np.full(len(g), self.table_z)])
        Rs, ps = self.rb.eef_pose()
        res = {"attempts": []}
        st = {"relation": "on", "tolerance": "loose"}
        self.log("park", object=obj.name)
        self.place_object(obj, table, st, np.r_[ps[:2], self.table_z], res, region_override=reg)

    def _safe_put_down(self):
        """After a failure while holding an object: lower it back near where it is and release."""
        try:
            Rs, ps = self.rb.eef_pose()
            pay = self._payload_world(Rs, ps)
            drop = pay[:, 2].min() - (self.table_z + 0.01)
            self.move_line(Rs, ps - np.array([0, 0, max(0.0, drop)]), grip=1.0, speed=0.05)
            self.ex.gripper(-1.0)
            obj = self.objects[self.held]
            self._finish_release(obj, self.table_z)
            Rs2, ps2 = self.rb.eef_pose()
            self.move_line(Rs2, ps2 + np.array([0, 0, 0.1]), grip=-1.0, speed=0.1, finger=0.04)
            head = self.capture(HEAD_CAMERA)
            self.frames.append(head)
            self.refresh_object(obj, [head], [obj.category, obj.name])
            self.rebuild_map(self.frames[-8:])
            self.log("put_down", object=obj.name, center=obj.footprint_center())
        except Exception as e:
            self.log("put_down_error", error=repr(e))

    def retreat_home(self):
        """Lift the hand to a clear height above the workspace (end pose for verification)."""
        Rs, ps = self.rb.eef_pose()
        target = np.array([ps[0], ps[1], max(ps[2], self.table_z + 0.30)])
        q = self.rm.solve_ik(Rs, target, self.rb.q()[:7], n_random=6)
        if len(q):
            self.go_configs(list(q[:2]), NEAR, grip=-1.0, finger=0.04, tag="home")


def _turn_angle(R0, R1):
    """Rotation angle (rad) between two hand orientations."""
    return float(np.arccos(np.clip((np.trace(np.asarray(R0).T @ np.asarray(R1)) - 1.0) / 2.0, -1.0, 1.0)))


def _short(res):
    if not isinstance(res, dict):
        return {}
    keep = {k: v for k, v in res.items() if k in ("object", "reference", "relation", "relation_check", "measured",
                                                  "direction", "motion")}
    return json.loads(json.dumps(keep, default=_jsonable))


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    if isinstance(o, (np.floating,)):
        return round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return repr(o)
