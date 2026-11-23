#!/usr/bin/env python
"""Task dashboard video for one CP-5 episode, rendered after the episode from <run>/rec + events.jsonl.

  render_task_video.py RUN_DIR [--out PATH]                  -> RUN_DIR/task_video.mp4 (1600x960, 10 fps, H.264)
  render_task_video.py RUN_DIR --frames I,J,K --png-dir DIR   -> only those timeline items as PNGs (review mode)

The frame list is vis_timeline.build_timeline(load_run(run)) with default arguments, shared with the 3D obstacle-map
video, so the two videos are frame-aligned (same length, same hold frames). Visual language follows the earlier
harness dashboard: dark theme, agent / wrist cameras on the left, program panel, three world projections, phase
bar and a telemetry panel. Nothing here is read by the policy; it only reads what the episode recorded.
"""
from __future__ import annotations

import argparse
import bisect
import math
import shutil
import subprocess
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vis_timeline import (PHASE_LABELS, PHASES, S1_EVENTS, build_timeline, current_map,  # noqa: E402
                          current_plan, events_upto, gap_label, load_run, phase_of, program_state)

W, H = 1600, 960
FPS = 10
VIDEO_NAME = "task_video.mp4"
SIM_DT = 0.05                      # simulator step (s)

# ---------------------------------------------------------------- palette (earlier harness dashboard)
BG = (12, 18, 30)
PANEL = (22, 31, 47)
EDGE = (44, 56, 74)
TEXT = (232, 239, 250)
MUTED = (154, 173, 197)
DIM = (110, 126, 150)
CYAN = (59, 201, 236)
YELLOW = (255, 220, 84)
GREEN = (102, 227, 161)
RED = (250, 129, 125)
ORANGE = (225, 150, 60)            # object boxes / projection titles
ORANGE_DIM = (128, 92, 52)         # box of a dropped object that S2 has not found again yet (last seen in the hand)
PLAN_YELLOW = (245, 210, 70)       # planned EEF path
PINK = (225, 95, 175)              # measured EEF path (current stage)
PINK_DIM = (128, 66, 110)          # measured EEF path (earlier stages)
GRID = (31, 43, 62)
TABLE_LINE = (100, 116, 140)
MAP_COL = (78, 104, 140)           # S2 voxel map (occupied, above the table)
ACTIVE_FILL = (32, 91, 90)
OK_FILL = (17, 50, 41)
BAD_FILL = (62, 25, 31)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

# ---------------------------------------------------------------- layout (reference dashboard geometry, plots taller)
CAMS = ((139, "AGENT VIEW (external)", "head camera, upright", "agent"),
        (535, "WRIST VIEW", "eye-in-hand camera, upright", "wrist"))
CAM_BOX_W, CAM_BOX_H = 352, 336
TOP = (425, 139, 1576, 389)
TOP_LIST_Y = 250                   # first row of the stage list / S1 calls
TOP_LIST_B = 381                   # lowest text baseline area of the top panel
COL_X = 1124                       # divider between stage list and S1 calls
# (panel box, (horizontal axis, vertical axis), vertical axis points down, title, subtitle).
# XY is drawn in head-camera orientation (+y right, +x down toward the camera, robot base at the top) so the
# agent view, XY and YZ share left / right.
PROJ = (((425, 401, 795, 662), (1, 0), True, "XY (top)", "camera orientation: robot base at the top"),
        ((815, 401, 1185, 662), (0, 2), False, "XZ (side)", ""),
        ((1205, 401, 1576, 662), (1, 2), False, "YZ (front)", "as the head camera sees it: y to the right"))
PHASE_Y = (674, 717)
BOTTOM = (425, 729, 1576, 916)
LABEL_FONT = 11
# label-placement costs (object box labels in the projections)
# (effectively lexicographic: other labels > markers > plan end > 'table top' caption > box edges > paths)
COST_LABEL, COST_MARKER, COST_PLAN_END, COST_SOFT = 1000.0, 500.0, 100.0, 20.0
COST_EDGE, COST_INSIDE = 10.0, 0.25
PATH_COST, PATH_CAP = 0.007, 1.5   # per covered path-mask unit (bright path / plan pixel = 3, earlier stages = 1)
# keep last frame's label spot unless another is better by more than this: covering a path never moves a label;
# a label, marker or box-edge crossing does (a held object's label, moving with its box, tolerates one edge)
STICKY_SLACK, STICKY_SLACK_HELD = 1.6, 11.6
# events that share a timestamp: the most informative one leads the status line (negative = omit when grouped)
EVENT_RANK = {"end": 9, "crash": 9, "timeout": 9, "stage_failed": 8, "actuated": 7, "placed": 7, "dropped": 7,
              "relocated": 6, "grasp_plan": 6, "place_plan": 6, "part_grasp": 6, "part_grasp_fail": 6,
              "payload": 5, "close": 5, "held": 5, "descend": 5, "blocked_descent": 5, "twist_stall": 5,
              "place_descend": 5, "stage_done": 1, "stage_start": 1, "payload_centering": -1,
              "gripper_geometry": -1, "recorder_error": -1}
NOT_S2 = ("gripper_geometry", "recorder_error")

MOVE_PHRASE = {"look": "look move: wrist camera to a view point", "look_toward": "look move toward an object",
               "side_look_back": "return from a side look", "to_part": "approach to the part",
               "to_pregrasp": "transit to pre-grasp", "to_preplace": "transit to place", "retreat": "retreat",
               "home": "retreat to the home pose", "move": "move"}
MOVE_SHORT = {"look": "LOOK MOVE", "look_toward": "LOOK MOVE", "side_look_back": "SIDE-LOOK RETURN",
              "to_part": "APPROACH TO THE PART", "to_pregrasp": "TRANSIT TO PRE-GRASP",
              "to_preplace": "TRANSIT TO PLACE", "retreat": "RETREAT", "home": "RETREAT HOME", "move": "MOVE"}
# what the harness does right after a completed move with this tag (the next event closes it)
MOVE_FOLLOW = {"to_pregrasp": ("force-guarded descent, then closing the gripper", "GRASP DESCENT + CLOSE"),
               "to_part": ("grasping and operating the part", "OPERATING THE PART"),
               "to_preplace": ("force-guarded descent to the placement", "PLACE DESCENT"),
               "look": ("capturing RGB-D views", "CAPTURING VIEWS"),
               "look_toward": ("capturing RGB-D views", "CAPTURING VIEWS")}
HOW_PHRASE = {"direct": "direct route", "lift": "lift route", "rrt": "RRT route"}
PLAN_HOLDS = ("S2 grasp planning", "S2 placement planning", "S2 part-grasp planning", "S2 motion planning")
PERCEIVE_HOLDS = ("S2 grounding", "S2 checking")


# ================================================================ small helpers
@lru_cache(maxsize=64)
def font(size, bold=False, mono=False):
    name = "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    for cand in (str(FONT_DIR / name), name):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(text, f, width):
    """Truncate `text` with an ellipsis so it fits `width` pixels."""
    text = str(text)
    if width <= 0:
        return ""
    if f.getlength(text) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if f.getlength(text[:mid].rstrip() + "…") <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


def wrap(text, f, width, max_lines):
    """Greedy word wrap by pixel width; the last line ends with an ellipsis when text is left over."""
    words = str(text).split()
    lines, cur, i = [], "", 0
    while i < len(words) and len(lines) < max_lines:
        cand = words[i] if not cur else cur + " " + words[i]
        if f.getlength(cand) <= width:
            cur, i = cand, i + 1
            continue
        if not cur:
            lines.append(fit(words[i], f, width))
            i += 1
        else:
            lines.append(cur)
            cur = ""
    if cur and len(lines) < max_lines:
        lines.append(cur)
        cur = ""
    if (i < len(words) or cur) and lines:
        rest = " ".join(([cur] if cur else []) + words[i:])
        lines[-1] = fit(lines[-1] + " " + rest, f, width)
        if not lines[-1].endswith("…"):
            lines[-1] = fit(lines[-1] + " …", f, width)
    return lines


def runs(draw, x, y, parts):
    """Draw [(text, font, colour), ...] left to right; returns the end x."""
    for text, f, col in parts:
        if text:
            draw.text((x, y), text, font=f, fill=col)
            x += f.getlength(text)
    return x


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def vec(v, d=3):
    try:
        return "[" + ", ".join(f"{float(a):+.{d}f}" for a in v) + "]"
    except (TypeError, ValueError):
        return "[?]"


def deg(rad):
    return math.degrees(rad) if isnum(rad) else float("nan")


def mm(m, d=1):
    return f"{m * 1000:.{d}f} mm" if isnum(m) else "? mm"


def libero_success(s):
    """(success, detail) from summary.json: LIBERO's rule, the first control step where check_success() is true
    (run_episode.py since 2026-11-20). Older runs only have the finish / after-hold checks (strict = both)."""
    if "success" in s:
        return s["success"] is True, (f"LIBERO done at step {s.get('success_step')}" if s["success"]
                                      else "LIBERO done never reached")
    fin, hold = s.get("raw_success_at_finish"), s.get("raw_success_after_hold")
    yn = {True: "yes", False: "no"}
    return fin is True and hold is True, f"old rule, at finish: {yn.get(fin, 'n/a')}, after the hold: {yn.get(hold, 'n/a')}"


def short_cat(cat):
    """Short object name for labels: the S1 category if short, else its last two words."""
    cat = str(cat or "").strip()
    if len(cat) <= 16:
        return cat
    words = cat.split()
    return " ".join(words[-2:]) if len(words) > 2 else cat


def pts3(v):
    try:
        a = np.asarray(v, float).reshape(-1, 3)
        return a[np.isfinite(a).all(1)]
    except (TypeError, ValueError):
        return np.empty((0, 3))


def overlaps(r, q, pad=1):
    return not (r[2] + pad <= q[0] or r[0] >= q[2] + pad or r[3] + pad <= q[1] or r[1] >= q[3] + pad)


# ================================================================ camera streams
class CamReader:
    """Sequential reader of an upright camera stream; keeps the last decoded frame (hold frames repeat it)."""

    def __init__(self, path):
        self.cap = None
        self.k = -1
        self.img = None
        self.ended = False
        try:
            if Path(path).exists():
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    self.cap = cap
        except Exception:
            self.cap = None

    def get(self, k):
        """(RGB image resized for the panel or None, stale flag)."""
        if self.cap is None:
            return None, False
        if k < self.k:                                   # never needed for a sequential timeline
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.k, self.img, self.ended = -1, None, False
        while self.k < k and not self.ended:
            if self.k + 1 < k:
                if not self.cap.grab():
                    self.ended = True
                    break
                self.k += 1
                continue
            ok, fr = self.cap.read()
            if not ok or fr is None:
                self.ended = True
                break
            self.k += 1
            h, w = fr.shape[:2]
            r = min(CAM_BOX_W / w, CAM_BOX_H / h)
            fr = cv2.resize(fr, (max(1, round(w * r)), max(1, round(h * r))),
                            interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_NEAREST)
            self.img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        return self.img, self.k < k


# ================================================================ world -> panel projection
class Projection:
    """Equal-aspect orthographic projection of world axes (a, b) into a plot rectangle (true scale). Axis a runs to
    the right; axis b runs up, or down when `vdown` (XY in head-camera orientation: +x toward the camera)."""

    def __init__(self, rect, axes, lo, hi, vdown=False):
        self.rect = tuple(int(v) for v in rect)
        self.axes = tuple(axes)
        self.vdown = bool(vdown)
        x0, y0, x1, y1 = self.rect
        self.size = (x1 - x0, y1 - y0)
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        span = np.maximum(hi - lo, 0.05)
        c = (lo + hi) / 2
        self.scale = min(self.size[0] / span[0], self.size[1] / span[1])      # px per metre on both axes
        span = np.array(self.size, float) / self.scale
        self.lo, self.hi = c - span / 2, c + span / 2

    def vpix(self, val):
        """Vertical pixel (from the plot's top edge) of a value on the vertical axis."""
        d = (np.asarray(val, float) - self.lo[1]) * self.scale
        return d if self.vdown else self.size[1] - d

    def vworld(self, rows):
        """Vertical-axis world value at pixel rows (inverse of vpix)."""
        rows = np.asarray(rows, float)
        return self.lo[1] + rows / self.scale if self.vdown else self.hi[1] - rows / self.scale

    def local(self, P):
        """World points -> pixel coordinates inside the plot rectangle (origin at its top-left)."""
        P = np.asarray(P, float)
        a, b = P[..., self.axes[0]], P[..., self.axes[1]]
        return np.stack(((a - self.lo[0]) * self.scale, self.vpix(b)), -1)

    def lpt(self, p):
        u, v = self.local(np.asarray(p, float)[None])[0]
        return float(u), float(v)


# ================================================================ the dashboard
def tolerant_load_run(run_dir):
    """Same structure as vis_timeline.load_run, but a corrupt JSON file is treated as missing (used only when
    load_run itself raises, e.g. on a summary.json truncated by a crash; telemetry/events parse identically)."""
    import json

    def obj(p):
        try:
            v = json.loads(p.read_text())
            return v if isinstance(v, dict) else None
        except (OSError, ValueError):
            return None

    def jl(p):
        out = []
        try:
            lines = p.read_text().splitlines() if p.exists() else []
        except OSError:
            lines = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
        return out

    d = Path(run_dir)
    rec = d / "rec"
    run = {"dir": d, "rec": rec, "meta": obj(rec / "meta.json") or {}, "telemetry": jl(rec / "telemetry.jsonl"),
           "plans": jl(rec / "plans.jsonl"), "maps": jl(rec / "maps" / "index.jsonl"), "events": jl(d / "events.jsonl"),
           "summary": obj(d / "summary.json"), "args": obj(d / "args.json") or {}}
    run["event_t"] = [float(e.get("t", 0.0)) for e in run["events"]]
    run["plan_t"] = [float(p.get("t", 0.0)) for p in run["plans"]]
    run["map_t"] = [float(m.get("t", 0.0)) for m in run["maps"]]
    return run


class Dashboard:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        try:
            self.run = load_run(self.dir)
        except Exception as e:
            print(f"[render_task_video] load_run failed ({e!r}); continuing with corrupt files ignored",
                  file=sys.stderr)
            self.run = tolerant_load_run(self.dir)
        self.tl = build_timeline(self.run)
        self.meta = self.run.get("meta") or {}
        self.effort = self.meta.get("effort") or (self.run.get("args") or {}).get("effort") or "?"
        self.events = self.run.get("events") or []
        self.tel = self.run.get("telemetry") or []
        self.tel_t = [float(r.get("t", 0.0)) for r in self.tel]
        self.summary = self.run.get("summary") if isinstance(self.run.get("summary"), dict) else None
        self.n_frames = len(self.tel)
        self.errors = {}
        self.cams = {}
        self.eef = np.array([r.get("eef") if isinstance(r.get("eef"), (list, tuple)) and len(r.get("eef")) == 3
                             else [np.nan] * 3 for r in self.tel], float).reshape(-1, 3)
        self.table_z = self._table_z()
        self._prep_s1()
        self._prep_objects()
        self._prep_ranges()
        self.map_cache = {}
        self.plot_base = {}
        self.label_mem = {}                     # (panel, object key) -> label candidate used on the previous frame
        self.table_text = {}                    # panel -> rect of the 'table top' caption (labels avoid it)
        for j in range(len(self.proj)):
            self._plot_base_img(j)
        self.bg = self._background()

    # ------------------------------------------------------------ preparation
    def _table_z(self):
        if isnum(self.meta.get("table_z")):
            return float(self.meta["table_z"])
        for name, key in (("support_plane", "resting"), ("scene", "table_z")):
            for e in self.events:
                if e.get("event") == name and isnum(e.get(key)):
                    return float(e[key])
        return None

    def _prep_s1(self):
        """Every S1 (Astra) call with its in-flight interval on the harness clock (end - latency .. end)."""
        self.s1 = []
        for e in self.events:
            ev = e.get("event")
            if ev not in S1_EVENTS:
                continue
            rec = e.get("record") if isinstance(e.get("record"), dict) else {}
            lat = rec.get("latency_s")
            t_end = float(e.get("t", 0.0))
            self.s1.append({"kind": ev, "t_end": t_end, "lat": lat if isnum(lat) else None,
                            "t_start": max(0.0, t_end - lat) if isnum(lat) else None,
                            "ok": str(rec.get("status", "ok")) == "ok" and rec.get("ok", True) is not False,
                            "n": rec.get("n") if isnum(rec.get("n")) else len(self.s1) + 1, "event": e})
        self.end_event = next((e for e in reversed(self.events) if e.get("event") == "end"), None)
        self.survey_t = next((float(e.get("t", 0)) for e in self.events if e.get("event") == "survey"), None)

    def _prep_objects(self):
        """Box updates per tracked object (harness oid): grounded / placed / relocated events, released payloads."""
        self.by_name = {}                       # object name -> oid key
        self.sid_cat = {}                       # S1 id -> category (prompts[0])
        self.box_updates = {}                   # key -> sorted [(t, lo, hi, kind)]
        self.obj_info = {}                      # key -> {sid, cat, name, role, size, how}
        self.drop_t = {}                        # key -> times of 'dropped' events
        for e in self.events:
            ev = e.get("event")
            t = float(e.get("t", 0.0))
            if ev == "grounded" and isinstance(e.get("summary"), dict):
                s = e["summary"]
                try:
                    c = np.asarray(s.get("centroid"), float).reshape(3)
                    sz = np.asarray(s.get("size"), float).reshape(3)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(np.r_[c, sz]).all():
                    continue
                lo, hi = c - sz / 2, c + sz / 2
                if isnum(s.get("top_z")):                         # exact z extent of the measured bbox
                    hi[2] = float(s["top_z"])
                    lo[2] = hi[2] - sz[2]
                key = e.get("oid") if e.get("oid") is not None else e.get("object")
                info = self.obj_info.setdefault(key, {})
                info.update(sid=e.get("object"), cat=s.get("category") or s.get("name"), name=s.get("name"),
                            role=s.get("role"), size=sz, how=e.get("how"))
                if s.get("name"):
                    self.by_name[s["name"]] = key
                if e.get("object") is not None:
                    self.sid_cat[e.get("object")] = s.get("category") or s.get("name")
                self.box_updates.setdefault(key, []).append((t, lo, hi, "grounded"))
            elif ev == "ground_alias" and e.get("oid") in self.obj_info:
                self.obj_info[e["oid"]].setdefault("aliases", []).append((t, e.get("object")))
                if e.get("object") is not None:
                    self.sid_cat.setdefault(e.get("object"), self.obj_info[e["oid"]].get("cat"))
        for e in self.events:
            ev = e.get("event")
            t = float(e.get("t", 0.0))
            key = self.by_name.get(e.get("object")) if isinstance(e.get("object"), str) else None
            if key is None or key not in self.obj_info:
                continue
            sz = self.obj_info[key]["size"]
            if ev == "placed" and isinstance(e.get("check"), dict):
                ck = e["check"]
                cxy, bot = ck.get("center"), ck.get("bottom")
                if isinstance(cxy, (list, tuple)) and len(cxy) == 2 and all(map(isnum, cxy)) and isnum(bot):
                    lo = np.array([cxy[0] - sz[0] / 2, cxy[1] - sz[1] / 2, bot])
                    self.box_updates[key].append((t, lo, lo + sz, "placed"))
            elif ev == "relocated":
                cxy, top = e.get("center"), e.get("top")
                if isinstance(cxy, (list, tuple)) and len(cxy) == 2 and all(map(isnum, cxy)) and isnum(top):
                    lo = np.array([cxy[0] - sz[0] / 2, cxy[1] - sz[1] / 2, top - sz[2]])
                    self.box_updates[key].append((t, lo, lo + sz, "relocated"))
            elif ev == "dropped":
                self.drop_t.setdefault(key, []).append(t)
        # held payload per recorded frame (world points of the attached collision model)
        self.payload_bb = {}
        try:
            with np.load(self.dir / "rec" / "robot_points.npz") as z:
                for n in z.files:
                    if n.startswith("payload_") and n[8:].isdigit():
                        a = np.asarray(z[n], np.float32).reshape(-1, 3)
                        a = a[np.isfinite(a).all(1)]
                        if len(a):
                            self.payload_bb[int(n[8:])] = (a.min(0).astype(float), a.max(0).astype(float))
        except Exception:
            self.payload_bb = {}
        # held object per frame; a released object stays where its payload was last seen until a newer event
        self.held_k = [r.get("held") for r in self.tel]
        self.held_from = {}
        start = None
        for k, h in enumerate(self.held_k):
            prev = self.held_k[k - 1] if k else None
            if h is not None and h != prev:
                start = k
            if h is not None:
                self.held_from[k] = start
            if h is None and prev is not None and prev in self.obj_info:
                bb = self.payload_bb.get(k - 1)
                if bb is None:                       # no payload points: the box as carried with the hand
                    ups = [u for u in self.box_updates.get(prev, []) if u[0] <= self.tel_t[k - 1]]
                    d = self.eef[k - 1] - self.eef[self.held_from.get(k - 1, k - 1)]
                    if ups and np.isfinite(d).all():
                        u = max(ups, key=lambda x: x[0])
                        bb = (u[1] + d, u[2] + d)
                if bb is not None:
                    self.box_updates.setdefault(prev, []).append((self.tel_t[k], bb[0], bb[1], "released"))
        for key in self.box_updates:
            self.box_updates[key].sort(key=lambda u: u[0])

    def objects(self, t, k):
        """Object boxes at time t (recorded frame k): [{key, sid, label, lo, hi, held}]."""
        out = []
        held = self.held_k[k] if 0 <= k < len(self.held_k) else None
        for key, ups in self.box_updates.items():
            info = self.obj_info.get(key)
            if info is None or not ups:
                continue
            first = next((u[0] for u in ups if u[3] == "grounded"), None)
            i = bisect.bisect_right([u[0] for u in ups], t)
            if first is None or first > t or i == 0:
                continue
            t_up, lo, hi, kind = ups[i - 1]
            sid = info.get("sid")
            for ta, s in info.get("aliases", []):
                if ta <= t:
                    sid = s
            is_held = held is not None and held == key
            # released by a drop and not re-perceived yet: the box is only where the hand last carried it
            dropped = (not is_held and kind == "released"
                       and any(abs(td - t_up) <= 1.0 and td <= t for td in self.drop_t.get(key, ())))
            if is_held:
                bb = self.payload_bb.get(k)
                if bb is not None:
                    lo, hi = bb
                else:                                            # no payload points: carry the box with the hand
                    d = self.eef[k] - self.eef[self.held_from.get(k, k)]
                    if np.isfinite(d).all():
                        lo, hi = lo + d, hi + d
            label = (f"{sid} " if sid else "") + short_cat(info.get("cat"))
            label += " (held)" if is_held else (" (dropped, last seen)" if dropped else "")
            out.append({"key": key, "sid": sid, "label": label, "lo": np.asarray(lo, float),
                        "hi": np.asarray(hi, float), "held": is_held, "dropped": dropped})
        return out

    def _prep_ranges(self):
        """Fixed plot ranges for the whole video (no jitter): everything the plots show, plus a margin. Height is
        capped 0.35 m above the highest object so a single high look move does not shrink the scene."""
        groups = [self.eef]
        for p in self.run.get("plans") or []:
            groups.append(pts3(p.get("eef")))
        tops = []
        for ups in self.box_updates.values():
            for _, lo, hi, _ in ups:
                groups.append(np.array([lo, hi]))
                tops.append(hi[2])
        for bb in self.payload_bb.values():
            groups.append(np.array(bb))
        for e in self.events:
            if e.get("event") in ("grasp_plan", "part_grasp"):
                groups.append(pts3(e.get("p")))
                tops += list(pts3(e.get("p"))[:, 2])
        groups = [g for g in groups if len(g)]
        allp = np.concatenate(groups) if groups else np.empty((0, 3))
        allp = allp[np.isfinite(allp).all(1)] if len(allp) else allp
        try:
            mlo = np.asarray(self.meta.get("map_lo"), float).reshape(3)
            mhi = mlo + np.asarray(self.meta.get("map_shape"), float).reshape(3) * float(self.meta.get("map_res"))
        except (TypeError, ValueError):
            mlo, mhi = np.array([-1.2, -1.0, 0.5]), np.array([0.8, 1.0, 2.0])
        if len(allp):
            allp = np.clip(allp, mlo, mhi)
            lo, hi = allp.min(0), allp.max(0)
        else:
            lo, hi = np.array([-0.4, -0.3, 0.85]), np.array([0.2, 0.3, 1.3])
        if self.table_z is not None:
            lo[2] = min(lo[2], self.table_z - 0.02)
        lo, hi = lo - 0.035, hi + 0.035
        if tops:
            hi[2] = max(min(hi[2], max(tops) + 0.35), lo[2] + 0.25)
        self.proj = []
        for (L, T, R, B), axes, vdown, _, _ in PROJ:
            self.proj.append(Projection((L + 46, T + 30, R - 14, B - 26), axes, lo[list(axes)], hi[list(axes)],
                                        vdown))

    # ------------------------------------------------------------ static background
    def _background(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        m = self.meta
        task = m.get("task") or (self.summary or {}).get("task") or "?"
        ep = m.get("episode", (self.summary or {}).get("episode"))
        try:
            ep_s = f"ep{int(ep):03d}"
        except (TypeError, ValueError):
            ep_s = "ep?"
        effort = self.effort
        d.text((24, 14), "ASTRA HARNESS V5 | CP-5: PROGRAM ONCE, SOLVE LOCALLY", font=font(24, True), fill=TEXT)
        sub = ([f"build {m['build']}"] if m.get("build") else []) + [f"{task} {ep_s}", f"Astra {effort}"]
        d.text((24, 49), fit(" | ".join(sub), font(17), 1540), font=font(17), fill=MUTED)
        instr = m.get("instruction") or (self.summary or {}).get("language") or ""
        d.text((24, 74), fit(instr, font(21), 1550), font=font(21), fill=TEXT)
        if m.get("hint"):
            x = runs(d, 24, 108, [("Operator note: ", font(14, True), MUTED)])
            d.text((x, 108), fit(m["hint"], font(14), 1570 - x), font=font(14), fill=MUTED)
        for y, label, caption, _ in CAMS:
            d.rounded_rectangle((24, y, 401, y + 381), radius=12, fill=PANEL)
            d.text((40, y + 6), label, font=font(16, True), fill=TEXT)
            size = m.get("image_size", 512)
            d.text((40, y + 26), f"{caption}  |  RGB {size} x {size}", font=font(12), fill=MUTED)
        d.rounded_rectangle(TOP, radius=12, fill=PANEL)
        d.text((444, 151), "S1 PROGRAM / S2 STAGES", font=font(19, True), fill=CYAN)
        d.line((444, 241, 1558, 241), fill=EDGE, width=1)
        d.line((COL_X, TOP_LIST_Y, COL_X, TOP[3] - 10), fill=EDGE, width=1)
        for (box, _, _, title, sub), pr in zip(PROJ, self.proj):
            self._proj_static(d, box, title, sub, pr)
        d.rounded_rectangle(BOTTOM, radius=12, fill=PANEL)
        self._legend(d)
        d.text((24, 943), "Rendered after the episode from recorded telemetry; S2 used only RGB-D + proprioception. "
                          "Hold frames: S2 planning or Astra waits (simulator paused).", font=font(12), fill=MUTED)
        return img

    def _legend(self, d):
        f = font(13)
        x, y = 24, 922
        cy = y + 8
        items = [("box", ORANGE, "S2-grounded objects (S1 ids)"), ("map", MAP_COL, "S2 voxel map (occupied)"),
                 ("line", PLAN_YELLOW, "planned EEF path"), ("line", PINK, "measured EEF path (bright: this stage)"),
                 ("dot", TEXT, "current EEF"), ("ring", GREEN, "grasp target"), ("diamond", CYAN, "place target"),
                 ("outline", YELLOW, "S1 call in flight"), ("outline", GREEN, "S2 work inside an S1 hold")]
        for kind, col, label in items:
            if kind == "box":
                d.rectangle((x, cy - 5, x + 14, cy + 5), outline=col, width=2)
            elif kind == "map":
                d.rectangle((x, cy - 5, x + 14, cy + 5), fill=col)
            elif kind == "line":
                d.line((x, cy, x + 16, cy), fill=col, width=3)
            elif kind == "dot":
                d.ellipse((x + 2, cy - 5, x + 12, cy + 5), fill=col)
            elif kind == "ring":
                d.ellipse((x + 1, cy - 6, x + 13, cy + 6), outline=col, width=2)
            elif kind == "diamond":
                d.polygon([(x + 7, cy - 7), (x + 14, cy), (x + 7, cy + 7), (x, cy)], outline=col)
            elif kind == "outline":
                d.rounded_rectangle((x, cy - 6, x + 16, cy + 6), radius=3, outline=col, width=2)
            x += 22
            d.text((x, y), label, font=f, fill=MUTED)
            x += f.getlength(label) + 14

    def _proj_static(self, d, box, title, sub, pr):
        L, T, R, B = box
        axes = pr.axes
        d.rounded_rectangle(box, radius=12, fill=PANEL)
        x = runs(d, L + 12, T + 8, [(title, font(14, True), ORANGE)])
        if sub:
            d.text((x + 8, T + 11), fit(sub, font(11), R - 12 - x - 8), font=font(11), fill=MUTED)
        x0, y0, x1, y1 = pr.rect
        names = "xyz"
        d.rectangle((x0, y0, x1, y1), outline=EDGE)
        f = font(10)
        hstep = 0.1 if 0.1 * pr.scale >= 44 else 0.2
        vstep = 0.1 if 0.1 * pr.scale >= 20 else 0.2
        hname = names[axes[0]] + " [m]"
        name_w = font(11, True).getlength(hname)
        nb = (x1 - name_w - 6, x1 + 4)
        d.text((x1 - name_w, y1 + 7), hname, font=font(11, True), fill=TEXT)
        v = math.ceil(pr.lo[0] / hstep - 1e-9) * hstep
        while v <= pr.hi[0] + 1e-9:
            u = x0 + (v - pr.lo[0]) * pr.scale
            s = "0.0" if abs(v) < 1e-6 else f"{v:+.1f}"
            tw = f.getlength(s)
            if u - tw / 2 >= L + 6 and u + tw / 2 <= R - 6 and (u + tw / 2 < nb[0] or u - tw / 2 > nb[1]):
                d.line((u, y1, u, y1 + 3), fill=MUTED)
                d.text((u - tw / 2, y1 + 6), s, font=f, fill=MUTED)
            v += hstep
        # vertical axis name above the tick column (in XY the tick values grow downward: +x toward the camera)
        vname = names[axes[1]] + " [m]"
        d.text((L + 8, y0 - 2), vname, font=font(11, True), fill=TEXT)
        v = math.ceil(pr.lo[1] / vstep - 1e-9) * vstep
        while v <= pr.hi[1] + 1e-9:
            w = y0 + float(pr.vpix(v))
            s = "0.0" if abs(v) < 1e-6 else f"{v:+.1f}"
            if y0 + 18 <= w <= y1 - 4:
                d.line((x0 - 3, w, x0, w), fill=MUTED)
                d.text((x0 - 5 - f.getlength(s), w - 6), s, font=f, fill=MUTED)
            v += vstep

    def _plot_base_img(self, j):
        """Plot-area image with the 0.1 m grid and the table line (cached per projection)."""
        if j in self.plot_base:
            return self.plot_base[j]
        pr = self.proj[j]
        pw, ph = pr.size
        im = Image.new("RGB", (pw, ph), PANEL)
        d = ImageDraw.Draw(im)
        for axis in (0, 1):
            v = math.ceil(pr.lo[axis] / 0.1 - 1e-9) * 0.1
            while v <= pr.hi[axis] + 1e-9:
                if axis == 0:
                    u = (v - pr.lo[0]) * pr.scale
                    d.line((u, 0, u, ph), fill=GRID)
                else:
                    w = float(pr.vpix(v))
                    d.line((0, w, pw, w), fill=GRID)
                v += 0.1
        if pr.axes[1] == 2 and self.table_z is not None and pr.lo[1] <= self.table_z <= pr.hi[1]:
            w = float(pr.vpix(self.table_z))
            for u in range(0, pw, 8):
                d.line((u, w, min(u + 4, pw), w), fill=TABLE_LINE)
            f = font(10)
            tw = f.getlength("table top")
            d.text((pw - 6 - tw, w - 13), "table top", font=f, fill=TABLE_LINE)
            self.table_text[j] = (pw - 7 - tw, w - 13, pw - 5, w - 1)       # labels avoid it (soft)
        self.plot_base[j] = im
        return im

    # ------------------------------------------------------------ per-frame state
    def state(self, i):
        it = self.tl[i]
        k = int(it["frame"])
        t = float(it["t"])
        rec = self.tel[k] if 0 <= k < len(self.tel) else {}
        st = {"i": i, "k": k, "t": t, "rec": rec, "step": int(it.get("step") or rec.get("step") or 0),
              "hold": bool(it.get("hold")) and not it.get("final"), "final": bool(it.get("final")),
              "label": it.get("label")}
        evs = events_upto(self.run, t)
        st["evs"] = evs
        st["latest"] = evs[-1] if evs else None
        st["latest_s2"] = next((e for e in reversed(evs) if e.get("event") not in S1_EVENTS
                                and e.get("event") not in NOT_S2), None)
        st["group"] = self.same_time(evs, st["latest"], s2_only=False)       # events sharing the latest time
        st["group_s2"] = self.same_time(evs, st["latest_s2"], s2_only=True)
        st["held_now"] = rec.get("held")
        plan = current_plan(self.run, t)
        st["plan"] = plan
        lt = float(st["latest"].get("t", -1)) if st["latest"] else -1.0
        st["plan_active"] = plan is not None and float(plan.get("t", -1)) > lt and not st["final"]
        prog, status, cur, calls, diag = program_state(self.run, t)
        st.update(prog=prog, status=status, cur=cur, calls=calls, diag=diag)
        st["stage"], st["stage_t"] = None, None
        results, prog_src = {}, None
        for e in evs:
            ev = e.get("event")
            if ev in ("program", "repair") and isinstance(e.get("program"), dict) and e["program"].get("stages") is not None:
                results, prog_src = {}, e
            elif ev == "stage_start" and isinstance(e.get("stage"), dict):
                st["stage"], st["stage_t"] = e["stage"], float(e.get("t", 0.0))
            elif ev == "stage_done":
                results[e.get("stage")] = ("done", e.get("result"))
            elif ev == "stage_failed":
                results[e.get("stage")] = ("failed", e.get("reason"))
        st["results"], st["prog_src"] = results, prog_src
        if st["hold"]:
            t0 = self.tel_t[k] if 0 <= k < len(self.tel_t) else t
            t1 = self.tel_t[k + 1] if k + 1 < len(self.tel_t) else t
            st["hold_iv"] = (t0, t1)
        st["pending"] = next((c for c in self.s1 if c["t_start"] is not None and c["t_start"] <= t < c["t_end"]), None)
        st["ended"] = (not st["final"] and not st["hold"] and st["latest"] is not None
                       and st["latest"].get("event") == "end")           # evaluator's 10-step hold after the policy
        st["hold_label"], st["s1_returned"] = self._hold_label(st) if st["hold"] else (None, None)
        st["phase"], st["phase_also"] = self._phase(st)
        return st

    @staticmethod
    def same_time(evs, e, s2_only):
        """Events logged at the same harness time as `e` (in file order), e.g. actuated + stage_done + stage_start."""
        if e is None:
            return []
        te = float(e.get("t", 0.0))
        out = []
        for x in reversed(evs):
            if abs(float(x.get("t", 0.0)) - te) > 1e-6:
                if float(x.get("t", 0.0)) < te:
                    break
                continue
            if s2_only and (x.get("event") in S1_EVENTS or x.get("event") in NOT_S2):
                continue
            out.append(x)
        return out[::-1]

    def _hold_label(self, st):
        """Caption of a hold frame. The timeline labels a whole gap; inside an S1 gap, once the call has returned
        the rest of the gap is S2 work (e.g. grounding the program's objects), so say that instead.
        Returns (label, S1 call that returned in this gap or None)."""
        lab = st["label"] or "HOLD (simulator paused)"
        if not lab.startswith("S1 ") or st["pending"] is not None:
            return lab, None
        t0, t1 = st.get("hold_iv", (st["t"], st["t"]))
        call = next((c for c in reversed(self.s1) if t0 < c["t_end"] <= st["t"]), None)
        if call is None:
            return lab, None
        rest = [e for e in self.events if call["t_end"] < float(e.get("t", 0.0)) <= t1
                and e.get("event") not in S1_EVENTS and e.get("event") not in NOT_S2]
        sub = gap_label(rest) if rest else "S2 processing"
        parts = lab.split("  |  ")
        return f"S1 {call['kind'].upper()} returned → {sub}" + (f"  |  {parts[-1]}" if len(parts) > 1 else ""), call

    def _phase(self, st):
        """(filled phase, outlined 'also running' phase). Filled: the latest event's phase; a running planned move
        counts as that move; the lift after the grasp (payload events) is execution; during hold frames the phase
        named by the hold label (S1 waits -> phase 2, S2 planning waits -> phase 3); none after the 'end' event.
        Outlined: S2 work inside an S1 hold once the call has returned (e.g. grounding -> phase 1)."""
        if st["final"] or st["ended"]:
            return None, None
        latest = st["latest"]
        if st["hold"]:
            lab = st["label"] or ""
            if lab.startswith("S1 "):
                also = None
                if st["s1_returned"] is not None:
                    sub = (st["hold_label"] or "").split("→", 1)[-1].strip()
                    also = (2 if sub.startswith(PLAN_HOLDS) else
                            (0 if sub.startswith(PERCEIVE_HOLDS) else None))
                return 1, also
            if lab.startswith(PLAN_HOLDS):
                return 2, None
            if lab.startswith(PERCEIVE_HOLDS):
                return 0, None
            return (phase_of(latest) if latest else None), None
        if st["plan_active"]:
            return phase_of({"event": "move", "tag": st["plan"].get("tag", "")}), None
        if latest is not None and latest.get("event") in ("payload", "payload_centering"):
            return 3, None
        return (phase_of(latest) if latest else None), None

    # ------------------------------------------------------------ text for events
    def obj_text(self, ref):
        """'o3 control knob' (S1 id + short category) for an S1 id or an object name."""
        if ref is None:
            return "-"
        if ref in self.sid_cat:
            return f"{ref} {short_cat(self.sid_cat[ref])}"
        key = self.by_name.get(ref)
        if key is not None and key in self.obj_info:
            info = self.obj_info[key]
            return f"{info.get('sid') or ''} {short_cat(info.get('cat'))}".strip()
        return str(ref)

    def prog_obj(self, prog, sid):
        for o in (prog or {}).get("objects", []) or []:
            if o.get("id") == sid:
                cat = (o.get("prompts") or [None])[0] or o.get("name")
                return f"{sid} {short_cat(cat)}"
        return self.obj_text(sid)

    @staticmethod
    def move_tag(tag):
        tag = str(tag or "move")
        retry = tag.endswith("_retry")
        return (tag[:-6] if retry else tag), retry

    def move_phrase(self, tag):
        base, retry = self.move_tag(tag)
        return MOVE_PHRASE.get(base, base.replace("_", " ")) + (" (retry)" if retry else "")

    @staticmethod
    def measured_text(m):
        if not isinstance(m, dict):
            return "no measurement"
        if isnum(m.get("rotation_deg")):
            return f"rotated {m['rotation_deg']:.1f}° {m.get('sense') or ''}".strip()
        if isnum(m.get("travel_m")):
            return f"moved {m['travel_m'] * 1000:.0f} mm" + (f" (stop: {m['stop']})" if m.get("stop") else "")
        return ", ".join(f"{k} {v}" for k, v in m.items())

    def describe(self, e, follow=False, detail=False):
        """One-line description of a harness event (what S2 just did)."""
        if not isinstance(e, dict):
            return ""
        ev = str(e.get("event", ""))
        g = e.get
        try:
            if ev == "move":
                base, _ = self.move_tag(g("tag"))
                status = str(g("status") or "?")
                how = HOW_PHRASE.get(g("how"), f"{g('how')} route")
                s = f"{self.move_phrase(g('tag'))}: {'done' if status == 'done' else status.upper()} ({how}"
                s += f", {g('steps')} sim steps)" if detail and isnum(g("steps")) else ")"
                if follow and status == "done" and base in MOVE_FOLLOW:
                    s += f"  →  {MOVE_FOLLOW[base][0]}…"
                return s
            if ev == "scene":
                sh = g("shape") or []
                res = self.meta.get("map_res", 0.01)
                return (f"S2 voxel map initialised: {' x '.join(map(str, sh))} voxels at {res * 100:.0f} cm"
                        if len(sh) == 3 and isnum(res) else "S2 voxel map initialised")
            if ev == "survey":
                return f"survey done: {g('views')} wrist views of the workspace"
            if ev == "program":
                p = g("program") or {}
                return (f"S1 PROGRAM received: {len(p.get('stages') or [])} stage(s), "
                        f"{len(p.get('objects') or [])} object(s)")
            if ev == "repair":
                p = g("program") or {}
                return f"S1 REPAIR received: {len(p.get('stages') or [])} stage(s)"
            if ev == "verify":
                r = g("result") or {}
                return ("S1 VERIFY: task complete" if r.get("complete")
                        else f"S1 VERIFY: not complete - {r.get('problem') or 'no reason given'}")
            if ev == "grounded":
                s = g("summary") or {}
                txt = f"grounded {g('object')} {short_cat(s.get('category') or s.get('name'))} ({g('how')}"
                txt += f", {s['n_points']} points)" if isnum(s.get("n_points")) else ")"
                if detail:
                    try:
                        size = "[" + ", ".join(f"{float(a):.3f}" for a in s.get("size", [])) + "]"
                    except (TypeError, ValueError):
                        size = "[?]"
                    txt += f", centroid {vec(s.get('centroid', []))} m, size {size} m"
                return txt
            if ev == "ground_fail":
                return f"grounding FAILED for {g('object')}"
            if ev == "ground_alias":
                return f"{g('object')} is the tracked object {g('oid')} ({g('why')})"
            if ev == "support_plane":
                return (f"support plane fitted: table top at z {g('resting'):.3f} m" if isnum(g("resting"))
                        else "support plane fitted")
            if ev == "mount_frame":
                return (f"part mount frame fitted: protrusion {mm(g('protrusion'), 0)}, {g('inliers')} inliers"
                        + (f", axis point {vec(g('axis_point'))} m" if detail else ""))
            if ev == "part_grasp_cands":
                return f"part-grasp candidates: {g('n')}"
            if ev == "part_grasp_prefilter":
                rej = g("rejected") or {}
                return f"part-grasp prefilter: kept {g('kept')}" + (
                    " (rejected " + ", ".join(f"{k} {v}" for k, v in rej.items()) + ")" if rej else "")
            if ev == "part_grasp":
                return (f"part grasp planned: width {mm(g('width'))}, pitch {deg(g('pitch')):.0f}°"
                        + (f", p {vec(g('p'))} m" if detail else ""))
            if ev == "part_grasp_fail":
                return f"part grasp FAILED: {g('why')}"
            if ev == "actuated":
                r = g("result") or {}
                return f"actuated {self.obj_text(r.get('object'))}: {self.measured_text(r.get('measured'))}"
            if ev == "twist_stall":
                return f"twist stalled: {g('achieved'):.0f}° of {g('commanded'):.0f}°"
            if ev == "stage_start":
                s = g("stage") or {}
                return f"stage {s.get('id')} started: {s.get('op')} {self.obj_text(s.get('object'))}"
            if ev == "stage_done":
                return f"stage {g('stage')} done"
            if ev == "stage_failed":
                return f"stage {g('stage')} FAILED: {g('reason')}"
            if ev == "grasp_cands":
                return f"grasp candidates: {g('n')}"
            if ev == "grasp_prefilter":
                return f"grasp prefilter: kept {g('kept')} of {g('of')}"
            if ev == "grasp_plan":
                return (f"grasp planned: width {mm(g('width'))}, yaw {deg(g('yaw')):.0f}°, depth {mm(g('depth'), 0)}"
                        + (f", p {vec(g('p'))} m" if detail else ""))
            if ev == "descend":
                return f"grasp descent stopped early: {g('status')}"
            if ev == "blocked_descent":
                return f"grasp descent blocked {mm(g('short'), 0)} above the grasp"
            if ev == "close":
                return f"gripper closed: opening {mm(g('opening'))} (expected {mm(g('expect'))})"
            if ev == "payload_centering":
                return f"payload re-centred between the pads ({g('shift') * 1000:+.1f} mm)"
            if ev == "payload":
                return f"payload attached: {self.obj_text(g('object'))} ({g('points')} collision points), lifting"
            if ev == "held":
                return f"holding {self.obj_text(g('object'))} after the lift (opening {mm(g('opening'))})"
            if ev == "place_cands":
                return f"placement candidates: {g('n')} ({g('region_cells')} region cells)"
            if ev == "place_precheck":
                return f"placement precheck: {g('n')} candidates ranked"
            if ev == "place_plan":
                xy = g("xy") or [float("nan")] * 2
                return (f"placement planned at xy [{xy[0]:+.3f}, {xy[1]:+.3f}] m, yaw {deg(g('yaw')):.0f}°, "
                        f"drop {mm(g('drop'), 0)}")
            if ev in ("place_ik_fail", "place_ik"):
                return "placement IK " + ("failed" if ev.endswith("fail") else "solved")
            if ev == "place_descend":
                return f"placement descent {g('status')}" + ("  →  releasing" if follow else "")
            if ev == "placed":
                c = g("check") or {}
                s = f"placed {self.obj_text(g('object'))}: check {'satisfied' if c.get('satisfied') else 'NOT satisfied'}"
                if isnum(c.get("dist_to_target")):
                    s += f", {c['dist_to_target'] * 1000:.0f} mm from the target"
                return s
            if ev == "side_look":
                if isnum(g("gap_deg")) and isnum(g("gap_after_deg")):
                    return (f"side look at {self.obj_text(g('object'))}: coverage gap {g('gap_deg'):.0f}° → "
                            f"{g('gap_after_deg'):.0f}°")
                return f"side look at {self.obj_text(g('object'))}"
            if ev == "look_failed":
                return "look move failed (no reachable view)"
            if ev == "plan_fail":
                return f"no path for the {self.move_phrase(g('tag'))} ({g('how')})"
            if ev == "rrt":
                return f"RRT path found: {g('nodes')} nodes in {g('seconds')} s"
            if ev == "rrt_fail":
                return f"RRT failed after {g('seconds')} s"
            if ev == "escape":
                return "escape move " + ("done" if g("moved") else "failed")
            if ev == "dropped":
                return f"DROPPED {self.obj_text(g('object'))}"
            if ev == "relocated":
                return f"relocated {self.obj_text(g('object'))} after the drop"
            if ev == "park":
                return f"parking {self.obj_text(g('object'))} on the table"
            if ev == "put_down":
                return f"put {self.obj_text(g('object'))} down"
            if ev == "timeout":
                return "wall-time budget exhausted (timeout)"
            if ev == "crash":
                return f"harness error: {g('error')}"
            if ev == "end":
                return f"episode end: {g('status')}"
            if ev == "repair_nothing_left":
                return "S1 REPAIR: nothing left to do"
            if ev == "repair_error":
                return f"S1 REPAIR failed: {g('error')}"
            if ev == "repair_ground_fail":
                return "grounding failed after the REPAIR"
            if ev == "program_warnings":
                return f"program warnings: {len(g('warnings') or [])}"
            if ev == "gripper_geometry":
                return "gripper geometry calibrated"
        except Exception:
            pass
        return ev.replace("_", " ")

    def short_event(self, e):
        """Compact description of a secondary event in a same-time group."""
        ev = e.get("event")
        if ev == "stage_done":
            return f"stage {e.get('stage')} done"
        if ev == "stage_start":
            s = e.get("stage") or {}
            return f"stage {s.get('id')} started ({str(s.get('op') or '?')} {self.obj_text(s.get('object'))})"
        return self.describe(e)

    def describe_group(self, group, follow=False, detail=False):
        """Description of events logged at one time: the most informative leads (actuated / placed / plans before
        stage bookkeeping), the others follow in short form."""
        group = [e for e in group if isinstance(e, dict)]
        if not group:
            return ""
        keep = [e for e in group if EVENT_RANK.get(e.get("event"), 3) >= 0] or group[-1:]
        if len(keep) == 1:
            return self.describe(keep[0], follow=follow, detail=detail)
        lead = max(keep, key=lambda e: EVENT_RANK.get(e.get("event"), 3))      # first of the best rank
        rest = [self.short_event(e) for e in keep if e is not lead]
        return self.describe(lead, follow=follow, detail=detail) + "  →  " + ", ".join(rest)

    def held_text(self, key):
        info = self.obj_info.get(key) or {}
        return f"{info.get('sid') or ''} {short_cat(info.get('cat') or info.get('name') or key)}".strip()

    def last_held(self, k):
        """Object held most recently at or before recorded frame k (None if never)."""
        for kk in range(min(k, len(self.held_k) - 1), -1, -1):
            if self.held_k[kk] is not None:
                return self.held_k[kk]
        return None

    def released(self, st):
        """After a placement descent: has the telemetry seen the hand let go? (Only trusted when the telemetry
        recorded the object as held before, i.e. the recording carries 'held' at all.)"""
        return st["held_now"] is None and self.last_held(st["k"]) is not None

    def now_text(self, st):
        """Status line: what S2 is doing at this timeline item."""
        if st["final"]:
            status = (self.end_event or {}).get("status") or (self.summary or {}).get("status")
            return f"episode over: {status}" if status else "recording ended (no 'end' event in events.jsonl)"
        latest = st["latest"]
        if st["ended"]:
            return (f"policy finished (end: {latest.get('status') or '?'}); the evaluator holds the gripper command "
                    "for up to 10 more simulator steps (LIBERO's done ends the episode once the goal holds)")
        if st["hold"]:
            pend = st["pending"]
            if pend is not None:
                return (f"waiting for S1 {pend['kind'].upper()} (Astra): {st['t'] - pend['t_start']:.1f} s so far, "
                        f"simulator paused")
            t0, _ = st.get("hold_iv", (st["t"], st["t"]))
            if latest is not None and float(latest.get("t", -1)) > t0:
                return self.describe_group(st["group"])
            return (st["hold_label"] or "S2 processing").split("  |  ")[0] + "…"
        if st["plan_active"]:
            p = st["plan"]
            how = HOW_PHRASE.get(p.get("how"), f"{p.get('how')} route")
            return f"executing {self.move_phrase(p.get('tag'))} ({how}, {p.get('n_waypoints', '?')} waypoints)"
        if latest is None:
            return "starting"
        ev = latest.get("event")
        if ev == "close":
            return self.describe(latest) + "  →  lifting"
        if ev == "place_descend" and self.released(st):
            return f"released {self.held_text(self.last_held(st['k']))}, retreating"
        return self.describe_group(st["group"], follow=True)

    def activity(self, st):
        """Short upper-case activity for the telemetry header."""
        if st["hold"]:
            return st["hold_label"] or "HOLD (simulator paused)"
        if st["ended"]:
            return "POLICY END  |  evaluator 10-step hold"
        if st["plan_active"]:
            base, retry = self.move_tag(st["plan"].get("tag"))
            return "EXECUTING " + MOVE_SHORT.get(base, base.replace("_", " ").upper()) + (" (RETRY)" if retry else "")
        e = st["latest"]
        ev = e.get("event") if e is not None else None
        if ev == "move" and e.get("status") == "done":
            base, _ = self.move_tag(e.get("tag"))
            if base in MOVE_FOLLOW:
                return MOVE_FOLLOW[base][1]
        if ev == "close":
            return "GRIPPER CLOSED, LIFTING"
        if ev in ("payload", "payload_centering"):
            obj = st["held_now"] if st["held_now"] is not None else self.by_name.get(e.get("object"))
            return ("LIFTING " + self.held_text(obj).upper()) if obj is not None else "LIFTING THE OBJECT"
        if ev == "held":
            return "HOLDING " + self.obj_text(e.get("object")).upper()
        if ev == "place_descend":
            return "RELEASED, RETREATING" if self.released(st) else "RELEASING"
        if ev == "stage_start":
            return f"STAGE {str((e.get('stage') or {}).get('id') or '').upper()} STARTED".replace("  ", " ")
        if ev == "stage_done":
            return f"STAGE {str(e.get('stage') or '').upper()} DONE"
        if ev == "stage_failed":
            return f"STAGE {str(e.get('stage') or '').upper()} FAILED"
        if ev == "descend":                                 # logged only when the guarded descent stopped early
            return f"GRASP DESCENT STOPPED ({str(e.get('status') or '?').upper()})"
        fixed = {"actuated": "ACTUATION DONE", "placed": "PLACEMENT CHECKED", "survey": "SURVEY DONE",
                 "scene": "MAP INITIALISED", "blocked_descent": "GRASP DESCENT BLOCKED",
                 "twist_stall": "TWIST STALLED", "dropped": "OBJECT DROPPED", "escape": "ESCAPE MOVE",
                 "program": "S1 PROGRAM RECEIVED", "repair": "S1 REPAIR RECEIVED", "verify": "S1 VERIFY RECEIVED"}
        if ev in fixed:
            return fixed[ev]
        ph = st["phase"]
        return PHASES[ph] if ph is not None else "RUNNING"

    # ------------------------------------------------------------ rendering
    def layout_only(self, i):
        """Advance the sticky label placement through timeline item i without drawing (review mode)."""
        try:
            self._plot_layout(self.state(i))
        except Exception:
            pass

    def render(self, i):
        st = self.state(i)
        img = self.bg.copy()
        d = ImageDraw.Draw(img)
        for name, fn in (("cameras", self._cameras), ("program panel", self._top_panel), ("projections", self._plots),
                         ("phase bar", self._phase_bar), ("telemetry", self._bottom)):
            try:
                fn(img, d, st)
            except Exception as e:                       # one broken panel must not lose the video
                self._panel_error(d, name, e)
        if st["final"]:
            try:
                self._result_banner(img, d, st)
            except Exception as e:
                self._panel_error(d, "result banner", e)
        return img

    def _panel_error(self, d, name, e):
        if name not in self.errors:
            self.errors[name] = repr(e)
            print(f"[render_task_video] {name} failed: {e!r}\n{traceback.format_exc()}", file=sys.stderr)
        anchors = {"cameras": (40, 300), "program panel": (444, 300), "projections": (444, 560),
                   "phase bar": (444, 690), "telemetry": (444, 860), "result banner": (444, 760)}
        x, y = anchors.get(name, (444, 500))
        d.text((x, y), fit(f"({name} unavailable: {e!r})", font(12), 1100), font=font(12), fill=RED)

    def _cameras(self, img, d, st):
        for y, _, _, key in CAMS:
            rd = self.cams.get(key)
            if rd is None:
                rd = self.cams[key] = CamReader(self.dir / "rec" / f"cam_{key}.mp4")
            im, stale = rd.get(st["k"])
            if im is None:
                d.rectangle((44, y + 42, 44 + CAM_BOX_H, y + 42 + CAM_BOX_H), outline=EDGE)
                d.text((92, y + 200), "no camera stream recorded", font=font(14), fill=MUTED)
            else:
                img.paste(im, (36 + (CAM_BOX_W - im.width) // 2, y + 42 + (CAM_BOX_H - im.height) // 2))
            if st["final"]:
                badge, col = "FINAL FRAME", MUTED
            elif st["hold"]:
                badge, col = "SIM PAUSED", YELLOW
            elif stale:
                badge, col = "STREAM ENDED", RED
            else:
                badge = None
            if badge:
                f = font(11, True)
                w = f.getlength(badge) + 14
                d.rounded_rectangle((386 - w, y + 7, 386, y + 24), radius=5, fill=col)
                d.text((386 - w + 7, y + 9), badge, font=f, fill=BG)
            else:
                s = f"frame {st['k']}" + (f" / {self.n_frames - 1}" if self.n_frames else "")
                f = font(11)
                d.text((386 - f.getlength(s), y + 10), s, font=f, fill=MUTED)

    # ---- top panel: program / stages / S1 calls
    def _top_panel(self, img, d, st):
        title_end = 444 + font(19, True).getlength("S1 PROGRAM / S2 STAGES") + 24
        badge = None
        if st["final"]:
            badge, col = "EPISODE END", MUTED
        elif st["hold"]:
            badge, col = "HOLD  " + (st["hold_label"] or ""), YELLOW
        elif st["ended"]:
            badge, col = "POLICY DONE  |  EVALUATOR 10-STEP HOLD", MUTED
        if badge:
            f = font(14, True)
            room = 1558 - title_end - 16
            if f.getlength(badge) > room and "  |  " in badge:       # drop the gap length before truncating
                badge = badge.rsplit("  |  ", 1)[0]
            text = fit(badge, f, room)
            w = f.getlength(text) + 16
            d.rounded_rectangle((1558 - w, 148, 1558, 175), radius=7, fill=col)
            d.text((1558 - w + 8, 153), text, font=f, fill=BG)
        prog = st["prog"]
        stage = st["stage"] if st["cur"] is not None else None
        obj = self.prog_obj(prog, stage.get("object")) if stage else "-"
        cur_s = st["status"].get(st["cur"]) if st["cur"] is not None and st["status"] else None
        sid = (st["cur"] or "-") + (f" ({cur_s})" if cur_s in ("done", "failed") else "")
        line = f"step {st['step']}  |  stage {sid}  |  object {obj}"
        d.text((444, 184), fit(line, font(17), 1558 - 444), font=font(17), fill=TEXT)
        x = runs(d, 444, 212, [("Status: ", font(16), MUTED)])
        d.text((x, 212), fit(self.now_text(st), font(16), 1558 - x), font=font(16),
               fill=YELLOW if st["hold"] else CYAN)
        self._stage_list(d, st)
        self._s1_calls(d, st)

    def _stage_list(self, d, st):
        x0, x1, y = 444, COL_X - 14, TOP_LIST_Y
        prog, status, results = st["prog"], st["status"] or {}, st["results"]
        fh = font(12, True)
        if prog is None:
            d.text((x0, y), "PROGRAM", font=fh, fill=MUTED)
            if st["pending"] is not None:
                msg = ("S1 PROGRAM call in flight; S2 surveyed the scene meanwhile and now waits"
                       if self.survey_t is not None and st["t"] >= self.survey_t
                       else "S1 PROGRAM call in flight; S2 surveys the scene meanwhile")
            else:
                msg = "no S1 program yet"
            d.text((x0, y + 22), fit(msg, font(15), x1 - x0), font=font(15), fill=MUTED)
            return
        src = st["prog_src"] or {}
        rec = src.get("record") if isinstance(src.get("record"), dict) else {}
        n = rec.get("n")
        stages = prog.get("stages") or []
        head = (f"PROGRAM  from S1 {str(src.get('event', 'program')).upper()}" + (f" #{n}" if isnum(n) else "")
                + (f" at {float(src.get('t', 0)):.1f} s" if src else "") + f"  |  {len(stages)} stage(s)")
        d.text((x0, y), fit(head, fh, x1 - x0), font=fh, fill=MUTED)
        y += 21
        rows = max(1, (TOP_LIST_B - 6 - y) // 24)
        show = stages
        if len(stages) > rows:                      # keep the running stage in view
            cur_i = next((j for j, s in enumerate(stages) if s.get("id") == st["cur"]), 0)
            a = max(0, min(cur_i - 1, len(stages) - (rows - 1)))
            show = stages[a:a + rows - 1]
        f, fb, fs = font(15), font(15, True), font(13)
        for s in show:
            sid = s.get("id")
            stt = status.get(sid, "pending")
            col = {"done": GREEN, "running": CYAN, "failed": RED}.get(stt, MUTED)
            self._marker(d, x0 + 7, y + 10, stt, col)
            res = results.get(sid)
            right = stt.upper()
            if res is not None and res[0] == "done":
                extra = self._result_text(res[1])
                right = "DONE" + (f"  {extra}" if extra else "")
            elif res is not None and res[0] == "failed":
                right = "FAILED  " + str(res[1] or "")
            right = fit(right, fs, 210)
            rw = fs.getlength(right)
            d.text((x1 - rw, y + 2), right, font=fs, fill=col)
            xi = runs(d, x0 + 20, y, [(str(sid), fb, col)]) + 10
            room = x1 - rw - 16 - xi
            desc = self._stage_desc(s, prog)
            if f.getlength(desc) > room:                # drop the tolerance note before truncating
                desc = self._stage_desc(s, prog, tolerance=False)
            d.text((xi, y), fit(desc, f, room), font=f, fill=col)
            y += 24
        if len(stages) > len(show):
            d.text((x0 + 20, y), f"+ {len(stages) - len(show)} more stage(s)", font=font(12), fill=MUTED)
            y += 16
        fin = prog.get("finish_condition")
        if fin and y + 15 <= TOP_LIST_B:
            for ln in wrap("finish when: " + str(fin), font(13), x1 - x0 - 20, max(1, (TOP_LIST_B - y) // 17)):
                d.text((x0 + 20, y + 2), ln, font=font(13), fill=MUTED)
                y += 17

    @staticmethod
    def _marker(d, cx, cy, stt, col):
        if stt == "done":
            d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=col)
            d.line([(cx - 4, cy), (cx - 1, cy + 3), (cx + 4, cy - 3)], fill=BG, width=2)
        elif stt == "running":
            d.polygon([(cx - 5, cy - 7), (cx + 7, cy), (cx - 5, cy + 7)], fill=col)
        elif stt == "failed":
            d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=col)
            d.line((cx - 3, cy - 3, cx + 3, cy + 3), fill=BG, width=2)
            d.line((cx - 3, cy + 3, cx + 3, cy - 3), fill=BG, width=2)
        else:
            d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=col, width=2)

    def _stage_desc(self, s, prog, tolerance=True):
        op = str(s.get("op") or "?")
        obj = self.prog_obj(prog, s.get("object"))
        if op == "transfer":
            rel = str(s.get("relation") or "?").replace("_", " ")
            txt = f"TRANSFER  {obj}  {rel.upper()}  {self.prog_obj(prog, s.get('reference'))}"
            if tolerance and s.get("tolerance") and s.get("tolerance") != "normal":
                txt += f"  ({s['tolerance']})"
            return txt
        if op == "actuate":
            txt = "ACTUATE  " + " ".join(p for p in (s.get("motion"), obj) if p)
            if s.get("goal"):
                txt += f"  → {s['goal']}"
            if s.get("turn_direction"):
                txt += f", {s['turn_direction']}"
            return txt
        return f"{op.upper()}  {obj}"

    @staticmethod
    def _result_text(res):
        if not isinstance(res, dict):
            return ""
        m = res.get("measured")
        if isinstance(m, dict):
            if isnum(m.get("rotation_deg")):
                sense = {"counterclockwise": "ccw", "clockwise": "cw"}.get(m.get("sense"), m.get("sense") or "")
                return f"{m['rotation_deg']:.1f}° {sense}".strip()
            if isnum(m.get("travel_m")):
                return f"{m['travel_m'] * 1000:.0f} mm"
        ck = res.get("relation_check")
        if isinstance(ck, dict):
            if isnum(ck.get("dist_to_target")):
                return f"{ck['dist_to_target'] * 1000:.0f} mm off target"
            return "check ok" if ck.get("satisfied") else "check failed"
        return ""

    def _s1_calls(self, d, st):
        x0, x1, y = COL_X + 14, 1558, TOP_LIST_Y
        effort = self.effort
        d.text((x0, y), fit(f"S1 CALLS  (Astra, effort {effort})", font(12, True), x1 - x0), font=font(12, True),
               fill=MUTED)
        y += 21
        t = st["t"]
        pend = st["pending"]
        entries = [c for c in self.s1 if c["t_end"] <= t] + ([pend] if pend is not None else [])
        diag = st["diag"]
        f, fb = font(14), font(14, True)
        max_rows = 2 if diag else max(1, (TOP_LIST_B - y) // 20)
        if not entries:
            d.text((x0, y), "none yet", font=f, fill=MUTED)
            y += 20
        if len(entries) > max_rows:
            hidden = len(entries) - (max_rows - 1)
            d.text((x0, y + 1), f"+ {hidden} earlier call(s)", font=font(12), fill=MUTED)
            y += 18
            entries = entries[hidden:]
        for c in entries:
            parts = [(f"#{c['n']} ", f, MUTED), (c["kind"].upper(), fb, YELLOW if c is pend else TEXT)]
            if c is pend:
                parts += [(f"   in flight, {t - c['t_start']:.1f} s so far", f, YELLOW)]
            else:
                parts += [(f"   {c['lat']:.2f} s" if c["lat"] is not None else "   ? s", f, TEXT),
                          ("   ok" if c["ok"] else "   FAILED", f, GREEN if c["ok"] else RED),
                          (f"   at {c['t_end']:.1f} s", font(12), MUTED)]
                if c["kind"] == "verify":
                    r = c["event"].get("result") or {}
                    parts += [("   complete" if r.get("complete") else "   NOT complete", font(12, True),
                               GREEN if r.get("complete") else RED)]
            runs(d, x0, y, parts)
            y += 20
        if diag:
            y += 3
            d.text((x0, y), "LAST REPAIR DIAGNOSIS", font=font(12, True), fill=MUTED)
            y += 16
            for ln in wrap(diag, font(12), x1 - x0, max(1, (TOP_LIST_B + 4 - y) // 15)):
                d.text((x0, y), ln, font=font(12), fill=TEXT)
                y += 15

    # ---- projections
    def _voxels(self, fname):
        """Occupied voxels above the table of one map snapshot, kept sparse: (ijk (N,3) int, label (N,) int,
        lo, res, shape). Cached per snapshot file."""
        key = ("vox", fname)
        if key not in self.map_cache:
            vox = None
            try:
                with np.load(self.dir / "rec" / "maps" / fname) as z:
                    occ = np.asarray(z["state"]) == 1
                    lab = np.asarray(z["label"]) if "label" in z.files else None
                    mlo = np.asarray(z["lo"], float).reshape(3)
                    res = float(z["res"])
                    tz = float(z["table_z"]) if "table_z" in z.files else float("nan")
                tz = tz if math.isfinite(tz) else self.table_z
                if tz is not None:
                    occ[:, :, :max(0, int(math.floor((tz + 0.012 - mlo[2]) / res)) + 1)] = False
                ijk = np.argwhere(occ)                              # C order, same as lab[occ]
                labels = (lab[occ].astype(np.int64) if lab is not None and lab.shape == occ.shape
                          else np.zeros(len(ijk), np.int64))
                vox = (ijk, labels, mlo, res, occ.shape)
            except Exception:
                vox = None
            self.map_cache[key] = vox
        return self.map_cache[key]

    def _map_exclude(self, mrec, t, k):
        """Object ids whose voxels the shown snapshot must not draw: the held object and its parts (S2 clears them
        from its map when the payload is attached) and objects released / relocated after the snapshot was taken
        (they are no longer where the snapshot has them)."""
        ex = set()
        held = self.held_k[k] if 0 <= k < len(self.held_k) else None
        if held is not None:
            ex.add(held)
        t_snap = float(mrec.get("t", 0.0) or 0.0)
        for key, ups in self.box_updates.items():
            if any(kind in ("released", "relocated") and t_snap < tu <= t for tu, _, _, kind in ups):
                ex.add(key)
        objs = mrec.get("objects") if isinstance(mrec.get("objects"), dict) else {}
        for oid, info in objs.items():                       # parts of an excluded object move with it
            if isinstance(info, dict) and info.get("part_of") is not None and any(
                    str(info.get("part_of")) in (str(x), str((self.obj_info.get(x) or {}).get("name"))) for x in ex):
                ex.add(oid)
        out = set()
        for x in ex:
            try:
                out.add(int(x))
            except (TypeError, ValueError):
                pass
        return frozenset(out)

    def _map_layer(self, rec, j, exclude=frozenset()):
        """Cached (RGB, mask) of the S2 map's occupied voxels above the table (minus `exclude` object ids),
        projected for panel j."""
        if rec is None or not rec.get("file"):
            return None
        fname = rec["file"]
        key = (fname, j, exclude)
        if key in self.map_cache:
            return self.map_cache[key]
        layer = None
        try:
            vox = self._voxels(fname)
            if vox is not None:
                ijk, labels, mlo, res, shape = vox
                if exclude:
                    keep = ~np.isin(labels, np.fromiter(exclude, np.int64))
                    ijk = ijk[keep]
                pr = self.proj[j]
                a, b = pr.axes
                na, nb = int(shape[a]), int(shape[b])
                cnt = np.bincount(ijk[:, a] * nb + ijk[:, b], minlength=na * nb).reshape(na, nb)
                pw, ph = pr.size
                ua = pr.lo[0] + (np.arange(pw) + 0.5) / pr.scale
                vb = pr.vworld(np.arange(ph) + 0.5)
                ia = np.floor((ua - mlo[a]) / res).astype(int)
                ib = np.floor((vb - mlo[b]) / res).astype(int)
                ok = ((ib >= 0) & (ib < nb))[:, None] & ((ia >= 0) & (ia < na))[None, :]
                val = cnt[np.clip(ia, 0, na - 1)][:, np.clip(ib, 0, nb - 1)].T * ok
                w = (0.4 + 0.6 * np.clip(val / 6.0, 0, 1) ** 0.6)[..., None]
                col = np.array(PANEL, float) * (1 - w) + np.array(MAP_COL, float) * w
                layer = (Image.fromarray(col.astype(np.uint8)), Image.fromarray(((val > 0) * 255).astype(np.uint8)))
        except Exception:
            layer = None
        self.map_cache[key] = layer
        return layer

    def _targets(self, st, objs):
        """Grasp target (grasp_plan / part_grasp p) and place target (place_plan xy at the destination's top) of the
        current stage."""
        grasp = place = None
        if st["stage_t"] is None:
            return None, None
        for e in st["evs"]:
            if float(e.get("t", 0)) < st["stage_t"]:
                continue
            if e.get("event") in ("grasp_plan", "part_grasp") and len(pts3(e.get("p"))):
                grasp = pts3(e.get("p"))[0]
            elif e.get("event") == "place_plan" and isinstance(e.get("xy"), (list, tuple)) and len(e["xy"]) == 2 \
                    and all(map(isnum, e["xy"])):
                place = (float(e["xy"][0]), float(e["xy"][1]))
        if place is not None:
            zt = None
            ref = (st["stage"] or {}).get("reference")
            for o in objs:
                if o["sid"] == ref and not o["held"]:
                    zt = float(o["hi"][2])
            if zt is None:
                zt = self.table_z if self.table_z is not None else 0.9
            place = np.array([place[0], place[1], zt])
        return grasp, place

    def _plot_layout(self, st):
        """What the three projections show at this timeline item, with the box labels placed. Label positions are
        sticky across consecutive frames (self.label_mem), so this must run for every timeline item in order (the
        video does; --frames runs it for the skipped items too). Cached on st."""
        if "layout" in st:
            return st["layout"]
        t, k = st["t"], st["k"]
        objs = self.objects(t, k)
        mrec = current_map(self.run, t)
        exclude = self._map_exclude(mrec, t, k) if mrec is not None else frozenset()
        plan = st["plan"]
        plan_pts = pts3(plan.get("eef")) if plan else np.empty((0, 3))
        meas = self.eef[:k + 1]
        meas = meas[np.isfinite(meas).all(1)] if len(meas) else meas
        k_stage = bisect.bisect_left(self.tel_t, st["stage_t"]) if st["stage_t"] is not None else 0
        k_stage = min(k_stage, max(0, len(meas) - 1))
        grasp, place = self._targets(st, objs)
        cur = self.eef[k] if 0 <= k < len(self.eef) else None
        panels = []
        for j, pr in enumerate(self.proj):
            pw, ph = pr.size
            boxes = []
            for o in objs:
                (ua, va), (ub, vb) = pr.lpt(o["lo"]), pr.lpt(o["hi"])
                bx = (min(ua, ub), min(va, vb), max(ua, ub), max(va, vb))
                if bx[2] < 0 or bx[0] > pw or bx[3] < 0 or bx[1] > ph:
                    continue
                boxes.append((o, bx))
            plan_px = pr.local(plan_pts) if len(plan_pts) > 1 else None
            meas_px = pr.local(meas) if len(meas) > 1 else None
            gpx = pr.lpt(grasp) if grasp is not None else None
            ppx = pr.lpt(place) if place is not None else None
            eef = self._eef_geom(pr, cur) if cur is not None and np.isfinite(cur).all() else None
            markers = [((m[0] - 8, m[1] - 8, m[0] + 8, m[1] + 8), COST_MARKER) for m in (gpx, ppx) if m is not None]
            if eef is not None:
                markers.append((eef["rect"], COST_MARKER))
            if plan_px is not None:
                u, v = plan_px[-1]
                markers.append(((u - 5, v - 5, u + 5, v + 5), COST_PLAN_END))
            if j in self.table_text:
                markers.append((self.table_text[j], COST_SOFT))
            integ = self._path_integral((pw, ph), [
                (plan_px, 3), (meas_px[:k_stage + 1] if meas_px is not None and k_stage >= 1 else None, 1),
                (meas_px[k_stage:] if meas_px is not None else None, 3)])
            labels = self._place_labels(j, (pw, ph), boxes, markers, integ)
            panels.append({"boxes": boxes, "plan_px": plan_px, "meas_px": meas_px, "k_stage": k_stage,
                           "grasp": gpx, "place": ppx, "eef": eef, "labels": labels})
        st["layout"] = {"panels": panels, "mrec": mrec, "exclude": exclude}
        return st["layout"]

    def _plots(self, img, d, st):
        lay = self._plot_layout(st)
        mrec, exclude = lay["mrec"], lay["exclude"]
        f = font(LABEL_FONT)
        for j, (pr, P) in enumerate(zip(self.proj, lay["panels"])):
            x0, y0, x1, y1 = pr.rect
            sub = self._plot_base_img(j).copy()
            sd = ImageDraw.Draw(sub)
            layer = self._map_layer(mrec, j, exclude)
            if layer is not None:
                sub.paste(layer[0], (0, 0), layer[1])
            for o, bx in P["boxes"]:
                sd.rectangle(bx, outline=ORANGE_DIM if o.get("dropped") else ORANGE, width=2 if o["held"] else 1)
            if P["plan_px"] is not None:
                sd.line([tuple(p) for p in P["plan_px"]], fill=PLAN_YELLOW, width=2)
                u, v = P["plan_px"][-1]
                sd.rectangle((u - 3, v - 3, u + 3, v + 3), outline=PLAN_YELLOW)
            M, ks = P["meas_px"], P["k_stage"]
            if M is not None:
                if ks >= 1:
                    sd.line([tuple(p) for p in M[:ks + 1]], fill=PINK_DIM, width=2)
                if len(M) - ks > 1:
                    sd.line([tuple(p) for p in M[ks:]], fill=PINK, width=2)
            # labels first, then the markers on top: a label can never hide a target or the current EEF
            for r, text in P["labels"]:
                sd.rectangle(r, fill=PANEL)
                sd.text((r[0] + 2, r[1] - 1), text, font=f, fill=ORANGE)
            if P["grasp"] is not None:
                u, v = P["grasp"]
                sd.ellipse((u - 6, v - 6, u + 6, v + 6), outline=GREEN, width=2)
                sd.line((u - 3, v, u + 3, v), fill=GREEN)
                sd.line((u, v - 3, u, v + 3), fill=GREEN)
            if P["place"] is not None:
                u, v = P["place"]
                for r in (7, 6):
                    sd.polygon([(u, v - r), (u + r, v), (u, v + r), (u - r, v)], outline=CYAN)
                sd.ellipse((u - 1.5, v - 1.5, u + 1.5, v + 1.5), fill=CYAN)
            if P["eef"] is not None:
                self._draw_eef(sd, P["eef"])
            img.paste(sub, (x0, y0))
            d.rectangle((x0, y0, x1, y1), outline=EDGE)
        if mrec is not None:                         # which S2 map snapshot is shown (XZ title row)
            L, T, R, B = PROJ[1][0]
            idx = str(mrec.get("file", "")).replace("map_", "").replace(".npz", "").lstrip("0") or "0"
            s = f"S2 voxel map #{idx} (from {float(mrec.get('t', 0)):.1f} s)"
            fc = font(11)
            d.text((R - 12 - fc.getlength(s), T + 11), s, font=fc, fill=MUTED)

    @staticmethod
    def _eef_geom(pr, cur):
        """Current-EEF marker geometry: a dot, or (out of view) an arrow at the border with the out-of-range
        coordinate. 'rect' is what box labels must keep clear of."""
        pw, ph = pr.size
        u, v = pr.lpt(cur)
        if 0 <= u < pw and 0 <= v < ph:                     # the dot is drawn with radius 5 (outline included)
            return {"inside": True, "u": u, "v": v, "rect": (u - 6, v - 6, u + 6, v + 6)}
        cu, cv = min(max(u, 9), pw - 10), min(max(v, 9), ph - 10)
        names = "xyz"
        if v < 0 or v >= ph:
            ax = pr.axes[1]
            tri = ([(cu, cv - 8), (cu - 6, cv + 2), (cu + 6, cv + 2)] if v < 0 else
                   [(cu, cv + 8), (cu - 6, cv - 2), (cu + 6, cv - 2)])
        else:
            ax = pr.axes[0]
            tri = ([(cu - 8, cv), (cu + 2, cv - 6), (cu + 2, cv + 6)] if u < 0 else
                   [(cu + 8, cv), (cu - 2, cv - 6), (cu - 2, cv + 6)])
        txt = f"EEF {names[ax]} {cur[ax]:+.2f}"
        tw = font(10).getlength(txt)
        tx = min(max(cu + 10, 2), pw - tw - 2)
        ty = min(max(cv - 6, 2), ph - 13)
        rect = (min(cu - 8, tx - 2), min(cv - 8, ty), max(cu + 8, tx + tw + 2), max(cv + 8, ty + 12))
        return {"inside": False, "tri": tri, "txt": txt, "tx": tx, "ty": ty, "tw": tw, "rect": rect}

    @staticmethod
    def _draw_eef(sd, g):
        if g["inside"]:
            u, v = g["u"], g["v"]
            sd.ellipse((u - 5, v - 5, u + 5, v + 5), fill=TEXT, outline=BG, width=2)
            return
        sd.polygon(g["tri"], fill=TEXT)
        sd.rectangle((g["tx"] - 2, g["ty"], g["tx"] + g["tw"] + 2, g["ty"] + 12), fill=PANEL)
        sd.text((g["tx"], g["ty"] - 1), g["txt"], font=font(10), fill=TEXT)

    @staticmethod
    def _path_integral(size, weighted):
        """Integral image of the drawn EEF paths [(pixel polyline, weight)], so a label's path coverage is O(1)."""
        pw, ph = size
        weighted = [(P, w) for P, w in weighted if P is not None and len(P) > 1]
        if not weighted:
            return None
        m = Image.new("L", (pw, ph), 0)
        md = ImageDraw.Draw(m)
        for P, w in sorted(weighted, key=lambda pw_: pw_[1]):        # heavier paths drawn last (win overlaps)
            md.line([tuple(p) for p in P], fill=int(w), width=2)
        S = np.zeros((ph + 1, pw + 1), np.int32)
        S[1:, 1:] = np.asarray(m, np.int32).cumsum(0).cumsum(1)
        return S

    @staticmethod
    def _rect_sum(S, r):
        ph, pw = S.shape[0] - 1, S.shape[1] - 1
        x0, x1 = (int(min(max(v, 0), pw)) for v in (math.floor(r[0]), math.ceil(r[2])))
        y0, y1 = (int(min(max(v, 0), ph)) for v in (math.floor(r[1]), math.ceil(r[3])))
        return int(S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0])

    @staticmethod
    def _label_cands(box, tw, th, pw, ph):
        """Candidate label rectangles (background included) around a box: above, below, beside, inside last."""
        bx0, by0, bx1, by1 = box
        raw = [(bx0, by0 - th - 1), (bx1 - tw, by0 - th - 1), (bx0, by1 + 2), (bx1 - tw, by1 + 2),
               (bx1 + 3, by0), (bx0 - tw - 3, by0), (bx1 + 3, by1 - th), (bx0 - tw - 3, by1 - th),
               (bx0 + 2, by0 + 2)]
        out = []
        for lx, ly in raw:
            lx = min(max(lx, 1), pw - tw - 1)
            ly = min(max(ly, 1), ph - th - 1)
            out.append((lx, ly, lx + tw, ly + th))
        return out

    def _place_labels(self, j, size, boxes, markers, integ):
        """Place box labels: never over another label or a marker (target, EEF) if any candidate avoids it, then
        avoid crossing box edges and covering the paths. A label keeps last frame's position while that is still
        about as good as the best (no jumping between equivalent spots)."""
        f = font(LABEL_FONT)
        pw, ph = size
        th = 13
        items = []
        for o, bx in sorted(boxes, key=lambda ob: (bool(ob[0]["held"]), str(ob[0]["key"]))):   # static first
            text = o["label"]
            items.append((o, bx, text, self._label_cands(bx, f.getlength(text) + 4, th, pw, ph)))

        def costs_of(bx, cands, placed):
            out = []
            for ci, r in enumerate(cands):
                c = COST_LABEL * sum(overlaps(r, q) for q in placed)
                c += sum(w for q, w in markers if overlaps(r, q))
                for _, b in boxes:
                    if not overlaps(r, b, 0):
                        continue
                    inside = r[0] > b[0] + 1 and r[2] < b[2] - 1 and r[1] > b[1] + 1 and r[3] < b[3] - 1
                    if b is bx:
                        c += 0.0 if inside else COST_EDGE
                    else:
                        c += COST_INSIDE if inside else COST_EDGE
                if integ is not None:
                    c += min(PATH_CAP, PATH_COST * self._rect_sum(integ, r))
                c += 0.01 * ci + (0.5 if ci == len(cands) - 1 else 0.0)      # prefer outside, in order
                out.append(c)
            return out

        chosen, placed = {}, []
        # pass 1: labels keep last frame's spot while it is clear of markers / kept labels and about as good as
        # the best spot, so one displaced label does not shuffle the others
        for a, (o, bx, _, cands) in enumerate(items):
            prev = self.label_mem.get((j, o["key"]))
            if prev is None or prev >= len(cands):
                continue
            cs = costs_of(bx, cands, placed)
            slack = STICKY_SLACK_HELD if o["held"] else STICKY_SLACK
            if cs[prev] < COST_MARKER and cs[prev] <= min(cs) + slack:
                chosen[a] = prev
                placed.append(cands[prev])
        # pass 2: new labels and displaced ones take the best free spot around the kept labels
        for a, (o, bx, _, cands) in enumerate(items):
            if a not in chosen:
                cs = costs_of(bx, cands, placed)
                chosen[a] = int(np.argmin(cs))
                placed.append(cands[chosen[a]])
        out = []
        for a, (o, _, text, cands) in enumerate(items):
            self.label_mem[(j, o["key"])] = chosen[a]
            out.append((cands[chosen[a]], text))
        return out

    # ---- phase bar
    def _phase_bar(self, img, d, st):
        if st["final"]:
            return
        active, also = st["phase"], st["phase_also"]
        s1_flight = st["pending"] is not None and active != 1
        f = font(13, True)
        for idx, label in enumerate(PHASE_LABELS):
            x = 425 + idx * 291
            on = idx == active
            if on:
                outline, fg = GREEN, TEXT
            elif idx == 1 and s1_flight:
                outline, fg = YELLOW, YELLOW             # S1 call in flight while S2 works
            elif idx == also:
                outline, fg = GREEN, GREEN               # S2 work inside an S1 hold after the call returned
            else:
                outline, fg = EDGE, MUTED
            d.rounded_rectangle((x, PHASE_Y[0], x + 277, PHASE_Y[1]), radius=8, fill=ACTIVE_FILL if on else PANEL,
                                outline=outline, width=2)
            d.text((x + 10, PHASE_Y[0] + 14), fit(label, f, 277 - 18), font=f, fill=fg)

    # ---- telemetry panel
    def _bottom(self, img, d, st):
        if st["final"]:
            return
        rec = st["rec"]
        R = BOTTOM[2]
        head = f"STEP {st['step']:04d}  |  {self.activity(st)}"
        fh = font(20, True)
        if st["hold"] and fh.getlength(head) > R - 17 - 442 and head.count("  |  ") > 1:
            head = head.rsplit("  |  ", 1)[0]                     # drop the gap length before truncating
        d.text((442, 741), fit(head, fh, R - 17 - 442), font=fh,
               fill=YELLOW if st["hold"] else (TEXT if st["ended"] else CYAN))
        f, fm = font(16), font(15)
        eef = rec.get("eef")
        d.text((442, 777), "EEF world m: " + (vec(eef, 4) if isinstance(eef, (list, tuple)) else "n/a"), font=f,
               fill=TEXT)
        fg = rec.get("finger")
        d.text((1025, 777), "Finger gap: " + (f"{fg * 1000:.2f} mm" if isnum(fg) else "n/a"), font=f, fill=TEXT)
        try:
            ftxt = f"Wrist force |F|: {float(np.linalg.norm(np.asarray(rec.get('force'), float))):.1f} N"
        except (TypeError, ValueError):
            ftxt = "Wrist force |F|: n/a"
        d.text((442, 805), ftxt, font=f, fill=TEXT)
        gc = rec.get("grip")
        if isnum(gc):
            gname = "open" if gc < -0.5 else ("close" if gc > 0.5 else "hold the finger target")
            gtxt = f"Gripper command: {'+1' if gc > 0.5 else ('-1' if gc < -0.5 else '0')} ({gname})"
        else:
            gtxt = "Gripper command: n/a"
        d.text((1025, 805), gtxt, font=f, fill=YELLOW)
        held = rec.get("held")
        if held is not None:
            info = self.obj_info.get(held, {})
            htxt = f"{info.get('sid') or ''} {info.get('cat') or info.get('name') or held}".strip()
            runs(d, 442, 833, [("Held object: ", f, TEXT), (fit(htxt, font(16, True), 500), font(16, True), GREEN)])
        else:
            runs(d, 442, 833, [("Held object: ", f, TEXT), ("none", f, MUTED)])
        d.text((1025, 833), f"Sim time {st['step'] * SIM_DT:.2f} s  |  wall {st['t']:.1f} s", font=f, fill=TEXT)
        e = st["latest_s2"]
        if e is not None:
            x = runs(d, 442, 863, [(f"Last S2 event @ {float(e.get('t', 0)):.1f} s:  ", fm, MUTED)])
            d.text((x, 863), fit(self.describe_group(st["group_s2"] or [e], detail=True), fm, R - 17 - x), font=fm,
                   fill=TEXT)
        else:
            d.text((442, 863), "No S2 event yet", font=fm, fill=MUTED)
        if st["hold"]:
            note = (f"Hold frame: simulated time stands still at step {st['step']} (image and robot state frozen) "
                    "while the harness works; the status above advances.")
        elif st["ended"]:
            note = ("The policy has finished; these frames are the evaluator's 10-step hold, ended early by "
                    "LIBERO's done if the goal holds (result follows).")
        else:
            note = ("Sim time = step x 0.05 s (one recorded frame every 2 steps); wall = seconds since the harness "
                    "started (events.jsonl clock).")
        d.text((442, 889), fit(note, font(13), R - 17 - 442), font=font(13), fill=MUTED)

    # ---- result banner (final tail frames)
    def _result_banner(self, img, d, st):
        L, T, R, B = 425, PHASE_Y[0], 1576, BOTTOM[3]
        s = self.summary
        if s is None:
            d.rounded_rectangle((L, T, R, B), radius=12, fill=PANEL, outline=MUTED, width=3)
            end = (self.end_event or {}).get("status") or "unknown"
            d.text((446, T + 16), f"EPISODE END: {str(end).upper()}", font=font(26, True), fill=YELLOW)
            d.text((446, T + 64), "No evaluator result: summary.json is missing for this run.", font=font(17),
                   fill=MUTED)
            calls = ", ".join(f"{c['kind'].upper()} {c['lat']:.2f} s" for c in self.s1 if c["lat"] is not None)
            d.text((446, T + 98), fit(f"S1 calls: {len(self.s1)}" + (f"  ({calls})" if calls else ""), font(17),
                                      R - 20 - 446), font=font(17), fill=TEXT)
            return
        ok, detail = libero_success(s)
        scene, added = s.get("scene_collision_free"), s.get("added_obstacle_collision_free")
        all_ok = ok and scene is True and added is True
        col = GREEN if all_ok else RED
        d.rounded_rectangle((L, T, R, B), radius=12, fill=OK_FILL if all_ok else BAD_FILL, outline=col, width=3)
        title = ("RESULT: SUCCESS, COLLISION-FREE" if all_ok else
                 ("RESULT: SUCCESS WITH COLLISIONS" if ok else "RESULT: TASK NOT ACHIEVED"))
        d.text((446, T + 14), title, font=font(26, True), fill=col)
        info = f"status {s.get('status', '?')}  |  repairs {s.get('repairs', 0)}  |  Astra {self.effort}"
        d.text((R - 20 - font(15).getlength(info), T + 22), info, font=font(15), fill=MUTED)

        def item(x, y, ok, label, extra=""):
            c = GREEN if ok is True else (RED if ok is False else MUTED)
            d.ellipse((x, y + 1, x + 20, y + 21), fill=c)
            mark = "✓" if ok is True else ("✗" if ok is False else "?")
            fb = font(14, True)
            d.text((x + 10 - fb.getlength(mark) / 2, y + 2), mark, font=fb, fill=BG)
            xx = runs(d, x + 30, y, [(label, font(17, True), TEXT)])
            if extra:
                d.text((xx + 10, y + 2), extra, font=font(15), fill=MUTED)

        y = T + 60
        item(446, y, ok, "Task success", detail)
        item(446, y + 34, scene, "Scene collision-free")
        item(446, y + 68, added, "Added-obstacle collision-free")
        x2, f17 = 1040, font(17)
        wall = s.get("wall_s_policy")
        d.text((x2, y + 1), f"Policy wall time: {wall:.1f} s" if isnum(wall) else "Policy wall time: n/a",
               font=f17, fill=TEXT)
        sim = s.get("sim_time_s")
        d.text((x2, y + 35), (f"Simulated time: {sim:.1f} s ({s.get('sim_steps', '?')} steps)" if isnum(sim)
                              else "Simulated time: n/a"), font=f17, fill=TEXT)
        model = s.get("model") if isinstance(s.get("model"), dict) else {}
        kinds, lats = model.get("kinds") or [], model.get("latency_s") or []
        calls = ", ".join(f"{str(kk).upper()} {ll:.1f} s" for kk, ll in zip(kinds, lats) if isnum(ll))
        d.text((x2, y + 69), fit(f"S1 calls: {model.get('calls', len(self.s1))}" + (f"  ({calls})" if calls else ""),
                                 f17, R - 20 - x2), font=f17, fill=TEXT)
        yb = T + 168
        pairs = [p for p in (s.get("colliding_pairs") or []) if isinstance(p, dict)]
        if pairs:
            txt = "Colliding pairs: " + "; ".join(" / ".join(map(str, p.get("owners") or [])) +
                                                   (f" ({p.get('onsets')} onsets)" if p.get("onsets") else "")
                                                   for p in pairs)
            d.text((446, yb), fit(txt, font(15), R - 20 - 446), font=font(15), fill=RED)
            yb += 22
        ledger = []
        for ent in s.get("ledger") or []:
            if not isinstance(ent, dict):
                continue
            stt = str(ent.get("status", "?"))
            extra = self._result_text(ent) if stt == "done" else str(ent.get("reason") or "")
            ledger.append(f"{ent.get('stage', '?')} {stt}" + (f" ({extra})" if extra else ""))
        if ledger and yb + 18 <= B - 28:
            d.text((446, yb), fit("Harness ledger: " + ";  ".join(ledger), font(15), R - 20 - 446), font=font(15),
                   fill=TEXT)
        d.text((446, B - 25), fit("Evaluator only, never visible to the policy: LIBERO's done (check_success() each "
                                  "step) ends the episode; contact monitor on every physics substep.", font(13),
                                  R - 20 - 446), font=font(13), fill=MUTED)


# ================================================================ output
def encode(dash, out):
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + ".partial.mp4")
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-movflags", "+faststart", str(partial)]
    errf = partial.with_suffix(".log")
    with open(errf, "w+b") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        n = 0
        try:
            for i in range(len(dash.tl)):
                proc.stdin.write(dash.render(i).tobytes())
                n += 1
            proc.stdin.close()
            code = proc.wait(timeout=600)
        except BaseException:
            proc.kill()
            proc.wait()
            if partial.exists():
                partial.unlink()
            raise
        err.seek(0)
        msg = err.read().decode(errors="replace")
    errf.unlink(missing_ok=True)
    if code != 0:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"ffmpeg failed ({code}): {msg.strip()[-600:]}")
    partial.replace(out)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the CP-5 task dashboard video of one episode.")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help=f"output video (default RUN_DIR/{VIDEO_NAME})")
    ap.add_argument("--frames", default=None, help="comma-separated timeline indices to render as PNGs (no video)")
    ap.add_argument("--png-dir", default=None, help="directory for the --frames PNGs (required with --frames)")
    a = ap.parse_args(argv)
    if a.frames and not a.png_dir:
        ap.error("--frames needs --png-dir DIR (review PNGs are never written into the run directory)")
    run_dir = Path(a.run_dir)
    if not run_dir.is_dir():
        print(f"[render_task_video] no such run directory: {run_dir}", file=sys.stderr)
        return 2
    t0 = time.time()
    dash = Dashboard(run_dir)
    n = len(dash.tl)
    if n == 0:
        print(f"[render_task_video] nothing to render: no recorded telemetry in {run_dir / 'rec'}", file=sys.stderr)
        return 2
    if a.frames:
        png_dir = Path(a.png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        idx = set()
        for tok in a.frames.split(","):
            tok = tok.strip()
            if tok:
                try:
                    idx.add(min(max(int(tok), 0), n - 1))
                except ValueError:
                    print(f"[render_task_video] ignoring frame index {tok!r}", file=sys.stderr)
        # label positions are sticky across frames: walk the skipped items too (layout only) so every PNG is
        # identical to the same frame of the video
        for i in range((max(idx) + 1) if idx else 0):
            if i in idx:
                p = png_dir / f"task_{i:04d}.png"
                dash.render(i).save(p)
                print(p)
            else:
                dash.layout_only(i)
        print(f"[render_task_video] {len(idx)} PNGs of {n} timeline items in {time.time() - t0:.1f} s")
        return 0
    out = Path(a.out) if a.out else run_dir / VIDEO_NAME
    k = encode(dash, out)
    print(f"[render_task_video] wrote {out} ({k} frames, {W}x{H} @ {FPS} fps) in {time.time() - t0:.1f} s")
    if dash.errors:
        print(f"[render_task_video] panels with errors: {dash.errors}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:                          # report cleanly; the runner logs the tail of the output
        traceback.print_exc()
        print(f"[render_task_video] failed: {exc!r}", file=sys.stderr)
        sys.exit(1)
