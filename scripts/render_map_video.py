#!/usr/bin/env python
"""Post-episode 3D obstacle-map video: S2's own voxel map (what the planner treats as obstacles), frame-aligned
with the task video (same `vis_timeline.build_timeline` frame list, so both play side by side).

    python scripts/render_map_video.py RUN_DIR [--out PATH] [--frames I,J,K --png-dir DIR]

Inputs (recorder.py layout): <run>/rec/{meta.json, cam_agent.mp4, telemetry.jsonl, robot_points.npz, maps/*,
plans.jsonl}, <run>/events.jsonl, <run>/summary.json. Output: RUN_DIR/map_video.mp4 (1600x960, 10 fps, H.264), or
with --frames only PNGs of those timeline indices. The 3D labels are sticky across frames, so --frames replays the
label placement of the skipped items first: a review PNG equals the video frame with the same index.

What is drawn as an obstacle replicates cp5.perception.SceneMap.obstacles(): occupied + unknown voxels, nothing
below the table-surface voxel, and that surface voxel only under something standing on it (the planner treats the
table as an analytic plane). Between two snapshots the harness changes the map in two places without a snapshot:
it clears a grasped object's voxels when it attaches the payload, and re-inserts the object (surface + 2.5D column
fill, like SceneMap.insert_object) where it was released. Both are emulated from the telemetry 'held' field and the
events, so the map never shows an object twice or not at all. The status line uses the task video's wording (a
port of render_task_video.py's status helpers), so both videos read the same at the same timeline index.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vis_timeline import (PHASES, S1_EVENTS, build_timeline, current_map, current_plan, events_upto,  # noqa: E402
                          gap_label, last_event, load_run, phase_of, program_state)

VIDEO_NAME = "map_video.mp4"
W, H, FPS = 1600, 960, 10

# ---------------------------------------------------------------- palette (task-video dashboard theme)
BG = (12, 18, 30)
PANEL = (22, 31, 47)
TEXT = (232, 239, 250)
MUTED = (154, 173, 197)
CYAN = (59, 201, 236)
YELLOW = (255, 220, 84)
GREEN = (102, 227, 161)
RED = (250, 129, 125)
PLAN_YELLOW = (245, 210, 70)       # planned EEF path        (same as the task video)
PINK = (225, 95, 175)              # executed EEF path, current stage
PINK_DIM = (128, 66, 110)          # executed EEF path, earlier stages
OK_FILL = (17, 50, 41)
BAD_FILL = (62, 25, 31)
ORANGE = (255, 146, 40)            # occupied, label 0: an obstacle S1 never named
VIEW_BG = (15, 22, 35)
FOG = (176, 188, 206)              # unknown space
ROBOT = (206, 212, 222)
PEDESTAL = (92, 100, 116)          # robot base links (static, not in the collision samples)
PAYLOAD = (64, 214, 245)
TABLE = (58, 74, 98)
TD_FREE = (30, 41, 58)             # top-down: observed free table area
# object colours: only hues that no other meaning uses (orange = unsegmented, yellow = plan, pink = executed path,
# cyan = payload / place, mint = grasp, white / grey = robot, EEF, unknown); after the six bases, darker variants
_OBJ_BASE = ((86, 136, 255), (168, 112, 255), (236, 72, 72), (112, 204, 64), (170, 168, 58), (38, 168, 150))
OBJ_COLORS = _OBJ_BASE + tuple(tuple(int(round(c * 0.62)) for c in col) for col in _OBJ_BASE)

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

# ---------------------------------------------------------------- layout
MAIN = (24, 106, 1040, 870)                 # 3D panel (x0, y0, x1, y1)
VIEW = (34, 140, 1030, 862)                 # 3D render area inside it (996 x 722)
RIGHT_X0, RIGHT_X1 = 1052, 1576
AGENT_PANEL = (RIGHT_X0, 106, RIGHT_X1, 466)
INSET = 320
INSET_X, INSET_Y = RIGHT_X0 + 10, 138       # agent-view inset (320 x 320)
LEGEND_X0 = INSET_X + INSET + 10            # legend column right of the inset
TD_PANEL = (RIGHT_X0, 478, RIGHT_X1, 870)
STATUS = (24, 878, 1576, 930)
FOOTER_Y = 937

SS = 2                                      # supersampling of the cached voxel layer
SHIFT = 3                                   # fixed-point bits for sub-pixel polygon vertices
FRONTIER_CAP = 25000
BAND_M = 0.35                               # unknown space shown from the table up to table_z + 0.35 m
FOG_SPLAT, FOG_ALPHA, FOG_MAX = 0.55, 0.13, 0.6   # unknown splat width (voxels), alpha per splat, alpha cap
ROBOT_SPLAT, PAYLOAD_SPLAT = 0.014, 0.011   # splat width of robot / payload points (m)
CAM = (-0.22, 0.10, -12.0, 50.0, 2.2)       # 3D view: target x, target height above table, azimuth, elevation, distance
FIT_PAD = 0.10                              # table margin kept in frame around the occupied voxels (m)
FIT_ZCAP = 0.55                             # robot points / paths higher than table_z + this are not framed: the
                                            # obstacle map gets the pixels, the arm's top may leave the view
PEDESTAL_R = 0.085                          # radius of the drawn robot base (m)

# 3D label placement (view pixels)
LABEL_H = 19
HARD_TOL = 30.0                             # overlap (px^2) with a label / protected mark that forces a label to move
SWITCH_MARGIN = 1500.0                      # otherwise it moves only for a spot cheaper by this much (no flicker)
EASE_N = 3                                  # frames over which a label slides to a new spot
VOXEL_COST = 0.7                            # cost per voxel pixel a label would hide

# ---------------------------------------------------------------- status phrases (render_task_video.py wording)
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
MOVE_FOLLOW = {"to_pregrasp": ("force-guarded descent, then closing the gripper", "GRASP DESCENT + CLOSE"),
               "to_part": ("grasping and operating the part", "OPERATING THE PART"),
               "to_preplace": ("force-guarded descent to the placement", "PLACE DESCENT"),
               "look": ("capturing RGB-D views", "CAPTURING VIEWS"),
               "look_toward": ("capturing RGB-D views", "CAPTURING VIEWS")}
HOW_PHRASE = {"direct": "direct route", "lift": "lift route", "rrt": "RRT route"}
PLAN_HOLDS = ("S2 grasp planning", "S2 placement planning", "S2 part-grasp planning", "S2 motion planning")
PERCEIVE_HOLDS = ("S2 grounding", "S2 checking")


# ================================================================ small helpers
@lru_cache(maxsize=None)
def font(size, bold=False, mono=False, cond=False):
    if mono:
        name = "DejaVuSansMono.ttf"
    elif cond:
        name = "DejaVuSansCondensed-Bold.ttf" if bold else "DejaVuSansCondensed.ttf"
    else:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def fit_text(text, fnt, width):
    """Text cut (with an ellipsis) to at most `width` pixels."""
    text = str(text)
    if width <= 0:
        return ""
    if fnt.getlength(text) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fnt.getlength(text[:mid].rstrip() + "…") <= width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


def blend_rect(img, x0, y0, x1, y1, color, alpha):
    """Alpha-blend a solid rectangle into a uint8 RGB numpy image (clipped)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    reg = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (reg * (1 - alpha) + np.asarray(color, np.float32) * alpha).astype(np.uint8)


def fmt_int(n):
    return f"{int(n):,}"


@lru_cache(maxsize=None)
def square_offsets(size):
    a = -(size // 2)
    return tuple((dx, dy) for dy in range(a, a + size) for dx in range(a, a + size))


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def deg(rad):
    return math.degrees(rad) if isnum(rad) else float("nan")


def mm(m, d=1):
    return f"{m * 1000:.{d}f} mm" if isnum(m) else "? mm"


def short_cat(cat):
    """Short object name (same rule as the task video): the S1 category if short, else its last two words."""
    cat = str(cat or "").strip()
    if len(cat) <= 16:
        return cat
    words = cat.split()
    return " ".join(words[-2:]) if len(words) > 2 else cat


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _overlaps(a, boxes):
    """Overlap areas of box a with each row of boxes (N x 4)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    if not len(b):
        return np.zeros(0)
    w = np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0])
    h = np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1])
    return np.where((w > 0) & (h > 0), w * h, 0.0)


def _rect_sum(integ, r):
    """Number of set pixels of a mask inside rect r, from its integral image (cv2.integral layout)."""
    if integ is None:
        return 0.0
    h, w = integ.shape[0] - 1, integ.shape[1] - 1
    x0, x1 = int(np.clip(math.floor(r[0]), 0, w)), int(np.clip(math.ceil(r[2]), 0, w))
    y0, y1 = int(np.clip(math.floor(r[1]), 0, h)), int(np.clip(math.ceil(r[3]), 0, h))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])


@lru_cache(maxsize=1)
def _scatter_last_wins():
    """True when a fancy assignment with repeated indices keeps the last value (NumPy does this in practice)."""
    a = np.zeros(3)
    idx = np.array([1, 1, 1, 2, 2, 1, 2])
    a[idx] = np.arange(1.0, 8.0)
    return bool(a[1] == 6.0 and a[2] == 7.0)


def _yn(v):
    return "yes" if v is True else "no" if v is False else "n/a"


def libero_success(s):
    """(success, detail) from summary.json: LIBERO's rule, the first control step where check_success() is true
    (run_episode.py since 2026-11-20). Older runs only have the finish / after-hold checks (strict = both)."""
    if "success" in s:
        return s["success"] is True, (f"LIBERO done at step {s.get('success_step')}" if s["success"]
                                      else "LIBERO done never reached")
    fin, hold = s.get("raw_success_at_finish"), s.get("raw_success_after_hold")
    return fin is True and hold is True, f"old rule, at finish: {_yn(fin)}, after the hold: {_yn(hold)}"


def pill(d, right, top, text, color, fnt):
    """Filled rounded badge (task-video style), right-aligned at `right`."""
    w = fnt.getlength(text) + 16
    d.rounded_rectangle((right - w, top, right, top + 25), radius=7, fill=color)
    d.text((right - w + 8, top + 4), text, font=fnt, fill=BG)


def load_run_tolerant(run_dir):
    """vis_timeline.load_run, or the same structure with corrupt JSON files treated as missing when it raises."""
    try:
        return load_run(run_dir)
    except Exception as e:                                       # e.g. a summary.json truncated by a crash
        print(f"render_map_video: load_run failed ({e!r}); reading with corrupt files ignored", file=sys.stderr)

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
                    v = json.loads(line)
                except ValueError:
                    continue
                if isinstance(v, dict):
                    out.append(v)
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


def timeline_of(run):
    """build_timeline(run) with default arguments (shared with the task video); telemetry records without a
    numeric time / step are dropped only if the shared builder cannot cope with them."""
    try:
        return build_timeline(run)
    except Exception as e:
        print(f"render_map_video: build_timeline failed ({e!r}); dropping malformed telemetry records",
              file=sys.stderr)
        run["telemetry"] = [r for r in run["telemetry"] if isinstance(r, dict) and isnum(r.get("t"))
                            and isinstance(r.get("step"), int)]
        try:
            return build_timeline(run)
        except Exception as e2:
            print(f"render_map_video: no usable timeline ({e2!r})", file=sys.stderr)
            return []


# ================================================================ camera
class PinholeCam:
    """Pinhole camera looking from `eye` at `target` (z up); focal length / principal point fitted so that
    `fit_points` fill the (width x height) viewport with the given margins."""

    def __init__(self, eye, target, width, height, fit_points, margin=(18, 16, 18, 16)):
        self.eye = np.asarray(eye, float)
        f = np.asarray(target, float) - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        d = np.cross(f, r)
        self.R = np.stack([r, d, f])                     # rows: image right, image down, forward
        self.W, self.H = int(width), int(height)
        pc = (np.asarray(fit_points, float).reshape(-1, 3) - self.eye) @ self.R.T
        xn, yn = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
        ml, mt, mr, mb = margin
        fx = (self.W - ml - mr) / max(1e-6, xn.max() - xn.min())
        fy = (self.H - mt - mb) / max(1e-6, yn.max() - yn.min())
        self.f = min(fx, fy)
        self.cx = ml + (self.W - ml - mr) / 2 - self.f * (xn.max() + xn.min()) / 2
        self.cy = mt + (self.H - mt - mb) / 2 - self.f * (yn.max() + yn.min()) / 2

    def project(self, P, scale=1.0):
        pc = (np.asarray(P, np.float64).reshape(-1, 3) - self.eye) @ self.R.T
        z = np.maximum(pc[:, 2], 1e-6)
        u = (self.f * pc[:, 0] / z + self.cx) * scale
        v = (self.f * pc[:, 1] / z + self.cy) * scale
        return u, v, pc[:, 2]

    def uv(self, P, scale=1.0):
        u, v, _ = self.project(P, scale)
        return np.stack([u, v], 1)


# ================================================================ run data
class VideoFrames:
    """Forward-only reader of an upright RGB camera stream (one frame per recorded frame)."""

    def __init__(self, path):
        self.path = Path(path)
        self.cap = None
        self.pos = -1
        self.cur = None
        self._open()

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(self.path)) if self.path.exists() else None
        if self.cap is not None and not self.cap.isOpened():
            self.cap = None
        self.pos, self.cur = -1, None
        self.stale = False

    def get(self, k):
        if k < self.pos:
            self._open()
        while self.cap is not None and self.pos < k:
            ok, fr = self.cap.read()
            if not ok:
                self.cap.release()
                self.cap = None
                break
            self.pos += 1
            self.cur = fr
        self.stale = self.cur is not None and self.pos < k
        return None if self.cur is None else self.cur[:, :, ::-1]


class RobotPoints:
    def __init__(self, path):
        self.z = None
        self.keys = set()
        try:
            if Path(path).exists():
                self.z = np.load(path)
                self.keys = set(self.z.files)
        except Exception as e:                                   # corrupt / truncated archive: draw no robot
            print("robot_points.npz unreadable:", repr(e))
        self._cache = {}
        pf = []
        if "payload_frames" in self.keys:
            try:
                pf = sorted(int(x) for x in np.asarray(self.z["payload_frames"]).ravel())
            except Exception:
                pf = []
        if not pf:
            pf = sorted(int(k.split("_")[1]) for k in self.keys if k.startswith("payload_") and k[8:].isdigit())
        self.payload_frames = pf

    def get(self, kind, k):
        key = f"{kind}_{int(k)}"
        if key not in self.keys:
            return None
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            try:
                a = np.asarray(self.z[key], np.float64).reshape(-1, 3)
                a = a[np.all(np.isfinite(a), 1)]
            except Exception:
                a = None
            self._cache[key] = a
        return self._cache[key]

    def last_payload_upto(self, k):
        i = bisect.bisect_right(self.payload_frames, int(k))
        return self.payload_frames[i - 1] if i > 0 else None

    def scan(self, n_frames, base_xy, every=4, per_frame=60):
        """(subsample of all robot points over the episode for framing the 3D view, lowest robot point within
        0.12 m of the base axis = where the collision samples start)."""
        out, low = [], None
        for k in range(0, n_frames, every):
            a = self.get("robot", k)
            if a is None or not len(a):
                continue
            out.append(a[np.linspace(0, len(a) - 1, min(per_frame, len(a))).astype(int)])
            near = np.hypot(a[:, 0] - base_xy[0], a[:, 1] - base_xy[1]) < 0.12
            if near.any():
                z = float(a[near, 2].min())
                low = z if low is None else min(low, z)
        self._cache.clear()
        return (np.concatenate(out) if out else None), low


# ================================================================ map snapshots
def _neighbour_any(m):
    """6-neighbourhood dilation (no wrap-around) of a boolean grid."""
    out = np.zeros_like(m)
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


def _shadow_of(state, label, oid):
    """Unknown voxels hidden under object `oid`'s observed surface (SceneMap.finalize shadow attribution)."""
    nx, ny, nz = state.shape
    cur = np.zeros((nx, ny), bool)
    out = np.zeros(state.shape, bool)
    for k in range(nz - 1, -1, -1):
        s, lab = state[:, :, k], label[:, :, k]
        cur = np.where((s == 1) & (lab > 0), lab == oid, cur)
        cur = np.where((s == 0) | ((s == 1) & (lab == 0)), False, cur)
        out[:, :, k] = cur & (s == -1)
    return out


def _insert_solid(st, lab, lo, res, oid, pts):
    """SceneMap.insert_object: the object's points plus its 2.5D solid (ObjectModel: 5 mm top-height map of the
    points with small holes closed; every voxel whose centre lies between the object's bottom and that height)."""
    shape = np.array(st.shape)
    idx = np.floor((pts - lo) / res).astype(int)
    ok = np.all((idx >= 0) & (idx < shape), 1)
    idx = idx[ok]
    if not len(idx):
        return
    st[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    lab[idx[:, 0], idx[:, 1], idx[:, 2]] = oid
    cell = 0.005
    lo_xy = pts[:, :2].min(0) - 2 * cell
    n = np.ceil((pts[:, :2].max(0) + 2 * cell - lo_xy) / cell).astype(int) + 1
    hm = np.full(n, -np.inf)
    ij = np.floor((pts[:, :2] - lo_xy) / cell).astype(int)
    np.maximum.at(hm, (ij[:, 0], ij[:, 1]), pts[:, 2])
    try:
        from scipy import ndimage
        hm = ndimage.grey_closing(np.where(np.isfinite(hm), hm, -10.0), size=(3, 3))
        hm[hm < -5] = -np.inf
    except ImportError:
        pass
    bottom = float(pts[:, 2].min())                     # resting on its support after the release
    a = np.clip(np.floor((np.r_[pts[:, :2].min(0), bottom] - lo) / res).astype(int), 0, shape - 1)
    b = np.clip(np.floor((pts.max(0) - lo) / res).astype(int), 0, shape - 1)
    G = np.stack(np.meshgrid(*[np.arange(a[d], b[d] + 1) for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
    C = lo + (G + 0.5) * res
    q = np.floor((C[:, :2] - lo_xy) / cell).astype(int)
    okq = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < hm.shape[0]) & (q[:, 1] < hm.shape[1])
    top = np.full(len(G), -np.inf)
    top[okq] = hm[q[okq, 0], q[okq, 1]]
    G = G[np.isfinite(top) & (C[:, 2] <= top) & (C[:, 2] >= bottom)]
    st[G[:, 0], G[:, 1], G[:, 2]] = 1
    lab[G[:, 0], G[:, 1], G[:, 2]] = oid


class MapView:
    """One displayed map state: a snapshot, possibly with a grasped object removed / a released one re-inserted."""

    def __init__(self, data, rec, hide=(), insert=()):
        self.rec = rec
        self.inserted = tuple(oid for oid, _ in insert)
        self.hidden = tuple(o for o in hide if o not in self.inserted)
        st = np.asarray(data["state"]).astype(np.int8).copy()
        lab = np.asarray(data["label"]).astype(np.int32).copy()
        self.lo = np.asarray(data["lo"], float).reshape(3)
        self.res = float(data["res"])
        tz = float(data["table_z"]) if "table_z" in data else float("nan")
        vtz = float(data["visual_table_z"]) if "visual_table_z" in data else float("nan")
        self.table_z = tz if math.isfinite(tz) else (vtz if math.isfinite(vtz) else self.lo[2])
        plane_z = vtz if math.isfinite(vtz) else self.table_z
        for oid in hide:                                  # harness: scene.clear_label(oid) when the payload attaches
            sel = (lab == oid) | _shadow_of(st, lab, oid)
            st[sel] = 0
            lab[sel] = 0
        for oid, pts in insert:                           # harness: scene.insert_object(obj) at the release pose
            _insert_solid(st, lab, self.lo, self.res, oid, pts)
        self.state, self.label = st, lab
        self.shape = st.shape
        nz = st.shape[2]
        k0 = int(np.clip(np.floor((plane_z - self.lo[2]) / self.res + 1e-6), 0, nz - 2))   # == SceneMap.layer()
        self.k0 = k0
        occ, unk = st == 1, st == -1
        obs = occ | unk                                   # == SceneMap.obstacles(exclude=())
        obs[:, :, :k0] = False
        obs[:, :, k0] &= st[:, :, k0 + 1] == 1
        self.occ = occ & obs
        ktop = int(np.clip(np.floor((self.table_z + BAND_M - self.lo[2]) / self.res - 0.5), k0, nz - 1))
        band = unk & obs
        band[:, :, ktop + 1:] = False
        self.unknown_band = band
        self.n_unknown = int(band.sum())
        front = band & _neighbour_any(st == 0)
        self.n_frontier = int(front.sum())
        fidx = np.argwhere(front)
        if len(fidx) > FRONTIER_CAP:
            keep = np.sort(np.random.default_rng(0).choice(len(fidx), FRONTIER_CAP, replace=False))
            fidx = fidx[keep]
        self.frontier = fidx
        self.occ_idx = np.argwhere(self.occ)
        self.occ_lab = lab[self.occ_idx[:, 0], self.occ_idx[:, 1], self.occ_idx[:, 2]] if len(self.occ_idx) \
            else np.zeros(0, int)
        self.n_occ = int(len(self.occ_idx))
        self.n_unseg = int((self.occ_lab == 0).sum())

    def centers(self, idx):
        return self.lo + (np.asarray(idx, float) + 0.5) * self.res


# ================================================================ status text (task-video wording)
class Narrator:
    """What S2 is doing at a timeline item, in the task video's words: a port of render_task_video.py's
    Dashboard.state / now_text / activity / describe (same phrases, same event ranking), so the status lines of the
    two videos read the same at the same timeline index."""

    def __init__(self, run, tl):
        self.run, self.tl = run, tl
        self.meta = run.get("meta") or {}
        self.events = run.get("events") or []
        self.tel = run.get("telemetry") or []
        self.tel_t = [float(r.get("t", 0.0)) for r in self.tel]
        self.summary = run.get("summary") if isinstance(run.get("summary"), dict) else None
        self.s1 = []
        for e in self.events:
            ev = e.get("event")
            if ev not in S1_EVENTS:
                continue
            rec = e.get("record") if isinstance(e.get("record"), dict) else {}
            lat = rec.get("latency_s")
            t_end = float(e.get("t", 0.0))
            self.s1.append({"kind": ev, "t_end": t_end, "lat": lat if isnum(lat) else None,
                            "t_start": max(0.0, t_end - lat) if isnum(lat) else None, "event": e})
        self.end_event = next((e for e in reversed(self.events) if e.get("event") == "end"), None)
        self.by_name, self.sid_cat, self.obj_info = {}, {}, {}
        for e in self.events:
            ev = e.get("event")
            if ev == "grounded" and isinstance(e.get("summary"), dict):
                s = e["summary"]
                key = e.get("oid") if e.get("oid") is not None else e.get("object")
                try:
                    info = self.obj_info.setdefault(key, {})
                except TypeError:
                    continue
                info.update(sid=e.get("object"), cat=s.get("category") or s.get("name"), name=s.get("name"),
                            role=s.get("role"))
                if isinstance(s.get("name"), str):
                    self.by_name[s["name"]] = key
                if isinstance(e.get("object"), str):
                    self.sid_cat[e["object"]] = s.get("category") or s.get("name")
            elif ev == "ground_alias" and isinstance(e.get("object"), str):
                try:
                    info = self.obj_info.get(e.get("oid"))
                except TypeError:
                    info = None
                if info is not None:
                    self.sid_cat.setdefault(e["object"], info.get("cat"))
        self.held_k = [r.get("held") for r in self.tel]

    # ---------------------------------------------------------------- state
    def state(self, i):
        it = self.tl[i]
        k = int(it["frame"])
        t = float(it["t"])
        rec = self.tel[k] if 0 <= k < len(self.tel) else {}
        st = {"i": i, "k": k, "t": t, "step": int(it.get("step") or rec.get("step") or 0),
              "hold": bool(it.get("hold")) and not it.get("final"), "final": bool(it.get("final")),
              "label": it.get("label")}
        evs = events_upto(self.run, t)
        st["latest"] = evs[-1] if evs else None
        st["group"] = self.same_time(evs, st["latest"], s2_only=False)
        st["held_now"] = rec.get("held")
        plan = current_plan(self.run, t)
        st["plan"] = plan
        lt = float(st["latest"].get("t", -1)) if st["latest"] else -1.0
        st["plan_active"] = plan is not None and float(plan.get("t", -1)) > lt and not st["final"]
        if st["hold"]:
            t0 = self.tel_t[k] if 0 <= k < len(self.tel_t) else t
            t1 = self.tel_t[k + 1] if k + 1 < len(self.tel_t) else t
            st["hold_iv"] = (t0, t1)
        st["pending"] = next((c for c in self.s1 if c["t_start"] is not None and c["t_start"] <= t < c["t_end"]), None)
        st["ended"] = (not st["final"] and st["latest"] is not None and st["latest"].get("event") == "end")
        st["hold_label"], st["s1_returned"] = self._hold_label(st) if st["hold"] else (None, None)
        st["phase"] = self._phase(st)
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
        """Caption of a hold frame; inside an S1 gap, once the call has returned, the rest of the gap is S2 work."""
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
        if st["final"] or st["ended"]:
            return None
        latest = st["latest"]
        if st["hold"]:
            lab = st["label"] or ""
            if lab.startswith("S1 "):
                return 1
            if lab.startswith(PLAN_HOLDS):
                return 2
            if lab.startswith(PERCEIVE_HOLDS):
                return 0
            return phase_of(latest) if latest else None
        if st["plan_active"]:
            return phase_of({"event": "move", "tag": st["plan"].get("tag", "")})
        if latest is not None and latest.get("event") in ("payload", "payload_centering"):
            return 3
        return phase_of(latest) if latest else None

    # ---------------------------------------------------------------- phrases
    def obj_text(self, ref):
        """'o3 control knob' (S1 id + short category) for an S1 id or an object name."""
        if ref is None:
            return "-"
        if isinstance(ref, str) and ref in self.sid_cat:
            return f"{ref} {short_cat(self.sid_cat[ref])}"
        key = self.by_name.get(ref) if isinstance(ref, str) else None
        if key is not None and key in self.obj_info:
            info = self.obj_info[key]
            return f"{info.get('sid') or ''} {short_cat(info.get('cat'))}".strip()
        return str(ref)

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

    def describe(self, e, follow=False):
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
                s = f"{self.move_phrase(g('tag'))}: {'done' if status == 'done' else status.upper()} ({how})"
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
                return txt + (f", {s['n_points']} points)" if isnum(s.get("n_points")) else ")")
            if ev == "ground_fail":
                return f"grounding FAILED for {g('object')}"
            if ev == "ground_alias":
                return f"{g('object')} is the tracked object {g('oid')} ({g('why')})"
            if ev == "support_plane":
                return (f"support plane fitted: table top at z {g('resting'):.3f} m" if isnum(g("resting"))
                        else "support plane fitted")
            if ev == "mount_frame":
                return f"part mount frame fitted: protrusion {mm(g('protrusion'), 0)}, {g('inliers')} inliers"
            if ev == "part_grasp_cands":
                return f"part-grasp candidates: {g('n')}"
            if ev == "part_grasp_prefilter":
                rej = g("rejected") or {}
                return f"part-grasp prefilter: kept {g('kept')}" + (
                    " (rejected " + ", ".join(f"{k} {v}" for k, v in rej.items()) + ")" if rej else "")
            if ev == "part_grasp":
                return f"part grasp planned: width {mm(g('width'))}, pitch {deg(g('pitch')):.0f}°"
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
                return f"grasp planned: width {mm(g('width'))}, yaw {deg(g('yaw')):.0f}°, depth {mm(g('depth'), 0)}"
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
                return f"placement descent {g('status')}  →  releasing"
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
        ev = e.get("event")
        if ev == "stage_done":
            return f"stage {e.get('stage')} done"
        if ev == "stage_start":
            s = e.get("stage") or {}
            return f"stage {s.get('id')} started ({str(s.get('op') or '?')} {self.obj_text(s.get('object'))})"
        return self.describe(e)

    def describe_group(self, group, follow=False):
        """Events logged at one time: the most informative leads, the others follow in short form."""
        group = [e for e in group if isinstance(e, dict)]
        if not group:
            return ""
        keep = [e for e in group if EVENT_RANK.get(e.get("event"), 3) >= 0] or group[-1:]
        if len(keep) == 1:
            return self.describe(keep[0], follow=follow)
        lead = max(keep, key=lambda e: EVENT_RANK.get(e.get("event"), 3))
        rest = [self.short_event(e) for e in keep if e is not lead]
        return self.describe(lead, follow=follow) + "  →  " + ", ".join(rest)

    def held_text(self, key):
        try:
            info = self.obj_info.get(key) or {}
        except TypeError:
            info = {}
        return f"{info.get('sid') or ''} {short_cat(info.get('cat') or info.get('name') or key)}".strip()

    def last_held(self, k):
        for kk in range(min(k, len(self.held_k) - 1), -1, -1):
            if self.held_k[kk] is not None:
                return self.held_k[kk]
        return None

    def now_text(self, st):
        """Status line: what S2 is doing at this timeline item (render_task_video.Dashboard.now_text)."""
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
        if ev == "place_descend" and st["held_now"] is None:
            obj = self.last_held(st["k"])
            return (f"released {self.held_text(obj)}, retreating" if obj is not None
                    else "released the object, retreating")
        return self.describe_group(st["group"], follow=True)

    def activity(self, st):
        """Short upper-case activity (render_task_video.Dashboard.activity)."""
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
            return "RELEASED, RETREATING" if st["held_now"] is None else "RELEASING"
        if ev == "stage_start":
            return f"STAGE {str((e.get('stage') or {}).get('id') or '').upper()} STARTED".replace("  ", " ")
        if ev == "stage_done":
            return f"STAGE {str(e.get('stage') or '').upper()} DONE"
        if ev == "stage_failed":
            return f"STAGE {str(e.get('stage') or '').upper()} FAILED"
        if ev == "descend":
            return f"GRASP DESCENT STOPPED ({str(e.get('status') or '?').upper()})"
        fixed = {"actuated": "ACTUATION DONE", "placed": "PLACEMENT CHECKED", "survey": "SURVEY DONE",
                 "scene": "MAP INITIALISED", "blocked_descent": "GRASP DESCENT BLOCKED",
                 "twist_stall": "TWIST STALLED", "dropped": "OBJECT DROPPED", "escape": "ESCAPE MOVE",
                 "program": "S1 PROGRAM RECEIVED", "repair": "S1 REPAIR RECEIVED", "verify": "S1 VERIFY RECEIVED"}
        if ev in fixed:
            return fixed[ev]
        ph = st["phase"]
        return PHASES[ph] if ph is not None else "RUNNING"


# ================================================================ renderer
class MapVideo:
    def __init__(self, run_dir):
        self.run = load_run_tolerant(run_dir)
        self.tl = timeline_of(self.run)
        self.meta = self.run.get("meta") or {}
        self.tel = self.run.get("telemetry") or []
        self.rec_dir = self.run["rec"]
        self.maps = self.run.get("maps") or []
        self.map_pos = {m.get("file"): i for i, m in enumerate(self.maps)}
        self.robot = RobotPoints(self.rec_dir / "robot_points.npz")
        self.agent = VideoFrames(self.rec_dir / "cam_agent.mp4")
        self.narr = Narrator(self.run, self.tl)
        self._raw = {}
        self._ok, self._bad = set(), set()
        self._legend_cache = {}
        self._views = {}
        self._layers = {}
        self._td = {}
        self._inp = None
        self._tag_memory, self._unseg_reg = {}, []            # sticky 3D labels (see labels_for)
        self._lab_i, self._lab_out = -1, None
        self.objmeta = self._object_meta()
        self.eef = np.array([r.get("eef") if isinstance(r.get("eef"), (list, tuple)) and len(r.get("eef")) == 3
                             else [np.nan] * 3 for r in self.tel], float).reshape(-1, 3)
        self._tel_t = [float(r.get("t", 0.0)) for r in self.tel]
        self._geometry()
        self._chrome = self._make_chrome()

    # ------------------------------------------------------------ static facts
    def _object_meta(self):
        """S2 object id -> {'sids': [(t, S1 id)], 'cat', 'role', 'name'}: S1 ids and categories from the grounded /
        ground_alias events (as the task video names objects), names and roles from the snapshot index."""
        out = {}
        for m in self.maps:
            for key, v in (m.get("objects") or {}).items():
                oid = _as_int(key)
                if oid is None or not isinstance(v, dict):
                    continue
                d = out.setdefault(oid, {"sids": []})
                for f in ("name", "role"):
                    if v.get(f):
                        d[f] = v[f]
        for e in self.run.get("events") or []:
            ev = e.get("event")
            if ev not in ("grounded", "ground_alias"):
                continue
            oid = _as_int(e.get("oid"))
            if oid is None:
                continue
            d = out.setdefault(oid, {"sids": []})
            if isinstance(e.get("object"), str):
                d["sids"].append((float(e.get("t", 0.0)), e["object"]))
            s = e.get("summary") if isinstance(e.get("summary"), dict) else {}
            if s.get("category") or s.get("name"):
                d["cat"] = s.get("category") or s.get("name")
            if s.get("role"):
                d.setdefault("role", s["role"])
        for d in out.values():
            d["sids"].sort(key=lambda x: x[0])
        return out

    def sid_at(self, oid, t):
        d = self.objmeta.get(oid) or {}
        sids = d.get("sids") or []
        if not sids:
            return f"#{oid}"
        i = bisect.bisect_right([s[0] for s in sids], t)
        return sids[i - 1][1] if i > 0 else sids[0][1]

    def obj_name(self, oid, t):
        """'o3 control knob': S1 id + short category, like the task video."""
        d = self.objmeta.get(oid) or {}
        return f"{self.sid_at(oid, t)} {short_cat(d.get('cat') or d.get('name') or '')}".strip()

    def obj_tag(self, oid, t, mv=None):
        role = None
        if mv is not None:
            v = (mv.rec.get("objects") or {}).get(str(oid))
            role = v.get("role") if isinstance(v, dict) else None
        role = role or (self.objmeta.get(oid) or {}).get("role")
        return self.obj_name(oid, t) + (f" ({role})" if role else "")

    def _load_raw(self, i):
        if i not in self._raw:
            rec = self.maps[i]
            try:
                with np.load(self.rec_dir / "maps" / rec["file"]) as z:
                    self._raw[i] = {k: np.asarray(z[k]) for k in z.files}
                if "state" not in self._raw[i] or "label" not in self._raw[i] or self._raw[i]["state"].ndim != 3:
                    raise ValueError("no 3-D state / label arrays")
                self._ok.add(i)
            except Exception as e:
                if i not in self._bad:
                    print("render_map_video: map snapshot unreadable, using the nearest readable one:",
                          rec.get("file"), repr(e))
                self._bad.add(i)
                self._raw[i] = None
            while len(self._raw) > 4:
                self._raw.pop(next(iter(self._raw)))
        return self._raw[i]

    def _geometry(self):
        m = self.meta
        lo = m.get("map_lo")
        shape = m.get("map_shape")
        res = m.get("map_res")
        if lo is None or shape is None or res is None:
            for i in range(len(self.maps)):
                d = self._load_raw(i)
                if d is not None:
                    lo, shape, res = d["lo"].tolist(), list(d["state"].shape), float(d["res"])
                    break
        if lo is None:
            lo, shape, res = [-0.91, -0.75, 0.84], [140, 150, 80], 0.01
        self.lo = np.asarray(lo, float)
        self.res = float(res)
        self.shape = tuple(int(s) for s in shape)
        self.hi = self.lo + np.array(self.shape) * self.res
        tz = m.get("table_z")
        if tz is None:
            for i in range(len(self.maps)):
                d = self._load_raw(i)
                if d is not None and math.isfinite(float(d.get("table_z", np.nan))):
                    tz = float(d["table_z"])
                    break
        self.table_z = float(tz) if tz is not None else float(self.lo[2] + 0.06)
        self.base = np.asarray(m.get("robot_base") or [self.lo[0] + 0.25, 0.0, self.table_z + 0.01], float)
        rs, low = self.robot.scan(len(self.tel), self.base[:2])
        # the drawn base column ends where the collision samples start (they skip the static base links)
        self.pedestal_top = (low - 0.004) if low is not None and low > self.table_z + 0.03 else None
        # camera: from the head-camera side (+x), high, looking down at the workspace centre (+y right, robot back)
        target = np.array([CAM[0], 0.0, self.table_z + CAM[1]])
        az, el, dist = math.radians(CAM[2]), math.radians(CAM[3]), CAM[4]
        eye = target + dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        x0, y0, x1, y1 = self.lo[0], self.lo[1], self.hi[0], self.hi[1]
        tz = self.table_z
        zcap = tz + FIT_ZCAP
        # frame the obstacle map, not the whole map box (the top-down panel shows the full extent): the occupied
        # voxels of all snapshots and the table around them, the robot base, and robot points / EEF paths up to
        # zcap (higher look poses let the arm's top leave the view: the map gets the pixels)
        fit = [np.asarray(self.base, float)[None]]
        if self.pedestal_top is not None:
            fit.append(np.array([[self.base[0], self.base[1], self.pedestal_top]]))
        if rs is not None:
            fit.append(rs[rs[:, 2] <= zcap])
        occ = self._occupied_samples()
        if occ is not None and len(occ):
            q_lo, q_hi = np.percentile(occ, 0.5, axis=0), np.percentile(occ, 99.5, axis=0)
            fit.append(occ[np.all((occ >= q_lo) & (occ <= q_hi), 1)])
            xa, ya = q_lo[0] - FIT_PAD, q_lo[1] - FIT_PAD
            xb, yb = q_hi[0] + FIT_PAD, q_hi[1] + FIT_PAD
        else:
            xa, xb, ya, yb = self.base[0], self.base[0] + 0.8, -0.4, 0.4
        xa, xb = max(xa, x0), min(xb, x1)
        ya, yb = max(ya, y0), min(yb, y1)
        fit.append(np.array([[xa, ya, tz], [xa, yb, tz], [xb, ya, tz], [xb, yb, tz]]))
        paths = [np.asarray(p.get("eef") or [], float).reshape(-1, 3) for p in self.run["plans"]
                 if isinstance(p.get("eef"), list)]
        paths.append(self.eef)
        for pth in paths:
            pth = pth[np.all(np.isfinite(pth), 1)] if len(pth) else pth
            pth = pth[pth[:, 2] <= zcap] if len(pth) else pth
            if len(pth):
                fit.append(pth)
        fit = np.concatenate(fit)
        fit = fit[np.all(np.isfinite(fit), 1)]
        self.cam = PinholeCam(eye, target, VIEW[2] - VIEW[0], VIEW[3] - VIEW[1], fit, margin=(26, 20, 26, 24))
        self._pedestal = self._pedestal_faces()

    def _occupied_samples(self):
        """Centres of (a subsample of) the occupied obstacle voxels over all snapshots."""
        out = []
        for i in range(len(self.maps)):
            d = self._load_raw(i)
            if d is None:
                continue
            try:
                st = d["state"]
                lo, res = np.asarray(d["lo"], float), float(d["res"])
                vtz = float(d.get("visual_table_z", np.nan))
                tz = vtz if math.isfinite(vtz) else float(d.get("table_z", np.nan))
                k0 = int(np.clip(np.floor((tz - lo[2]) / res + 1e-6), 0, st.shape[2] - 2)) if math.isfinite(tz) else 0
                occ = st == 1
                occ[:, :, :k0] = False
                occ[:, :, k0] &= st[:, :, k0 + 1] == 1
                idx = np.argwhere(occ)[::5]
                out.append(lo + (idx + 0.5) * res)
            except Exception:
                continue
        return np.concatenate(out) if out else None

    def _pedestal_faces(self):
        """Robot base column (the base links are not in the collision samples): cylinder side quads facing the
        camera and the top cap as triangles, as (quads (N,4,3), colours (N,3), face centres (N,3))."""
        if self.pedestal_top is None:
            return None
        z0, z1 = self.table_z, self.pedestal_top
        bx, by = float(self.base[0]), float(self.base[1])
        n = 40
        th = np.linspace(0.0, 2 * np.pi, n + 1)
        quads, cols, ctrs = [], [], []
        col = np.asarray(PEDESTAL, float)
        for a, b in zip(th[:-1], th[1:]):
            pa = (bx + PEDESTAL_R * math.cos(a), by + PEDESTAL_R * math.sin(a))
            pb = (bx + PEDESTAL_R * math.cos(b), by + PEDESTAL_R * math.sin(b))
            q = np.array([[pa[0], pa[1], z0], [pb[0], pb[1], z0], [pb[0], pb[1], z1], [pa[0], pa[1], z1]])
            c = q.mean(0)
            nrm = np.array([math.cos((a + b) / 2), math.sin((a + b) / 2), 0.0])
            view = self.cam.eye - c
            view /= np.linalg.norm(view)
            if nrm @ view <= 0:
                continue
            quads.append(q)
            cols.append(col * (0.55 + 0.45 * float(nrm @ view)))
            ctrs.append(c)
            cap = np.array([[bx, by, z1], [pa[0], pa[1], z1], [pb[0], pb[1], z1], [pb[0], pb[1], z1]])
            quads.append(cap)
            cols.append(col * 1.12)
            ctrs.append(cap[:3].mean(0))
        for a, b in zip(th[:-1], th[1:]):                           # cap triangles of the back half as well
            pa = (bx + PEDESTAL_R * math.cos(a), by + PEDESTAL_R * math.sin(a))
            pb = (bx + PEDESTAL_R * math.cos(b), by + PEDESTAL_R * math.sin(b))
            nrm = np.array([math.cos((a + b) / 2), math.sin((a + b) / 2), 0.0])
            c = np.array([(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, z1])
            if nrm @ (self.cam.eye - c) <= 0:
                cap = np.array([[bx, by, z1], [pa[0], pa[1], z1], [pb[0], pb[1], z1], [pb[0], pb[1], z1]])
                quads.append(cap)
                cols.append(col * 1.12)
                ctrs.append(cap[:3].mean(0))
        if not quads:
            return None
        return np.array(quads), np.clip(np.array(cols), 0, 255), np.array(ctrs)

    # ------------------------------------------------------------ per-frame map state
    def map_state(self, item):
        """Key of the map shown at this timeline item: (snapshot index, cleared oids, ((oid, payload frame), ...)).

        Between snapshots the harness edits its map without a new snapshot: scene.clear_label(oid) when a grasped
        object's payload attaches ('payload' event; telemetry 'held' from then on) and scene.insert_object(obj) at
        the release pose right after a placement ('place_descend'); a dropped object stays cleared ('dropped')."""
        rec = current_map(self.run, item["t"])
        if rec is None:
            return None
        si = self._readable(self.map_pos.get(rec.get("file"), 0))
        if si is None:
            return None
        rec = self.maps[si]
        k = item["frame"]
        held_now = self._held(k)
        ev_t = self.run["event_t"]
        evs = self.run["events"][bisect.bisect_right(ev_t, float(rec.get("t", 0.0))):bisect.bisect_right(ev_t, item["t"])]
        kp = self.robot.last_payload_upto(k)
        hide, insert = [], []
        for key, v in (rec.get("objects") or {}).items():
            oid = _as_int(key)
            if oid is None or not isinstance(v, dict) or v.get("virtual"):
                continue
            name = v.get("name")
            if oid == held_now:                                  # in the hand now: not in the map
                if not v.get("held"):
                    hide.append(oid)
                continue
            if v.get("held"):                                    # snapshot taken while held, released since
                if kp is not None and not any(e.get("event") == "dropped" and e.get("object") == name for e in evs):
                    insert.append((oid, kp))
                continue
            pay = [e for e in evs if e.get("event") == "payload" and e.get("object") == name]
            if pay:                                              # grasped and released between two snapshots
                hide.append(oid)
                tp = float(pay[-1].get("t", 0.0))
                if kp is not None and any(e.get("event") == "place_descend" and float(e.get("t", 0.0)) >= tp
                                          for e in evs):
                    insert.append((oid, kp))
        return si, tuple(hide), tuple(insert)

    def _readable(self, si):
        """Index of the snapshot to show for snapshot `si`: itself, else the closest earlier (then later) readable one
        (a corrupt / missing file must not blank the map)."""
        order = [si] + list(range(si - 1, -1, -1)) + list(range(si + 1, len(self.maps)))
        for j in order:
            if j in self._bad:
                continue
            if j in self._ok or self._load_raw(j) is not None:
                return j
        return None

    def map_view(self, key):
        if key not in self._views:
            si, hide, insert = key
            d = self._load_raw(si)
            if d is None:
                self._views[key] = None
            else:
                ins = []
                for oid, kp in insert:
                    pts = self.robot.get("payload", kp)
                    if pts is not None and len(pts):
                        ins.append((oid, pts))
                self._views[key] = MapView(d, self.maps[si], hide=hide, insert=ins)
            while len(self._views) > 6:
                self._views.pop(next(iter(self._views)))
        return self._views[key]

    def color_of(self, lab):
        if lab is None or lab <= 0:
            return ORANGE
        return OBJ_COLORS[(int(lab) - 1) % len(OBJ_COLORS)]

    def snap_id(self, mv, sep="; "):
        """'#11 (12/17; rebuild, step 530, t 52.3 s)': the snapshot's file index (as the task video numbers it)."""
        si = self.map_pos.get(mv.rec.get("file"), 0)
        f = str(mv.rec.get("file", ""))
        idx = (f.replace("map_", "").replace(".npz", "").lstrip("0") or "0") if f.startswith("map_") else str(si)
        t = mv.rec.get("t")
        return (f"#{idx} ({si + 1}/{len(self.maps)}{sep}{mv.rec.get('reason', '?')}, step {mv.rec.get('step', '?')}"
                + (f", t {float(t):.1f} s)" if isnum(t) else ")"))

    # ------------------------------------------------------------ cached 3D layer (per map view)
    def layer(self, key):
        """(view image, view depth, object tags, occupied-voxel pixels, integral image of the solid pixels)."""
        if key in self._layers:
            return self._layers[key]
        mv = self.map_view(key) if key is not None else None
        cam = self.cam
        Wv, Hv = cam.W, cam.H
        Ws, Hs = Wv * SS, Hv * SS
        img = np.empty((Hs, Ws, 3), np.uint8)
        img[:] = VIEW_BG
        dep = np.full((Hs, Ws), np.inf, np.float32)
        self._draw_table(img)
        self._draw_solids(img, dep, mv)
        tags = []
        occ_uv = np.zeros((0, 2))
        if mv is not None:
            tags = self._tags(mv)
            self._draw_fog(img, dep, mv)
            if len(mv.occ_idx):
                occ_uv = cam.uv(mv.centers(mv.occ_idx[::3]))
        small = cv2.resize(img, (Wv, Hv), interpolation=cv2.INTER_AREA)
        dsmall = dep.reshape(Hv, SS, Wv, SS).min(axis=(1, 3))
        integ = cv2.integral(np.isfinite(dsmall).astype(np.uint8))
        self._layers[key] = (small, dsmall, tags, occ_uv, integ)
        if len(self._layers) > 12:
            self._layers.pop(next(iter(self._layers)))
        return self._layers[key]

    def _draw_table(self, img):
        cam, tz = self.cam, self.table_z
        x0, y0, x1, y1 = self.lo[0], self.lo[1], self.hi[0], self.hi[1]
        corners = np.array([[x0, y0, tz], [x1, y0, tz], [x1, y1, tz], [x0, y1, tz]])
        poly = np.round(cam.uv(corners, SS) * (1 << SHIFT)).astype(np.int32)
        over = img.copy()
        cv2.fillConvexPoly(over, poly, TABLE, cv2.LINE_AA, SHIFT)
        cv2.addWeighted(over, 0.55, img, 0.45, 0, dst=img)
        grid = img.copy()
        step = 0.1
        for x in np.arange(math.ceil(x0 / step) * step, x1 + 1e-6, step):
            a, b = cam.uv([[x, y0, tz], [x, y1, tz]], SS)
            cv2.line(grid, tuple(np.round(a * 8).astype(int)), tuple(np.round(b * 8).astype(int)), (96, 116, 146),
                     SS, cv2.LINE_AA, 3)
        for y in np.arange(math.ceil(y0 / step) * step, y1 + 1e-6, step):
            a, b = cam.uv([[x0, y, tz], [x1, y, tz]], SS)
            cv2.line(grid, tuple(np.round(a * 8).astype(int)), tuple(np.round(b * 8).astype(int)), (96, 116, 146),
                     SS, cv2.LINE_AA, 3)
        cv2.addWeighted(grid, 0.35, img, 0.65, 0, dst=img)
        cv2.polylines(img, [poly], True, (110, 132, 164), SS, cv2.LINE_AA, SHIFT)
        # robot base footprint
        th = np.linspace(0, 2 * np.pi, 64)
        ring = np.stack([self.base[0] + PEDESTAL_R * np.cos(th), self.base[1] + PEDESTAL_R * np.sin(th),
                         np.full_like(th, tz + 0.001)], 1)
        rp = np.round(cam.uv(ring, SS) * (1 << SHIFT)).astype(np.int32)
        over = img.copy()
        cv2.fillPoly(over, [rp], (70, 82, 104), cv2.LINE_AA, SHIFT)
        cv2.addWeighted(over, 0.6, img, 0.4, 0, dst=img)
        cv2.polylines(img, [rp], True, (150, 162, 184), SS, cv2.LINE_AA, SHIFT)

    _FACES = (  # (axis, sign, shade, corner offsets of the quad, in voxel units from the voxel's min corner)
        (2, 1, 1.00, ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        (0, 1, 0.80, ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        (1, 1, 0.66, ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
        (1, -1, 0.58, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        (0, -1, 0.50, ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    )

    def _draw_solids(self, img, dep, mv):
        """Occupied voxels as shaded cubes (exposed faces that face the camera) and the robot base column, in one
        far-to-near painter's order; writes the per-face depth into `dep`."""
        cam = self.cam
        quads, fills, edges, dist, depth = [], [], [], [], []
        if mv is not None and len(mv.occ_idx):
            idx = mv.occ_idx
            solid = mv.occ
            shp = np.array(mv.shape)
            vmin = mv.lo + idx * mv.res
            ctr = vmin + 0.5 * mv.res
            vdist = np.linalg.norm(ctr - cam.eye, axis=1)
            _, _, zc = cam.project(ctr)
            cols = np.array([self.color_of(int(l)) for l in mv.occ_lab], float)
            hfrac = np.clip((ctr[:, 2] - self.table_z) / 0.25, 0, 1)
            vox_shade = 0.78 + 0.28 * hfrac
            for axis, sign, shade, offs in self._FACES:
                nb = idx.copy()
                nb[:, axis] += sign
                inside = (nb[:, axis] >= 0) & (nb[:, axis] < shp[axis])
                exposed = np.ones(len(idx), bool)
                exposed[inside] = ~solid[nb[inside, 0], nb[inside, 1], nb[inside, 2]]
                plane = vmin[:, axis] + (mv.res if sign > 0 else 0.0)
                facing = sign * (cam.eye[axis] - plane) > 0
                sel = np.nonzero(exposed & facing)[0]
                if not len(sel):
                    continue
                quads.append(vmin[sel, None, :] + np.asarray(offs, float)[None] * mv.res)
                fc = np.clip(cols[sel] * (shade * vox_shade[sel])[:, None], 0, 255)
                fills.append(fc)
                edges.append(np.clip(fc * 0.55, 0, 255))
                dist.append(vdist[sel])
                depth.append(zc[sel])
        if self._pedestal is not None:
            pq, pc, pctr = self._pedestal
            quads.append(pq)
            fills.append(pc)
            edges.append(pc)
            dist.append(np.linalg.norm(pctr - cam.eye, axis=1))
            depth.append(cam.project(pctr)[2])
        if not quads:
            return
        quads = np.concatenate(quads)
        fills = np.concatenate(fills).astype(int).tolist()
        edges = np.concatenate(edges).astype(int).tolist()
        depth = np.concatenate(depth).astype(float).tolist()
        order = np.argsort(-np.concatenate(dist), kind="stable")
        pts = np.round(cam.uv(quads.reshape(-1, 3), SS).reshape(-1, 4, 2) * (1 << SHIFT)).astype(np.int32)
        for i in order.tolist():
            p = pts[i]
            cv2.fillConvexPoly(img, p, fills[i], cv2.LINE_8, SHIFT)
            if edges[i] != fills[i]:
                cv2.polylines(img, [p], True, edges[i], 1, cv2.LINE_8, SHIFT)
            cv2.fillConvexPoly(dep, p, depth[i], cv2.LINE_8, SHIFT)

    def _tags(self, mv):
        """What the 3D labels name in this map view: each labelled object and the three largest unsegmented
        components (>= 120 voxels), with their projected boxes (view pixels)."""
        tags = []
        if not len(mv.occ_idx):
            return tags
        ctr = mv.centers(mv.occ_idx)

        def box(P):
            u, v, z = self.cam.project(P)
            half = 0.5 * self.cam.f * mv.res / max(1e-3, float(np.median(z)))
            return (float(u.min() - half), float(v.min() - half), float(u.max() + half), float(v.max() + half))

        for lab in sorted(set(int(l) for l in np.unique(mv.occ_lab)) - {0}):
            sel = mv.occ_lab == lab
            if sel.sum() < 5:
                continue
            tags.append(dict(kind="obj", oid=lab, box=box(ctr[sel]), prio=1))
        sel0 = mv.occ_lab == 0
        if sel0.sum() >= 30:
            try:
                from scipy import ndimage
            except ImportError:
                return tags
            g = np.zeros(mv.shape, bool)
            i0 = mv.occ_idx[sel0]
            g[i0[:, 0], i0[:, 1], i0[:, 2]] = True
            comp, n = ndimage.label(g, structure=np.ones((3, 3, 3), bool))
            if n:
                sizes = ndimage.sum(g, comp, index=np.arange(1, n + 1))
                for ci in np.argsort(-sizes)[:3]:
                    if sizes[ci] < 120:
                        break
                    P = mv.centers(np.argwhere(comp == ci + 1))
                    tags.append(dict(kind="unseg", centroid=P.mean(0), box=box(P), prio=2))
        return tags

    def _draw_fog(self, img, dep, mv):
        """Unknown frontier as faint splats (about half a voxel wide), alpha-accumulated, hidden behind solids."""
        if not len(mv.frontier):
            return
        cam = self.cam
        Hs, Ws = dep.shape
        P = mv.centers(mv.frontier)
        u, v, z = cam.project(P, SS)
        size = np.clip(np.round(FOG_SPLAT * cam.f * SS * mv.res / np.maximum(z, 1e-3)), 2, 12).astype(int)
        dflat = dep.reshape(-1)
        hits = []
        for sz in np.unique(size):
            sel = size == sz
            ui, vi, zs = np.floor(u[sel]).astype(np.int64), np.floor(v[sel]).astype(np.int64), z[sel]
            for dx, dy in square_offsets(int(sz)):
                uu, vv = ui + dx, vi + dy
                ok = (uu >= 0) & (uu < Ws) & (vv >= 0) & (vv < Hs)
                pix = vv[ok] * Ws + uu[ok]
                hits.append(pix[zs[ok] < dflat[pix]])
        if not hits:
            return
        pix = np.concatenate(hits)
        if not len(pix):
            return
        cnt = np.bincount(pix, minlength=Hs * Ws)
        nz = np.nonzero(cnt)[0]
        a = np.minimum(1.0 - np.power(1.0 - FOG_ALPHA, cnt[nz].astype(np.float32)), FOG_MAX)[:, None]
        flat = img.reshape(-1, 3)
        flat[nz] = (flat[nz].astype(np.float32) * (1 - a) + np.asarray(FOG, np.float32) * a).astype(np.uint8)

    # ------------------------------------------------------------ top-down (per map view)
    def topdown(self, key, size):
        ck = (key, size)
        if ck in self._td:
            return self._td[ck]
        Wt, Ht = size
        mv = self.map_view(key) if key is not None else None
        nx, ny = self.shape[0], self.shape[1]
        img = np.empty((nx, ny, 3), np.float32)
        img[:] = TD_FREE
        if mv is not None and mv.state.shape[:2] == (nx, ny):
            cnt = mv.unknown_band.sum(2).astype(np.float32)
            a = np.clip(cnt / 10.0, 0, 1)[..., None] * 0.45
            img = img * (1 - a) + np.asarray(FOG, np.float32) * 0.8 * a
            occ = mv.occ
            has = occ.any(2)
            nz = occ.shape[2]
            ktop = nz - 1 - np.argmax(occ[:, :, ::-1], axis=2)
            ii, jj = np.nonzero(has)
            labs = mv.label[ii, jj, ktop[ii, jj]]
            zt = mv.lo[2] + (ktop[ii, jj] + 0.5) * mv.res
            shade = 0.55 + 0.45 * np.clip((zt - self.table_z) / 0.2, 0, 1)
            cols = np.array([self.color_of(int(l)) for l in labs], np.float32).reshape(-1, 3)
            img[ii, jj] = cols * shade[:, None]
        big = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        grid = big.copy()                                             # 10 cm grid
        for x in np.arange(math.ceil(self.lo[0] / 0.1) * 0.1, self.hi[0] + 1e-6, 0.1):
            r = int(round((x - self.lo[0]) / (nx * self.res) * Ht))
            cv2.line(grid, (0, r), (Wt - 1, r), (80, 98, 124), 1)
        for y in np.arange(math.ceil(self.lo[1] / 0.1) * 0.1, self.hi[1] + 1e-6, 0.1):
            c = int(round((y - self.lo[1]) / (ny * self.res) * Wt))
            cv2.line(grid, (c, 0), (c, Ht - 1), (80, 98, 124), 1)
        cv2.addWeighted(grid, 0.35, big, 0.65, 0, dst=big)
        bc = self.td_px(self.base[None], size)[0]
        rad = PEDESTAL_R / (ny * self.res) * Wt
        cv2.circle(big, (int(round(bc[0] * 4)), int(round(bc[1] * 4))), int(round(rad * 4)), (150, 162, 184), 1,
                   cv2.LINE_AA, 2)
        self._td[ck] = big
        if len(self._td) > 12:
            self._td.pop(next(iter(self._td)))
        return big

    def td_px(self, P, size):
        Wt, Ht = size
        P = np.asarray(P, float).reshape(-1, 3)
        c = (P[:, 1] - self.lo[1]) / (self.shape[1] * self.res) * Wt
        r = (P[:, 0] - self.lo[0]) / (self.shape[0] * self.res) * Ht
        return np.stack([c, r], 1)

    # ------------------------------------------------------------ static chrome
    def _make_chrome(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((24, 12), "S2 3D OBSTACLE MAP | WHAT THE PLANNER SEES", font=font(27, True), fill=TEXT)
        instr = self.meta.get("instruction") or (self.run["summary"] or {}).get("language") or ""
        d.text((24, 75), fit_text(instr, font(20), W - 48), font=font(20), fill=TEXT)
        d.rounded_rectangle(MAIN, radius=12, fill=PANEL)
        d.text((MAIN[0] + 16, MAIN[1] + 8), "3D MAP  |  view from the head-camera side", font=font(18, True),
               fill=CYAN)
        d.rounded_rectangle(AGENT_PANEL, radius=12, fill=PANEL)
        d.rounded_rectangle(TD_PANEL, radius=12, fill=PANEL)
        d.text((TD_PANEL[0] + 14, TD_PANEL[1] + 8), "TOP-DOWN OCCUPANCY (XY)", font=font(16, True), fill=TEXT)
        title_end = TD_PANEL[0] + 14 + font(16, True).getlength("TOP-DOWN OCCUPANCY (XY)") + 14
        for note in ("highest obstacle per column, grey = unknown", "top obstacle, grey = unknown",
                     "grey = unknown"):
            if TD_PANEL[2] - 14 - font(12).getlength(note) >= title_end:
                break
        d.text((TD_PANEL[2] - 14 - font(12).getlength(note), TD_PANEL[1] + 11), note, font=font(12), fill=MUTED)
        d.rounded_rectangle(STATUS, radius=12, fill=PANEL)
        arr = np.asarray(img).copy()
        ticks, ox, oy, Wt, Ht = self._td_axes(arr)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        f12, f12b = font(12), font(12, True)
        for xy, lbl in ticks:
            d.text(xy, lbl, font=f12, fill=MUTED)
        # axis names: y runs along the horizontal ticks (right), x down the vertical tick column (toward the camera)
        d.text((ox + Wt - int(f12b.getlength("y [m] →")), oy + Ht + 22), "y [m] →", font=f12b, fill=TEXT)
        d.text((ox - 50, oy - 18), "x [m] ↓", font=f12b, fill=TEXT)
        d.text((ox - 50 + int(f12b.getlength("x [m] ↓")) + 8, oy - 18), "toward the head camera", font=f12,
               fill=MUTED)
        d.text((24, FOOTER_Y), f"S2 voxel map, {self.res * 100:g} cm voxels, rendered after the episode from recorded "
                               "snapshots. Obstacles for planning = occupied + unknown space; the table is a plane.",
               font=font(14), fill=MUTED)
        return np.asarray(img).copy()

    # ------------------------------------------------------------ per-frame overlays
    def overlays(self, item):
        """Everything event-driven for this timeline item."""
        t = item["t"]
        k = item["frame"]
        prog, status, cur, calls, diag = program_state(self.run, t)
        stage = None
        if prog and cur is not None:
            stage = next((s for s in prog.get("stages", []) if isinstance(s, dict) and s.get("id") == cur), None)
        evs = events_upto(self.run, t)
        stage_t = None
        for e in reversed(evs):
            if e.get("event") == "stage_start":
                stage_t = float(e.get("t", 0.0))
                break
        grasp = place = None
        for e in evs:
            if stage_t is not None and float(e.get("t", 0.0)) < stage_t:
                continue
            ev = e.get("event")
            try:
                if ev in ("grasp_plan", "part_grasp") and e.get("p") is not None:
                    g = np.asarray(e["p"], float).reshape(3)
                    grasp = g if np.all(np.isfinite(g)) else grasp
                elif ev == "place_plan" and e.get("xy") is not None:
                    place = self._place_point(e)
            except (TypeError, ValueError):
                pass
        plan = current_plan(self.run, t)
        plan_active = False
        if plan is not None:
            lo = bisect.bisect_right(self.run["event_t"], float(plan.get("t", 0.0)))
            hi = bisect.bisect_right(self.run["event_t"], t)
            plan_active = not item["final"] and not any(e.get("event") == "move" for e in self.run["events"][lo:hi])
        k_stage = bisect.bisect_left(self._tel_t, stage_t) if stage_t is not None and self.tel else 0
        return dict(prog=prog, status=status, cur=cur, stage=stage, grasp=grasp, place=place, plan=plan,
                    plan_active=plan_active, k=k, k_stage=k_stage)

    def _place_point(self, e):
        xy = np.asarray(e["xy"], float).reshape(2)
        z = None
        for c in reversed([c for c in self.run["events"] if c.get("event") == "place_cands"
                           and float(c.get("t", 0)) <= float(e.get("t", 0))]):
            for cand in c.get("top") or []:
                try:
                    if cand.get("xy") is not None and np.allclose(cand["xy"], xy, atol=1e-3) and cand.get("rest_z"):
                        z = float(cand["rest_z"])
                        break
                except (TypeError, ValueError):
                    continue
            if z is not None:
                break
        return np.array([xy[0], xy[1], z if z is not None else self.table_z])

    def inputs(self, i):
        """Per-item inputs shared by label placement and drawing (memoised for the last item)."""
        if self._inp is not None and self._inp["i"] == i:
            return self._inp
        item = self.tl[i]
        k = item["frame"]
        ov = self.overlays(item)
        key = self.map_state(item)
        mv = self.map_view(key) if key is not None else None
        lkey = key if mv is not None else None
        self._inp = dict(i=i, item=item, k=k, ov=ov, key=lkey, mv=mv, layer=self.layer(lkey),
                         payload=self.robot.get("payload", k))
        return self._inp

    # ------------------------------------------------------------ sticky 3D labels
    def labels_for(self, i):
        """Placed 3D labels (and the result-banner rect) for timeline item i. Placement is sticky over consecutive
        items (hysteresis + easing), so it runs for every item in order; asking for a later item replays the skipped
        ones first (placement only), which makes --frames PNGs identical to the video frames."""
        if i < self._lab_i:
            self._tag_memory, self._unseg_reg = {}, []
            self._lab_i, self._lab_out = -1, None
        while self._lab_i < i:
            j = self._lab_i + 1
            self._lab_out = self._place_labels(j, self.inputs(j))
            self._lab_i = j
        return self._lab_out

    def _unseg_id(self, c, used):
        """Stable id of an unsegmented component across snapshots: the nearest one seen before (< 12 cm)."""
        best, bd = None, 0.12
        for uid, rc in enumerate(self._unseg_reg):
            if uid in used:
                continue
            dd = float(np.linalg.norm(rc - c))
            if dd < bd:
                best, bd = uid, dd
        if best is None:
            best = len(self._unseg_reg)
            self._unseg_reg.append(np.asarray(c, float))
        else:
            self._unseg_reg[best] = np.asarray(c, float)
        used.add(best)
        return best

    @staticmethod
    def _label_cands(it, tw, th):
        bx0, by0, bx1, by1 = it["box"]
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        if it.get("marker"):
            return [(bx1 + 4, cy - th / 2), (bx0 - 4 - tw, cy - th / 2), (cx - tw / 2, by0 - 4 - th),
                    (cx - tw / 2, by1 + 4), (bx1 + 4, by1 + 2), (bx0 - 4 - tw, by0 - th - 2),
                    (bx1 + 4, by0 - th - 2), (bx0 - 4 - tw, by1 + 2)]
        return [(cx - tw / 2, by1 + 6), (bx1 + 6, cy - th / 2), (bx0 - 6 - tw, cy - th / 2),
                (cx - tw / 2, by0 - 6 - th), (bx1 + 4, by1 + 2), (bx0 - 4 - tw, by1 + 2),
                (cx - tw / 2, by1 + 10 + th), (cx - tw / 2, by1 + 14 + 2 * th),
                (bx1 + 4, by0 - th - 2), (bx0 - 4 - tw, by0 - th - 2)]

    def _place_labels(self, i, inp):
        """Labels next to what they name, avoiding the EEF, the markers, the active planned path, the result banner,
        each other and the drawn solids (pixel-exact, from the layer's integral image, so small unlabelled
        components count too). A label keeps last frame's spot (relative to its object's box) unless that spot now
        collides with another label or a protected mark, or another spot is much cheaper; a move slides over
        EASE_N frames."""
        cam = self.cam
        item, ov, k = inp["item"], inp["ov"], inp["k"]
        t = float(item["t"])
        _, _, tags, occ_uv, integ = inp["layer"]
        banner = self._banner_rect(self.robot.get("robot", k), occ_uv) if item["final"] else None
        items, used = [], set()
        for tg in tags:
            if tg["kind"] == "obj":
                items.append(dict(key=f"obj:{tg['oid']}", text=self.obj_tag(tg["oid"], t, inp["mv"]),
                                  color=self.color_of(tg["oid"]), box=tg["box"], prio=1))
            else:
                uid = self._unseg_id(tg["centroid"], used)
                items.append(dict(key=f"unseg:{uid}", text="unsegmented", color=ORANGE, box=tg["box"], prio=2))
        payload = inp["payload"]
        held = self._held(k)
        if payload is not None and len(payload) and held is not None:
            u, v, _ = cam.project(payload)
            items.append(dict(key="payload", text=f"{self.obj_name(held, t)} (held)", color=PAYLOAD,
                              box=(float(u.min()), float(v.min()), float(u.max()), float(v.max())), prio=3))
        prot, prot_w = [], []
        for P, label, color in ((ov["grasp"], "grasp", GREEN), (ov["place"], "place", PAYLOAD)):
            if P is not None:
                u, v = cam.uv(P[None])[0]
                items.append(dict(key=label, text=label, color=color, box=(u - 9, v - 9, u + 9, v + 9), prio=0,
                                  marker=True))
                prot.append((u - 10, v - 10, u + 10, v + 10))
                prot_w.append(3.0)
        if 0 <= k < len(self.eef) and np.all(np.isfinite(self.eef[k])):
            u, v = cam.uv(self.eef[k][None])[0]
            prot.append((u - 14, v - 14, u + 14, v + 14))
            prot_w.append(3.0)
        plan = ov["plan"]
        if plan is not None and ov["plan_active"] and isinstance(plan.get("eef"), list) and plan["eef"]:
            P = np.asarray(plan["eef"], float).reshape(-1, 3)
            P = P[np.all(np.isfinite(P), 1)]
            if len(P):
                for x, y in cam.uv(P[::max(1, len(P) // 40)]):
                    prot.append((x - 6, y - 6, x + 6, y + 6))
                    prot_w.append(3.0)
        if banner is not None:
            prot.append((banner[0] - VIEW[0] - 4, banner[1] - VIEW[1] - 4, banner[2] - VIEW[0] + 4,
                         banner[3] - VIEW[1] + 4))
            prot_w.append(40.0)
        prot = np.array(prot, float).reshape(-1, 4)
        prot_w = np.array(prot_w, float)
        boxes = np.array([it["box"] for it in items], float).reshape(-1, 4)
        Wv, Hv = cam.W, cam.H
        fnt = font(13, True)
        placed, out = [], []
        for n, it in sorted(enumerate(items), key=lambda x: x[1]["prio"]):
            tw, th = int(fnt.getlength(it["text"])) + 10, LABEL_H
            bx0, by0, bx1, by1 = it["box"]
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            soft_w = np.where(np.arange(len(items)) == n, 0.0, 0.6)
            rects, costs, hard = [], [], []
            for rank, (x, y) in enumerate(self._label_cands(it, tw, th)):
                x = float(np.clip(x, 4, Wv - tw - 4))
                y = float(np.clip(y, 6, Hv - th - 4))
                r = (x, y, x + tw, y + th)
                lab_ov = _overlaps((r[0] - 5, r[1] - 3, r[2] + 5, r[3] + 3), placed)
                pro_ov = _overlaps(r, prot)
                c = 4.0 * float(lab_ov.sum()) + float((pro_ov * prot_w).sum())
                c += VOXEL_COST * _rect_sum(integ, r) + float((_overlaps(r, boxes) * soft_w).sum()) + 25.0 * rank
                rects.append(r)
                costs.append(c)
                hard.append(float(lab_ov.sum() + pro_ov.sum()))
            best = int(np.argmin(costs))
            m = self._tag_memory.get(it["key"])
            chosen = best
            if m is not None and 0 <= m["rank"] < len(rects) and m["rank"] != best:
                p = m["rank"]
                if hard[p] <= HARD_TOL and costs[p] - costs[best] < SWITCH_MARGIN:
                    chosen = p                                           # hysteresis: stay put
            r = rects[chosen]
            target = (r[0] - cx, r[1] - cy)                              # spot relative to the object's box
            cont = m is not None and m["seen"] == i - 1
            if cont and chosen != m["rank"]:
                start, left = m["shown"], EASE_N - 1
            elif cont and m["left"] > 0:
                start, left = m["start"], m["left"] - 1
            else:
                start, left = target, 0
            s = (EASE_N - left) / EASE_N
            x = float(np.clip(cx + start[0] + (target[0] - start[0]) * s, 4, Wv - tw - 4))
            y = float(np.clip(cy + start[1] + (target[1] - start[1]) * s, 6, Hv - th - 4))
            shown = (x, y, x + tw, y + th)
            self._tag_memory[it["key"]] = dict(rank=chosen, seen=i, shown=(x - cx, y - cy), start=start, left=left)
            placed.append(shown)
            out.append((it, shown))
        return dict(labels=out, banner=banner)

    def _draw_tags(self, canvas, texts, labels):
        """Leaders first, then the label boxes (a box is never crossed by another label's leader)."""
        fnt = font(13, True)
        for it, r in labels:
            x0, y0, x1, y1 = (int(round(r[0])) + VIEW[0], int(round(r[1])) + VIEW[1],
                              int(round(r[2])) + VIEW[0], int(round(r[3])) + VIEW[1])
            bx0, by0, bx1, by1 = it["box"]
            ob = (bx0 + VIEW[0], by0 + VIEW[1], bx1 + VIEW[0], by1 + VIEW[1])
            ocx, ocy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
            pt = (float(np.clip(ocx, x0, x1)), float(np.clip(ocy, y0, y1)))
            pb = (float(np.clip((x0 + x1) / 2, ob[0], ob[2])), float(np.clip((y0 + y1) / 2, ob[1], ob[3])))
            if math.hypot(pt[0] - pb[0], pt[1] - pb[1]) > 3:
                cv2.line(canvas, (int(pt[0]), int(pt[1])), (int(pb[0]), int(pb[1])), it["color"], 1, cv2.LINE_AA)
        for it, r in labels:
            x0, y0, x1, y1 = (int(round(r[0])) + VIEW[0], int(round(r[1])) + VIEW[1],
                              int(round(r[2])) + VIEW[0], int(round(r[3])) + VIEW[1])
            blend_rect(canvas, x0, y0, x1, y1, (8, 12, 20), 0.8)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), it["color"], 1, cv2.LINE_AA)
            texts.append(((x0 + 5, y0 + 2), it["text"], fnt, it["color"]))

    # ------------------------------------------------------------ frame
    def render(self, i):
        lab = self.labels_for(i)
        inp = self.inputs(i)
        item, ov, mv, k = inp["item"], inp["ov"], inp["mv"], inp["k"]
        canvas = self._chrome.copy()
        texts = []                                        # (xy, text, font, fill) drawn with PIL at the end

        # ---- 3D view
        base, dep, _, _, _ = inp["layer"]
        view = base.copy()
        robot = self.robot.get("robot", k)
        payload = inp["payload"]
        self._splat_points(view, dep, robot, payload)
        self._draw_paths_3d(view, ov, k)
        vx, vy = VIEW[0], VIEW[1]
        canvas[vy:vy + view.shape[0], vx:vx + view.shape[1]] = view
        banner = lab["banner"]
        self._draw_tags(canvas, texts, lab["labels"])
        if banner is None or banner[3] < VIEW[3] - 120:
            self._triad(canvas, texts)
        if mv is None:
            texts.append(((VIEW[0] + 20, VIEW[1] + 20), "no map snapshot recorded", font(20), MUTED))

        # ---- right column
        self._agent_panel(canvas, texts, k, item)
        self._legend(canvas, mv, k, float(item["t"]))
        self._topdown_panel(canvas, inp["key"], ov, robot, payload, k)

        # ---- status bar + header
        self._status(texts, i, item, ov, mv)
        pil = Image.fromarray(canvas)
        d = ImageDraw.Draw(pil)
        d.text((24, 48), fit_text(self._subtitle(mv, item), font(18), W - 48), font=font(18), fill=MUTED)
        for xy, txt, fnt, fill in texts:
            d.text(xy, txt, font=fnt, fill=fill)
        if item["final"]:
            pill(d, MAIN[2] - 14, MAIN[1] + 7, "EPISODE END", MUTED, font(14, True))
            self._result_banner(d, banner)
        elif item["hold"]:
            pill(d, MAIN[2] - 14, MAIN[1] + 7, "SIM PAUSED", YELLOW, font(14, True))
        return np.asarray(pil)

    def _banner_rect(self, robot, occ_uv):
        """Result banner over the top or the bottom of the 3D view, whichever hides less of the robot / map."""
        s = self.run["summary"] or {}
        bh = (150 + (22 if s.get("colliding_pairs") else 0)) if s else 100
        cands = [(VIEW[0] + 14, VIEW[3] - 10 - bh, VIEW[2] - 14, VIEW[3] - 10),
                 (VIEW[0] + 14, VIEW[1] + 10, VIEW[2] - 14, VIEW[1] + 10 + bh)]
        off = np.array([VIEW[0], VIEW[1]])

        def hidden(P, r):
            return int(np.sum((P[:, 0] >= r[0]) & (P[:, 0] <= r[2]) & (P[:, 1] >= r[1]) & (P[:, 1] <= r[3])))

        rob = self.cam.uv(robot[::3]) + off if robot is not None and len(robot) else np.zeros((0, 2))
        occ = occ_uv + off if len(occ_uv) else np.zeros((0, 2))
        # the end-state map matters more than the arm: hide the fewest occupied voxels, then the fewest robot points
        cost = [(hidden(occ, r), hidden(rob, r)) for r in cands]
        return cands[min(range(len(cands)), key=lambda j: cost[j])]

    def _subtitle(self, mv, item):
        m, a, sm = self.meta, self.run["args"] or {}, self.run["summary"] or {}
        parts = [f"build {m['build']}"] if m.get("build") else []
        task = m.get("task", a.get("task", sm.get("task")))
        ep = m.get("episode", a.get("episode", sm.get("episode")))
        if task is not None:
            parts.append(f"{task} ep{int(ep):03d}" if isnum(ep) else f"{task}")
        if mv is not None:
            si = self.map_pos.get(mv.rec.get("file"), 0)
            txt = f"map snapshot {self.snap_id(mv)}"
            cur = current_map(self.run, item["t"])
            ci = self.map_pos.get(cur.get("file"), si) if cur else si
            if ci != si:
                txt += f", snapshot {ci + 1} unreadable"
            t = float(item["t"])
            for o in mv.hidden:
                txt += f"; {self.sid_at(o, t)} grasped: cleared from the map"
            for o in mv.inserted:
                txt += f"; {self.sid_at(o, t)} released: re-inserted"
            parts.append(txt)
        else:
            parts.append("no map snapshot")
        return "  |  ".join(parts)

    def _splat_points(self, view, dep, robot, payload):
        """Robot collision samples (light grey) and payload (cyan) as z-buffered splats, depth-tested against the
        cached solid depth (1.5 cm in the points' favour: fingers touching an object stay visible)."""
        cam = self.cam
        Hv, Wv = dep.shape
        sets = []
        if robot is not None and len(robot):
            sets.append((robot, ROBOT, ROBOT_SPLAT))
        if payload is not None and len(payload):
            sets.append((payload, PAYLOAD, PAYLOAD_SPLAT))
        if not sets:
            return
        P = np.concatenate([s[0] for s in sets])
        col = np.concatenate([np.repeat(np.asarray(c, float)[None], len(p), 0) for p, c, _ in sets])
        metres = np.concatenate([np.full(len(p), m) for p, _, m in sets])
        u, v, z = cam.project(P)
        zn = (z - z.min()) / max(1e-6, z.max() - z.min())
        col = np.clip(col * (1.1 - 0.45 * zn)[:, None], 0, 255).astype(np.uint8)
        size = np.clip(np.round(cam.f * metres / np.maximum(z, 1e-3)), 2, 9).astype(int)
        # all (point, footprint offset) pairs at once, ordered far point -> near point; one scatter where the
        # last write wins = nearest point per pixel (checked at start-up; exact sort otherwise)
        far = np.argsort(-z, kind="stable")
        ui, vi = np.floor(u).astype(np.int64)[far], np.floor(v).astype(np.int64)[far]
        zs, size, col = z[far], size[far], col[far]
        smax = int(size.max())
        off = np.arange(-(smax // 2), -(smax // 2) + smax)
        dx, dy = np.meshgrid(off, off)
        dx, dy = dx.ravel()[None, :], dy.ravel()[None, :]
        lo_ = -(size // 2)[:, None]
        own = (dx >= lo_) & (dx < lo_ + size[:, None]) & (dy >= lo_) & (dy < lo_ + size[:, None])
        uu, vv = ui[:, None] + dx, vi[:, None] + dy
        ok = own & (uu >= 0) & (uu < Wv) & (vv >= 0) & (vv < Hv)
        pix = (vv * Wv + uu)[ok]
        ci = np.broadcast_to(np.arange(len(far))[:, None], ok.shape)[ok]
        vis = zs[ci] < dep.reshape(-1)[pix] + 0.015
        pix, ci = pix[vis], ci[vis]
        if not len(pix):
            return
        flat = view.reshape(-1, 3)
        if _scatter_last_wins():
            flat[pix] = col[ci]
        else:
            n = len(far)
            key = np.sort(pix * n + (n - 1 - ci))            # per pixel: largest ci (= nearest) first
            p, c = key // n, n - 1 - key % n
            first = np.r_[True, p[1:] != p[:-1]]
            flat[p[first]] = col[c[first]]

    def _draw_paths_3d(self, view, ov, k):
        cam = self.cam
        S = 1 << SHIFT

        def poly(P):
            P = np.asarray(P, float).reshape(-1, 3)
            P = P[np.all(np.isfinite(P), 1)]
            if len(P) < 2:
                return None
            return np.round(cam.uv(P) * S).astype(np.int32)

        plan = ov["plan"]
        if plan is not None and isinstance(plan.get("eef"), list) and plan["eef"]:
            try:
                p = poly(plan["eef"])
            except (TypeError, ValueError):
                p = None
            if p is not None:
                if ov["plan_active"]:
                    cv2.polylines(view, [p], False, (20, 24, 32), 5, cv2.LINE_AA, SHIFT)
                    cv2.polylines(view, [p], False, PLAN_YELLOW, 2, cv2.LINE_AA, SHIFT)
                    q = p[-1]
                    cv2.rectangle(view, (int(q[0] / S) - 3, int(q[1] / S) - 3), (int(q[0] / S) + 3, int(q[1] / S) + 3),
                                  PLAN_YELLOW, 1, cv2.LINE_AA)
                else:                                     # already executed: kept faint until the next plan
                    over = view.copy()
                    cv2.polylines(over, [p], False, PLAN_YELLOW, 2, cv2.LINE_AA, SHIFT)
                    cv2.addWeighted(over, 0.4, view, 0.6, 0, dst=view)
        if len(self.eef) and k >= 1:
            ks = int(np.clip(ov.get("k_stage", 0), 0, k))
            if ks > 1:
                p = poly(self.eef[:ks + 1])
                if p is not None:
                    cv2.polylines(view, [p], False, PINK_DIM, 2, cv2.LINE_AA, SHIFT)
            p = poly(self.eef[ks:k + 1])
            if p is not None:
                cv2.polylines(view, [p], False, PINK, 2, cv2.LINE_AA, SHIFT)
        if ov["grasp"] is not None:
            u, v = cam.uv(ov["grasp"][None])[0]
            c = (int(round(u * S)), int(round(v * S)))
            cv2.circle(view, c, 9 * S, (10, 20, 16), 4, cv2.LINE_AA, SHIFT)
            cv2.circle(view, c, 9 * S, GREEN, 2, cv2.LINE_AA, SHIFT)
            cv2.line(view, (c[0] - 4 * S, c[1]), (c[0] + 4 * S, c[1]), GREEN, 1, cv2.LINE_AA, SHIFT)
            cv2.line(view, (c[0], c[1] - 4 * S), (c[0], c[1] + 4 * S), GREEN, 1, cv2.LINE_AA, SHIFT)
        if ov["place"] is not None:
            u, v = cam.uv(ov["place"][None])[0]
            dia = np.round(np.array([[u, v - 10], [u + 10, v], [u, v + 10], [u - 10, v]]) * S).astype(np.int32)
            cv2.polylines(view, [dia], True, (10, 20, 30), 4, cv2.LINE_AA, SHIFT)
            cv2.polylines(view, [dia], True, CYAN, 2, cv2.LINE_AA, SHIFT)
            cv2.circle(view, (int(round(u * S)), int(round(v * S))), 2 * S, CYAN, -1, cv2.LINE_AA, SHIFT)
        if 0 <= k < len(self.eef) and np.all(np.isfinite(self.eef[k])):
            e = self.eef[k]
            u, v = cam.uv(e[None])[0]
            ub, vb = cam.uv(np.array([[e[0], e[1], self.table_z]]))[0]
            n = 14
            for j in range(n):
                if j % 2 == 0:
                    a, b = j / n, (j + 1) / n
                    cv2.line(view, (int(u + (ub - u) * a), int(v + (vb - v) * a)),
                             (int(u + (ub - u) * b), int(v + (vb - v) * b)), (200, 206, 218), 1, cv2.LINE_AA)
            cv2.circle(view, (int(round(ub)), int(round(vb))), 3, (200, 206, 218), 1, cv2.LINE_AA)
            cv2.circle(view, (int(round(u * 4)), int(round(v * 4))), 7 * 4, BG, -1, cv2.LINE_AA, 2)
            cv2.circle(view, (int(round(u * 4)), int(round(v * 4))), 5 * 4, TEXT, -1, cv2.LINE_AA, 2)

    def _triad(self, canvas, texts):
        cam = self.cam
        o = np.array([-0.24, 0.0, self.table_z])
        ox, oy = VIEW[0] + 52, VIEW[3] - 46
        u0, v0 = cam.uv(o[None])[0]
        for ax, name, col in ((0, "+x", (255, 120, 110)), (1, "+y", (120, 230, 140)), (2, "+z", (120, 170, 255))):
            p = o.copy()
            p[ax] += 0.1
            u1, v1 = cam.uv(p[None])[0]
            dd = np.array([u1 - u0, v1 - v0])
            dd = dd / max(1e-6, np.linalg.norm(dd)) * 34
            tip = (int(ox + dd[0]), int(oy + dd[1]))
            cv2.arrowedLine(canvas, (ox, oy), tip, col, 2, cv2.LINE_AA, tipLength=0.25)
            texts.append(((tip[0] + (4 if dd[0] >= 0 else -22), tip[1] - (16 if dd[1] < 0 else 2)), name,
                          font(13, True), col))

    def _agent_panel(self, canvas, texts, k, item):
        x0, y0 = INSET_X, AGENT_PANEL[1] + 8
        f15b, f13 = font(15, True), font(13)
        texts.append(((x0, y0), "AGENT VIEW", f15b, TEXT))
        texts.append(((x0 + int(f15b.getlength("AGENT VIEW ")), y0 + 2), "(context)", f13, MUTED))
        cap_end = x0 + int(f15b.getlength("AGENT VIEW ") + f13.getlength("(context)"))
        ix, iy = INSET_X, INSET_Y
        img = self.agent.get(k)
        stale = img is not None and self.agent.stale
        if item["final"] or item["hold"] or stale:
            word, col = ("STREAM ENDED", RED) if stale else (("FINAL FRAME", MUTED) if item["final"] else
                                                              ("SIM PAUSED", YELLOW))
            f = font(11, True)
            w = int(f.getlength(word)) + 14
            px0 = max(ix + INSET - w, cap_end + 8)                  # never over the caption
            cv2.rectangle(canvas, (px0, y0 + 1), (px0 + w, y0 + 18), col, -1)
            texts.append(((px0 + 7, y0 + 3), word, f, BG))
        else:
            f = font(11)
            lbl = f"frame {k}" + (f" / {len(self.tel) - 1}" if self.tel else "")
            texts.append(((ix + INSET - int(f.getlength(lbl)), y0 + 4), lbl, f, MUTED))
        if img is None:
            blend_rect(canvas, ix, iy, ix + INSET, iy + INSET, (8, 12, 20), 1.0)
            texts.append(((ix + (INSET - int(font(15).getlength("no camera frame"))) // 2, iy + INSET // 2 - 9),
                          "no camera frame", font(15), MUTED))
        else:
            small = cv2.resize(np.ascontiguousarray(img), (INSET, INSET), interpolation=cv2.INTER_AREA)
            canvas[iy:iy + INSET, ix:ix + INSET] = (small * 0.45).astype(np.uint8) if stale else small

    def _held(self, k):
        """S2 object id in the hand at recorded frame k (telemetry), or None."""
        try:
            h = self.tel[k].get("held") if 0 <= k < len(self.tel) else None
            return int(h) if h is not None else None
        except (TypeError, ValueError, AttributeError):
            return None

    # ------------------------------------------------------------ legend
    def _legend_entries(self, mv, k, t):
        objs = {}
        if mv is not None:
            for key, v in (mv.rec.get("objects") or {}).items():
                oid = _as_int(key)
                if oid is not None and isinstance(v, dict) and not v.get("virtual"):
                    objs[oid] = v
        present = set(int(l) for l in np.unique(mv.occ_lab)) if mv is not None and len(mv.occ_lab) else set()
        held_now = self._held(k)
        entries = []
        for oid in sorted(objs):                                     # stable order: S2 object id
            held = (oid == held_now) if self.tel else bool(objs[oid].get("held"))
            txt = self.obj_name(oid, t) + (" (held)" if held else "")
            entries.append(("box", self.color_of(oid), txt, oid in present or held))
        if not entries:
            entries.append(("none", MUTED, "no named objects yet", False))
        std = [("box", ORANGE, "unsegmented obstacle", True), ("fog", FOG, "unknown space (obstacle)", True),
               ("plane", TABLE, "table plane (analytic)", True), ("dots", ROBOT, "robot collision points", True),
               ("pedestal", PEDESTAL, "robot base (not sampled)", True), ("dots", PAYLOAD, "held payload", True),
               ("line", PLAN_YELLOW, "planned EEF path", True), ("line2", PINK, "executed EEF path", True),
               ("dot", TEXT, "current EEF", True), ("ring", GREEN, "grasp target", True),
               ("diamond", CYAN, "place target", True)]
        return tuple(entries), tuple(std)

    def _legend(self, canvas, mv, k, t):
        """Legend column right of the agent inset; the drawn patch is cached per entry list."""
        X0, Y0, X1, Y1 = LEGEND_X0 - 4, AGENT_PANEL[1] + 4, AGENT_PANEL[2] - 4, AGENT_PANEL[3] - 4
        objs, std = self._legend_entries(mv, k, t)
        hit = self._legend_cache.get((objs, std))
        if hit is None:
            patch = canvas[Y0:Y1, X0:X1].copy()
            sub = []
            self._legend_draw(patch, sub, list(objs), list(std), X0, Y0)
            pil = Image.fromarray(patch)
            d = ImageDraw.Draw(pil)
            for xy, txt, fnt, fill in sub:
                d.text((xy[0] - X0, xy[1] - Y0), txt, font=fnt, fill=fill)
            hit = np.asarray(pil).copy()
            if len(self._legend_cache) > 16:
                self._legend_cache.clear()
            self._legend_cache[(objs, std)] = hit
        canvas[Y0:Y1, X0:X1] = hit

    def _legend_draw(self, canvas, texts, obj_entries, std, OX, OY):
        """Draw the legend into a patch whose top-left is canvas point (OX, OY)."""
        x0 = LEGEND_X0 - OX
        x1 = AGENT_PANEL[2] - 8 - OX
        y = AGENT_PANEL[1] + 8 - OY
        texts.append(((x0 + OX, y + OY), "LEGEND", font(15, True), TEXT))
        y = INSET_Y - OY
        width = x1 - x0 - 21
        avail = INSET_Y + INSET - (y + OY)
        total = len(obj_entries) + len(std)
        row = int(np.clip(avail // max(1, total), 16, 22))           # many objects: tighter rows
        max_rows = avail // row
        if total > max_rows:                                          # too many: objects in the map first
            keep = max(1, max_rows - len(std) - 1)
            idx = sorted(sorted(range(len(obj_entries)), key=lambda j: 0 if obj_entries[j][3] else 1)[:keep])
            obj_entries = [obj_entries[j] for j in idx] + [("none", MUTED, f"+{len(obj_entries) - keep} more objects",
                                                            False)]
        f13, f12 = font(13, cond=True), font(12, cond=True)
        for kind, col, txt, live in (obj_entries + std)[:max_rows]:
            cy = y + row // 2 - 1
            if kind == "box":
                cv2.rectangle(canvas, (x0, cy - 6), (x0 + 13, cy + 7), col, -1)
            elif kind == "fog":
                for dx in range(0, 14, 4):
                    for dy in range(-6, 8, 4):
                        cv2.rectangle(canvas, (x0 + dx, cy + dy), (x0 + dx + 1, cy + dy + 1), col, -1)
            elif kind == "plane":
                cv2.rectangle(canvas, (x0, cy - 5), (x0 + 13, cy + 6), col, -1)
                cv2.rectangle(canvas, (x0, cy - 5), (x0 + 13, cy + 6), (110, 132, 164), 1)
            elif kind == "pedestal":
                cv2.rectangle(canvas, (x0 + 2, cy - 4), (x0 + 11, cy + 6), col, -1)
                cv2.ellipse(canvas, (x0 + 6, cy - 4), (5, 2), 0, 0, 360, tuple(min(255, int(c * 1.12)) for c in col),
                            -1, cv2.LINE_AA)
            elif kind == "dots":
                for dx in range(1, 14, 4):
                    for dy in (-4, 0, 4):
                        cv2.rectangle(canvas, (x0 + dx, cy + dy), (x0 + dx + 1, cy + dy + 1), col, -1)
            elif kind == "line":
                cv2.line(canvas, (x0, cy), (x0 + 14, cy), col, 2, cv2.LINE_AA)
            elif kind == "line2":
                cv2.line(canvas, (x0, cy), (x0 + 6, cy), PINK_DIM, 2, cv2.LINE_AA)
                cv2.line(canvas, (x0 + 7, cy), (x0 + 14, cy), col, 2, cv2.LINE_AA)
            elif kind == "dot":
                cv2.circle(canvas, (x0 + 7, cy), 5, BG, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x0 + 7, cy), 4, col, -1, cv2.LINE_AA)
            elif kind == "diamond":
                pts = np.array([[x0 + 7, cy - 7], [x0 + 14, cy], [x0 + 7, cy + 7], [x0, cy]], np.int32)
                cv2.polylines(canvas, [pts], True, col, 2, cv2.LINE_AA)
                cv2.circle(canvas, (x0 + 7, cy), 1, col, -1, cv2.LINE_AA)
            elif kind == "ring":
                cv2.circle(canvas, (x0 + 7, cy), 6, col, 2, cv2.LINE_AA)
                cv2.line(canvas, (x0 + 5, cy), (x0 + 9, cy), col, 1)
                cv2.line(canvas, (x0 + 7, cy - 2), (x0 + 7, cy + 2), col, 1)
            fnt = f13 if f13.getlength(txt) <= width and row >= 18 else f12
            texts.append(((x0 + 21 + OX, cy - 8 + OY), fit_text(txt, fnt, width), fnt,
                          TEXT if kind != "none" and (live or kind != "box") else MUTED))
            y += row

    # ------------------------------------------------------------ top-down panel
    def _td_geom(self):
        """Top-down plot placement: (ox, oy, (Wt, Ht)); rows = x (down = +x, toward the head camera), cols = y."""
        px0, py0, px1, py1 = TD_PANEL
        nx, ny = self.shape[0], self.shape[1]
        avail_w = px1 - px0 - 80
        avail_h = py1 - py0 - 94
        s = min(avail_w / ny, avail_h / nx)
        Wt, Ht = int(ny * s), int(nx * s)
        return px0 + 58 + (avail_w - Wt) // 2, py0 + 50, (Wt, Ht)

    def _td_axes(self, canvas):
        """Static frame and ticks of the top-down plot (drawn once into the chrome)."""
        ox, oy, (Wt, Ht) = self._td_geom()
        nx, ny = self.shape[0], self.shape[1]
        cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + Wt, oy + Ht), (70, 86, 110), 1)
        f12 = font(12)
        ticks = []
        for yv in np.arange(math.ceil(self.lo[1] / 0.2) * 0.2, self.hi[1] + 1e-6, 0.2):
            c = ox + (yv - self.lo[1]) / (ny * self.res) * Wt
            cv2.line(canvas, (int(c), oy + Ht), (int(c), oy + Ht + 4), MUTED, 1)
            lbl = f"{yv:+.1f}" if abs(yv) > 1e-6 else "0"
            ticks.append(((int(c - f12.getlength(lbl) / 2), oy + Ht + 6), lbl))
        for xv in np.arange(math.ceil(self.lo[0] / 0.2) * 0.2, self.hi[0] + 1e-6, 0.2):
            r = oy + (xv - self.lo[0]) / (nx * self.res) * Ht
            cv2.line(canvas, (ox - 5, int(r)), (ox - 1, int(r)), MUTED, 1)
            lbl = f"{xv:+.1f}" if abs(xv) > 1e-6 else "0"
            ticks.append(((int(ox - 8 - f12.getlength(lbl)), int(r - 7)), lbl))
        return ticks, ox, oy, Wt, Ht

    def _topdown_panel(self, canvas, key, ov, robot, payload, k):
        ox, oy, size = self._td_geom()
        Wt, Ht = size
        img = self.topdown(key, size).copy()
        S = 1 << SHIFT

        def P2(P):
            return np.round(self.td_px(P, size) * S).astype(np.int32)

        if robot is not None and len(robot):
            q = self.td_px(robot, size).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < Wt) & (q[:, 1] >= 0) & (q[:, 1] < Ht)
            q = q[ok]
            sub = img[q[:, 1], q[:, 0]].astype(np.float32)
            img[q[:, 1], q[:, 0]] = (sub * 0.35 + np.asarray(ROBOT, np.float32) * 0.65).astype(np.uint8)
        if payload is not None and len(payload):
            q = self.td_px(payload, size).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < Wt) & (q[:, 1] >= 0) & (q[:, 1] < Ht)
            q = q[ok]
            img[q[:, 1], q[:, 0]] = PAYLOAD
        plan = ov["plan"]
        if plan is not None and isinstance(plan.get("eef"), list) and len(plan["eef"]) >= 2:
            try:
                pp = P2(np.asarray(plan["eef"], float).reshape(-1, 3))
            except (TypeError, ValueError):
                pp = None
            if pp is not None:
                if ov["plan_active"]:
                    cv2.polylines(img, [pp], False, PLAN_YELLOW, 2, cv2.LINE_AA, SHIFT)
                else:
                    over = img.copy()
                    cv2.polylines(over, [pp], False, PLAN_YELLOW, 1, cv2.LINE_AA, SHIFT)
                    cv2.addWeighted(over, 0.5, img, 0.5, 0, dst=img)
        if len(self.eef) and k >= 1:
            ks = int(np.clip(ov.get("k_stage", 0), 0, k))
            for seg, col in ((self.eef[:ks + 1] if ks > 1 else self.eef[:0], PINK_DIM), (self.eef[ks:k + 1], PINK)):
                seg = seg[np.all(np.isfinite(seg), 1)] if len(seg) else seg
                if len(seg) >= 2:
                    cv2.polylines(img, [P2(seg)], False, col, 1, cv2.LINE_AA, SHIFT)
        if ov["grasp"] is not None:
            c = self.td_px(ov["grasp"][None], size)[0]
            cv2.circle(img, (int(round(c[0] * S)), int(round(c[1] * S))), 6 * S, GREEN, 2, cv2.LINE_AA, SHIFT)
        if ov["place"] is not None:
            c = self.td_px(ov["place"][None], size)[0]
            dia = np.round(np.array([[c[0], c[1] - 7], [c[0] + 7, c[1]], [c[0], c[1] + 7], [c[0] - 7, c[1]]]) * S)
            cv2.polylines(img, [dia.astype(np.int32)], True, CYAN, 2, cv2.LINE_AA, SHIFT)
        if 0 <= k < len(self.eef) and np.all(np.isfinite(self.eef[k])):
            c = self.td_px(self.eef[k][None], size)[0]
            cv2.circle(img, (int(round(c[0] * S)), int(round(c[1] * S))), 5 * S, BG, -1, cv2.LINE_AA, SHIFT)
            cv2.circle(img, (int(round(c[0] * S)), int(round(c[1] * S))), 4 * S, TEXT, -1, cv2.LINE_AA, SHIFT)
        canvas[oy:oy + Ht, ox:ox + Wt] = img

    # ------------------------------------------------------------ status bar
    def _stage_text(self, ov):
        stage = ov["stage"]
        if not stage:
            return "-"
        bits = [str(stage.get("id")), str(stage.get("op") or "")]
        if stage.get("object"):
            bits.append(str(stage["object"]))
        if stage.get("relation") and stage.get("reference"):
            bits.append(f"{stage['relation']} {stage['reference']}")
        txt = " ".join(b for b in bits if b)
        stt = (ov.get("status") or {}).get(stage.get("id"))
        return txt + (f" ({stt})" if stt and stt != "running" else "")

    def _status(self, texts, i, item, ov, mv):
        x0, y0, x1, y1 = STATUS
        st = self.narr.state(i)
        if item["final"]:
            act, now = "EPISODE END", self._verdict()
        else:
            act, now = self.narr.activity(st), self.narr.now_text(st)
        head = f"STEP {int(item['step']):04d}  |  stage {self._stage_text(ov)}  |  "
        fb, fr = font(17, True), font(16)
        color = YELLOW if st["hold"] else (TEXT if (st["ended"] or item["final"]) else CYAN)
        avail = x1 - x0 - 32
        if st["hold"] and "  |  " in act and fb.getlength(head + act) + 22 + fr.getlength(now) > avail:
            act = act.rsplit("  |  ", 1)[0]          # drop the gap length (the SIM PAUSED pill says it) before cutting
        line = fit_text(head + act, fb, avail)
        texts.append(((x0 + 16, y0 + 6), line, fb, color))
        if now.rstrip("…").strip() == act.split("  |  ")[0].strip():
            now = ""                                             # the hold caption already says it
        elif "simulator paused" in line:
            now = now.replace(", simulator paused", "")
        xr = x0 + 16 + int(fb.getlength(line)) + 22
        if now and not line.endswith("…") and x1 - 16 - xr > 140:
            if fr.getlength(now) > x1 - 16 - xr:
                fr = font(15)                                    # one size smaller before cutting the text
            texts.append(((xr, y0 + 7 + (16 - fr.size) // 2 + 1 if fr.size < 16 else y0 + 7),
                          fit_text(now, fr, x1 - 16 - xr), fr, TEXT))
        if mv is not None:
            line2 = (f"map snapshot {self.snap_id(mv, sep=': ')}   |   occupied obstacle voxels {fmt_int(mv.n_occ)} "
                     f"(unsegmented {fmt_int(mv.n_unseg)})   |   unknown voxels, table to +{int(BAND_M * 100)} cm: "
                     f"{fmt_int(mv.n_unknown)} (frontier drawn {fmt_int(min(mv.n_frontier, FRONTIER_CAP))})")
        else:
            line2 = "no map snapshot recorded"
        texts.append(((x0 + 16, y0 + 31), fit_text(line2, font(14), x1 - x0 - 32), font(14), MUTED))

    def _verdict(self):
        s = self.run["summary"]
        status = (self.narr.end_event or {}).get("status") or (s or {}).get("status")
        if not s:
            return f"episode over: {status or 'unknown'}; no evaluator result (summary.json missing)"
        ok, _ = libero_success(s)
        coll = [n for n, kk in (("scene", "scene_collision_free"), ("added obstacle", "added_obstacle_collision_free"))
                if s.get(kk) is False]
        return (f"episode over: {status or '?'}; evaluator: " + ("task success" if ok else "task NOT achieved")
                + (", collisions: " + " + ".join(coll) if coll else ", collision-free"))

    def _result_banner(self, d, rect):
        L, T, R, B = rect
        s = self.run["summary"]
        if not s:
            d.rounded_rectangle(rect, radius=12, fill=PANEL, outline=MUTED, width=3)
            end = last_event(self.run, float("inf"), ("end",))
            status = (end or {}).get("status") or "unknown"
            d.text((L + 20, T + 14), f"EPISODE END: {str(status).upper()}", font=font(24, True), fill=YELLOW)
            d.text((L + 20, T + 58), "No evaluator result: summary.json is missing for this run.", font=font(16),
                   fill=MUTED)
            return
        ok, detail = libero_success(s)
        scene, added = s.get("scene_collision_free"), s.get("added_obstacle_collision_free")
        all_ok = ok and scene is True and added is True
        col = GREEN if all_ok else RED
        d.rounded_rectangle(rect, radius=12, fill=OK_FILL if all_ok else BAD_FILL, outline=col, width=3)
        title = "RESULT: SUCCESS, COLLISION-FREE" if all_ok else (
            "RESULT: SUCCESS WITH COLLISIONS" if ok else "RESULT: TASK NOT ACHIEVED")
        d.text((L + 20, T + 12), title, font=font(24, True), fill=col)
        info = f"status {s.get('status', '?')}  |  repairs {s.get('repairs', 0)}"
        if self.meta.get("effort"):
            info += f"  |  Astra {self.meta['effort']}"
        d.text((R - 20 - font(14).getlength(info), T + 20), info, font=font(14), fill=MUTED)

        def item(x, y, ok, label, extra=""):
            c = GREEN if ok is True else (RED if ok is False else MUTED)
            d.ellipse((x, y + 1, x + 20, y + 21), fill=c)
            mark = "✓" if ok is True else ("✗" if ok is False else "?")
            fb = font(14, True)
            d.text((x + 10 - fb.getlength(mark) / 2, y + 2), mark, font=fb, fill=BG)
            d.text((x + 30, y + 1), label, font=font(16, True), fill=TEXT)
            if extra:
                d.text((x + 38 + font(16, True).getlength(label), y + 3), extra, font=font(13), fill=MUTED)

        y = T + 54
        item(L + 20, y, ok, "Task success", detail)
        item(L + 20, y + 31, scene, "Scene collision-free")
        item(L + 20, y + 62, added, "Added-obstacle collision-free")
        x2 = L + 540
        f15 = font(15)
        wall = s.get("wall_s_policy")
        d.text((x2, y + 2), f"Policy wall time: {wall:.1f} s" if isnum(wall) else "Policy wall time: n/a", font=f15,
               fill=TEXT)
        sim = s.get("sim_time_s")
        d.text((x2, y + 33), f"Simulated time: {sim:.1f} s ({s.get('sim_steps', '?')} steps)"
               if isnum(sim) else "Simulated time: n/a", font=f15, fill=TEXT)
        model = s.get("model") if isinstance(s.get("model"), dict) else {}
        kinds, lats = model.get("kinds") or [], model.get("latency_s") or []
        calls = ", ".join(f"{str(a).upper()} {b:.1f} s" for a, b in zip(kinds, lats) if isnum(b))
        d.text((x2, y + 64), fit_text(f"S1 calls: {model.get('calls', len(kinds))}" + (f"  ({calls})" if calls else ""),
                                      f15, R - 20 - x2), font=f15, fill=TEXT)
        pairs = s.get("colliding_pairs") or []
        if pairs:
            txt = "Colliding pairs: " + "; ".join(
                " / ".join(map(str, p.get("owners") or [])) + (f" ({p.get('onsets')} onsets)" if p.get("onsets") else "")
                for p in pairs if isinstance(p, dict))
            d.text((L + 20, B - 30), fit_text(txt, font(14), R - L - 40), font=font(14), fill=RED)


# ================================================================ main
def parse_frames(spec, n):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            i = int(tok)
        except ValueError:
            continue
        if n:
            out.append(int(np.clip(i if i >= 0 else n + i, 0, n - 1)))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help=f"output video (default RUN_DIR/{VIDEO_NAME})")
    ap.add_argument("--frames", default=None, help="comma-separated timeline indices: write PNGs only")
    ap.add_argument("--png-dir", default=None, help="directory for --frames PNGs")
    args = ap.parse_args()
    if args.frames and not args.png_dir:
        ap.error("--frames needs --png-dir (review PNGs are never written into the run directory)")
    t0 = time.time()
    run_dir = Path(args.run_dir)
    mvid = MapVideo(run_dir)
    n = len(mvid.tl)
    if n == 0:
        print("render_map_video: no recorded frames (rec/telemetry.jsonl empty or missing); nothing to render")
        return 0
    if args.frames:
        png_dir = Path(args.png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        idx = parse_frames(args.frames, n)
        for i in idx:
            fr = mvid.render(i)
            Image.fromarray(fr).save(png_dir / f"map_{i:04d}.png")
        print(f"render_map_video: {len(idx)} PNGs -> {png_dir} ({time.time() - t0:.1f} s)")
        return 0
    out = Path(args.out) if args.out else run_dir / VIDEO_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".part.mp4")
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            fr = mvid.render(i)
            proc.stdin.write(np.ascontiguousarray(fr, np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        if tmp.exists():
            tmp.unlink()
        raise
    if rc != 0:
        if tmp.exists():
            tmp.unlink()
        print(f"render_map_video: ffmpeg failed (exit {rc})")
        return 1
    os.replace(tmp, out)
    print(f"render_map_video: {n} frames -> {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
