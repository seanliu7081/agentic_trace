#!/usr/bin/env python
"""TRACE planning video: how every motion of an episode was planned, request by request.

    python scripts/trace_capture.py RUN_DIR             # once: replays the requests with recording hooks
    python scripts/render_trace_video.py RUN_DIR [--out PATH] [--frames I,J --png-dir DIR]

-> RUN_DIR/trace_video.mp4 (1920x1080, 10 fps, H.264). For each request: the start pose and the end-pose candidates
(kept / dropped and why), the straight-line check (fast path), and when TRACE searches: the A* corridors and fitted
seeds, every continuation level (tolerance inflation, scene blur, control poses, penalty), the sampled paths of every
explore iteration with each seed's best, the gradient refinement, which seeds survive, the certificate attempts, and
the result. The final path and certificate are the run's own (trace/req_*.pkl); the search in between comes from the
replay (trace/viz), and a request whose replay ends differently is marked. The 3D view is the map video's (S2's voxel
map at the request time, the arm where it stood), with TRACE's collision spheres drawn on it. Evaluator-side only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_map_video as rmv  # noqa: E402

VIDEO_NAME = "trace_video.mp4"
W, H, FPS = 1920, 1080, 10
RX0, RX1 = 1052, W - 24
REQ_PANEL = (RX0, 106, RX1, 388)
PIPE_PANEL = (RX0, 398, RX1, 640)
CERT_PANEL = (RX0, 650, RX1, 870)
STATUS = (24, 878, RX1, 930)
STRIP = (24, 940, RX1, 984)
FOOTER_Y = 994
CAM = 250

rmv.W, rmv.H = W, H
rmv.RIGHT_X0, rmv.RIGHT_X1 = RX0, RX1
rmv.STATUS = STATUS

font, fit_text, pill, isnum = rmv.font, rmv.fit_text, rmv.pill, rmv.isnum
BG, PANEL, TEXT, MUTED = rmv.BG, rmv.PANEL, rmv.TEXT, rmv.MUTED
CYAN, YELLOW, GREEN, RED, ORANGE = rmv.CYAN, rmv.YELLOW, rmv.GREEN, rmv.RED, (255, 160, 70)
EDGE = (44, 56, 74)
ACTIVE_FILL = (32, 91, 90)
BODY_COL = (196, 190, 255)            # TRACE's collision spheres at the start pose
SEED_COLS = [(255, 196, 84), (178, 140, 255), (90, 210, 255), (255, 120, 190), (150, 230, 120), (240, 240, 240)]
ROLE_COL = {"safety": RED, "require": YELLOW, "prefer": MUTED}
SHIFT = 3
TAG_PHRASE = {"look": "look move (wrist camera to a view point)", "look_toward": "look move toward an object",
              "side_look_back": "return from a side look", "to_pregrasp": "transit to the pre-grasp pose",
              "lift": "lift the grasped object", "v2_above": "carry the held object above the place point",
              "release_lift": "lift away after the release", "home": "retreat to the home pose",
              "to_press": "approach the part to press", "press_back": "back off after pressing",
              "back_off": "back off", "to_part": "approach the part", "retreat": "retreat",
              "to_preplace": "transit to the place pose", "to_above_place": "move above the place point"}
RESULT_COL = {"certified": GREEN, "no_path": RED, "conflict": ORANGE, "not_certifiable": ORANGE,
              "program_error": RED}
STEPS = ("1  END POSES", "2  STRAIGHT LINE", "3  SEEDS", "4  REFINE", "5  CERTIFY", "RESULT")


def wrap_px(text, fnt, width, max_lines):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if fnt.getlength(trial) <= width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = w if fnt.getlength(w) <= width else fit_text(w, fnt, width)
        if len(lines) == max_lines:
            lines[-1] = fit_text(lines[-1] + " …", fnt, width)
            return lines
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit_text(lines[-1] + " …", fnt, width)
    return lines


def tag_phrase(tag):
    base = str(tag or "move").split(":")[0]
    retry = base.endswith("_retry")
    base = base.replace("_retry", "")
    return TAG_PHRASE.get(base, base.replace("_", " ")) + (" (retry)" if retry else "")


def reason_text(c):
    """Why a certificate attempt was rejected, in words."""
    r = str(c.get("reason") or "")
    if r.startswith("arm:"):
        arm = c.get("arm") or {}
        return ("arm check: " + r[4:].replace("_", " ") +
                (f" at t={float(arm['at_t']):.2f}" if isnum(arm.get("at_t")) else ""))
    if r.startswith("terms:"):
        bad = sorted([x for x in c["rows"] if x["role"] != "prefer" and x["max_violation"] > 1.0],
                     key=lambda x: -x["max_violation"])
        return "over tolerance: " + ", ".join(f"{x['term']} {x['max_violation']:.1f}x" for x in bad[:3])
    return r or "rejected"


def result_text(res, stage):
    if res == "certified":
        return {"fast_path": "certified on the straight line", "search": "certified after the search",
                "seed": "certified on the harness's seed path"}.get(stage, "certified")
    return {"no_path": "no path", "conflict": "conflict (requirements cannot all hold)",
            "not_certifiable": "not certifiable with this perception (look closer)",
            "program_error": "program error"}.get(res, str(res)) + (" (at the end-pose check)" if stage == "terminal"
                                                                    else "")


class TraceVideo(rmv.MapVideo):
    def __init__(self, run_dir):
        super().__init__(run_dir)
        viz = Path(run_dir) / "trace" / "viz"
        idx = json.loads((viz / "index.json").read_text())["requests"]
        self.reqs = [pickle.load(open(viz / f"req_{r['id']}.pkl", "rb")) for r in idx]
        self.wrist = None
        self.head_cam = None
        f = self.rec_dir / "head_camera.json"
        if f.exists():
            c = json.loads(f.read_text())
            self.head_cam = (np.asarray(c["K"], float), np.asarray(c["T_wc"], float), float(c.get("size", 512)))
        for R in self.reqs:                   # the attempt the run reported: first failed search, else first failed fast path
            certs = [e for e in R["events"] if e["kind"] == "certify" and not e["ok"]]
            R["_reported"] = next((e for e in certs if e["phase"] != "fast_path"), certs[0] if certs else None)
            R["_failed"] = certs
        self.frames = self._build()
        self.tl = [dict(frame=fr["frame"], step=fr["step"], t=fr["t"], hold=True, label=None, final=False)
                   for fr in self.frames]
        self._inp, self._lab_i, self._lab_out = None, -1, None
        self._cam0, self._cam_ri, self._cam_cache = self.cam, None, {}

    # ------------------------------------------------------------------ per-request camera (zoomed on the move)
    def _req_cam(self, ri):
        if ri in self._cam_cache:
            return self._cam_cache[ri]
        R = self.reqs[ri]
        pts = [np.asarray(R["start_p"], float)[None]]
        for e in R["events"]:
            k = e["kind"]
            if k == "end":
                pts.append(np.asarray(e["p"], float)[None])
            elif k == "certify":
                pts.append(np.asarray(e["path"], float))
            elif k == "astar":
                pts += [np.asarray(c, float) for c in e["corridors"]]
            elif k in ("seeds", "search_end"):
                pts.append(np.asarray(e["paths"], float).reshape(-1, 3))
        if R["logged"]["path"] is not None:
            pts.append(np.asarray(R["logged"]["path"], float))
        P = np.concatenate(pts)
        P = P[np.all(np.isfinite(P), 1)]
        lo, hi = P.min(0) - 0.08, P.max(0) + 0.08
        c = (lo + hi) / 2
        half = np.maximum((hi - lo) / 2, 0.2)                 # at least 40 cm across: keep some context
        lo, hi = c - half, c + half
        lo[2] = min(lo[2], self.table_z)                       # the table under the move stays in view
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        base = self._cam0
        f = base.R[2]
        dist = float(np.clip(2.0 * np.linalg.norm(hi - lo), 1.0, 2.2))
        cam = rmv.PinholeCam(c - f * dist, c, base.W, base.H, corners, margin=(30, 64, 30, 30))
        self._cam_cache[ri] = cam
        return cam

    def _use_cam(self, ri):
        if self._cam_ri == ri:
            return
        self.cam = self._req_cam(ri)
        self._pedestal = self._pedestal_faces()
        self._layers.clear()
        self._tag_memory, self._unseg_reg = {}, []
        self._cam_ri = ri

    def inputs(self, i):
        self._use_cam(self.frames[i]["ri"])
        return super().inputs(i)

    # ------------------------------------------------------------------ frame list
    def _build(self):
        out = []
        for ri, R in enumerate(self.reqs):
            st = dict(ri=ri, pstep="request", ends=[], cands=None, cert=None, cert_phase=None, corridors=None,
                      seeds=None, ids=[], level=None, explore=None, exploit=None, select=None, dropped=[],
                      search_end=None, J=[], result=False, checking=False, note=None, gi=0, fast=None)
            base = dict(frame=int(R["frame"]), step=int(R["step"]), t=float(R["t"]))

            def emit(n, **kw):
                for j in range(n):
                    d = dict(st, **kw)
                    d.update(base, sub=j, n_sub=n)
                    d["J"] = list(st["J"])
                    out.append(d)
            ev = R["events"]
            st["ends"] = [e for e in ev if e["kind"] == "end"]
            emit(4, note="request")
            next_id = 0
            for e in ev:
                k = e["kind"]
                if k == "candidates":
                    st.update(cands=e, pstep="ends")
                    emit(5)
                elif k == "certify":
                    stp = "straight" if e["phase"] == "fast_path" else "certify"
                    st.update(pstep=stp, cert=e, cert_phase=e["phase"], checking=True)
                    if e["phase"] == "fast_path":
                        st["fast"] = e
                    emit(3)
                    st.update(checking=False)
                    emit(4)
                elif k == "astar":
                    st.update(pstep="seeds", corridors=e, seeds=None, cert=None)
                    emit(6)
                elif k == "seeds":
                    n = e["paths"].shape[0]
                    st.update(pstep="seeds", seeds=e, ids=list(range(n)), dropped=[], cert=None)
                    next_id = n
                    emit(5)
                elif k == "level":
                    st.update(pstep="refine", level=e, explore=None, exploit=None, select=None, dropped=[])
                    emit(3, note="level")
                elif k == "explore":
                    M = e["mean"].shape[0]
                    if M > len(st["ids"]):                # the harness's seed path joins at its K
                        add = M - len(st["ids"])
                        st["ids"] = st["ids"] + list(range(next_id, next_id + add))
                        next_id += add
                    st.update(pstep="refine", explore=e, exploit=None, select=None)
                    st["gi"] += 1
                    st["J"].append((st["gi"], e["level"], {st["ids"][m]: float(e["J_best"][m]) for m in range(M)}))
                    emit(1, note=None)
                elif k == "exploit":
                    M = e["after"].shape[0]
                    st.update(pstep="refine", exploit=e)
                    st["gi"] += 1
                    st["J"].append((st["gi"], e["level"], {st["ids"][m]: float(e["J"][m]) for m in range(M)
                                                           if m < len(st["ids"])}))
                    emit(3, note="exploit")
                elif k == "select":
                    idx = list(e["idx"])
                    if len(idx) < len(st["ids"]):
                        st["dropped"] = [st["ids"][i] for i in range(len(st["ids"])) if i not in idx]
                        st.update(select=e)
                        emit(3, note="select")
                    st["ids"] = [st["ids"][i] for i in idx if i < len(st["ids"])]
                    st["dropped"] = []
                elif k == "search_end":
                    st.update(search_end=e, explore=None, exploit=None, pstep="certify", cert=None)
                    emit(2, note="search_end")
            st.update(pstep="result", result=True, checking=False)
            emit(14)
        return out

    # ------------------------------------------------------------------ drawing helpers
    def _splat(self, view, dep, P, radii, color, alpha=1.0, test=True):
        cam = self.cam
        Hv, Wv = dep.shape
        P = np.asarray(P, float).reshape(-1, 3)
        if not len(P):
            return
        u, v, z = cam.project(P)
        zn = (z - z.min()) / max(1e-6, z.max() - z.min())
        col = np.clip(np.asarray(color, float)[None] * (1.1 - 0.4 * zn)[:, None], 0, 255)
        size = np.clip(np.round(cam.f * 2 * np.asarray(radii, float) / np.maximum(z, 1e-3)), 2, 14).astype(int)
        far = np.argsort(-z, kind="stable")
        ui, vi = np.floor(u).astype(np.int64)[far], np.floor(v).astype(np.int64)[far]
        zs, size, col = z[far], size[far], col[far]
        smax = int(size.max())
        off = np.arange(-(smax // 2), -(smax // 2) + smax)
        dx, dy = np.meshgrid(off, off)
        dx, dy = dx.ravel()[None, :], dy.ravel()[None, :]
        lo_ = -(size // 2)[:, None]
        own = (dx >= lo_) & (dx < lo_ + size[:, None]) & (dy >= lo_) & (dy < lo_ + size[:, None])
        rr = (dx - (lo_ + size[:, None] / 2 - 0.5)) ** 2 + (dy - (lo_ + size[:, None] / 2 - 0.5)) ** 2
        own &= rr <= (size[:, None] / 2) ** 2 + 0.5                                  # round splats
        uu, vv = ui[:, None] + dx, vi[:, None] + dy
        ok = own & (uu >= 0) & (uu < Wv) & (vv >= 0) & (vv < Hv)
        pix = (vv * Wv + uu)[ok]
        ci = np.broadcast_to(np.arange(len(far))[:, None], ok.shape)[ok]
        if test:
            vis = zs[ci] < dep.reshape(-1)[pix] + 0.015
            pix, ci = pix[vis], ci[vis]
        if not len(pix):
            return
        order = np.argsort(pix * len(far) + (len(far) - 1 - ci), kind="stable")   # per pixel the nearest first
        pix, ci = pix[order], ci[order]
        first = np.r_[True, pix[1:] != pix[:-1]]
        pix, ci = pix[first], ci[first]
        flat = view.reshape(-1, 3)
        if alpha >= 1.0:
            flat[pix] = col[ci].astype(np.uint8)
        else:
            flat[pix] = (flat[pix] * (1 - alpha) + col[ci] * alpha).astype(np.uint8)

    def _poly(self, view, P, col, th=2, alpha=1.0, outline=True, dashed=False):
        P = np.asarray(P, float).reshape(-1, 3)
        P = P[np.all(np.isfinite(P), 1)]
        if len(P) < 2:
            return
        S = 1 << SHIFT
        pts = np.round(self.cam.uv(P) * S).astype(np.int32)
        tgt = view if alpha >= 1.0 else view.copy()
        if dashed:
            segs = [pts[i:i + 2] for i in range(0, len(pts) - 1, 2)]
            if outline:
                cv2.polylines(tgt, segs, False, (14, 18, 26), th + 3, cv2.LINE_AA, SHIFT)
            cv2.polylines(tgt, segs, False, col, th, cv2.LINE_AA, SHIFT)
        else:
            if outline:
                cv2.polylines(tgt, [pts], False, (14, 18, 26), th + 3, cv2.LINE_AA, SHIFT)
            cv2.polylines(tgt, [pts], False, col, th, cv2.LINE_AA, SHIFT)
        if alpha < 1.0:
            cv2.addWeighted(tgt, alpha, view, 1 - alpha, 0, dst=view)

    def _dots(self, view, P, col, r=3):
        for u, v in self.cam.uv(np.asarray(P, float).reshape(-1, 3)):
            cv2.circle(view, (int(round(u)), int(round(v))), r + 1, (14, 18, 26), -1, cv2.LINE_AA)
            cv2.circle(view, (int(round(u)), int(round(v))), r, col, -1, cv2.LINE_AA)

    def _bodies_at(self, R, p, B, groups=None):
        c, r, g = B["centers"], B["radii"], B["groups"]
        sel = np.ones(len(r), bool) if groups is None else np.isin(g, groups)
        return c[sel] @ np.asarray(R, float).T + np.asarray(p, float), r[sel]

    # ------------------------------------------------------------------ 3D overlays
    def _overlay(self, view, dep, fr, texts):
        R = self.reqs[fr["ri"]]
        B = R["bodies"]
        vx, vy = rmv.VIEW[0], rmv.VIEW[1]
        start_P, start_r = self._bodies_at(R["start_R"], R["start_p"], B)
        self._splat(view, dep, start_P, start_r, BODY_COL, alpha=0.85)
        lab = []
        # end poses: grey until the check, then kept (green) / dropped (red, with the reason)
        status = {}
        if fr["cands"] is not None:
            for d in fr["cands"]["items"]:
                status[d["index"]] = d
        for e in fr["ends"]:
            d = status.get(e["index"])
            col = MUTED if d is None else (GREEN if d["ok"] else RED)
            look = bool(e.get("flag")) and float(e["flag"].get("uncertainty", 1.0)) > float(
                (R.get("margins") or {}).get("look_floor", 0.0))
            if look:
                col = ORANGE
            P, r = self._bodies_at(e["R"], e["p"], B, groups=[0, 1, 3])
            self._splat(view, dep, P, r, col, alpha=0.55 if not fr["result"] else 0.3)
            if d is not None and not d["ok"] and d.get("reason") and not fr["result"]:
                u, v = self.cam.uv(np.asarray(e["p"], float)[None])[0]
                lab.append((u, v, f"end {e['index']}: {d['reason']}", RED))
            elif look and not fr["result"]:
                u, v = self.cam.uv(np.asarray(e["p"], float)[None])[0]
                lab.append((u, v, f"end {e['index']}: needs a closer look", ORANGE))
        if not fr["ends"] and fr["cands"] is not None:             # free end: TRACE's own end poses
            for d in fr["cands"]["items"]:
                P, r = self._bodies_at(d["R"], d["p"], B, groups=[0, 1, 3])
                self._splat(view, dep, P, r, GREEN if d["ok"] else RED, alpha=0.5)
        # straight line from the start to the first end (fast path), always faint once past it
        tgt = next((e for e in fr["ends"]), None)
        if tgt is not None and fr["pstep"] in ("request", "ends"):
            self._poly(view, [R["start_p"], tgt["p"]], TEXT, 1, 0.6, outline=False, dashed=True)
        # search state
        if fr["corridors"] is not None and fr["pstep"] == "seeds":
            n = len(fr["corridors"]["corridors"])
            show = n if fr["seeds"] is not None else max(1, int(math.ceil((fr["sub"] + 1) / fr["n_sub"] * n)))
            for P in fr["corridors"]["corridors"][:show]:
                self._poly(view, P, CYAN, 2, 0.9 if fr["seeds"] is None else 0.45)
                self._dots(view, P[1:-1:2], CYAN, 2)
        if fr["seeds"] is not None and fr["pstep"] == "seeds":
            for m, P in enumerate(fr["seeds"]["paths"]):
                self._poly(view, P, SEED_COLS[m % len(SEED_COLS)], 2)
                self._dots(view, fr["seeds"]["ctrl"][m], SEED_COLS[m % len(SEED_COLS)], 3)
        if fr["pstep"] == "refine":
            ids = fr["ids"]
            ex, xp = fr["explore"], fr["exploit"]
            if ex is not None:
                for m in range(ex["mean"].shape[0]):
                    col = SEED_COLS[ids[m] % len(SEED_COLS)] if m < len(ids) else TEXT
                    for P in ex["samples"][m].astype(np.float32):
                        self._poly(view, P, col, 1, 0.45, outline=False)
                    self._poly(view, ex["mean"][m], col, 1, 0.8, outline=False)
                    self._poly(view, ex["best"][m], col, 3)
                    self._dots(view, ex["best_ctrl"][m], col, 3)
            if xp is not None:
                for m in range(xp["after"].shape[0]):
                    col = SEED_COLS[ids[m] % len(SEED_COLS)] if m < len(ids) else TEXT
                    self._poly(view, xp["after"][m], col, 3)
                    self._dots(view, xp["ctrl"][m], col, 4)
            if fr["select"] is not None and fr["dropped"]:
                lab.append((vx + 40, vy + 70, "worse half dropped: seed " + ", ".join(map(str, fr["dropped"])), MUTED))
            if ex is None and xp is None and fr["level"] is not None and fr["seeds"] is not None:
                for m, P in enumerate(fr["seeds"]["paths"]):
                    self._poly(view, P, SEED_COLS[m % len(SEED_COLS)], 2, 0.6)
        if fr["search_end"] is not None and fr["pstep"] == "certify" and fr["cert"] is None:
            for m, P in enumerate(fr["search_end"]["paths"]):
                col = SEED_COLS[fr["ids"][m] % len(SEED_COLS)] if m < len(fr["ids"]) else TEXT
                self._poly(view, P, col, 3 if m == 0 else 2, 1.0 if m == 0 else 0.5)
        # certificate attempt
        c = fr["cert"]
        if c is not None and fr["pstep"] in ("straight", "certify"):
            col = TEXT if fr["checking"] else (GREEN if c["ok"] else RED)
            self._poly(view, c["path"], col, 3)
            if not fr["checking"] and not c["ok"]:
                self._worst_marker(view, c, lab)
        # result: the run's own path (logged), with the gripper swept along it
        if fr["result"]:
            L = R["logged"]
            if L["path"] is not None:
                for j in np.linspace(0, len(L["path"]) - 1, 7).astype(int)[1:-1]:
                    P, r = self._bodies_at(L["R"][j], L["path"][j], B, groups=[0, 1, 3])
                    self._splat(view, dep, P, r, (150, 236, 190), alpha=0.35)
                self._poly(view, L["path"], GREEN, 4)
                self._dots(view, L["path"][-1:], GREEN, 5)
            elif R.get("_reported") is not None:
                for e in R["_failed"]:
                    if e is not R["_reported"]:
                        self._poly(view, e["path"], RED, 1, 0.45, outline=False)
                self._poly(view, R["_reported"]["path"], RED, 3)
                self._worst_marker(view, R["_reported"], lab)
        for u, v, txt, col in lab:
            f = font(13, True)
            tw = f.getlength(txt)
            x = int(np.clip(u + 12, 0, view.shape[1] - tw - 12))
            y = int(np.clip(v - 10, 4, view.shape[0] - 24))
            texts.append(("box", (vx + x - 4, vy + y - 2, vx + x + tw + 4, vy + y + 17), BG))
            texts.append(((vx + x, vy + y), txt, f, col))

    def _worst_marker(self, view, c, lab):
        if str(c.get("reason") or "").startswith("arm:"):
            arm = c.get("arm") or {}
            j = int(round(float(arm.get("at_t") or 0.0) * (len(c["path"]) - 1)))
            self._x_marker(view, np.asarray(c["path"][j], float), lab, reason_text(c))
            return
        rows = [r for r in c["rows"] if r["role"] != "prefer"]
        if not rows:
            return
        w = max(rows, key=lambda r: r["max_violation"])
        if w["max_violation"] <= 1.0:
            w = max(c["rows"], key=lambda r: r["max_violation"])
        if w.get("at_x") is not None:
            X = np.asarray(w["at_x"], float)
        else:
            j = int(round(float(w.get("at_t") or 0.0) * (len(c["path"]) - 1)))
            X = np.asarray(c["path"][j], float)
        self._x_marker(view, X, lab, f"{w['term']}: {w['max_violation']:.1f}x tolerance at t={float(w['at_t']):.2f}")

    def _x_marker(self, view, X, lab, text):
        u, v = self.cam.uv(np.asarray(X, float)[None])[0]
        cv2.circle(view, (int(u), int(v)), 11, (14, 18, 26), 4, cv2.LINE_AA)
        cv2.circle(view, (int(u), int(v)), 11, RED, 2, cv2.LINE_AA)
        cv2.line(view, (int(u) - 7, int(v) - 7), (int(u) + 7, int(v) + 7), RED, 2, cv2.LINE_AA)
        cv2.line(view, (int(u) - 7, int(v) + 7), (int(u) + 7, int(v) - 7), RED, 2, cv2.LINE_AA)
        lab.append((u, v + 12, text, RED))

    # ------------------------------------------------------------------ panels
    def _head(self, canvas, R, fr):
        x0, y0 = REQ_PANEL[0] + 12, REQ_PANEL[1] + 34
        key = (R["frame"], fr["ri"])
        img = self.agent.get(int(R["frame"]))
        if img is None:
            return
        small = cv2.resize(np.ascontiguousarray(img), (CAM, CAM), interpolation=cv2.INTER_AREA)
        if self.head_cam is not None:
            K, T, size = self.head_cam
            sc = CAM / size

            def uv(P):
                pc = (np.asarray(P, float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]
                z = np.maximum(pc[:, 2], 1e-3)
                return np.stack([(K[0, 0] * pc[:, 0] / z + K[0, 2] - 0.5) * sc,
                                 (K[1, 1] * pc[:, 1] / z + K[1, 2] - 0.5) * sc], 1)
            S = 1 << SHIFT
            L = R["logged"]
            if L["path"] is not None and fr["result"]:
                cv2.polylines(small, [np.round(uv(L["path"]) * S).astype(np.int32)], False, GREEN, 2, cv2.LINE_AA,
                              SHIFT)
            s = uv(R["start_p"][None])[0]
            cv2.circle(small, (int(s[0]), int(s[1])), 5, BG, -1, cv2.LINE_AA)
            cv2.circle(small, (int(s[0]), int(s[1])), 4, TEXT, -1, cv2.LINE_AA)
            for e in fr["ends"][:3]:
                q = uv(np.asarray(e["p"], float)[None])[0]
                cv2.circle(small, (int(q[0]), int(q[1])), 6, GREEN, 2, cv2.LINE_AA)
        canvas[y0:y0 + CAM, x0:x0 + CAM] = small

    def _req_panel(self, canvas, texts, fr):
        R = self.reqs[fr["ri"]]
        x0, y0, x1, y1 = REQ_PANEL
        texts.append(((x0 + 12, y0 + 9), "HEAD CAMERA", font(13, True), MUTED))
        self._head(canvas, R, fr)
        tx, tw = x0 + 12 + CAM + 16, x1 - (x0 + 12 + CAM + 16) - 14
        y = y0 + 9
        n = len(self.reqs)
        texts.append(((tx, y), f"REQUEST {fr['ri'] + 1}/{n}  ·  {R['id']}", font(17, True), CYAN))
        y += 26
        for line in wrap_px(tag_phrase(R["tag"]), font(16, True), tw, 2):
            texts.append(((tx, y), line, font(16, True), TEXT))
            y += 21
        mode = {"straight": "straight line only (fast path)", "trace": "fast path, then full search",
                "v0": "fast path, then the harness's seed paths"}.get(R["mode"], R["mode"])
        f14 = font(14)
        info = [f"Mode: {mode}  |  budget {R['budget_ms']:.0f} ms",
                f"Holding: {R['holding'] or 'nothing'}  |  end candidates: {R['n_ends'] or 'free end'}"]
        for line in info:
            texts.append(((tx, y + 3), fit_text(line, f14, tw), f14, MUTED))
            y += 19
        y += 6
        texts.append(((tx, y), f"Requirements ({len(R['terms'])} terms, each a residual / its tolerance):",
                      font(13, True), TEXT))
        y += 19
        groups = {}
        for t in R["terms"]:
            what = t["prim"].replace("_", " ")
            body = (t.get("args") or {}).get("body")
            groups.setdefault(t["role"], []).append(what + (f" {body}" if isinstance(body, str) else ""))
        for role in ("safety", "require", "prefer"):
            if role not in groups:
                continue
            items = sorted(set(groups[role]), key=groups[role].index)
            line = f"{role} ({len(groups[role])}): " + ", ".join(items)
            for ln in wrap_px(line, font(13), tw, 2):
                texts.append(((tx, y), ln, font(13), ROLE_COL[role]))
                y += 17

    def _pipe_panel(self, canvas, texts, fr):
        R = self.reqs[fr["ri"]]
        x0, y0, x1, y1 = PIPE_PANEL
        texts.append(((x0 + 12, y0 + 8), "HOW TRACE PLANS THIS MOVE", font(16, True), CYAN))
        order = {"request": -1, "ends": 0, "straight": 1, "seeds": 2, "refine": 3, "certify": 4, "result": 5}
        cur = order.get(fr["pstep"], -1)
        n = len(STEPS)
        bx, gap = x0 + 12, 8
        bw = (x1 - x0 - 24 - gap * (n - 1)) / n
        searched = any(e["kind"] in ("seeds", "search_start") for e in R["events"])
        for j, lab in enumerate(STEPS):
            a, b = int(bx + j * (bw + gap)), int(bx + j * (bw + gap) + bw)
            skipped = j in (2, 3) and not searched and cur >= 4
            on = j == cur
            cv2.rectangle(canvas, (a, y0 + 34), (b, y0 + 62), ACTIVE_FILL if on else PANEL, -1)
            cv2.rectangle(canvas, (a, y0 + 34), (b, y0 + 62), CYAN if on else EDGE, 2 if on else 1)
            texts.append(((a + 8, y0 + 40), fit_text(lab, font(13, True), b - a - 12), font(13, True),
                          TEXT if on else (EDGE if skipped else MUTED)))
        # level schedule
        tx, ty = x0 + 12, y0 + 74
        cols = ("level", "tolerance x", "scene blur", "control poses K", "penalty mu")
        cx = [tx, tx + 60, tx + 150, tx + 245, tx + 370]
        for c, x in zip(cols, cx):
            texts.append(((x, ty), c, font(12, True), MUTED))
        lev_now = fr["level"]["level"] if fr["level"] is not None and fr["pstep"] == "refine" else None
        for li, L in enumerate(rmv_levels()):
            yy = ty + 18 + li * 19
            on = li == lev_now
            if on:
                cv2.rectangle(canvas, (tx - 6, yy - 2), (cx[-1] + 80, yy + 16), ACTIVE_FILL, -1)
            vals = (str(li), f"{L['s']:g}", f"{L['beta'] * 100:g} cm", str(L["K"]), f"{L['mu']:g}")
            for v, x in zip(vals, cx):
                texts.append(((x, yy), v, font(13, on), TEXT if on else MUTED))
        if fr["pstep"] == "refine" and fr["explore"] is not None:
            e = fr["explore"]
            txt = (f"explore ({e['solver'].upper()}): iteration {e['it'] + 1}/{e['n_it']}, {e['n_samples']} sampled "
                   f"paths per seed")
        elif fr["pstep"] == "refine" and fr["exploit"] is not None:
            txt = f"exploit ({fr['exploit']['solver'].upper()}): {fr['exploit']['iters']} gradient iterations per seed"
        elif fr["pstep"] == "refine" and fr["select"] is not None:
            txt = "keep the better half of the seeds, refine their splines exactly to the next K"
        elif fr["pstep"] == "seeds":
            ns = 0 if fr["seeds"] is None else fr["seeds"]["paths"].shape[0]
            txt = (f"seeds: the straight line + {ns - 1} A* corridors on the coarse grid, fitted with K=4" if ns
                   else "A* on a 2 cm grid of the inflated obstacles; each corridor penalises the next")
        elif fr["pstep"] == "straight":
            txt = "fast path: check the straight line before any search"
        elif fr["pstep"] == "ends":
            c = fr["cands"]
            txt = f"end poses: {len(c['kept'])} of {len(c['items'])} kept (collision-free, reachable)"
        elif fr["pstep"] == "certify":
            txt = "certificate: every safety / require term within tolerance on 128 samples, then the arm check"
        elif fr["pstep"] == "result":
            txt = "result from the run's own log"
        else:
            txt = "request received"
        texts.append(((tx, ty + 18 + 4 * 19 + 6), fit_text(txt, font(13), cx[-1] + 80 - tx), font(13), YELLOW))
        # cost plot: best J per seed (log scale) over explore / exploit steps
        px0, py0, px1, py1 = cx[-1] + 104, ty, x1 - 14, y1 - 26
        cv2.rectangle(canvas, (px0, py0), (px1, py1), (16, 23, 36), -1)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), EDGE, 1)
        texts.append(((px0 + 6, py0 + 4), "cost J of each seed's best (log)", font(12, True), MUTED))
        pts = fr["J"]
        if pts:
            gmax = max(10, max(p[0] for p in pts) if fr["pstep"] != "result" else max(p[0] for p in pts))
            allj = [v for p in pts for v in p[2].values() if v > 0]
            lo = math.log10(max(min(allj) if allj else 1e-3, 1e-3))
            hi = math.log10(max(max(allj) if allj else 1.0, 1e-2))
            if hi - lo < 1:
                hi = lo + 1
            X = lambda g: px0 + 8 + (px1 - px0 - 16) * (g - 1) / max(1, gmax - 1)
            Y = lambda v: py1 - 8 - (py1 - py0 - 30) * (math.log10(max(v, 1e-3)) - lo) / (hi - lo)
            series = {}
            for g, lev, d in pts:
                for sid, v in d.items():
                    series.setdefault(sid, []).append((X(g), Y(v)))
            lev_starts = {}
            for g, lev, d in pts:
                lev_starts.setdefault(lev, g)
            for lev, g in lev_starts.items():
                xg = int(X(g))
                cv2.line(canvas, (xg, py0 + 20), (xg, py1), EDGE, 1)
                texts.append(((xg + 3, py1 - 18), f"L{lev}", font(11), MUTED))
            for sid, s in series.items():
                if len(s) >= 2:
                    cv2.polylines(canvas, [np.round(np.array(s)).astype(np.int32)], False,
                                  SEED_COLS[sid % len(SEED_COLS)], 2, cv2.LINE_AA)
                cv2.circle(canvas, (int(s[-1][0]), int(s[-1][1])), 3, SEED_COLS[sid % len(SEED_COLS)], -1)
        else:
            texts.append(((px0 + 8, py0 + 30), "no search for this request" if fr["pstep"] == "result" else
                          "(search not started)", font(12), MUTED))

    def _cert_panel(self, canvas, texts, fr):
        R = self.reqs[fr["ri"]]
        x0, y0, x1, y1 = CERT_PANEL
        L = R["logged"]
        if fr["result"]:
            rows, title = L["report"] or [], "CERTIFICATE (the run's)"
            ok = L["result"] == "certified"
        elif fr["cert"] is not None:
            rows, ok = fr["cert"]["rows"], fr["cert"]["ok"]
            title = "CERTIFICATE ATTEMPT (" + ("straight line" if fr["cert_phase"] == "fast_path" else "searched path") + ")"
        else:
            rows, ok, title = [], None, "CERTIFICATE"
        texts.append(((x0 + 12, y0 + 8), title, font(16, True), CYAN))
        if fr["result"]:
            res = result_text(L["result"], L["stage"])
            col = RESULT_COL.get(L["result"], TEXT)
            texts.append(((x0 + 12 + int(font(16, True).getlength(title)) + 14, y0 + 9), res.upper(),
                          font(15, True), col))
        why = None
        if fr["result"] and L["result"] != "certified":
            parts = []
            if R.get("_reported") is not None:
                parts.append("best attempt rejected by the " + reason_text(R["_reported"]))
            if L.get("budget_exhausted"):
                parts.append(f"the {R['budget_ms']:.0f} ms budget ran out")
            why = "; ".join(parts) or None
        elif fr["cert"] is not None and not fr["checking"]:
            verdict = "ACCEPTED" if ok else "REJECTED: " + reason_text(fr["cert"])
            texts.append(((x0 + 12 + int(font(16, True).getlength(title)) + 14, y0 + 9), fit_text(verdict, font(14, True), 380),
                          font(14, True), GREEN if ok else RED))
        if not rows:
            msg = "waiting for a path to check"
            fc = fr.get("fast")
            if fr["pstep"] in ("seeds", "refine", "certify") and fc is not None and not fc["ok"]:
                bad = [r for r in fc["rows"] if r["role"] != "prefer" and r["max_violation"] > 1.0]
                w = max(bad, key=lambda r: r["max_violation"]) if bad else None
                msg = ("The straight line was rejected" + (f" (worst: {w['term']}, {w['max_violation']:.1f}x its tolerance)" if w else f" ({fc['reason']})") + ". TRACE searches; the searched path is checked here when the search ends.")
            if fr["result"]:
                bits = []
                if L.get("quote"):
                    bits.append("conflict: " + "; ".join(f"{q.get('term')} needs {q.get('needed')} {q.get('unit') or ''}"
                                                         f" (has {q.get('value')})" for q in L["quote"][:2]))
                if L.get("look_at"):
                    la = L["look_at"]
                    bits.append(f"look closer at {la.get('entity')}: uncertainty {la.get('uncertainty')} > tolerance "
                                f"{la.get('tolerance')}")
                if L["stage"] == "terminal" and not bits:
                    bits.append("every end pose collides or cannot be reached, so no path is planned")
                if why:
                    bits.append(why)
                msg = " | ".join(bits) or msg
            for j, line in enumerate(wrap_px(msg, font(14), x1 - x0 - 24, 4)):
                texts.append(((x0 + 12, y0 + 40 + 20 * j), line, font(14), MUTED))
            return
        top = y0 + 38
        if why:
            texts.append(((x0 + 12, y0 + 34), fit_text(why[0].upper() + why[1:], font(14, True), x1 - x0 - 24),
                          font(14, True), RED))
            top = y0 + 60
        rows = sorted(rows, key=lambda r: -float(r.get("max_violation") or 0.0))[:7 if not why else 6]
        bx0, bx1 = x0 + 250, x1 - 70
        for j, r in enumerate(rows):
            yy = top + j * 24
            v = float(r.get("max_violation") or 0.0)
            col = ROLE_COL.get(r.get("role"), MUTED)
            texts.append(((x0 + 12, yy), fit_text(f"{r.get('term')} ({r.get('role')})", font(13), 230), font(13), col))
            frac = min(v, 3.0) / 3.0
            cv2.rectangle(canvas, (bx0, yy + 3), (bx1, yy + 15), (16, 23, 36), -1)
            bad = v > 1.0 and r.get("role") != "prefer"
            cv2.rectangle(canvas, (bx0, yy + 3), (int(bx0 + frac * (bx1 - bx0)), yy + 15),
                          RED if bad else (GREEN if v <= 1.0 else MUTED), -1)
            x1v = int(bx0 + (bx1 - bx0) / 3.0)
            cv2.line(canvas, (x1v, yy), (x1v, yy + 18), TEXT, 1)
            texts.append(((bx1 + 8, yy), f"{v:.2f}" if v < 100 else f"{v:.0f}", font(13, True), RED if bad else TEXT))
        texts.append(((bx0, y1 - 22), "max violation / tolerance (white line = 1: at the edge of acceptable; "
                      "prefer terms never block)", font(11), MUTED))
        arm = (fr["cert"] or {}).get("arm") if not fr["result"] else None
        if arm:
            texts.append(((x0 + 12, y1 - 22), f"arm check: {'ok' if arm.get('ok') else arm.get('reason')}",
                          font(12, True), GREEN if arm.get("ok") else RED))

    def _strip(self, canvas, texts, fr):
        x0, y0, x1, y1 = STRIP
        n = len(self.reqs)
        w = (x1 - x0) / n
        for j, R in enumerate(self.reqs):
            a, b = int(x0 + j * w) + 1, int(x0 + (j + 1) * w) - 1
            L = R["logged"]
            col = RESULT_COL.get(L["result"], MUTED)
            if L["result"] == "certified" and L["stage"] == "search":
                col = CYAN
            fill = tuple(int(c * (0.85 if j == fr["ri"] else 0.35)) for c in col)
            cv2.rectangle(canvas, (a, y0), (b, y1), fill, -1)
            if j == fr["ri"]:
                cv2.rectangle(canvas, (a - 1, y0 - 3), (b + 1, y1 + 3), TEXT, 2)
            if w >= 26:
                lab = str(j + 1)
                texts.append(((a + (b - a - font(11).getlength(lab)) / 2, y0 + 14), lab, font(11, j == fr["ri"]),
                              BG if j == fr["ri"] else TEXT))

    def _status_line(self, texts, fr):
        R = self.reqs[fr["ri"]]
        L, P = R["logged"], R["replay"]
        x0, y0, x1, y1 = STATUS
        stepname = {"request": "REQUEST", "ends": "END POSES", "straight": "STRAIGHT LINE", "seeds": "SEEDS",
                    "refine": "REFINE", "certify": "CERTIFY", "result": "RESULT"}[fr["pstep"]]
        if fr["pstep"] == "refine" and fr["level"] is not None:
            stepname += f" level {fr['level']['level']}"
        head = f"REQUEST {fr['ri'] + 1}/{len(self.reqs)}  |  {R['id']}  |  {stepname}  |  "
        ran = f"in the run: {result_text(L['result'], L['stage'])}, {L['solve_ms']:.0f} ms"
        if L["stage"] == "search" and L.get("levels_run") is not None:
            ran += f", {L['levels_run']} levels, {int(L.get('evals') or 0):,} cost evaluations"
        fb = font(17, True)
        texts.append(((x0 + 16, y0 + 6), head, fb, YELLOW if fr["pstep"] != "result" else TEXT))
        texts.append(((x0 + 16 + int(fb.getlength(head)), y0 + 7), fit_text(ran, font(16), x1 - x0 - 40 - int(fb.getlength(head))),
                      font(16), RESULT_COL.get(L["result"], TEXT)))
        same = L["result"] == P["result"]
        note = ("replay of the search (saved snapshot, logged arm answers) ends the same way" if same else
                f"replay ends {P['result']} ({P['stage']}), the run {L['result']}: the replay checks clearance on the "
                "saved voxel grid, the run on the live map; the search shown is the replay's")
        texts.append(((x0 + 16, y0 + 31), fit_text(note, font(13), x1 - x0 - 32), font(13), MUTED if same else ORANGE))

    # ------------------------------------------------------------------ frame
    def _make_chrome(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((24, 12), "TRACE MOTION PLANNING  |  HOW EACH MOVE IS PLANNED", font=font(27, True), fill=TEXT)
        instr = self.meta.get("instruction") or (self.run["summary"] or {}).get("language") or ""
        d.text((24, 75), fit_text(instr, font(20), W - 48), font=font(20), fill=TEXT)
        d.rounded_rectangle(rmv.MAIN, radius=12, fill=PANEL)
        d.text((rmv.MAIN[0] + 16, rmv.MAIN[1] + 8), "3D  |  TRACE's scene: S2's voxel map at the request",
               font=font(17, True), fill=CYAN)
        for box in (REQ_PANEL, PIPE_PANEL, CERT_PANEL, STATUS):
            d.rounded_rectangle(box, radius=12, fill=PANEL)
        f = font(13)
        d.text((24, FOOTER_Y), "Lavender = TRACE's collision spheres at the start (gripper, held object, wrist); green / "
               "red ghosts = end poses kept / dropped; dashed = straight line; cyan = A* corridors; colours = seeds "
               "(faint: sampled paths, bold: each seed's best, dots: control poses).", font=f, fill=MUTED)
        d.text((24, FOOTER_Y + 19), "Green path = the path the run executed (TRACE's certified output), with the "
               "gripper swept along it; red X = worst requirement of a rejected path. Strip: one chip per request "
               "(green: certified on the straight line, cyan: certified after search, red: no path, orange: conflict / "
               "look closer).", font=f, fill=MUTED)
        return np.asarray(img).copy()

    def _subtitle_txt(self, fr):
        R = self.reqs[fr["ri"]]
        a = self.meta
        return (f"build {str(a.get('build', '?'))[:16]}  |  {a.get('task', '?')} ep{int(a.get('episode', 0)):03d}  |  "
                f"request {R['id']} at t {R['t']:.1f} s (step {R['step']})")

    def render(self, i):
        fr = self.frames[i]
        lab = self.labels_for(i)
        inp = self.inputs(i)
        k = inp["k"]
        canvas = self._chrome.copy()
        texts = []
        base, dep, _, _, _ = inp["layer"]
        view = base.copy()
        self._splat_points(view, dep, self.robot.get("robot", k), inp["payload"])
        boxes = []
        self._overlay(view, dep, fr, boxes)
        vx, vy = rmv.VIEW[0], rmv.VIEW[1]
        canvas[vy:vy + view.shape[0], vx:vx + view.shape[1]] = view
        self._draw_tags(canvas, texts, lab["labels"])
        self._triad(canvas, texts)
        self._req_panel(canvas, texts, fr)
        self._pipe_panel(canvas, texts, fr)
        self._cert_panel(canvas, texts, fr)
        self._strip(canvas, texts, fr)
        self._status_line(texts, fr)
        pil = Image.fromarray(canvas)
        d = ImageDraw.Draw(pil)
        d.text((24, 48), fit_text(self._subtitle_txt(fr), font(18), W - 48), font=font(18), fill=MUTED)
        for item in boxes:
            if item[0] == "box":
                d.rectangle(item[1], fill=item[2])
            else:
                d.text(item[0], item[1], font=item[2], fill=item[3])
        for xy, txt, fnt, fill in texts:
            d.text(xy, txt, font=fnt, fill=fill)
        if fr["pstep"] == "refine" and fr["level"] is not None and fr.get("note") == "level":
            L = fr["level"]
            msg = (f"LEVEL {L['level']}: tolerances x{L['s']:g}, scene blurred {L['beta'] * 100:g} cm, "
                   f"K = {L['K']} control poses, penalty {L['mu']:g}")
            f = font(20, True)
            tw = f.getlength(msg)
            cx = (rmv.MAIN[0] + rmv.MAIN[2]) / 2
            d.rounded_rectangle((cx - tw / 2 - 16, vy + 16, cx + tw / 2 + 16, vy + 54), radius=10, fill=ACTIVE_FILL,
                                outline=CYAN, width=2)
            d.text((cx - tw / 2, vy + 24), msg, font=f, fill=TEXT)
        pill(d, rmv.MAIN[2] - 14, rmv.MAIN[1] + 7, "PLANNING (simulator paused)", YELLOW, font(14, True))
        return np.asarray(pil)


def rmv_levels():
    from traceplan.objective import LEVELS
    return LEVELS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--png-dir", default=None)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    t0 = time.time()
    run_dir = Path(args.run_dir)
    vid = TraceVideo(run_dir)
    n = len(vid.frames)
    if args.frames:
        png = Path(args.png_dir)
        png.mkdir(parents=True, exist_ok=True)
        for i in rmv.parse_frames(args.frames, n):
            Image.fromarray(vid.render(i)).save(png / f"trace_{i:04d}.png")
        print(f"render_trace_video: PNGs -> {png} ({time.time() - t0:.1f} s)")
        return 0
    out = Path(args.out) if args.out else run_dir / VIDEO_NAME
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            proc.stdin.write(np.ascontiguousarray(vid.render(i), np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        if tmp.exists():
            tmp.unlink()
        raise
    if rc != 0:
        print(f"render_trace_video: ffmpeg failed (exit {rc})")
        return 1
    os.replace(tmp, out)
    print(f"render_trace_video: {n} frames -> {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
