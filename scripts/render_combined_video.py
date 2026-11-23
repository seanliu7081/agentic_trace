#!/usr/bin/env python
"""Combined post-episode video: S2's 3D scene map with the planned and executed EEF paths, the head and wrist
cameras (planned path also drawn on the head view), and S1's inference: each Astra call with the image it was
given, its answer marked on that image, and its text.

    python scripts/render_combined_video.py RUN_DIR [--out PATH] [--frames I,J,K --png-dir DIR]

-> RUN_DIR/combined_video.mp4 (1920x1080, 10 fps, H.264). Inputs: what the episode recorded with --video (rec/ in
recorder.py's layout), events.jsonl, summary.json, model/ (the images of every S1 request) and rec/head_camera.json
(scripts/head_camera.py; without it the head view shows no path). Built on render_map_video.MapVideo (3D map, labels,
paths, status line, result banner) and vis_timeline (same frame list as the task and map videos: one frame per
recorded step = real time, plus hold frames while the simulator is paused for S1 inference or S2 planning).
Evaluator-side only: it reads what the episode recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_map_video as rmv  # noqa: E402
from vis_timeline import PERCEIVE_MOVES, S1_EVENTS, S1_KIND  # noqa: E402

VIDEO_NAME = "combined_video.mp4"
W, H, FPS = 1920, 1080, 10
RX0, RX1 = 1052, W - 24                       # right column (cameras + legend, S1 inference)
CAM = 320
CAM_PANEL = (RX0, 106, RX1, 466)
HEAD_X, CAM_Y = RX0 + 10, 138
WRIST_X = HEAD_X + CAM + 10
INF_PANEL = (RX0, 478, RX1, 870)
THUMB = 270
PHASE_BOX = (24, 940, RX1, 984)
FOOTER_Y = 996

# MapVideo reads its layout from module globals at call time: point them at this layout
rmv.W, rmv.H = W, H
rmv.RIGHT_X0, rmv.RIGHT_X1 = RX0, RX1
rmv.AGENT_PANEL = CAM_PANEL                   # legend column: top and right edge
rmv.INSET, rmv.INSET_X, rmv.INSET_Y = CAM, HEAD_X, CAM_Y
rmv.LEGEND_X0 = WRIST_X + CAM + 10
rmv.TD_PANEL = INF_PANEL                      # the top-down plot is not drawn here
rmv.STATUS = (24, 878, RX1, 930)
rmv.FOOTER_Y = FOOTER_Y

font, fit_text, pill, isnum = rmv.font, rmv.fit_text, rmv.pill, rmv.isnum
BG, PANEL, TEXT, MUTED = rmv.BG, rmv.PANEL, rmv.TEXT, rmv.MUTED
CYAN, YELLOW, GREEN, RED = rmv.CYAN, rmv.YELLOW, rmv.GREEN, rmv.RED
PLAN_YELLOW, PINK, PINK_DIM = rmv.PLAN_YELLOW, rmv.PINK, rmv.PINK_DIM
ACTIVE_FILL = (32, 91, 90)
EDGE = (44, 56, 74)
SHIFT = 3
PHASES = ("1  PERCEIVE  (RGB-D + SAM)", "2  S1 INFERENCE  (Astra)", "3  S2 PLAN  (TRACE)", "4  EXECUTE / MONITOR")
ASKED = {"program": "Write the program: name and point at the objects in IMAGE 1, list the stages.",
         "repair": "A stage failed: diagnose it and rewrite the rest of the program.",
         "verify": "Final check: is the whole task done?",
         "effect_check": "Did the stage have its intended effect?",
         "place_pose": "Where should the held object be put down? A pixel in IMAGE 1 and the long side's direction.",
         "place_look": "After the release: is the object placed as the stage requires?"}
LONG_DIR = {"left_right": (1.0, 0.0), "near_far": (0.0, 1.0), "diagonal_rising": (0.7071, -0.7071),
            "diagonal_falling": (0.7071, 0.7071)}


def wrap_px(text, fnt, width, max_lines):
    """Greedy pixel-width word wrap; the last kept line ends with '…' when text is cut."""
    words, lines, cur = str(text).split(), [], ""
    for i, w in enumerate(words):
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


def stage_text(s):
    if not isinstance(s, dict):
        return str(s)
    op, obj = str(s.get("op") or "").upper(), s.get("object")
    if s.get("op") == "transfer":
        t = f"TRANSFER {obj} {s.get('relation')} {s.get('reference')}"
        return t + (f"  (release: {s['release']})" if s.get("release") else "")
    if s.get("op") == "actuate":
        return f"ACTUATE {obj}: {s.get('motion')} until {s.get('goal')}"
    return f"{op} {obj}"


_describe = rmv.Narrator.describe


def describe(self, e, follow=False):
    """Status-line wording for this fork's events (TRACE moves, V2 placement); everything else as in the map video."""
    g = e.get if isinstance(e, dict) else (lambda k: None)
    ev = g("event")
    try:
        if ev == "trace_request":
            return f"TRACE planning the {self.move_phrase(g('tag'))}"
        if ev == "trace_result":
            return (f"TRACE {str(g('result')).replace('_', ' ')} ({str(g('stage') or '').replace('_', ' ')}, "
                    f"{g('solve_ms'):.0f} ms)" if isnum(g("solve_ms")) else f"TRACE {g('result')}")
        if ev == "trace_seed":
            return f"TRACE seed path ({g('how')}): " + ("found" if g("ok") else "none")
        if ev == "trace_reaction":
            return f"TRACE {str(g('result')).replace('_', ' ')}: {str(g('reaction')).replace('_', ' ')}"
        if ev == "exec_timed":
            info = g("info") or {}
            dur = info.get("duration") if isinstance(info, dict) else None
            return "executed the planned move: " + str(g("status")) + (f" ({dur:.1f} s)" if isnum(dur) else "")
        if ev == "v2_pose":
            a = g("answer") or {}
            return f"S1 PLACE_POSE: put it down at pixel ({a.get('x')}, {a.get('y')})"
        if ev == "v2_put":
            if g("status") == "released":
                return (f"put down: lowered {g('lowered_cm')} cm, " + ("stopped on contact" if g("ended_on_contact")
                                                                       else "no contact") + f", let go {g('let_go')}")
            return f"put down: {str(g('status')).replace('_', ' ')}"
        if ev == "v2_look":
            a = g("answer") or {}
            return "S1 PLACE_LOOK: " + ("placed as required" if a.get("done") else "not placed, put it again")
        if ev == "open":
            return f"gripper opened ({g('gap')})"
    except Exception:
        pass
    return _describe(self, e, follow)


rmv.Narrator.describe = describe


class CombinedVideo(rmv.MapVideo):
    def __init__(self, run_dir):
        super().__init__(run_dir)
        for c in self.narr.s1:                           # "S1 V2_POSE" -> "S1 PLACE_POSE" in the status line
            c["kind"] = S1_KIND.get(c["kind"], c["kind"])
        self.wrist = rmv.VideoFrames(self.rec_dir / "cam_wrist.mp4")
        self.head_cam = None
        f = self.rec_dir / "head_camera.json"
        if f.exists():
            c = json.loads(f.read_text())
            self.head_cam = (np.asarray(c["K"], float), np.asarray(c["T_wc"], float), float(c.get("size", 512)))
        self.calls = self._load_calls()
        self._thumbs = {}

    # ------------------------------------------------------------ S1 calls
    def _load_calls(self):
        out, model = [], Path(self.run["dir"]) / "model"
        for e in self.run.get("events") or []:
            ev = e.get("event")
            if ev not in S1_EVENTS:
                continue
            rec = e.get("record") if isinstance(e.get("record"), dict) else {}
            kind = rec.get("kind") or S1_KIND.get(ev, ev)
            lat = rec.get("latency_s")
            t1 = float(e.get("t", 0.0))
            t0 = t1 - float(lat) if isnum(lat) else t1
            n, img = rec.get("n"), None
            if isinstance(n, int) and model.is_dir():
                files = []
                for p in model.glob(f"{n:02d}_{kind}_img*.png"):
                    try:
                        files.append((int(p.stem.split("_img")[1].split("_")[0]), p))
                    except (IndexError, ValueError):
                        continue
                img = min(files)[1] if files else None          # first image of the request = IMAGE 1 (head)
            out.append(dict(ev=ev, kind=kind, n=n, t0=t0, t1=t1, lat=lat, img=img, e=e,
                            ok=str(rec.get("status", "ok")) == "ok"))
        return out

    def _answer(self, c):
        """Answer of one call: ([(text, colour)], [(mark kind, x, y, label)]) in IMAGE 1 pixels."""
        e, ev, L, M = c["e"], c["ev"], [], []

        def pt(x, y):
            return isnum(x) and isnum(y)
        if ev in ("program", "repair"):
            p = e.get("program") if isinstance(e.get("program"), dict) else {}
            if ev == "repair" and e.get("diagnosis"):
                L.append(("Diagnosis: " + str(e["diagnosis"]), TEXT))
            if ev == "program" and p.get("scene"):
                L.append(("Scene: " + str(p["scene"]), MUTED))
            for j, s in enumerate(p.get("stages") or [], 1):
                L.append((f"Stage {j}: " + stage_text(s), CYAN))
            if p.get("finish_condition"):
                L.append(("Done when: " + str(p["finish_condition"]), MUTED))
            for o in p.get("objects") or []:
                if isinstance(o, dict) and pt(o.get("x"), o.get("y")):
                    M.append(("obj", o["x"], o["y"], str(o.get("id"))))
            for s in p.get("stages") or []:
                if not isinstance(s, dict):
                    continue
                for kx, ky, what in (("place_x", "place_y", "place"), ("effect_x", "effect_y", "effect")):
                    if pt(s.get(kx), s.get(ky)):
                        M.append(("target", s[kx], s[ky], f"{s.get('id')}: {what}"))
        elif ev == "v2_pose":
            a = e.get("answer") if isinstance(e.get("answer"), dict) else {}
            L.append((f"Put it down at pixel ({a.get('x')}, {a.get('y')}); long side: {a.get('long_side')}", CYAN))
            if a.get("why"):
                L.append(("Why: " + str(a["why"]), TEXT))
            if pt(a.get("x"), a.get("y")):
                M.append(("put", a["x"], a["y"], a.get("long_side")))
        elif ev == "v2_look":
            a = e.get("answer") if isinstance(e.get("answer"), dict) else {}
            done = bool(a.get("done"))
            L.append(("Placed as required: " + ("YES" if done else "NO, pick it up and put it again"),
                      GREEN if done else RED))
            if a.get("evidence"):
                L.append(("Evidence: " + str(a["evidence"]), TEXT))
            if not done and pt(a.get("x"), a.get("y")):
                L.append((f"New point: ({a['x']}, {a['y']}); long side: {a.get('long_side')}", CYAN))
                M.append(("put", a["x"], a["y"], a.get("long_side")))
        elif ev == "verify":
            r = e.get("result") if isinstance(e.get("result"), dict) else {}
            comp = r.get("complete")
            L.append(("Task complete: " + ("YES" if comp else "NO"), GREEN if comp else RED))
            for s in r.get("stages") or []:
                if isinstance(s, dict):
                    ok = s.get("status") == "done"
                    L.append((f"{s.get('id')}: {s.get('status')}. {s.get('evidence', '')}", TEXT if ok else RED))
            if r.get("problem"):
                L.append(("Problem: " + str(r["problem"]), RED))
            if r.get("hint"):
                L.append(("Hint: " + str(r["hint"]), MUTED))
        else:
            body = {k: v for k, v in e.items() if k not in ("t", "step", "event", "record")}
            L.append((json.dumps(body, default=str)[:600], TEXT))
        return L, M

    def _thumb(self, ci, answered):
        key = (ci, answered)
        if key in self._thumbs:
            return self._thumbs[key]
        c = self.calls[ci]
        canvas = np.full((THUMB, THUMB, 3), (8, 12, 20), np.uint8)
        im = cv2.imread(str(c["img"]), cv2.IMREAD_COLOR) if c["img"] is not None else None
        if im is None:
            pil = Image.fromarray(canvas)
            ImageDraw.Draw(pil).text((70, THUMB // 2 - 8), "no image saved", font=font(15), fill=MUTED)
            out = np.asarray(pil).copy()
        else:
            h0, w0 = im.shape[:2]
            sc = THUMB / max(h0, w0)
            im = cv2.resize(im[:, :, ::-1], (int(round(w0 * sc)), int(round(h0 * sc))), interpolation=cv2.INTER_AREA)
            canvas[:im.shape[0], :im.shape[1]] = im
            pil = Image.fromarray(canvas)
            d = ImageDraw.Draw(pil)
            if answered:
                f = font(12, True)

                def tag(u, v, label, col):
                    tw = f.getlength(label)
                    x = u + 9 if u + 9 + tw + 6 < THUMB else u - 9 - tw - 6
                    d.rectangle((x, v - 8, x + tw + 6, v + 8), fill=BG)
                    d.text((x + 3, v - 7), label, font=f, fill=col)
                for kind, x, y, label in self._answer(c)[1]:
                    u, v = x * sc, y * sc
                    if kind == "obj":
                        d.ellipse((u - 5, v - 5, u + 5, v + 5), fill=CYAN, outline=BG, width=2)
                        tag(u, v, label, CYAN)
                    elif kind == "target":
                        d.polygon([(u, v - 8), (u + 8, v), (u, v + 8), (u - 8, v)], outline=GREEN, width=2)
                        tag(u, v, label, GREEN)
                    elif kind == "put":
                        d.ellipse((u - 11, v - 11, u + 11, v + 11), outline=YELLOW, width=3)
                        d.line((u - 17, v, u - 5, v), fill=YELLOW, width=2)
                        d.line((u + 5, v, u + 17, v), fill=YELLOW, width=2)
                        d.line((u, v - 17, u, v - 5), fill=YELLOW, width=2)
                        d.line((u, v + 5, u, v + 17), fill=YELLOW, width=2)
                        dx, dy = LONG_DIR.get(str(label), (0.0, 0.0))
                        if dx or dy:                              # the long side's direction: double arrow
                            a, b = (u - 34 * dx, v - 34 * dy), (u + 34 * dx, v + 34 * dy)
                            d.line((*a, *b), fill=YELLOW, width=2)
                            for (px, py), s in ((a, -1), (b, 1)):
                                nx, ny = -dy, dx
                                d.polygon([(px, py), (px - s * 9 * dx + 5 * nx, py - s * 9 * dy + 5 * ny),
                                           (px - s * 9 * dx - 5 * nx, py - s * 9 * dy - 5 * ny)], fill=YELLOW)
                        tag(u + 8, v + 14, "put here" + (f", long side {label}" if label not in (None, "any") else ""),
                            YELLOW)
            out = np.asarray(pil).copy()
        if len(self._thumbs) > 64:
            self._thumbs.clear()
        self._thumbs[key] = out
        return out

    def _inference(self, canvas, texts, item):
        x0, y0, x1, y1 = INF_PANEL
        t = float(item["t"])
        f17b, f15, f14, f13 = font(17, True), font(15), font(14), font(13)
        eff = self.meta.get("effort")
        texts.append(((x0 + 14, y0 + 9), "S1 INFERENCE", f17b, CYAN))
        texts.append(((x0 + 14 + int(f17b.getlength("S1 INFERENCE  ")), y0 + 11),
                      "Astra (System 1)" + (f", effort {eff}" if eff else "") + ": what it was asked, what it saw, "
                      "what it answered", f13, MUTED))
        started = [j for j, c in enumerate(self.calls) if c["t0"] <= t]
        flight = [j for j in started if self.calls[j]["t1"] > t]
        done = [j for j in started if self.calls[j]["t1"] <= t]
        cur = flight[-1] if flight else (max(done, key=lambda j: self.calls[j]["t1"]) if done else None)
        if cur is None:
            for j, line in enumerate(wrap_px("No S1 call yet. S1 writes the program once from the head and wrist "
                                             "images; with V2 placement it also points where to put the held object "
                                             "down and judges the result after each release.", f15, x1 - x0 - 28, 4)):
                texts.append(((x0 + 14, y0 + 48 + 22 * j), line, f15, MUTED))
            return
        c, in_flight = self.calls[cur], cur in flight
        head = f"CALL #{c['n'] if c['n'] is not None else cur + 1}  {str(c['kind']).upper()}"
        texts.append(((x0 + 14, y0 + 38), head, font(16, True), YELLOW if in_flight else TEXT))
        hx = x0 + 14 + int(font(16, True).getlength(head)) + 16
        if in_flight:
            wait = f"IN FLIGHT  {max(0.0, t - c['t0']):.1f} s" + (f" of {c['lat']:.1f} s" if isnum(c["lat"]) else "")
            texts.append(((hx, y0 + 40), wait + "  |  simulator paused", f14, YELLOW))
        else:
            info = (f"{c['lat']:.1f} s" if isnum(c["lat"]) else "") + ("  ok" if c["ok"] else "  FAILED") + \
                f"  |  answered at {c['t1']:.1f} s"
            texts.append(((hx, y0 + 40), info, f14, MUTED if c["ok"] else RED))
        # the image S1 was given (IMAGE 1, head camera) with its answer marked
        tx, ty = x0 + 14, y0 + 66
        th = self._thumb(cur, not in_flight)
        canvas[ty:ty + th.shape[0], tx:tx + th.shape[1]] = th
        cv2.rectangle(canvas, (tx - 1, ty - 1), (tx + THUMB, ty + THUMB), YELLOW if in_flight else EDGE, 1)
        texts.append(((tx, ty + THUMB + 4), "IMAGE 1 (head camera) as sent to S1", font(12), MUTED))
        # text column: the question, then the answer
        cx, cw = tx + THUMB + 16, x1 - (tx + THUMB + 16) - 14
        y, ylim = ty - 2, y1 - 40
        for line in wrap_px("Asked: " + ASKED.get(str(c["kind"]), str(c["kind"])), f13, cw, 2):
            texts.append(((cx, y), line, f13, MUTED))
            y += 17
        y += 6
        if in_flight:
            texts.append(((cx, y), "Waiting for the answer…", f15, YELLOW))
        else:
            for txt, col in self._answer(c)[0]:
                room = int((ylim - y) // 20)
                if room <= 0:
                    break
                for line in wrap_px(txt, f15, cw, room):
                    texts.append(((cx, y), line, f15, col))
                    y += 20
                y += 4
        # all calls so far
        toks = [(f"#{self.calls[j]['n'] if self.calls[j]['n'] is not None else j + 1} "
                 f"{str(self.calls[j]['kind']).upper()}" +
                 (f" {self.calls[j]['lat']:.1f} s" if isnum(self.calls[j]["lat"]) and j not in flight else
                  (" …" if j in flight else "")), j == cur) for j in started]
        total = sum(self.calls[j]["lat"] for j in done if isnum(self.calls[j]["lat"]))
        lead = f"S1 calls so far: {len(started)}, {total:.0f} s of inference  |  "
        fx = x0 + 14
        texts.append(((fx, y1 - 30), lead, f13, MUTED))
        fx += int(f13.getlength(lead))
        avail = x1 - 14 - fx
        while toks and f13.getlength("   ".join(s for s, _ in toks)) > avail:
            toks = toks[1:]
            if toks:
                toks[0] = ("… " + toks[0][0].lstrip("… "), toks[0][1])
        for s, is_cur in toks:
            texts.append(((fx, y1 - 30), s, f13, (YELLOW if cur in flight else CYAN) if is_cur else TEXT))
            fx += int(f13.getlength(s + "   "))

    # ------------------------------------------------------------ cameras
    def _project(self, P, scale):
        K, T, _ = self.head_cam
        pc = (np.asarray(P, float).reshape(-1, 3) - T[:3, 3]) @ T[:3, :3]
        z = pc[:, 2]
        ok = z > 0.05
        z = np.where(ok, z, 1.0)
        u = (K[0, 0] * pc[:, 0] / z + K[0, 2] - 0.5) * scale
        v = (K[1, 1] * pc[:, 1] / z + K[1, 2] - 0.5) * scale
        return np.stack([u, v], 1), ok

    def _head_overlay(self, img, k, ov):
        """Planned EEF path (yellow), executed path of this stage (pink), place / grasp targets, current EEF."""
        if self.head_cam is None:
            return img
        sc = CAM / self.head_cam[2]
        S = 1 << SHIFT

        def poly(P, col, th, alpha=1.0):
            uv, ok = self._project(P, sc)
            uv = uv[ok]
            if len(uv) < 2:
                return
            pts = np.round(uv * S).astype(np.int32)
            if alpha >= 1.0:
                cv2.polylines(img, [pts], False, col, th, cv2.LINE_AA, SHIFT)
            else:
                over = img.copy()
                cv2.polylines(over, [pts], False, col, th, cv2.LINE_AA, SHIFT)
                cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)
        if len(self.eef) and k >= 1:
            ks = int(np.clip(ov.get("k_stage", 0), 0, k))
            seg = self.eef[ks:k + 1]
            seg = seg[np.all(np.isfinite(seg), 1)]
            if len(seg) >= 2:
                poly(seg, PINK, 2)
        plan = ov.get("plan")
        if plan is not None and isinstance(plan.get("eef"), list) and len(plan["eef"]) >= 2:
            try:
                poly(np.asarray(plan["eef"], float), PLAN_YELLOW, 2, 1.0 if ov.get("plan_active") else 0.45)
            except (TypeError, ValueError):
                pass
        for P, kind in ((ov.get("grasp"), "grasp"), (ov.get("place"), "place")):
            if P is None:
                continue
            uv, ok = self._project(P, sc)
            if not ok[0]:
                continue
            u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
            if kind == "grasp":
                cv2.circle(img, (u, v), 7, GREEN, 2, cv2.LINE_AA)
            else:
                cv2.polylines(img, [np.array([[u, v - 8], [u + 8, v], [u, v + 8], [u - 8, v]], np.int32)], True, CYAN,
                              2, cv2.LINE_AA)
        if 0 <= k < len(self.eef) and np.all(np.isfinite(self.eef[k])):
            uv, ok = self._project(self.eef[k], sc)
            if ok[0]:
                c = (int(round(uv[0, 0])), int(round(uv[0, 1])))
                cv2.circle(img, c, 5, BG, -1, cv2.LINE_AA)
                cv2.circle(img, c, 4, TEXT, -1, cv2.LINE_AA)
        return img

    def _cams(self, canvas, texts, k, item, ov):
        y0 = CAM_PANEL[1] + 8
        f15b, f11 = font(15, True), font(11)
        for name, x, reader in (("HEAD CAMERA", HEAD_X, self.agent), ("WRIST CAMERA", WRIST_X, self.wrist)):
            texts.append(((x, y0), name, f15b, TEXT))
            img = reader.get(k)
            stale = img is not None and reader.stale
            if item["final"] or item["hold"] or stale:
                word, col = ("STREAM ENDED", RED) if stale else (("FINAL FRAME", MUTED) if item["final"] else
                                                                  ("SIM PAUSED", YELLOW))
                f = font(11, True)
                w = int(f.getlength(word)) + 14
                px0 = x + CAM - w
                cv2.rectangle(canvas, (px0, y0 + 1), (px0 + w, y0 + 18), col, -1)
                texts.append(((px0 + 7, y0 + 3), word, f, BG))
            else:
                lbl = f"frame {k}" + (f" / {len(self.tel) - 1}" if self.tel else "")
                texts.append(((x + CAM - int(f11.getlength(lbl)), y0 + 4), lbl, f11, MUTED))
            if img is None:
                rmv.blend_rect(canvas, x, CAM_Y, x + CAM, CAM_Y + CAM, (8, 12, 20), 1.0)
                texts.append(((x + 100, CAM_Y + CAM // 2 - 9), "no camera frame", font(15), MUTED))
                continue
            small = cv2.resize(np.ascontiguousarray(img), (CAM, CAM), interpolation=cv2.INTER_AREA)
            if name == "HEAD CAMERA":
                small = self._head_overlay(np.ascontiguousarray(small), k, ov)
            canvas[CAM_Y:CAM_Y + CAM, x:x + CAM] = (small * 0.45).astype(np.uint8) if stale else small

    # ------------------------------------------------------------ V2 place target
    def overlays(self, item):
        ov = super().overlays(item)
        if ov.get("place") is None:                      # V2: S1's pointed spot at the destination's local top
            t, stage_t = float(item["t"]), None
            for e in reversed(rmv.events_upto(self.run, t)):
                if e.get("event") == "stage_start":
                    stage_t = float(e.get("t", 0.0))
                    break
            for e in reversed(rmv.events_upto(self.run, t)):
                if stage_t is not None and float(e.get("t", 0.0)) < stage_t:
                    break
                if e.get("event") == "v2_pose" and e.get("xy") is not None and isnum(e.get("z_top")):
                    try:
                        ov["place"] = np.array([float(e["xy"][0]), float(e["xy"][1]), float(e["z_top"])])
                    except (TypeError, ValueError, IndexError):
                        pass
                    break
        return ov

    # ------------------------------------------------------------ chrome, phase bar, frame
    def _make_chrome(self):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((24, 12), "AGENTIC_TRACE V2  |  3D SCENE MAP, CAMERAS, PLANNED PATHS, S1 INFERENCE",
               font=font(27, True), fill=TEXT)
        instr = self.meta.get("instruction") or (self.run["summary"] or {}).get("language") or ""
        d.text((24, 75), fit_text(instr, font(20), W - 48), font=font(20), fill=TEXT)
        d.rounded_rectangle(rmv.MAIN, radius=12, fill=PANEL)
        d.text((rmv.MAIN[0] + 16, rmv.MAIN[1] + 8), "3D SCENE MAP  |  S2's voxel map: what the planner sees",
               font=font(18, True), fill=CYAN)
        for box in (CAM_PANEL, INF_PANEL, rmv.STATUS):
            d.rounded_rectangle(box, radius=12, fill=PANEL)
        d.text((24, FOOTER_Y), "3D map: S2's 1 cm voxel map from snapshots recorded during the run (planning obstacles "
               "= occupied + unknown space; the table is a plane). Yellow = planned EEF path, also drawn on the head "
               "camera; pink = executed path of the current stage.", font=font(14), fill=MUTED)
        d.text((24, FOOTER_Y + 22), "Real time while the robot moves (one frame per 0.1 s of simulation). SIM PAUSED "
               "frames = the simulator waits while S1 infers or S2 plans; long waits are compressed (2 frames per "
               "second of wait, at most 3 s).", font=font(14), fill=MUTED)
        return np.asarray(img).copy()

    def _phase(self, i, item, ov):
        if item["final"]:
            return None
        t = float(item["t"])
        if any(c["t0"] <= t < c["t1"] for c in self.calls):
            return 1
        if item["hold"]:
            lab = str(self.narr.state(i).get("hold_label") or item.get("label") or "")
            if "→" in lab:                               # "S1 X returned → <S2 work in the rest of the gap>"
                lab = lab.split("→", 1)[1].strip()
            if lab.startswith("S1"):
                return 1
            if lab.startswith(("S2 grounding", "S2 checking")):
                return 0
            return 2
        plan = ov.get("plan")
        if plan is not None and ov.get("plan_active") and str(plan.get("tag", "")) in PERCEIVE_MOVES:
            return 0
        return 3

    def _phase_bar(self, canvas, texts, i, item, ov):
        x0, y0, x1, y1 = PHASE_BOX
        n, gap = len(PHASES), 12
        w = (x1 - x0 - gap * (n - 1)) / n
        act = self._phase(i, item, ov)
        f = font(16, True)
        for j, lab in enumerate(PHASES):
            a, b = int(x0 + j * (w + gap)), int(x0 + j * (w + gap) + w)
            on = j == act
            cv2.rectangle(canvas, (a, y0), (b, y1), ACTIVE_FILL if on else PANEL, -1)
            cv2.rectangle(canvas, (a, y0), (b, y1), CYAN if on else EDGE, 2 if on else 1)
            texts.append(((a + 14, y0 + 12), lab, f, TEXT if on else MUTED))

    def render(self, i):
        lab = self.labels_for(i)
        inp = self.inputs(i)
        item, ov, mv, k = inp["item"], inp["ov"], inp["mv"], inp["k"]
        canvas = self._chrome.copy()
        texts = []
        base, dep, _, _, _ = inp["layer"]
        view = base.copy()
        robot = self.robot.get("robot", k)
        self._splat_points(view, dep, robot, inp["payload"])
        self._draw_paths_3d(view, ov, k)
        vx, vy = rmv.VIEW[0], rmv.VIEW[1]
        canvas[vy:vy + view.shape[0], vx:vx + view.shape[1]] = view
        banner = lab["banner"]
        self._draw_tags(canvas, texts, lab["labels"])
        if banner is None or banner[3] < rmv.VIEW[3] - 120:
            self._triad(canvas, texts)
        if mv is None:
            texts.append(((rmv.VIEW[0] + 20, rmv.VIEW[1] + 20), "no map snapshot recorded", font(20), MUTED))
        self._cams(canvas, texts, k, item, ov)
        self._legend(canvas, mv, k, float(item["t"]))
        self._inference(canvas, texts, item)
        self._phase_bar(canvas, texts, i, item, ov)
        self._status(texts, i, item, ov, mv)
        pil = Image.fromarray(canvas)
        d = ImageDraw.Draw(pil)
        d.text((24, 48), fit_text(self._subtitle(mv, item), font(18), W - 48), font=font(18), fill=MUTED)
        for xy, txt, fnt, fill in texts:
            d.text(xy, txt, font=fnt, fill=fill)
        if item["final"]:
            pill(d, rmv.MAIN[2] - 14, rmv.MAIN[1] + 7, "EPISODE END", MUTED, font(14, True))
            self._result_banner(d, banner)
        elif item["hold"]:
            pill(d, rmv.MAIN[2] - 14, rmv.MAIN[1] + 7, "SIM PAUSED", YELLOW, font(14, True))
        return np.asarray(pil)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None, help=f"output video (default RUN_DIR/{VIDEO_NAME})")
    ap.add_argument("--frames", default=None, help="comma-separated timeline indices: write PNGs only")
    ap.add_argument("--png-dir", default=None, help="directory for --frames PNGs")
    args = ap.parse_args()
    if args.frames and not args.png_dir:
        ap.error("--frames needs --png-dir")
    t0 = time.time()
    run_dir = Path(args.run_dir)
    vid = CombinedVideo(run_dir)
    n = len(vid.tl)
    if n == 0:
        print("render_combined_video: no recorded frames; nothing to render")
        return 0
    if args.frames:
        png = Path(args.png_dir)
        png.mkdir(parents=True, exist_ok=True)
        for i in rmv.parse_frames(args.frames, n):
            Image.fromarray(vid.render(i)).save(png / f"combined_{i:04d}.png")
        print(f"render_combined_video: PNGs -> {png} ({time.time() - t0:.1f} s)")
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
        print(f"render_combined_video: ffmpeg failed (exit {rc})")
        return 1
    os.replace(tmp, out)
    print(f"render_combined_video: {n} frames -> {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
