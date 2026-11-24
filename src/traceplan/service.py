"""TraceService.plan: the contract's single entry point (DESIGN §2.10, TRACE integration.md).

Flow: lint -> compile (start/end clearances, auto standoffs, fusion) -> program tests -> terminal candidates ->
straight-line fast path -> seeds and coupled continuation within budget_ms -> certificate (dense exact check, the
harness's arm check, retiming) -> diagnosis of the best failure (conflict quote, look, no path). Seeded by the
request and logged per call, so replay is exact at no model cost.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .certify import certify
from .contract import GROUPS, TraceResponse, to_jsonable
from .diagnose import auto_standoff, conflict, end_window_term, look_rule, window_flag
from .objective import LEVELS, Objective
from .program import Program, fuse, lint, run_program_tests
from .scene import field_distance
from .search import CFG as SEARCH
from .search import Cand, continuation, end_test, free_end_candidates, path_seeds, score_candidates
from .spline import SE3Spline

CFG = {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "n_dense": 128, "z": 1.64, "search": {}}
FAST_PATH_ENDS = 4          # straight lines certified per request (the kept ends, best first)


class TraceService:
    def __init__(self, cfg=None, log_dir=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.scfg = {**SEARCH, **self.cfg["search"]}
        self.device = torch.device(self.cfg["device"])
        self.log_dir = None if log_dir is None else Path(log_dir)

    def plan(self, request, snapshot, arm_check, ik_check=None):
        t0 = time.perf_counter()
        resp = _Plan(self, request, snapshot, arm_check, ik_check, t0).run()
        resp.stats["solve_ms"] = (time.perf_counter() - t0) * 1e3
        self._log(request, snapshot, resp)
        return resp

    def _log(self, req, snap, resp):
        if self.log_dir is None:
            return
        (self.log_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        f = self.log_dir / "snapshots" / f"v{snap.version}.npz"
        if not f.exists():
            snap.save(f)
        rec = {"time": time.time(),
               "request": {"request_id": req.request_id, "scene_version": req.scene_version, "seed": req.seed,
                           "mode": req.mode, "budget_ms": req.budget_ms, "n_ends": len(req.ends),
                           "n_terms": len(req.terms), "holding": req.holding, "start_p": req.start_p,
                           "tag": req.meta.get("tag")},
               "response": {"result": resp.result, "end_index": resp.end_index, "standoff": resp.standoff,
                            "tube_mm": (resp.valid_while or {}).get("tube_mm"), "quote": resp.quote,
                            "look_at": resp.look_at, "findings": resp.findings, "stats": resp.stats}}
        with open(self.log_dir / "trace.jsonl", "a") as fh:
            fh.write(json.dumps(to_jsonable(rec)) + "\n")
        with open(self.log_dir / f"req_{req.request_id}.pkl", "wb") as fh:
            pickle.dump((replace(req, meta={k: v for k, v in req.meta.items() if not callable(v)}), resp), fh)


def _clean(rows):
    """Report rows for the response: the worst sphere position stays (as "at"), so the caller can name what blocks."""
    out = []
    for r in rows:
        row = {k: v for k, v in r.items() if k != "at_x"}
        if r.get("at_x") is not None:
            row["at"] = np.round(np.asarray(r["at_x"], float), 4).tolist()
        out.append(row)
    return out


def _jsonable_test(d):
    return {k: (np.round(np.asarray(v, float), 4).tolist() if k == "at" else v) for k, v in d.items()}


class _Plan:
    """State of one plan call."""

    def __init__(self, svc, req, snap, arm_check, ik_check, t0):
        self.svc, self.req, self.snap, self.arm, self.ik = svc, req, snap, arm_check, ik_check
        self.cfg, self.dev, self.n_dense = svc.scfg, svc.device, svc.cfg["n_dense"]
        self.deadline = t0 + req.budget_ms / 1e3
        np.random.seed(req.seed % 2 ** 32)
        torch.manual_seed(req.seed)
        self.rng = np.random.default_rng(req.seed)
        self.gen = torch.Generator().manual_seed(req.seed)       # CPU draws: identical noise on any device
        self.stats = {"stage": None, "certify_ms": 0.0, "evals": 0, "seeds": 0, "levels_run": 0,
                      "candidates_tried": 0, "budget_exhausted": False}
        self.dropped = []                                         # ends that left before any search, and why
        self.start = {"R": np.asarray(req.start_R, float), "p": np.asarray(req.start_p, float)}
        self.levels = [dict(L, K=k) for L, k in zip(LEVELS, self.cfg["k_schedule"])]
        self.allow = req.meta.get("allow_contact", [])

    def resp(self, result, **kw):
        return TraceResponse(self.req.request_id, result, stats=self.stats, **kw)

    # ------------------------------------------------------------------ compile
    def group_clear(self, t, pose):
        b = self.req.bodies
        sel = np.ones(len(b.radii), bool) if t.args["body"] == "all" else b.groups == GROUPS.index(t.args["body"])
        if not sel.any():
            return None
        x = b.centers[sel] @ pose["R"].T + pose["p"]
        d = field_distance(self.snap, t.args.get("field"), x) if t.prim == "clearance" else x[:, 2] - t.args["z"]
        return float((d - b.radii[sel]).min())

    def program(self, terms, end):
        """Start/end clearances on the clearance terms (the zone rule), then fusion."""
        out = []
        for t in terms:
            if t.prim in ("clearance", "plane_clearance"):
                a = dict(t.args)
                for key, pose in (("c_start", self.start), ("c_end", end)):
                    c = None if pose is None else self.group_clear(t, pose)
                    if c is not None:
                        a[key] = c
                t = replace(t, args=a)
            out.append(t)
        return fuse(Program(out, self.req.holding, self.allow), self.snap, self.svc.cfg["z"])

    def candidate(self, i, e):
        a = None if e.approach is None else np.asarray(e.approach, float) / np.linalg.norm(e.approach)
        s, flag, extra = 0.0, None, []
        slide = tuple(e.standoff) if isinstance(e.standoff, (tuple, list)) else None
        if a is not None:
            auto = e.standoff == "auto" or slide is not None
            s = auto_standoff(a, e.entity, self.snap, self.req.margins) if auto else float(e.standoff)
            if slide is not None:
                s = max(s, float(slide[0]))
            if e.window is not None:
                flag = window_flag(a, e.entity, e.window, self.snap, self.req.margins)
                extra = [end_window_term(a, e.window, e.entity)]
        R, p = np.asarray(e.R, float), np.asarray(e.p, float) - (0.0 if a is None else s * a)
        q = None if e.q is None else np.asarray(e.q, float)
        test, ik_done = None, False
        if slide is not None and a is not None:
            s, p, q, test, ik_done = self.slide(e, R, a, s, float(slide[1]), q)
        c = Cand(i, R, p, q, a, s, e.entity, e.window, e, flag=flag)
        c.test, c.ik_done = test, ik_done
        c.program, c.flags = self.program(list(self.req.terms) + extra, c.end())
        return c

    def slide(self, e, R, a, s0, s_max, q):
        """The hand-off sits where certification ends: from the contact pose back along the approach, the first
        standoff (1 cm steps) whose end pose passes the end test and has an arm posture. Returns (s, p, q, test,
        ik_done); when none passes, the last test (the caller's terminal screen drops the end with its reason)."""
        held = (self.req.holding,) if self.req.holding else ()
        test, last, ik_tries, next_ik = None, None, 0, -np.inf
        for s in np.arange(s0, s_max + 1e-9, 0.01):
            p = np.asarray(e.p, float) - s * a
            test = end_test(self.snap, self.req.bodies, R, p, held, e.entity, contact=True)
            if last is None or not last[3].get("no_posture"):
                last = (float(s), p, q, test, False)
            if not test["ok"] or s < next_ik - 1e-9:
                continue
            if self.ik is None:
                return float(s), p, q, test, False
            ok, q_s = self.ik(R, p, q)
            if ok:
                return float(s), p, q_s, test, True
            # no posture here: two more tries, 4 cm further back each (IK is the costly part of the slide)
            last = (float(s), p, q, dict(test, ok=False, no_posture=True), False)
            ik_tries += 1
            if ik_tries >= 3:
                break
            next_ik = s + 0.04
        return last

    # ------------------------------------------------------------------ helpers
    def objective(self, c, K):
        sp = SE3Spline(self.start["R"], self.start["p"], c.R, c.p, K, pin_end=True, device=self.dev)
        return Objective(c.program, self.snap, self.req.bodies, sp, self.start, c.end(), self.dev,
                         self.cfg["n_eval"], self.req.margins, self.cfg["hard"], self.levels, flags=c.flags), sp

    def certify(self, obj, th, c):
        t = time.perf_counter()
        out = certify(obj, th, self.req, c, self.arm, self.n_dense)
        self.stats["certify_ms"] += (time.perf_counter() - t) * 1e3
        arm = out[1].get("arm")
        if not out[0] and arm is not None and not arm.get("ok"):
            self.stats["arm"] = {k: v for k, v in arm.items() if k != "ok"} | {"end": c.index}
        return out

    def certified(self, c, sp, th, cert, stage):
        self.stats["stage"] = stage
        if c.closed:
            self.stats["closed_by_contact"] = c.closed
        if getattr(c, "unseen", None):
            self.stats["end_unseen"] = _jsonable_test(c.unseen)
        tm = cert["timed"]
        return self.resp("certified", timed=tm, spline=sp.to_dict(th), end_index=c.index, end_R=c.R, end_p=c.p,
                         standoff=c.standoff, report=_clean(cert["report"]),
                         valid_while={"scene_version": self.snap.version, "tube_mm": float(tm.tube.max() * 1e3),
                                      "holding": self.req.holding, "start_q": np.asarray(self.req.start_q, float)})

    def provided(self, c):
        if c.src is None or c.src.p_path is None:
            return None, None
        sp = SE3Spline(self.start["R"], self.start["p"], c.R, c.p, self.levels[-1]["K"], device=self.dev)
        return sp, sp.theta_from_path(c.src.p_path, c.src.R_path)

    # ------------------------------------------------------------------ flow
    def run(self):
        req, snap = self.req, self.snap
        found = lint(req.terms, snap)
        if found:
            return self.resp("program_error", findings=found)
        if req.ends:
            cands = [self.candidate(i, e) for i, e in enumerate(req.ends)]
            prog, R_t, p_t, a_t = cands[0].program, cands[0].R, cands[0].p, cands[0].approach
        else:
            fe = req.free_end
            ent = snap.entities[fe["entity"]]
            p_t = ent.keypoint("top_center" if fe["relation"] == "above" else "centroid") + [0, 0, fe.get("height", 0)]
            prog, _ = self.program(list(req.terms), None)
            R_t, a_t = self.start["R"], None
        found = run_program_tests(prog, snap, lambda: SE3Spline(self.start["R"], self.start["p"], R_t, p_t, 4, False,
                                                                device="cpu", dtype=torch.float64),
                                  req.bodies, self.rng, a_t)
        if found:
            return self.resp("program_error", findings=found)
        if not req.ends:
            cands = []
            for i, (R, p, q) in enumerate(free_end_candidates(req.free_end, prog, snap, req.bodies, self.start,
                                                              req.holding, self.ik, self.gen, self.cfg, self.dev)):
                c = Cand(i, R, p, q, None, 0.0, req.free_end["entity"], None, None)
                c.program, c.flags = self.program(list(req.terms), c.end())
                cands.append(c)
        for c in cands:      # at the perception floor a closer look cannot help: the lateral axis is closed by contact
            if c.flag and c.flag["uncertainty"] <= req.margins.get("look_floor", 0.0):
                c.closed, c.flag = c.flag, None
        flagged = [c for c in cands if c.flag]
        cands = [c for c in cands if not c.flag]
        if flagged and not cands:
            self.stats["stage"] = "terminal"
            return self.resp("not_certifiable", look_at=flagged[0].flag)
        keep = int(req.meta.get("keep_ends", self.cfg["G"]))
        cands = score_candidates(cands, snap, req.bodies, self.start, req.holding, self.ik, keep, self.dropped)
        self.stats["ends_dropped"] = [_jsonable_test(d) for d in self.dropped]
        if not cands:
            self.stats["stage"] = "terminal"
            return self.resp("no_path")

        best = None                                   # (cand, obj, spline, theta, rows) of the best failure
        for c in cands[:FAST_PATH_ENDS]:              # fast path: certify the straight line
            obj, sp = self.objective(c, self.levels[0]["K"])
            th = sp.straight_theta()
            ok, cert = self.certify(obj, th, c)
            self.stats["candidates_tried"] += 1
            if ok:
                return self.certified(c, sp, th, cert, "fast_path")
            best = best or (c, obj, sp, th, cert["report"])
        if req.mode == "straight":
            self.stats["stage"] = "fast_path"
            return self.diagnose(best, search=False)
        if req.mode == "v0":
            for c in cands:
                sp, th = self.provided(c)
                if sp is not None:
                    obj = Objective(c.program, snap, req.bodies, sp, self.start, c.end(), self.dev,
                                    self.cfg["n_eval"], req.margins, flags=c.flags)
                    ok, cert = self.certify(obj, th, c)
                    if ok:
                        return self.certified(c, sp, th, cert, "seed")
            self.stats["stage"] = "seed"
            return self.diagnose(best, search=False)

        searched = None
        warm = req.meta.get("warm_spline")
        # the search gets the whole budget: screening a set of hand-off ends (end test, slide, arm postures) and their
        # straight lines must not starve it (a set of ends is the point of the hand-off, CONTACT_STAGE_PLAN 4.2)
        self.deadline = time.perf_counter() + req.budget_ms / 1e3
        for k, c in enumerate(cands):
            if k and time.perf_counter() >= self.deadline:     # the first candidate certifies at least its seeds
                self.stats["budget_exhausted"] = True
                break
            obj, sp = self.objective(c, self.levels[0]["K"])
            sp_p, th_p = self.provided(c)
            if warm is not None and np.linalg.norm(np.asarray(warm["pg"]) - c.p) < 0.01:
                # closed-loop reuse: refit the previous solution from the current start, rerun the last level
                spw, thw = SE3Spline.from_dict(warm)
                Rw, pw = spw.sample_np(thw, np.linspace(0.0, 1.0, 64))
                sp = sp.like(K=self.levels[-1]["K"])
                seeds, kw = sp.theta_from_path(pw, Rw)[None], {"warm": True}
            else:
                seeds, kw = path_seeds(sp, c, snap, req.bodies, self.gen, self.cfg, self.deadline), {}
            prov = None if th_p is None else th_p[None]
            self.stats["seeds"] += len(seeds) + (0 if prov is None else 1)
            sp2, th, J, info = continuation(obj, sp, seeds, self.gen, self.cfg, self.deadline, provided=prov, **kw)
            self.stats["levels_run"] += info["levels_run"]
            if info["levels_run"] < (1 if kw else len(self.levels)):
                self.stats["budget_exhausted"] = True
            self.stats["evals"] += obj.evals
            self.stats["candidates_tried"] += 1
            for i in range(len(th)):                  # certify in order of J
                ok, cert = self.certify(obj, th[i], c)
                if ok:
                    return self.certified(c, sp2, th[i], cert, "search")
                searched = searched or (c, obj, sp2, th[i], cert["report"])
            if sp_p is not None:                      # then the unrefined provided seed path
                objp = Objective(c.program, snap, req.bodies, sp_p, self.start, c.end(), self.dev,
                                 self.cfg["n_eval"], req.margins, flags=c.flags)
                ok, cert = self.certify(objp, th_p, c)
                if ok:
                    return self.certified(c, sp_p, th_p, cert, "seed")
        self.stats["stage"] = "search"
        return self.diagnose(searched or best, search=True)

    def diagnose(self, best, search):
        """Conflict and look claims need a finished search: when the budget cut it, the answer is no_path."""
        c, obj, sp, th, rows = best
        if self.stats["budget_exhausted"]:
            return self.resp("no_path", report=_clean(rows), end_index=c.index)
        if search and any(r["role"] != "prefer" and r["max_violation"] > 1.0 for r in rows):
            quotes = conflict(obj, sp, th, rows, self.gen, self.cfg, self.n_dense)
            if quotes:
                return self.resp("conflict", quote=quotes, report=_clean(rows), end_index=c.index)
        look = look_rule(rows, self.snap, self.req.margins)
        if look is not None:
            return self.resp("not_certifiable", look_at=look, report=_clean(rows), end_index=c.index)
        return self.resp("no_path", report=_clean(rows), end_index=c.index)
