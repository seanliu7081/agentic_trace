"""Every contact-free segment through the TRACE contract (DESIGN §3.5-3.9).

TraceMotion builds one TraceRequest per segment from the measured state, the stage's terms (level table) and the
harness's safety terms, calls TRACE (imported lazily, so atrace works without it), and maps its four results to
the sequencer's reactions:
  certified       execute with the tracking tube, the per-step guard and the contact detector; a stop on a
                  certified path is a defect: logged, re-observe, re-request once;
  conflict        relax what the plan allowed in advance (no model call), else StageFailure("trace_conflict") so
                  one REPAIR call carries the quote;
  not_certifiable look closer at the entity and re-request (at most 2 looks per move);
  no_path         None: the caller's ladder tries its next candidate (a search failure, not a fact);
  program_error   drop the offending language terms (reported at the next REPAIR) and re-request.
The straight line is tried first (mode "straight"); only when it fails is CP-5's planner path computed and added
as a seed for the full request (mode "v0" or "trace").
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np
from traceplan.contract import TraceRequest

from .body import hand_spheres, make_arm_check, make_ik_check, payload_spheres
from .guard import StepGuard
from .levels import LADDER, LIMITS, LIMITS_HOLDING, ZONE, can_relax, compile_terms, relax, safety_terms
from .motion import densify
from .perception import mask_points
from .sim_iface import WRIST_CAMERA
from .snapshot import build_snapshot, entity_of

TRACE = {"max_looks": 2, "q_tol": 0.02, "seed_budget_s": 2.0, "carve_step": 0.15}


def _stage_index(term_id):
    return int(term_id.split("_")[0][4:])          # lang<i>_<prim>_<of>: index of the stage term it came from


class TraceMotion:
    def __init__(self, h, mode="trace", device="cuda:0", budget_ms=2000.0, log_dir=None, cfg=TRACE):
        self.h, self.mode, self.budget_ms, self.cfg = h, mode, float(budget_ms), cfg
        self.svc_cfg, self.log_dir = {"device": device}, log_dir
        self._svc = None
        self.n = 0                     # per-episode request counter, also the request's seed
        self.depth = 0                 # > 0 while a look reaction moves the arm: no nested looks
        self.true_scene_hook = None    # evaluator side (--true-scene): called with the request id only
        self.counts = {"results": Counter(), "reactions": Counter(), "stops": Counter(), "executions": 0}
        self.wall_ms, self.fast = [], 0
        self._hands = {}

    @property
    def svc(self):
        if self._svc is None:
            from traceplan.service import TraceService
            self._svc = TraceService(self.svc_cfg, log_dir=self.log_dir)
        return self._svc

    # ------------------------------------------------------------------------------------------- one segment
    def move(self, ends, tag, grip=None, finger=0.04, entity=None, approach=None, seed_goal=None, goal_near=False,
             retreat=False):
        """One segment to the first reachable end candidate. Returns the ExecResult of the executed certified path
        (done, dropped, or how it stopped after the one re-request), or None when nothing was certified. A retreat
        never looks or calls the model: anything but a certificate returns None (the caller's move_line)."""
        from .harness import StageFailure
        h = self.h
        looks, slow, retried, seeded = 0, False, False, False
        while True:
            h.carve_robot()
            h._clear_held_ghost()              # the held object's own volume (unlabelled map remains) is no obstacle
            snap = build_snapshot(h)
            req = self.request(snap, ends, tag, finger, entity, approach, slow, "straight")
            resp = self.plan(req, snap, finger)
            r = resp.result
            # no end survived its own check (lateral window, clearance, IK): seeds and search cannot change that
            terminal = (r == "not_certifiable" and resp.look_at.get("axis") == "lateral") or \
                resp.stats.get("stage") == "terminal"
            if r in ("conflict", "no_path", "not_certifiable") and not terminal:
                if not seeded:
                    self.add_seeds(ends, finger, seed_goal, goal_near)
                    seeded = True
                req = self.request(snap, ends, tag, finger, entity, approach, slow, self.mode)
                resp = self.plan(req, snap, finger)
                r = resp.result
            if r == "certified":
                if not self.valid(req, resp, snap):
                    self.react(req, "void_certificate")
                    continue
                res = self.execute(req, resp, snap, grip, finger)
                if res.status in ("done", "dropped") or retried:
                    return res
                self.react(req, "reobserve_rerequest", status=res.status)
                h.reobserve()
                retried, seeded = True, self.clear_seeds(ends)
                continue
            if r == "conflict":
                terms = {t.id: t for t in req.terms}
                if resp.quote and all(can_relax(q, terms[q["term"]]) for q in resp.quote):
                    slow |= self.relax(resp.quote, terms)
                    self.react(req, "relax", quote=resp.quote)
                    continue
                if not retreat:
                    self.react(req, "repair", quote=resp.quote)
                    raise StageFailure("trace_conflict", quote=resp.quote, segment=tag,
                                       terms=[{k: q.get(k) for k in ("term", "prim", "level", "relax_to")}
                                              for q in resp.quote])
            if r == "not_certifiable" and not retreat and looks < self.cfg["max_looks"] and self.look(snap, resp.look_at):
                looks += 1
                seeded = self.clear_seeds(ends)            # the arm moved and the map changed: seeds are stale
                self.react(req, "look", look_at=resp.look_at)
                continue
            if r == "program_error":
                self.drop(req, resp.findings)
                self.react(req, "drop_terms", findings=resp.findings)
                continue
            self.react(req, "fallback" if retreat else "next_candidate", result=r)
            h.last_move = {"tag": tag, "status": "no_path", "planner": "trace", "result": r,
                           "request": req.request_id}
            return None

    def request(self, snap, ends, tag, finger, entity, approach, slow, mode):
        h = self.h
        self.n += 1
        holding = entity_of(h, h.held) if h.held is not None else None
        phase = "carry" if holding is not None else "move"
        st = h.cur_stage
        terms = compile_terms((st.get("terms") or []) if st else [], h, phase, holding, approach, entity)
        terms += [relax(t) if slow and t.level == "strict" else t for t in safety_terms(h, phase, LADDER, ZONE, holding)]
        key = round(float(finger), 4)
        if key not in self._hands:
            self._hands[key] = hand_spheres(h.rm, finger)
        bodies = self._hands[key] + payload_spheres(h.checker.attached) if holding is not None else self._hands[key]
        Rs, ps = h.rb.eef_pose()
        return TraceRequest(f"{self.n:04d}_{tag}", snap.version, self.n, self.budget_ms, Rs, ps, h.rb.q()[:7].copy(),
                            float(finger), holding, ends, terms, bodies, LADDER,
                            LIMITS_HOLDING if holding is not None else LIMITS, mode=mode, zone=ZONE,
                            meta={"tag": tag, "stage": st["id"] if st else None, "phase": phase})

    def plan(self, req, snap, finger):
        h = self.h
        h.log("trace_request", id=req.request_id, tag=req.meta["tag"], n_ends=len(req.ends), mode=req.mode,
              scene_version=req.scene_version, n_terms=len(req.terms))
        arm, ik = make_arm_check(h, LADDER, ZONE, finger), make_ik_check(h, finger)
        calls = req.meta.setdefault("callbacks", [])    # the harness's answers, logged with the request for replay

        def arm_rec(R, p, s, q_ref):
            out = arm(R, p, s, q_ref)
            calls.append(("arm", np.asarray(p, np.float32), out))
            return out

        def ik_rec(R, p, q_seed=None):
            out = ik(R, p, q_seed)
            calls.append(("ik", np.asarray(p, float), out))
            return out
        t = time.time()
        resp = self.svc.plan(req, snap, arm_rec, ik_rec)
        ms = 1e3 * (time.time() - t)
        h._tick("plan", t)
        st = resp.stats
        self.counts["results"][resp.result] += 1
        self.wall_ms.append(ms)
        self.fast += resp.result == "certified" and st.get("stage") == "fast_path"
        h.log("trace_result", id=req.request_id, result=resp.result, stage=st.get("stage"), solve_ms=st.get("solve_ms"),
              wall_ms=round(ms, 1), tube_mm=(resp.valid_while or {}).get("tube_mm"), standoff=resp.standoff,
              end_index=resp.end_index, quote=resp.quote, look_at=resp.look_at, findings=resp.findings,
              closed_by_contact=st.get("closed_by_contact"))
        if self.true_scene_hook is not None:
            self.true_scene_hook(req.request_id)
        return resp

    # ------------------------------------------------------------------------------------------ execution
    def valid(self, req, resp, snap):
        """Certificate validity (DESIGN §3.7): the same held entity, start q within q_tol, and the same snapshot
        version, or no occupancy / unknown change inside the swept tube."""
        h = self.h
        vw = resp.valid_while
        holding = entity_of(h, h.held) if h.held is not None else None
        if vw["holding"] != holding or np.abs(np.asarray(vw["start_q"]) - h.rb.q()[:7]).max() > self.cfg["q_tol"]:
            return False
        now = build_snapshot(h)
        if vw["scene_version"] == now.version:
            return True
        from traceplan.scene import changed_inside
        T, b = resp.timed, req.bodies
        X = (T.p[:, None, :] + np.einsum("kij,sj->ksi", T.R, b.centers)).reshape(-1, 3)
        return not changed_inside(snap, now, X, float(T.tube.max() + b.radii.max()))

    def execute(self, req, resp, snap, grip, finger):
        h = self.h
        T, tag = resp.timed, req.meta["tag"]
        guard = StepGuard(h, snap, req, T, LADDER, ZONE)
        h._rec("on_plan", T.q, finger, tag, "trace")
        t = time.time()
        res = h.ex.run_timed(T, cm=h.cm, grip=grip, finger=finger, guard=guard, detector=h.detector.guard("contact"),
                             drop_check=h._drop_check if h.held is not None else None)
        h._tick("execute", t)
        self.counts["executions"] += 1
        h.log("exec_timed", request_id=req.request_id, status=res.status, steps=res.steps,
              first_step=res.info["first_step"], last_step=res.info["last_step"], info=res.info)
        h.last_move = {"tag": tag, "status": res.status, "planner": f"trace:{resp.stats.get('stage')}",
                       "request": req.request_id}
        if res.status not in ("done", "dropped"):
            self.defect(req, resp, res)            # before carving: the map version tells whether the scene moved
        if res.status == "done":
            n = len(T.q)
            h.carve_robot(densify(T.q[np.unique(np.r_[np.arange(0, n, 5), n - 1])], self.cfg["carve_step"]))
        else:
            h.carve_robot()                        # stopped part-way: the rest of the path was never occupied
        if res.status == "contact":
            W, att = h.checker.world_points(h.rb.q()[None, :7], finger)
            pts = W[0][h.checker.pg >= 1] if att is None else np.concatenate([W[0][h.checker.pg >= 1], att[0]])
            h.last_move.update(force_n=np.round(res.info.get("force", []), 1).tolist(),
                               contact_near=h.nearest_obstacle(pts))
        h._rec("on_map", "move:" + tag)
        return res

    def defect(self, req, resp, res):
        """A stop on a certified path, with one of three explanations (DESIGN §3.7)."""
        h, i = self.h, res.info
        expl = "scene_changed" if build_snapshot(h).version != resp.valid_while["scene_version"] else \
            "tracking_left_tube" if i["ratio_max"] > 1.0 else "geometry_disagreement"
        self.counts["stops"][res.status] += 1
        h.log("guard_defect", request_id=req.request_id, path_index=i.get("step"), kind=res.status,
              group=i.get("group"), clearance=i.get("clearance"), tracking_err=i.get("tracking_err"),
              tube=i.get("tube"), scene_version=resp.valid_while["scene_version"], explanation=expl,
              detector=h.detector.last)

    # -------------------------------------------------------------------------------------------- reactions
    def react(self, req, reaction, **kw):
        self.counts["reactions"][reaction] += 1
        self.h.log("trace_reaction", id=req.request_id, reaction=reaction, **kw)

    def relax(self, quote, terms):
        """Switch the quoted terms to the levels the plan allowed in advance; True when unknown space becomes slow."""
        slow, st = False, self.h.cur_stage
        for q in quote:
            t = terms[q["term"]]
            if t.src == "language":
                src = st["terms"][_stage_index(t.id)]
                src["level"], src["relax_to"] = t.relax_to, None
            else:
                slow = True
        return slow

    def drop(self, req, findings):
        """Interface repair: the language terms the program tests rejected leave the stage (reported at the next
        REPAIR). A finding on a harness term is a bug and raises."""
        terms = {t.id: t for t in req.terms}
        bad = {f["term"] for f in findings}
        if any(terms[b].src != "language" for b in bad):
            raise RuntimeError(f"TRACE program tests rejected harness terms: {findings}")
        st = self.h.cur_stage
        gone = [st["terms"][_stage_index(b)] for b in bad]
        st["terms"] = [t for t in st["terms"] if not any(t is g for g in gone)]
        self.h.dropped_terms += [{"stage": st["id"], **{k: g[k] for k in ("prim", "of", "level", "role")},
                                  "findings": [f for f in findings if f["term"] in bad]} for g in gone]

    def look(self, snap, look_at):
        """Look closer (CP-5's grasp-time look from above and model refresh) at the entity TRACE could not certify
        against, then rebuild the map. False when a look cannot help (no entity, the held object, or one the wrist
        camera has already seen close: its covariance is at its floor) or the view is unreachable."""
        h = self.h
        e = snap.entities.get(look_at.get("entity"))
        if e is None or self.depth > 0:
            return False
        obj = h.objects[e.label]
        if obj.held or any(v[0] == WRIST_CAMERA for v in obj.views):
            return False
        self.depth += 1
        try:
            fr = h.look_from_above(np.r_[obj.footprint_center(), obj.top_z], height=0.26)
        finally:
            self.depth -= 1
        if fr is None:
            return False
        mt = h.mask_matching(fr, [obj.category] + ([obj.name] if obj.name != obj.category else []), obj)
        if mt is not None:
            obj.add_points(h.filter_points(mask_points(fr, mt[0]), obj))
            obj.views.append((fr.name, "trace_look", float(mt[3])))
        h.rebuild_map(h.frames[-8:])
        return True

    def add_seeds(self, ends, finger, seed_goal, goal_near):
        """CP-5's planner path (the v0 seed, and the last seed in trace mode), computed only after the straight line
        failed: one planner call over the candidates' configurations (or the caller's pre-pose), attached to the
        candidate it reaches as q_path / p_path / R_path."""
        from .harness import NEAR, TRANSIT
        h = self.h
        pairs = [(ends[0], np.asarray(seed_goal, float))] if seed_goal is not None else \
            [(e, np.asarray(e.q, float)) for e in ends if e.q is not None]
        if not pairs:
            return
        q0 = h.rb.q()[:7]
        goals = [g for _, g in pairs]
        pl = h.planner
        pl.set_relax(q0, finger=finger)
        pl.goal_relax = (goals, NEAR) if goal_near else None
        t = time.time()
        path, how = pl.plan(q0, goals, TRANSIT, finger=finger, lift_z=h._lift_heights(q0, goals, finger),
                            time_budget=self.cfg["seed_budget_s"], allow_rrt=True)
        pl.set_relax(None)
        pl.goal_relax = None
        h._tick("plan", t)
        h.log("trace_seed", how=how, ok=path is not None, seconds=round(time.time() - t, 2))
        if path is None:
            # the joint-space straight line: joint-feasible by construction, and its orientation profile is the one
            # the arm turns through naturally (the eef geodesic can drive the wrist past its limit when the base
            # turns); TRACE certifies its clearance with its own margins
            path = np.array([q0, goals[0]])
        Q = densify(path, 0.03)
        Rq, pq = h.rm.site_pose(h.rm.full_q(Q, finger))
        e = pairs[int(np.argmin([np.abs(Q[-1] - g).max() for g in goals]))][0]
        e.q_path, e.p_path, e.R_path = Q, pq, Rq

    @staticmethod
    def clear_seeds(ends):
        for e in ends:
            e.q_path = e.p_path = e.R_path = None
        return False

    # ------------------------------------------------------------------------------------------------ summary
    def summary(self):
        ms = np.array(self.wall_ms) if self.wall_ms else np.zeros(1)
        return {"mode": self.mode, "requests": self.n, "results": dict(self.counts["results"]),
                "reactions": dict(self.counts["reactions"]), "executions": self.counts["executions"],
                "fast_path_share": round(self.fast / max(self.n, 1), 3),
                "solve_ms_p50": round(float(np.percentile(ms, 50)), 1),
                "solve_ms_p95": round(float(np.percentile(ms, 95)), 1),
                "guard_stops": dict(self.counts["stops"])}
