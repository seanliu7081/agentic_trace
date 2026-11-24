#!/usr/bin/env python
"""Capture how TRACE planned every request of a run, for the planning video (scripts/render_trace_video.py).

Replays each logged request (trace/req_*.pkl) on its saved snapshot with the run's logged arm / IK answers, as
scripts/trace_replay.py does, with recording hooks on the planner's steps:
  - end-pose candidates (kept / dropped and why; window flags = "look closer"),
  - every certificate attempt (path, per-term violations, arm check, reason),
  - the A* corridors and the fitted seeds,
  - every continuation level (tolerance inflation s, scene blur, K control poses, penalty mu),
  - every explore iteration (per seed: the sampling mean, the best so far and a few samples, with J),
  - the exploit (gradient) result and which seeds survive each level.
Time spent in the hooks is taken off TRACE's clock, so the budget cuts the search where it would without them.
The final path and certificate shown in the video always come from the run's own log; the replay supplies the
search in between, and a request whose replay ends differently is marked (the replay checks clearance on the saved
voxel grid, the run on the harness's live map).

    python scripts/trace_capture.py RUN_DIR [--device cuda:0]   -> RUN_DIR/trace/viz/{index.json, req_<id>.pkl}
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import pickle
import sys
import time as _time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import traceplan.search as S  # noqa: E402
import traceplan.service as SV  # noqa: E402
import traceplan.solvers as SO  # noqa: E402
from traceplan.contract import GROUPS, SceneSnapshot  # noqa: E402
from traceplan.objective import LEVELS, Objective  # noqa: E402
from traceplan.scene import exact_distance  # noqa: E402
from traceplan.spline import SE3Spline  # noqa: E402
from trace_replay import recorded  # noqa: E402

NTS = np.linspace(0.0, 1.0, 32)          # samples per drawn path
N_KEEP = 12                              # explore samples kept per seed and iteration (evenly spaced: the spread)


# ------------------------------------------------------------------------------------------------ clock
class _Clock:
    """time stand-in for the planner modules: perf_counter minus the time spent in the recording hooks."""
    overhead = 0.0

    @staticmethod
    def perf_counter():
        return _time.perf_counter() - _Clock.overhead


class _Hook:
    def __enter__(self):
        self.t = _time.perf_counter()

    def __exit__(self, *a):
        _Clock.overhead += _time.perf_counter() - self.t


S.time = _Clock
SV.time = _Clock


# ------------------------------------------------------------------------------------------------ recorder
class Rec:
    def __init__(self):
        self.events, self.phase, self.cand = [], "fast_path", None

    def add(self, kind, **kw):
        self.events.append(dict(kind=kind, phase=self.phase, cand=self.cand, **kw))


REC = Rec()
ORIG = {}


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def paths(spline, theta):
    """(M, len(NTS), 3) float32 positions and (M, K, 3) control points of theta rows on this spline."""
    th = torch.as_tensor(theta).to(device=spline.device, dtype=spline.dtype).reshape(-1, spline.dim)
    with torch.no_grad():
        p = spline.eval(th, NTS)["p"]
        P, _ = spline.ctrl(th)
    return _np(p).astype(np.float32), _np(P).astype(np.float32)


def rec_explore(name, J, theta, budget_evals, gen, cfg):
    """solvers.explore with recording (same draws from `gen`, same result)."""
    if name == "none":
        return theta
    c = {**SO.DEFAULTS, **cfg}
    theta = theta.detach()
    M, D = theta.shape
    Sn = min(c["n_samples"], budget_evals)
    A = SO.noise_factor(c["layout"], c["alpha_pos"], c["alpha_rot"], c["smooth_eps"]).to(theta)
    method = SO.EXPLORERS[name](theta, A, c, Sn)
    best, best_J = theta.clone(), theta.new_full((M,), math.inf)
    rows = torch.arange(M, device=theta.device)
    n_it = budget_evals // Sn
    keep = np.unique(np.linspace(1, Sn - 1, min(N_KEEP, Sn - 1)).astype(int)) if Sn > 1 else np.zeros(0, int)
    with torch.no_grad():
        for it in range(n_it):
            z = torch.randn((M, Sn, D), generator=gen, device=gen.device, dtype=theta.dtype).to(theta.device)
            z[:, 0] = 0.0
            X = method.propose(z)
            Jb = J(X.reshape(M * Sn, D)).reshape(M, Sn)
            j, i = Jb.min(1)
            better = j < best_J
            best = torch.where(better[:, None], X[rows, i], best)
            best_J = torch.where(better, j, best_J)
            with _Hook():
                mean_p, mean_c = paths(J.spline, X[:, 0])
                best_p, best_c = paths(J.spline, best)
                samp_p = paths(J.spline, X[:, keep].reshape(-1, D))[0].reshape(M, len(keep), len(NTS), 3) \
                    if len(keep) else np.zeros((M, 0, len(NTS), 3), np.float32)
                REC.add("explore", level=int(J.level), it=it, n_it=n_it, solver=name, n_samples=Sn,
                        mean=mean_p, best=best_p, best_ctrl=best_c, samples=samp_p.astype(np.float16),
                        J_best=_np(best_J).astype(float), J_min=_np(Jb.min(1).values).astype(float),
                        J_med=_np(Jb.median(1).values).astype(float))
            method.update(z, X, Jb)
    return best


def rec_exploit(name, J, theta, budget_iters, cfg):
    out = ORIG["exploit"](name, J, theta, budget_iters, cfg)
    with _Hook():
        with torch.no_grad():
            J1 = _np(J(out)).astype(float)
        p, ctrl = paths(J.spline, out)
        REC.add("exploit", level=int(J.level), solver=name, iters=int(budget_iters), after=p, ctrl=ctrl, J=J1)
    return out


def rec_set_level(self, level, spline):
    ORIG["set_level"](self, level, spline)
    with _Hook():
        L = LEVELS[level]
        REC.add("level", level=int(level), s=float(L["s"]), beta=float(L["beta"]), K=int(spline.K),
                mu=float(L["mu"]), M=int(getattr(self, "M", 0) or 0))


def rec_select(self, idx):
    ORIG["select"](self, idx)
    with _Hook():
        REC.add("select", level=int(getattr(self, "level", -1)), idx=_np(idx).astype(int).tolist())


def rec_astar(snap, p0, pg, inflate, n, cfg, deadline=math.inf):
    out = ORIG["astar"](snap, p0, pg, inflate, n, cfg, deadline)
    with _Hook():
        REC.add("astar", corridors=[np.asarray(P, np.float32) for P in out], inflate=float(inflate),
                res=float(cfg["astar_res"]))
    return out


def rec_path_seeds(spline, cand, snap, bodies, gen, cfg, deadline=math.inf):
    REC.phase = "search"
    th = ORIG["seeds"](spline, cand, snap, bodies, gen, cfg, deadline)
    with _Hook():
        p, ctrl = paths(spline, th)
        REC.add("seeds", paths=p, ctrl=ctrl, K=int(spline.K), n_astar=int(len(th) - 1),
                seeding=cfg["seeding"])
    return th


def rec_continuation(obj, spline, theta, gen, cfg, deadline, provided=None, warm=False):
    REC.phase = "search"
    with _Hook():
        REC.add("search_start", warm=bool(warm), n_seeds=int(len(theta)), budget_left_ms=float(
            max(0.0, (deadline - _Clock.perf_counter()) * 1e3)))
    out = ORIG["cont"](obj, spline, theta, gen, cfg, deadline, provided=provided, warm=warm)
    with _Hook():
        sp2, th, J, info = out
        p, ctrl = paths(sp2, th)
        REC.add("search_end", levels_run=int(info["levels_run"]), paths=p, ctrl=ctrl, J=_np(J).astype(float),
                K=int(sp2.K))
    return out


def rec_objective(self, c, K):
    REC.cand = int(c.index)
    return ORIG["objective"](self, c, K)


def rec_certify(obj, theta_1, req, cand, arm_check, n=128):
    ok, cert = ORIG["certify"](obj, theta_1, req, cand, arm_check, n)
    with _Hook():
        sp = obj.spline
        R, p = sp.sample_np(_np(theta_1).astype(float), np.linspace(0.0, 1.0, 48))
        rows = []
        for r in cert.get("report") or []:
            d = {k: r.get(k) for k in ("term", "prim", "role", "src", "max_violation", "at_t", "r_max", "sigma")}
            if r.get("at_x") is not None:
                d["at_x"] = np.asarray(r["at_x"], float).tolist()
            rows.append(d)
        REC.add("certify", cand_index=int(cand.index), ok=bool(ok), reason=cert.get("reason"), rows=rows,
                arm=cert.get("arm"), path=p.astype(np.float32), R=R.astype(np.float32), K=int(sp.K))
    return ok, cert


def _why_dropped(c, snap, bodies, holding, ik_check):
    held = (holding,) if holding else ()
    x = c.R @ bodies.centers.T + c.p[:, None]
    clear = exact_distance(snap, x.T, held) - bodies.radii
    finger = bodies.groups == GROUPS.index("finger")
    if c.approach is not None and c.entity is not None and finger.any():
        clear[finger] = exact_distance(snap, x.T[finger], held + (c.entity,)) - bodies.radii[finger]
    if snap.table_z is not None:
        clear = np.minimum(clear, x[2] - snap.table_z - bodies.radii)
    if clear.min() < 0:
        g = GROUPS[int(bodies.groups[int(clear.argmin())])]
        return f"{g} inside an obstacle at the end pose ({-clear.min() * 1000:.0f} mm)", float(clear.min())
    return "no IK solution at the end pose", float(clear.min())


def rec_score(cands, snap, bodies, start, holding, ik_check, G, dropped=None):
    # dropped: the planner's own list (score_candidates gained it after this tool was written); only the real call
    # below fills it, the per-candidate probes do not
    with _Hook():
        info = []
        for c in cands:
            kept1 = ORIG["score"]([c], snap, bodies, start, holding, ik_check, 1)
            d = dict(index=int(c.index), p=np.asarray(c.p, float), R=np.asarray(c.R, float),
                     approach=None if c.approach is None else np.asarray(c.approach, float),
                     standoff=float(c.standoff or 0.0), entity=c.entity, ok=bool(kept1), reason=None, clear=None)
            if not kept1:
                d["reason"], d["clear"] = _why_dropped(c, snap, bodies, holding, ik_check)
            info.append(d)
    out = ORIG["score"](cands, snap, bodies, start, holding, ik_check, G, dropped)
    with _Hook():
        kept = [int(c.index) for c in out]
        for d in info:
            if d["ok"] and d["index"] not in kept:
                d["ok"], d["reason"] = False, f"ranked out: only the best {G} ends are kept"
        REC.add("candidates", items=info, kept=kept, G=int(G))
    return out


def rec_candidate(self, i, e):
    c = ORIG["candidate"](self, i, e)
    with _Hook():
        REC.add("end", index=int(i), p=np.asarray(c.p, float), R=np.asarray(c.R, float),
                contact_p=np.asarray(e.p, float), standoff=float(c.standoff or 0.0),
                approach=None if c.approach is None else np.asarray(c.approach, float), entity=c.entity,
                flag=None if not c.flag else {k: (float(v) if isinstance(v, (int, float, np.floating)) else
                                                  (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v))
                                              for k, v in c.flag.items()})
    return c


def install():
    ORIG.update(explore=S.explore, exploit=S.exploit, astar=S.astar_paths, seeds=SV.path_seeds,
                score=SV.score_candidates, certify=SV.certify, cont=SV.continuation, set_level=Objective.set_level,
                select=Objective.select, objective=SV._Plan.objective, candidate=SV._Plan.candidate)
    S.explore, S.exploit, S.astar_paths = rec_explore, rec_exploit, rec_astar
    SV.path_seeds, SV.score_candidates, SV.certify, SV.continuation = rec_path_seeds, rec_score, rec_certify, \
        rec_continuation
    Objective.set_level, Objective.select = rec_set_level, rec_select
    SV._Plan.objective, SV._Plan.candidate = rec_objective, rec_candidate


# ------------------------------------------------------------------------------------------------ main
def logged_path(resp, n=64):
    if not resp.spline:
        return None, None
    sp, th = SE3Spline.from_dict(resp.spline)
    R, p = sp.sample_np(th, np.linspace(0.0, 1.0, n))
    return p.astype(np.float32), R.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run = Path(args.run_dir)
    tdir = run / "trace"
    out_dir = tdir / "viz"
    out_dir.mkdir(exist_ok=True)
    events = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
    when = {e.get("id"): e for e in events if e.get("event") == "trace_request"}
    done = {e.get("id"): e for e in events if e.get("event") == "trace_result"}
    tel = [json.loads(l) for l in (run / "rec" / "telemetry.jsonl").read_text().splitlines() if l.strip()] \
        if (run / "rec" / "telemetry.jsonl").exists() else []
    tel_steps = [int(r["step"]) for r in tel]
    install()
    svc = SV.TraceService({"device": args.device})
    snaps, index = {}, []
    t0 = _time.time()
    for f in sorted(tdir.glob("req_*.pkl")):
        req, old = pickle.load(open(f, "rb"))
        if req.scene_version not in snaps:
            snaps[req.scene_version] = SceneSnapshot.load(tdir / "snapshots" / f"v{req.scene_version}.npz")
        arm, ik, miss = recorded(req.meta.get("callbacks", []))
        REC.__init__()
        _Clock.overhead = 0.0
        new = svc.plan(req, snaps[req.scene_version], arm, ik)
        ev = when.get(req.request_id) or {}
        step = int(ev.get("step", 0))
        k = max(0, bisect.bisect_right(tel_steps, step) - 1) if tel_steps else 0
        lp, lR = logged_path(old)
        st = old.stats or {}
        terms = [dict(id=t.id, prim=t.prim, role=t.role, scope=t.scope if isinstance(t.scope, str) else list(t.scope),
                      sigma=float(t.sigma), src=t.src, entity=t.entity, args={a: v for a, v in t.args.items()
                                                                              if isinstance(v, (str, int, float))})
                 for t in req.terms]
        rec = dict(
            id=req.request_id, tag=req.meta.get("tag"), mode=req.mode, budget_ms=float(req.budget_ms),
            seed=int(req.seed), holding=req.holding, finger=float(req.finger), zone=float(req.zone),
            start_R=np.asarray(req.start_R, float), start_p=np.asarray(req.start_p, float),
            bodies=dict(centers=np.asarray(req.bodies.centers, float), radii=np.asarray(req.bodies.radii, float),
                        groups=np.asarray(req.bodies.groups, int)),
            terms=terms, n_ends=len(req.ends), free_end=req.free_end, margins=req.margins,
            t=float(ev.get("t", 0.0)), step=step, frame=int(k), t_result=float((done.get(req.request_id) or {})
                                                                              .get("t", ev.get("t", 0.0))),
            logged=dict(result=old.result, stage=st.get("stage"), solve_ms=float(st.get("solve_ms") or 0.0),
                        end_index=old.end_index, standoff=old.standoff, report=old.report, quote=old.quote,
                        look_at=old.look_at, findings=old.findings, path=lp, R=lR,
                        tube_mm=(old.valid_while or {}).get("tube_mm"),
                        duration_s=float(old.timed.t[-1]) if old.timed is not None and len(old.timed.t) else None,
                        levels_run=st.get("levels_run"), evals=st.get("evals"), seeds=st.get("seeds"),
                        budget_exhausted=st.get("budget_exhausted")),
            replay=dict(result=new.result, stage=(new.stats or {}).get("stage"),
                        solve_ms=float((new.stats or {}).get("solve_ms") or 0.0), unmatched=dict(miss)),
            events=REC.events)
        with open(out_dir / f"req_{req.request_id}.pkl", "wb") as fh:
            pickle.dump(rec, fh)
        index.append(dict(id=req.request_id, tag=rec["tag"], mode=req.mode, t=rec["t"], step=step, frame=k,
                          logged=old.result, logged_stage=st.get("stage"), replay=new.result,
                          replay_stage=(new.stats or {}).get("stage"), same=old.result == new.result,
                          n_events=len(REC.events)))
        print(f"{req.request_id:18} {req.mode:8} logged {old.result:15} {str(st.get('stage')):10} replay "
              f"{new.result:15} {str((new.stats or {}).get('stage')):10} events {len(REC.events)}", flush=True)
    (out_dir / "index.json").write_text(json.dumps({"run": str(run), "requests": index,
                                                    "seconds": round(_time.time() - t0, 1)}, indent=1))
    print(f"trace_capture: {len(index)} requests, {sum(r['same'] for r in index)} with the logged result -> {out_dir}")


if __name__ == "__main__":
    main()
