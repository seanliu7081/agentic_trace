"""Tests for traceplan (DESIGN §2): spline properties, primitives, fusion, the five program tests, the service on the
standard scenes, the perception limit, the snapshot round trip and CPU reproducibility."""
import numpy as np
import pytest
import torch

from traceplan.contract import GROUPS, EndCandidate, SceneSnapshot, Term
from traceplan.primitives import Body, EvalContext, Source, pose_tensors, residual
from traceplan.program import Program, fuse, run_program_tests
from traceplan.se3 import angle_between_np, so3_exp_np
from traceplan.service import TraceService
from traceplan.spline import K_SCHEDULE, SE3Spline
from traceplan.synthetic import R_DOWN, box_scene, default_hand, make_request, rotx, standard_scenes, trivial_arm_check

F64 = torch.float64
BLOCK = {"id": "block", "center": [0.5, 0.0, 0.1], "half": [0.05, 0.05, 0.1]}


@pytest.fixture(scope="module")
def scenes():
    return standard_scenes()


@pytest.fixture(scope="module")
def svc():
    return TraceService()


def _plan(svc, sc, **kw):
    """Outcome tests: a generous budget, so a loaded machine (or CUDA start-up in the first call) cannot cut the
    search; the timing targets are measured separately."""
    return svc.plan(make_request(sc, **{"budget_ms": 1e4, **kw}), sc["snap"], trivial_arm_check)


# --------------------------------------------------------------------------------------------- spline
def _spline(K, pin_end=True):
    rng = np.random.default_rng(K)
    return SE3Spline(so3_exp_np(rng.normal(size=3)), [0.3, -0.2, 0.3], so3_exp_np(rng.normal(size=3)),
                     [0.5, 0.2, 0.1], K, pin_end, dtype=F64)


def test_spline_straight_seed_is_the_geodesic():
    ts = np.linspace(0.0, 1.0, 101)
    for K in K_SCHEDULE:
        sp = _spline(K)
        th = sp.straight_theta()
        R, p = sp.sample_np(th, ts)
        geo_R = sp.R0 @ so3_exp_np(ts[:, None] * sp.w_ref)
        geo_p = sp.p0 + ts[:, None] * (sp.pg - sp.p0)
        assert np.abs(p - geo_p).max() <= 1e-12 and np.abs(R - geo_R).max() <= 1e-12
        out = sp.eval(th[None], ts)                                  # the differentiable path agrees
        assert np.abs(out["p"][0].numpy() - geo_p).max() <= 1e-12
        assert np.abs(out["R"][0].numpy() - geo_R).max() <= 1e-12


def test_spline_refine_is_exact():
    ts = np.linspace(0.0, 1.0, 101)
    gen = torch.Generator().manual_seed(0)
    for pin in (True, False):
        sp = _spline(K_SCHEDULE[0], pin)
        th = sp.straight_theta() + 0.2 * torch.randn(sp.dim, generator=gen, dtype=F64)
        for K in K_SCHEDULE[1:]:
            sp2, th2 = sp.refine(th[None], K)
            for a, b in zip(sp.sample_np(th, ts), sp2.sample_np(th2[0], ts)):
                assert np.abs(a - b).max() <= 1e-12
            sp, th = sp2, th2[0]


def test_spline_rotation_offset_bounded_by_control_norms():
    sp = _spline(11)
    th = 1.5 * torch.randn(6, sp.dim, generator=torch.Generator().manual_seed(1), dtype=F64)
    _, F = sp.ctrl(th)
    assert (F.norm(dim=-1) <= sp.phi_max + 1e-12).all() and (F.norm(dim=-1) > sp.phi_max - 1e-9).any()  # clamped
    phi = sp.eval(th, np.linspace(0.0, 1.0, 401))["phi"]
    assert (phi.norm(dim=-1).amax(1) <= F.norm(dim=-1).amax(1) + 1e-12).all()


# --------------------------------------------------------------------------------------------- primitives
def _ctx(snap, Rs, ps, start, end, holding=False, bodies=None):
    """EvalContext of explicit poses (one trajectory, one sample per pose), float64 on exact geometry."""
    t = lambda x: torch.as_tensor(np.asarray(x, float))
    z = torch.zeros(1, len(ps), 3, dtype=F64)
    out = {"R": t(Rs)[None], "p": t(ps)[None], "v": z, "acc": z}
    return EvalContext(out, np.linspace(0.0, 1.0, len(ps)), Body(bodies or default_hand(), "cpu", F64),
                       Source(snap, "exact", dtype=F64), pose_tensors(*start, None, F64, "cpu"),
                       pose_tensors(*end, None, F64, "cpu"), holding)


def test_clearance_sign():
    snap = box_scene(boxes=[BLOCK])
    ps = [[0.5, 0.0, 0.05], [0.5, 0.0, 0.5]]                     # palm inside the block, palm 0.33 m above it
    ctx = _ctx(snap, [R_DOWN, R_DOWN], ps, (R_DOWN, ps[0]), (R_DOWN, ps[1]))
    r = residual(ctx, Term("c", "clearance", {"body": "palm", "margin": 0.01}, "ineq", "safety"))[0].amax(-1)
    assert r[0] > 0 > r[1]
    hand = default_hand()
    palm = hand.groups == GROUPS.index("palm")
    x = hand.centers[palm] @ R_DOWN.T + ps[1]                     # palm spheres at the second pose
    assert float(r[1]) == pytest.approx(float((0.01 + hand.radii[palm] - snap.exact(x)).max()))


def test_tilt_angle():
    snap = box_scene(boxes=[BLOCK])
    R0, p0 = R_DOWN, [0.3, 0.0, 0.3]
    Rs = [rotx(0.3), rotx(0.3) @ R0]
    ctx = _ctx(snap, Rs, [p0, p0], (R0, p0), (R0, p0), holding=True)
    eef = residual(ctx, Term("t", "tilt_limit", {"u": "eef.z", "v": "world.up", "max": 0.1}, "ineq", "require"))
    assert float(eef[0, 0, 0]) == pytest.approx(0.2)                 # eef z tilted 0.3 rad from up
    pay = residual(ctx, Term("t", "tilt_limit", {"u": "payload.up", "v": "world.up", "max": 0.1, "u0": [0, 0, 1.0]},
                             "ineq", "require"))
    assert float(pay[0, 1, 0]) == pytest.approx(0.2)                 # the held object tilted with the hand
    assert float(pay[0, 0, 0]) == pytest.approx(np.pi - 0.3 - 0.1)   # rotx(0.3) is R0 turned upside down


def test_zone_relaxation():
    snap = box_scene(boxes=[BLOCK])
    ps = [[0.3, -0.2, 0.3], [0.3, 0.0, 0.3], [0.3, 0.2, 0.3]]
    ctx = _ctx(snap, [R_DOWN] * 3, ps, (R_DOWN, ps[0]), (R_DOWN, ps[2]))
    args = {"body": "palm", "margin": 0.05, "zone": 0.05}
    plain = residual(ctx, Term("c", "clearance", args, "ineq", "safety"))[0]
    relaxed = residual(ctx, Term("c", "clearance", dict(args, c_start=0.004, c_end=0.03), "ineq", "safety"))[0]
    d = (plain - relaxed).numpy()
    assert np.allclose(d[0], 0.05 - 0.003) and np.allclose(d[1], 0.0) and np.allclose(d[2], 0.05 - 0.029)


# --------------------------------------------------------------------------------------------- fusion
def test_fuse_attract_loosens_repel_tightens_flags():
    att = Term("att", "reach_point", {"a": "eef", "b": "block.top_center"}, "eq", "require", "terminal", 0.01,
               entity="block", effect="attract")
    rep = Term("rep", "clearance", {"body": "palm", "field": {"only": "block"}, "margin": 0.02}, "ineq", "require",
               sigma=0.01, entity="block", effect="repel")
    prog, flags = fuse(Program([att, rep]), box_scene(boxes=[BLOCK], cov=0.02), z=1.64)
    a, r = prog.terms
    assert a.sigma == pytest.approx(np.hypot(0.01, 0.02)) and a.args["sigma_accept"] == 0.01
    assert r.args["margin"] == pytest.approx(0.02 + 1.64 * 0.02) and r.sigma == 0.01
    assert not flags["att"]["certifiable"] and not flags["rep"]["certifiable"]
    _, flags = fuse(Program([att, rep]), box_scene(boxes=[BLOCK], cov=0.002))
    assert flags["att"]["certifiable"] and flags["rep"]["certifiable"]


# --------------------------------------------------------------------------------------------- program tests
def _findings(sc, extra):
    st, e = sc["start"], sc["ends"][0]
    make = lambda: SE3Spline(st["R"], st["p"], e.R, e.p, 4, False, device="cpu", dtype=F64)
    return run_program_tests(Program(list(sc["terms"]) + extra, sc["holding"]), sc["snap"], make, sc["bodies"],
                             np.random.default_rng(0))


@pytest.mark.parametrize("test, term", [
    ("lint", Term("bad", "reach_point", {"a": "eef", "b": "ghost.centroid"}, "eq", "require", "terminal")),
    ("range", Term("bad", "tilt_limit", {"u": "eef.z", "v": "world.up", "max": 10.0}, "ineq", "prefer",
                   sigma=0.05)),                                                  # degrees for radians
    ("theta_dependence", Term("bad", "above", {"a": "payload.base", "b": "block.top_center", "h": 0.0}, "ineq",
                              "prefer")),                                         # payload term, nothing held
    ("invariance", Term("bad", "reach_point", {"a": "eef", "b": [0.45, 0.15, 0.22]}, "eq", "prefer", "terminal")),
    ("start_feasibility", Term("bad", "above", {"a": "eef", "b": "block.top_center", "h": 0.4}, "ineq", "require")),
])
def test_program_tests_catch_injected_errors(scenes, test, term):
    found = _findings(scenes["S1"], [term])
    assert found and {f["term"] for f in found} == {"bad"} and test in {f["test"] for f in found}, found


def test_program_tests_pass_a_correct_program(scenes):
    ok = Term("over", "above", {"a": "eef", "b": "block.top_center", "h": 0.05}, "ineq", "require")
    assert _findings(scenes["S1"], [ok]) == []


# --------------------------------------------------------------------------------------------- service
def test_s1_certified_on_the_fast_path(svc, scenes):
    r = _plan(svc, scenes["S1"])
    assert r.result == "certified" and r.stats["stage"] == "fast_path" and r.timed is not None


def test_s2_certified_by_search(svc, scenes):
    r = _plan(svc, scenes["S2"])
    assert r.result == "certified" and r.stats["stage"] == "search"
    assert all(row["max_violation"] <= 1.0 for row in r.report if row["role"] != "prefer")


def test_s4_certified_upright(svc, scenes):
    sc = scenes["S4"]
    r = _plan(svc, sc)
    assert r.result == "certified"
    term = next(t for t in sc["terms"] if t.id == "upright")
    sp, th = SE3Spline.from_dict(r.spline)
    R, _ = sp.sample_np(th, np.linspace(0.0, 1.0, 201))
    up = R @ sc["start"]["R"].T @ np.asarray(term.args["u0"], float)
    assert angle_between_np(up, [0.0, 0.0, 1.0]).max() <= term.args["max"] + term.sigma   # certified tolerance


def test_s6_conflict_quotes_the_tilt_term(svc, scenes):
    r = _plan(svc, scenes["S6"])
    assert r.result == "conflict"
    q = next(q for q in r.quote if q["term"] == "upright")
    assert q["prim"] == "tilt_limit" and q["param"] == "max" and q["needed"] > q["value"]


def test_window_smaller_than_perception_needs_a_look(svc, scenes):
    snap = box_scene(boxes=[BLOCK], cov=0.02)
    sc = dict(scenes["S1"], snap=snap)
    end = EndCandidate(R_DOWN, np.r_[0.5, 0.0, 0.2], q=np.zeros(7), approach=np.r_[0.0, 0.0, -1.0], standoff="auto",
                       window=0.005, entity="block")
    r = svc.plan(make_request(sc, ends=[end]), snap, trivial_arm_check)
    assert r.result == "not_certifiable"
    assert r.look_at["entity"] == "block" and r.look_at["axis"] == "lateral"
    assert r.look_at["uncertainty"] > r.look_at["tolerance"] / 1.64


def test_snapshot_roundtrip(tmp_path):
    cov = {"block": np.diag([1e-4, 4e-4, 9e-6])}
    snap = box_scene(boxes=[BLOCK], unknown=[{"center": [0.62, 0.0, 0.1], "half": [0.05, 0.08, 0.1],
                                              "shadow_of": "block"}], cov=cov, version=7)
    snap.save(tmp_path / "s.npz")
    back = SceneSnapshot.load(tmp_path / "s.npz")
    assert back.version == 7 and back.res == snap.res and back.table_z == snap.table_z and back.meta == snap.meta
    for k in ("origin", "occ", "unknown", "labels", "shadow"):
        assert np.array_equal(getattr(back, k), getattr(snap, k)), k
    assert back.unknown.any() and back.shadow.any()
    e, f = snap.entities["block"], back.entities["block"]
    assert f.label == e.label and np.array_equal(f.points, e.points) and np.array_equal(f.cov, e.cov)
    assert all(np.array_equal(f.keypoints[k], e.keypoints[k]) for k in e.keypoints)


def test_same_seed_same_result_on_cpu(scenes):
    cpu = TraceService({"device": "cpu", "search": {"n_samples": 32, "explore_iters": (2, 2, 2, 2),
                                                     "exploit_iters": (2, 2, 2, 3)}})
    a, b = (_plan(cpu, scenes["S2"], budget_ms=1e6) for _ in range(2))
    assert a.result == b.result and a.stats["evals"] == b.stats["evals"] > 0
    assert a.spline == b.spline and [r["max_violation"] for r in a.report] == [r["max_violation"] for r in b.report]
