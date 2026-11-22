"""Offline tests of the harness side of the TRACE integration (no simulator, no model, no TRACE service)."""
from types import SimpleNamespace

import numpy as np
import pytest
from traceplan.contract import EndCandidate, Spheres, Timed, TraceResponse

from atrace import levels, tracemotion
from atrace.body import ControllerModel
from atrace.guard import ContactDetector
from atrace.harness import Harness, StageFailure
from atrace.levels import LADDER, LEVELS, can_relax, compile_terms, relax, safety_terms, tube
from atrace.motion import Attached, Checker, ExecResult, TAU_LEAD
from atrace.perception import FREE, OCC, UNKNOWN, ObjectModel, SceneMap
from atrace.program import merge_compiles, validate_program
from atrace.snapshot import build_snapshot, entity_cov

DEG = np.pi / 180


def _fake_h(held=None):
    R = np.diag([1.0, -1.0, -1.0])                       # top-down hand
    rb = SimpleNamespace(eef_pose=lambda: (R, np.array([0.5, 0.0, 0.3])), q=lambda: np.zeros(9), steps=0)
    att = Attached(3, np.zeros((4, 3)), 0.0, up_site=R.T @ np.array([0.0, 0.0, 1.0])) if held else None
    return SimpleNamespace(id_map={"o1": 3, "o2": 5, "o3": 3}, objects={3: None, 5: None}, held=held, rb=rb,
                           checker=SimpleNamespace(attached=att), scene=SimpleNamespace(table_z=0.05),
                           rm=SimpleNamespace(root_p=np.array([-0.6, 0.0, 0.9])))


STAGE_TERMS = [{"prim": "tilt_limit", "of": "o1", "level": "upright", "role": "require", "relax_to": "tilted"},
               {"prim": "clearance", "of": "o2", "level": "wide", "role": "prefer", "relax_to": "normal"},
               {"prim": "approach", "of": "o2", "level": "vertical", "role": "require", "relax_to": "steep"},
               {"prim": "clearance", "of": "o9", "level": "close", "role": "require", "relax_to": None}]


# ------------------------------------------------------------------------------------------- levels
def test_levels_compile_and_relax():
    h = _fake_h(held=3)
    t = {x.id: x for x in compile_terms(STAGE_TERMS, h, "carry", "o1", approach=[0, 0, -1.0], entity="o2")}
    tilt = t["lang0_tilt_limit_o1"]
    assert tilt.args["max"] == pytest.approx(10 * DEG) and tilt.sigma == pytest.approx(3 * DEG)
    assert tilt.src == "language" and tilt.level == "upright" and tilt.relax_to == "tilted" and tilt.param == "max"
    np.testing.assert_allclose(tilt.args["u0"], [0, 0, 1], atol=1e-12)         # upright at grasp, hand unchanged
    cl = t["lang1_clearance_o2"]
    assert cl.args["field"] == {"exclude": [], "only": "o2", "unknown": False} and cl.args["margin"] == 0.08
    assert cl.effect == "repel" and cl.entity == "o2" and cl.sigma == 0.02 and cl.role == "prefer"
    assert t["lang2_approach_o2"].prim == "approach_cone" and t["lang2_approach_o2"].args["half_angle"] == \
        pytest.approx(10 * DEG)
    assert t["lang3_clearance_o9"].args["field"]["only"] == "o9"              # unknown id kept for the program tests
    # no tilt while not carrying it, no approach without a matching contact end; an alias names the same entity
    ids = [x.id for x in compile_terms(STAGE_TERMS, h, "move", None)]
    assert ids == ["lang1_clearance_o2", "lang3_clearance_o9"]
    assert levels.entity_id(h, "o3") == "o1" and levels.entity_id(h, "zz") == "zz"
    s = {x.id: x for x in safety_terms(h, "carry", LADDER, 0.05, holding="o1")}
    tv = tube(LADDER, LADDER["v_min"])
    assert s["clear_palm"].args["margin"] == pytest.approx(0.008 + tv + 0.005 + levels.SAFETY_SIGMA)
    assert s["clear_payload"].args["field"]["exclude"] == ["o1"] and s["clear_payload"].role == "safety"
    assert s["plane_finger"].args["margin"] == pytest.approx(0.004 + tv + levels.SAFETY_SIGMA)
    assert s["plane_finger"].args["z"] == 0.05
    assert s["known_link"].role == "require" and s["known_link"].args["field"]["unknown"]
    np.testing.assert_allclose(s["reach"].args["center"], [-0.6, 0, 1.233])
    assert "clear_payload" not in {x.id for x in safety_terms(h, "move", LADDER, 0.05, holding=None)}
    # can_relax: the allowed level must admit the quoted value
    assert can_relax({"needed": 23 * DEG}, tilt) and not can_relax({"needed": 30 * DEG}, tilt)
    assert can_relax({"needed": 0.031}, cl) and not can_relax({"needed": 0.02}, cl)
    assert not can_relax({"needed": 0.5}, t["lang3_clearance_o9"])            # no relax_to: model call
    assert can_relax({"needed": 1.0}, s["known_palm"])
    r = relax(tilt)
    assert r.level == "tilted" and r.args["max"] == pytest.approx(25 * DEG) and r.relax_to is None
    k = relax(s["known_palm"])
    assert k.role == "prefer" and k.level == "slow" and k.args["slow"] == levels.LIMITS["v_slow"]


def test_program_terms_validate_and_merge():
    st = {"id": "s1", "op": "transfer", "object": "o1", "relation": "in", "reference": "o2", "place_x": None,
          "place_y": None, "effect": None, "effect_x": None, "effect_y": None,
          "terms": [{"prim": "tilt_limit", "of": "o1", "level": "wide", "role": "require", "relax_to": None},
                    {"prim": "clearance", "of": "o2", "level": "wide", "role": "require", "relax_to": "upright"}]}
    prog = {"objects": [{"id": "o1", "name": "a", "prompts": ["a"], "x": 1, "y": 1, "part_of": None},
                        {"id": "o2", "name": "b", "prompts": ["b"], "x": 2, "y": 2, "part_of": None}], "stages": [st]}
    warn = validate_program(prog)
    assert len(warn) == 2 and st["terms"] == [{"prim": "clearance", "of": "o2", "level": "wide", "role": "require",
                                               "relax_to": None}]
    t = lambda prim, of, level: {"prim": prim, "of": of, "level": level, "role": "require", "relax_to": None}
    progs = [{"stages": [{"terms": [t("tilt_limit", "o1", "upright"), t("clearance", "o2", "wide")]}]},
             {"stages": [{"terms": [t("tilt_limit", "o1", "tilted"), t("clearance", "o2", "wide")]}]},
             {"stages": [{"terms": [t("tilt_limit", "o1", "tilted")]}]},
             {"stages": [{"terms": [t("tilt_limit", "o1", "tilted"), t("approach", "o2", "vertical")]}]},
             {"stages": [{"terms": []}]}]
    m = {(x["prim"], x["of"]): x for x in merge_compiles(progs)["stages"][0]["terms"]}
    assert m[("tilt_limit", "o1")]["level"] == "tilted" and m[("tilt_limit", "o1")]["kappa"] == 0.8
    assert m[("clearance", "o2")]["role"] == "require" and m[("clearance", "o2")]["kappa"] == 0.4
    assert m[("approach", "o2")]["role"] == "prefer" and m[("approach", "o2")]["kappa"] == 0.2


# ----------------------------------------------------------------------------------------- snapshot
def test_snapshot_from_synthetic_scenemap():
    sc = SceneMap((0.0, -0.2, 0.0), (0.4, 0.2, 0.3), res=0.01, table_z=0.05)
    sc.state[:] = FREE
    sc.state[:, :, :6] = OCC                               # the table slab (below / at the table layer)
    sc.state[18:22, 8:12, 6:12] = OCC                      # a can, label 3
    sc.label[18:22, 8:12, 6:12] = 3
    sc.state[22:26, 8:12, 6:12] = UNKNOWN                  # the can's shadow
    sc.shadow[22:26, 8:12, 6:12] = 3
    sc.state[5:8, 25:28, 6:9] = OCC                        # a box, label 5
    sc.label[5:8, 25:28, 6:9] = 5
    can = ObjectModel(3, "can", "can", "target", 0.05)
    g = np.stack(np.meshgrid(np.arange(0.18, 0.22, 0.004), np.arange(-0.12, -0.08, 0.004), [0.06, 0.12]), -1)
    can.add_points(g.reshape(-1, 3))
    can.views.append(("head", "sam", 0.9))
    box = ObjectModel(5, "box", "box", "context", 0.05)
    box.add_points(np.array([[0.06, 0.06, 0.07], [0.07, 0.07, 0.08], [0.06, 0.07, 0.085], [0.075, 0.055, 0.07]]))
    box.views.append(("robot0_eye_in_hand", "sam", 0.9))
    objects = {3: can, 5: box}
    rm = SimpleNamespace(points=SimpleNamespace(group=np.zeros(0, int), radius=np.zeros(0)))
    h = SimpleNamespace(scene=sc, objects=objects, id_map={"o1": 3, "o2": 5}, held=None,
                        rb=SimpleNamespace(eef_pose=lambda: (np.eye(3), np.array([0.2, -0.1, 0.25]))),
                        checker=Checker(rm, sc, objects))
    s = build_snapshot(h)
    assert s.version == 1 and s.occ.shape == sc.shape and s.res == 0.01 and s.table_z == 0.05
    # the table is the analytic plane: its layer stays solid only under something standing on it (CP-5's rule)
    assert not s.occ[:, :, :5].any() and not s.occ[0, 0, 5] and s.occ[19, 9, 5] and s.occ[19, 9, 7]
    assert s.unknown[23, 9, 7] and s.shadow[23, 9, 7] == 3 and s.labels[19, 9, 7] == 3 and s.labels[19, 9, 20] == 0
    assert set(s.entities) == {"o1", "o2"} and s.entities["o1"].label == 3
    np.testing.assert_allclose(s.entities["o1"].cov, np.eye(3) * (0.003 ** 2 + 0.006 ** 2))
    np.testing.assert_allclose(s.entities["o2"].cov, np.eye(3) * 0.003 ** 2)    # seen close by the wrist camera
    np.testing.assert_allclose(s.entities["o1"].keypoints["top_center"][:2], can.footprint_center())
    assert s.entities["o1"].keypoints["base"][2] == 0.05
    p = np.array([[0.20, -0.10, 0.16], [0.20, -0.10, 0.27]])
    d = s.exact(p)
    assert 0.02 < d[0] < 0.06 and d[1] > d[0]
    assert s.exact(p, ("o1",))[0] > d[0]                                   # excluding the can opens the space
    assert build_snapshot(h) is s                                          # cached while nothing changed
    sc._touch()
    assert build_snapshot(h).version == 2
    # holding the can: it leaves occupied space and its entity is posed in the hand
    h.held, can.held = 3, True
    h.checker.attached = Attached(3, np.zeros((3, 3)), 0.0, model_site=np.array([[0, 0, 0.02], [0, 0, 0.05],
                                                                                  [0.01, 0, 0.03]]))
    s3 = build_snapshot(h)
    assert s3.version == 3 and not s3.occ[19, 9, 7] and not s3.unknown[23, 9, 7]
    np.testing.assert_allclose(s3.entities["o1"].points.mean(0), [0.2 + 0.01 / 3, -0.1, 0.25 + 0.1 / 3])
    assert entity_cov(box)[0, 0] == pytest.approx(0.003 ** 2)


# ------------------------------------------------------------------------------- controller, detector
def _rm_stub(M=None):
    """A 6-DOF 'robot' whose joints are the site pose: q = (p, rotation vector, -, fingers), J = [I | 0], so the
    controller model's coupling comes from the mass matrix M (7,7) alone."""
    from scipy.spatial.transform import Rotation
    M = np.eye(7) if M is None else M
    J = np.c_[np.eye(6), np.zeros(6)]
    return SimpleNamespace(limits=np.c_[np.full(9, -3.0), np.full(9, 3.0)], mass_matrix=lambda q: M,
                           jacobian=lambda q: np.repeat(J[None], len(q), 0),
                           site_pose=lambda q: (Rotation.from_rotvec(q[:, 3:6]).as_matrix(), q[:, :3].copy()))


def _coupled_M(seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(7, 7))
    return X @ X.T + 2.0 * np.eye(7)


def _trapezoid(L, vmax, amax, dt):
    ta = vmax / amax
    T = 2 * ta + (L - amax * ta ** 2) / vmax
    t = np.arange(int(np.ceil(T / dt)) + 1) * dt
    s = np.where(t < ta, 0.5 * amax * t ** 2, np.where(t < T - ta, 0.5 * amax * ta ** 2 + vmax * (t - ta),
                 L - 0.5 * amax * np.maximum(T - t, 0) ** 2))
    return t, np.clip(s, 0, L)


def test_controller_model_predict_and_command():
    cm = ControllerModel(_rm_stub())
    assert cm.kd / cm.kp == pytest.approx(TAU_LEAD)          # the executor's velocity lead is this model's kd/kp
    A, J = cm.coupling(np.zeros(9))
    np.testing.assert_allclose(A, np.eye(6), atol=1e-12)     # isotropic inertia: no position/orientation coupling
    e, v = cm.predict(A, np.zeros(6), np.zeros(6))           # at rest on the command: stays
    np.testing.assert_allclose(np.r_[e, v], 0.0)
    # decoupled: per axis the critically damped closed form with the command held, e(t) = (e0 + (v0 + w e0) t) e^-wt
    w, dt = np.sqrt(cm.kp), cm.dt
    vel = np.array([0.1, -0.2, 0.05, 0.3, 0.0, -0.1])
    e0 = -vel * TAU_LEAD
    e1, v1 = cm.predict(A, e0, vel)
    np.testing.assert_allclose(e1, (e0 + (vel + w * e0) * dt) * np.exp(-w * dt), atol=1e-12)
    np.testing.assert_allclose(v1, (vel - w * (vel + w * e0) * dt) * np.exp(-w * dt), atol=1e-12)
    # coupled loop (eigenvalues of A away from 1): the per-axis lead leaves the plan by millimetres on a 20 cm move,
    # the model-inverse command (run_timed's feedforward) follows it
    cm = ControllerModel(_rm_stub(_coupled_M()))
    A, J = cm.coupling(np.zeros(9))
    lam = np.linalg.eigvals(A).real
    assert lam.min() < 0.7 and lam.max() > 1.3
    t, s = _trapezoid(0.2, 0.3, 0.5, dt)
    X = np.c_[s, np.zeros((len(t), 5))]                      # 6-D path: x moves, orientation held
    K = len(t)
    Xp = np.r_[X[:1], X, X[-1:]]
    qdd = (Xp[2:] - 2 * Xp[1:-1] + Xp[:-2]) / dt ** 2        # J = [I | 0]: joint acceleration = site acceleration
    qdd = np.c_[0.5 * (qdd[1:] + qdd[:-1]), np.zeros(K - 1)]

    def track(cmd):
        x, v, worst = X[0].copy(), np.zeros(6), 0.0
        for k in range(K + 20):
            c = cmd(k) if k < K - 1 else X[-1]
            for _ in range(5):                               # the controller runs faster than the command
                cm.dt = dt / 5
                e, v = cm.predict(A, x - c, v)
                x = c + e
            cm.dt = dt
            worst = max(worst, np.linalg.norm(x[:3] - X[min(k + 1, K - 1), :3]))
        return worst

    lead = track(lambda k: np.array([np.interp(min(t[k] + TAU_LEAD, t[-1]), t, X[:, i]) for i in range(6)]))
    ff = track(lambda k: 0.5 * (X[k] + X[k + 1]) + cm.command(A, J, (X[k + 1] - X[k]) / dt, qdd[k]))
    assert lead > 0.005 and ff < 0.0015, (lead, ff)


def _descent(cm, n, v_des, wall=None, accel=0.25, noise=1e-4, seed=0, f_contact=True, lam=20.0):
    """Plant = the controller model (coupled when the stub's mass matrix is) + measurement noise; command = a
    run_path-like plan point led by TAU_LEAD along -z with a trapezoid speed profile; an optional rigid wall stops the
    hand, and the wrist force (arm on hand) is then the OSC's push down, task inertia lam."""
    rng = np.random.default_rng(seed)
    t = np.arange(n + 40) * cm.dt
    s = np.minimum(0.5 * accel * t ** 2, v_des * t - 0.5 * v_des ** 2 / accel)
    plan = lambda tt: np.array([0.0, 0.0, -np.interp(tt, t, s), 0.0, 0.0, 0.0])
    det = ContactDetector(cm)
    A, _ = cm.coupling(np.zeros(9))
    x, v = np.zeros(6), np.zeros(6)
    F0 = np.array([0.0, 0.0, 5.0])
    first, contact_step = None, None
    for k in range(n):
        c = plan(k * cm.dt + TAU_LEAD)
        e, v = cm.predict(A, x - c, v)
        x = c + e
        if wall is not None and x[2] < wall:
            x[2], v = wall, np.zeros(6)
            contact_step = k if contact_step is None else contact_step
        F = F0 if not (f_contact and contact_step is not None) else \
            F0 - np.array([0, 0, lam * cm.kp * max(0.0, wall - c[2])])
        q = np.r_[x + np.r_[rng.normal(0, noise, 3), 0, 0, 0], 0.0, 0.0, 0.0]
        if det.update(q, np.r_[v, 0.0, 0.0, 0.0], (np.eye(3), c[:3]), F, F0) and first is None:
            first = k
    return first, contact_step


def test_contact_detector_synthetic():
    for M in (None, _coupled_M()):
        cm = ControllerModel(_rm_stub(M))
        for v_des in (0.06, 0.4):                              # free motion (guarded descent speed, top transit speed)
            for seed in range(5):
                assert _descent(cm, 120, v_des, seed=seed)[0] is None
        for v_des, wall in ((0.06, -0.03), (0.05, -0.02), (0.04, -0.02), (0.12, -0.05)):
            for seed in range(5):
                first, c = _descent(cm, 120, v_des, wall=wall, seed=seed)
                assert c is not None and first is not None and first - c <= 1, (v_des, first, c)
                first, c = _descent(cm, 120, v_des, wall=wall, seed=seed, f_contact=False)   # prediction error alone
                assert first is not None and first - c <= 2, (v_des, first, c)
    # the CP-5 force rule is kept with its own reason; the detector stops with the caller's reason
    rob = SimpleNamespace(q=lambda: np.zeros(9), joint_vel=lambda: np.zeros(9), last_goal=(np.eye(3), np.zeros(3)))
    g = ContactDetector(cm).guard("early_contact", lambda r, F, F0, k: "early_contact" if F[2] > 12 else None)
    assert g(rob, np.array([0, 0, 13.0]), np.zeros(3), 3) == "early_contact"
    g = ContactDetector(cm).guard("support_contact", lambda r, F, F0, k: None)
    assert g(rob, np.zeros(3), np.zeros(3), 1) is None
    rob.last_goal = (np.eye(3), np.array([0, 0, -0.01]))
    assert g(rob, np.zeros(3), np.zeros(3), 2) == "support_contact"     # commanded down, did not move: blocked
    # the force along the motion counts only while the hand falls behind: leaving a jammed contact (hand on the
    # prediction) does not fire; a contact (short of it by more than e_push, less than e_min) does
    cm = ControllerModel(_rm_stub())
    A, _ = cm.coupling(np.zeros(9))
    c = np.array([0, 0, -0.01])
    e, _ = cm.predict(A, np.r_[-c, np.zeros(3)], np.zeros(6))
    for behind, fires in ((0.0, False), (0.0003, True)):
        det = ContactDetector(cm)
        det.update(np.zeros(9), np.zeros(9), (np.eye(3), c), np.zeros(3), np.zeros(3))
        q = np.r_[c + e[:3] + np.array([0, 0, behind]), np.zeros(6)]
        assert det.update(q, np.zeros(9), (np.eye(3), c), np.array([0, 0, -7.0]), np.zeros(3)) == fires


def test_path_distance_tube():
    from atrace.motion import path_distance
    P = np.c_[np.linspace(0, 0.1, 11), np.zeros(11), np.zeros(11)]
    assert path_distance(np.array([0.035, 0.0, 0.0]), P[3:6]) == pytest.approx(0.0)       # lag along the path
    assert path_distance(np.array([0.045, 0.003, 0.0]), P[3:6]) == pytest.approx(0.003)   # off the path
    assert path_distance(np.array([0.02, 0.0, 0.0]), P[3:6]) == pytest.approx(0.01)       # left behind the window
    assert path_distance(np.array([0.1, 0.0, 0.004]), P[10:11]) == pytest.approx(0.004)


# ------------------------------------------------------------------------------- trace_move reactions
class FakeService:
    def __init__(self, script):
        self.script, self.requests = list(script), []

    def plan(self, req, snap, arm_check, ik_check=None):
        self.requests.append(req)
        return self.script.pop(0)(req)


def _timed():
    t = np.arange(5) * 0.05
    return Timed(t, np.c_[np.linspace(0.5, 0.6, 5), np.zeros(5), np.full(5, 0.3)], np.repeat(np.eye(3)[None], 5, 0),
                 np.zeros((5, 7)), np.full(5, 0.01), np.full(5, 0.1))


def certified(req):
    return TraceResponse(req.request_id, "certified", timed=_timed(), end_index=0, standoff=0.0,
                         valid_while={"scene_version": 7, "tube_mm": 10, "holding": None, "start_q": np.zeros(7)},
                         stats={"stage": "fast_path"})


def result(kind, **kw):
    return lambda req: TraceResponse(req.request_id, kind, stats={"stage": "search"},
                                     **{k: (v(req) if callable(v) else v) for k, v in kw.items()})


class FakeHarness:
    def __init__(self, monkeypatch, script, exec_status=("done",), stage_terms=None):
        self.events, self.exec_status, self.n_reobserve, self.n_look = [], list(exec_status), 0, 0
        self.id_map, self.held, self.dropped_terms, self.frames, self.last_move = {"o1": 3, "o2": 5}, None, [], [], {}
        self.objects = {3: SimpleNamespace(views=[("head", "sam", 0.9)], held=False, category="can", name="can",
                                           footprint_center=lambda: np.zeros(2), top_z=0.1),
                        5: SimpleNamespace(views=[], held=False)}
        self.cur_stage = {"id": "s1", "terms": stage_terms if stage_terms is not None else []}
        self.rb = SimpleNamespace(eef_pose=lambda: (np.eye(3), np.array([0.5, 0, 0.3])), q=lambda: np.zeros(9),
                                  steps=0)
        self.checker = SimpleNamespace(attached=None, pg=np.zeros(0, int),
                                       world_points=lambda Q, f: (np.zeros((1, 0, 3)), None))
        self.scene = SimpleNamespace(table_z=0.0)
        self.rm = SimpleNamespace(root_p=np.zeros(3))
        self.detector = SimpleNamespace(guard=lambda reason, rule=None: None, last={})
        self.ex = SimpleNamespace(run_timed=self._run_timed)
        self.cm = None                                     # the controller model only reaches the (faked) executor
        self.svc = FakeService(script)
        self.tm = tracemotion.TraceMotion(self, mode="trace")
        self.tm._svc = self.svc
        snap = SimpleNamespace(version=7, entities={"o1": SimpleNamespace(label=3), "o2": SimpleNamespace(label=5)})
        sph = Spheres(np.zeros((1, 3)), np.full(1, 0.01), np.zeros(1, int))
        monkeypatch.setattr(tracemotion, "build_snapshot", lambda h: snap)
        monkeypatch.setattr(tracemotion, "hand_spheres", lambda rm, f: sph)
        monkeypatch.setattr(tracemotion, "make_arm_check", lambda *a: None)
        monkeypatch.setattr(tracemotion, "make_ik_check", lambda *a: None)
        monkeypatch.setattr(tracemotion, "StepGuard", lambda *a: None)

    def _run_timed(self, timed, **kw):
        return ExecResult(self.exec_status.pop(0), 5, {"first_step": 1, "last_step": 5, "ratio_max": 0.5, "step": 3})

    def log(self, event, **kw):
        self.events.append((event, kw))

    def reactions(self):
        return [kw["reaction"] for e, kw in self.events if e == "trace_reaction"]

    def carve_robot(self, Q=None):
        pass

    def _clear_held_ghost(self):
        pass

    def _tick(self, *a):
        pass

    def _rec(self, *a):
        pass

    def reobserve(self):
        self.n_reobserve += 1

    def look_from_above(self, point, height=0.3):
        self.n_look += 1
        return SimpleNamespace(name="robot0_eye_in_hand")

    def mask_matching(self, *a):
        return None

    def rebuild_map(self, frames):
        pass

    def nearest_obstacle(self, pts):
        return None

    def _drop_check(self, robot):
        return False


END = [EndCandidate(np.eye(3), np.array([0.6, 0.0, 0.3]))]
CLEAR = [{"prim": "clearance", "of": "o2", "level": "wide", "role": "require", "relax_to": "normal"}]


def test_trace_move_certified_and_no_path(monkeypatch):
    h = FakeHarness(monkeypatch, [certified])
    res = h.tm.move(END, "look", grip=0.0)
    assert res.status == "done" and h.svc.requests[0].mode == "straight" and h.reactions() == []
    assert [e for e, _ in h.events] == ["trace_request", "trace_result", "exec_timed"]
    assert h.last_move["status"] == "done"
    h = FakeHarness(monkeypatch, [result("no_path"), result("no_path")])
    assert h.tm.move(END, "look") is None
    assert [r.mode for r in h.svc.requests] == ["straight", "trace"] and h.reactions() == ["next_candidate"]
    assert h.last_move["status"] == "no_path"
    assert h.tm.summary()["results"] == {"no_path": 2}


def test_trace_move_conflict(monkeypatch):
    quote = lambda needed: (lambda req: [{"term": "lang0_clearance_o2", "prim": "clearance", "level": "wide",
                                          "relax_to": "normal", "param": "margin", "value": 0.08, "needed": needed,
                                          "unit": "m"}])
    h = FakeHarness(monkeypatch, [result("conflict", quote=quote(0.031))] * 2 + [certified], stage_terms=CLEAR)
    assert h.tm.move(END, "carry").status == "done" and h.reactions() == ["relax"]
    assert h.cur_stage["terms"][0]["level"] == "normal" and h.cur_stage["terms"][0]["relax_to"] is None
    lang = [t for t in h.svc.requests[-1].terms if t.src == "language"][0]
    assert lang.args["margin"] == LEVELS["clearance"]["normal"][0]
    h = FakeHarness(monkeypatch, [result("conflict", quote=quote(0.01))] * 2, stage_terms=[dict(CLEAR[0])])
    with pytest.raises(StageFailure) as e:
        h.tm.move(END, "carry")
    assert e.value.reason == "trace_conflict" and e.value.info["quote"][0]["needed"] == 0.01
    assert h.reactions() == ["repair"]
    # a retreat never calls the model: the caller falls back to move_line
    h = FakeHarness(monkeypatch, [result("conflict", quote=quote(0.01))] * 2, stage_terms=[dict(CLEAR[0])])
    assert h.tm.move(END, "lift", retreat=True) is None and h.reactions() == ["fallback"]
    # unknown space is always relaxable to slow guarded motion
    known = lambda req: [{"term": "known_palm", "prim": "clearance", "level": "strict", "relax_to": "slow",
                          "param": "margin", "value": 0.01, "needed": 0.0, "unit": "m"}]
    h = FakeHarness(monkeypatch, [result("conflict", quote=known)] * 2 + [certified])
    assert h.tm.move(END, "move").status == "done" and h.reactions() == ["relax"]
    kp = {t.id: t for t in h.svc.requests[-1].terms}["known_palm"]
    assert kp.role == "prefer" and kp.args["slow"] == levels.LIMITS["v_slow"]


def test_trace_move_look_program_error_defect(monkeypatch):
    look = {"entity": "o1", "term": "end_window", "uncertainty": 0.011, "tolerance": 0.01, "axis": "lateral"}
    h = FakeHarness(monkeypatch, [result("not_certifiable", look_at=look), certified])
    assert h.tm.move(END, "to_pregrasp").status == "done"
    assert h.n_look == 1 and h.reactions() == ["look"] and [r.mode for r in h.svc.requests] == ["straight", "straight"]
    h = FakeHarness(monkeypatch, [result("not_certifiable", look_at=look)] * 3)
    assert h.tm.move(END, "to_pregrasp") is None and h.n_look == 2
    assert h.reactions() == ["look", "look", "next_candidate"]
    h.objects[3].views.append(("robot0_eye_in_hand", "top", 0.8))            # already seen close: a look cannot help
    h.svc.script = [result("not_certifiable", look_at=look)]
    assert h.tm.move(END, "to_pregrasp") is None and h.n_look == 2
    # program_error: the offending language term leaves the stage and is reported at the next REPAIR
    terms = [dict(CLEAR[0]), {"prim": "clearance", "of": "o7", "level": "close", "role": "require", "relax_to": None}]
    find = lambda req: [{"test": "lint", "term": "lang1_clearance_o7", "detail": "unknown entity o7"}]
    h = FakeHarness(monkeypatch, [result("program_error", findings=find), certified], stage_terms=terms)
    assert h.tm.move(END, "carry").status == "done" and h.reactions() == ["drop_terms"]
    assert [t["of"] for t in h.cur_stage["terms"]] == ["o2"]
    assert h.dropped_terms[0]["of"] == "o7" and h.dropped_terms[0]["findings"][0]["test"] == "lint"
    # a stop on a certified path: defect logged, re-observe, re-request once
    h = FakeHarness(monkeypatch, [certified, certified], exec_status=("contact", "done"))
    assert h.tm.move(END, "carry").status == "done" and h.n_reobserve == 1
    d = [kw for e, kw in h.events if e == "guard_defect"][0]
    assert d["kind"] == "contact" and d["explanation"] == "geometry_disagreement" and d["scene_version"] == 7
    assert h.reactions() == ["reobserve_rerequest"] and h.tm.summary()["guard_stops"] == {"contact": 1}
    h = FakeHarness(monkeypatch, [certified, certified], exec_status=("tube_exit", "guard_stop"))
    assert h.tm.move(END, "carry").status == "guard_stop" and h.n_reobserve == 1


def test_harness_trace_move_drop_raises(monkeypatch):
    h = FakeHarness(monkeypatch, [certified], exec_status=("dropped",))
    with pytest.raises(StageFailure) as e:
        Harness.trace_move(h, END, "carry")
    assert e.value.reason == "dropped_in_transit"
    h = FakeHarness(monkeypatch, [result("no_path"), result("no_path")])
    assert Harness.trace_move(h, END, "look") is False
