"""LRL_PLAN M1 second half, harness side, without a simulator or a model (fakes for the policy, the detector and the
robot): candidates over faces and axes and their cycling on no_effect, near-duplicates, the grip search's exemption
of what the contract allows, the interim labels, the parts and the weights' resolution from a manifest, the tare of a
run in place on a loaded hand."""
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from atrace import lrl_models
from atrace.harness import Harness, StageFailure
from atrace.lrl import (CFG, LrlMixin, build_suffix, candidates, interim_label, is_marked, load_weights,
                        parse_parts, weights_record)
from atrace.lrl_skill import CFG as SKILL, Trace, _WIDTH

N = np.array([0.0, -1.0, 0.0])
TOUCH = np.array([0.5, 0.2, 0.9])


def face(n, why_not=None, p0=(0.0, 0.0, 0.0)):
    return {"n": np.asarray(n, float), "p0": np.asarray(p0, float), "why_not": why_not, "source": "parent",
            "score": 1.0, "lift": 0.0, "axis_point": np.zeros(3), "protrusion": 0.0, "plane_inliers": 100}


# ------------------------------------------------------------------------------- candidates over faces
class Stage(Harness):
    """A push / pull stage of Harness._actuate_linear with lrl_move scripted: labels[i] is the label of the i-th
    candidate; what the views read after a stroke is `closed` (one value per observation)."""

    def __init__(self, faces, labels, closed=(True,)):
        self.placement, self.max_wall_s, self.t0 = "lrl", 1e9, 0.0
        self.lrl_setup("lrl", ("skill",))
        self.checker = SimpleNamespace(free_near=None)
        self.faces, self.labels, self.closed = faces, list(labels), list(closed)
        self.used, self.observed, self.events, self.effect_checks = [], 0, [], 0

    def log(self, event, **kw):
        self.events.append((event, kw))

    elapsed = lambda self: 0.0
    _part_not_separated = lambda self, part, parent: False
    _look_at_face = lambda self, part, parent, mf: None
    _dir_words = lambda self, v: "image left"
    _contents_of = lambda self, part: []
    _closing_end_check = lambda self, *a: None

    def _face_hypotheses(self, part, parent):
        return [dict(f) for f in self.faces]

    def _effect_after_stroke(self, *a, **k):
        self.effect_checks += 1
        return {}

    def _observe_after_stroke(self, *a, **k):
        self.observed += 1
        c = self.closed.pop(0) if len(self.closed) > 1 else self.closed[0]
        return {"measured": True, "closed": c, "front_m": 0.0, "face_m": 0.0, "offset_m": 0.0 if c else 0.05,
                "face_shift_m": 0.02}

    def lrl_move(self, motion, part, parent, mf, failed, result, *a):
        self._lrl_stage(result)["entered"] += 1
        self.used.append(np.round(mf["n"], 3).tolist())
        label = self.labels.pop(0) if self.labels else "no_effect"
        stroke = {"mode": "press", "travel": 0.05, "stop": "force_limit", "d": -mf["n"], "disp": -0.05 * mf["n"],
                  "point": TOUCH, "family": "tip", "normal": mf["n"], "label": label}
        result["attempts"].append({"status": "press_push", "label": label})
        if label in CFG["marking"]:
            self.lrl_mark(result, stroke, label, scope="face" if label == "no_effect" else "candidate")
        return stroke


def _stage(h, motion="push", goal="closed"):
    part = SimpleNamespace(points=np.zeros((50, 3)), oid=3, part_of=None, name="drawer")
    result = {"stage": "s1", "attempts": []}
    return h._actuate_linear({"goal": goal}, part, None, motion, result), result


def test_no_effect_takes_the_next_face_without_a_look_or_a_model_call():
    up, front = [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]
    # the upward face first (v2's order), the front face vetoed by v2 but a candidate on the lrl path
    h = Stage([face(up), face(front, why_not="approach_side_solid")], ["no_effect", "reached"])
    out, result = _stage(h)
    assert out["relation_check"]["satisfied"] and h.used == [up, front]
    assert h.observed == 1 and h.effect_checks == 0           # the marked candidate was neither observed nor judged
    marks = [kw for e, kw in h.events if e == "lrl_candidate_marked"]
    assert len(marks) == 1 and marks[0]["label"] == "no_effect" and marks[0]["scope"] == "face"


def test_plausible_faces_come_first_and_the_stage_is_bounded():
    faces = [face(np.r_[np.cos(a), np.sin(a), 0.0], why_not="faces_down" if i < 2 else None, p0=(0, 0, 0.1 * i))
             for i, a in enumerate(np.linspace(0, 2 * np.pi, 12, endpoint=False))]
    h = Stage(faces, [])
    with pytest.raises(StageFailure):
        _stage(h, "pull", "open")
    assert len(h.used) == CFG["candidates"] == 8                       # every candidate said no_effect
    assert h.used[0] == np.round(faces[2]["n"], 3).tolist()             # the vetoed faces only after the others


def test_reached_that_v2_does_not_confirm_is_marked_and_the_next_candidate_taken():
    h = Stage([face([0.0, -1.0, 0.0])], ["reached", "reached"], closed=(False, True))
    out, result = _stage(h)
    assert out["relation_check"]["satisfied"] and h.observed == 2 and len(h.used) == 2
    dis = [kw for e, kw in h.events if e == "lrl_disagree"]
    assert len(dis) == 1 and dis[0]["v2"] == "again"
    assert [kw["why"] for e, kw in h.events if e == "lrl_candidate_marked"] == ["reached, v2's verdict again"]


def test_near_duplicates_of_a_marked_candidate_are_skipped():
    c0 = TOUCH
    marked = [("tip", c0, N)]
    left = candidates(c0, N, marked=marked)
    assert all(not (k[0] == "tip" and np.linalg.norm(k[5] - c0) < CFG["dup_radius"]) for k in left)
    assert any(k[0] == "tip" for k in left) and any(k[0] == "tip_lean" and np.allclose(k[5], c0) for k in left)
    assert is_marked(marked, "tip", c0 + [0.01, 0, 0], N) and not is_marked(marked, "tip", c0 + [0.03, 0, 0], N)
    assert not is_marked(marked, "tip", c0, np.array([0.0, 0.0, 1.0])) and not is_marked(marked, "side", c0, N)


def test_interim_labels():
    lab = interim_label
    assert lab("push", "guard", 0.1, True) == "outside"
    assert lab("push", "time", 0.05, True) == "reached" and lab("push", "time", 0.05, False) == "short"
    assert lab("push", "time", None) == "short"                       # not read: v2's observation decides
    assert lab("push", "force", 0.001, None, peak_n=30.0) == "no_effect"
    assert lab("push", "time", 0.001, None, peak_n=2.0) == "no_contact"
    assert lab("pull", "time", 0.1, True, gap=0.0) == "no_contact"
    assert lab("turn", "time", gap=0.02, turned=np.radians(40)) == "reached"
    assert lab("turn", "force", gap=0.02, turned=np.radians(3)) == "no_effect"
    assert lab("turn", "time", gap=0.0, turned=np.radians(40)) == "lost_grip"


# ------------------------------------------------------------------------------- the grip search's exemption
PART, FIXTURE, OTHER = 3, 7, 9


class Clearance:
    """A checker whose observed geometry is the part and the fixture (touching the hand) and, when `other`, another
    object as close: clearance 0 to whatever is not excluded and not freed."""

    def __init__(self, other=False):
        self.other, self.free_near = other, None
        self.pg, self.pr = np.array([0, 1, 2]), np.full(3, 0.005)

    def _near(self, exclude):
        touching = {PART, FIXTURE} | ({OTHER} if self.other else set())
        return 0.0 if touching - set(exclude) else 1.0

    def point_clearance(self, W, exclude=(), **k):
        return np.full(len(W), self._near(exclude))

    def clearance(self, Q, exclude=(), finger=0.0, **k):
        c = np.full(len(Q), self._near(exclude))
        return {g: c for g in ("arm", "palm", "finger")}

    def world_points(self, Q, finger):
        return np.zeros((len(Q), 3, 3)), None


class Grip(LrlMixin):
    def __init__(self, other=False):
        self.checker, self.table_z, self.events = Clearance(other), 0.0, []

    log = lambda self, event, **kw: self.events.append((event, kw))
    _hand_local = lambda self, f: (np.array([[0, 0, 0.0], [0.0, 0.0, 0.01]]), np.array([1, 2]), np.full(2, 0.005))
    nearest_obstacle = lambda self, pts, excl: {"nearest": "something"}


def test_grip_search_exempts_the_part_and_its_fixture_only():
    g = SimpleNamespace(R=np.eye(3), p=np.array([0.5, 0.0, 0.3]), width=0.02, info={})
    allowed, v2 = (PART, FIXTURE), (PART,)
    assert Grip()._lrl_hand_bad(g, allowed, [], {}) is None and "finger" in g.info
    assert Grip()._lrl_hand_bad(g, v2, [], {}) == "hand_scene"         # v2's veto: the fixture
    assert Grip(other=True)._lrl_hand_bad(g, allowed, [], {}) == "hand_scene"         # another object stays one
    low = SimpleNamespace(R=np.eye(3), p=np.array([0.5, 0.0, 0.004]), width=0.02, info={})
    assert Grip()._lrl_hand_bad(low, allowed, [], {}) == "hand_table"                # and so does the table
    q = np.zeros((1, 7))
    assert Grip()._lrl_meets(q, (PART, FIXTURE), allowed, []) is None
    assert Grip()._lrl_meets(q, (PART,), allowed, [])[0] == "arm"       # the arm keeps v2's exclusion
    assert Grip(other=True)._lrl_meets(q, (PART, FIXTURE), allowed, [])[0] == "arm"


def test_grip_search_runs_v2s_order_with_the_exemption(monkeypatch):
    from atrace import articulate
    gs = [SimpleNamespace(R=np.eye(3), p=np.array([0.5, 0.0, 0.3 + 0.03 * i]), width=0.02, score=1.0,
                          info={"tilt": 0.0, "pitch": 0.0, "yaw": 0.0}) for i in range(3)]
    monkeypatch.setattr(articulate, "part_grasps", lambda part, mf, R0, max_depth=None: gs)
    h = Grip()
    h.rb = SimpleNamespace(eef_pose=lambda: (np.eye(3), np.zeros(3)), q=lambda: np.zeros(9))
    h.ik_clear = lambda R, p, q, exclude=(), finger=0.0: np.zeros((1, 7))
    h._ik_near = lambda R, p, q: q
    h.line_configs = lambda R, p, q, finger=0.0: [q, q, q]
    part, parent = SimpleNamespace(oid=PART), SimpleNamespace(oid=FIXTURE)
    mf = face([0.0, -1.0, 0.0])
    plan, why = h._lrl_grip(part, parent, mf, [gs[0].p], "pull", (PART, FIXTURE), [], [("grip", gs[1].p, mf["n"])])
    assert plan[0] is gs[2]                       # the failed one and the marked one are skipped
    plan, why = h._lrl_grip(part, parent, mf, [], "pull", (PART,), [], [])
    assert plan is None and why["hand_scene"] == 3
    pre = [kw for e, kw in h.events if e == "part_grasp_prefilter"][-1]
    assert pre["blame"]["hand_scene"]["nearest"] == "something"


# ------------------------------------------------------------------------------- parts and weights
def _manifest(tmp_path, entries):
    for name in entries:
        (tmp_path / name).write_bytes(name.encode())
    (tmp_path / "manifest.json").write_text(json.dumps(entries))


def _fake_loaders(monkeypatch):
    load = lambda kind: lambda path: SimpleNamespace(path=str(path), sha256=(kind[0] * 8) + str(path)[-12:-3],
                                                    modes=("push", "pull"), threshold={"reached": 0.9}, kind=kind)
    monkeypatch.setattr(lrl_models, "load_policy", load("p"))
    monkeypatch.setattr(lrl_models, "load_detector", load("d"))


def test_weights_from_a_manifest(tmp_path, monkeypatch):
    _fake_loaders(monkeypatch)
    _manifest(tmp_path, {
        "move_part_a.pt": {"kind": "move_part", "part": "skill", "m1": True, "published": "2026-11-25T12:00"},
        "move_part_b.pt": {"kind": "move_part", "part": "skill", "m1": True, "published": "2026-11-25T18:00"},
        "move_part_c.pt": {"kind": "move_part", "part": "skill", "published": "2026-11-25T19:00"},
        "detector_d.pt": {"kind": "move_part", "part": "detector", "m1": True, "published": "2026-11-25T18:00"}})
    w = load_weights(tmp_path, parse_parts("skill,policy,detector"))
    assert w["policy"].path.endswith("move_part_b.pt") and w["detector"].path.endswith("detector_d.pt")
    rec = weights_record(w)
    assert rec["policy"]["sha256"] == w["policy"].sha256 and rec["detector"]["threshold"] == {"reached": 0.9}
    assert build_suffix(w) == f"+p{w['policy'].sha256[:8]}+d{w['detector'].sha256[:8]}"
    assert load_weights(tmp_path, ("skill",)) == {"dir": str(tmp_path), "policy": None, "detector": None}
    assert build_suffix(load_weights(tmp_path, ("skill", "policy"))).count("+") == 1
    h = SimpleNamespace()
    LrlMixin.lrl_setup(h, "lrl", ("skill", "policy"), w)
    assert h.lrl_policy is w["policy"] and h.lrl_detector is None      # loaded but switched off: not used
    with pytest.raises(AssertionError):
        LrlMixin.lrl_setup(h, "lrl", ("skill", "policy"), None)
    _manifest(tmp_path, {"move_part_a.pt": {"kind": "move_part", "part": "skill", "m1": True}})
    with pytest.raises(FileNotFoundError):
        load_weights(tmp_path, ("skill", "detector"))


# ------------------------------------------------------------------------------- tare of a run in place
def _trace(n=40, w=np.zeros(3), dp=0.0):
    tr = Trace(np.eye(3), np.zeros(3))
    for k in range(n):
        row = np.zeros(_WIDTH)
        row[1] = dp * k
        row[9:12] = w                                    # angular velocity (p, rv, v, w, ...)
        tr.rows.append(row)
    tr.stop = "time"
    return tr


class Detector:
    def __init__(self, labels, travel=0.03):
        self.labels, self.travel, self.calls = list(labels), travel, []

    def __call__(self, trace, contract):
        self.calls.append((sorted(trace), contract.goal.mode))
        label = self.labels.pop(0)
        return {"label": label, "probs": {c: float(c == label) for c in lrl_models.DETECTOR_CLASSES},
                "travel": self.travel, "travel_sd": 0.01}


class Hand(LrlMixin):
    """lrl_move / the turn's senses on a scripted hand whose wrist reads LOADED once the hand has touched."""
    FREE, LOADED = (np.array([1.0, 2.0, 3.0]), np.zeros(3)), (np.array([9.0, 9.0, 9.0]), np.ones(3))

    def __init__(self, detector, traces=None):
        self.lrl_detector, self.lrl_policy, self.events, self.tares, self.steps = detector, None, [], [], []
        self.loaded, self.traces = False, list(traces or [])
        self.objects = {3: SimpleNamespace(name="drawer")}
        self.tm = SimpleNamespace(counts={"stops": Counter()})
        self.ex = SimpleNamespace(gripper=lambda *a, **k: None, gripper_to=lambda *a, **k: None)
        self.rb = SimpleNamespace(eef_pose=lambda: (np.eye(3), TOUCH.copy()), finger_opening=lambda: 0.02,
                                  wrench=lambda: self.LOADED if self.loaded else self.FREE)

    log = lambda self, event, **kw: self.events.append((event, kw))
    _fixture = lambda self, part, obj: ((), np.zeros((0, 3)))
    _closing_end_check = lambda self, *a: None
    _contents_of = lambda self, part: []
    _same_face = staticmethod(Harness._same_face)
    _ik_near = lambda self, R, p, q: q
    _guard = lambda self, reason, rule: rule
    retreat = lambda self, *a, **k: None

    def _lrl_grip(self, part, parent, mf, failed, motion, allowed, own, marked, end_check=None):
        g = SimpleNamespace(R=np.eye(3), p=TOUCH, width=0.02, info={"finger": 0.02})
        return (g, np.zeros(7), np.zeros(7)), {}

    def reach(self, *a):
        return SimpleNamespace(R=np.eye(3), p=TOUCH)

    def move_line(self, R, p, **kw):
        self.loaded = True                               # the pads around the handle, a press on the face

    def _lrl_run(self, contract, moving, own, stage, tare=None, max_steps=None, policy=False):
        self.tares.append(tare)
        self.steps.append(max_steps)
        tr = self.traces.pop(0) if self.traces else _trace(dp=0.001)
        return tr, {"file": "x", "stop": tr.stop, "steps": len(tr.rows), "hand_travel_m": 0.05, "hand_end_m": 0.05,
                    "peak_force_n": 20.0}, {}


def test_a_run_in_place_tares_against_the_wrist_before_the_hand_was_loaded():
    h = Hand(Detector(["short", "reached"]))
    part = SimpleNamespace(points=np.random.default_rng(0).uniform(-0.05, 0.05, (60, 3)) + TOUCH, oid=3)
    result = {"stage": "s1", "attempts": []}
    stroke = h.lrl_move("pull", part, None, face(-N), [], result, "open", None, None, None)
    assert stroke["label"] == "reached" and stroke["mode"] == "grip" and len(h.tares) == 2
    assert h.tares[0] is None and np.allclose(np.hstack(h.tares[1]), np.hstack(Hand.FREE))
    runs = [kw for e, kw in h.events if e == "lrl_skill"]
    assert [r["label"] for r in runs] == ["short", "reached"] and runs[0]["probs"]["short"] == 1.0
    assert all("touch" in keys and mode == "pull" for keys, mode in h.lrl_detector.calls)
    # a label that marks: no run in place, the candidate is marked, no observation needed
    h = Hand(Detector(["no_contact"]))
    stroke = h.lrl_move("pull", part, None, face(-N), [], {"stage": "s1", "attempts": []}, "open", None, None, None)
    assert stroke["label"] == "no_contact" and len(h.tares) == 1
    assert [kw["scope"] for e, kw in h.events if e == "lrl_candidate_marked"] == ["candidate"]


def test_turn_senses_run_in_place_with_the_tare_and_the_base_law():
    n, ap = np.array([0.0, 0.0, 1.0]), TOUCH - [0.0, 0.0, 0.03]
    rate = np.radians(40.0) / (40 * SKILL["dt"])                 # 40 degrees over the 40 rows
    h = Hand(Detector(["no_effect", "reached"]), traces=[_trace(), _trace(w=-n * rate)])
    g = SimpleNamespace(R=np.eye(3), p=TOUCH)
    result = {"stage": "s1", "attempts": []}
    moved, eff = h._lrl_turn_senses({}, result, (3,), [], n, ap, g, [1.0, -1.0], Hand.FREE, None, None)
    assert eff is None and moved["sense"] == "clockwise" and abs(moved["rotation_deg"] + 40.0) < 0.5
    assert h.tares[0] is None and h.tares[1] is Hand.FREE
    base = SKILL["close_steps"] + int(np.ceil(CFG["turn_travel"] / np.radians(SKILL["turn_rate"]) / SKILL["dt"]))
    assert h.steps == [base, base] and base == 72
    marks = [kw for e, kw in h.events if e == "lrl_candidate_marked"]
    assert len(marks) == 1 and marks[0]["scope"] == "axis" and np.allclose(marks[0]["normal"], n)
    assert h._lrl_stage(result)["entered"] == 1                  # the second sense counted as a candidate
