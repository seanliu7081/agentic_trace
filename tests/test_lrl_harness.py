"""LRL_PLAN M1, harness side, without a simulator or a model: the flag and its parts, that lrl with no part runs
v2's code, the entry order of a push, the skill's stop in the verdict's words, what lrl_move does with a hold, an
early contact and a run that does not move the hand, and the runner's flags for stock tasks and stage truth.
(Candidates over faces and axes, the grip search's exemption, weights and tare: test_lrl_harness2.py.)"""
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atrace import articulate
from atrace.harness import Harness, StageFailure
from atrace.lrl import CFG, PARTS, LrlMixin, candidates, parse_parts, press_point, stroke_stop
from atrace.lrl_skill import Trace, _WIDTH

ROOT = Path(__file__).resolve().parents[1]


def test_parts_parsing():
    assert parse_parts("skill") == ("skill",) and parse_parts(" skill ,") == ("skill",)
    assert parse_parts("") == () and set(parse_parts(",".join(PARTS))) == set(PARTS)
    assert PARTS == ("skill", "policy", "detector") and parse_parts("skill,policy") == ("skill", "policy")
    for bad in ("skill,scorer_that_does_not_exist", "policy", "detector,policy"):
        with pytest.raises(ValueError):
            parse_parts(bad)


def test_flag_is_in_the_runner():
    text = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_episode.py"), "--help"], capture_output=True,
                          text=True, check=True).stdout
    assert "--lrl-parts" in text and "--lrl-weights" in text and "unified,lrl" in text


def test_setup_switches_parts_on_only_with_lrl():
    h = SimpleNamespace()
    LrlMixin.lrl_setup(h, "lrl", ("skill",), {"dir": "/some/weights"})
    assert h.lrl and h.lrl_parts == ("skill",) and h.lrl_weights == {"dir": "/some/weights", "policy": None,
                                                                    "detector": None}
    assert h.lrl_policy is None and h.lrl_detector is None
    LrlMixin.lrl_setup(h, "lrl")
    assert h.lrl and h.lrl_parts == ()
    LrlMixin.lrl_setup(h, "v2", ("skill",))
    assert not h.lrl and h.lrl_parts == ()
    with pytest.raises(AssertionError):
        LrlMixin.lrl_setup(h, "lrl", ("no_such_part",))


FACE = {"n": np.array([0.0, -1.0, 0.0]), "p0": np.zeros(3), "why_not": None, "source": "parent", "score": 1.0,
        "lift": 0.0, "axis_point": np.zeros(3), "protrusion": 0.0, "plane_inliers": 100}


class Bare(Harness):
    """A Harness with everything a push stage calls replaced by a recorder: which stroke code the stage goes to."""

    def __init__(self, placement, parts):
        self.placement, self.calls, self.max_wall_s, self.t0 = placement, [], 1e9, 0.0
        self.lrl_setup(placement, parts)
        self.checker = SimpleNamespace(free_near=None)

    log = lambda self, *a, **k: None
    elapsed = lambda self: 0.0
    _part_not_separated = lambda self, part, parent: False
    _face_hypotheses = lambda self, part, parent: [dict(FACE)]
    _look_at_face = lambda self, part, parent, mf: None
    _dir_words = lambda self, v: "image left"
    _contents_of = lambda self, part: []
    _observe_after_stroke = lambda self, *a, **k: {"measured": True, "closed": True, "front_m": 0.0, "face_m": 0.0,
                                                   "offset_m": 0.0, "face_shift_m": 0.1}

    def _stroke(self, name):
        self.calls.append(name)
        return {"mode": "press", "travel": 0.1, "stop": "force_limit", "d": -FACE["n"],
                "disp": -0.1 * FACE["n"], "point": np.zeros(3)}

    def plan_part_grasp(self, *a, **k):
        self.calls.append("grip_search")

    def _press_stroke(self, part, parent, mf, failed, result, **k):
        result["attempts"].append({"status": "press_push"})
        return self._stroke("press_stroke")

    def lrl_move(self, motion, part, parent, mf, failed, result, *a):
        result["attempts"].append({"status": "press_push"})
        self._lrl_stage(result)["entered"] += 1
        return {**self._stroke("lrl_move"), "label": "reached", "family": "tip", "normal": FACE["n"]}


def _push(placement, parts, motion="push"):
    h = Bare(placement, parts)
    part = SimpleNamespace(points=np.zeros((50, 3)), oid=3, part_of=None, name="drawer")
    result = {"stage": "s1", "attempts": []}
    out = h._actuate_linear({"goal": "closed"}, part, None, motion, result)
    assert out["relation_check"]["satisfied"]
    return h.calls


def test_lrl_without_parts_dispatches_as_v2():
    assert _push("v2", ()) == _push("lrl", ()) == ["grip_search", "press_stroke"]
    assert _push("lrl", ("skill",)) == ["lrl_move"]
    assert _push("v2", ("skill",)) == ["grip_search", "press_stroke"]         # parts mean nothing outside lrl


def test_pull_goes_through_the_skill_under_lrl():
    h = Bare("lrl", ("skill",))
    result = {"stage": "s1", "attempts": []}
    part = SimpleNamespace(points=np.zeros((50, 3)), oid=3, part_of=None, name="drawer")
    with pytest.raises(StageFailure):       # the fake reading says closed after a pull to open: the stage fails ...
        h._actuate_linear({"goal": "open"}, part, None, "pull", result)
    assert set(h.calls) == {"lrl_move"} and len(h.calls) == CFG["candidates"]      # ... after 8 lrl candidates
    assert _push("lrl", ("skill",), "pull") == ["lrl_move"]


def test_put_down_of_lrl_is_v2s():
    seen = []
    h = SimpleNamespace(placement="lrl", place_object_v2=lambda *a: seen.append(a) or "v2")
    assert Harness.place_object(h, "obj", "dest", {}, None, {}) == "v2" and len(seen) == 1


def test_candidate_order():
    n = np.array([0.0, -1.0, 0.0])                       # an upright face: every family has poses
    cands = candidates(np.array([0.5, 0.2, 0.9]), n)
    fams = [c[0] for c in cands]
    assert CFG["order"] == ("tip", "tip_lean", "tip_steep", "side")
    assert sorted(set(fams), key=fams.index) == list(CFG["order"])           # family blocks, in the measured order
    assert [fams.count(f) for f in CFG["order"]] == [28, 56, 56, 18]
    down = np.array([0.0, 0.0, -1.0])
    for fam, mode, a, R, p, c in cands:
        tilt = np.degrees(np.arccos(np.clip(a @ -n, -1, 1)))
        assert np.allclose(R[:, 2], a) and np.isclose(np.linalg.det(R), 1.0)
        if fam == "side":
            assert mode == "side" and np.allclose(a, down)
        else:
            assert mode == "tip" and any(np.isclose(tilt, t) for t in {"tip": (0.0,), "tip_lean": CFG["lean"],
                                                                      "tip_steep": CFG["steep"]}[fam])
            assert np.isclose(np.linalg.norm(p - c), 0.0092 + CFG["tip_gap"], atol=5e-3)   # fingertips at the face
    # within a family the centre point comes first; a failed point is left out for every family
    assert np.allclose(cands[0][5], [0.5, 0.2, 0.9])
    left = candidates(np.array([0.5, 0.2, 0.9]), n, failed=[np.array([0.5, 0.2, 0.9])])
    assert all(np.linalg.norm(c[5] - [0.5, 0.2, 0.9]) > CFG["failed_radius"] for c in left) and len(left) < len(cands)
    # a level face has fingertips straight down only
    assert {c[0] for c in candidates(np.zeros(3), np.array([0.0, 0.0, 1.0]))} == {"tip"}


def test_press_point_is_the_middle_of_the_outer_face():
    g = np.stack(np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(0.0, 0.3, 16), np.linspace(0.8, 0.9, 11)),
                 -1).reshape(-1, 3)                       # a box whose face y = 0 looks along -y
    c0, win = press_point(g, np.array([0.0, -1.0, 0.0]))
    assert np.allclose(c0, [0.0, 0.0, 0.85], atol=0.012) and 0.03 < win < 0.06


def _trace(stop, moved):
    tr = Trace(np.eye(3), np.zeros(3))
    for k in range(40):
        row = np.zeros(_WIDTH)
        row[0] = moved * k
        tr.rows.append(row)
    tr.stop = stop
    return tr


def test_stop_words_for_the_verdict():
    still, moving = articulate.STALL_DISP / 100, articulate.STALL_DISP
    assert stroke_stop(_trace("force", moving)) == "force_limit"
    assert stroke_stop(_trace("time", still)) == "stalled"
    assert stroke_stop(_trace("time", moving)) == "time" and stroke_stop(_trace("guard", still)) == "guard"
    verdict = lambda stop: Harness._linear_verdict(None, "closed", {"closed": True}, {"travel": 0.1, "stop": stop})
    assert verdict("force_limit") == verdict("stalled") == "done" and verdict("time") == verdict("guard") == "again"


# ------------------------------------------------------------------------------------------- lrl_push
N = np.array([0.0, -1.0, 0.0])
TOUCH = np.array([0.5, 0.2, 0.9])


class Push(LrlMixin):
    """lrl_move on a scripted hand: `hold` is what the grip search returns, `ends` what each transit returns (an end,
    or None with the hand left `short` metres before the touch after a stop of kind `stop`), `hand` the hand travel
    of each skill run."""

    def __init__(self, hold=False, ends=(True,), stop="contact", short=0.02, hand=(0.08, 0.001), gap=0.02):
        self.hold, self.ends, self.stop, self.short, self.hand, self.gap = hold, list(ends), stop, short, list(hand), gap
        self.events, self.moves, self.runs, self.p = [], [], [], TOUCH + 0.3 * N
        self.tm = SimpleNamespace(counts={"stops": Counter()})
        self.objects = {3: SimpleNamespace(name="drawer")}
        self.last_move = {}
        self.ex = SimpleNamespace(gripper=lambda *a, **k: self.moves.append("close"),
                                  gripper_to=lambda w, **k: self.moves.append("open_to"))
        self.rb = SimpleNamespace(eef_pose=lambda: (np.eye(3), self.p), finger_opening=lambda: self.gap,
                                  wrench=lambda: (np.zeros(3), np.zeros(3)))
        self.lrl_detector = self.lrl_policy = None

    log = lambda self, event, **kw: self.events.append((event, kw))
    _fixture = lambda self, part, obj: ((), np.zeros((0, 3)))
    _closing_end_check = lambda self, *a: None
    _contents_of = lambda self, part: []
    _same_face = staticmethod(Harness._same_face)
    _state_words = lambda self, state: "open"
    _ik_near = lambda self, R, p, q: q
    _guard = lambda self, reason, rule: rule
    _lrl_look = lambda self, *a: {"front_m": 0.05, "closed": False}

    def _lrl_grip(self, part, parent, mf, failed, motion, allowed, own, marked, end_check=None):
        assert motion == "push" and allowed == (3,)
        g = SimpleNamespace(R=np.eye(3), p=TOUCH, width=0.02, info={"finger": 0.02})
        return ((g, np.zeros(7), np.zeros(7)), {}) if self.hold else (None, {"no_hold": 1})

    def _lrl_entry(self, part, parent, n, c0, failed, marked, allowed, own):
        return ("tip", "tip", -N, np.eye(3), TOUCH, TOUCH, np.zeros(7)), {}, {}

    def reach(self, R, p, q, a, oid, win, tag, grip, finger, q_pre):
        self.moves.append(tag)
        if self.ends.pop(0):
            self.p = p - a * 0.02
            return SimpleNamespace(R=R, p=p)
        self.tm.counts["stops"][self.stop] += 1
        self.last_move = {"status": self.stop}
        self.p = p - a * self.short

    def move_line(self, R, p, **kw):
        self.moves.append("line")

    def retreat(self, R, p, **kw):
        self.moves.append(kw["tag"])

    def _lrl_run(self, contract, moving, own, stage, tare=None, max_steps=None, policy=False):
        self.runs.append(contract)
        went = self.hand.pop(0)
        tr = _trace("time", went / 40)
        return tr, {"file": "x", "stop": "time", "steps": 40, "hand_travel_m": went, "hand_end_m": went,
                    "peak_force_n": 5.0}, {}


def _lrl_push(h, failed=None):
    part = SimpleNamespace(points=np.random.default_rng(0).uniform(-0.05, 0.05, (60, 3)) + TOUCH, oid=3)
    result = {"stage": "s1", "attempts": []}
    return h.lrl_move("push", part, None, dict(FACE), [] if failed is None else failed, result, "closed", 0.1,
                      TOUCH + 0.1 * N, TOUCH), result


def test_a_hold_comes_first_and_is_a_grip_and_push():
    assert CFG["kinds"] == ("grip", "press")
    h = Push(hold=True)
    stroke, result = _lrl_push(h)
    entry = dict(h.events)["lrl_entry"]
    assert entry["family"] == "grip" and entry["passed"] == {}
    goal = h.runs[0].goal
    assert goal.mode == "pull" and np.allclose(goal.direction, -N) and h.runs[0].toucher == "pads"   # close, then lead
    assert h.moves == ["open_to", "to_part", "line", "open_to", "release_back"]
    assert stroke["mode"] == "grip" and result["attempts"][-1]["pose"] == "grip"
    # pads that closed on nothing: the stroke was a press with closed fingers
    assert _lrl_push(Push(hold=True, gap=0.0))[0]["mode"] == "press"
    # no hold: the press candidates, and the grip search is not repeated on this face while the part stands
    h = Push(hold=False)
    stroke, result = _lrl_push(h)
    entry = dict(h.events)["lrl_entry"]
    assert entry["family"] == "tip" and "grip" in entry["passed"] and h.runs[0].goal.mode == "push"
    assert len(h._lrl_mem["no_hold"]) == 1 and entry["passed"]["grip"] == {"no_hold": 1}
    assert h.moves == ["close", "to_press", "line", "press_back", "close"]


def test_contact_next_to_the_touch_is_an_arrival():
    h = Push(ends=(None,), short=0.02)
    failed = []
    stroke, result = _lrl_push(h, failed)
    arrival = dict(h.events)["lrl_arrival"]
    assert arrival["arrived"] and stroke is not None and not failed and len(h.runs) >= 1
    assert "line" not in h.moves                        # the skill starts from where the hand is
    # too far from the touch, or a stop that is not a contact: the transit failed as before
    for kw in ({"short": CFG["standoff"] + CFG["arrive"] + 0.01}, {"stop": "guard_stop"}):
        h, failed = Push(ends=(None,), **kw), []
        stroke, result = _lrl_push(h, failed)
        assert stroke is None and len(failed) == 1 and not h.runs
        assert result["attempts"][-1]["status"] == "transit_failed"


def test_no_second_run_when_the_hand_stood_still():
    def moving(h):                                      # the in-place reading says the part moved in every run
        h._lrl_look = lambda *a, k=iter((-0.15, -0.17, -0.19)): {"front_m": next(k), "closed": False}
        return h
    h = moving(Push(hand=(0.001, 0.08)))
    _lrl_push(h)
    assert len(h.runs) == 1                             # under CFG["again_hand"]: on to the next candidate
    h = moving(Push(hand=(0.08, 0.001, 0.08)))
    _lrl_push(h)
    assert len(h.runs) == 2                             # run 0 moved hand and part: once more in place, then no more


# ------------------------------------------------------------------------------------------- runner
def _runner(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run_episode.py"), *args], capture_output=True,
                          text=True)


def test_runner_has_stock_tasks_and_refuses_truth_on_test_tasks():
    text = _runner("--help").stdout
    assert all(flag in text for flag in ("--suite", "--suite-root", "--init", "--stage-truth"))
    for args in (("--task", "t03", "--episode", "0", "--stage-truth"),
                 ("--task", "KITCHEN_SCENE3_turn_on_the_stove", "--suite", "libero_90", "--stage-truth"),
                 ("--task", "t03")):
        r = _runner(*args)
        assert r.returncode == 2 and "error" in r.stderr


def test_dev_tasks_and_their_files():
    sys.path.insert(0, str(ROOT / "scripts"))
    import stage_truth
    assert stage_truth.is_dev("libero_90", "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet")
    assert stage_truth.is_dev("libero_goal", "turn_on_the_stove")
    assert not stage_truth.is_dev("libero_90", "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet")
    assert not stage_truth.is_dev("libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it")
    # beside the dev tasks, libero_90 holds the 5 held-out part-moving tasks of the M1 evaluation (LRL_PLAN.md M1)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_episode import HELDOUT_SCENES
    for suite, n, held in (("libero_90", 20, 5), ("libero_goal", 10, 0)):
        bddl = sorted(p.stem for p in (ROOT / "suites" / "bddl_files" / suite).glob("*.bddl"))
        init = sorted(p.stem for p in (ROOT / "suites" / "init_files" / suite).glob("*.pruned_init"))
        dev = [t for t in bddl if stage_truth.is_dev(suite, t)]
        assert bddl == init and len(dev) == n and len(bddl) - n == held
        assert all(t.startswith(HELDOUT_SCENES) for t in bddl if t not in dev)


def test_truth_probe_and_the_rules_of_the_error_script():
    import mujoco
    sys.path.insert(0, str(ROOT / "scripts"))
    import handoff_error
    import stage_truth
    # a drawer front 2 cm thick at y = 0.3 .. 0.32 with a handle sphere in front of it, on a slide along -y
    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body name="drawer" pos="0 0.31 0.5">
        <joint name="slide" type="slide" axis="0 -1 0" range="-0.16 0.01"/>
        <geom type="box" size="0.1 0.01 0.05"/><geom type="sphere" size="0.01" pos="0 -0.03 0"/>
        </body></worldbody></mujoco>""")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    axis = np.array([0.0, -1.0, 0.0])
    geoms = stage_truth._subtree(m, 1)
    assert len(geoms) == 2
    p = stage_truth._probe(m, d, geoms, axis, np.array([0.0, 0.25, 0.5]))      # 2 cm in front of the handle
    assert np.isclose(p["front"], -0.02, atol=1e-3) and np.isclose(p["back"], -0.07, atol=1e-3)
    assert np.isclose(p["near_m"], 0.02, atol=2e-3) and np.isclose(p["hi"], -0.02, atol=1e-3)
    off = stage_truth._probe(m, d, geoms, axis, np.array([0.05, 0.25, 0.5]))     # beside the handle: the panel
    assert np.isclose(off["front"], -0.05, atol=1e-3)
    assert stage_truth._probe(m, d, geoms, axis, np.array([0.3, 0.25, 0.5]))["front"] is None      # misses the part
    fix = {"default_open_ranges": [-0.16, -0.14], "default_close_ranges": [0.0, 0.005]}
    assert handoff_error.holds(0.001, "close", fix) and not handoff_error.holds(-0.002, "close", fix)
    assert handoff_error.holds(-0.15, "open", fix) and not handoff_error.holds(-0.10, "open", fix)
    assert handoff_error.holds(0.6, "turnon", {"default_turnon_ranges": [0.5, 2.1]})
    assert np.isclose(handoff_error.opening(-0.15, fix), 0.15) and handoff_error.opening(0.004, fix) == 0.0
    assert handoff_error.opening(0.3, {"default_open_ranges": [], "default_close_ranges": []}) is None
