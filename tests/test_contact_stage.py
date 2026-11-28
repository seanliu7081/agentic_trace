"""CONTACT_STAGE_PLAN M1-M3: the shared end test, the slide-back hand-off, leg verdicts, the fill rule, wrist-image
pointing, re-localisation of a known shape, the hand-off reasons."""
from types import SimpleNamespace

import numpy as np
import pytest

from traceplan.contract import EndCandidate, GROUPS
from traceplan.search import end_test
from traceplan.service import TraceService
from traceplan.synthetic import R_DOWN, box_scene, default_hand, make_request, standard_scenes, trivial_arm_check

from atrace.contact import ContactMixin, Outcome, judge
from atrace.perception import FREE, OCC, UNKNOWN, ObjectModel, SceneMap
from atrace.program import OBJECT_SCHEMA, validate_program

BLOCK = {"id": "block", "center": [0.5, 0.0, 0.05], "half": [0.04, 0.04, 0.05]}


def test_end_test_observed_only_reports_unseen():
    unseen = {"center": [0.5, 0.2, 0.1], "half": [0.05, 0.05, 0.1]}
    snap = box_scene(boxes=[BLOCK], unknown=[unseen], table_z=0.0)
    hand = default_hand()
    blocked = end_test(snap, hand, R_DOWN, np.array([0.5, 0.0, 0.08]))          # palm inside the block
    assert not blocked["ok"] and blocked["worst"]["clearance"] < 0
    in_unseen = end_test(snap, hand, R_DOWN, np.array([0.5, 0.2, 0.15]))       # in unseen space only
    assert in_unseen["ok"] and in_unseen["unseen"] is not None and in_unseen["unseen"]["depth"] > 0
    clear = end_test(snap, hand, R_DOWN, np.array([0.5, -0.25, 0.3]))
    assert clear["ok"] and clear["unseen"] is None


def test_slide_back_ends_where_the_end_test_passes():
    scenes = standard_scenes()
    sc = scenes["S1"]
    snap = box_scene(boxes=[BLOCK], table_z=0.0)
    contact = np.array([0.5, 0.0, 0.05])                    # inside the block: the hand-off must slide up out of it
    e = EndCandidate(R_DOWN, contact, approach=np.array([0.0, 0.0, -1.0]), standoff=(0.03, 0.30))
    req = make_request(sc, ends=[e], mode="straight")
    req.start_p = np.array([0.35, -0.2, 0.45])
    resp = TraceService({"device": "cpu"}).plan(req, snap, trivial_arm_check)
    assert resp.result == "certified" and resp.standoff > 0.05
    end = contact - resp.standoff * np.array([0.0, 0.0, -1.0])
    assert end_test(snap, req.bodies, R_DOWN, end)["ok"]
    assert not end_test(snap, req.bodies, R_DOWN, end - np.array([0.0, 0.0, 0.02]))["ok"]   # 2 cm lower is not
    # an end that never clears within the range: terminal no_path with the reason
    e2 = EndCandidate(R_DOWN, np.array([0.5, 0.0, 0.0]), approach=np.array([0.0, 0.0, -1.0]), standoff=(0.0, 0.02))
    req2 = make_request(sc, ends=[e2], mode="straight")
    r2 = TraceService({"device": "cpu"}).plan(req2, snap, trivial_arm_check)
    assert r2.result == "no_path" and r2.stats["stage"] == "terminal"
    assert r2.stats["ends_dropped"][0]["why"] == "blocked" and r2.stats["ends_dropped"][0]["group"] in GROUPS


def test_leg_verdicts():
    o = lambda cause, travel: Outcome(cause, travel, 0.0, None, 0.02, None)
    assert judge(o("contact", 0.03), (0.05, 0.10)) == "early"
    assert judge(o("force", 0.07), (0.05, 0.10)) == "as_expected"
    assert judge(o("done", 0.10), (0.05, 0.10)) == "none"
    assert judge(o("contact", 0.02), None) == "early" and judge(o("done", 0.1), None) == "as_expected"
    assert judge(o("no_line", 0.0), (0.0, 0.1)) == "no_line"


def test_fill_never_overwrites_seen_free_space():
    sc = SceneMap((0.0, 0.0, 0.0), (0.2, 0.2, 0.2), res=0.01, table_z=0.0)
    obj = ObjectModel(3, "box", "box", "destination", 0.0)
    top = np.stack(np.meshgrid(np.arange(0.05, 0.15, 0.004), np.arange(0.05, 0.15, 0.004), [0.12]), -1).reshape(-1, 3)
    obj.set_points(top)                                     # a roof seen from above: the fill would solidify below it
    seen = np.zeros(sc.shape, bool)
    seen[8:12, 8:12, 2:8] = True                            # a camera saw into the cavity under the roof
    sc.state[:] = UNKNOWN
    sc.insert_object(obj, seen_free=seen)
    assert (sc.state[8:12, 8:12, 2:8] == UNKNOWN).all()     # not filled: stays what it was (recalled free later)
    assert sc.state[6, 6, 5] == OCC                         # elsewhere under the roof: inferred solid
    sc.state[:] = UNKNOWN
    sc.insert_object(obj, seen_free=seen, force=True)       # an object just put there: its whole solid
    assert sc.state[9, 9, 5] == OCC


def test_wrist_image_pointing_field():
    assert "image" in OBJECT_SCHEMA["required"] and OBJECT_SCHEMA["properties"]["image"]["enum"] == [1, 2]
    prog = {"objects": [{"id": "o1", "name": "can", "prompts": ["can"], "x": 600, "y": 10, "image": 2,
                         "part_of": None, "role": "target"},
                        {"id": "o2", "name": "basket", "prompts": ["basket"], "x": 5, "y": 5, "part_of": None,
                         "role": "destination"}],
            "stages": [{"id": "s1", "op": "transfer", "object": "o1", "relation": "in", "reference": "o2",
                        "terms": []}]}
    validate_program(prog)
    assert prog["objects"][0]["image"] == 2 and prog["objects"][1]["image"] == 1 and prog["objects"][0]["x"] == 511


class _Frame:
    def __init__(self, pts):
        self.pts, self.robot_mask = pts, np.zeros((4, 4), bool)

    def backproject(self, mask=None, stride=1):
        return self.pts, None


def test_relocalize_registers_the_known_shape():
    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, 13), np.linspace(-0.02, 0.02, 9), np.linspace(0, 0.08, 17)),
                 -1).reshape(-1, 3)
    shell = g[(np.abs(g[:, 0]) > 0.029) | (np.abs(g[:, 1]) > 0.019) | (g[:, 2] > 0.079)] + [0.5, 0.0, 0.0]
    obj = ObjectModel(3, "box", "box", "target", 0.0)
    obj.set_points(shell + [0.012, -0.008, 0.0])           # where the model says it is (off by 1.4 cm)
    seen = shell + rng.normal(0, 0.0008, shell.shape)
    h = SimpleNamespace(objects={3: obj}, table_z=-0.01, _forget_free=lambda pts: None, log=lambda *a, **k: None)
    fit = ContactMixin.relocalize(SimpleNamespace(**vars(h), _without_others=lambda P, oid: P), obj, [_Frame(seen)])
    assert fit is not None and fit["inliers"] > 0.8
    assert np.linalg.norm(obj.points.mean(0) - shell.mean(0)) < 0.003


def test_rest_words_and_start_side():
    w = ContactMixin._rest_words({"would_rest_on_something_else": 5, "overhang unsegmented surface": 2})
    assert "rest on something else" in w and "unsegmented surface is in the way from above" in w
    from atrace.harness import Harness
    assert Harness._start_side({"kind": "path_blocked", "at_t": 0.05})
    assert not Harness._start_side({"kind": "path_blocked", "at_t": 0.6}) and not Harness._start_side(None)


# ------------------------------------------------------------------------------------------------ M4: direction
def _box_sdf(P, lo, hi):
    c, h = (np.asarray(lo) + np.asarray(hi)) / 2, (np.asarray(hi) - np.asarray(lo)) / 2
    q = np.abs(P - c) - h
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def _cavity_harness(boxes):
    """A stand-in for the harness: observed geometry = boxes (exact distances)."""
    def point_clearance(P, exclude=(), unknown=True):
        P = np.asarray(P, float)
        return np.min([_box_sdf(P, lo, hi) for lo, hi in boxes], axis=0)
    h = SimpleNamespace(checker=SimpleNamespace(point_clearance=point_clearance),
                        nearest_obstacle=lambda P, excl=(): {"object": "roof"})
    return h


def test_entry_search_slides_in_under_a_roof():
    # a cavity open toward -y: floor top z=0.10, roof bottom z=0.25, back wall at y=0.30, sides x=+-0.15; front at y=0
    boxes = [((-0.2, 0.0, 0.0), (0.2, 0.32, 0.10)), ((-0.2, 0.0, 0.25), (0.2, 0.32, 0.30)),
             ((-0.2, 0.30, 0.10), (0.2, 0.32, 0.25)), ((-0.2, 0.0, 0.10), (-0.15, 0.32, 0.25)),
             ((0.15, 0.0, 0.10), (0.2, 0.32, 0.25))]
    h = _cavity_harness(boxes)
    g = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, 5), np.linspace(-0.03, 0.03, 5), np.linspace(0, 0.08, 5)),
                 -1).reshape(-1, 3)
    Wp = g + np.array([0.0, 0.0, -0.04])               # the object, its bottom 4 cm below the site
    Lc = np.array([[0.0, 0.0, 0.0], [0.0, -0.05, 0.0], [0.0, -0.10, 0.0]])   # a hand sticking out toward -y
    radc = np.full(len(Lc), 0.01)
    p_rel = np.array([0.0, 0.18, 0.141])                # object 1 mm above the floor, well inside
    dirs = [np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    entry, blocked = ContactMixin.entry_search(h, p_rel, np.eye(3), Wp, Lc, radc, (), dirs)
    assert blocked == {"object": "roof"}
    assert entry["kind"] == "slide" and np.allclose(entry["dir"], [0.0, -1.0, 0.0])
    p_out, p_mid, p_end = entry["path"]
    assert p_out[1] < 0.0 - 0.03 + 0.10 and p_mid[2] > p_end[2] and np.allclose(p_end, p_rel)
    # open from above: straight down
    h2 = _cavity_harness([((-0.2, 0.0, 0.0), (0.2, 0.32, 0.10))])
    e2, _ = ContactMixin.entry_search(h2, p_rel, np.eye(3), Wp, Lc, radc, (), dirs)
    assert e2["kind"] == "vertical" and len(e2["path"]) == 1


def test_side_grasps_find_a_handle_pinch():
    from atrace import manip
    th = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    wall = np.concatenate([np.c_[r * np.cos(th), r * np.sin(th), np.full(len(th), z)]
                           for r in (0.045, 0.041) for z in np.arange(0, 0.10, 0.006)])
    bar = np.stack(np.meshgrid(np.arange(-0.087, -0.078, 0.003), np.arange(-0.009, 0.0091, 0.003),
                               np.arange(0.027, 0.086, 0.003)), -1).reshape(-1, 3)
    g = manip.side_grasp_candidates(np.r_[wall, bar], "mug", [np.array([-1.0, 0.0, 0.0])])
    pinch = [c for c in g if c.p[0] < -0.07]
    assert pinch, "no grasp on the handle"
    assert abs(pinch[0].R[2, 2]) < 0.5 and pinch[0].R[0, 2] > 0.8      # approach horizontal, toward the mug


def test_entry_hint_words():
    R = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # camera: right = +y, forward = -x
    h = SimpleNamespace(head0=SimpleNamespace(T_wc=np.r_[np.c_[R, np.zeros(3)], [[0, 0, 0, 1.0]]]))
    assert np.allclose(ContactMixin._entry_hint(h, "from_image_right"), [0.0, 1.0, 0.0])
    assert np.allclose(ContactMixin._entry_hint(h, "from_camera_side"), [1.0, 0.0, 0.0])
    assert ContactMixin._entry_hint(h, "from_above") is None and ContactMixin._entry_hint(h, "any") is None


def test_on_is_a_footprint_share():
    plate, other = 3, 7
    lb = np.full(100, plate)
    inside = np.ones(100, bool)
    share = lambda *a, **k: ContactMixin._footprint_share(*a, **k)
    assert share(lb, inside, [plate]) == 1.0
    in2, lb2 = inside.copy(), lb.copy()
    in2[50:], lb2[50:] = False, 0                      # half of it beyond the plate's support region
    assert share(lb2, in2, [plate]) == 0.5 < 0.6
    lb3 = lb.copy()
    lb3[:45] = other                                   # 45 % on another object lying on the plate
    assert abs(share(lb3, inside, [plate]) - 0.55) < 1e-9
    assert share(np.zeros(100, int), inside, [plate]) == 1.0       # unlabelled surface inside the support region
    assert share(np.zeros(100, int), None, [plate]) == 0.0         # no support region (too sparse): labels decide
    assert share(lb, np.zeros(100, bool), [plate], flat=True) == 1.0   # a flat mat: labels decide


def test_support_mask_closes_gaps():
    # a concave plate seen as a ring (rim) plus a sparse, patchy inner surface: the region is the whole plate
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rim = np.c_[0.09 * np.cos(th), 0.09 * np.sin(th), np.full(120, 0.015)]
    rng = np.random.default_rng(0)
    inner = rng.uniform(-0.07, 0.07, (60, 2))
    inner = np.c_[inner[np.linalg.norm(inner, axis=1) < 0.07], np.full(int((np.linalg.norm(inner, axis=1) < 0.07).sum()), 0.008)]
    dest = SimpleNamespace(points=np.r_[rim, inner], virtual=False)
    h = SimpleNamespace(table_z=0.0)
    lo = np.array([-0.15, -0.15])
    m, n = ContactMixin._support_mask(h, dest, lo, (60, 60))
    assert m is not None and n == len(dest.points)
    c = 0.005
    centre = np.floor((np.array([0.0, 0.0]) - lo) / c).astype(int)
    assert m[centre[0], centre[1]]                              # the plate's middle, where few points were seen
    disk = np.argwhere(np.ones((60, 60), bool))
    xy = lo + (disk + 0.5) * c
    in_plate = np.linalg.norm(xy, axis=1) < 0.085
    assert m.reshape(-1)[in_plate].mean() > 0.97               # nearly all of the plate
    assert m.reshape(-1)[np.linalg.norm(xy, axis=1) > 0.11].mean() < 0.02   # not the table around it
    few = SimpleNamespace(points=rim[:10], virtual=False)
    assert ContactMixin._support_mask(h, few, lo, (60, 60))[0] is None      # too sparse: labels decide


def test_hollow_container_interior_is_not_filled():
    from atrace.perception import UNKNOWN as UNK
    sc = SceneMap((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), res=0.01, table_z=0.0)
    obj = ObjectModel(5, "microwave", "microwave", "destination", 0.0)
    g = np.stack(np.meshgrid(np.arange(0.05, 0.25, 0.004), np.arange(0.05, 0.25, 0.004)), -1).reshape(-1, 2)
    roof = np.c_[g, np.full(len(g), 0.20)]                     # seen from above: only the roof's top
    obj.set_points(roof)
    sc.state[:] = UNK
    sc.insert_object(obj)
    assert sc.state[15, 15, 10] == OCC                         # default: inferred solid below the roof
    sc.state[:] = UNK
    sc.insert_object(obj, hollow=True)
    assert sc.state[15, 15, 10] == UNK                         # a container: the space under its roof stays unknown
    assert sc.state[15, 15, 19] == OCC                         # ... but the roof has its thickness


def test_bounded_drop_only_for_in_with_room():
    # the hand is blocked at the floor by a wall top at z = 0.02 (palm points below it), clear from 2.5 cm up
    def point_clearance(P, exclude=(), unknown=True):
        return np.asarray(P, float)[:, 2] - 0.02 + 0.01
    L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.005]])
    grp, rad = np.array([2, 1]), np.array([0.001, 0.001])
    h = SimpleNamespace(checker=SimpleNamespace(point_clearance=point_clearance),
                        nearest_obstacle=lambda P, excl=(): {"object": "wall"},
                        entry_search=lambda *a, **k: ({"kind": "vertical", "path": [a[0]], "approach": np.array([0, 0, -1.0])}, None),
                        _entry_dirs=lambda *a, **k: [], _entry_posture=lambda *a, **k: np.zeros(7))
    for name in ("_release_hand", "_bad", "_blocked_below"):
        setattr(h, name, getattr(ContactMixin, name).__get__(h))
    ctx = lambda rel, drop: {"held": (), "rel": rel, "drop": drop, "hint": None, "f_rel": 0.04, "L": L, "grp": grp,
                             "rad": rad, "Lc": L, "radc": rad, "wide": [(0.04, L, grp, rad)],
                             "tight": [(0.012, L, grp, rad), (0.04, L, grp, rad)], "q_now": np.zeros(7),
                             "why": {}, "dbg": []}
    spot = {"xy": np.zeros(2), "yaw": 0.0, "rest_z": 0.0, "rim": 0.02, "open_dirs": [], "wall_gap": 0.02, "score": 0}
    rest = np.array([0.0, 0.0, 0.0])
    c = ContactMixin._rest_3d(h, ctx("in", False), np.eye(3), np.zeros((1, 3)), rest, spot)
    assert c is not None and c["bounded_drop"] and 0.0 < c["release_dz"] <= 0.045 and c["release_finger"] == 0.012
    assert ContactMixin._rest_3d(h, ctx("on", False), np.eye(3), np.zeros((1, 3)), rest, spot) is None
    tight_spot = dict(spot, wall_gap=0.005)                  # the footprint too close to a wall: no drop
    assert ContactMixin._rest_3d(h, ctx("in", False), np.eye(3), np.zeros((1, 3)), rest, tight_spot) is None
    h.checker = SimpleNamespace(point_clearance=lambda P, exclude=(), unknown=True: np.full(len(P), -0.005))
    assert ContactMixin._rest_3d(h, ctx("in", False), np.eye(3), np.zeros((1, 3)), rest, spot) is None  # beside: no


def test_every_called_method_exists():
    import ast
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "atrace"
    texts = {f: (src / f).read_text() for f in ("contact.py", "harness.py", "lrl.py", "ucs_harness.py")}
    defined = {n.name for t in texts.values() for n in ast.walk(ast.parse(t)) if isinstance(n, ast.FunctionDef)}
    called = {c for t in texts.values() for c in re.findall(r"self\.(_?[a-zA-Z]\w*)\(", t)}
    attrs = {"astra", "rb", "ex", "tm", "rm", "checker", "scene", "detector", "segmenter"}      # objects, not methods
    assert not sorted(called - defined - attrs)


def test_side_frame_sees_through_a_handle_loop():
    th = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    wall = np.concatenate([np.c_[r * np.cos(th), r * np.sin(th), np.full(len(th), z)]
                           for r in (0.045, 0.041) for z in np.arange(0, 0.10, 0.004)])
    bar = np.stack(np.meshgrid(np.arange(-0.087, -0.078, 0.003), np.arange(-0.009, 0.0091, 0.003),
                               np.arange(0.027, 0.086, 0.003)), -1).reshape(-1, 3)
    top = np.stack(np.meshgrid(np.arange(-0.083, -0.044, 0.003), np.arange(-0.009, 0.0091, 0.003),
                               np.arange(0.080, 0.087, 0.003)), -1).reshape(-1, 3)
    bot = top - [0.0, 0.0, 0.055]
    mug = np.concatenate([wall, bar, top, bot])
    finger = np.array([[-0.064, 0.0, 0.05], [-0.064, 0.0, 0.06]])   # through the handle's loop (between bar and body)
    seen_from_above = ContactMixin.object_clearance(mug, finger)
    seen_across_loop = ContactMixin.object_clearance(mug, finger, side=np.array([0.0, -1.0, 0.0]))
    assert (seen_from_above < 0).all()                         # the 2.5D solid seen from above fills the loop
    assert (seen_across_loop > 0.005).all()                    # seen across the loop (a side grasp's frame): free
