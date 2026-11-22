"""Placement "v1": candidates are ranked instead of rejected on the destination side (pure geometry, no simulator)."""
import numpy as np

from atrace import manip
from atrace.manip import container_region, placement_candidates, soft_candidates
from atrace.perception import ObjectModel


def _box(inner_x, inner_y, wall=0.01, height=0.08, step=0.003):
    """An open box standing on z = 0: floor plus four walls, as surface points."""
    xs = np.arange(-inner_x / 2 - wall, inner_x / 2 + wall + 1e-9, step)
    ys = np.arange(-inner_y / 2 - wall, inner_y / 2 + wall + 1e-9, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    floor = np.c_[X.ravel(), Y.ravel(), np.full(X.size, 0.005)]
    inside = (np.abs(X) < inner_x / 2) & (np.abs(Y) < inner_y / 2)
    rim = np.c_[X[~inside], Y[~inside]]
    walls = np.concatenate([np.c_[rim, np.full(len(rim), z)] for z in np.arange(0.008, height + 1e-9, step)])
    m = ObjectModel(1, "box", "box", "destination", 0.0)
    m.set_points(np.concatenate([floor[inside.ravel()], walls]))
    return m


def _rect(lx, ly, step=0.004):
    xs = np.arange(-lx / 2, lx / 2 + 1e-9, step)
    ys = np.arange(-ly / 2, ly / 2 + 1e-9, step)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)


def test_soft_candidates_rank_a_payload_that_does_not_fit():
    dest = _box(0.10, 0.04)                       # a slot 10 x 4 cm
    reg = container_region(dest)
    assert len(reg["grid_xy"])
    fp = _rect(0.115, 0.043)                      # a book 11.5 x 4.3 cm: larger than the opening both ways
    yaws = [0.0, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 4, np.pi]
    strict = placement_candidates(reg, fp, float(np.linalg.norm(fp, axis=1).max()), yaws, need_clear=0.0)
    assert strict == []                           # the strict test turns every cell down
    soft = soft_candidates(reg, fp, yaws, None)
    assert soft and all(c["soft"] for c in soft)
    best = soft[0]
    assert best["inside"] > 0.6                   # most of it over the opening ...
    assert abs(np.sin(best["yaw"])) < 1e-6        # ... long side along the slot ...
    assert np.linalg.norm(best["xy"]) < 0.012     # ... near its middle
    cross = [c["inside"] for c in soft_candidates(reg, fp, [np.pi / 2], None)]
    assert best["inside"] > max(cross)            # crosswise covers less of the opening, so it ranks lower


def test_soft_candidates_keep_clear_of_what_may_not_be_touched():
    dest = _box(0.20, 0.20)
    reg = container_region(dest)
    fp = _rect(0.05, 0.05)
    ys = np.arange(-0.1, 0.1001, 0.002)
    block = np.c_[np.zeros(len(ys)), ys]          # something that counts stands across the middle
    soft = soft_candidates(reg, fp, [0.0], block)
    assert soft and all(c["block_clear"] >= 0.002 for c in soft)
    assert all(abs(c["xy"][0]) > 0.02 for c in soft)      # no footprint straddles it


def test_rest_candidates_soft_ranks_an_unstable_rest():
    # a support 4 cm wide; a payload 12 cm long whose centroid cannot lie 5 mm inside its contact area's far ends
    sup = np.c_[_rect(0.04, 0.04, 0.003), np.full(len(_rect(0.04, 0.04, 0.003)), 0.02)]
    centres = manip.support_centres(sup)
    fp = _rect(0.12, 0.03, 0.004)
    prof_xy, prof_b = manip.bottom_profile(np.c_[fp + np.array([0.05, 0.0]), np.zeros(len(fp))])   # held off-centre
    g = np.array([0.05, 0.0])                     # centroid 5 cm from the footprint centre
    none = np.zeros((0, 3))
    strict = manip.rest_candidates(centres, prof_xy - prof_xy.mean(0), prof_b, g, sup, none, [0.0], 0.0,
                                   need_support=0.0)
    soft = manip.rest_candidates(centres, prof_xy - prof_xy.mean(0), prof_b, g, sup, none, [0.0], 0.0,
                                 need_support=0.0, soft=True)
    assert len(soft) >= len(strict) and soft
    assert all(c["soft"] for c in soft)
    # the least unstable rests rank first
    assert soft[0]["lean"] <= soft[-1]["lean"] + 1e-9
