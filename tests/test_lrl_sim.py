"""Tier A stage simulator and scene builders (LRL_PLAN.md 6.1, 6.2). CPU, real simulator, no rendering."""
import mujoco
import numpy as np
import pytest

from atrace.articulate import translate
from atrace.geometry import frame_from_axes
from lrl_train import scenes
from lrl_train.sim import CFG, Shim, StageSim


@pytest.fixture(scope="module")
def drawers():
    sim = StageSim(scenes.sample_layout(np.random.default_rng(24), "drawer_stack"))
    yield sim
    sim.close()


def front_pose(part, standoff=0.03):
    """Closed fingertips pointing into the part's face, standoff clear of its handle."""
    n = part.face_normal
    return frame_from_axes(-n, np.cross(n, part.face_u)), part.face_point + n * (standoff + part.protrusion)


def test_drawer_stack_parts_match_model(drawers):
    m, d = drawers.robot.m, drawers.robot.d
    assert len(drawers.parts) >= 2
    base = np.array(scenes.CFG["base_xy"])
    for p in drawers.parts:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p.joint)
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p.body)
        assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE and m.jnt_bodyid[j] == b
        np.testing.assert_allclose(m.jnt_range[j], p.range, atol=1e-4)
        np.testing.assert_allclose(p.axis, p.face_normal, atol=1e-6)        # a drawer opens along its outward normal
        assert {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])} <= set(p.fixture_bodies)
        gid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, p.face_point + 0.2 * p.face_normal, -p.face_normal, None, 1, -1, gid)
        assert m.geom_bodyid[gid[0]] == b and 0.2 - p.protrusion - 1e-4 <= dist <= 0.2 + 1e-4
        to_base = base - p.face_point[:2]
        assert p.face_normal[:2] @ to_base / np.linalg.norm(to_base) >= np.cos(scenes.CFG["mount_yaw"]) - 1e-6
    assert all(pen < 1e-3 for _, _, pen in drawers.robot_contacts())


@pytest.mark.parametrize("finger", [CFG["finger_open"], CFG["finger_closed"]])
def test_set_arm_holds(drawers, finger):
    rb = drawers.robot
    q = drawers.q_home + np.array([0.2, 0.1, -0.1, 0.15, 0.1, -0.2, 0.3])
    drawers.set_arm(q, finger)
    p_set = rb.model.site_pose(rb.model.full_q(q, finger))[1][0]
    assert np.linalg.norm(rb.eef_pose()[1] - p_set) < 1e-3
    np.testing.assert_allclose(rb.last_goal[1], p_set, atol=1e-9)
    forces = []
    for _ in range(10):
        rb.hold(1)
        forces.append(rb.wrench()[0])
    assert np.ptp(np.array(forces), axis=0).max() < 0.2                     # N: steady wrist force
    assert np.linalg.norm(rb.eef_pose()[1] - p_set) < 1e-3
    assert abs(rb.finger_opening() - (2 * finger - 0.001 if finger else 0.0)) < 2e-3


def test_entry_in_front_of_face(drawers):
    part = drawers.parts[0]
    drawers.set_joint(part.joint, part.closed)
    R, p = front_pose(part)
    assert drawers.entry(R, p)
    Re, pe = drawers.robot.eef_pose()
    assert np.linalg.norm(pe - p) < 3e-3 and np.abs(Re - R).max() < 0.03    # IK tolerances of RobotModel.ik
    lim = drawers.robot.model.limits[:7]
    q = drawers.robot.q()[:7]
    assert np.minimum(q - lim[:, 0], lim[:, 1] - q).min() >= CFG["limit_slack"] - 1e-3
    assert not drawers.entry(R, p + np.array([0.0, 0.0, 2.0]))             # out of reach: no solution, arm untouched
    np.testing.assert_allclose(drawers.robot.q()[:7], q, atol=1e-6)


def test_translate_pushes_open_drawer_and_contacts_are_named(drawers):
    part = drawers.parts[0]
    drawers.set_joint(part.joint, part.open)
    opened = part.face_point.copy()
    assert drawers.entry(*front_pose(part))
    seen = set()
    drawers.robot.on_step = lambda rb, obs: seen.update((a, b) for a, b, _ in drawers.robot_contacts())
    travel, why, _ = translate(Shim(drawers.robot), part.face_normal, "push", grip_check=False)
    drawers.robot.on_step = None
    moved = drawers.part_travel(part)
    assert moved > 0.6 * abs(part.open - part.closed) and travel > moved - 0.01
    assert why in ("stalled", "force_limit", "max_travel")
    assert any(a.startswith("gripper0_") and b == part.body for a, b in seen)
    assert all(a.startswith(scenes.ROBOT) and not b.startswith(scenes.ROBOT) for a, b in seen)
    scenes.place_parts(drawers.parts, drawers.robot.m, drawers.robot.d)
    np.testing.assert_allclose(opened - part.face_point, moved * part.face_normal, atol=1e-6)


def test_snapshot_restore(drawers):
    part = drawers.parts[-1]
    drawers.set_joint(part.joint, 0.5 * part.open)
    assert drawers.entry(*front_pose(part))
    s = drawers.snapshot()
    q, goal = drawers.robot.q(), drawers.robot.last_goal[1].copy()

    def push():
        translate(Shim(drawers.robot), part.face_normal, "push", max_travel=0.05, grip_check=False)
        return drawers.joint(part.joint), drawers.robot.q()

    first = push()
    assert abs(first[0] - 0.5 * part.open) > 0.01
    drawers.restore(s)
    assert drawers.joint(part.joint) == 0.5 * part.open and drawers.part_travel(part) == 0.0
    np.testing.assert_array_equal(drawers.robot.q(), q)
    np.testing.assert_array_equal(drawers.robot.last_goal[1], goal)
    second = push()                                                          # the same push again: same outcome
    assert abs(second[0] - first[0]) < 1e-9 and np.abs(second[1] - first[1]).max() < 1e-9


def test_white_cabinet_layout_from_generated_bddl(tmp_path):
    layout = scenes.sample_layout(np.random.default_rng(5), "white_cabinet", out_dir=tmp_path)
    assert "white_cabinet_1 - white_cabinet" in open(layout.bddl).read()
    sim = StageSim(layout)
    m, d = sim.robot.m, sim.robot.d
    assert sorted(p.name for p in sim.parts) == [f"white_cabinet_1_{k}_level" for k in ("bottom", "middle", "top")]
    carcass = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "white_cabinet_1_base")
    centre = d.geom_xpos[(m.geom_bodyid == carcass) & (m.geom_contype > 0)].mean(0)
    for p in sim.parts:
        assert p.kind == "slide" and "white_cabinet_1_base" in p.fixture_bodies
        assert abs(p.face_normal[2]) < 1e-6 and (p.face_point - centre) @ p.face_normal > 0.05
        np.testing.assert_allclose(p.face_normal, p.axis * np.sign(p.open - p.closed), atol=1e-6)
        assert 0.08 < p.half_extents[0] < 0.13 and 0.02 < p.half_extents[1] < 0.05 and 0.01 < p.protrusion < 0.05
    top, bottom = (f(sim.parts, key=lambda p: p.face_point[2]) for f in (max, min))
    assert not sim.entry(*front_pose(bottom))           # the hand does not fit between this face and the table
    sim.set_joint(top.joint, top.open)
    assert sim.entry(*front_pose(top))
    translate(Shim(sim.robot), top.face_normal, "push", grip_check=False)
    assert sim.part_travel(top) > 0.12
    sim.close()
