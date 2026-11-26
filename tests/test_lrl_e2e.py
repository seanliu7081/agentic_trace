"""LRL_PLAN M1E: the shared end-to-end interface (frame A, start region, goal, action map, safety wrapper, proprio),
on a simulated hand and once in the stage simulator."""
import numpy as np
import pytest

from atrace.geometry import mat_to_rotvec, rotvec_to_mat
from atrace.lrl_e2e import (ACT_DIM, E2E_CFG, GOAL_DIM, PROPRIO_DIM, STEP_DIM, Belief, E2ESkill, Goal, action_to_command,
                            frame_a, start_region)

C = np.array([0.1, 0.2, 1.0])
N = np.array([-1.0, 0.2, 0.0]) / np.hypot(1.0, 0.2)


class FakeIO:
    """A hand that moves to the commanded pose with a first-order lag; past the wall plane n.p = off it reports a
    spring force against n; the fingers move 1 cm per step under a gripper command."""

    def __init__(self, R, p, alpha=0.3, wall=None, k_wall=9000.0, gap=0.08):
        self.R, self.p, self.alpha, self.wall, self.k_wall, self.gap = R.copy(), p.copy(), alpha, wall, k_wall, gap
        self.cmds = []

    def step(self, R_goal, p_goal, grip=None, posture=None):
        self.cmds.append((R_goal.copy(), p_goal.copy(), grip))
        self.p = self.p + self.alpha * (p_goal - self.p)
        self.R = rotvec_to_mat(self.alpha * mat_to_rotvec(R_goal @ self.R.T)) @ self.R
        self.gap = float(np.clip(self.gap - 0.01 * grip, 0.0, 0.08))

    def eef_pose(self):
        return self.R.copy(), self.p.copy()

    def wrench(self):
        if self.wall is None:
            return np.zeros(3), np.zeros(3)
        n, off = self.wall
        return -self.k_wall * max(n @ self.p - off, 0.0) * n, np.zeros(3)

    def finger_opening(self):
        return self.gap


def started(goal=Goal("open"), **io_kw):
    belief = Belief(C, N)
    R, p = start_region(belief)[0]
    frame = frame_a(p, belief)
    io = FakeIO(R, p, **io_kw)
    sk = E2ESkill()
    obs = sk.start(io, frame, goal)
    return sk, io, frame, obs


def test_frame_a_axes():
    belief = Belief(C, N)
    p = C + 0.15 * N + np.array([0.0, 0.0, 0.05])
    A = frame_a(p, belief)
    np.testing.assert_allclose(A.R.T @ A.R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(A.R[:, 2], [0, 0, 1])
    assert abs(np.linalg.det(A.R) - 1) < 1e-12
    q = A.point(p)
    assert q[0] > 0.14 and abs(q[1]) < 1e-9 and abs(q[2] - 0.05) < 1e-9     # the start lies on +x, level with z up
    np.testing.assert_allclose(A.to_world(A.point(p)), p)
    np.testing.assert_allclose(A.point(C), 0, atol=1e-12)


def test_frame_a_from_above_takes_normal_then_base():
    above = C + np.array([0.01, 0.0, 0.15])
    A = frame_a(above, Belief(C, N))
    np.testing.assert_allclose(A.R[:, 0], np.r_[N[:2], 0] / np.linalg.norm(N[:2]), atol=1e-9)
    base = np.array([-0.66, 0.0, 0.9])
    A = frame_a(above, Belief(C, [0.0, 0.05, 1.0]), base)
    h = np.r_[(base - C)[:2], 0]
    np.testing.assert_allclose(A.R[:, 0], h / np.linalg.norm(h), atol=1e-9)
    np.testing.assert_allclose(frame_a(above, Belief(C, [0, 0, 1.0])).R[:, 0], [-1, 0, 0])


def test_start_region_poses():
    belief = Belief(C, N)
    poses = start_region(belief)
    per = len(E2E_CFG["start_lateral"]) * len(E2E_CFG["start_rolls"])
    assert len(poses) == len(E2E_CFG["start_dist"]) * len(E2E_CFG["start_elev"]) * per
    u = np.cross(N, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    for R, p in poses:
        w = -R[:, 2]                                                        # the way out; fingers point back
        elev = np.degrees(np.arcsin(np.clip(w[2], -1, 1)))
        assert min(abs(elev - e) for e in E2E_CFG["start_elev"]) < 1e-6
        lat = (p - C) @ u
        assert min(abs(lat - x) for x in E2E_CFG["start_lateral"]) < 1e-9
        d = np.linalg.norm(p - C - lat * u)
        assert 0.1 - 1e-9 <= d <= 0.2 + 1e-9 and np.allclose((p - C - lat * u) / d, w)
    R, p = poses[0]
    np.testing.assert_allclose(p, C + E2E_CFG["start_dist"][0] * N)
    np.testing.assert_allclose(R[:, 2], -N, atol=1e-9)                     # first: level, on the normal
    assert abs(R[2, 0]) < 1e-9 and abs(poses[1][0][2, 0]) > 0.99            # closing axis level, then upright
    up = start_region(Belief(C, [0, 0, 1.0]), dists=(0.05,))
    assert len(up) == per                                                   # no raised way out from a top face
    np.testing.assert_allclose(up[0][1], C + [0, 0, 0.05])


def test_goal_vector():
    assert Goal("open").vector().shape == (GOAL_DIM,)
    v = Goal("turn_on", sense=-1, amount=1.2, unit="rad").vector()
    np.testing.assert_allclose(v, [0, 0, 1, 0, -1, 1, 0, 1.2])
    v = Goal("close", amount=0.1).vector()
    np.testing.assert_allclose(v, [0, 1, 0, 0, 0, 1, E2E_CFG["amount_scale"] * 0.1, 0])
    with pytest.raises(AssertionError):
        Goal("push")


def test_action_map_in_frame_a():
    sk, io, A, obs = started()
    assert obs.shape == (PROPRIO_DIM,) and PROPRIO_DIM == STEP_DIM * E2E_CFG["stack"] + 1 + GOAL_DIM
    R, p = io.eef_pose()
    for a_x, want in ((0.5, 0.25), (-1.0, -1.0), (0.0, 0.0)):
        a = np.zeros(ACT_DIM)
        a[0] = a_x
        R_cmd, lead = action_to_command(a, R, p, A)
        np.testing.assert_allclose(A.vec(lead), [want * E2E_CFG["dp"], 0, 0], atol=1e-12)
        np.testing.assert_allclose(R_cmd, R, atol=1e-12)
    a = np.zeros(ACT_DIM)
    a[5] = 1.0                                                              # about A's z (up)
    R_cmd, _ = action_to_command(a, R, p, A)
    np.testing.assert_allclose(mat_to_rotvec(R_cmd @ R.T), [0, 0, np.radians(E2E_CFG["dr"])], atol=1e-9)
    a = np.ones(ACT_DIM)
    assert np.linalg.norm(action_to_command(a, R, p, A)[1]) <= E2E_CFG["lead_max"] + 1e-12


def test_step_moves_in_a_and_reports():
    sk, io, A, _ = started()
    p0 = io.p.copy()
    a = np.zeros(ACT_DIM)
    a[0] = -0.6                                                             # towards the part
    for _ in range(5):
        obs = sk.step(a)
    d = A.vec(io.p - p0)
    assert d[0] < -0.01 and abs(d[1]) < 1e-9 and abs(d[2]) < 1e-9
    step = obs[(E2E_CFG["stack"] - 1) * STEP_DIM:E2E_CFG["stack"] * STEP_DIM]
    np.testing.assert_allclose(step[:3], E2E_CFG["p_scale"] * A.point(io.p), atol=1e-9)
    assert step[9] < 0                                                      # velocity along -x of A
    np.testing.assert_allclose(step[-ACT_DIM:], a)                          # the previous action
    assert obs[-GOAL_DIM - 1] == pytest.approx(1 - 5 / sk.n_max)
    assert sk.stop is None and len(sk.trace.rows) == 5


def test_box_clips_the_command():
    sk, io, A, _ = started()
    a = np.zeros(ACT_DIM)
    a[2] = 1.0
    for _ in range(200):
        sk.step(a)
        if sk.stop:
            break
    assert A.point(io.cmds[-1][1])[2] <= E2E_CFG["box_hi"][2] + 1e-9


def test_speed_limit_scales_the_lead():
    sk, io, A, _ = started(alpha=1.0)
    a = np.zeros(ACT_DIM)
    a[1] = 1.0
    sk.step(a)
    sk.v = np.array([0.6, 0.0, 0.0])                                        # four times the limit
    p = io.p.copy()
    sk.step(a)
    assert np.linalg.norm(io.p - p) == pytest.approx(E2E_CFG["dp"] / 4, rel=1e-6)


def test_force_cap_counts_the_lead_as_the_force_it_makes():
    """Pressing with the felt force F along u: the lead along u is cut to (f_cap + b_osc v.u) / k_contact, whatever F
    already is (the old cap, (f_cap - F) / k, held a steady push under f_cap / 2); the lead across u is free."""
    c = E2E_CFG
    sk, io, A, _ = started(alpha=1.0)
    u = np.array([-1.0, 0.0, 0.0])
    a = np.zeros(ACT_DIM)
    a[0] = -1.0                                     # 2.5 cm of lead along -x of A
    for v_u, want in ((0.0, c["f_cap"] / c["k_contact"]), (0.05, (c["f_cap"] + c["b_osc"] * 0.05) / c["k_contact"])):
        sk.raw = np.r_[20.0 * (A.R @ u), np.zeros(3)]     # 20 N felt along -x of A
        sk.v = v_u * (A.R @ u)                      # moving along it at v_u
        p = io.p.copy()
        sk.step(a)
        lead = A.vec(io.cmds[-1][1] - p)
        assert lead @ u == pytest.approx(want, rel=1e-6)
        assert want < c["dp"]                       # the cap is what bound it
    a = np.zeros(ACT_DIM)
    a[1] = 1.0                                      # across the felt force: untouched
    sk.raw = np.r_[20.0 * (A.R @ u), np.zeros(3)]
    sk.v = np.zeros(3)
    p = io.p.copy()
    sk.step(a)
    assert A.vec(io.cmds[-1][1] - p)[1] == pytest.approx(c["dp"], rel=1e-6)


def test_gripper_commands():
    sk, io, A, _ = started()
    for g, want in ((0.9, 1.0), (0.1, 0.0), (-0.5, -1.0), (0.33, 0.0)):
        a = np.zeros(ACT_DIM)
        a[6] = g
        sk.step(a)
        assert io.cmds[-1][2] == want


def test_done_force_and_time_stops():
    sk, *_ = started()
    a = np.zeros(ACT_DIM)
    a[7] = 0.1
    sk.step(a)
    assert sk.stop == "done" and sk.trace.stop == "done" and sk.trace.steps == 0
    # a wall 2 cm in front of the start along the approach: pushing into it stops on force, then holds the pose
    belief = Belief(C, N)
    R, p = start_region(belief)[0]
    sk, io, A, _ = started(wall=(-N, -N @ p + 0.02))
    a = np.zeros(ACT_DIM)
    a[0] = -1.0
    for _ in range(100):
        sk.step(a)
        if sk.stop:
            break
    assert sk.stop == "force"
    np.testing.assert_allclose(io.cmds[-1][1], io.cmds[-2][1], atol=0.01)  # the back-off holds what it measured
    sk, *_ = started()
    sk.n_max = 7
    while sk.stop is None:
        sk.step(np.zeros(ACT_DIM))
    assert sk.stop == "time" and sk.trace.steps == 7
    arr = sk.trace.arrays()
    assert arr["action"].shape == (7, ACT_DIM) and arr["gap"].shape == (7,)


def test_in_stage_simulator():
    """The skill in Tier A: the hand moves along A's axes as commanded, the gripper closes, nothing else moves."""
    from lrl_train import scenes
    from lrl_train.sim import StageSim
    sim = StageSim(scenes.sample_layout(np.random.default_rng(24), "drawer_stack"))
    part = sim.parts[0]
    belief = Belief(part.face_point, part.face_normal)
    assert any(sim.entry(R, p, finger=0.04) for R, p in start_region(belief))
    R, p = sim.robot.eef_pose()
    A = frame_a(p, belief, base=np.r_[scenes.CFG["base_xy"], scenes.CFG["table_z"]])
    sk = E2ESkill()
    sk.start(sim.robot, A, Goal("open"))
    a = np.zeros(ACT_DIM)
    a[2], a[6] = 0.8, 1.0                                                   # up, close
    for _ in range(20):
        sk.step(a)
    d = A.vec(sim.robot.eef_pose()[1] - p)
    assert d[2] > 0.05 and abs(d[0]) < 0.01 and abs(d[1]) < 0.01
    assert sim.robot.finger_opening() < 0.01 and sk.stop is None
    sim.close()
