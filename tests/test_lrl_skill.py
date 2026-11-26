"""LRL_PLAN M0: the contract data types and the skill loop, on a simulated hand (no simulator)."""
import numpy as np
import pytest

from atrace.geometry import mat_to_rotvec, rotvec_to_mat
from atrace.lrl_contract import LIMITS, Contract, Entry, Goal
from atrace.lrl_skill import ACT_DIM, CFG, OBS_DIM, OBS_STEP, Skill, Trace, neutral_action

R0 = rotvec_to_mat([0.3, -0.5, 0.8])
P0 = np.array([0.5, 0.1, 0.3])
D = np.array([1.0, 1.0, 0.5]) / 1.5


class FakeIO:
    """A hand that follows the commanded pose with a first-order lag. Past the wall plane n.p = off it reports a
    spring force against n; the fingers move 1 cm per step under a gripper command."""

    def __init__(self, R=R0, p=P0, alpha=0.2, wall=None, k_wall=4000.0, gap=0.08):
        self.R, self.p, self.alpha, self.wall, self.k_wall, self.gap = R.copy(), p.copy(), alpha, wall, k_wall, gap
        self.grips = []

    def step(self, R_goal, p_goal, grip=None, posture=None):
        self.p = self.p + self.alpha * (p_goal - self.p)
        self.R = rotvec_to_mat(self.alpha * mat_to_rotvec(R_goal @ self.R.T)) @ self.R
        self.gap = float(np.clip(self.gap - 0.01 * grip, 0.0, 0.08))
        self.grips.append(grip)

    def eef_pose(self):
        return self.R.copy(), self.p.copy()

    def q(self):
        return np.zeros(9)

    def joint_vel(self):
        return np.zeros(9)

    def wrench(self):
        if self.wall is None:
            return np.zeros(3), np.zeros(3)
        n, off = self.wall
        return -self.k_wall * max(n @ self.p - off, 0.0) * n, np.zeros(3)

    def finger_opening(self):
        return self.gap


def contract(mode="push", d=D, travel=0.2, axis_point=None, **limits):
    entry = Entry(R0, P0, R0[:, 2], (0.0, 0.05))
    return Contract("move_part", "tips", ("part", "fixture"), [entry], Goal(mode, d, axis_point, travel),
                    {**LIMITS, **limits}, ("reached",))


def const(**parts):
    a = neutral_action()
    for k, v in parts.items():
        a[int(k[1:])] = v
    return lambda obs: a


def lateral(x, d=D):
    r = x - P0
    return np.linalg.norm(r - np.outer(r @ d, d), axis=1)


def test_dims_and_contract_defaults():
    assert (OBS_STEP, OBS_DIM, ACT_DIM) == (38, 38 * CFG["stack"], 11)
    assert neutral_action().shape == (ACT_DIM,) and not neutral_action().any()
    c = Contract("move_part", "tips", (), [], Goal("push", D))
    assert c.limits == LIMITS and c.limits is not LIMITS
    seen = []
    Skill(lambda o: seen.append(o) or neutral_action()).run(FakeIO(), contract(), max_steps=3)
    assert all(o.shape == (OBS_DIM,) for o in seen) and len(seen) == 3
    assert not seen[0][:-OBS_STEP].any()                      # zero-padded before the first step
    last = seen[0][-OBS_STEP:]
    assert np.allclose(last[3:9], np.eye(3)[:, :2].T.ravel()) and last[21] == 0.8
    assert np.allclose(last[29:32], [1, 0, 0]) and np.allclose(last[32:35], R0.T @ D)
    assert last[35] == 0.2 and last[36] == 0.0 and last[37] == 1.0


def test_push_base_moves_along_d():
    io = FakeIO()
    tr = Skill().run(io, contract(), max_steps=40)
    a = tr.arrays()
    assert tr.stop == "time" and tr.steps == 40
    assert (a["p"][-1] - P0) @ D > 0.08 and lateral(a["p"]).max() < 1e-9
    assert np.allclose(a["p_cmd"][0], P0 + CFG["lead"] * D)
    assert np.allclose(a["rv_cmd"], mat_to_rotvec(R0)) and set(io.grips) == {1.0}


def test_neutral_action_equals_base():
    a = Skill().run(FakeIO(), contract(), max_steps=30).arrays()
    b = Skill(const()).run(FakeIO(), contract(), max_steps=30).arrays()
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_effort_changes_the_step_length():
    lead = {}
    for e in (-1.0, 0.0, 1.0):
        a = Skill(const(a6=e)).run(FakeIO(), contract(speed=1.0), max_steps=20).arrays()
        before = np.vstack([P0, a["p"][:-1]])
        lead[e] = (a["p_cmd"] - before) @ D
    assert np.allclose(lead[-1.0], 0.3 * CFG["lead"]) and np.allclose(lead[0.0], CFG["lead"])
    assert np.allclose(lead[1.0], 2.0 * CFG["lead"])


def test_speed_limit_scales_the_lead():
    free = Skill().run(FakeIO(), contract(), max_steps=20).arrays()
    slow = Skill().run(FakeIO(), contract(speed=0.03), max_steps=20).arrays()
    assert (slow["p"][-1] - P0) @ D < 0.8 * ((free["p"][-1] - P0) @ D)


def test_force_stop_backs_off():
    io = FakeIO(wall=(D, P0 @ D + 0.03))
    tr = Skill().run(io, contract())
    a = tr.arrays()
    n_back = CFG["backoff_steps"]
    s = tr.steps - n_back - 1                                # the step that stopped the contact
    assert tr.stop == "force" and tr.steps == len(a["p"])
    f = np.vstack([np.zeros((2, 3)), a["f"]])                # the entry reading is zero by taring
    lp = np.linalg.norm((f[:-2] + f[1:-1] + f[2:]) / 3, axis=1)
    over = lp > LIMITS["force"]
    assert over[s] and over[s - 1] and not over[:s - 1].any()
    assert np.allclose(a["p_cmd"][-n_back:], a["p"][s] - CFG["backoff"] * D)
    assert (a["p"][-1] - a["p"][s]) @ D < -0.004
    assert not a["action"][-n_back:].any()


def test_tube_clips_the_command():
    side = R0.T @ np.cross(D, [0.0, 0.0, 1.0])               # a lateral direction, in C
    pol = const(**{f"a{i}": x for i, x in enumerate(side / np.abs(side).max())})
    a = Skill(pol).run(FakeIO(), contract(tube=0.02), max_steps=60).arrays()
    lat = lateral(a["p_cmd"])
    assert lat.max() < 0.02 + 1e-9 and np.isclose(lat[-1], 0.02)
    wide = Skill(pol).run(FakeIO(), contract(tube=1.0), max_steps=60).arrays()
    assert lateral(wide["p_cmd"]).max() > 0.03


def test_tube_angle_clips_the_orientation_and_does_not_wind_up():
    calls = []

    def pol(obs):                                            # 20 steps one way, then back
        calls.append(0)
        return const(a3=1.0 if len(calls) <= 20 else -1.0)(obs)

    a = Skill(pol).run(FakeIO(), contract(tube_angle=30.0), max_steps=24).arrays()
    ang = np.degrees(np.linalg.norm(mat_to_rotvec(rotvec_to_mat(a["rv_cmd"]) @ R0.T), axis=1))
    assert np.allclose(ang[:10], 3.0 * np.arange(1, 11)) and np.allclose(ang[10:20], 30.0)
    assert np.allclose(ang[20:], 30.0 - 3.0 * np.arange(1, 5))


def test_tube_follows_the_steered_path():
    # a door's handle moves on an arc: steering turns the direction, and the tube goes with it (M1 second half);
    # with the tube about the initial line alone the hand could never leave it by more than the tube
    a = Skill(const(a7=1.0)).run(FakeIO(), contract(tube=0.02), max_steps=40).arrays()
    assert lateral(a["p"]).max() > 0.05


def test_steer_turns_the_direction():
    a = Skill(const(a7=1.0)).run(FakeIO(), contract(tube=1.0), max_steps=6).arrays()
    step = a["p_cmd"] - np.vstack([P0, a["p"][:-1]])
    ang = np.degrees(np.arccos(step @ D / np.linalg.norm(step, axis=1)))
    assert np.allclose(ang, CFG["steer"] * np.arange(1, 7))


def test_time_stop():
    assert Skill().run(FakeIO(), contract(time=1.0)).steps == 20
    tr = Skill().run(FakeIO(), contract(time=1.0), max_steps=5)
    assert tr.stop == "time" and tr.steps == 5


def test_done_flag():
    calls = []

    def pol(obs):
        calls.append(0)
        return const(a10=1.0 if len(calls) == 6 else -1.0)(obs)

    tr = Skill(pol).run(FakeIO(), contract())
    assert tr.stop == "done" and tr.steps == 5


def test_guard_stop():
    tr = Skill().run(FakeIO(), contract(), guard=lambda io: (io.p - P0) @ D > 0.02)
    a = tr.arrays()
    assert tr.stop == "guard" and (a["p"][-1] - P0) @ D > 0.02 and (a["p"][-2] - P0) @ D <= 0.02


def test_pull_closes_first_and_obeys_the_gripper_action():
    io = FakeIO()
    n = CFG["close_steps"]
    a = Skill(const(a9=-1.0)).run(io, contract("pull"), max_steps=n + 5).arrays()
    assert np.allclose(a["p_cmd"][:n], P0) and np.allclose(a["p"][:n], P0)
    assert a["gap"][n - 1] == 0.0 and io.grips == [1.0] * n + [-1.0] * 5
    assert ((a["p"][-1] - P0) @ D) > 0.01
    io = FakeIO()
    Skill(const(a9=-1.0)).run(io, contract("push"), max_steps=5)
    assert io.grips == [1.0] * 5                             # a push keeps the fingers closed


def test_turn_base_turns_about_the_axis():
    axis = R0[:, 2]
    point = P0 + 0.05 * R0[:, 0]
    n, m = CFG["close_steps"], 40
    io = FakeIO(alpha=0.5)
    tr = Skill().run(io, contract("turn", axis, np.radians(90), point), max_steps=n + m)
    a = tr.arrays()
    want = np.radians(CFG["turn_rate"]) * CFG["dt"] * m       # 60 deg: past tube_angle, the tube follows the turn
    assert np.allclose(rotvec_to_mat(a["rv_cmd"][-1]), rotvec_to_mat(want * axis) @ R0)
    r = a["p_cmd"][n:] - point
    assert np.allclose(r @ axis, 0.0) and np.allclose(np.linalg.norm(r, axis=1), 0.05)
    assert np.allclose(a["p_cmd"][-1], point + rotvec_to_mat(want * axis) @ (P0 - point))
    turned = mat_to_rotvec(io.R @ R0.T)
    assert np.allclose(np.cross(turned, axis), 0.0, atol=1e-9) and want - 0.1 < turned @ axis < want
    assert abs(np.linalg.norm(io.p - point) - 0.05) < 0.005 and np.allclose(a["p_cmd"][:n], P0)


def test_put_and_grasp_are_later_milestones():
    with pytest.raises(NotImplementedError):
        Skill().run(FakeIO(), contract("put"))


def test_trace_arrays_shapes():
    tr = Skill().run(FakeIO(), contract(), max_steps=7)
    a = tr.arrays()
    assert {k: v.shape for k, v in a.items()} == {
        "p": (7, 3), "rv": (7, 3), "v": (7, 3), "w": (7, 3), "f": (7, 3), "t": (7, 3), "gap": (7,),
        "p_cmd": (7, 3), "rv_cmd": (7, 3), "action": (7, ACT_DIM)}
    assert tr.R_c.shape == (3, 3) and tr.p_c.shape == (3,)
    assert Trace(R0, P0).arrays()["p"].shape == (0, 3)


def test_way_in_touches_then_re_anchors():
    a = R0[:, 2]
    io = FakeIO(wall=(a, a @ P0 + 0.04))            # the part stands 4 cm ahead along the approach axis
    tr = Skill().run(io, contract(d=a), way_in=0.06)
    x = tr.arrays()
    assert tr.touch > 0 and not x["action"][:tr.touch].any()        # the policy is not asked on the way in
    assert 0.04 < (tr.p_c - P0) @ a < 0.052                         # the contact frame sits at the touch
    assert lateral(x["p"], a).max() < 1e-9
    assert tr.stop == "force"                                       # then the push loads the wall up to the limit
    free = Skill().run(FakeIO(), contract(d=a), way_in=0.06, max_steps=80)
    assert abs((free.p_c - P0) @ a - 0.09) < 0.012                  # nothing there: way_in + approach_over, less lag
