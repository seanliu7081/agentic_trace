"""M1E track E0 (LRL_PLAN.md M1E step 1): the skill's base-law switch. With the default the skill is M1's, bit for
bit; with the base law off the policy's position and orientation numbers are the whole motion, and the training
environment charges the change of action instead of the residual's size."""
import hashlib

import numpy as np

from atrace.geometry import mat_to_rotvec, rotvec_to_mat
from atrace.lrl_contract import LIMITS, Contract, Entry, Goal
from atrace.lrl_skill import ACT_DIM, CFG, Skill, neutral_action

R0 = rotvec_to_mat([0.3, -0.5, 0.8])
P0 = np.array([0.5, 0.1, 0.3])
D = np.array([1.0, 1.0, 0.5]) / 1.5
FREE = CFG | {"base": False}


class Hand:
    """A hand that follows the commanded pose with a first-order lag; past a wall plane it reports a spring force,
    the fingers move 1 cm per step under a gripper command (as test_lrl_skill.FakeIO)."""

    def __init__(self, alpha=0.3, wall=None, k_wall=3000.0):
        self.R, self.p, self.alpha, self.wall, self.k_wall, self.gap = R0.copy(), P0.copy(), alpha, wall, k_wall, 0.08

    def step(self, R_goal, p_goal, grip=None, posture=None):
        self.p = self.p + self.alpha * (p_goal - self.p)
        self.R = rotvec_to_mat(self.alpha * mat_to_rotvec(R_goal @ self.R.T)) @ self.R
        self.gap = float(np.clip(self.gap - 0.01 * grip, 0.0, 0.08))

    def eef_pose(self):
        return self.R.copy(), self.p.copy()

    def wrench(self):
        if self.wall is None:
            return np.zeros(3), np.zeros(3)
        n, off = self.wall
        return -self.k_wall * max(n @ self.p - off, 0.0) * n, np.array([0.1, -0.2, 0.05]) * (n @ self.p - off > 0)

    def finger_opening(self):
        return self.gap


def contract(mode, d=D, travel=0.2, axis_point=None, **limits):
    return Contract("move_part", "tips" if mode == "push" else "pads", ("part",), [Entry(R0, P0, R0[:, 2], (0.0, 0.05))],
                    Goal(mode, d, axis_point, travel), {**LIMITS, **limits})


def scenario(mode, cfg=CFG, n=70, seed=0):
    """A contact driven by random actions over all eleven numbers (beyond [-1, 1] at times, done kept off): every
    observation and every trace row, hashed."""
    rng = np.random.default_rng(seed)
    if mode == "push":
        a = R0[:, 2]
        io, c, way_in = Hand(wall=(a, a @ P0 + 0.04), k_wall=700.0), contract("push", d=a), 0.06
    elif mode == "pull":
        io, c, way_in = Hand(wall=(D, D @ P0 + 0.06), k_wall=500.0), contract("pull"), 0.0
    else:
        io, c, way_in = Hand(alpha=0.5), contract("turn", R0[:, 2], np.radians(90), P0 + 0.05 * R0[:, 0]), 0.0
    sk = Skill(cfg=cfg)
    obs = [sk.start(io, c, max_steps=n, way_in=way_in)]
    while sk.stop is None:
        act = rng.uniform(-1.3, 1.3, ACT_DIM)
        act[10] = -1.0
        obs.append(sk.step(act))
    tr = sk.trace
    h = hashlib.sha256()
    for x in (np.array(obs), np.array(tr.rows), tr.R_c, tr.p_c, np.array([tr.touch, tr.steps], float)):
        h.update(np.ascontiguousarray(x, float).tobytes())
    h.update(tr.stop.encode())
    return h.hexdigest(), tr


# digests of scenario(mode) computed with M1's lrl_skill.py (sha256 513ffad5..., copy in
# runs/lrl_m1e_20261125/e0/m1_ref) before the switch was added
M1_DIGEST = {
    "push": "928ffedde5debd7e598b1fdfeb139b569c6cbce8e13ef1f066ebb277a75916e3",
    "pull": "0333dc74dd81d81f27a350c0ac6132137f97cdce92dabf0656841db90cf5ab95",
    "turn": "f0ec3a411b1d887fb125efed53cde33c9b2617a98b11a00d095fb76decea4317",
}


def test_default_is_m1_bit_for_bit():
    assert CFG["base"] is True
    for mode, digest in M1_DIGEST.items():
        assert scenario(mode)[0] == digest, mode


def run(mode, act, n, **limits):
    point = P0 + 0.05 * R0[:, 0]
    c = contract(mode, R0[:, 2], np.radians(90), point, **limits) if mode == "turn" else contract(mode, **limits)
    io = Hand(alpha=1.0)
    sk = Skill(cfg=FREE)
    sk.start(io, c, max_steps=n)
    for _ in range(n):
        sk.step(act)
    return sk.trace.arrays(), io


def test_free_neutral_action_holds_the_hand():
    k = CFG["close_steps"]
    for mode in ("push", "pull", "turn"):
        a, io = run(mode, neutral_action(), 30)
        assert np.allclose(a["p_cmd"], P0) and np.allclose(io.p, P0), mode
        assert np.allclose(rotvec_to_mat(a["rv_cmd"][-1]), R0) and len(a["p"]) == 30 + (mode != "push") * k


def test_free_numbers_are_the_whole_motion():
    act = neutral_action()
    act[0], act[6] = 1.0, 1.0              # position along x of C; effort does nothing without the base law
    a, _ = run("push", act, 5, speed=10.0, tube=1.0)
    step = a["p_cmd"] - np.vstack([P0, a["p"][:-1]])
    assert np.allclose(step, CFG["free_p"] * R0[:, 0])
    slow, _ = run("push", act, 5, speed=0.1, tube=1.0)          # the speed clip scales the policy's numbers
    assert np.linalg.norm(slow["p"][-1] - P0) < 0.9 * np.linalg.norm(a["p"][-1] - P0)


def test_free_turn_is_the_policys_and_the_tube_follows_it():
    axis, point, k, m = R0[:, 2], P0 + 0.05 * R0[:, 0], CFG["close_steps"], 15
    act = neutral_action()
    act[3:6] = R0.T @ axis                  # the orientation numbers along the believed axis, in C
    a, io = run("turn", act, m, speed=10.0)
    want = np.radians(CFG["free_r"]) * m    # 90 deg: past the 40 deg tube, which follows the believed path
    assert np.allclose(rotvec_to_mat(a["rv_cmd"][-1]), rotvec_to_mat(want * axis) @ R0)
    assert np.allclose(a["p_cmd"][-1], point + rotvec_to_mat(want * axis) @ (P0 - point))
    assert np.allclose(a["p_cmd"][:k], P0)


def test_env_charges_the_change_of_action_with_the_base_off(tmp_path):
    from lrl_train.env import CFG as ENV, REWARD, StageEnv
    act = np.r_[np.full(9, 0.2), 0.6, -1.0]
    rewards = {}
    for base in (True, False):
        env = StageEnv(5, tmp_path, ENV | {"base": base, "kinds": ("drawer_stack",), "inject": 0.0, "pull_share": 0.0},
                       REWARD | {"residual": 1000.0, "change": 1000.0}, randomize=False)
        env.reset(0.0)
        assert env.skill.cfg["base"] is base
        rewards[base] = [env.step(a)[2] for a in (np.r_[np.zeros(9), 0.6, -1.0], act, act)]
        env.close()
    # 1000 x 0.04 = 40 per charged step; every other term of a step is below 3 here
    assert rewards[True][0] > -3 and rewards[True][1] < -37 and rewards[True][2] < -37
    assert rewards[False][0] > -3 and rewards[False][1] < -37 and rewards[False][2] > -3


def test_free_checkpoints_are_published_apart(tmp_path, monkeypatch):
    import json
    from lrl_train import ppo, train_move_part as T
    from lrl_train.env import CRIT_DIM, OBS_DIM
    monkeypatch.setattr(T, "WEIGHTS", tmp_path)
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM)
    for base in (True, False):
        ppo.save(tmp_path / f"c{base}.pt", model, {"env": {"base": base}, "modes": ["push", "pull", "turn"],
                                                  "steps": int(base)})
        T.publish(tmp_path / f"c{base}.pt")
    man = json.loads((tmp_path / "manifest.json").read_text())
    kinds = {e["kind"]: (name, e.get("m1")) for name, e in man.items()}
    assert kinds["move_part"][1] is True and kinds["move_part_free"][1] is None
    assert kinds["move_part_free"][0].startswith("move_part_free_")
