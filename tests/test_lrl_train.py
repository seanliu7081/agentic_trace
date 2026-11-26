"""Training side of the lrl move_part policy (LRL_PLAN.md 6.3, M1): PPO pieces on their own, then StageEnv in the
three modes, injected failures, labels and dumps, a small vector in the real simulator, the detector's operating
point. CPU, no rendering."""
import pickle

import numpy as np
import pytest
import torch

from lrl_train import ppo
from lrl_train.env import ACT_DIM, CFG, CRIT_DIM, OBS_DIM, REWARD, TRAIN_KINDS, UNSEEN_KINDS, VAL_SEED, StageEnv
from lrl_train.vec import VecEnv

DRAWERS = CFG | {"kinds": ("drawer_stack",), "pull_share": 0.0, "inject": 0.0}


def trained_a_little(seed=0):
    """A model whose weights and running statistics have left their initial values."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM)
    n = 512
    obs, crit = 3.0 * rng.normal(size=(n, OBS_DIM)) + 1.0, rng.normal(size=(n, CRIT_DIM))
    act, logp, _ = model.act(obs, crit)
    ppo.PPO(model).update(obs, crit, act, logp, rng.normal(size=n), rng.normal(size=n))
    return model, obs


def test_checkpoint_round_trip_reproduces_actions(tmp_path):
    model, obs = trained_a_little()
    assert {"obs_norm.mean", "obs_norm.var", "obs_norm.count", "crit_norm.mean"} <= set(model.state_dict())
    assert abs(float(model.obs_norm.mean.mean()) - 1.0) < 0.2          # the statistics saw the batch
    ppo.save(tmp_path / "m.pt", model, {"steps": 7})
    loaded, meta = ppo.load(tmp_path / "m.pt")
    assert meta == {"steps": 7}
    a, b = ppo.Policy(model, seed=3), ppo.Policy(loaded, seed=3)
    for o in obs[:20]:
        np.testing.assert_array_equal(a(o), b(o))
    with torch.no_grad():
        o = torch.as_tensor(obs)
        assert torch.equal(model.dist(o).mean, loaded.dist(o).mean)
    # a fresh model with untouched statistics acts differently: the round trip is not passing by accident
    fresh = ppo.Policy(ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM), seed=3)
    assert np.abs(fresh(obs[0])[:10] - a(obs[0])[:10]).max() > 1e-6


def test_untrained_policy_is_the_base_law_and_rarely_raises_done():
    torch.manual_seed(0)
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM)
    obs, crit = np.random.default_rng(0).normal(size=(20000, OBS_DIM)), np.zeros((20000, CRIT_DIM))
    a = model.act(obs, crit)[0]
    assert (a[:, -1] > 0).mean() < 0.01
    with torch.no_grad():
        assert model.dist(torch.as_tensor(obs)).mean[:, :-1].abs().max() < 0.2


def test_gae_by_hand():
    g, lam = 0.9, 0.5
    adv, ret = ppo.gae([1.0, 2.0, 3.0], np.array([0.5, 0.4, 0.3]), [0.0, 1.0, 0.0], 2.0, g, lam)
    d2 = 3.0 + g * 2.0 - 0.3
    d1 = 2.0 - 0.4                                  # the episode ended here: nothing is carried across
    d0 = 1.0 + g * 0.4 - 0.5
    np.testing.assert_allclose(adv, [d0 + g * lam * d1, d1, d2])
    np.testing.assert_allclose(ret, adv + [0.5, 0.4, 0.3])


def test_ppo_update_moves_the_mean_towards_the_better_action():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM)
    algo = ppo.PPO(model)
    obs, crit = rng.normal(size=(4096, OBS_DIM)), rng.normal(size=(4096, CRIT_DIM))
    for _ in range(5):
        act, logp, _ = model.act(obs, crit)
        algo.update(obs, crit, act, logp, act[:, 0].astype(float), np.zeros(4096))     # advantage: a[0] itself
    with torch.no_grad():
        mean = model.dist(torch.as_tensor(obs)).mean.mean(0)
    assert mean[0] > 0.05 and mean[1:-1].abs().max() < mean[0]


def test_a_plinth_too_thin_to_write_is_left_out():
    # training run 3 ended on a generated layout whose plinth height was written as a box of zero size
    from lrl_train import scenes
    lo, hi = [-0.2, -0.1, 4e-6], [0.0, 0.1, 0.2]
    thin = scenes._carcass("m", lo, hi, False, 500.0, 4e-6)
    assert "m_plinth" not in thin and "m_plinth" in scenes._carcass("m", lo, hi, False, 500.0, 0.05)
    scenes._finish(f'<body name="m">{thin}</body>', [])                         # compiles


def test_training_never_sees_the_unseen_kinds():
    assert not set(TRAIN_KINDS) & set(UNSEEN_KINDS) and CFG["kinds"] == TRAIN_KINDS and CFG["seeds"][1] <= VAL_SEED


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    e = StageEnv(11, tmp_path_factory.mktemp("bddl"), DRAWERS)
    yield e
    e.close()


def test_stage_env_base_law_episode(env):
    obs, crit, info = env.reset(0.0)
    assert obs.shape == (OBS_DIM,) and crit.shape == (CRIT_DIM,) and info["kind"] == "drawer_stack"
    assert info["layout"] < VAL_SEED and info["way_in"] > 0
    assert crit[0] <= 1.0 and abs(np.linalg.norm(crit[2:5]) - 1.0) < 1e-6      # share of travel left, true push direction
    total, ended, n = 0.0, False, 0
    while not ended:
        obs, crit, r, ended, end = env.step(np.zeros(ACT_DIM))
        total += r
        n += 1
        assert np.isfinite(obs).all() and np.isfinite(crit).all() and (ended or end == {})
    assert end["stop"] in ("force", "time") and end["steps"] == n and abs(end["return"] - total) < 1e-9
    assert n + info["way_in"] <= round(CFG["time"]["push"] / 0.05)
    # no error, the base law, a generated drawer: reward follows the travel that was covered
    covered = 1.0 - end["left"] / max(env.left0, 1e-9)
    assert total <= REWARD["progress"] * covered + 1e-6
    assert not end["success"] or env.left0 == REWARD["travel_floor"] or total > 0.5 * REWARD["progress"]
    sim, part = env.sim, env.part
    drawn = sim.randomize(np.random.default_rng(0), part, CFG)
    dof = sim._adr(part.joint)[1]
    assert sim.robot.m.dof_damping[dof] == pytest.approx(sim._loaded[0][dof] * drawn["damping"])
    again = sim.randomize(np.random.default_rng(1), part, CFG)                  # factors do not pile up
    assert sim.robot.m.dof_damping[dof] == pytest.approx(sim._loaded[0][dof] * again["damping"])
    np.testing.assert_allclose(sim.robot.ctrl.kp, sim._loaded[5] * again["gain"])
    assert CFG["gain"][0] <= again["gain"] <= CFG["gain"][1]


def test_a_start_is_run_again_from_the_same_state(tmp_path):
    env = StageEnv(4, tmp_path, DRAWERS | {"episodes_per_start": 2}, randomize=False)

    def episode():
        obs, crit, info = env.reset(0.0)
        seen, ended = [obs], False
        while not ended:
            obs, _, r, ended, end = env.step(np.zeros(ACT_DIM))
            seen.append(obs)
        return info, np.array(seen), end

    (i1, o1, e1), (i2, o2, e2), (i3, o3, e3) = episode(), episode(), episode()
    # clean sensors: the same start. The first steps repeat exactly; on this start the two runs part by up to 1e-3 in
    # the observation from the tenth step on (state outside MuJoCo's integration state, not found: train2/NOTES.md)
    assert i1 == i2 and o1.shape == o2.shape and np.abs(o1[:8] - o2[:8]).max() < 1e-9
    assert e1["return"] == pytest.approx(e2["return"], abs=0.05) and e1["steps"] == e2["steps"]
    assert (i3["part"], i3["left0"], i3["family"]) != (i1["part"], i1["left0"], i1["family"])     # then a new start
    env.close()


def test_done_on_an_open_part_ends_the_episode_with_the_penalty(env):
    env.reset(1.0)
    a = np.zeros(ACT_DIM)
    a[-1] = 1.0
    _, _, r, ended, end = env.step(a)
    assert ended and end["stop"] == "done" and not end["closed_at_end"]
    assert r == pytest.approx(REWARD["done_open"] - REWARD["step"], abs=0.05)
    assert end["steps"] == 1


def test_vector_resets_are_explicit(tmp_path):
    vec = VecEnv(2, tmp_path, seed=5, cfg=DRAWERS)
    out = vec.reset(0.5)
    assert out.live.all() and out.fresh.all() and not out.stepped.any()
    assert all(i["scale"] == 0.5 for i in out.info)
    a = np.zeros((2, ACT_DIM))
    a[0, -1] = 1.0                                  # environment 0 raises done at once
    seen_end = seen_fresh = False
    steps = np.zeros(2, int)
    for _ in range(4000):
        out = vec.step(a)
        assert not (out.stepped & out.fresh).any()                              # a row is a transition or a first obs
        assert (out.live == ((out.stepped & ~out.ended) | out.fresh)).all()
        assert not out.reward[~out.stepped].any() and not out.ended[~out.stepped].any()
        steps += out.stepped
        if out.ended[0]:
            assert out.info[0]["stop"] == "done" and not out.live[0]
            seen_end = True
        if out.fresh[0]:
            assert seen_end and "family" in out.info[0]                         # the reset came after the terminal step
            seen_fresh = True
        if seen_fresh and steps[1] > 0:
            break
    assert seen_fresh and steps[1] > 0
    for o in vec.settle():
        assert not o.fresh[o.stepped].any()
    assert not vec.busy.any()
    vec.close()


def _episode(env, scale=0.0):
    obs, crit, info = env.reset(scale)
    ended = False
    while not ended:
        obs, crit, r, ended, end = env.step(np.zeros(ACT_DIM))
        assert np.isfinite(obs).all() and np.isfinite(crit).all()
    return info, end


def test_pull_and_turn_episodes_label_and_dump(tmp_path):
    from atrace.lrl_models import DETECTOR_CLASSES
    for kind, mode in (("wooden_cabinet", "pull"), ("flat_stove", "turn")):
        env = StageEnv(3, tmp_path / kind, CFG | {"kinds": (kind,), "pull_share": 1.0, "inject": 0.0,
                                                  "dump_share": 1.0, "dump_chunk": 2}, dump_dir=tmp_path / "dump")
        for _ in range(2):
            info, end = _episode(env)
            # no way in: the skill's first 12 steps close the pads around the handle
            assert info["mode"] == mode and info["way_in"] == 12 and env.contract.toucher == "pads"
            assert end["label"] in DETECTOR_CLASSES and np.isfinite(end["travel"])
            assert (end["label"] == "reached") == end["closed_at_end"]
        env.close()
    recs = [r for f in sorted((tmp_path / "dump").glob("*.pkl")) for r in pickle.load(open(f, "rb"))]
    assert len(recs) == 4 and {r["mode"] for r in recs} == {"pull", "turn"}
    for r in recs:
        # the pads closed for 12 steps first; a force stop holds three steps more
        assert r["steps"] + 12 <= len(r["trace"]["p"]) <= r["steps"] + 15 and r["touch"] == 0
        assert r["label"] in DETECTOR_CLASSES and r["goal"]["mode"] == r["mode"]


def test_a_blocked_part_does_not_move(tmp_path):
    env = StageEnv(5, tmp_path, CFG | {"kinds": ("wooden_cabinet",), "pull_share": 1.0, "inject": 1.0})
    seen, blocked = set(), 0
    for _ in range(60):
        info, end = _episode(env)
        if info["inject"] == "blocked":
            assert end["label"] != "reached" and abs(end["travel"]) < 0.01
            blocked += 1
            if blocked == 2:
                break
        seen.add(info["inject"])
    assert seen <= {"wrong_axis", "blocked", "no_contact", "slip"} and blocked == 2
    env.close()


def test_reached_threshold_is_the_lowest_with_the_precision():
    from lrl_train.train_detector import threshold_for
    p = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    y = np.array([True, True, True, False, True, False])
    assert threshold_for(p, y, 0.98) == 0.9                 # 0.7 has 4 of 5 = 80 %
    assert threshold_for(p, y, 0.75) == 0.7
    assert threshold_for(p, np.zeros(6, bool), 0.98) == 1.0
