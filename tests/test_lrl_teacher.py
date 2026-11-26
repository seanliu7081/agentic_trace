"""LRL_PLAN M1E step 2: the teacher's scenes, environment, vector and loader (stage simulator, CPU)."""
import functools

import mujoco
import numpy as np
import pytest

from atrace.lrl_e2e import ACT_DIM, E2E_CFG, PROPRIO_DIM, Belief
from lrl_train import ppo, scenes
from lrl_train.teacher_env import CFG, CRIT_DIM, OBS_DIM, PRIV_DIM, REWARD, TeacherEnv, span_of
from lrl_train.train_teacher import bc_anchor, load_teacher
from lrl_train.vec import VecEnv


def test_new_handle_kinds_geometry():
    """Low bars sit near the bottom edge of their front; a recess is a slot through the front."""
    seen = set()
    for i in range(60):
        xml, parts = scenes.E2E_GENERATED["drawer_e2e"](np.random.default_rng([5, i]))
        m = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{xml}</worldbody></mujoco>")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        scenes.place_parts(parts, m, d)
        for p in parts:
            seen.add(p.handle_kind)
            bottom = p.face_point[2] - p.half_extents[1]
            if p.handle_kind == "low_bar":
                assert 0.012 - 1e-3 <= p.handle_point[2] - bottom <= 0.035 + 1e-3
                assert 0.012 - 1e-6 <= p.protrusion <= 0.032 + 1e-6
            if p.handle_kind == "recess":
                below = p.handle_point - [0, 0, p.handle_half[1] + 0.005]       # just under the top strip
                gid = np.zeros(1, np.int32)
                dist = mujoco.mj_ray(m, d, below + 0.05 * p.face_normal, -p.face_normal, None, 1, -1, gid)
                assert dist < 0 or dist > 0.05 + scenes.CFG["panel_t"] + 1e-3   # through the front
                assert p.protrusion == 0.0
    assert {"low_bar", "recess", "bar", "none"} <= seen


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    # no sensor delay (the test moves the arm under the skill); region starts only (no near starts of the reverse
    # curriculum, whose distance is 0)
    e = TeacherEnv(3, tmp_path_factory.mktemp("teacher"), CFG | {"kinds": ("drawer_e2e",), "episodes_per_start": 1,
                                                                 "near_share": 0.0}, randomize=False)
    yield e
    e.close()


def test_reset_start_and_observation(env):
    obs, crit, info = env.reset(0.0)
    assert obs.shape == (OBS_DIM,) and crit.shape == (CRIT_DIM,) and OBS_DIM == PROPRIO_DIM + PRIV_DIM
    assert np.isfinite(obs).all() and np.isfinite(crit).all()
    assert not env.sim._touching()
    part, A = env.part, env.frame
    np.testing.assert_allclose(A.o, part.face_point, atol=1e-9)              # scale 0: no hand-off error
    assert info["err_pos"] < 1e-9 and info["err_angle"] < 1e-3 and not info["wrong"]
    assert 0.04 - 1e-9 <= info["dist"] <= 0.08 + 1e-9
    priv = obs[PROPRIO_DIM:]
    np.testing.assert_allclose(priv[:3], E2E_CFG["p_scale"] * A.point(part.face_point), atol=1e-9)
    np.testing.assert_allclose(priv[3:6], A.vec(part.face_normal), atol=1e-9)
    p = env.sim.robot.eef_pose()[1]
    assert A.point(p)[0] > 0.03                                              # the start lies on +x of A


def test_goal_progress_and_done(env):
    env.reset(0.0)
    s_open, end, span = span_of(env.part)
    x_goal = span if env.dirn > 0 else 0.0
    if env.amount is not None:
        x_goal = env.x0 + env.dirn * env.amount
    q_goal = env.part.closed + s_open * x_goal
    assert not env._reached()
    # the part is set, not pushed: a done step does not step the simulator, so nothing moves it back
    env.sim.set_joint(env.part.joint, env.part.closed + s_open * (env.x0 + 0.5 * (x_goal - env.x0)))
    assert env._left() == pytest.approx(0.5 * env.left0, rel=1e-6) and not env._reached()
    env.sim.set_joint(env.part.joint, q_goal)
    assert env._left() < 1e-9 and env._reached()
    env.rw = dict(env.rw, done_reached=100.0)
    a = np.zeros(ACT_DIM)
    a[7] = 1.0
    _, _, r, ended, info = env.step(a)
    assert ended and info["stop"] == "done" and info["success"] and info["label"] == "reached"
    # the done bonus and the whole progress, less at most the shaping of a handle moved 11 cm past a still hand
    assert r > 100.0 + env.rw["progress"] - 1.1


def test_belief_hook(tmp_path):
    seen = {}

    def belief_fn(sim, part):
        seen["arm"] = sim.robot.q()[:7].copy()
        return Belief(part.face_point + [0.0, 0.0, 0.02], [1.0, 0.0, 0.0], np.zeros((5, 3)))
    e = TeacherEnv(4, tmp_path, CFG | {"kinds": ("drawer_e2e",)}, belief_fn=belief_fn)
    e.reset(0.0)
    np.testing.assert_allclose(e.frame.o, e.part.face_point + [0.0, 0.0, 0.02], atol=1e-9)
    np.testing.assert_allclose(seen["arm"], e.sim.q_home, atol=1e-6)        # read with the arm at rest
    e.close()


def test_load_teacher_matches_actor(tmp_path):
    model = ppo.ActorCritic(OBS_DIM, CRIT_DIM, ACT_DIM, ppo.CFG | {"hidden": 32, "layers": 2})
    model.obs_norm.update(np.random.default_rng(0).normal(size=(50, OBS_DIM)))
    ppo.save(tmp_path / "t.pt", model)
    act = load_teacher(tmp_path / "t.pt")
    obs = np.random.default_rng(1).normal(size=(4, OBS_DIM))
    import torch
    with torch.no_grad():
        want = model.actor(model.obs_norm(torch.as_tensor(obs))).numpy()
    np.testing.assert_allclose(act(obs), want, atol=1e-6)
    assert act.done_std == pytest.approx(ppo.CFG["done_std0"], rel=1e-5)


def test_vector_runs_teacher_env(tmp_path):
    vec = VecEnv(2, tmp_path, 0, CFG | {"kinds": ("knob_e2e",)}, REWARD, env_cls=functools.partial(TeacherEnv, source="t"),
                 dims=(OBS_DIM, CRIT_DIM))
    out = vec.reset(0.0)
    assert all(o.shape == (OBS_DIM,) for o in out.obs) and out.live.all()
    for _ in range(3):
        out = vec.step(np.zeros((2, ACT_DIM)))
    vec.settle()
    vec.close()


def test_e2e_placement_turns_fronts_up_to_90_deg_within_reach():
    """E2E layouts (e2 request 5): fronts up to 90 deg from facing the robot, the point 15 cm out within reach_start;
    M1's generators keep 60 deg."""
    base = np.array(scenes.CFG["base_xy"])
    seen = []
    for i in range(80):
        xml, parts = scenes.E2E_GENERATED["door_e2e"](np.random.default_rng([6, i]))
        p = parts[0]
        n = p.face_normal[:2] / np.linalg.norm(p.face_normal[:2])
        back = (base - p.face_point[:2]) / np.linalg.norm(base - p.face_point[:2])
        seen.append(np.degrees(np.arccos(np.clip(n @ back, -1, 1))))
    assert max(seen) <= 90.0 + 0.5              # the face centre is the mount's origin: the angle is the yaw drawn
    assert max(seen) > 65.0                     # beyond M1's 60 deg
    for i in range(40):
        xml, parts = scenes.door(np.random.default_rng([6, i]))
        n = parts[0].face_normal[:2] / np.linalg.norm(parts[0].face_normal[:2])
        back = base - parts[0].face_point[:2]
        assert np.degrees(np.arccos(np.clip(n @ back / np.linalg.norm(back), -1, 1))) <= 60.0 + 0.5


def test_door_counts_open_at_end_tol_door_and_neighbours_open_only_for_a_close(tmp_path):
    e = TeacherEnv(8, tmp_path, CFG | {"kinds": ("door_e2e",), "near_share": 0.0, "amount": 0.0, "inject": 0.0},
                   randomize=False)
    while True:
        e.reset(0.0)
        if e.effect == "open":
            break
    s_open, _, span = span_of(e.part)
    e.sim.set_joint(e.part.joint, e.part.closed + s_open * (1 - CFG["end_tol_door"] - 0.02) * span)
    assert not e._reached()
    e.sim.set_joint(e.part.joint, e.part.closed + s_open * (1 - CFG["end_tol_door"] + 0.02) * span)
    assert e._reached()
    e.close()
    e = TeacherEnv(9, tmp_path, CFG | {"kinds": ("drawer_e2e",), "near_share": 0.0, "others_open": 1.0, "inject": 0.0},
                   randomize=False)
    for _ in range(12):
        e.reset(0.0)
        others = [o for o in e.sim.parts if o is not e.part]
        if others:
            opened = any(abs(e.sim.joint(o.joint) - o.closed) > 1e-6 for o in others)
            assert opened == (e.effect == "close")
    e.close()


def test_expert_carries_a_grasped_drawer_open_through_the_skill(tmp_path):
    """The privileged expert from grasped starts on a cabinet drawer, in the training setting: its actions stay in
    bounds, and it brings the part to the goal through lrl_e2e's action space and ends on its done (some layouts put
    the open drawer beyond the arm's reach: then it reports no contact, so one success in three is asked)."""
    from lrl_train.expert import Expert
    e = TeacherEnv(12, tmp_path, CFG | {"kinds": ("wooden_cabinet",), "near_share": 1.0, "near_grasped": 1.0,
                                        "inject": 0.0, "amount": 0.0, "episodes_per_start": 1})
    ex = Expert(e)
    ok = []
    for _ in range(12):
        e.reset(0.0)
        if e.info["near"] != "grasped":
            continue
        ex.start()
        while True:
            a = ex()
            assert a.shape == (ACT_DIM,) and np.all(np.abs(a) <= 1.0)
            _, _, _, ended, info = e.step(a)
            if ended:
                break
        ok.append(info["success"] and info["stop"] == "done")
        if len(ok) == 3:
            break
    e.close()
    assert any(ok)


def test_bc_collect_train_and_anchor(tmp_path):
    """bc.py: a collection job writes kept steps with labels and returns; training writes a ppo checkpoint that
    load_teacher reads; the fine-tuning anchor gives a finite loss."""
    from lrl_train import bc
    cfg_env = dict(bc.ENV)
    bc.ENV.update({"kinds": ("flat_stove",), "near_share": 1.0, "near_grasped": 1.0, "inject": 0.0})
    try:
        stats, meta = bc.collect_job((0, 3, str(tmp_path), None))
    finally:
        bc.ENV.clear()
        bc.ENV.update(cfg_env)
    assert stats["episodes"] == 3 and len(meta) == 3
    if not stats["kept"]:
        pytest.skip("no expert episode reached the goal in this draw")
    d = bc.load(tmp_path)
    n = len(d["obs"])
    assert d["obs"].shape == (n, OBS_DIM) and d["crit"].shape == (n, CRIT_DIM) and d["act"].shape == (n, ACT_DIM)
    assert d["first"].sum() == stats["kept"] and np.isfinite(d["ret"]).all()
    old = dict(bc.CFG)
    bc.CFG.update({"epochs": 1, "hidden": 32, "layers": 2})
    try:
        import argparse
        bc.train(argparse.Namespace(data=tmp_path, out=tmp_path / "bc.pt", device="cpu"))
    finally:
        bc.CFG.clear()
        bc.CFG.update(old)
    act = load_teacher(tmp_path / "bc.pt")
    assert act(d["obs"][:4]).shape == (4, ACT_DIM)
    model = ppo.load(tmp_path / "bc.pt")[0]
    assert np.isfinite(bc_anchor(tmp_path, "cpu")(model).item())
