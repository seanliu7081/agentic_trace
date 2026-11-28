"""Step 0 contracts and simulator locks. EGL, MuJoCo and project .venv required."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from atrace.gcp import (ContactContract, FAMILIES, GripStateObserver, LeakyPLAI, MultiHypothesisEntrySampler,
                        OBS_SHAPES, UniversalContactPolicy, decode_axis)


@pytest.mark.parametrize("dof", [6, 7])
def test_tared_effort_projection_preserves_tcp_axes_at_a_singularity(dof):
    from atrace.gcp import JointTorqueObserver
    from atrace.geometry import rotvec_to_mat

    scale = np.array([1., .8, .6, 1., .5, 0.])
    jacobian = np.zeros((6, dof))
    jacobian[:, :6] = np.diag(scale)
    limits = np.arange(10., 10. + dof)
    observer = JointTorqueObserver(limits, damping=.04)
    free = np.linspace(-2., 2., dof)
    bias = np.linspace(3., -.5, dof)
    rotation = rotvec_to_mat(np.array([0., 0., np.pi/2]))
    observer.tare(free + bias, free)
    np.testing.assert_allclose(observer.observe(free + bias, free, jacobian, rotation), 0., atol=1e-7)

    external = np.array([3., -4., 5., .5, .8, 1.2])
    residual = jacobian.T @ external
    residual[-1] += .7  # effort in a direction unobservable at the TCP
    observed = observer.observe(free + bias + residual, free, jacobian, rotation)
    fitted = external * scale**2 / (scale**2 + .04**2)
    expected = np.r_[rotation.T @ fitted[:3], rotation.T @ fitted[3:],
                     np.max(np.abs(residual) / limits), np.linalg.norm(residual - jacobian.T @ fitted)]
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


def test_coulomb_reversal_preserves_sign_in_deadband_and_transient_decays():
    from lrl_train.gcp_env import ReversalNoise

    rng = np.random.default_rng(47)
    noise = ReversalNoise(rng, np.full(6, 8.))
    assert np.all(noise.sigma > 0)
    noise.sigma[:] = 0.  # isolate the friction term from additive current noise
    forward = noise.sample(rng, np.full(6, .08))
    zero = noise.sample(rng, np.zeros(6))
    backward = noise.sample(rng, np.full(6, -.08))
    settling = noise.sample(rng, np.full(6, -.08))
    np.testing.assert_allclose(forward, noise.coulomb)
    np.testing.assert_array_equal(zero, forward)
    np.testing.assert_allclose(backward, -2 * noise.coulomb)
    assert np.all((backward < settling) & (settling < -noise.coulomb))


def test_grip_observation_recovers_from_commanded_close_with_external_opening():
    sensor = GripStateObserver((-.012, .004), .008)
    site = np.zeros(3)
    assert not sensor.update(0., .014, site, True)
    assert sensor.update(0., .014, site, True)
    assert not sensor.update(0., .035, site, True)
    for _ in range(4):
        assert not sensor.update(0., .075, site, True)
    assert not sensor.update(.08, .077, site, True)
    for width in (.045, .014, .014):
        assert not sensor.update(0., width, site, True)
    assert sensor.update(0., .014, site, True)
    assert not sensor.update(0., .014, [.03,0.,0.], True)


def test_narrow_grasp_requires_stopped_pads_and_releases_on_open_command():
    sensor = GripStateObserver((-.012, .004), .008)
    for gap in (.0049, .0044, .0040, .0035):
        assert not sensor.update(0., gap, np.zeros(3), True)
    assert not sensor.update(0., .00345, np.zeros(3), True)
    assert sensor.update(0., .00344, np.zeros(3), True)
    assert sensor.retains_grip(0., .00342)
    assert not sensor.retains_grip(.02, .00342)
    assert not sensor.update(.02, .00342, np.zeros(3), False)


def test_stationary_zero_command_leaks_unsaturated_pose_error_and_alpha_caps_stalls():
    from atrace.geometry import mat_to_rotvec, rotvec_to_mat
    R, p = rotvec_to_mat([.3, -.2, .05]), np.array([.4, .1, .9])
    dpos, drot = np.array([.002, -.001, .001]), np.array([.012, -.010, .018])
    core = LeakyPLAI(rotvec_to_mat(drot) @ R, p + dpos)
    for tick in range(1,21):
        cmd_R, cmd_p, _, _ = core.command(np.zeros(8), R, p)
        np.testing.assert_allclose(cmd_p - p, dpos * .975**tick, atol=1e-12)
        np.testing.assert_allclose(mat_to_rotvec(cmd_R @ R.T), drot * .975**tick, atol=1e-10)
    for scale, cap in [(-1., .004), (1., .014)]:
        act=np.ones(8)
        act[6]=scale
        for _ in range(12):
            cmd_R, cmd_p, alpha, _ = core.command(act, R, p)
            assert np.linalg.norm(cmd_p-p) <= cap + 1.e-12
            assert np.linalg.norm(mat_to_rotvec(cmd_R @ R.T)) <= np.deg2rad(6.)*alpha + 1.e-12
        assert np.linalg.norm(cmd_p-p) == pytest.approx(cap, abs=1e-12)


def test_insertion_corridor_rotates_the_held_grasp_offset():
    from atrace.geometry import rotvec_to_mat
    goal = np.array([.08, .11, .04])
    grasp = np.array([.025, .003, 0.])
    p = np.array([-.08, .11, .08])
    c = ContactContract("place", "held", "container", goal_region_ref=goal)
    for R in [np.eye(3), rotvec_to_mat([.2, .7, .0])]:
        for destination in (goal - R @ grasp, lambda rot: goal - rot @ grasp):
            paths = list(MultiHypothesisEntrySampler._motions(
                c, R, p, np.eye(3), [1., 0., 0.], .20, destination))
            assert len(paths) == 2
            for _, span, pose in paths:
                assert span > .04
                R_final, tcp = pose(1.)
                np.testing.assert_allclose(tcp + R_final @ grasp, goal, atol=1e-12)
            direct, raised = (path[-1] for path in paths)
            np.testing.assert_allclose(raised(.99)[1][:2], direct(1.)[1][:2], atol=1.e-8)
            assert raised(.99)[1][2] > raised(1.)[1][2]


@pytest.fixture(scope="session", autouse=True)
def select_render_device():
    from lrl_train.render import egl_device
    egl_device(int(os.environ.get("GCP_TEST_GPU", 0)))


@pytest.fixture
def env(tmp_path):
    from lrl_train.gcp_env import GCPEnv
    instance = GCPEnv(seed=0, out_dir=tmp_path, preset="all_5_families", staged_reset_prob=.5)
    try:
        yield instance
    finally:
        instance.close()


def test_one_actor_all_contracts_and_dimensional_contract():
    targets = {"pick": (.06, 0.), "turn": (0., -.6), "push_pull": (-.09, 0.),
               "place": (0., 0.), "planar_push": (0., 0.)}
    obs = {key: torch.zeros(len(FAMILIES), *shape) for key, shape in OBS_SHAPES.items()}
    for i, family in enumerate(FAMILIES):
        trans, angle = targets[family]
        region = [.2, -.14, .06] if family in ("place", "planar_push") else None
        c = ContactContract(family, "target", "mount", [1,0,0], [0,0,-1], trans, angle, region)
        header = c.goal_header(np.eye(3), np.eye(3), 0., 0., np.zeros(3), False, 1., .5, .3, .4)
        np.testing.assert_allclose(decode_axis(header[3:5]), [0., 0., -1.])
        obs["goal"][i] = torch.from_numpy(header)
        held = (family == "pick")
        assert c.goal_reached(trans, angle, region, held)
        assert not c.goal_reached(trans, angle, region, not held)
    policy = UniversalContactPolicy()
    out = policy(obs)
    assert out.shape == (5, 8) and torch.isfinite(out).all() and (out.abs() <= 1).all()
    out.mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


def test_zero_wrist_full_sampling_step_and_guard(env, monkeypatch):
    from atrace.body import ControllerModel
    from atrace.guard import ContactDetector
    from atrace.motion import Executor
    from atrace.sim_iface import SimRobot
    from lrl_train.fast_sim import FastSimRobot

    def forbidden(robot):
        raise RuntimeError("Read wrist F/T inside a sensorless path")

    with monkeypatch.context() as patch:
        patch.setattr(SimRobot, "wrench", forbidden)
        patch.setattr(FastSimRobot, "wrench", forbidden)
        obs, info = env.reset(seed=0, kind="drawer", staged=False)
        assert info["entry_ok"] and info["render_count"] == 1
        assert all(obs[key].shape == shape for key, shape in OBS_SHAPES.items())
        assert np.unique(obs["scene_pts"][:, 4]).tolist() == [0., 1.]
        actor = UniversalContactPolicy().eval()
        with torch.no_grad():
            action = actor({k: torch.from_numpy(v) for k, v in obs.items()}).numpy()
        _, _, _, after = env.step(action)
        assert after["render_count"] == 2 and after["steps"] == 1
        rb = env.ss.robot
        q = rb.q()[:7]
        path = np.stack([q, q + np.array([.005, 0, 0, 0, 0, 0, 0])])
        cd = ContactDetector(ControllerModel(rb.model))
        # Force-based user rules must be disabled along with wrist sampling.
        guard = cd.guard("kinematic_contact", rule=lambda *_: pytest.fail("force rule invoked"))
        result = Executor(rb, rb.model).run_path(path, guard=guard, settle=False)
        assert result.steps >= 2 and cd.k > 0
    for rb in (env.ss.robot, env.fast_rb):
        with pytest.raises(RuntimeError, match="Wrist F/T"):
            rb.wrench()


@pytest.mark.parametrize("family", ["pick", "turn", "planar_push"])
def test_additional_families_enter_and_execute_on_real_depth(env, family):
    obs, info = env.reset(seed=0, kind=family, staged=False)
    assert info["entry_ok"] and info["family"] == family
    assert set(obs) == set(OBS_SHAPES)
    new, reward, done, after = env.step(env.expert_action())
    assert not done and np.isfinite(reward)
    assert after["render_count"] == info["render_count"] + 1 and after["steps"] == 1
    assert all(new[k].shape == s and np.isfinite(new[k]).all() for k, s in OBS_SHAPES.items())


def test_contacted_staged_reset_matches_uninterrupted_first_step(env):
    _, info = env.reset(seed=0, kind="drawer", staged=False)
    assert info["entry_ok"]
    for _ in range(100):
        _, _, done, info = env.step(env.expert_action())
        if env.buffer or done:
            break
    assert len(env.buffer) == 1, "script never established verified contact for a staged snapshot"
    c = info["contact_facts"]
    assert c["firm"] or c["push_face_loaded"]
    before = {k: arr.copy() for k, arr in env.obs.items()}
    tare = env.observer.delta_tau0.copy()
    state = env.d.qpos.copy(), env.d.qvel.copy()
    action = env.expert_action()
    uninterrupted, reward, _, info1 = env.step(action)
    first_row = uninterrupted["history"][-1].copy()
    restored, restarted = env.reset(seed=0, kind="drawer", staged=True)
    assert restarted["staged"] and restarted["steps"] == info["steps"]
    for key in before:
        np.testing.assert_array_equal(before[key], restored[key])
    np.testing.assert_array_equal(env.observer.delta_tau0, tare)
    np.testing.assert_array_equal(env.d.qpos, state[0])
    np.testing.assert_array_equal(env.d.qvel, state[1])
    replayed, reward2, _, info2 = env.step(action)
    row = replayed["history"][-1]
    # Real load can be significant. A jump is the difference against the same
    # physical continuation with the same sensor noise and controller state.
    np.testing.assert_allclose(row[12:18], first_row[12:18], atol=1e-7, rtol=0)
    np.testing.assert_allclose(row[28:36], first_row[28:36], atol=1e-5, rtol=0)
    np.testing.assert_allclose(replayed["goal"], uninterrupted["goal"], atol=1e-6)
    assert reward2 == pytest.approx(reward, abs=1e-6)
    assert info2["target_delta"] == pytest.approx(info1["target_delta"], abs=1e-8)


@pytest.mark.parametrize("wall,normal_dim,inward_sign", [("roof", 2, -1), ("left", 1, 1)])
def test_held_box_against_cabinet_in_endpoint_and_substep_scanners(env, monkeypatch, wall, normal_dim, inward_sign):
    import xml.etree.ElementTree as ET
    import mujoco
    from atrace.geometry import rotvec_to_mat
    from lrl_train import gcp_env

    build_layout = gcp_env._layout

    def layout_with_movable_wall(*args, **kwargs):
        layout, fixture = build_layout(*args, **kwargs)
        cabinet = ET.fromstring(layout.mjcf[0])
        face = cabinet.find(f"geom[@name='{fixture.fixture_body}_{wall}']")
        # Compile the original cabinet panel with its own pose. The robot,
        # complete payload and other panels retain their ordinary collisions.
        panel = ET.Element("body", dict(cabinet.attrib, name="scan_panel", mocap="true"))
        cabinet.remove(face)
        panel.append(face)
        layout.mjcf[0] = ET.tostring(cabinet, encoding="unicode")
        layout.mjcf.append(ET.tostring(panel, encoding="unicode"))
        return layout, fixture

    monkeypatch.setattr(gcp_env, "_layout", layout_with_movable_wall)
    _, initial = env.reset(seed=1, kind="place", staged=False)
    assert initial["entry_ok"] and initial["contact_facts"]["target_opposition"]
    m, d = env.m, env.d
    plane = m.geom(f"{env.fixture.fixture_body}_{wall}").id
    box = m.geom("ucs_payload_box").id
    box_R = d.geom_xmat[box].reshape(3, 3)
    panel_R = d.geom_xmat[plane].reshape(3, 3).copy()
    if wall == "left":
        panel_R = rotvec_to_mat([0., 0., np.pi/2]) @ box_R
    extent = m.geom_size[plane] + np.abs(panel_R.T @ box_R) @ m.geom_size[box]
    # Engage the corner facing away from the hand. Leave a 12 mm overlap
    # along the face of the side panel and 3 mm along its normal.
    offset = (np.array([extent[0]-.012, 0., -extent[2]+.012]) if wall == "left" else
              np.array([.010-m.geom_size[plane, 0], 0., 0.]))
    offset[normal_dim] = -inward_sign * (extent[normal_dim] - .003)
    panel_origin = d.geom_xpos[box] + panel_R @ (offset - m.geom_pos[plane])
    pose_id = m.body_mocapid[m.geom_bodyid[plane]]
    d.mocap_pos[pose_id] = panel_origin
    mujoco.mju_mat2Quat(d.mocap_quat[pose_id], np.ascontiguousarray(panel_R).ravel())
    mujoco.mj_forward(m, d)
    assert env._contacts()["firm"]
    measured = max(-c.dist for c in d.contact[:d.ncon] if {box, plane} == {c.geom1, c.geom2})
    assert .0028 < measured <= .00301
    assert env.scan_penetration() == pytest.approx(measured, abs=1e-9)
    env._pairs(False)
    assert env.scan_penetration() < measured - .0005
    env._pairs(True)
    _, _, _, after = env.step(np.r_[np.zeros(6), -1., -1.])
    assert env.fast_rb.last_substep_max_pen_m >= measured - 1e-8
    assert after["max_pen_m"] == pytest.approx(
        max(initial["max_pen_m"], env.fast_rb.last_substep_max_pen_m, env.scan_penetration()), abs=1e-8)


def test_single_pad_transition_preserves_payload_wall_pair_until_release(env, monkeypatch):
    _, info = env.reset(seed=1, kind="place", staged=False)
    assert info["held"] and env.payload_owned
    target = env.m.geom("ucs_payload_box").id
    wall = env.m.geom(f"{env.fixture.fixture_body}_left").id
    assert env.pairs[target,wall]
    # Inject two contact-classifier observations into the same step path as the
    # real wall test. The open command rules out the encoder-obstruction branch.
    measure = env._contacts
    left = True
    def pad_transition():
        facts = measure()
        return dict(facts, left_target=left, right_target=False, firm=False,
                    target_opposition=False, width_m=.075)
    monkeypatch.setattr(env, "_contacts", pad_transition)
    open_action = np.r_[np.zeros(7), 1.]
    env.step(open_action)
    assert env.payload_owned and env.pairs[target,wall] and env.pairs[wall,target]
    left = False
    env.step(open_action)
    assert not env.payload_owned and not env.pairs[target,wall]


def test_exact_relative_goal_requires_ten_consecutive_settled_steps(env):
    _, info = env.reset(seed=0, kind="door", target_deg=75, staged=False)
    assert info["entry_ok"]
    action = np.r_[np.zeros(7), 1.]
    env.ss.set_joint(env.part.joint, env.initial_joint + np.deg2rad(60.))
    _, _, _, short = env.step(action)
    assert not short["goal_satisfied"] and not short["success"]
    env.ss.set_joint(env.part.joint, env.initial_joint + np.deg2rad(75.))
    _, _, _, transient = env.step(action)
    assert transient["goal_satisfied"] and not transient["success"]
    env.ss.set_joint(env.part.joint, env.initial_joint + np.deg2rad(60.))
    _, _, _, short = env.step(action)
    assert not short["success"] and env.good_steps == 0
    env.ss.set_joint(env.part.joint, env.initial_joint + np.deg2rad(75.))
    for i in range(10):
        _, _, _, settled = env.step(action)
        assert settled["goal_satisfied"]
        assert settled["success"] == (i == 9)


def test_exogenous_slip_is_not_written_into_actor_actions_or_references(env):
    import copy
    from atrace.geometry import mat_to_rotvec
    _, info = env.reset(seed=403561, kind="door", target_deg=75., staged=False, slip_at_deg=15.)
    assert info["entry_ok"]
    for _ in range(env.horizon-10):
        if info["angle_deg"] >= 15.:
            break
        _, _, _, info = env.step(env.expert_action())
    assert info["angle_deg"] >= 15. and not info["slip_injected"]
    nominal = copy.deepcopy(env.core)
    sensor = copy.deepcopy(env.grip_observer)
    R, p = env.fast_rb.eef_pose()
    requested = np.array([.6, -.3, .2, .7, .2, -.2, .8, -1.], dtype=np.float32)
    expected_R, expected_p, _, expected_width = nominal.command(requested, R, p)
    obs, _, _, info = env.step(requested)
    assert info["slip_injected"] and env.release_left > 0
    assert info["contact_facts"]["command_width_m"] == pytest.approx(.08)
    assert env.requested_width == pytest.approx(expected_width) == 0.
    np.testing.assert_array_equal(obs["history"][-1,20:28], requested)
    np.testing.assert_allclose(env.policy_reference.p_des, expected_p, atol=1e-11)
    np.testing.assert_allclose(env.policy_reference.R_des, expected_R, atol=1e-11)
    assert np.linalg.norm(env.core.p_des - expected_p) > .001
    R, p = env.fast_rb.eef_pose()
    track = np.r_[env.R0.T @ (expected_p - p), env.R0.T @ mat_to_rotvec(expected_R @ R.T)]
    np.testing.assert_allclose(obs["history"][-1,12:18], track, atol=1e-7)
    # Even the grip observation must compare against the actor's request.
    site = R.T @ (env.tracker.transform(env.grip_site0_world) - p)
    assert bool(obs["goal"][11]) == sensor.update(expected_width, env.fast_rb.finger_opening(), site,
                                                   env.tracker.visible)


def test_actor_and_visual_progress_invariant_to_posthandoff_target_segmentation(env, monkeypatch):
    from dataclasses import replace
    _, info = env.reset(seed=403561, kind="door", target_deg=75., staged=False)
    assert info["entry_ok"]
    snap = env.snapshot()
    action = env.expert_action()
    expected, _, _, _ = env.step(action)
    expected = {k:v.copy() for k,v in expected.items()}
    progress = env.tracker.value
    env.restore(snap)
    real_render = env.cameras.render
    erased_targets = []
    def render_without_live_labels():
        views = []
        for view in real_render():
            body = view.body
            valid = body >= 0
            is_robot = valid & env.ss._is_robot[np.maximum(body,0)]
            is_target = valid & env.body_target[np.maximum(body,0)]
            erased_targets.append(int(is_target.sum()))
            # World body 0 is a nonrobot, nontarget ID. Depth, intrinsics and
            # robot silhouettes are the same; every other ID is anonymized.
            assert not env.ss._is_robot[0] and not env.body_target[0]
            views.append(replace(view, body=np.where(is_robot,body,0)))
        return views
    monkeypatch.setattr(env.cameras,"render",render_without_live_labels)
    observed, _, _, _ = env.step(action)
    assert sum(erased_targets)>0
    for k in OBS_SHAPES:
        np.testing.assert_array_equal(observed[k],expected[k])
    assert env.tracker.value == progress


def test_fifty_scripted_physical_labels():
    from lrl_train.audit_gcp_labels import audit
    from atrace.gcp_labels import LABELS
    report = audit()
    assert report["count"] == report["correct"] == 50
    assert {r["expected"] for r in report["trials"]} == set(LABELS)
    for name in LABELS:
        row = report["confusion"][name]
        assert row[name] == sum(row.values()) >= 8


def test_occluded_frozen_payload_uses_measured_fk_prior():
    import copy
    from atrace.gcp_tracking import ReferenceDepthTracker
    rng = np.random.default_rng(324)
    center = np.array([.04, 0., .08])
    target = rng.uniform(-.012,.012,(160,3)) + center
    table = rng.uniform(-.2,.2,(200,3))
    table[:,2] = 0.
    tracker = ReferenceDepthTracker(target,table,np.zeros(3),np.zeros(3),False,rng)
    tracker.record_grip(True,np.eye(3),center)
    previous = copy.deepcopy(tracker)
    displacement = np.array([.006,-.002,.003])
    prediction = tracker.carried_translation(np.eye(3),center+displacement)
    # Both cameras lose the carried object. The header retains its measured
    # motion; absent depth evidence is still exposed as an occlusion.
    tracker.update(table,rng,free=True,grasp_prediction=prediction)
    assert not tracker.visible
    np.testing.assert_allclose(tracker.translation,displacement,atol=1e-10)
    np.testing.assert_array_equal(previous.translation,np.zeros(3))
    tracker.record_grip(False,np.eye(3),center+displacement)
    assert tracker.carried_translation(np.eye(3),center) is None


def test_gap2_free_object_tracking_ignores_exposed_table_hole_and_tracks_flat_plate():
    from atrace.gcp_tracking import ReferenceDepthTracker
    from atrace.geometry import rotvec_to_mat
    rng = np.random.default_rng(71)
    # 16 mm tall flat circular plate resting on z=0.90 table; table beneath plate is occluded at t=0!
    r_plate, h_plate, z_tab = .075, .016, .900
    angles = rng.uniform(0., 2 * np.pi, 220)
    radii = np.sqrt(rng.uniform(0., 1., 220)) * r_plate
    top_pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.full(220, z_tab + h_plate)))
    rim_ang = np.linspace(0., 2 * np.pi, 60, endpoint=False)
    rim_pts = np.column_stack((r_plate * np.cos(rim_ang), r_plate * np.sin(rim_ang), rng.uniform(z_tab + .005, z_tab + h_plate, 60)))
    target0 = np.vstack((top_pts, rim_pts))

    # Background table has a complete hole under r < r_plate at t=0.
    bg_xy = rng.uniform(-.25, .25, (1200, 2))
    bg_xy = bg_xy[np.linalg.norm(bg_xy, axis=1) > r_plate + .005]
    bg0 = np.column_stack((bg_xy, np.full(len(bg_xy), z_tab) + rng.normal(0., .0006, len(bg_xy))))

    tracker = ReferenceDepthTracker(target0, bg0, np.zeros(3), np.zeros(3), False, rng)
    assert tracker.support_z is not None and abs(tracker.support_z - z_tab) < .003

    # Slide plate +9 cm along +x (with 15 deg yaw) in 12 steps while the bare tabletop hole is exposed,
    # AND drop 80% of rim edge pixels to simulate D9 edge_hole corruption!
    errs_mm = []
    for step in range(1, 13):
        frac = step / 12.
        true_t = np.array([.090 * frac, -.025 * frac, 0.])
        true_R = rotvec_to_mat(np.array([0., 0., np.deg2rad(15.) * frac]))
        c0 = tracker.center0
        moved_top = c0 + true_t + (top_pts - c0) @ true_R.T + rng.normal(0., .0008, top_pts.shape)
        keep_rim = rng.random(len(rim_pts)) > .80
        moved_rim = c0 + true_t + (rim_pts[keep_rim] - c0) @ true_R.T + rng.normal(0., .0008, (keep_rim.sum(), 3))
        exposed_hole = np.column_stack((rng.uniform(-r_plate, r_plate, (180, 2)), np.full(180, z_tab) + rng.normal(0., .0008, 180)))
        live_world = np.vstack((moved_top, moved_rim, bg0[:300], exposed_hole))
        tracker.update(live_world, rng, free=True)
        errs_mm.append(1000. * float(np.linalg.norm(tracker.translation - true_t)))

    assert tracker.visible
    assert np.percentile(errs_mm, 95) < 5.0


def test_gap1_all_5_families_bidirectional_modes_and_diverse_payloads(tmp_path):
    from lrl_train.gcp_env import _layout
    from lrl_train.gcp_env import GCPEnv

    # 1. Fast layout-level coverage check across seeds (no EGL reload overhead):
    payload_geoms, place_fixtures = set(), set()
    for s in range(2, 26):
        lay_pick, _ = _layout("pick", s, tmp_path, preset="all_5_families")
        lay_push, _ = _layout("planar_push", s, tmp_path, preset="all_5_families")
        _, fix_place = _layout("place", s, tmp_path, preset="all_5_families")
        for frag in lay_pick.mjcf + lay_push.mjcf:
            for gname in ("ucs_payload_plate", "ucs_payload_book", "ucs_payload_cyl", "ucs_payload_base", "ucs_payload_box"):
                if gname in frag:
                    payload_geoms.add(gname)
        place_fixtures.add(fix_place.fixture_body)
    assert {"ucs_payload_plate", "ucs_payload_book", "ucs_payload_cyl", "ucs_payload_base", "ucs_payload_box"}.issubset(payload_geoms)
    assert {"ucs_cavity", "ucs_ohdrawer", "ucs_caddy"}.issubset(place_fixtures)

    # 2. Full GCPEnv bidirectional mode check on drawer & door (reusing loaded layouts across resets!):
    inst = GCPEnv(seed=10, out_dir=tmp_path / "env", preset="all_5_families", staged_reset_prob=0.)
    try:
        drawer_signs, door_signs = set(), set()
        for s in range(10, 18):
            inst.reset(seed=s, kind="drawer", staged=False)
            drawer_signs.add(int(np.sign(inst.contract.target_trans_m)))
            inst.reset(seed=s, kind="door", staged=False)
            door_signs.add(int(np.sign(inst.contract.target_rot_rad * (inst.part.open - inst.part.closed))))
        assert {-1, 1}.issubset(drawer_signs)
        assert {-1, 1}.issubset(door_signs)
    finally:
        inst.close()



def test_stationary_half_occluded_plate_has_no_tangential_creep():
    from atrace.gcp_tracking import ReferenceDepthTracker
    rng = np.random.default_rng(9008)
    angles = np.linspace(0., 2*np.pi, 120, endpoint=False)
    radii = np.linspace(.003, .075, 17)
    a, r = np.meshgrid(angles, radii)
    top = np.column_stack(((r*np.cos(a)).ravel(), (r*np.sin(a)).ravel(), np.full(r.size, .916)))
    rim = np.column_stack((.075*np.cos(angles), .075*np.sin(angles),
                           .904 + .010*rng.random(len(angles))))
    target = np.vstack((top, rim))
    table = rng.uniform(-.22, .22, (1800, 3))
    table[:, 2] = .900
    bg = table[np.linalg.norm(table[:, :2], axis=1) > .078]
    tracker = ReferenceDepthTracker(target, bg, np.zeros(3), np.zeros(3), False, rng)
    # Calibrated downward camera. A closer foreground occluder covers the
    # left half of the disc in both images, the right half is still measured.
    K = np.array([[270.,0.,64.],[0.,270.,64.],[0.,0.,1.]])
    pose = np.diag([1.,-1.,-1.,1.])
    pose[:3,3] = [0.,0.,1.36]
    image = np.full((128,128), 1.36-.916)
    image[:, :64] = .30
    exposed = target[target[:,0] >= 0.]
    second_pose = pose.copy()
    second_pose[0, 3] = .016
    second_image = np.full_like(image, 1.36-.916)
    second_image[:, :54] = .30
    views = [(image, K, pose), (second_image, K, second_pose)]
    for _ in range(12):
        tracker.update(np.vstack((bg, exposed)), rng, free=True, views=views)
        assert tracker.visible
        assert np.linalg.norm(tracker.translation) < .002


def test_depth_buffer_accepts_a_point_visible_to_either_camera():
    from atrace.gcp_tracking import _unoccluded
    K = np.array([[100.,0.,8.],[0.,100.,8.],[0.,0.,1.]])
    pose = np.eye(4)
    points = np.array([[0.,0.,1.], [.02,0.,1.], [.2,0.,1.], [0.,0.,-1.]])
    image = np.full((16,16), np.nan)
    image[8,8] = .991  # occluded by more than 8 mm
    image[8,10] = .993  # inside the z-buffer tolerance
    other = np.full_like(image, np.nan)
    other[8,8] = 1.
    np.testing.assert_array_equal(_unoccluded(points, [(image,K,pose)]), [False,True,False,False])
    np.testing.assert_array_equal(_unoccluded(points, [(image,K,pose),(other,K,pose)]),
                                  [True,True,False,False])


def test_stationary_drawer_sidewall_cannot_observe_prismatic_motion():
    from atrace.gcp_tracking import ReferenceDepthTracker
    rng = np.random.default_rng(1124)
    x, z = np.meshgrid(np.linspace(-.08,.08,31), np.linspace(.92,1.02,25))
    side = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    tracker = ReferenceDepthTracker(side, np.empty((0,3)), np.zeros(3), np.array([1.,0.,0.]), False, rng)
    for _ in range(12):
        tracker.update(side[side[:,0] > .025], rng)
        assert tracker.value == 0.
        assert not tracker.visible


def test_confirmed_place_grasp_survives_darkness_then_releases_on_command(env, monkeypatch):
    from dataclasses import replace
    _, info = env.reset(seed=6, kind="place", staged=False)
    assert info["entry_ok"] and env.sensed_hold
    assert env.tracker.center_in_grip is not None
    anchor = env.tracker.center_in_grip.copy()
    render = env.cameras.render
    monkeypatch.setattr(env.cameras, "render",
                        lambda: [replace(v, depth=np.full_like(v.depth, np.nan)) for v in render()])
    priors = []
    update = env.tracker.update
    def capture(*args, **kwargs):
        priors.append(kwargs["grasp_prediction"])
        return update(*args, **kwargs)
    monkeypatch.setattr(env.tracker, "update", capture)
    action = np.zeros(8)
    action[7] = -1.
    for _ in range(15):
        env.step(action)
        assert not env.tracker.visible and env.sensed_hold
        np.testing.assert_array_equal(env.tracker.center_in_grip, anchor)
        assert priors[-1] is not None
    action[7] = 1.
    env.step(action)
    assert priors[-1] is None
    assert not env.sensed_hold and env.tracker.center_in_grip is None


def test_invisible_handoff_cannot_acquire_or_reacquire_a_grasp():
    sensor = GripStateObserver((-.012, .004), .008, carried_at_handoff=True)
    for _ in range(15):
        assert not sensor.update(0., .014, np.zeros(3), False)
    assert not sensor.update(0., .014, np.zeros(3), True)
    assert sensor.update(0., .014, np.zeros(3), True)
    assert not sensor.update(0., .040, np.zeros(3), False)
    assert not sensor.update(0., .014, np.zeros(3), False)


def test_released_turn_seed418205_holds_measured_angle(env):
    _, info = env.reset(seed=418205, kind="turn", staged=False)
    assert info["entry_ok"]
    errors = []
    for _ in range(130):
        _, _, _, info = env.step(env.expert_action())
        true = env.progress()[1]
        if not env.facts["left_target"] and not env.facts["right_target"] and abs(true) > np.deg2rad(25.):
            errors.append(abs(np.rad2deg(env.tracker.progress_value - true)))
    assert len(errors) >= 45
    assert max(errors) <= 5., max(errors)


def test_released_book_in_caddy_tracks_body_origin(env):
    _, info = env.reset(seed=12, kind="place", staged=False)
    assert info["entry_ok"] and env.fixture.kind == "slot_place"
    errors = []
    for _ in range(130):
        _, _, _, info = env.step(env.expert_action())
        if not env.sensed_hold:
            pose = env.tracker.transform(env.q_initial_world)
            errors.append(np.linalg.norm(pose - env.target_pos()) * 1000.)
    assert info["success"] and len(errors) >= 40
    assert np.percentile(errors, 95) <= 12. and max(errors) <= 30.


def test_loaded_payload_stays_level_in_low_clearance_transport(env):
    _, info = env.reset(seed=839016, kind="place", staged=False)
    assert info["entry_ok"]
    errors = []
    for _ in range(130):
        _, _, _, info = env.step(env.expert_action())
        errors.append(1000.*np.linalg.norm(env.tracker.transform(env.q_initial_world) - env.target_pos()))
    assert info["success"] and env.teacher_place_release is not None
    assert max(errors) <= 30.


def test_reachable_outer_contact_survives_deeper_pad_candidate(env):
    _, info = env.reset(seed=984032, kind="turn", staged=False)
    assert info["entry_ok"]
    for _ in range(130):
        _, _, _, info = env.step(env.expert_action())
    assert info["success"]

def test_half_open_goal_sampling_is_bidirectional_for_each_start_bin():
    from types import SimpleNamespace
    from lrl_train.gcp_env import _door_command, _drawer_command, _turn_command

    rng = np.random.default_rng(37)
    for sign in (-1., 1.):
        for angular in (False, True):
            span = np.deg2rad(100.) if angular else .15
            part = SimpleNamespace(closed=0., open=sign*span, range=(0., sign*span),
                                   handle_point_b=np.ones(3), protrusion=.03, handle_half=(.02, .01))
            sample = _door_command if angular else _drawer_command
            rows = np.asarray([sample(rng, part, None, None) for _ in range(4000)])
            start = rows[:, 0] / part.open
            end = (rows[:, 0] + rows[:, 1] if angular else rows[:, 1]) / part.open
            left, right = (.20, .45) if angular else (.25, .70)
            bins = np.linspace(left, right, 5)
            assert np.mean((start >= left) & (start <= right)) >= .4
            assert np.all((end >= -1e-12) & (end <= 1.+1e-12))
            for low, high in zip(bins, bins[1:]):
                in_bin = (start >= low) & (start < high)
                assert .41 < np.mean(end[in_bin] > start[in_bin]) < .59
            half = (start >= left) & (start <= right)
            if angular:
                opened = end[half & (end > start)]
                assert np.all((opened >= .60) & (opened <= .80))
            else:
                assert np.any(half & (end < start) & (end >= .15) & (end <= .40))
                assert np.any((end > start) & (end >= .15) & (end <= .40))
        rows = np.asarray([_turn_command(rng, part, None, None) for _ in range(1200)])
        mag = np.abs(np.rad2deg(rows[:, 1]))
        assert 30. <= mag.min() < 31. and 79. < mag.max() <= 80.
        assert len(np.unique(np.round(mag, 1))) > 300
        assert set(np.sign(rows[:, 1])) == {-1., 1.}


@pytest.mark.parametrize("kind,seed,opening", [("drawer", 11, .033), ("door", 11, 25.), ("turn", 931004, 24.)])
def test_signed_override_only_changes_goal_at_same_handoff(env, kind, seed, opening):
    angular = kind != "drawer"
    field = "target_deg" if angular else "target_trans_m"
    env.reset(seed=seed, kind=kind, initial_fraction=.50, staged=False, **{field: opening})
    assert env.entry_ok
    snap = env.snapshot()
    for command in (-opening, opening*.6):
        reference = env.restore(snap)[0]
        # Capture an independently restored reference to compare complete sensor inputs.
        restored, _ = env.restore(snap, **{field: command})
        for key in ("scene_pts", "payload_pts", "history"):
            np.testing.assert_array_equal(restored[key], reference[key])
        assert env.rng.bit_generator.state == snap["rng"]
        np.testing.assert_array_equal(env.ss.snapshot()["state"], snap["sim"]["state"])
        assert env.good_steps == 0
        delta = np.deg2rad(command-opening) if angular else command-opening
        rem = 6 if angular else 5
        assert restored["goal"][rem] - reference["goal"][rem] == pytest.approx(delta, abs=1e-7)
        np.testing.assert_allclose(restored["goal"][:3], np.sign(command)*reference["goal"][:3], atol=1e-6)
    with pytest.raises(ValueError, match="range"):
        env.restore(snap, **{field: opening*10.})


@pytest.mark.parametrize("kind,original,actual", [("drawer", .09, .035), ("door", .95, .55), ("turn", 1.05, .60)])
def test_joint_counterfactual_pairs_share_the_original_tracking_error(monkeypatch, kind, original, actual):
    from types import SimpleNamespace
    from lrl_train.gcp_env import GCPEnv

    env = object.__new__(GCPEnv)
    env.hinge = kind != "drawer"
    field = "target_rot_rad" if env.hinge else "target_trans_m"
    slot = 6 if env.hinge else 5
    env.part = SimpleNamespace(range=(0., 1.4 if env.hinge else .14), handle_point_b=np.ones(3))
    env.contract = ContactContract("turn" if kind == "turn" else "push_pull", "target", "frame",
                                   [1., 0., 0.], [0., 0., 1.], **{field: original})
    env.entry_ok, env.evaluation, env.steps, env.horizon, env.initial_joint = True, False, 34, 130, 0.
    env.preset = "all_5_families"
    env.obs = {"goal": np.arange(16, dtype=np.float32)}
    env.obs["goal"][slot] = original - actual + .005
    env.progress = lambda: (0., actual) if env.hinge else (actual, 0.)
    monkeypatch.setattr(GCPEnv, "expert_action", lambda view: getattr(view.contract, field))
    header, alternative = env.counterfactual_expert()
    assert alternative == pytest.approx(actual)
    assert header[slot] == pytest.approx(.005, abs=1e-7)
    np.testing.assert_array_equal(np.delete(header, slot), np.delete(env.obs["goal"], slot))
    assert getattr(env.contract, field) == original


@pytest.mark.parametrize("degrees", [-75., 75.])
def test_rolling_arc_initial_displacement_follows_signed_contract(monkeypatch, degrees):
    from types import SimpleNamespace
    from atrace import gcp_entry
    from atrace.gcp import ContactContract, EntryCandidate

    axis = np.array([0., 0., 1.])
    tangent = np.array([0., np.sign(degrees), 0.])
    contact = np.zeros(3)
    approach, closing = np.eye(3)[:2]
    contract = ContactContract("push_pull", "door", "frame", tangent, axis,
                               target_rot_rad=np.deg2rad(degrees))
    candidate = EntryCandidate(np.eye(3), -approach*.05, np.eye(3), contact, np.zeros(7),
                               np.zeros((1, 7)), approach, "test")
    sampler = SimpleNamespace(model=SimpleNamespace(limits=np.tile([-3., 3.], (7, 1))))
    paths = []
    def inspect_path(sampler, candidate, contact, axis, angles, path, finger, fractions, limits):
        paths.append(np.asarray(path))
        return .30
    monkeypatch.setattr(gcp_entry, "_score_existing", inspect_path)
    gcp_entry.rank_rolling_arc(sampler, contract, [candidate], np.zeros(7), contact, approach,
                               closing, np.eye(3), .02, np.random.default_rng(42), 1)
    assert len(paths) == 1
    first_step = paths[0][1] - paths[0][0]
    assert first_step @ tangent > .99 * np.linalg.norm(first_step) > 0.


def test_all5_staged_restore_rebinds_nonzero_drawer_joint(tmp_path):
    from lrl_train.gcp_env import GCPEnv
    env = GCPEnv(1101, tmp_path, preset="all_5_families", staged_reset_prob=0.)
    try:
        env.reset(seed=100, kind="drawer", staged=False)
        assert env.entry_ok and env.part.body.endswith("_p2")
        for _ in range(35):
            env.step(env.expert_action())
            if env.facts["firm"] or env.facts["push_face_loaded"]:
                break
        assert env.facts["firm"] or env.facts["push_face_loaded"]
        shot = env.snapshot()
        progress = env.progress()
        names = ("body_target", "target_g", "grasp_g", "support_g", "pairs")
        masks = {name: getattr(env, name).copy() for name in names}
        def check_restored():
            env.restore(shot)
            assert env.part in env.ss.parts and env.part.body == env.contract.target_entity
            assert env.part.joint == shot["state"]["part_joint_name"]
            assert env.ss._body[env.bid] == env.part.body
            assert env.progress() == progress
            assert not env.hinge and not env.free
            for name, mask in masks.items():
                np.testing.assert_array_equal(getattr(env, name), mask)
        # Choose the intervening layer explicitly; episode goal draws may
        # otherwise land on the original layer and miss the binding regression.
        env.rng = np.random.default_rng(1)
        env.reset(staged=False)  # another training start in the same layout
        assert env.key == ("drawer", 100) and env.part.body != shot["state"]["contract"].target_entity
        check_restored()
        env.reset(seed=1, kind="pick", staged=False)
        check_restored()
        for name in env._binding_fields:
            shot["state"].pop(name)
        del shot["state"]["tracker"].legacy
        del shot["state"]["grip_observer"].legacy
        env.reset(seed=1, kind="pick", staged=False)
        env.restore(shot)
        assert env.progress() == progress and env.part.body == env.contract.target_entity
        assert not env.tracker.legacy and not env.grip_observer.legacy
        for name, mask in masks.items():
            np.testing.assert_array_equal(getattr(env, name), mask)
        env.step(env.expert_action())
    finally:
        env.close()


def test_all5_snapshot_reverses_push_close_door_goal_tangent(tmp_path):
    from lrl_train.gcp_env import GCPEnv
    env = GCPEnv(11, tmp_path, preset="all_5_families", staged_reset_prob=0.)
    try:
        env.reset(seed=11, kind="door", staged=False)
        assert env.contract.target_rot_rad < 0. and env.surface_push
        shot = env.snapshot()
        obs, _ = env.restore(shot, target_deg=20.)
        assert not env.surface_push and env.contract.target_rot_rad == pytest.approx(np.deg2rad(20.))
        assert env.initial_joint + env.contract.target_rot_rad < env.part.open
        expected = np.cross(env.axis, env.p_contact() - env.anchor)
        expected /= np.linalg.norm(expected)
        R, _ = env.fast_rb.eef_pose()
        assert (R @ obs["goal"][:3]) @ expected > .999
        np.testing.assert_allclose(env.Rref @ env.contract.goal_dir_ref, env.direction)
        positive = env.snapshot()
        env.restore(positive, target_deg=-20.)
        expected = -np.cross(env.axis, env.p_contact() - env.anchor)
        expected /= np.linalg.norm(expected)
        assert env.surface_push and (R @ env.obs["goal"][:3]) @ expected > .999
    finally:
        env.close()


def test_all5_driveable_door_distribution(tmp_path):
    from xml.etree import ElementTree
    from lrl_train.gcp_env import ALL5_MECH_CFG, _door_push_close, _layout

    pulls, handles = [], {True: set(), False: set()}
    damping, friction = [], []
    for seed in range(256):
        closing = _door_push_close(seed)
        layout, _ = _layout("door", seed, tmp_path, preset="all_5_families")
        part = layout.parts[0]
        handles[closing].add(part.handle_kind)
        pulls.append(not closing)
        if not closing:
            assert part.handle_point_b is not None
        joint = ElementTree.fromstring(layout.mjcf[0]).find(".//joint")
        damping.append(float(joint.attrib["damping"]))
        friction.append(float(joint.attrib["frictionloss"]))
    assert .59 < np.mean(pulls) < .71
    assert "none" not in handles[False] and "none" in handles[True]
    assert len(handles[False]) >= 2
    assert ALL5_MECH_CFG["heavy_damping"] == (.20, 3.8)
    assert ALL5_MECH_CFG["heavy_friction"] == (.04, .85)
    assert 3.5 < max(damping) <= 3.8 and max(friction) > .75


def test_slide_encoder_prior_cannot_publish_a_visual_measurement():
    from atrace.gcp_tracking import ReferenceDepthTracker
    rng = np.random.default_rng(210)
    x, z = np.meshgrid(np.linspace(-.08, .08, 21), np.linspace(-.03, .03, 11))
    points = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    tracker = ReferenceDepthTracker(points, np.empty((0, 3)), np.zeros(3), np.array([1., 0., 0.]), False, rng)
    tracker.update(points, rng, angle_prior=.040)
    assert not tracker.visible and tracker.progress_value == 0.
    tracker.record_grip(True, np.eye(3), np.zeros(3))
    prior = tracker.carried_joint(np.eye(3), np.array([.040, 0., 0.]))
    assert prior == pytest.approx(.040)
    tracker.update(points, rng, angle_prior=prior)
    assert not tracker.visible and tracker.value == 0.
    assert tracker.progress_value == .040
    np.testing.assert_allclose(tracker.transform(np.zeros(3)), [.040, 0., 0.])
    tracker.record_grip(False, np.eye(3), np.zeros(3))
    tracker.update(points, rng, angle_prior=.110)
    assert tracker.progress_value == .040


def test_stationary_visible_joint_overrules_slipping_wrist_prediction():
    from atrace.gcp_tracking import ReferenceDepthTracker
    rng = np.random.default_rng(74)
    y, z = np.meshgrid(np.linspace(-.045, .045, 23), np.linspace(.01, .07, 17))
    face = np.column_stack((np.zeros(y.size), y.ravel(), z.ravel()))
    tracker = ReferenceDepthTracker(face, np.empty((0, 3)), np.zeros(3),
                                    np.array([1., 0., 0.]), False, rng)
    assert tracker.visible
    tracker.record_grip(True, np.eye(3), np.zeros(3))
    prior = tracker.carried_joint(np.eye(3), np.array([.040, 0., 0.]))
    assert prior == pytest.approx(.040)
    tracker.update(face, rng, angle_prior=prior)
    assert tracker.visible
    assert abs(tracker.progress_value) < .002


def test_subpixel_depth_edges_do_not_claim_rotation_of_a_symmetric_surface():
    from atrace.gcp_tracking import ReferenceDepthTracker
    from atrace.geometry import rotvec_to_mat

    rng = np.random.default_rng(18)
    focal = 250.
    v, u = np.mgrid[:96, :96]
    rays = np.dstack(((u+.5-48)/focal, (v+.5-48)/focal, np.ones_like(u)))
    face_z = .4
    mask = np.linalg.norm(face_z*rays[:, :, :2], axis=2) < .030
    depth = np.where(mask, face_z, .45)
    points = depth[:, :, None]*rays
    k = np.array([[focal, 0., 48.], [0., focal, 48.], [0., 0., 1.]])
    views = [(depth, k, np.eye(4))]
    axis, origin = np.array([0., 0., 1.]), np.array([0., 0., face_z])
    tracker = ReferenceDepthTracker(points[mask], points[~mask], origin, axis, True, rng,
                                    camera_targets=[points[mask]], views=views)
    tracker.record_grip(True, np.eye(3), origin)
    prediction = tracker.carried_joint(rotvec_to_mat(axis*.25), origin)
    tracker.update(points.reshape(-1, 3), rng, angle_prior=prediction, views=views)
    assert not tracker.visible and tracker.progress_value == pytest.approx(.25)
    tracker.record_grip(False, np.eye(3), origin)
    tracker.update(points.reshape(-1, 3), rng, views=views)
    assert not tracker.visible and tracker.progress_value == pytest.approx(.25)


@pytest.mark.parametrize("kind,seed", [("planar_push", 2), ("place", 25)])
def test_all5_disk_rim_and_sill_corridor_complete_in_130_ticks(env, kind, seed):
    _, info = env.reset(seed=seed, kind=kind, staged=False)
    assert info["entry_ok"]
    if kind == "planar_push":
        radius = env.m.geom_size[env.m.body_geomadr[env.bid], 0]
        assert np.linalg.norm(env.hint_local[:2]) == pytest.approx(radius, abs=1.e-8)
        assert np.isfinite(env.contact_R).all()
    else:
        assert env.fixture.kind == "overhang_place"
        assert env.facts["firm"]
        assert env.entry_payload_pen < .001
    for _ in range(130):
        _, _, _, info = env.step(env.expert_action())
    assert info["success"]


def test_slot_place_setdown_opens_once_and_retreats_without_moving_cup(env):
    env.reset(seed=418205, kind="place", staged=False)
    assert env.entry_ok and env.fixture.kind == "slot_place"
    positions = []
    for _ in range(env.horizon):
        action = env.expert_action()
        if env.teacher_place_release is not None:
            _, held_width, _ = env.teacher_place_release
            assert .04 * (action[7] + 1.) > held_width
        _, _, _, info = env.step(action)
        if env.teacher_place_release is not None:
            positions.append(env.target_pos().copy())
    assert len(positions) > 30 and info["success"]
    positions = np.stack(positions)
    assert np.max(np.linalg.norm(positions - positions[0], axis=1)) < .006


def test_all5_diagonal_box_push_remains_on_table_for_full_stroke(env):
    env.reset(seed=602422, kind="planar_push", staged=False)
    assert env.entry_ok and "ucs_payload_box" in env.geom_names
    origin = env.target_pos().copy()
    for _ in range(env.horizon):
        _, _, _, info = env.step(env.expert_action())
        travel = env.target_pos() - origin
        sideways = travel - (travel @ env.direction) * env.direction
        # This diagonal contact previously launched the body by over 0.4 m.
        assert np.linalg.norm(sideways) < .025
    assert info["success"]


def test_shallow_push_transmits_side_load_below_mass_center(env):
    import mujoco
    env.reset(seed=931015, kind="planar_push", staged=False)
    assert env.entry_ok and env.shallow_push_entry
    candidate = next(iter(env.pose_solutions.values()))[0][0]
    assert candidate.visibility_score >= 1.
    initial_pitch = np.rad2deg(np.arcsin(-env.contact_R[2, 2]))
    assert 15. <= initial_pitch <= 35.01
    original = env.snapshot()
    rich_contact = False
    for _ in range(80):
        env.step(env.expert_action())
        R, _ = env.fast_rb.eef_pose()
        pitch = np.rad2deg(np.arcsin(-R[2, 2]))
        if not 14. <= pitch <= 36.:
            continue
        for k, contact in enumerate(env.d.contact[:env.d.ncon]):
            a, b = contact.geom1, contact.geom2
            if env.fingers[a] and env.target_g[b]:
                normal = contact.frame[:3]
            elif env.fingers[b] and env.target_g[a]:
                normal = -contact.frame[:3]
            else:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(env.m, env.d, k, wrench)
            if (wrench[0] > .25 and normal @ env.direction > .72
                    and normal[2] < .25 and contact.pos[2] < env.d.xipos[env.bid, 2]+.001):
                fingers = np.flatnonzero(env.fingers & ((env.m.geom_contype | env.m.geom_conaffinity) != 0))
                floors = np.flatnonzero(env.support_g)
                assert all(mujoco.mj_geomDistance(env.m, env.d, g, floor, .02, None) > 0.
                           for g in fingers for floor in floors)
                rich_contact = True
                break
        if rich_contact:
            break
    assert rich_contact
    env.reset(seed=2, kind="planar_push", staged=False)
    assert env.entry_ok and not env.shallow_push_entry
    env.restore(original)
    assert env.shallow_push_entry and env.push_tcp_height == original["state"]["push_tcp_height"]
    np.testing.assert_array_equal(env.push_contact_offset, original["state"]["push_contact_offset"])
    # Staged records before shallow contacts stored no tool-to-patch offset.
    original["state"].pop("shallow_push_entry")
    original["state"].pop("push_contact_offset")
    env.restore(original)
    assert not env.shallow_push_entry
    np.testing.assert_array_equal(env.push_contact_offset, np.zeros(3))


def test_air_sweep_does_not_move_the_visual_goal(env):
    from atrace.geometry import frame_from_axes, mat_to_rotvec

    obs, info = env.reset(seed=2, kind="planar_push", staged=False)
    assert info["entry_ok"]
    origin = env.target_pos().copy()
    estimate = env.tracker.transform(env.q_initial_world)
    initial_remaining = obs["goal"][5]
    direction = env.direction.copy()
    rear = env.hint_world.copy()
    attitude = frame_from_axes([0., 0., -1.], direction)
    pairs = (env.robot_g[:, None] & env.target_g[None, :])
    pairs |= pairs.T.copy()
    env.fast_rb.set_penetration_pairs(pairs)
    travel = []
    for tick in range(130):
        R, _ = env.fast_rb.eef_pose()
        # Rotate clear of the target before moving across it. Verification covers
        # the approach as well as the 90+ mm horizontal motion.
        clearance = .060 if tick < 25 else .030
        goal = rear + np.array([0., 0., clearance]) + (-.055 if tick < 55 else .100) * direction
        translation = R.T @ (goal - env.core.p_des) / .008
        rotation = R.T @ mat_to_rotvec(attitude @ env.core.R_des.T) / np.deg2rad(4.5)
        obs, _, _, _ = env.step(np.clip(np.r_[translation, rotation, 1., -.78], -1., 1.))
        travel.append(env.fast_rb.eef_pose()[1] @ direction)
        contacts = env.d.contact.geom[:env.d.ncon]
        assert not pairs[contacts[:, 0], contacts[:, 1]].any()
        assert env.fast_rb.last_substep_max_pen_m == 0.
        assert np.linalg.norm(env.target_pos() - origin) < .001
        assert np.linalg.norm(env.tracker.transform(env.q_initial_world) - estimate) < .003
        assert abs(obs["goal"][5] - initial_remaining) < .003
    assert max(travel[55:]) - travel[54] >= .090


@pytest.mark.parametrize("kind,seed", [("pick", 931015), ("place", 9)])
def test_contact_physics_match_independent_stagesim(env, tmp_path, kind, seed):
    from lrl_train.gcp_env import _layout
    from lrl_train.sim import StageSim

    env.reset(kind=kind, seed=seed, staged=False)
    layout, _ = _layout(kind, seed, tmp_path, preset="all_5_families")
    reference = StageSim(layout)
    try:
        model = reference.robot.m
        for field in ("geom_condim", "geom_friction", "geom_solref", "geom_solimp"):
            np.testing.assert_array_equal(getattr(env.m, field), getattr(model, field))
        for field in ("enableflags", "disableflags", "noslip_iterations", "solver", "cone"):
            assert getattr(env.m.opt, field) == getattr(model.opt, field)
    finally:
        reference.close()


@pytest.mark.parametrize("kind,seed", [("planar_push", 931007), ("planar_push", 931015), ("turn", 931004)])
def test_blind_contact_tracking_and_teacher(env, kind, seed):
    _, info = env.reset(seed=seed, kind=kind, staged=False)
    assert info["entry_ok"]
    errors = []
    for _ in range(130):
        _, _, _, info = env.step(env.expert_action())
        if env.free:
            errors.append(1000 * np.linalg.norm(env.tracker.transform(env.q_initial_world) - env.target_pos()))
        else:
            errors.append(abs(np.rad2deg(env.tracker.progress_value - env.progress()[1])))
    assert info["success"]
    assert np.percentile(errors, 95) <= (12. if env.free else 5.)
    if env.free:
        assert max(errors) <= 30.


def test_signed_opening_displacement_injects_slip_and_recovers(env):
    _, info = env.reset(seed=13, kind="door", target_deg=-55., slip_at_deg=15., staged=False)
    assert info["entry_ok"] and env.part.open < env.part.closed and not env.surface_push
    for _ in range(130):
        angle_before = env.progress()[1]
        injected = env.slip_injected
        _, _, _, info = env.step(env.expert_action())
        if env.slip_injected and not injected:
            assert -np.rad2deg(angle_before) >= 15. and env.before_slip_held
    assert env.slip_injected and env.slip_lost and env.regrasped and info["success"]


def test_demonstrator_rollouts_preserve_real_plant_and_actor(env):
    import copy
    import mujoco
    from lrl_train.gcp_teacher import GripRollout

    _, info = env.reset(seed=931004, kind="turn", staged=False)
    assert info["entry_ok"]
    nominal = np.array([0., 0., 0., 0., 0., .45, 1., -1.])
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    def registers():
        state = np.empty(mujoco.mj_stateSize(env.m, state_spec))
        mujoco.mj_getState(env.m, env.d, state, state_spec)
        robot = env.fast_rb
        return [a.copy() for a in (state, robot._kp, robot._kd, robot._grip_cur, robot._q_init,
                                   robot.base.ctrl.kp, robot.base.ctrl.kd, env.core.p_des,
                                   env.core.R_des, *env.obs.values())]

    before = registers()
    random_state = copy.deepcopy(env.rng.bit_generator.state)
    real_steps = env.fast_rb.base.steps
    planner = GripRollout(env)
    decision = planner.select(env, nominal)
    assert planner.select(env, nominal) == decision
    assert env.fast_rb.base.steps == real_steps and env.rng.bit_generator.state == random_state
    for original, actual in zip(before, registers()):
        np.testing.assert_array_equal(original, actual)


def test_opening_pads_do_not_move_a_stationary_shaft_estimate(env):
    # Reproduce the original -0.65 rad handoff. The released-pad regression
    # needs the shaft at rest by frame 60 regardless of curriculum draws.
    _, info = env.reset(seed=713218, kind="turn", staged=False,
                        initial_fraction=.9485332967846805, target_deg=np.rad2deg(-.65))
    assert info["entry_ok"]
    released = []
    for tick in range(130):
        _, _, _, info = env.step(env.expert_action())
        if tick >= 60:
            assert not env.sensed_hold
            released.append(np.rad2deg([env.progress()[1], env.tracker.progress_value]))
    truth, estimate = np.asarray(released).T
    assert info["success"] and np.ptp(truth) < .2
    assert np.max(np.abs(estimate-truth)) < 5.


def test_consecutive_depth_rejects_an_inconsistent_free_pose(env, monkeypatch):
    from atrace import gcp_release

    env.reset(seed=770410, kind="planar_push", staged=False)
    tracker = env.tracker
    surfaces = tracker.reference_surfaces
    clouds = [surface.points + tracker.center0 for surface in surfaces]
    views = [[surface.view] for surface in surfaces]
    # Reacquisition away from the original handoff. Simulate the wrong match
    # to a parallel surface in the old template, with two identical depth frames.
    tracker.anchor_center = tracker.center0 - .020*env.direction
    proposed = np.zeros(3)
    def ambiguous_fit(tracked, clouds, views):
        tracked.visible = True
        tracked.translation = proposed.copy()
    monkeypatch.setattr(gcp_release, "_fit_reference", ambiguous_fit)
    gcp_release.register_surfaces(tracker, clouds, views)
    assert any(patch is not None for patch in tracker.previous_depth_patches)
    proposed[:] = .025*env.direction
    gcp_release.register_surfaces(tracker, clouds, views)
    assert np.linalg.norm(tracker.translation) < .006


@pytest.mark.parametrize("shape", ["box", "mug", "bowl", "plate", "thin_book"])
def test_flat_support_witnesses_preserve_rigid_body_and_collision_boundary(shape):
    import itertools
    import xml.etree.ElementTree as ET
    import mujoco
    from lrl_train.ucs_scenes import make_payload_xml

    fragment, _, _ = make_payload_xml(np.random.default_rng(95), shape)
    body = ET.fromstring(fragment)
    def compile_body():
        return mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{ET.tostring(body, encoding='unicode')}</worldbody></mujoco>")

    model = compile_body()
    witnesses = [g for g in body.findall("geom") if g.get("mass") == "0"]
    assert len(witnesses) == 4
    for g in witnesses:
        body.remove(g)
    original = compile_body()

    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_allclose(model.body_ipos, original.body_ipos, atol=1e-14)
    def inertia(m):
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[1])
        axes = rot.reshape(3, 3)
        return axes @ np.diag(m.body_inertia[1]) @ axes.T
    np.testing.assert_allclose(inertia(model), inertia(original), atol=1e-14, rtol=1e-13)
    np.testing.assert_array_equal(model.geom_friction[:original.ngeom], original.geom_friction)
    np.testing.assert_array_equal(model.geom_rgba[:original.ngeom], original.geom_rgba)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    # All eight corners of the enclosing cube must be in one original convex
    # solid. Convexity then certifies containment for every point of the sphere
    # in every pose, also for a thin base under an empty interior.
    for g in range(original.ngeom, model.ngeom):
        assert model.geom_rgba[g, 3] == 0.
        radius = model.geom_size[g, 0]
        corners = data.geom_xpos[g] + radius * np.array(list(itertools.product((-1., 1.), repeat=3)))
        inside = []
        for parent in range(original.ngeom):
            ray = np.array([.223, -.413, .883])
            ray /= np.linalg.norm(ray)
            low = [mujoco.mju_rayGeom(data.geom_xpos[parent], data.geom_xmat[parent],
                   model.geom_size[parent], c-.5*ray, ray, model.geom_type[parent]) for c in corners]
            high = [mujoco.mju_rayGeom(data.geom_xpos[parent], data.geom_xmat[parent],
                    model.geom_size[parent], c+.5*ray, -ray, model.geom_type[parent]) for c in corners]
            if np.all(np.asarray(low) >= 0.) and np.all(np.asarray(high) >= 0.) and max(low+high) <= .5+1e-9:
                inside.append(parent)
        assert len(inside) and any(np.array_equal(model.geom_friction[i], model.geom_friction[g]) for i in inside)


def test_older_payload_snapshot_rebinds_collision_indices(env):
    _, info = env.reset(seed=1, kind="place", staged=False)
    assert info["entry_ok"] and env.payload_owned
    snapshot = env.snapshot()
    old = np.array([not label.startswith("ucs_support_") for label in env.geom_names])
    for field in ("target_g", "grasp_g", "support_g"):
        snapshot["state"][field] = snapshot["state"][field][old]
    snapshot["penetration_pairs"] = snapshot["penetration_pairs"][np.ix_(old, old)]
    expected = env.pairs.copy()
    env.restore(snapshot)
    np.testing.assert_array_equal(env.pairs, expected)
    assert env.payload_owned and env.target_g.shape == (env.m.ngeom,)
    _, _, _, after = env.step(env.expert_action())
    assert after["entry_ok"] and np.isfinite(after["max_pen_m"])


def test_rear_hint_finds_the_exposed_corner_of_a_compound_footprint():
    import mujoco
    from lrl_train.gcp_env import _rear_surface

    xml = """<mujoco><worldbody><body euler="0 0 31">
        <geom type="box" size=".04 .033 .02"/>
        <geom type="box" pos="-.05 .017 0" size=".018 .009 .014"/>
        </body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    direction = np.array([1., 0., 0.])
    hint = _rear_surface(m, d, np.arange(m.ngeom), np.zeros(3), direction, .002)
    corners = np.array([[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]])
    surfaces = np.vstack([d.geom_xpos[g] + (corners*m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T
                          for g in range(m.ngeom)])
    expected = surfaces[np.argmin(surfaces @ direction)]
    np.testing.assert_allclose(hint[:2], expected[:2], atol=1e-6)
    assert hint[2] == pytest.approx(.002)


def test_foreground_support_plane_cannot_cut_through_handoff_surface():
    from atrace.gcp_tracking import ReferenceDepthTracker

    rng = np.random.default_rng(12)
    target = rng.uniform([-.025, -.025, -.028], [.025, .025, .028], (400, 3))
    shelf = rng.uniform(-.26, .26, (2000, 3))
    shelf[:, 2] = 0.
    tracker = ReferenceDepthTracker(target, shelf, np.zeros(3), np.zeros(3), False, rng,
                                    camera_targets=[target], camera_backgrounds=[shelf])
    assert tracker.support_planes[0] is None
    retained = tracker.reference_surfaces[0].points + tracker.center0
    assert np.count_nonzero(retained[:, 2] < -.01) > 40
    assert np.count_nonzero(retained[:, 2] > .01) > 40


def test_occluding_outer_contour_preserves_last_measurable_translation():
    from atrace.gcp_tracking import ReferenceDepthTracker

    intrinsics = np.array([[320., 0., 48.], [0., 320., 48.], [0., 0., 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = .50
    rows, cols = np.mgrid[:96, :96]
    rays = np.stack(((cols + .5 - 48) / 320, -(rows + .5 - 48) / 320), axis=-1)
    top = rays * .485

    def measure(shift, occlude=False):
        footprint = (np.abs(top[..., 0] - shift) < .040) & (np.abs(top[..., 1]) < .030)
        depth = np.where(footprint, .485, .50)
        blocked = occlude & ((np.abs(top[..., 0]) > .015) | (np.abs(top[..., 1]) > .020))
        depth[blocked] = .30
        points = np.dstack((rays * depth[..., None], .50 - depth))
        return points[footprint & ~blocked], points[~footprint & ~blocked], (depth, intrinsics, pose)

    rng = np.random.default_rng(0)
    target, table, view = measure(0.)
    tracker = ReferenceDepthTracker(target, table, np.zeros(3), np.zeros(3), False, rng,
                                    camera_targets=[target], camera_backgrounds=[table], views=[view])
    for displacement in np.linspace(0., .018, 10):
        target, table, view = measure(displacement)
        cloud = np.vstack((target, table))
        tracker.update(cloud, rng, free=True, camera_points=[cloud], views=[view])
    before = tracker.translation.copy()
    assert tracker.visible and abs(before[0] - .018) < .003

    target, table, view = measure(.018, occlude=True)
    cloud = np.vstack((target, table))
    tracker.update(cloud, rng, free=True, camera_points=[cloud], views=[view])
    assert not tracker.visible
    np.testing.assert_array_equal(tracker.translation, before)


def test_hiding_a_depth_discrepancy_cannot_certify_object_motion():
    from atrace.gcp_release import _ray_cost

    depth = np.ones((8, 8))
    depth[:, 1] = .90  # Foreground occluder alongside the discrepant rays.
    depth[:, 0] = 1.02
    intrinsics = np.diag([100., 100., 1.])
    patch = np.array([[(u+.25)/100., (v+.25)/100., 1.]
                      for u in (0, 3) for v in range(1, 7)])
    candidates = np.stack((patch, patch + [.010, 0., 0.], patch + [0., 0., .010]))
    stationary, hidden, closer = _ray_cost(candidates, (depth, intrinsics, np.eye(4)))
    assert hidden == pytest.approx(stationary)
    assert closer < stationary


def test_opposing_depth_motion_cannot_reverse_a_reference_update(monkeypatch):
    from types import SimpleNamespace
    from atrace import gcp_release

    center = np.array([0., 0., 1.])
    points = np.array([[u*.002, v*.002, 0.] for u in range(-4, 5) for v in range(-3, 4)])
    intrinsics = np.array([[100., 0., 16.], [0., 100., 16.], [0., 0., 1.]])
    view = (np.ones((32, 32)), intrinsics, np.eye(4))
    tracker = SimpleNamespace(center0=center, anchor_center=center-np.array([.020, 0., 0.]),
                              translation=np.zeros(3), rotation=np.eye(3), visible=True)
    tracker._free_surface = lambda: points + center + tracker.translation
    live = [points + center]
    proposal = np.zeros(3)

    def fit_reference(state, *_):
        state.visible = True
        state.translation = proposal.copy()

    monkeypatch.setattr(gcp_release, "_fit_reference", fit_reference)
    monkeypatch.setattr(gcp_release, "_ray_translation",
                        lambda pairs, position, rotation: position + np.array([.010, 0., 0.]))
    gcp_release.register_surfaces(tracker, live, [[view]])
    assert tracker.previous_depth_patches[0] is not None
    gcp_release.register_surfaces(tracker, live, [[view]])
    np.testing.assert_array_equal(tracker.translation, np.zeros(3))
    proposal[0] = -.009
    gcp_release.register_surfaces(tracker, live, [[view]])
    np.testing.assert_array_equal(tracker.translation, np.zeros(3))


def test_place_jitter_settles_with_measured_grasp_and_survives_snapshot(env):
    env.place_grasp_jitter = True
    _, info = env.reset(seed=450241, kind="place", staged=False)
    assert info["place_grasp_jitter"] == "applied"
    assert info["entry_ok"] and info["held"] and info["max_pen_m"] < .001
    grip_R, grip_p = env.fast_rb.eef_pose()
    held_R, held_offset = env.place_grasp_transform
    body_R = env.d.xmat[env.bid].reshape(3, 3)
    np.testing.assert_allclose(grip_R @ held_R, body_R, atol=1e-10)
    np.testing.assert_allclose(grip_p + grip_R @ held_offset, env.target_pos(), atol=1e-10)
    assert 2. < np.rad2deg(np.arccos(body_R[2, 2])) < 10.

    snapshot = env.snapshot()
    env.step(np.array([0., 0., 0., 0., 0., 0., 0., 1.], dtype=np.float32))
    _, info = env.restore(snapshot)
    assert info["held"] and info["place_grasp_jitter"] == "applied"
    for actual, saved in zip(env.place_grasp_transform, (held_R, held_offset)):
        np.testing.assert_array_equal(actual, saved)


def test_place_jitter_rejection_restores_nominal_physics_and_sampling(tmp_path):
    from lrl_train.gcp_env import GCPEnv

    starts = []
    for jitter in (True, None):
        env = GCPEnv(0, tmp_path / str(jitter), preset="all_5_families", place_grasp_jitter=jitter)
        try:
            obs, info = env.reset(seed=450240, kind="place", staged=False)
            assert info["place_grasp_jitter"] == ("nominal_fallback" if jitter else "off")
            assert info["entry_ok"] and info["held"]
            assert info["render_count"] == 1
            starts.append(dict(qpos=env.d.qpos.copy(), qvel=env.d.qvel.copy(), **obs))
        finally:
            env.close()
    for name in starts[0]:
        np.testing.assert_array_equal(starts[0][name], starts[1][name], err_msg=name)
