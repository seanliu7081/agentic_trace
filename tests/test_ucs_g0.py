"""Unit and simulator tests for Unified Contact Stage (`atrace.ucs` and `lrl_train.fast_sim`)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from atrace.geometry import rotvec_to_mat
from atrace.ucs import (
    LeakyPLAICompliantController,
    ObjectPermanenceStore,
    OnlineScrewTracker,
    ScrewHypothesis,
    WholeStrokeSampler,
)
from lrl_train import scenes, sim
from lrl_train.fast_sim import FastSimRobot


def test_screw_propagation_prismatic_revolute_rotary():
    R0 = np.eye(3)
    p0 = np.array([0.0, -0.2, 1.0])

    # Prismatic (drawer / cavity slide)
    sp = ScrewHypothesis("prismatic", np.array([0.0, -1.0, 0.0]), p0, span=0.15, sign=1.0)
    R1, p1 = sp.propagate_pose(R0, p0, 1.0)
    assert np.allclose(R1, R0)
    assert np.allclose(p1, [0.0, -0.35, 1.0])

    # Revolute (90 deg door swing about z axis at x=-0.2, y=-0.2)
    sr = ScrewHypothesis("revolute", np.array([0.0, 0.0, 1.0]), np.array([-0.2, -0.2, 1.0]), span=np.pi / 2, sign=1.0)
    R2, p2 = sr.propagate_pose(R0, p0, 1.0)
    assert np.allclose(p2, [-0.2, 0.0, 1.0], atol=1e-6)
    assert np.isclose(np.linalg.det(R2), 1.0)

    # Rotary (in-place stove knob)
    sk = ScrewHypothesis("rotary", np.array([0.0, 0.0, 1.0]), p0, span=0.6, sign=-1.0)
    R3, p3 = sk.propagate_pose(R0, p0, 1.0)
    assert np.allclose(p3, p0)
    assert not np.allclose(R3, R0)


def test_leaky_plai_caps_normal_force_lead():
    plai = LeakyPLAICompliantController()
    R0, p0 = np.eye(3), np.array([0.1, 0.2, 0.95])
    plai.reset_state(R0, p0, np.zeros(6))

    # Under 25 N normal contact force along +x with zero velocity, commanded +x lead must shrink to <= f_steady_cap / k_contact (~4.56 mm at kp=45)
    R_cmd, p_cmd, info = plai.compute_command(
        delta_pos_world=np.array([0.015, 0.0, 0.0]),
        delta_rot_world=np.array([0.2, 0.0, 0.0]),
        alpha=1.0,
        R_meas=R0,
        p_meas=p0,
        v_meas=np.zeros(3),
        wrench_world=np.array([-25.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    expected_cap = plai.cfg["f_steady_cap"] / plai.cfg["k_contact"] + 1e-6
    assert (p_cmd[0] - p0[0]) <= expected_cap
    assert info["lead_mm"] <= 1000.0 * expected_cap


def test_online_screw_tracker_adapts_to_arc_and_endstop():
    tr = OnlineScrewTracker(np.array([0.0, 1.0, 0.0]))
    # Move along 35-deg angled arc
    u_true = np.array([np.sin(np.radians(35)), np.cos(np.radians(35)), 0.0])
    for k in range(10):
        out = tr.update(p_meas=u_true * (0.004 * k), v_meas=u_true * 0.08, f_tared=np.array([0.0, -3.0, 0.0]))
    assert float(out["tangent"] @ u_true) > 0.96
    assert not out["at_endstop"]

    # Hit rigid endstop (v ~ 0, axial force = 15 N for 6 steps)
    p_stop = u_true * 0.04
    for _ in range(6):
        out = tr.update(p_meas=p_stop, v_meas=np.zeros(3), f_tared=-15.0 * out["tangent"])
    assert out["at_endstop"]


def test_object_permanence_store_fk_and_icp():
    rng = np.random.default_rng(42)
    # Create an asymmetric mug-like L-cloud so 3D rotation/translation ICP is unambiguous
    theta = np.linspace(0, 2 * np.pi, 80)
    z = np.linspace(-0.04, 0.04, 5)
    th_g, z_g = np.meshgrid(theta, z)
    cyl = np.stack([0.035 * np.cos(th_g.ravel()), 0.035 * np.sin(th_g.ravel()), z_g.ravel()], axis=1)
    handle = np.stack([np.linspace(0.035, 0.065, 40), np.zeros(40), np.linspace(-0.02, 0.02, 40)], axis=1)
    pts0 = np.vstack([cyl, handle]) + np.array([0.0, 0.1, 0.94])

    store = ObjectPermanenceStore()
    store.register_seen("mug_1", "yellow mug", pts0)
    store.register_seen("stove_1", "flat stove", np.array([[0.0, -0.2, 0.91]]) + rng.uniform(-0.08, 0.08, (200, 3)))

    R_h0, p_h0 = np.eye(3), np.array([0.0, 0.1, 1.02])
    store.on_grasp("mug_1", R_h0, p_h0)

    # Carry to new pose
    R_h1 = rotvec_to_mat(np.array([0.0, 0.0, 0.4]))
    p_h1 = np.array([-0.15, -0.05, 0.99])
    store.update_held(R_h1, p_h1)
    assert store.entities["mug_1"].status == "held"

    # Simulate release settling (+8 mm shift, 5 deg yaw settle)
    dR_settle = rotvec_to_mat(np.array([0.0, 0.0, np.radians(5.0)]))
    dt_settle = np.array([0.006, -0.005, -0.008])
    true_settled = store.entities["mug_1"].world_points() @ dR_settle.T + dt_settle
    scene_cloud = np.vstack([true_settled, store.entities["stove_1"].world_points()])

    res = store.on_release_and_icp("mug_1", scene_cloud)
    assert res["verified"] is True
    err = np.linalg.norm(store.entities["mug_1"].world_points().mean(axis=0) - true_settled.mean(axis=0))
    assert err < 0.003


def test_fast_sim_robot_matches_robosuite():
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(7)
        lay = scenes.sample_layout(rng, "short_cabinet", Path(td))
        ss = sim.StageSim(lay)
        ss.set_arm(ss.q_home)
        snap = ss.snapshot()
        R0, p0 = ss.robot.eef_pose()
        Rg = rotvec_to_mat(np.array([0.04, -0.03, 0.05])) @ R0
        pg = p0 + np.array([0.015, -0.01, 0.01])

        ss.robot.step(Rg, pg, 1.0)
        mujoco.mj_forward(ss.robot.m, ss.robot.d)
        q_ref = ss.robot.d.qpos.copy()

        ss.restore(snap)
        fast = FastSimRobot(ss.robot)
        fast.sync_from_robosuite()
        fast.step(Rg, pg, 1.0)
        q_fast = ss.robot.d.qpos.copy()

        assert np.max(np.abs(q_fast - q_ref)) < 1e-11

        # Also smoke-test WholeStrokeSampler on cavity insert
        sampler = WholeStrokeSampler(ss.robot.model)
        ins = sampler.sample_cavity_or_overhang_insert(
            p_rest_site=p0 + np.array([0.12, 0.0, -0.05]),
            R_grasp_candidates=[R0],
            open_directions=[np.array([-1.0, 0.0, 0.0])],
            finger_held=0.02,
            q_ref=ss.q_home,
            slide_len=0.08,
        )
        assert len(ins) >= 1
        ss.close()
