"""LRL_PLAN M1: run-time loading of the move_part policy and the outcome detector, and the skill's tare keyword."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from atrace.lrl_contract import LIMITS, Contract, Entry, Goal
from atrace.lrl_models import (DETECTOR_CLASSES, GOAL_FEATS, STEP_FEATS, DetectorNet, find_weights, load_detector,
                               load_policy, trace_features)
from atrace.lrl_skill import ACT_DIM, OBS_DIM, Skill, neutral_action
from lrl_train import ppo

from test_lrl_skill import D, P0, R0, FakeIO

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "lrl"
PUSH = WEIGHTS / "move_part_push_20261125_c3fae1aa.pt"


def _contract(mode="push", travel=0.2):
    return Contract("move_part", "tips", ("part",), [Entry(R0, P0, R0[:, 2], (0.0, 0.05))],
                    Goal(mode, D, None if mode != "turn" else P0, travel), dict(LIMITS), ("reached",))


@pytest.mark.skipif(not PUSH.exists(), reason="published push policy not present")
def test_policy_reproduces_ppo_mean_action():
    pol = load_policy(PUSH)
    model, _ = ppo.load(PUSH)
    assert pol.sha256.startswith("c3fae1aa") and pol.modes == ("push",)
    rng = np.random.default_rng(0)
    mu, sd = model.obs_norm.mean.numpy(), np.sqrt(model.obs_norm.var.numpy())
    for _ in range(5):
        obs = mu + sd * rng.normal(size=OBS_DIM) * 1.5
        a = pol(obs)
        with torch.no_grad():
            ref = model.actor(model.obs_norm(torch.as_tensor(obs)[None]))[0].numpy()
        assert a.shape == (ACT_DIM,) and np.allclose(a, np.clip(ref, -1, 1), atol=1e-5)
        assert np.array_equal(a, pol(obs))                     # deterministic


@pytest.mark.skipif(not PUSH.exists(), reason="published push policy not present")
def test_policy_drives_skill():
    tr = Skill(load_policy(PUSH)).run(FakeIO(wall=(D, D @ P0 + 0.05)), _contract(), max_steps=20)
    assert tr.stop in ("force", "time", "done") and len(tr.rows) > 0


def test_tare_keyword():
    # a hand that starts loaded with 6 N along d: tared against zero, the skill reads the load; by default it does not
    io = FakeIO(wall=(D, D @ P0 - 6.0 / 4000.0))
    s = Skill()
    s.start(io, _contract(), max_steps=2)
    assert np.linalg.norm(s.lp[:3]) < 1.0
    io = FakeIO(wall=(D, D @ P0 - 6.0 / 4000.0))
    s.start(io, _contract(), max_steps=2, tare=(np.zeros(3), np.zeros(3)))
    assert np.linalg.norm(s.lp[:3]) > 1.0
    tr = Skill().run(FakeIO(), _contract(), max_steps=3, tare=(np.zeros(3), np.zeros(3)))
    assert tr.steps == 3


def _detector_file(tmp_path, threshold):
    torch.manual_seed(0)
    net = DetectorNet()
    path = tmp_path / "det.pt"
    torch.save({"format": "lrl_detector", "cfg": {}, "state": net.state_dict(), "classes": DETECTOR_CLASSES,
                "threshold": {"reached": threshold}, "feat_mean": np.zeros(STEP_FEATS, np.float32),
                "feat_std": np.ones(STEP_FEATS, np.float32), "meta": {}}, path)
    return path


@pytest.mark.parametrize("mode", ["push", "pull", "turn"])
def test_detector_interface(tmp_path, mode):
    c = _contract(mode, 1.0 if mode == "turn" else 0.2)
    sk = Skill()
    sk.start(FakeIO(), c, max_steps=15)
    while sk.stop is None:
        sk.step(neutral_action())
    tr = sk.trace.arrays() | {"touch": sk.trace.touch}
    x, g = trace_features(tr, c)
    assert x.shape == (len(sk.trace.rows), STEP_FEATS) and g.shape == (GOAL_FEATS,) and np.isfinite(x).all()
    for thr, reached in ((0.0, True), (1.01, False)):
        out = load_detector(_detector_file(tmp_path, thr))(tr, c)
        assert set(out) == {"label", "probs", "travel", "travel_sd"}
        assert set(out["probs"]) == set(DETECTOR_CLASSES) and abs(sum(out["probs"].values()) - 1) < 1e-5
        assert (out["label"] == "reached") == reached and out["travel_sd"] > 0


def test_find_weights(tmp_path):
    man = {"a.pt": {"kind": "move_part", "part": "skill", "m1": True, "published": "2026-11-25T10:00"},
           "b.pt": {"kind": "move_part", "part": "skill", "m1": True, "published": "2026-11-25T11:00"},
           "c.pt": {"kind": "move_part", "part": "skill", "published": "2026-11-25T12:00"},
           "d.pt": {"kind": "move_part", "part": "detector", "m1": True, "published": "2026-11-25T09:00"},
           "e.pt": {"kind": "put", "part": "skill", "m1": True, "published": "2026-11-25T13:00"}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    assert find_weights(tmp_path, "move_part") == {"policy": str(tmp_path / "b.pt"), "detector": str(tmp_path / "d.pt")}
    assert find_weights(tmp_path, "grasp") == {"policy": None, "detector": None}
