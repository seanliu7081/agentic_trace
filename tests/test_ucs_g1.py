"""Unit tests for Gate 1 UCS Skill, Outcome Detector calibration, and Harness bundle loader."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from atrace.ucs_harness import load_ucs_bundle
from atrace.ucs_skill import (
    EXEC_ACT_DIM,
    EXEC_MODES,
    EXEC_OBS_DIM,
    EXEC_STEP_DIM,
    ExecGoal,
    UCSExecSkill,
    neutral_exec_action,
)
from lrl_train import ppo
from lrl_train.train_ucs import DET_CLASSES, TRAIN_CFG, train_detector_from_dumps
from lrl_train.ucs_env import UCS_CRIT_DIM


class _DummyIO:
    def __init__(self):
        self._p = np.array([0.0, 0.0, 1.0])
        self._R = np.eye(3)
        self._q = np.zeros(9)
        self._w = (np.zeros(3), np.zeros(3))
        self.kp = np.full(6, 150.0)
        self.kd = 2.0 * np.sqrt(self.kp)

    def eef_pose(self):
        return self._R.copy(), self._p.copy()

    def wrench(self):
        return self._w[0].copy(), self._w[1].copy()

    def finger_opening(self):
        return 0.03

    def q(self):
        return self._q.copy()

    def step(self, R_goal, p_goal, grip=None, posture=None):
        self._p = self._p + 0.35 * (np.asarray(p_goal) - self._p)
        self._R = np.asarray(R_goal).copy()


def test_ucs_exec_skill_dimensions_and_loop():
    limits = np.array([[-2.89, 2.89]] * 7)
    skill = UCSExecSkill(limits)
    io = _DummyIO()
    goal = ExecGoal(
        mode="pull_open",
        believed_tangent=np.array([1.0, 0.0, 0.0]),
        believed_axis=np.array([1.0, 0.0, 0.0]),
        believed_anchor=np.zeros(3),
        target_span=0.06,
        approach_axis=np.array([-1.0, 0.0, 0.0]),
        way_in_m=0.0,
    )
    obs = skill.start(io, goal, max_steps=40)
    assert obs.shape == (EXEC_OBS_DIM,)
    while skill.stop is None:
        obs = skill.step(neutral_exec_action())
    assert skill.stop in ("reached_span", "time")
    assert len(skill.trace_rows) >= 2


def test_detector_calibration_and_runtime_bundle_roundtrip():
    rng = np.random.default_rng(123)
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        dump_dir = tdir / "dump"
        dump_dir.mkdir()
        rows = []
        for i in range(240):
            cls = DET_CLASSES[i % len(DET_CLASSES)]
            L = int(rng.integers(15, 60))
            tr = rng.normal(scale=0.2, size=(L, EXEC_STEP_DIM)).astype(np.float32)
            # Inject class-separable signature on channel 0..3 so calibration & argmax are meaningful
            tr[:, i % 4] += 3.0
            rows.append({
                "trace": tr,
                "mode": EXEC_MODES[i % len(EXEC_MODES)],
                "target_span": 0.12,
                "label": cls,
                "success": cls == "reached_goal",
            })
        with open(dump_dir / "ucs_trace_0_0000.pkl", "wb") as f:
            pickle.dump(rows, f)

        det_info = train_detector_from_dumps(dump_dir, tdir / "detector_ucs.pt", device="cpu")
        assert det_info["trained"] is True
        assert det_info["val_acc"] > 0.85

        model = ppo.ActorCritic(EXEC_OBS_DIM, UCS_CRIT_DIM, EXEC_ACT_DIM, cfg=TRAIN_CFG)
        ppo.save(tdir / "ckpt_best.pt", model, meta={"modes": EXEC_MODES})

        bundle = load_ucs_bundle(tdir, use_actor=True)
        assert bundle["actor"] is not None
        assert bundle["detector"] is not None
        act = bundle["actor"](np.zeros(EXEC_OBS_DIM))
        assert act.shape == (EXEC_ACT_DIM,)
        pred = bundle["detector"]([rows[0]["trace"][k] for k in range(len(rows[0]["trace"]))], rows[0]["mode"], 0.12)
        assert pred["label"] == "reached_goal"
