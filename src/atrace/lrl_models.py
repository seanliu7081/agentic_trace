"""lrl learned models at run time (LRL_PLAN.md 5.2, 5.3): the skill policy, the outcome detector, and the lookup of
published weights in weights/lrl/manifest.json.

This module only loads and runs models; it does not import the training code (src/lrl_train), so the harness needs
torch and the weight files only. The detector's input features are computed here (trace_features), and the trainer
uses the same function, so training and run time cannot drift apart.

Checkpoint formats:
- policy: the file lrl_train.ppo.save writes, {"dims", "cfg", "state", "meta"}; only the actor and the observation
  statistics are used here.
- detector: {"format": "lrl_detector", "cfg", "state", "classes", "threshold", "feat_mean", "feat_std", "meta"}.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .geometry import mat_to_rotvec, rotvec_to_mat
from .lrl_skill import ACT_DIM, MODES, OBS_DIM

DETECTOR_CLASSES = ("reached", "short", "no_effect", "no_contact", "outside", "lost_grip")

FEAT = {
    "max_rows": 320,       # T  rows of a trace the detector reads, the last ones (way in <= ~60 + 12 s at 20 Hz)
    "p_unit": 0.1,         # T  m: displacement unit of the features
    "v_unit": 0.1,         # T  m/s
    "f_unit": 10.0,        # T  N (as lrl_skill.CFG f_scale)
    "cmd_unit": 0.01,      # T  m: command error unit
    "gap_unit": 0.1,       # T  m
    "time_unit": 240.0,    # T  rows: 12 s at 20 Hz (lrl_contract.LIMITS time)
}
STEP_FEATS = 40
GOAL_FEATS = 5


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------- policy
class Policy:
    """A trained move_part actor: observation (OBS_DIM,) -> deterministic action (ACT_DIM,) in [-1, 1].

    The actor is rebuilt from the state dict (Linear layers with tanh between them) and normalizes its input with
    the running statistics stored in the checkpoint, exactly as lrl_train.ppo.ActorCritic.actor(obs_norm(obs))."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.path, self.sha256 = str(path), _sha256(path)
        st, cfg = ck["state"], ck["cfg"]
        self.meta = ck.get("meta", {})
        self.modes = tuple(self.meta.get("modes", ("push",)))
        n = sorted({int(k.split(".")[1]) for k in st if k.startswith("actor.") and k.endswith(".weight")})
        layers = []
        for i in n:
            w = st[f"actor.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": st[f"actor.{i}.bias"]})
            layers += [lin, nn.Tanh()]
        self.net = nn.Sequential(*layers[:-1]).eval()
        if layers[0].in_features != OBS_DIM or layers[-2].out_features != ACT_DIM:
            raise ValueError(f"{path}: actor is {layers[0].in_features} -> {layers[-2].out_features}, "
                             f"the skill needs {OBS_DIM} -> {ACT_DIM}")
        self.mean, self.var = st["obs_norm.mean"], st["obs_norm.var"]
        self.clip = float(cfg["obs_clip"])

    @torch.no_grad()
    def __call__(self, obs):
        x = torch.as_tensor(np.asarray(obs, float), dtype=torch.float64)
        x = ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip).float()
        return np.clip(self.net(x[None])[0].numpy().astype(float), -1.0, 1.0)


def load_policy(path):
    return Policy(path)


# ----------------------------------------------------------------------------------------------------- detector
def _goal_frame(d):
    """Columns d, e1, e2: a frame fixed by the goal direction and gravity, so the features of a contact do not
    depend on its heading in the room (a wrist sensor and the arm's kinematics give both on a real robot)."""
    d = d / np.linalg.norm(d)
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, d)
    e1 /= np.linalg.norm(e1)
    return np.stack([d, e1, np.cross(d, e1)], 1)


def trace_features(trace, contract):
    """(rows, STEP_FEATS) per-step features and (GOAL_FEATS,) goal features of one contact.

    trace: Trace.arrays() plus "touch" (rows before it are the way in). Positions and rotations are taken from the
    pose at the touch (the last way-in row; the first row when there was no way in) and expressed in the goal frame."""
    u, g = FEAT, contract.goal
    F = _goal_frame(np.asarray(g.direction, float))
    p, rv = np.asarray(trace["p"], float), np.asarray(trace["rv"], float)
    n, touch = len(p), int(trace["touch"])
    ref = max(touch - 1, 0)
    R_ref = rotvec_to_mat(rv[ref])
    rot = np.array([mat_to_rotvec(rotvec_to_mat(r) @ R_ref.T) for r in rv]) @ F
    dp = (p - p[ref]) @ F
    rot_cmd = np.array([mat_to_rotvec(rotvec_to_mat(c) @ rotvec_to_mat(r).T)
                        for c, r in zip(np.asarray(trace["rv_cmd"], float), rv)]) @ F
    gap = np.asarray(trace["gap"], float)
    gap_rate = np.diff(gap, prepend=gap[:1]) / 0.05
    travel = max(float(g.travel), 1e-3)
    along = rot[:, 0] if g.mode == "turn" else dp[:, 0]
    x = np.hstack([
        dp / u["p_unit"], rot,
        np.asarray(trace["v"], float) @ F / u["v_unit"], np.asarray(trace["w"], float) @ F,
        np.asarray(trace["f"], float) @ F / u["f_unit"], np.asarray(trace["t"], float) @ F,
        gap[:, None] / u["gap_unit"], gap_rate[:, None] / u["v_unit"] * u["gap_unit"],
        (np.asarray(trace["p_cmd"], float) - p) @ F / u["cmd_unit"], rot_cmd,
        np.asarray(trace["action"], float),
        (np.arange(n) >= touch)[:, None], (along / travel)[:, None], (np.arange(n) / u["time_unit"])[:, None]])
    goal = np.hstack([np.array(MODES) == g.mode, travel / (1.0 if g.mode == "turn" else u["p_unit"]),
                      n / u["time_unit"]])
    return x[-u["max_rows"]:].astype(np.float32), goal.astype(np.float32)


class DetectorNet(nn.Module):
    """Temporal convolutions over the per-step features, masked mean and max pooling, the last step's features and
    the goal, then a small MLP: class logits, travel (in feature units) and its log standard deviation."""

    def __init__(self, width=96, layers=4, kernel=5, hidden=192, n_cls=len(DETECTOR_CLASSES)):
        super().__init__()
        convs, c_in = [], STEP_FEATS
        for i in range(layers):
            convs += [nn.Conv1d(c_in, width, kernel, padding=kernel // 2 * (1 + i % 2), dilation=1 + i % 2), nn.GELU()]
            c_in = width
        self.conv = nn.Sequential(*convs)
        self.head = nn.Sequential(nn.Linear(2 * width + STEP_FEATS + GOAL_FEATS, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_cls + 2))
        self.n_cls = n_cls

    def forward(self, x, mask, goal, last):
        """x (B, T, STEP_FEATS) normalized, mask (B, T) bool of real rows, goal (B, GOAL_FEATS), last (B, STEP_FEATS)."""
        # padded rows are zeroed after every layer, so a padded batch in training computes what one unpadded
        # trace computes at run time (where the convolutions' own zero padding is all there is)
        h, keep = x.transpose(1, 2), mask[:, None, :].float()
        for layer in self.conv:
            h = layer(h)
            if isinstance(layer, nn.GELU):
                h = h * keep
        h = h.transpose(1, 2)
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = h.masked_fill(~mask[..., None], -1e4).max(1).values
        out = self.head(torch.cat([mean, mx, last, goal], -1))
        return out[:, :self.n_cls], out[:, self.n_cls], out[:, self.n_cls + 1]


class Detector:
    """Outcome of one contact from its trace and contract (LRL_PLAN.md 5.3)."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("format") != "lrl_detector":
            raise ValueError(f"{path} is not an lrl detector checkpoint")
        self.path, self.sha256 = str(path), _sha256(path)
        self.classes = tuple(ck["classes"])
        self.threshold = dict(ck["threshold"])
        self.meta = ck.get("meta", {})
        self.net = DetectorNet(**ck["cfg"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        self.mu, self.sd = (torch.as_tensor(ck[k], dtype=torch.float32) for k in ("feat_mean", "feat_std"))

    @torch.no_grad()
    def __call__(self, trace, contract):
        x, goal = trace_features(trace, contract)
        x = (torch.as_tensor(x) - self.mu) / self.sd
        logits, t, log_sd = self.net(x[None], torch.ones(1, len(x), dtype=torch.bool), torch.as_tensor(goal)[None],
                                     x[None, -1])
        probs = torch.softmax(logits[0], -1).numpy().astype(float)
        p = dict(zip(self.classes, probs))
        if p["reached"] >= self.threshold["reached"]:
            label = "reached"
        else:
            label = max((c for c in self.classes if c != "reached"), key=p.get)
        unit = 1.0 if contract.goal.mode == "turn" else FEAT["p_unit"]
        return {"label": label, "probs": p, "travel": float(t[0]) * unit, "travel_sd": float(np.exp(log_sd[0])) * unit}


def load_detector(path):
    return Detector(path)


# ------------------------------------------------------------------------------------------------------ weights
def find_weights(weights_dir, kind):
    """The manifest entries marked "m1": true for `kind`: dict(policy=path or None, detector=path or None). With
    several marked entries of one part, the latest published wins."""
    weights_dir = Path(weights_dir)
    man = json.loads((weights_dir / "manifest.json").read_text())
    out = {"policy": None, "detector": None}
    stamp = {"policy": "", "detector": ""}
    for name, e in man.items():
        if not e.get("m1") or e.get("kind") != kind:
            continue
        part = "detector" if e.get("part") == "detector" else "policy"
        if e.get("published", "") >= stamp[part]:
            out[part], stamp[part] = str(weights_dir / name), e.get("published", "")
    return out


__all__ = ["DETECTOR_CLASSES", "Policy", "Detector", "DetectorNet", "load_policy", "load_detector", "find_weights",
           "trace_features", "FEAT", "STEP_FEATS", "GOAL_FEATS"]
