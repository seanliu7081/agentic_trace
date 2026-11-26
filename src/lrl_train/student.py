"""The student policy of M1E (LRL_PLAN.md M1E, E2): the end-to-end contact policy on run-time inputs only, distilled
from the privileged teacher by DAgger (dagger.py). Same action as the teacher (lrl_e2e: ACT_DIM numbers in [-1, 1],
the last one the done flag).

Network: a PointNet-style encoder over the point cloud of student_obs (each point: x, y, z in A, the same relative to
the hand, part flag, wrist flag; one shared MLP per point, max-pooled separately over the live points and the part
points, so the part's few points are not drowned by the scene's many), an MLP over lrl_e2e's proprio, and a head on
both. Outputs: the first ACT_DIM - 1 numbers through tanh (regressed on the teacher's), and a logit for done (the
teacher's done > 0, a classification: done is a threshold decision). Observation scaling is fixed by student_obs and
lrl_e2e (no running statistics, so nothing is lost on a save-and-load round trip).
"""
from __future__ import annotations

import torch
from torch import nn

from atrace.lrl_e2e import ACT_DIM, E2E_CFG, PROPRIO_DIM, STEP_DIM

from .student_obs import CFG as OBS, N_POINTS, POINT_DIM

CFG = {
    "point_mlp": (64, 128, 256),     # T  shared per-point layers
    "proprio_mlp": (256, 256),       # T
    "head": (256, 256),              # T
    "done_weight": 0.5,              # T  weight of the done classification against the action regression
}
HAND = slice((E2E_CFG["stack"] - 1) * STEP_DIM, (E2E_CFG["stack"] - 1) * STEP_DIM + 3)   # latest hand position in A
                                                                                          # (scaled as the points)


def _mlp(sizes, last_act=True):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if last_act or i < len(sizes) - 2:
            layers += [nn.LayerNorm(b), nn.ReLU()]
    return nn.Sequential(*layers)


class Student(nn.Module):
    def __init__(self, cfg=CFG, n_live=OBS["live_points"]):
        super().__init__()
        self.cfg, self.n_live = cfg, n_live
        self.points = _mlp((POINT_DIM + 3,) + tuple(cfg["point_mlp"]))
        self.proprio = _mlp((PROPRIO_DIM,) + tuple(cfg["proprio_mlp"]))
        width = 2 * cfg["point_mlp"][-1] + cfg["proprio_mlp"][-1]
        self.head = _mlp((width,) + tuple(cfg["head"]) + (ACT_DIM,), last_act=False)

    def forward(self, obs):
        """obs (B, OBS_DIM) -> (actions (B, ACT_DIM - 1) in [-1, 1], done logit (B,))."""
        proprio = obs[:, :PROPRIO_DIM]
        pts = obs[:, PROPRIO_DIM:].reshape(len(obs), N_POINTS, POINT_DIM)
        rel = pts[..., :3] - proprio[:, None, HAND]
        f = self.points(torch.cat([pts[..., :3], rel, pts[..., 3:]], -1))
        live, part = f[:, :self.n_live].amax(1), f[:, self.n_live:].amax(1)
        out = self.head(torch.cat([live, part, self.proprio(proprio)], -1))
        return torch.tanh(out[:, :-1]), out[:, -1]

    @torch.no_grad()
    def act(self, obs, draw=True):
        """The action the env takes: the motion and gripper numbers, and done as +1 / -1, drawn with the predicted
        probability (as the teacher's own done is drawn: its mean need never cross zero, ppo.Policy) or, with
        draw=False, at probability 0.5."""
        a, done = self(obs)
        p = torch.sigmoid(done)
        u = torch.rand_like(p) if draw else torch.full_like(p, 0.5)
        return torch.cat([a, torch.where(p > u, 1.0, -1.0)[:, None]], -1)

    def loss(self, obs, target):
        """target (B, ACT_DIM): the teacher's mean numbers, its last column the teacher's probability of done.
        Regression on the numbers, cross-entropy on done; also the agreement of the done decision at 0.5."""
        a, done = self(obs)
        reg = ((a - target[:, :-1].clamp(-1.0, 1.0)) ** 2).mean()
        cls = nn.functional.binary_cross_entropy_with_logits(done, target[:, -1])
        acc = ((done > 0) == (target[:, -1] > 0.5)).float().mean()
        return reg + self.cfg["done_weight"] * cls, {"mse": reg.item(), "done_bce": cls.item(), "done_acc": acc.item()}


def save(path, model, meta=None):
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg, "n_live": model.n_live, "meta": meta or {}}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Student(ck["cfg"], ck["n_live"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["meta"]


__all__ = ["CFG", "Student", "save", "load"]
