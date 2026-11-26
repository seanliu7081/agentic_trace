"""PPO with clipping and GAE, in torch, on CPU or one GPU (LRL_PLAN.md 6.3).

The actor sees the skill's observation; the critic sees it too, plus a vector of simulator truth that the actor never
gets. Actions are Gaussian and the skill clips them to [-1, 1]. The done flag is one of the action numbers (the skill
ends the contact when it is positive), so its mean starts well below zero: an untrained policy would otherwise end
half of its contacts at the first step.

Observation statistics are running means and variances held in persistent buffers, so they are part of the state
dict and of every checkpoint.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

CFG = {
    "hidden": 256, "layers": 3,     # T  LRL_PLAN.md 5.2
    "std0": 0.4,                    # T  initial standard deviation of the residual numbers, one value or one per
                                    #    number (unit noise would bury the base law)
    "done_mean0": -1.0,             # T  initial mean of the done number: with done_std0, done is raised by chance in
    "done_std0": 0.3,               # T  0.04 % of the steps (runs 1-3: -1.4 and 0.5, whose draws ended contacts
                                    #    on an open part: train/REPORT.md)
    "obs_clip": 10.0,               # T  normalized inputs are clipped here
    "gamma": 0.99, "lam": 0.95,     # T
    "clip": 0.2,                    # T
    "epochs": 5, "minibatch": 2048,  # T
    "lr": 3e-4, "lr_end": 0.3,      # T  Adam; the rate falls linearly to this share
    "vf_coef": 0.5, "ent_coef": 0.0, "max_grad": 0.5,   # T
    "target_kl": 0.03,              # T  an update stops at the first minibatch whose KL is 1.5 times this
}


class RunningNorm(nn.Module):
    def __init__(self, dim, clip):
        super().__init__()
        self.clip = clip
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    def update(self, x):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        n, mean, var = x.shape[0], x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = mean - self.mean
        self.var.copy_((self.var * self.count + var * n + delta ** 2 * self.count * n / tot) / tot)
        self.mean.add_(delta * n / tot)
        self.count.copy_(tot)

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip).float()


def _mlp(n_in, n_out, cfg, out_gain):
    dims = [n_in] + [cfg["hidden"]] * cfg["layers"]
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.Tanh()]
    layers.append(nn.Linear(dims[-1], n_out))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 2 ** 0.5)
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(layers[-1].weight, out_gain)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, crit_dim, act_dim, cfg=CFG):
        super().__init__()
        self.dims, self.cfg = (obs_dim, crit_dim, act_dim), dict(cfg)
        self.obs_norm, self.crit_norm = RunningNorm(obs_dim, cfg["obs_clip"]), RunningNorm(crit_dim, cfg["obs_clip"])
        self.actor = _mlp(obs_dim, act_dim, cfg, 0.01)      # near-zero output: the policy starts as the base law
        self.critic = _mlp(obs_dim + crit_dim, 1, cfg, 1.0)
        std = torch.cat([torch.as_tensor(cfg["std0"], dtype=torch.float32).expand(act_dim - 1),
                         torch.tensor([cfg["done_std0"]])])
        self.log_std = nn.Parameter(std.log())
        with torch.no_grad():
            self.actor[-1].bias[-1] = cfg["done_mean0"]
            if cfg.get("mean0") is not None:            # initial means of the other numbers (default 0)
                self.actor[-1].bias[:-1] = torch.as_tensor(cfg["mean0"], dtype=torch.float32)

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(self.obs_norm(obs)), self.log_std.exp())

    def value(self, obs, crit):
        return self.critic(torch.cat([self.obs_norm(obs), self.crit_norm(crit)], -1)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, crit, noise=None):
        """Sampled actions for rows of observations (numpy in, numpy out): action, log probability, value.
        `noise`: the standard normal draws to use, one row per observation (a trainer that correlates them in time)."""
        # written out (no distribution object, the observation normalized once): this runs once per vector call
        dev = self.log_std.device
        x = self.obs_norm(torch.as_tensor(obs, device=dev))
        mean, std = self.actor(x), self.log_std.exp()
        a = mean + std * (torch.randn_like(mean) if noise is None
                          else torch.as_tensor(noise, dtype=torch.float32, device=dev))
        logp = (-0.5 * ((a - mean) / std) ** 2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        value = self.critic(torch.cat([x, self.crit_norm(torch.as_tensor(crit, device=dev))], -1)).squeeze(-1)
        out = torch.stack([*a.T, logp, value], 1).cpu().numpy()        # one transfer back
        return out[:, :-2], out[:, -2], out[:, -1]


class Policy:
    """A trained actor as lrl_skill.Skill takes it: observation (OBS_DIM,) -> action (ACT_DIM,). The residual numbers
    are the mean. The done number is drawn as in training, from a generator with a fixed seed: the trained policy
    raises done through the probability of a positive draw, and its mean alone need never cross zero."""

    def __init__(self, model, seed=0):
        self.model, self.rng = model, np.random.default_rng(seed)
        self.done_std = model.log_std[-1].exp().item()

    @torch.no_grad()
    def __call__(self, obs):
        a = self.model.actor(self.model.obs_norm(torch.as_tensor(obs)[None]))[0].numpy().astype(float)
        a[-1] += self.done_std * self.rng.standard_normal()
        return np.clip(a, -1.0, 1.0)


def save(path, model, meta=None):
    state = {k: v.cpu() for k, v in model.state_dict().items()}       # a checkpoint loads on any device
    torch.save({"dims": model.dims, "cfg": model.cfg, "state": state, "meta": meta or {}}, path)


def load(path):
    """(model, meta) of a checkpoint written by save."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCritic(*ck["dims"], ck["cfg"])
    model.load_state_dict(ck["state"])
    return model.eval(), ck["meta"]


def gae(reward, value, ended, last_value, gamma, lam):
    """Advantages and returns of one environment's transitions in order. `ended` marks the last step of an episode;
    last_value is the value of the observation after the final transition (unused when that one ended)."""
    adv = np.zeros(len(reward))
    nxt, run = last_value, 0.0
    for t in range(len(reward) - 1, -1, -1):
        go = 1.0 - ended[t]
        run = reward[t] + gamma * go * nxt - value[t] + gamma * lam * go * run
        adv[t], nxt = run, value[t]
    return adv, adv + value


class PPO:
    def __init__(self, model, cfg=CFG):
        self.model, self.cfg = model, cfg
        self.opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], eps=1e-5)

    def update(self, obs, crit, act, logp, adv, ret, progress=0.0, aux=None, critic_only=False):
        """One PPO update on a batch of transitions (numpy arrays). progress in [0, 1] sets the learning rate.
        The batch's observations enter the running statistics first. aux(model) -> a loss added to every
        minibatch's (a behaviour-cloning anchor); critic_only: the policy loss is left out (a critic warm-up)."""
        c, m = self.cfg, self.model
        dev = m.log_std.device
        m.obs_norm.update(torch.as_tensor(obs, device=dev))
        m.crit_norm.update(torch.as_tensor(crit, device=dev))
        for g in self.opt.param_groups:
            g["lr"] = c["lr"] * (1.0 - (1.0 - c["lr_end"]) * min(1.0, progress))
        obs, crit, act, logp, adv, ret = (torch.as_tensor(x, dtype=torch.float32 if k > 1 else torch.float64,
                                                          device=dev)
                                          for k, x in enumerate((obs, crit, act, logp, adv, ret)))
        n = len(obs)
        stats = {}
        kl, steps = 0.0, 0
        for idx in (i for _ in range(c["epochs"]) for i in torch.randperm(n, device=dev).split(c["minibatch"])):
            if kl > 1.5 * c["target_kl"]:
                break
            d = m.dist(obs[idx])
            lp = d.log_prob(act[idx]).sum(-1)
            ratio = (lp - logp[idx]).exp()
            a = adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pi_loss = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
            v_loss = 0.5 * (m.value(obs[idx], crit[idx]) - ret[idx]).pow(2).mean()
            ent = d.entropy().sum(-1).mean()
            self.opt.zero_grad()
            loss = (0.0 if critic_only else pi_loss - c["ent_coef"] * ent) + c["vf_coef"] * v_loss
            if aux is not None and not critic_only:
                a_loss = aux(m)
                stats_aux = a_loss.item()
                loss = loss + a_loss
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), c["max_grad"])
            self.opt.step()
            kl, steps = (logp[idx] - lp).mean().item(), steps + 1      # of the policy before this step
            stats = {"pi_loss": pi_loss.item(), "v_loss": v_loss.item(), "entropy": ent.item(),
                     "clipped": ((ratio - 1).abs() > c["clip"]).float().mean().item()}
            if aux is not None and not critic_only:
                stats["aux"] = stats_aux
        with torch.no_grad():
            v = m.value(obs, crit)
        return stats | {"kl": kl, "epochs": steps * c["minibatch"] / n,
                        "explained": float(1 - (ret - v).var() / (ret.var() + 1e-8)),
                        "std": m.log_std[:-1].exp().mean().item(), "done_std": m.log_std[-1].exp().item()}
