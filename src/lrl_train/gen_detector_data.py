"""Episodes for the outcome detector beyond the training rollouts (LRL_PLAN.md 6.4): StageEnv with every episode
written to disk, driven by a given action source, on a given set of kinds and layout seeds.

Sources: base (the base law, zero residual), random (the base law plus white noise on the residual numbers, the
gripper kept closed), policy (a published policy, deterministic, as the harness runs it). Injected failures as in
training (env.CFG inject). Layout seeds: training layouts (below VAL_SEED) for more training data, validation
layouts (VAL_SEED on) for calibration and the report, and the unseen kinds for the transfer report.

  python -m lrl_train.gen_detector_data --source base --set val --episodes 60 --procs 32 --out <dir>
  (episodes land in <dir>/<set>/<source>/dump)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from atrace.lrl_models import load_policy
from atrace.lrl_skill import ACT_DIM

from .env import CFG, StageEnv, TRAIN_KINDS, UNSEEN_KINDS, VAL_SEED

SETS = {
    "train": (TRAIN_KINDS, (0, VAL_SEED)),
    "val": (TRAIN_KINDS, (VAL_SEED, 2 * VAL_SEED)),
    "unseen": (UNSEEN_KINDS, (VAL_SEED, 2 * VAL_SEED)),
}
GEN = {
    "noise": 0.5,           # T  standard deviation of the random source's residual numbers
}


def run(job):
    source, kinds, seeds, n, seed, out, weights = job
    torch.set_num_threads(1)
    policy = load_policy(weights) if source == "policy" else None
    rng = np.random.default_rng(seed)
    # one episode per start: a second one from the same start would be a near copy for the detector
    env = StageEnv(seed, Path(out) / "bddl" / f"w{seed}", CFG | {"kinds": kinds, "seeds": seeds, "dump_share": 1.0,
                                                                "max_loads": 10 ** 6, "episodes_per_start": 1},
                   dump_dir=Path(out) / "dump", source=source)
    for _ in range(n):
        obs, _, _ = env.reset(float(rng.uniform()))         # error levels across the whole range
        ended = False
        while not ended:
            if source == "policy":
                a = policy(obs)
            else:
                a = np.zeros(ACT_DIM)
                if source == "random":
                    a[:9] = GEN["noise"] * rng.standard_normal(9)
                a[9] = 1.0                                  # the pads stay closed
                a[10] = -1.0
            obs, _, _, ended, _ = env.step(a)
    env.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("base", "random", "policy"), required=True)
    ap.add_argument("--set", choices=tuple(SETS), required=True)
    ap.add_argument("--episodes", type=int, default=60, help="per process")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--weights", type=Path, help="the policy of --source policy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    kinds, seeds = SETS[a.set]
    jobs = [(a.source, kinds, seeds, a.episodes, 1000 * a.seed + i, str(a.out / a.set / a.source), str(a.weights))
            for i in range(a.procs)]
    with mp.get_context("fork").Pool(a.procs) as pool:
        print(sum(pool.map(run, jobs)), "episodes")


if __name__ == "__main__":
    main()
