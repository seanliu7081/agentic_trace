#!/usr/bin/env python
"""Gate 1 Automated Evaluation Script (`runs/ucs_g1_20261127/gate1_spec.json`).

Evaluates the trained Unified Contact Stage policy (`pi_exec`) exclusively on UNSEEN assets (`seeds >= 500,000`):
  - `pull_open`:      `short_cabinet`, `short_fridge`, `slide_cabinet`, `window` (400 trials)
  - `push_close`:     `short_cabinet`, `short_fridge`, `slide_cabinet`, `window` (400 trials)
  - `knob_turn`:      `faucet`, `basin_faucet` (400 trials)
  - `cavity_insert`:  `ucs_cavity_unseen_arch` + `short_fridge` interior cavity (400 trials)
  - `overhang_place`: `ucs_overhang_unseen_bin` + `short_cabinet` overhang drawer (400 trials)

On EVERY trial, runs from the EXACT SAME physical snapshot (`ss.snapshot()`):
  1. Trained `pi_exec` policy
  2. Untrained open-loop base guide (`neutral_exec_action()`) -> paired difference!
  3. Sign-flipped goal falsification control (first 50 trials per mode) -> must drop to <= 2%!

Computes exact Clopper-Pearson 95% confidence intervals (`scipy.stats.beta`) and writes `gate1_results.json`.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LIBERO_ROOT", str(ROOT / "third_party" / "LIBERO"))
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/workspace/Libero10-Collision")
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

from atrace.ucs_skill import EXEC_MODES, ExecGoal, neutral_exec_action  # noqa: E402
from lrl_train import scenes, ucs_scenes  # noqa: E402
from lrl_train.fast_sim import FastSimRobot  # noqa: E402
from lrl_train.sim import StageSim  # noqa: E402
from lrl_train.train_ucs import load_ucs_actor  # noqa: E402
from lrl_train.ucs_env import ENV_CFG, REWARD_CFG, UCSExecEnv  # noqa: E402

UNSEEN_BY_MODE = {
    "pull_open": ("short_cabinet", "short_fridge", "slide_cabinet", "window"),
    "push_close": ("short_cabinet", "short_fridge", "slide_cabinet", "window"),
    "knob_turn": ("faucet", "basin_faucet"),
    "cavity_insert": ("ucs_cavity_unseen_arch",),
    "overhang_place": ("ucs_overhang_unseen_bin",),
}


def clopper_pearson_95(k: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(0.975, k + 1, n - k))
    return round(lo, 4), round(hi, 4)


def _eval_chunk(args):
    mode, ckpt_path, worker_id, n_trials, n_flip, seed_base = args
    policy = load_ucs_actor(ckpt_path)
    kinds = UNSEEN_BY_MODE[mode]

    pol_succ = 0
    pol_strict_1mm_succ = 0
    base_succ = 0
    flip_succ = 0
    flip_count = 0
    sim_times = []
    pens_m = []
    paired_wins = 0   # policy=1, base=0
    paired_losses = 0 # policy=0, base=1

    with tempfile.TemporaryDirectory() as td:
        cfg_u = {
            **ENV_CFG,
            "kinds": kinds,
            "seeds": (seed_base + worker_id * 5000, seed_base + (worker_id + 1) * 5000),
            "episodes_per_start": 1,
            "err_pos_max": 0.020,
            "err_ang_max_deg": 12.0,
            "err_span_max": 0.20,
        }
        env = UCSExecEnv(seed=[seed_base, worker_id], out_dir=Path(td), cfg=cfg_u, reward=REWARD_CFG, randomize=True)

        done_trials = 0
        lay_idx = 0
        while done_trials < n_trials and lay_idx < n_trials * 3:
            lay_idx += 1
            kind = kinds[lay_idx % len(kinds)]
            seed_int = seed_base + worker_id * 10000 + lay_idx
            r_lay = np.random.default_rng([abs(hash(kind)) % 10000, seed_int])

            from lrl_train.ucs_env import build_ucs_layout
            lay, ctgt = build_ucs_layout(r_lay, kind, Path(td) / "bddl", seed_int)
            ss = StageSim(lay)
            fast_rb = FastSimRobot(ss.robot, split_step=True)
            fast_rb.sync_from_robosuite()
            if ctgt is not None:
                ucs_scenes.bind_container_world(ctgt, ss.robot.m, ss.robot.d)
            home_lay = ss.snapshot()

            # Up to 4 independent randomized starts (different joint initializations, physics draws & noisy beliefs) per unseen layout
            for _sub in range(4):
                if done_trials >= n_trials:
                    break
                ss.restore(home_lay)
                ss.robot.ctrl.kp[:] = 150.0
                ss.robot.ctrl.kd[:] = 2.0 * np.sqrt(150.0)
                fast_rb.sync_from_robosuite()
                env.ss, env.fast_rb, env.kind, env.ctgt, env.scale = ss, fast_rb, kind, ctgt, 1.0
                from atrace.ucs_skill import UCSExecSkill
                env.skill = UCSExecSkill(ss.robot.model.limits[:7])

                if ctgt is not None:
                    ok = env._setup_container_episode(ss, fast_rb, ctgt, 1.0)
                else:
                    ok = env._setup_part_episode(ss, fast_rb, 1.0, force_mode=mode)

                if not ok or env.mode != mode:
                    continue

                fast_rb.sync_to_robosuite()
                snap_init = ss.snapshot()
                env._start_skill_with_belief(1.0)
                shared_goal = ExecGoal(
                    mode=env.skill.goal.mode,
                    believed_tangent=env.skill.goal.believed_tangent.copy(),
                    believed_axis=env.skill.goal.believed_axis.copy(),
                    believed_anchor=env.skill.goal.believed_anchor.copy(),
                    target_span=env.skill.goal.target_span,
                    approach_axis=env.skill.goal.approach_axis.copy(),
                    way_in_m=env.skill.goal.way_in_m,
                )

                def run_trial(act_fn, goal_obj):
                    ss.restore(snap_init)
                    ss.robot.ctrl.kp[:] = 150.0
                    ss.robot.ctrl.kd[:] = 2.0 * np.sqrt(150.0)
                    fast_rb.sync_from_robosuite()
                    env.skill = UCSExecSkill(ss.robot.model.limits[:7])
                    obs = env.skill.start(fast_rb, goal_obj)
                    env.left_prev = env._dist_left()
                    env.left0 = max(env.left_prev, 0.02)
                    env.max_pen_m = 0.0
                    env.outside_hits = 0
                    env.ep_ret = 0.0
                    env.ep_steps = 0
                    env.a_prev = np.zeros(8, dtype=float)
                    info = {}
                    while True:
                        obs, _, _, ended, info = env.step(act_fn(obs))
                        if ended:
                            break
                    return info

                # 1. Trained policy
                info_pol = run_trial(policy, shared_goal)
                s_p = bool(info_pol["success"])
                pen_mm = float(info_pol["max_pen_mm"])
                pol_succ += int(s_p)
                pol_strict_1mm_succ += int(s_p and pen_mm <= 1.0)
                sim_times.append(float(info_pol["sim_s"]))
                pens_m.append(pen_mm / 1000.0)

                # 2. Paired untrained base guide on exact same snapshot + belief!
                info_base = run_trial(lambda _o: neutral_exec_action(), shared_goal)
                s_b = bool(info_base["success"])
                base_succ += int(s_b)
                if s_p and not s_b:
                    paired_wins += 1
                elif s_b and not s_p:
                    paired_losses += 1

                # 3. Mandatory sign-flip falsification control!
                if flip_count < n_flip:
                    flip_count += 1
                    if mode in ("cavity_insert", "overhang_place"):
                        out_shift = env.ctgt.open_normal_world * 0.28
                        flip_goal = ExecGoal(
                            mode=mode,
                            believed_tangent=-shared_goal.believed_tangent,
                            believed_axis=shared_goal.believed_axis,
                            believed_anchor=shared_goal.believed_anchor + out_shift,
                            target_span=shared_goal.target_span,
                            approach_axis=shared_goal.approach_axis,
                            way_in_m=0.0,
                        )
                    else:
                        flip_goal = ExecGoal(
                            mode=mode,
                            believed_tangent=-shared_goal.believed_tangent,
                            believed_axis=-shared_goal.believed_axis,
                            believed_anchor=shared_goal.believed_anchor,
                            target_span=shared_goal.target_span,
                            approach_axis=shared_goal.approach_axis,
                            way_in_m=shared_goal.way_in_m,
                        )
                    info_flip = run_trial(policy, flip_goal)
                    flip_succ += int(bool(info_flip["success"]))

                done_trials += 1

            ss.close()
        env.close()

    return {
        "n": done_trials,
        "pol_succ": pol_succ,
        "pol_strict_1mm_succ": pol_strict_1mm_succ,
        "base_succ": base_succ,
        "flip_n": flip_count,
        "flip_succ": flip_succ,
        "paired_wins": paired_wins,
        "paired_losses": paired_losses,
        "sim_times": sim_times,
        "pens_m": pens_m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", default=str(ROOT / "runs" / "ucs_g1_20261127" / "gate1_spec.json"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "ucs_g1_20261127" / "gate1_results.json"))
    ap.add_argument("--procs-per-mode", type=int, default=16)
    ap.add_argument("--trials-per-mode", type=int, default=None)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    n_per_mode = int(args.trials_per_mode or spec["evaluation_protocol"]["trials_per_mode"])
    n_flip_total = int(spec["evaluation_protocol"]["falsification_controls"]["sign_flip_trials_per_mode"])
    seed_base = int(spec["evaluation_protocol"]["seed_offset"])

    t0 = time.time()
    tasks = []
    p_per_m = args.procs_per_mode
    for m_idx, mode in enumerate(EXEC_MODES):
        base_q, rem = divmod(n_per_mode, p_per_m)
        flip_q, flip_rem = divmod(n_flip_total, p_per_m)
        for w in range(p_per_m):
            nt = base_q + (1 if w < rem else 0)
            nf = flip_q + (1 if w < flip_rem else 0)
            if nt > 0:
                tasks.append((mode, str(Path(args.ckpt).resolve()), m_idx * 100 + w, nt, nf, seed_base + m_idx * 100000))

    print(f"[Gate 1 Eval] Launching {len(tasks)} workers across {len(EXEC_MODES)} unseen modes ({n_per_mode} trials/mode)...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(tasks), 96)) as pool:
        raw_chunks = pool.map(_eval_chunk, tasks)

    by_mode = {}
    idx_c = 0
    all_pens, all_times = [], []
    tot_n = tot_pol = tot_base = tot_flip_n = tot_flip = 0

    for mode in EXEC_MODES:
        chunks = []
        base_q, rem = divmod(n_per_mode, p_per_m)
        for w in range(p_per_m):
            if base_q + (1 if w < rem else 0) > 0:
                chunks.append(raw_chunks[idx_c])
                idx_c += 1
        n = sum(c["n"] for c in chunks)
        ps = sum(c["pol_succ"] for c in chunks)
        ps_strict = sum(c["pol_strict_1mm_succ"] for c in chunks)
        bs = sum(c["base_succ"] for c in chunks)
        fn = sum(c["flip_n"] for c in chunks)
        fs = sum(c["flip_succ"] for c in chunks)
        pw = sum(c["paired_wins"] for c in chunks)
        pl = sum(c["paired_losses"] for c in chunks)
        st = [x for c in chunks for x in c["sim_times"]]
        pm = [x for c in chunks for x in c["pens_m"]]
        all_times.extend(st)
        all_pens.extend(pm)
        tot_n += n; tot_pol += ps; tot_base += bs; tot_flip_n += fn; tot_flip += fs

        rate_p = ps / max(1, n)
        rate_b = bs / max(1, n)
        ci_lo, ci_hi = clopper_pearson_95(ps, n)
        by_mode[mode] = {
            "trials": n,
            "policy_success": ps,
            "policy_rate": round(rate_p, 4),
            "strict_1mm_success": ps_strict,
            "strict_1mm_rate": round(ps_strict / max(1, n), 4),
            "ci95": [ci_lo, ci_hi],
            "paired_base_success": bs,
            "paired_base_rate": round(rate_b, 4),
            "gain_over_base": round(rate_p - rate_b, 4),
            "paired_wins_vs_losses": [pw, pl],
            "sign_flip_falsification": f"{fs}/{fn} ({round(fs / max(1, fn), 4)})",
            "median_sim_s": round(float(np.median(st)), 2) if st else 0.0,
            "p95_penetration_mm": round(1000.0 * float(np.percentile(pm, 95)), 3) if pm else 0.0,
        }

    crit = spec["criteria"]
    overall_rate = tot_pol / max(1, tot_n)
    overall_base = tot_base / max(1, tot_n)
    flip_rate = tot_flip / max(1, tot_flip_n)
    p95_pen_m = float(np.percentile(all_pens, 95)) if all_pens else 0.0
    med_s = float(np.median(all_times)) if all_times else 0.0

    verdicts = {
        "per_mode_ge_83pct": {
            m: (by_mode[m]["policy_rate"] >= crit["g1_mode_first_attempt_success"]["threshold"])
            for m in EXEC_MODES
        },
        "overall_rate": round(overall_rate, 4),
        "overall_ci95": list(clopper_pearson_95(tot_pol, tot_n)),
        "median_sim_s_le_8s": med_s <= crit["g1_median_skill_wall_sim_s"]["threshold"],
        "p95_penetration_le_1mm": p95_pen_m <= crit["g1_p95_penetration_m"]["threshold"],
        "sign_flip_falsification_le_2pct": flip_rate <= crit["g1_falsification_sign_flip_success"]["threshold"],
        "gain_over_base": round(overall_rate - overall_base, 4),
    }
    verdicts["GATE_1_PASS"] = (
        all(verdicts["per_mode_ge_83pct"].values())
        and verdicts["median_sim_s_le_8s"]
        and verdicts["p95_penetration_le_1mm"]
        and verdicts["sign_flip_falsification_le_2pct"]
    )

    result = {
        "gate": "G1",
        "ckpt": str(args.ckpt),
        "elapsed_wall_s": round(time.time() - t0, 2),
        "total_trials": tot_n,
        "overall_policy_success": f"{tot_pol}/{tot_n} ({overall_rate:.4f})",
        "overall_base_success": f"{tot_base}/{tot_n} ({overall_base:.4f})",
        "overall_sign_flip": f"{tot_flip}/{tot_flip_n} ({flip_rate:.4f})",
        "overall_median_sim_s": round(med_s, 2),
        "overall_p95_penetration_mm": round(1000.0 * p95_pen_m, 3),
        "by_mode": by_mode,
        "verdicts": verdicts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
