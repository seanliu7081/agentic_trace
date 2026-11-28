"""Paired depth/physics replay for all six all_5_families scene kinds.

Run with PYTHONPATH=src, MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl.
One row per attempted layout, including entry rejections. Retain every one of
the 130 frames after entry, including after task completion and loss of sight.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


KINDS = ("door", "drawer", "turn", "pick", "place", "planar_push")
FREE = ("pick", "place", "planar_push")
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("src/contact_contracts.py", "src/atrace/harness.py", "src/lrl_train/render.py",
           "src/lrl_train/scenes.py", "src/lrl_train/ucs_scenes.py", "src/lrl_train/sim.py",
           "src/lrl_train/fast_sim.py", "src/lrl_train/_fast_osc.cpp", "src/lrl_train/_fast_osc.so",
           "src/lrl_train/student_obs.py", "src/atrace/geometry.py", "src/atrace/robot_model.py",
           "scripts/eval_gcp_tracking.py")


def source_hashes():
    paths = {ROOT / name for name in SOURCES}
    for pattern in ("src/atrace/gcp*.py", "src/lrl_train/gcp*.py"):
        paths.update(ROOT.glob(pattern))
    return {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def replay(job):
    gpu, kind, seeds, out_dir = job
    from lrl_train.render import egl_device
    egl_device(gpu)
    from lrl_train.gcp_env import GCPEnv
    rows = []
    with TemporaryDirectory(prefix="gcp-replay-") as layouts:
        env = GCPEnv(123, layouts, preset="all_5_families", gpu=gpu, evaluation=True,
                     staged_reset_prob=0, horizon=130)
        try:
            for seed in seeds:
                _, info = env.reset(seed=seed, kind=kind, staged=False)
                row = dict(seed=seed, kind=kind, entry=info["entry_ok"], dropped=False,
                           success=False, errors=[], visible=[], held=[], truth=[], estimate=[])
                if kind == "place":
                    row["fixture"] = env.fixture.fixture_body
                if env.contract.goal_region_ref is not None:
                    row["goal_world"] = (env.reference + env.Rref @ env.contract.goal_region_ref).tolist()
                if kind in FREE:
                    row["geometry"] = [name for name, mask in zip(env.geom_names, env.target_g) if mask]
                if env.entry_ok:
                    falling_from = None
                    for tick in range(130):
                        _, _, done, info = env.step(env.expert_action())
                        assert done == (tick == 129)
                        if env.free:
                            gt = env.target_pos()
                            est = env.tracker.transform(env.q_initial_world)
                            error = 1000 * np.linalg.norm(est - gt)
                        else:
                            factor = 180 / np.pi if env.hinge else 1000
                            est = env.tracker.progress_value * factor
                            gt = env.progress()[int(env.hinge)] * factor
                            error = abs(est - gt)
                        if not np.isfinite(error):
                            raise ValueError(f"non-finite tracking output for {kind} {seed} at {tick}")
                        row["errors"].append(float(error))
                        row["truth"].append(np.asarray(gt).tolist())
                        row["estimate"].append(np.asarray(est).tolist())
                        row["visible"].append(bool(env.tracker.visible))
                        row["held"].append(bool(env.sensed_hold))
                        if kind == "place":
                            # Independent of tracking error and task success. A payload lost
                            # during commanded closure that falls >3 cm without either pad
                            # or designated support is a drop. Commanded releases are scored.
                            c = env.d.contact.geom[:env.d.ncon]
                            supported = np.any((env.target_g[c[:,0]] & env.support_g[c[:,1]]) |
                                               (env.target_g[c[:,1]] & env.support_g[c[:,0]]))
                            pad = env.facts["left_target"] or env.facts["right_target"]
                            if pad or supported or env.requested_width >= .04:
                                falling_from = None
                            else:
                                falling_from = gt[2] if falling_from is None else max(falling_from, gt[2])
                                if falling_from - gt[2] > .03:
                                    row["dropped"] = True
                    row["success"] = bool(info["success"])
                rows.append(row)
        finally:
            env.close()
    path = out_dir / f"{kind}-{seeds[0]}.json"
    path.write_text(json.dumps(rows, separators=(",", ":")))
    return path


def stats(rows):
    err = [v for r in rows for v in r["errors"]]
    return dict(attempts=len(rows), entered=sum(r["entry"] for r in rows),
                completed=sum(r["success"] for r in rows), frames=len(err),
                p95=float(np.percentile(err, 95)) if err else None, max=max(err, default=None),
                frames_over_30=sum(e > 30 for e in err))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(602401, 602433)))
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--chunk-size", type=int, default=4)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or args.chunk_size < 1:
        p.error("seeds must be distinct and chunk-size must be positive")
    args.out.mkdir(exist_ok=True, parents=True)
    revision = source_hashes()
    jobs = [(kind, args.seeds[i:i + args.chunk_size])
            for kind in args.kinds for i in range(0, len(args.seeds), args.chunk_size)]
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        files = list(pool.map(replay, [(args.gpus[n % len(args.gpus)], k, s, args.out)
                                       for n, (k, s) in enumerate(jobs)]))
    rows = [row for f in files for row in json.loads(f.read_text())]
    report = dict(seeds=args.seeds, physical_gpus=args.gpus, families={}, subsets={}, source_sha256=revision,
                  sources_unchanged=source_hashes() == revision)
    for kind in args.kinds:
        cases = [r for r in rows if r["kind"] == kind]
        all_cases = stats(cases)
        metric = all_cases
        limit = 12. if kind in FREE else 5.
        passed = (metric["p95"] is not None and metric["p95"] <= limit
                  and (kind not in FREE or metric["max"] <= 30.))
        completion = (metric["completed"] / metric["entered"]) if metric["entered"] else 0.
        report["families"][kind] = dict(all=all_cases, scored=metric, completion_rate=completion,
                                        passes_completion_limit=completion >= .80,
                                        dropped_seeds=[r["seed"] for r in cases if r["dropped"]],
                                        unit="deg" if kind in ("door","turn") else "mm",
                                        passes_frame_error_limits=passed)
    for name, subset in (("overhang_bowl", [r for r in rows if r.get("fixture") == "ucs_ohdrawer"
                                           and "ucs_payload_base" in r["geometry"]]),
                         ("flat_plate", [r for r in rows if r["kind"] == "planar_push"
                                        and "ucs_payload_plate" in r["geometry"]])):
        report["subsets"][name] = stats(subset)
    report["passes_frame_error_limits"] = (
        report["sources_unchanged"] and all(v["passes_frame_error_limits"] for v in report["families"].values()))
    report["passes_completion_limits"] = all(v["passes_completion_limit"] for v in report["families"].values())
    report["full_requested_coverage"] = (
        set(args.kinds) == set(KINDS) and len(args.seeds) == 32
        and all(v["all"]["frames"] == 130 * v["all"]["entered"] for v in report["families"].values()))
    (args.out/"summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passes_frame_error_limits"] and report["passes_completion_limits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
