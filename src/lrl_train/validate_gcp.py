"""Select frozen GCP checkpoints on a fixed procedural development panel."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .eval_gcp_smoke import source_hashes, trial
from .gcp_devices import parse_gpus


def implementation_hashes():
    src = Path(__file__).resolve().parent.parent
    names = [path.relative_to(src).as_posix() for path in src.glob("atrace/gcp*.py")]
    names += ["contact_contracts.py", "lrl_train/gcp_evaluation.py", "lrl_train/gcp_devices.py",
              "lrl_train/gcp_teacher.py", "lrl_train/validate_gcp.py", "lrl_train/gcp_pilot_report.py"]
    return {**source_hashes(), **{name: hashlib.sha256((src / name).read_bytes()).hexdigest()
                                 for name in names}}


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def validate(checkpoint, out, n, workers, seed, sources=None, preset="smoke_door_drawer", gpus=(0, 1, 2)):
    sources = implementation_hashes() if sources is None else sources
    checkpoint = checkpoint.resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    output = out / f"{preset}_{digest}_{signature}_{seed}_{n}.json"
    if output.exists():
        saved = json.loads(output.read_text())
        if not saved.get("trajectories") or Path(saved["trajectories"]).is_file():
            return saved
    metadata = {"checkpoint": str(checkpoint), "sha256": digest, "implementation_sha256": sources,
                "n_per_scenario": n, "first_seed": seed, "physical_gpus": list(gpus), "result_file": str(output.resolve()),
                "preset": preset}
    if preset == "all_5_families":
        from .gcp_evaluation import run_panel
        saved = dict(**metadata, **run_panel(checkpoint, output.with_suffix(".jsonl.gz").resolve(), n, workers, seed, gpus))
        if implementation_hashes() != sources or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
            return None
        atomic_json(output, saved)
        return saved

    starts = [(kind, i, seed + offset + i, checkpoint, out, False, gpus)
              for kind, offset in (("door", 0), ("drawer", 50000)) for i in range(n)]
    by_scenario = defaultdict(list)
    with mp.get_context("spawn").Pool(processes=workers) as pool:
        for result in pool.imap_unordered(trial, starts):
            for name, row in result["conditions"].items():
                if name not in ("door40", "door75", "slip75", "drawer"):
                    continue
                traj = row["trajectory"]
                first, last = traj[0], traj[-1]
                fraction = 1 - last["reference_remaining"] / max(1e-5, first["reference_remaining"])
                by_scenario[name].append({
                    "seed": result["seed"], "entered": row["entry"], "success": row["passed"],
                    "safe_success": row["passed"] and last["max_pen_m"] < .001,
                    "max_pen_m": last["max_pen_m"], "max_angle_deg": max(t["angle_deg"] for t in traj),
                    "final_angle_deg": last["angle_deg"], "final_error_m": last["reference_remaining"],
                    "progress_fraction": float(np.clip(fraction, 0., 1.)) if row["entry"] else 0.,
                    "bilateral_contact": any(t["left"] and t["right"] for t in traj),
                    "released_at_end": not (last["left"] or last["right"]),
                    "slip_injected": any(t["slip"] for t in traj), "regrasped": last["regrasp"]})
    wins, progress = [], []
    summary = {}
    for name, cases in sorted(by_scenario.items()):
        cases.sort(key=lambda r: r["seed"])
        rates = {key: float(np.mean([r[key] for r in cases])) for key in
                 ("entered", "success", "safe_success", "bilateral_contact", "released_at_end",
                  "slip_injected", "regrasped", "progress_fraction", "max_angle_deg", "final_angle_deg")}
        summary[name] = rates
        wins.append(rates["safe_success" if name == "drawer" else "success"])
        progress.append(rates["progress_fraction"])
    drawer_pens = [row["max_pen_m"] for row in by_scenario["drawer"] if row["entered"]]
    pen95 = float(np.percentile(drawer_pens,95)) if drawer_pens else None
    ratios = {
        "both_commanded_angles": min(summary["door40"]["success"], summary["door75"]["success"]) / .85,
        "slip_recovery": summary["slip75"]["success"] / .60,
        "drawer_success_and_clearance": min(summary["drawer"]["success"] / .85,
                                            .001 / max(1e-12, pen95) if pen95 is not None else 0.)}
    successes_at_40 = {r["seed"] for r in by_scenario["door40"] if r["success"]}
    successes_at_75 = {r["seed"] for r in by_scenario["door75"] if r["success"]}
    paired = len(successes_at_40 & successes_at_75) / n
    key = sorted(min(1., value) for value in ratios.values()) + [paired, float(np.mean(wins)),
                                                               min(progress), float(np.mean(progress))]
    if implementation_hashes() != sources:
        return None  # Re-evaluate on one implementation if the source changed mid-panel.
    saved = {**metadata, "checkpoint": str(checkpoint), "sha256": digest, "implementation_sha256": sources,
             "n_per_scenario": n, "first_seed": seed,
             "selection_protocol_version": 2, "selection_key": key, "normalized_gate_ratios": ratios,
             "drawer_p95_m": pen95, "success_on_both_angles_same_start": paired,
             "rule": "Lexicographically maximize the three ascending, capped threshold ratios (weaker door angle, slip and drawer including P95). Ties: paired angle success, mean strict success, worst and mean final physical goal progress.",
             "summary": summary, "trials": dict(by_scenario)}
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    temp.replace(output)
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", type=Path, nargs="*")
    ap.add_argument("--watch-run", type=Path, help="evaluate each archived checkpoint as it appears and the final one on completion")
    ap.add_argument("--reference", type=Path, help="also compare the immutable warm-start weight under the same runtime")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preset", choices=("smoke_door_drawer", "all_5_families"), default="smoke_door_drawer")
    ap.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2"))
    ap.add_argument("--n", type=int)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    all5 = args.preset == "all_5_families"
    if args.n is None:
        args.n = 16 if all5 else 6
    if args.seed is None:
        args.seed = 450000 if all5 else 440739
    if all5 and args.seed < 450000:
        ap.error("Step 1B evaluation requires a disjoint development seed >= 450000")
    if args.n <= 0 or args.workers <= 0:
        ap.error("positive --n and --workers required")
    if bool(args.watch_run) == bool(args.checkpoints):
        ap.error("pass explicit checkpoints or --watch-run")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[variable] = "1"
    sources = None
    while True:
        current_sources = implementation_hashes()
        if sources != current_sources:
            reviewed, results = set(), []
            sources = current_sources
        finished = args.watch_run is None or (args.watch_run / "training.json").exists()
        paths = list(args.checkpoints)
        if args.watch_run is not None:
            paths = sorted(args.watch_run.glob("policy_step*.pt"))
            if finished:
                paths.append(args.watch_run / "policy.pt")
        if args.reference is not None:
            paths.insert(0, args.reference)
        for path in paths:
            if path in reviewed:
                continue
            record = validate(path, args.out, args.n, args.workers, args.seed, sources, args.preset, args.gpus)
            if record is None:
                finished = False
                break
            reviewed.add(path)
            # An archived payload and the final path can have identical bytes.
            # Retain both names in the manifest; the content-addressed audit is shared.
            results.append({**record, "checkpoint": str(path.resolve())})
            best = max(results, key=lambda row: row["selection_key"])
            selected = {"selected": best["checkpoint"], "sha256": best["sha256"],
                        "selection_key": best["selection_key"], "rule": best["rule"],
                        "candidates": [{k: r[k] for k in ("checkpoint", "sha256", "selection_key", "result_file")}
                                       for r in results]}
            atomic_json(args.out / "selection.json", selected)
            print(json.dumps({"validated": str(path), "file": record["result_file"],
                              "selection_key": record["selection_key"], "selected": best["checkpoint"]}), flush=True)
        if finished:
            if all5 and args.watch_run is not None:
                from .gcp_pilot_report import write_report
                final_path = (args.watch_run / "policy.pt").resolve()
                final = next(r for r in results if r["checkpoint"] == str(final_path))
                write_report(args.watch_run, best, final, args.out)
            break
        time.sleep(15)


if __name__ == "__main__":
    main()
