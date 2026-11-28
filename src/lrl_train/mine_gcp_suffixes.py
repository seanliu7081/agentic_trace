"""Training-only DAgger: physically rescue suffixes from a frozen actor's own states."""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .mine_gcp_demos import PROPOSALS, DemoProposal, _candidate_rank, score

PRIORITIES = ("regrasp_45_65", "late_65_plus", "lost_8_35")


def _capture_region(env):
    if not env.slip_lost or env.release_left or env.horizon-env.steps<12:
        return None
    angle = np.rad2deg(env.progress()[1])
    if env.regrasped and 45.<=angle<=65.:
        return PRIORITIES[0]
    if env.regrasped and 65.<angle<=85.:
        return PRIORITIES[1]
    if not env.regrasped and 8.<=angle<=35.:
        return PRIORITIES[2]
    return None


def _digest(env,obs):
    from .eval_gcp_smoke import paired_state_hash,physics_hash
    return (physics_hash(env),paired_state_hash(env,obs),env.steps,float(env.initial_joint),
            float(env.contract.target_rot_rad),env.release_left,env.slip_lost,env.regrasped)


@lru_cache(maxsize=1)
def _copy_mjdata():
    """MuJoCo 3.2.6 provides mj_copyData in the library, but not its Python API."""
    import mujoco
    library = Path(mujoco.__file__).parent / f"libmujoco.so.{mujoco.__version__}"
    copy_data = ctypes.CDLL(str(library)).mj_copyData
    copy_data.argtypes = [ctypes.c_void_p] * 3
    copy_data.restype = ctypes.c_void_p
    return copy_data


def _branch(env,snapshot,digest,prefix,policy):
    from .eval_gcp_smoke import play
    # Generic restore rebuilds derived state with mj_forward. Under contact
    # that extra constraint solve can change the warmstart, acceleration and
    # derived sensors. Restore the exact native frame in place; FastSim and
    # the renderer retain their original MjData address. Frames live only in
    # this worker and only for this layout/model.
    if snapshot["model_address"] != env.m._address:
        raise RuntimeError("Mid-contact frame belongs to a different compiled model")
    obs,_ = env.restore(snapshot)
    restored = _copy_mjdata()(env.d._address, env.m._address, snapshot["full_mjdata"]._address)
    if restored != env.d._address:
        raise RuntimeError("MuJoCo failed to restore the full MjData snapshot")
    info = env._info()
    if _digest(env,obs)!=digest:
        raise RuntimeError("Mid-contact restore changed physics, controller, sensing, progress or remaining time")
    suffix,_ = play(env,policy,obs,info)
    start = digest[2]
    rows = prefix[:start+1]+suffix["trajectory"][1:]
    if ([r["t"] for r in rows]!=list(range(env.horizon+1)) or
            len(suffix["trajectory"])-1!=env.horizon-start):
        raise RuntimeError("The branch must use only the original episode's remaining horizon")
    return score(dict(entry=True,trajectory=rows),"slip75",env.contract.trans_tolerance),suffix


def _save_suffix(out,seed,expert,rows,start):
    samples = {k:np.stack([x[k] for x in expert.samples]) for k in expert.samples[0]}
    samples.update(forced=np.array([r["forced"] for r in rows[1:]]),
                   post_slip=np.array([r["slip"] for r in rows[:-1]]),
                   regrasped=np.array([r["regrasp"] for r in rows[:-1]]),
                   goal_satisfied=np.array([r["goal_now"] for r in rows[:-1]]),
                   angle_deg=np.array([r["angle_deg"] for r in rows[:-1]],np.float32),
                   episode_tick=np.arange(start,start+len(expert.samples),dtype=np.int16))
    path = out/f"{seed:05d}_slip75_suffix.npz"
    with tempfile.NamedTemporaryFile(dir=out,suffix=".npz",delete=False) as f:
        np.savez_compressed(f,**samples)
    Path(f.name).replace(path)
    return path.name,hashlib.sha256(path.read_bytes()).hexdigest()


def _job(args):
    from .render import egl_device
    gpu = (mp.current_process()._identity[0]-1)%4
    egl_device(gpu)
    from .eval_gcp_smoke import play
    from .gcp_env import GCPEnv
    from .train_gcp import load_policy
    import torch
    torch.set_num_threads(1)
    seed,path,out = args
    actor = load_policy(path,device="cpu")
    with tempfile.TemporaryDirectory() as td:
        env = GCPEnv(seed,td,gpu=gpu,evaluation=True,staged_reset_prob=0.)
        try:
            obs,info = env.reset(seed=seed,kind="door",target_deg=75.,staged=False,slip_at_deg=15.)
            captured = {}
            def observe_actor(data):
                phase = _capture_region(env)
                if phase is not None and phase not in captured:
                    snapshot = env.snapshot()
                    snapshot["full_mjdata"] = copy.copy(env.d)
                    snapshot["model_address"] = env.m._address
                    captured[phase] = (snapshot,_digest(env,env.obs))
                return actor(data)
            actual,_ = play(env,observe_actor,obs,info)
            original = score(actual,"slip75",env.contract.trans_tolerance)
            record = dict(seed=seed,condition="slip75",start="learner_prefix",goal_deg=75.,
                          intervention_deg=15.,original=original,captured_regions=list(captured),
                          branch_region=None,branch_tick=None,branch_angle_deg=None,reference_joint_origin=None,
                          snapshot_sha256=None,original_policy_replay=None,proposals={},selected=None,
                          shard=None,frames=0)
            if original["passed"] or not captured:
                return record
            phase = next(k for k in PRIORITIES if k in captured)
            snapshot,digest = captured[phase]
            start = digest[2]
            prefix = actual["trajectory"]
            record.update(branch_region=phase,branch_tick=start,branch_angle_deg=prefix[start]["angle_deg"],
                          reference_joint_origin=digest[3],snapshot_sha256=digest[1],
                          handoff_sha256=digest[0],remaining_steps=env.horizon-start)
            baseline,suffix = _branch(env,snapshot,digest,prefix,actor)
            record["original_policy_replay"] = baseline
            record["actor_replay_max_angle_difference_deg"] = max(
                abs(a["angle_deg"]-b["angle_deg"]) for a,b in zip(prefix[start+1:],suffix["trajectory"][1:]))
            if prefix[start+1:] != suffix["trajectory"][1:]:
                raise RuntimeError(f"Restored actor trajectory differs from the original at seed {seed}, tick {start}")
            record["exact_original_actor_replay"] = True
            if baseline["passed"]:
                return record
            best = None
            for mode in PROPOSALS:
                expert = DemoProposal(env,mode)
                judge,result = _branch(env,snapshot,digest,prefix,expert)
                record["proposals"][mode] = judge
                if judge["eligible_demo"] and (best is None or _candidate_rank(judge)>best[0]):
                    best = (_candidate_rank(judge),mode,expert,result["trajectory"])
            if best is not None:
                _,mode,expert,rows = best
                name,sha = _save_suffix(out,seed,expert,rows,start)
                record.update(selected=mode,frames=len(expert.samples),shard=name,shard_sha256=sha)
            return record
        finally:
            env.close()


def main():
    from .eval_gcp_smoke import source_hashes
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--expected-sha256",required=True)
    ap.add_argument("--seed-start",type=int,required=True)
    ap.add_argument("--num-seeds",type=int,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=4)
    args = ap.parse_args()
    if args.num_seeds<=0 or args.seed_start<0 or args.seed_start+args.num_seeds>100000 or args.workers<=0:
        ap.error("Use positive workers and an interval entirely within training [0,100000)")
    path = args.checkpoint.resolve()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha!=args.expected_sha256:
        ap.error("Immutable checkpoint SHA-256 mismatch")
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = args.out/"manifest.json"
    if manifest.exists():
        ap.error("Output contains a completed manifest")
    sources=source_hashes()
    sources.update(demo_proposals=hashlib.sha256(Path(__file__).with_name("mine_gcp_demos.py").read_bytes()).hexdigest(),
                   suffix_collector=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata = dict(actor_sha256=sha,checkpoint=str(path),actor_keys=["scene_pts","payload_pts","goal","history"],
                    label_key="action",seed_start=args.seed_start,num_seeds=args.num_seeds,conditions=["slip75"],
                    source_hashes=sources,capture_priority=PRIORITIES,prefix_label_frames=0,horizon=130,
                    criteria="Capture first eligible state in the highest-priority available phase. Collect suffixes only if both original and restored actor fail, a physically executed expert suffix passes the full original-episode slip gate and peak forbidden penetration <1 mm. Pilot expansion requires >=4 rescues, >=2 from the primary phase, >=160 actual suffix frames.")
    (args.out/"protocol.json").write_text(json.dumps(metadata,indent=2))
    jobs = [(s,path,args.out) for s in range(args.seed_start,args.seed_start+args.num_seeds)]
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for completed,row in enumerate(pool.imap_unordered(_job,jobs),1):
            rows.append(row)
            if completed%8==0:
                print(f"{completed} / {len(jobs)} actor episodes",flush=True)
    rows.sort(key=lambda r:r["seed"])
    chosen = [r for r in rows if r["selected"] is not None]
    phases = dict(Counter(r["branch_region"] for r in chosen))
    replay_drift = [r["actor_replay_max_angle_difference_deg"] for r in rows if r["original_policy_replay"]]
    frames = sum(r["frames"] for r in chosen)
    summary = dict(starts=len(rows),entries=sum(r["original"]["entered"] for r in rows),
                   original_actor_passes=sum(r["original"]["passed"] for r in rows),candidate_regions=(
                       {p:sum(p in r["captured_regions"] for r in rows) for p in PRIORITIES}),
                   replayed_actor_failures=sum(r["original_policy_replay"] is not None and not
                                                r["original_policy_replay"]["passed"] for r in rows),
                   rescue_episodes=len(chosen),safe_demo_shards=len(chosen),rescued_by_region=phases,frames=frames,
                   actor_replay_max_trajectory_angle_difference_deg=max(replay_drift,default=None),
                   pilot_expansion_supported=len(chosen)>=4 and phases.get(PRIORITIES[0],0)>=2 and frames>=160)
    if any(source_hashes()[k]!=v for k,v in sources.items() if k not in ("demo_proposals","suffix_collector")):
        raise RuntimeError("Runtime or scorer changed during collection")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        raise RuntimeError("Actor changed during collection")
    metadata.update(summary={"slip75":summary},records=rows)
    with tempfile.NamedTemporaryFile("w",dir=args.out,delete=False) as f:
        json.dump(metadata,f,allow_nan=False)
    Path(f.name).replace(manifest)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
