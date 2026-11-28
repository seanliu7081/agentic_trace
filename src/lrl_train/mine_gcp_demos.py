"""Mine physically scored demonstrations on training-domain seeds for offline BC.

Each proposal sees the same full handoff snapshot. The deployment runtime has
no dependencies on this optional collector or its privileged proposals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np

PROPOSALS = ("baseline", "load_brake", "seat_then_transport")
CONDITIONS = ("door40", "door75", "slip75", "drawer")


class DemoProposal:
    def __init__(self, env, mode):
        self.env = env
        self.mode = mode
        self.frame = None
        self.samples = []

    def __call__(self, data):
        import torch
        from atrace.geometry import mat_to_rotvec, rotvec_to_mat
        e = self.env
        was_releasing = e.teacher_release_pending
        action = e.expert_action()
        if self.mode == "load_brake" and e.hinge:
            # Replace the default brake and its latch. Preserve the default
            # approach/transport as long as the custom brake is inactive.
            e.teacher_release_pending = was_releasing
            R,p = e.fast_rb.eef_pose()
            h,angle = e.p_contact(),e.progress()[1]
            c = e.contract
            err = c.target_rot_rad-angle
            vel = float(e.d.qvel[e.ss._adr(e.part.joint)[1]])
            if abs(err)>.15:
                e.teacher_release_pending = False
            if abs(err)<.23 and (e.grasp_latched or e.teacher_release_pending):
                if abs(err-.10*vel)<np.deg2rad(3.7) and abs(vel)<np.deg2rad(24.):
                    e.teacher_release_pending = True
                if e.teacher_release_pending:
                    target = p if e.facts["left_target"] or e.facts["right_target"] else h-.048*R[:,2]
                    Rgoal = R
                    action[7] = 1.
                else:
                    lead = np.clip(err-.12*vel+np.clip(.6*err,-.048,.048),-.19,.19)
                    target = e.anchor+rotvec_to_mat(e.axis*lead)@(h-e.anchor)
                    roll = .5 if abs(e.axis@e.contact_R[:,1])>.85 else 1.
                    Rgoal = rotvec_to_mat(e.axis*(roll*(angle+lead)))@e.contact_R
                    action[7] = -1.
                action[:3] = np.clip(R.T@(target-e.core.p_des)/.008,-1.,1.)
                action[3:6] = np.clip(R.T@mat_to_rotvec(Rgoal@e.core.R_des.T)/np.deg2rad(4.5),-1.,1.)
        elif self.mode == "seat_then_transport" and e.hinge and not e.teacher_release_pending:
            if not e.grasp_latched:
                self.frame = None
            R,p = e.fast_rb.eef_pose()
            angle = e.progress()[1]
            if self.frame is None and e.grasp_streak>=3:
                self.frame = (R.copy(),angle)
            if self.frame is None and e.grasp_latched:
                action[:3] = np.clip(R.T@(e.p_contact()-e.core.p_des)/.008,-1.,1.)
                action[3:6] = np.clip(R.T@mat_to_rotvec(R@e.core.R_des.T)/np.deg2rad(4.5),-1.,1.)
                action[7] = -1.
            elif self.frame is not None and abs(e.axis@e.contact_R[:,1])>.85:
                grip_R,grip_angle = self.frame
                err = e.contract.target_rot_rad-angle
                near = abs(err)<.23
                velocity = float(e.d.qvel[e.ss._adr(e.part.joint)[1]])
                lead = np.clip(err-.12*velocity,-.15,.15) if near else np.clip(err,-.16,.16)
                desired = rotvec_to_mat(e.axis*(.5*(angle+lead-grip_angle)))@grip_R
                reference = e.core.R_des if near else R
                action[3:6] = np.clip(R.T@mat_to_rotvec(desired@reference.T)/np.deg2rad(4.5),-1.,1.)
        sample = {k:v[0].numpy().astype(np.float16 if k.endswith("_pts") else np.float32,copy=True)
                  for k,v in data.items()}
        sample["action"] = action.copy()
        self.samples.append(sample)
        return torch.from_numpy(action[None])


def score(episode,condition,tolerance):
    rows = episode["trajectory"]
    final,tail = rows[-1],rows[-10:]
    score = dict(entered=episode["entry"],frames=len(rows)-1,peak_pen_m=final["max_pen_m"],
                 final_angle_deg=final["angle_deg"],visual_error_p95_deg=None)
    if condition=="drawer":
        score["passed"] = (episode["entry"] and len(rows)>=11 and rows[0]["upper_open_fraction"]>=.5
                           and all(abs(r["bottom_open_m"])<=tolerance and r["goal_now"] for r in tail))
        score["final_error"] = abs(final["bottom_open_m"])
    else:
        goal = 40. if condition=="door40" else 75.
        score["final_error"] = abs(final["angle_deg"]-goal)
        score["passed"] = (episode["entry"] and len(tail)==10 and final["success"] and
                           all(abs(r["angle_deg"]-goal)<=5. for r in tail))
    score["zero_pad_open_at_end"] = not(final["left"] or final["right"]) and final["cmd_width_m"]>.030
    if condition=="slip75":
        forced = [i for i,r in enumerate(rows) if r["forced"]]
        regrips = [r for r in rows if r["regrasp"] and r["left"] and r["right"] and not r["forced"]]
        lost = any(r["slip"] and not(r["left"] or r["right"]) for r in rows)
        opened = returned = False
        if forced:
            window = rows[max(0,forced[0]-1):forced[-1]+13]
            score["visual_error_p95_deg"] = float(np.percentile([
                abs(75.-np.rad2deg(r["header_remaining_rad"])-r["angle_deg"]) for r in window],95))
        if regrips:
            first = regrips[0]
            opened = (first["policy_width_m"]<.025 and any(
                r["slip"] and not r["forced"] and r["t"]<first["t"] and r["policy_width_m"]>.030 for r in rows))
            peak_gap = max((r["separation_m"] for r in rows if r["forced"]),default=0.)
            returned = peak_gap-first["separation_m"]>=.005
        origin_kept = all(r["reference_joint_origin"]==rows[0]["reference_joint_origin"] for r in rows)
        score.update(injected=bool(forced),loss=lost,bilateral_regrasp=bool(regrips),
                     autonomous_open_close=opened,returned=returned,origin_kept=origin_kept)
        score["passed"] = bool(score["passed"] and lost and regrips and opened and returned and origin_kept
                               and score["visual_error_p95_deg"] is not None and score["visual_error_p95_deg"]<=5.)
    score["eligible_demo"] = bool(score["passed"] and score["peak_pen_m"]<.001)
    return score


def _candidate_rank(s):
    return (s["eligible_demo"],s["zero_pad_open_at_end"],-s["final_error"],-s["peak_pen_m"])


def collect_seed(job):
    from .eval_gcp_smoke import physics_hash, play
    from .render import egl_device
    gpu = (mp.current_process()._identity[0]-1)%4
    egl_device(gpu)
    from .gcp_env import GCPEnv
    import torch
    torch.set_num_threads(1)
    seed,conditions,out = job
    records = []
    with tempfile.TemporaryDirectory() as td:
        env = GCPEnv(seed,td,gpu=gpu,evaluation=True,staged_reset_prob=0.)
        try:
            for kind in ("door","drawer"):
                branches = [c for c in conditions if (c=="drawer")== (kind=="drawer")]
                if not branches:
                    continue
                env.reset(seed=seed,kind=kind,target_deg=75. if kind=="door" else None,staged=False)
                state = env.snapshot()
                digest = physics_hash(env)
                for name in branches:
                    rows = {}
                    best_rank,best_shard,choice = None,None,None
                    for mode in (("baseline",) if kind=="drawer" else PROPOSALS):
                        goal = None if kind=="drawer" else (40. if name=="door40" else 75.)
                        obs,info = env.restore(state,target_deg=goal,slip_at_deg=15. if name=="slip75" else None)
                        if physics_hash(env)!=digest:
                            raise RuntimeError("Physical handoff changed during demo mining")
                        actor = DemoProposal(env,mode)
                        episode,_ = play(env,actor,obs,info)
                        s = score(episode,name,env.contract.trans_tolerance)
                        rows[mode] = s
                        rank = _candidate_rank(s)
                        if s["eligible_demo"] and (best_rank is None or rank>best_rank):
                            best_rank,choice = rank,mode
                            best_shard = {k:np.stack([x[k] for x in actor.samples]) for k in actor.samples[0]}
                            # Training-only sampling metadata. These arrays are
                            # deliberately absent from the four actor inputs.
                            best_shard["forced"] = np.array([r["forced"] for r in episode["trajectory"][1:]])
                            best_shard["post_slip"] = np.array([r["slip"] for r in episode["trajectory"][:-1]])
                            best_shard["regrasped"] = np.array([r["regrasp"] for r in episode["trajectory"][:-1]])
                            best_shard["goal_satisfied"] = np.array([r["goal_now"] for r in episode["trajectory"][:-1]])
                            best_shard["angle_deg"] = np.array([r["angle_deg"] for r in episode["trajectory"][:-1]],np.float32)
                    case = dict(seed=seed,condition=name,layout=kind,start="fresh",goal_deg=goal,
                                intervention_deg=15 if name=="slip75" else None,handoff_sha256=digest,
                                proposals=rows,selected=choice,frames=0,shard=None)
                    if best_shard is not None:
                        target = out/f"{seed:05d}_{name}.npz"
                        with tempfile.NamedTemporaryFile(dir=out,suffix=".npz",delete=False) as tmp:
                            np.savez_compressed(tmp,**best_shard)
                        Path(tmp.name).replace(target)
                        case.update(frames=len(best_shard["action"]),shard=target.name,
                                    shard_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                    records.append(case)
        finally:
            env.close()
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--seed-start",required=True,type=int)
    ap.add_argument("--num-seeds",required=True,type=int)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--conditions",nargs="+",choices=CONDITIONS,default=list(CONDITIONS))
    ap.add_argument("--drawer-every",type=int,default=1,
                    help="Collect one drawer per this many training seeds, starting at the first seed")
    args = ap.parse_args()
    if not (0<=args.seed_start and 0<args.num_seeds and args.seed_start+args.num_seeds<=100000):
        ap.error("Use only layout seeds in the training domain [0,100000)")
    if not (args.workers>0 and args.drawer_every>0 and len(args.conditions)==len(set(args.conditions))):
        ap.error("Provide a positive worker count and distinct conditions")
    os.environ.setdefault("MUJOCO_GL","egl")
    os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/"manifest.json").exists():
        ap.error("This output already contains a completed manifest")
    from .eval_gcp_smoke import source_hashes
    hashes = source_hashes()
    records = []
    starts = range(args.seed_start,args.seed_start+args.num_seeds)
    jobs = ((s,[c for c in args.conditions if c!="drawer" or (s-args.seed_start)%args.drawer_every==0],
             args.out) for s in starts)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for completed,batch in enumerate(pool.imap_unordered(collect_seed,jobs),1):
            records.extend(batch)
            if completed%8==0:
                print(f"{completed} / {args.num_seeds} training layouts",flush=True)
    records.sort(key=lambda r:(r["seed"],r["condition"]))
    summary = {}
    for name in args.conditions:
        cases = [r for r in records if r["condition"]==name]
        proposals = cases[0]["proposals"]
        summary[name] = dict(starts=len(cases),entries=sum(c["proposals"]["baseline"]["entered"] for c in cases),
                baseline_passes=sum(c["proposals"]["baseline"]["passed"] for c in cases),
                proposal_passes={k:sum(c["proposals"][k]["passed"] for c in cases) for k in proposals},
                any_pass=sum(any(s["passed"] for s in c["proposals"].values()) for c in cases),
                safe_demo_shards=sum(c["selected"] is not None for c in cases),frames=sum(c["frames"] for c in cases))
    if source_hashes()!=hashes:
        raise RuntimeError("Physical or evaluation implementation changed during mining")
    manifest = dict(source_hashes=hashes,collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    actor_keys=["scene_pts","payload_pts","goal","history"],label_key="action",seed_start=args.seed_start,
                    num_seeds=args.num_seeds,conditions=args.conditions,drawer_every=args.drawer_every,summary=summary,records=records)
    target = args.out/"manifest.json"
    with tempfile.NamedTemporaryFile("w",dir=args.out,delete=False) as f:
        json.dump(manifest,f,allow_nan=False)
    Path(f.name).replace(target)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
