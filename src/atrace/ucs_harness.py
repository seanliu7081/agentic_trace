"""Harness integration (`UCSMixin`) for Unified Contact Stage (`--placement ucs`).

Implements the unified contact-stage loop inside `agentic_trace`'s `Harness`:
  1. `WholeStrokeSampler` (`ucs.py`) with cylindrical roll relaxation & steep approach tilts (`0/28/52/68 deg`,
     solving low-bar bottom-drawer grasps `4.6 cm` above the table where `v2` and `M1` scored `0/3`), checked
     against observed scene clearance (`allowed` = part + fixture + fixture skin) over the full stroke.
  2. `UCSExecSkill` (`ucs_skill.py`) running our `<1.0 mm` `LeakyPLAICompliantController` + `OnlineScrewTracker`
     at 20 Hz (using the validated screw-compliant base guide by default, or the learned residual actor when
     `use_actor=True`).
  3. `UCSOutcomeDetector` (`detector_ucs.pt`, temperature-calibrated 1D-CNN) for in-place continuation on
     `progress_short` and fast `<4 s` zero-S1-call candidate switching on `slipped_or_missed` / `blocked_obstacle`
     (marking only the candidate pose, never crossing off a whole valid face on a single attempt).
  4. `ObjectPermanenceStore` (`ucs.py`) hooked into `place_object` so grasped shapes stay frozen and post-release
     poses are registered via depth ICP.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import articulate, manip
from .body import hand_spheres, payload_spheres
from .guard import StepGuard
from .levels import LADDER
from .sim_iface import HEAD_CAMERA
from .ucs import (
    ObjectPermanenceStore,
    ScrewHypothesis,
    WholeStrokeSampler,
    _unit,
)
from .ucs_skill import (
    EXEC_CFG,
    EXEC_MODES,
    EXEC_STEP_DIM,
    ExecGoal,
    UCSExecSkill,
    neutral_exec_action,
)

UCS_HARNESS_CFG = {
    "max_candidates": 6,         # T  max contact candidates entered per stage
    "max_in_place_runs": 2,      # T  continuation runs in place when detector / travel reports partial progress
    "open_span_m": 0.38,         # T  max pull arc/travel (end-stop stops drawers at 16 cm; hinged doors swing >=75 deg!)
    "turn_span_rad": 1.05,       # T  ~60 deg per turn run (covers LIBERO 0.5 rad & RoboCasa 0.35 rad cleanly)
    "fixture_skin_m": 0.012,     # T  unlabelled surface within 1.2 cm of fixture belongs to fixture
    "limit_slack_rad": 0.10,     # T  escape posture if look leaves arm near joint limit
    "backoff_m": 0.06,           # T  guarded retreat distance after contact
    "backoff_speed": 0.12,       # T  m/s
}


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:8]


def load_ucs_bundle(weights_dir: str | Path, use_actor: bool = False) -> dict:
    """Load `detector_ucs.pt` (and optionally `ckpt_best.pt` actor when `use_actor=True`)."""
    wdir = Path(weights_dir)
    ckpt = wdir / "ckpt_best.pt"
    if not ckpt.exists():
        ckpt = wdir / "ckpt_latest.pt"
    det_pt = wdir / "detector_ucs.pt"

    actor_fn = None
    actor_sha = None
    if use_actor and ckpt.exists():
        actor_sha = _sha256_short(ckpt)
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        st, cfg = raw["state"], raw["cfg"]
        idx_l = sorted({int(k.split(".")[1]) for k in st if k.startswith("actor.") and k.endswith(".weight")})
        layers = []
        for i in idx_l:
            w = st[f"actor.{i}.weight"]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.load_state_dict({"weight": w, "bias": st[f"actor.{i}.bias"]})
            layers.extend([lin, torch.nn.Tanh()])
        net = torch.nn.Sequential(*layers[:-1]).eval()
        mean, var = st["obs_norm.mean"], st["obs_norm.var"]
        clip = float(cfg["obs_clip"])

        @torch.no_grad()
        def _act(obs: np.ndarray) -> np.ndarray:
            x = torch.as_tensor(np.asarray(obs, float), dtype=torch.float64)
            xn = ((x - mean) / torch.sqrt(var + 1e-8)).clamp(-clip, clip).float()
            return np.clip(net(xn[None])[0].numpy().astype(float), -1.0, 1.0)

        actor_fn = _act

    det_fn = None
    det_sha = None
    if det_pt.exists():
        det_sha = _sha256_short(det_pt)
        draw = torch.load(det_pt, map_location="cpu", weights_only=False)
        classes = tuple(draw["classes"])
        temp = float(draw.get("temperature", 1.0))

        hidden = 128
        conv = torch.nn.Sequential(
            torch.nn.Conv1d(EXEC_STEP_DIM, hidden, 5, padding=2), torch.nn.GELU(),
            torch.nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), torch.nn.GELU(),
            torch.nn.Conv1d(hidden, hidden, 3, padding=1), torch.nn.GELU(),
        )
        head = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden + EXEC_STEP_DIM + len(EXEC_MODES) + 1, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, len(classes)),
        )
        sd = draw["state_dict"]
        conv.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("conv.")})
        head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
        conv.eval()
        head.eval()

        @torch.no_grad()
        def _detect(trace_rows: list[np.ndarray], mode: str, target_span: float) -> dict:
            if not trace_rows:
                return {"label": "slipped_or_missed", "confidence": 1.0, "probs": {}}
            arr = np.asarray(trace_rows[-140:], dtype=np.float32)
            x = torch.as_tensor(arr)[None]
            h = conv(x.transpose(1, 2))
            mp = h.mean(dim=-1)
            xp = h.amax(dim=-1)
            last = x[:, -1, :]
            meta = torch.as_tensor(
                np.hstack([(np.array(EXEC_MODES) == mode).astype(np.float32), [float(target_span)]]),
                dtype=torch.float32,
            )[None]
            logits = head(torch.cat([mp, xp, last, meta], dim=-1))[0]
            probs = torch.softmax(logits / max(0.2, temp), dim=-1).numpy()
            best = int(np.argmax(probs))
            return {
                "label": classes[best],
                "confidence": round(float(probs[best]), 4),
                "probs": {c: round(float(p), 4) for c, p in zip(classes, probs)},
            }

        det_fn = _detect

    suffix = "+ucs" + (f"a{actor_sha}" if actor_sha else "base") + (f"d{det_sha}" if det_sha else "")
    return {
        "dir": str(wdir),
        "ckpt": str(ckpt) if actor_sha else None,
        "detector_ckpt": str(det_pt) if det_sha else None,
        "use_actor": bool(use_actor and actor_fn is not None),
        "actor": actor_fn,
        "detector": det_fn,
        "suffix": suffix,
    }


def _extract_part_handle_geometry(part_pts: np.ndarray, n: np.ndarray, p_face_0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract handle center, principal handle axis, across-width, and standoff-behind from observed 3D part points."""
    P = np.asarray(part_pts, dtype=float)
    n = _unit(n)
    proj = (P - p_face_0) @ n
    p92 = float(np.percentile(proj, 92))
    p20 = float(np.percentile(proj, 20))
    protrusion = max(0.0, p92 - p20)

    # Outer proud slab = handle / outermost graspable feature
    slab = P[proj >= p92 - max(0.012, 0.45 * protrusion)]
    if len(slab) < 8:
        slab = P
    center = np.median(slab, axis=0)

    # Principal axis in the plane perpendicular to n
    slab_2d = slab - np.outer((slab - center) @ n, n)
    cov = np.cov((slab_2d - slab_2d.mean(axis=0)).T) + 1e-9 * np.eye(3)
    w, V = np.linalg.eigh(cov)
    # V[:, 2] is longest axis, V[:, 1] is across-width axis in the face plane
    long_ax = V[:, 2] - float(V[:, 2] @ n) * n
    if np.linalg.norm(long_ax) < 1e-4:
        up = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.85 else np.array([1.0, 0.0, 0.0])
        long_ax = np.cross(up, n)
    long_ax = _unit(long_ax)
    across_ax = _unit(np.cross(n, long_ax))

    across_proj = (slab - center) @ across_ax
    width = float(np.clip(np.percentile(across_proj, 95) - np.percentile(across_proj, 5), 0.012, 0.068))
    stand_behind = float(np.clip(protrusion, 0.018, 0.045))
    return center, long_ax, width, stand_behind


class UCSMixin:
    """Methods mixed into `Harness` for `--placement ucs`."""

    def ucs_setup(self, placement: str, ucs_bundle: dict | None = None):
        self.ucs = placement == "ucs"
        self.ucs_store = ObjectPermanenceStore()
        self.ucs_bundle = ucs_bundle if self.ucs else None

    def _ucs_act_fn(self):
        if self.ucs_bundle and self.ucs_bundle.get("actor") is not None:
            return self.ucs_bundle["actor"]
        return lambda _obs: neutral_exec_action()

    def _ucs_detect(self, trace_rows: list[np.ndarray], mode: str, span: float) -> dict:
        if self.ucs_bundle and self.ucs_bundle.get("detector") is not None:
            return self.ucs_bundle["detector"](trace_rows, mode, span)
        return {"label": "reached_goal", "confidence": 0.5, "probs": {}}

    def _ucs_stage_mem(self, result: dict) -> dict:
        m = getattr(self, "_ucs_mem", None)
        if m is None or m["result"] is not result:
            m = self._ucs_mem = {"result": result, "entered": 0, "tried_pts": [], "dead_axes": []}
        return m

    def _ucs_scene_col_fn(self, part, parent, allowed, own):
        """Collision check callback `(q7, finger, step_i, s_frac) -> bool` using Harness's observed voxel/robot model.
        At `s_frac < 0` (handoff standoff), checks full arm + hand clearance; during stroke (`s_frac >= 0`), allows
        fingers/hand near `allowed` (part + fixture) while keeping arm links clear of table & obstacles!"""
        excl = (part.oid,) + ((parent.oid,) if parent is not None else ())

        def check_fn(q7: np.ndarray, finger: float, _step_i: int, s_frac: float) -> bool:
            if s_frac < 0.0:
                bad = self._lrl_meets(q7[None], excl, allowed, own, finger=finger)
                return bad is None
            # During stroke, verify arm links stay clear of table and non-fixture obstacles
            cl = self.checker.clearance(q7[None], exclude=allowed, finger=finger)
            if float(cl["arm"][0]) < 0.008 or float(cl["plane_arm"][0]) < 0.012:
                return False
            # Also keep hand above the tabletop (1 mm margin for low bottom-drawer handles 4.6 cm above table)!
            W, _ = self.checker.world_points(q7[None], finger)
            hand_z_min = float(W[0][self.checker.pg >= 1, 2].min())
            return hand_z_min >= self.table_z + 0.001

        return check_fn

    def _ucs_sample_part_candidates(self, ucs_mode: str, part, parent, mf: dict, span: float, allowed, own, failed_pts):
        """Combine `WholeStrokeSampler` (steep low-bar tilts + cylindrical roll relaxation) with `part_grasps`."""
        n = _unit(np.asarray(mf["n"], float))
        p0 = np.asarray(mf["p0"], float)
        P = part.points
        hp, ha, hw, stand = _extract_part_handle_geometry(P, n, p0)

        if ucs_mode == "knob_turn":
            screw = ScrewHypothesis("revolute", n, np.asarray(mf["axis_point"], float), span, sign=1.0)
        else:
            s_dir = -1.0 if ucs_mode == "push_close" else 1.0
            screw = ScrewHypothesis("prismatic", n, hp, span, sign=s_dir)
            if ucs_mode == "push_close":
                # Seat closed fingertips just outside handle / outer slab
                hp = hp + n * 0.006
                stand = 0.002

        col_fn = self._ucs_scene_col_fn(part, parent, allowed, own)
        sampler = WholeStrokeSampler(self.rm, collision_fn=col_fn)
        q_seed = getattr(self, "q_home", self.rb.q()[:7])

        max_f = 6 if ucs_mode == "knob_turn" else 4
        # Try 3 offsets along handle long axis (center, +2.5cm, -2.5cm)
        cands = []
        for off in (0.0, 0.025, -0.025):
            pt = hp + ha * off
            if any(np.linalg.norm(pt - fp) < 0.012 for fp in failed_pts):
                continue
            found = sampler.sample_handle_or_knob(
                mode=ucs_mode,
                handle_center=pt,
                handle_axis=ha,
                handle_width=hw,
                face_normal=n,
                screw=screw,
                q_ref=q_seed,
                standoff_behind=stand,
                max_found=max_f,
                n_random=10,
            )
            cands.extend(found)
            if len(cands) >= max_f:
                break

        # Also supplement with Harness's antipodal part_grasps (if WholeStrokeSampler found < 2 on low/irregular shapes)
        if len(cands) < 2 and ucs_mode in ("pull_open", "knob_turn"):
            plan, _ = self._lrl_grip(part, parent, mf, failed_pts, "pull" if ucs_mode == "pull_open" else "rotate",
                                     allowed, own, [])
            if plan is not None:
                g, q_pre, q_g = plan
                from .ucs import ContactCandidate
                cands.append(ContactCandidate(
                    name="antipodal_fallback",
                    mode=ucs_mode,
                    R_handoff=g.R,
                    p_handoff=g.p - g.R[:, 2] * 0.06,
                    R_contact=g.R,
                    p_contact=g.p,
                    approach=g.R[:, 2],
                    finger_open=g.info["finger"],
                    q_handoff=q_pre,
                    q_chain=np.stack([q_pre, q_g], axis=0),
                    screw=screw,
                    min_limit_margin_rad=0.12,
                ))
        return cands

    # -------------------------------------------------------------------------------- linear actuation (push / pull)
    def ucs_move(self, motion: str, part, parent, mf: dict, failed: list, result: dict, want, remaining, front_pt, fixture_pt):
        """Execute one unified contact candidate on mounting face `mf` (`push` or `pull`).
        Returns stroke dict compatible with `Harness._actuate_linear`, `'no_plan'`, or `None` (transit failed)."""
        P = part.points
        if len(P) < 20:
            result["attempts"].append({"status": "no_feasible_grip_or_press", "why": "no part model"})
            return "no_plan"

        n = _unit(np.asarray(mf["n"], float))
        mem = self._ucs_stage_mem(result)
        fix_oids, fix_pts = self._fixture(part, part)
        allowed = (part.oid,) + fix_oids
        own = [(fix_pts, UCS_HARNESS_CFG["fixture_skin_m"], True)] if len(fix_pts) else []

        ucs_mode = "push_close" if motion == "push" else "pull_open"
        span = (
            float(remaining + 0.025) if (motion == "push" and remaining is not None)
            else UCS_HARNESS_CFG["open_span_m"]
        )

        cands = self._ucs_sample_part_candidates(ucs_mode, part, parent, mf, span, allowed, own, failed + mem["tried_pts"])
        if not cands:
            result["attempts"].append({"status": "no_ucs_candidate", "face_n": np.round(n, 3).tolist()})
            return "no_plan"

        cand = cands[0]
        mem["entered"] += 1
        mem["tried_pts"].append(cand.p_contact.copy())

        is_push = ucs_mode == "push_close"
        f_trans = 0.0 if is_push else cand.finger_open
        if is_push:
            self.ex.gripper(1.0, max_steps=20)
        else:
            self.ex.gripper_to(2.0 * (cand.finger_open + manip.PAD_OFFSET))

        q_touch = cand.q_chain[min(3, len(cand.q_chain) - 1)]
        alts = [
            {
                "R": c_alt.R_contact,
                "p": c_alt.p_contact,
                "q": c_alt.q_chain[min(3, len(c_alt.q_chain) - 1)],
                "approach": c_alt.approach,
                "window": None,
            }
            for c_alt in cands[1:]
        ]
        with self._lrl_free(own):
            end = self.reach(
                cand.R_contact,
                cand.p_contact,
                q_touch,
                cand.approach,
                part.oid,
                None,
                "ucs_to_part",
                1.0 if is_push else 0.0,
                f_trans,
                cand.q_handoff,
                alts=alts,
                standoff=(0.0, 0.14),
            )
        if end is None:
            dist_short = float(np.linalg.norm(self.rb.eef_pose()[1] - cand.p_contact))
            if dist_short > 0.085:
                failed.append(cand.p_contact)
                result["attempts"].append({"status": "transit_failed", "pose": cand.name, "why": (self.last_move or {}).get("why")})
                return None
        else:
            k_sel = int(getattr(end, "k", 0))
            if k_sel > 0 and k_sel - 1 < len(cands) - 1:
                cand = cands[k_sel]
            cand_R_actual = np.asarray(end.R, float)
            cand_p_actual = np.asarray(end.p, float)
            cand_a_actual = cand_R_actual[:, 2]
        if end is None:
            cand_R_actual, cand_p_actual, cand_a_actual = cand.R_contact, cand.p_contact, cand.approach

        # Cover the final centimeters from TRACE's standoff to cand_p_actual with Harness's model-inverse feedforward
        # (`move_line`, <1mm lag) so tilted low-bar grasps land squarely around the bar instead of 1.3cm short!
        if is_push:
            def _touch_g(robot, F, F0, k):
                return "contact" if np.linalg.norm(F - F0) > 8.0 and k > 2 else None
            self.move_line(cand_R_actual, cand_p_actual + cand_a_actual * 0.008, grip=1.0, speed=0.05,
                           guard=self._guard("contact", _touch_g), finger=0.0)
        else:
            def _early_g(robot, F, F0, k):
                return "early_contact" if np.linalg.norm(F - F0) > 12.0 and k > 2 else None
            self.move_line(cand_R_actual, cand_p_actual, grip=0.0, speed=0.05,
                           guard=self._guard("early_contact", _early_g), finger=cand.finger_open)
            opening = self.ex.gripper(1.0, max_steps=22)
            if opening < 0.002:
                # Closed on empty air: fast <2s local backoff & try next candidate without S1 call!
                self.ex.gripper_to(2.0 * (cand.finger_open + manip.PAD_OFFSET))
                Rs, ps = self.rb.eef_pose()
                self.retreat(Rs, ps - cand_a_actual * 0.06, grip=0.0, speed=0.12, finger=cand.finger_open, tag="ucs_miss_back")
                failed.append(cand_p_actual)
                result["attempts"].append({"status": "missed_part", "pose": cand.name, "opening": round(float(opening), 4)})
                return None

        w_free = np.hstack(self.rb.wrench())
        R_start, p_start = self.rb.eef_pose()
        way_in = 0.0
        moving = allowed + tuple(m.oid for m in self._contents_of(part))
        guard_cb = self._lrl_guard(moving)

        act_fn = self._ucs_act_fn()
        far_p = p_start.copy()
        s_sign = -1.0 if is_push else 1.0
        runs_log = []
        det_read = {"label": "progress_short"}

        from .ucs import LeakyPLAICompliantController
        try:
            with self._lrl_free(own):
                for run_i in range(UCS_HARNESS_CFG["max_in_place_runs"]):
                    p_cur = self.rb.eef_pose()[1]
                    done_so_far = max(0.0, float((p_cur - p_start) @ (s_sign * n)))
                    rem_span = max(0.04, span - done_so_far)
                    goal = ExecGoal(
                        mode=ucs_mode,
                        believed_tangent=s_sign * n,
                        believed_axis=s_sign * n,
                        believed_anchor=cand_p_actual,
                        target_span=rem_span,
                        approach_axis=cand_a_actual,
                        way_in_m=way_in if run_i == 0 else 0.0,
                    )
                    skill = UCSExecSkill(self.rm.limits[:7])
                    obs = skill.start(self.rb, goal, tare_wrench=w_free)
                    while skill.stop is None:
                        obs = skill.step(act_fn(obs))
                        p_now = self.rb.eef_pose()[1]
                        if float((p_now - p_start) @ (s_sign * n)) > float((far_p - p_start) @ (s_sign * n)):
                            far_p = p_now.copy()
                        if skill.k % 4 == 0:
                            self.swept.append(self.rb.q()[:7])
                        g_hit = guard_cb(self.rb.q())
                        if g_hit is not None:
                            skill.stop = "guard"
                            break

                    det_read = self._ucs_detect(skill.trace_rows, ucs_mode, rem_span)
                    step_disp = float(np.linalg.norm(self.rb.eef_pose()[1] - p_cur))
                    runs_log.append({
                        "run": run_i,
                        "stop": skill.stop,
                        "steps": skill.k,
                        "disp_m": round(step_disp, 4),
                        "detector": det_read["label"],
                        "conf": det_read.get("confidence"),
                    })
                    self.log("ucs_skill_run", stage=result["stage"], candidate=cand.name, mode=ucs_mode, **runs_log[-1])
                    if skill.stop in ("endstop", "reached_span", "guard", "force") or step_disp < 0.008:
                        break
                    if det_read["label"] != "progress_short":
                        break
        finally:
            LeakyPLAICompliantController.restore_nominal_gains(self.rb)

        # Clean release & guarded retreat
        Rs, ps = self.rb.eef_pose()
        if not is_push:
            self.ex.gripper_to(2.0 * (cand.finger_open + manip.PAD_OFFSET))
        self.retreat(
            Rs,
            ps - cand_a_actual * UCS_HARNESS_CFG["backoff_m"],
            grip=1.0 if is_push else 0.0,
            speed=UCS_HARNESS_CFG["backoff_speed"],
            finger=0.0 if is_push else cand.finger_open,
            tag="ucs_backoff",
        )
        if is_push:
            self.ex.gripper(-1.0, max_steps=10)

        disp = far_p - p_start
        travel = float(np.linalg.norm(disp))
        stop_word = "force_limit" if skill.stop == "force" else ("stalled" if skill.stop == "endstop" else str(skill.stop))
        how = "press" if is_push else "grip"
        result["attempts"].append({
            "status": "press_push" if is_push else "pull",
            "mode": how,
            "pose": cand.name,
            "travel_m": round(travel, 3),
            "stop": stop_word,
            "detector": det_read["label"],
            "runs": runs_log,
        })
        return {
            "mode": how,
            "travel": travel,
            "stop": stop_word,
            "d": s_sign * n,
            "disp": disp,
            "point": cand_p_actual,
            "family": cand.name,
            "normal": n,
            "label": "reached" if det_read["label"] == "reached_goal" else "short",
        }

    # -------------------------------------------------------------------------------- rotary actuation (knobs / valves)
    def ucs_turn(self, st: dict, part, parent, direction, result: dict):
        """Unified rotary actuation (`knob_turn`) across dominant surface-normal hypotheses."""
        mf0 = self._mount_frame(part, parent)
        q_seen = self.rb.q()[:7].copy()
        if self._look_at_face(part, parent, mf0) is not None:
            mf0 = self._mount_frame(part, parent)
        q = self.rb.q()[:7]
        if np.minimum(q - self.cm.lo, self.cm.hi - q).min() < UCS_HARNESS_CFG["limit_slack_rad"]:
            self.log("ucs_after_look", moved=self.go_configs([q_seen], None, finger=self._finger(), tag="ucs_after_look"))

        axes = self._lrl_axes(part, parent, mf0)
        # Physical prior for tabletop rotary controls: small knobs (<7cm) prefer upward/outward axes over side slivers
        part_span = float(np.ptp(part.points, axis=0).max()) if len(part.points) else 0.1
        if part_span < 0.085:
            axes.sort(key=lambda ax: -(1.0 if ax["n"][2] > 0.7 else 0.0))

        self.checker.free_near = [(part.points, 0.015, True)] + ([(parent.points, 0.010, False)] if parent is not None else [])
        fix_oids, fix_pts = self._fixture(part, part)
        allowed = (part.oid,) + fix_oids
        own = [(fix_pts, UCS_HARNESS_CFG["fixture_skin_m"], True)] if len(fix_pts) else []

        has_effect = bool(st.get("effect"))
        ref0, measure = None, None
        if has_effect:
            ref0 = self.capture(HEAD_CAMERA)
            self.save_img("effect_ref", ref0.rgb)
            region = self._effect_region(ref0, part, parent, st)
            has_px = st.get("effect_x") is not None and st.get("effect_y") is not None

            def measure(cur):
                n_ch, n_vis = self._effect_change(ref0, cur, region)
                seen = n_vis >= (max(self.EFFECT_MIN_VISIBLE, 0.5 * int(region.sum())) if has_px else self.EFFECT_MIN_VISIBLE)
                need = max(40, self.EFFECT_FRAC * n_vis) if has_px else 250
                return {"changed_px": n_ch, "visible_px": n_vis, "seen": bool(seen), "hit": bool(seen and n_ch >= need)}

        mem = self._ucs_stage_mem(result)
        failed = []
        act_fn = self._ucs_act_fn()

        for ax in axes:
            if mem["entered"] >= UCS_HARNESS_CFG["max_candidates"]:
                break
            n, ap = _unit(np.asarray(ax["n"], float)), np.asarray(ax["axis_point"], float)
            cands = self._ucs_sample_part_candidates(
                "knob_turn", part, parent, ax, UCS_HARNESS_CFG["turn_span_rad"], allowed, own, failed
            )
            if not cands:
                result["attempts"].append({"status": "no_ucs_grip", "axis": np.round(n, 3).tolist()})
                continue

            cand = cands[0]
            mem["entered"] += 1
            width = 2.0 * (cand.finger_open + manip.PAD_OFFSET)
            self.ex.gripper_to(width)
            q_touch = cand.q_chain[min(3, len(cand.q_chain) - 1)]
            alts_k = [
                {
                    "R": c_alt.R_contact,
                    "p": c_alt.p_contact,
                    "q": c_alt.q_chain[min(3, len(c_alt.q_chain) - 1)],
                    "approach": c_alt.approach,
                    "window": None,
                }
                for c_alt in cands[1:]
            ]
            with self._lrl_free(own):
                end = self.reach(
                    cand.R_contact, cand.p_contact, q_touch, cand.approach,
                    part.oid, None, "ucs_to_knob", 0.0, cand.finger_open, cand.q_handoff,
                    alts=alts_k, standoff=(0.0, 0.12),
                )
            if end is None:
                failed.append(cand.p_contact)
                result["attempts"].append({"status": "transit_failed", "why": (self.last_move or {}).get("why")})
                continue

            cand_R_actual = np.asarray(end.R, float)
            cand_p_actual = np.asarray(end.p, float)
            cand_a_actual = cand_R_actual[:, 2]
            w_free = np.hstack(self.rb.wrench())

            def _early_k(robot, F, F0, k):
                return "early_contact" if np.linalg.norm(F - F0) > 12.0 and k > 2 else None
            self.move_line(cand_R_actual, cand_p_actual, grip=0.0, speed=0.05,
                           guard=self._guard("early_contact", _early_k), finger=cand.finger_open)
            opening_k = self.ex.gripper(1.0, max_steps=22)
            if opening_k < 0.002:
                self.ex.gripper_to(width)
                Rs, ps = self.rb.eef_pose()
                self.retreat(Rs, ps - cand_a_actual * 0.06, grip=0.0, speed=0.12, finger=cand.finger_open, tag="ucs_knob_miss")
                failed.append(cand_p_actual)
                result["attempts"].append({"status": "missed_part", "opening": round(float(opening_k), 4)})
                continue
            way_in = 0.0
            senses = [1, -1] if direction != "clockwise" else [-1, 1]
            if direction == "counterclockwise":
                senses = [1, -1]

            done_turn = None
            from .ucs import LeakyPLAICompliantController
            try:
                for s_idx, sgn in enumerate(senses):
                    goal = ExecGoal(
                        mode="knob_turn",
                        believed_tangent=np.cross(sgn * n, cand_p_actual - ap + 1e-4),
                        believed_axis=sgn * n,
                        believed_anchor=ap,
                        target_span=UCS_HARNESS_CFG["turn_span_rad"],
                        approach_axis=cand_a_actual,
                        way_in_m=way_in if s_idx == 0 else 0.0,
                    )
                    skill = UCSExecSkill(self.rm.limits[:7])
                    obs = skill.start(self.rb, goal, tare_wrench=w_free, max_steps=65)
                    while skill.stop is None:
                        obs = skill.step(act_fn(obs))

                    turned_deg = round(float(np.rad2deg(skill.rot_accum)), 1)
                    det_read = self._ucs_detect(skill.trace_rows, "knob_turn", UCS_HARNESS_CFG["turn_span_rad"])
                    att = {
                        "status": "turn",
                        "sign": sgn,
                        "achieved_deg": turned_deg,
                        "stop": skill.stop,
                        "detector": det_read["label"],
                    }
                    result["attempts"].append(att)
                    self.log("ucs_turn_run", stage=result["stage"], **att)

                    moved = {"rotation_deg": turned_deg, "sense": "counterclockwise" if sgn > 0 else "clockwise"}
                    if measure is None:
                        if abs(np.deg2rad(turned_deg)) >= articulate.MIN_FOLLOW:
                            done_turn = (moved, None)
                            break
                    else:
                        LeakyPLAICompliantController.restore_nominal_gains(self.rb)
                        self._hold(4)
                        cur = self.capture(HEAD_CAMERA)
                        m_eff = measure(cur)
                        ok_eff, why_eff = (
                            (None, "no change in watched area") if not m_eff["hit"]
                            else (None, "no model budget") if not self._effect_budget_ok()
                            else self._confirm_effect(st, ref0, cur, np.deg2rad(turned_deg))
                        )
                        att["effect"] = {**m_eff, "confirmed_by_s1": ok_eff}
                        if ok_eff:
                            self.save_img("effect_seen", cur.rgb)
                            done_turn = (moved, {"expected": st.get("effect"), "observed": True, "s1_evidence": why_eff, **m_eff})
                            break
                        # Undo unsuccessful direction before trying opposite sense
                        if abs(np.deg2rad(turned_deg)) > np.deg2rad(5.0):
                            articulate.twist_about(self, ap, n, -np.deg2rad(turned_deg), settle=6)
            finally:
                LeakyPLAICompliantController.restore_nominal_gains(self.rb)

            failed.append(cand_p_actual)
            self.ex.gripper_to(width)
            Rs, ps = self.rb.eef_pose()
            self.retreat(Rs, ps - Rs[:, 2] * UCS_HARNESS_CFG["backoff_m"], grip=0.0, speed=UCS_HARNESS_CFG["backoff_speed"],
                         finger=cand.finger_open, tag="ucs_release_back")
            head = self.capture(HEAD_CAMERA)
            self.frames.append(head)
            self.rebuild_map(self.frames[-8:])

            if done_turn is not None:
                moved, eff = done_turn
                result["measured"] = moved
                if eff is None:
                    result["relation_check"] = {"satisfied": True, **moved}
                else:
                    after = measure(head)
                    eff["after_release"] = after
                    result["effect"] = eff
                    gone = after["seen"] and not after["hit"]
                    result["relation_check"] = {
                        "satisfied": not gone,
                        **({"unverified": True} if gone else {}),
                        "effect_observed": True if not gone else None,
                        "confirmed_by_s1": True,
                        **moved,
                    }
                self.log("actuated", result=result)
                return result

        from .harness import StageFailure
        if has_effect:
            raise StageFailure("no_visible_effect", expected_effect=st.get("effect"), attempts=result["attempts"][-4:])
        raise StageFailure("actuation_failed", attempts=result["attempts"][-4:])
