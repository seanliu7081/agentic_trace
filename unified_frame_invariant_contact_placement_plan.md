# Unified Frame-Invariant Contact & Placement Plan (`agentic_trace`)

**Date:** 2026-11-23  
**Scope:** `src/atrace` (harness, perception, grasp/place, guarded execution) and `src/traceplan` (TRACE numerical planner)  
**Goal:** Eliminate per-contact-type code branches (e.g., separate top-down placement, side-entry insertion, and ad-hoc articulation routines) by replacing hardcoded world-$z$ assumptions with four $SE(3)$ frame-invariant abstractions.

---

## 1. Architectural Motivation

The current harness fails on side-opening containers (such as `t09`'s microwave) not because side entry requires bespoke robotics primitives, but because four general geometric and control operations are currently hardcoded to the vertical world axis:

1. **Perception (`2.5D` solid-fill):** `ObjectModel` fills occluded regions beneath top surfaces as solid obstacles, turning interior cavities into solid boxes and forcing separate top-view (`container_region`) vs. side-view (`side_opening`) detectors.
2. **Contact execution (`-z` / $F_z$ only):** Placement descends strictly along $-e_z$, monitors only $F_z$, opens the gripper fully, and lifts along $+e_z$.
3. **Decoupled grasping:** Grasp generation biases toward top-down poses ($\le 35^\circ$ tilt) without evaluating whether the resulting grasp is feasible at the destination.
4. **Discrete approach enumeration:** Workarounds for tight entries enumerate fixed yaw/pitch angles in the harness instead of delegating continuous pose/corridor refinement to TRACE.

Adding a standalone `insert_object` path would fix one failure mode while risking relapse into a brittle rule stack. Instead, lifting these four pieces to frame-invariant $SE(3)$ operations unifies tabletop placement, container insertion, shelf/cubby placement, and articulated part actuation under a single pipeline—while keeping TRACE strictly responsible for contact-free motion up to a computed standoff.

---

## 2. Core Design: Four Unified Abstractions

### A. True 3D Volumetric Representation & Unified Corridor Discovery

**Replace 2.5D height-map solid-filling with ray-carved 3D occupancy (`free` / `occupied` / `unknown`) and general approach-corridor search.**

* **Volumetric object & scene model (`src/atrace/perception.py`, `src/atrace/snapshot.py`):**
  * Remove the downward solid-fill assumption beneath observed surfaces. Maintain object surface point clouds and 3D voxel states directly from multi-view depth ray-carving, marking unviewed interiors as `unknown` rather than `occupied`.
  * Preserve `SceneSnapshot`'s existing three-state grid (`occupied`, `free`, `unknown`) and per-entity covariance $\Sigma$ so TRACE and the execution guard read identical 3D geometry.
* **Corridor discovery replaces opening classifiers (`src/atrace/manip.py`):**
  * Retire separate top-opening vs. side-opening heuristics and fix the aim-snap defect (never snap a target pixel to a distant surface or a rim lacking interior support).
  * Given a grounded 3D target region $p_{\text{target}}$ (and optional directional hint from System 1), evaluate candidate approach unit vectors $\hat{a} \in S^2$ (sampled over the hemisphere/sphere around $p_{\text{target}}$, weighted by kinematic manipulability and gravity preference) by sweeping the payload bounding volume along the exit ray $p(s) = p_{\text{target}} - s\hat{a}$:
    * **Clear of `occupied` and mostly `free`:** valid approach corridor $\hat{a}$ immediately available.
    * **Clear of `occupied` but intersects `unknown`:** automatically trigger a generic `look_toward(target=p_target, view_dir=-a_hat)` wrist observation to carve the cavity free before re-evaluating.
  * **Emergent behavior:** Open plates/baskets naturally yield $\hat{a} \approx -e_z$; microwaves, shelves, and cubbies block overhead rays via their occupied roofs/walls and naturally yield horizontal/angled corridors pointing out of the cavity mouth without any container-specific code.

### B. Generalized $SE(3)$ Contact Primitive (`GuardedScrew`) & SDF-Adaptive Release

**Collapse vertical placement, horizontal insertion, and articulated actuation into one parameterized local contact primitive.**

Define a single low-level executor primitive in `src/atrace/motion.py` backed by `src/atrace/guard.py`:

```python
GuardedScrew(
    axis: ScrewAxis,          # unit twist xi = (v, omega) in se(3), e.g., pure translation or joint rotation
    max_travel: float,        # distance (m) or angle (rad)
    speed: float,             # conservative contact speed (e.g., 2.5-5 cm/s)
    stop_on: StopCondition,   # union of {target_travel, contact_wrench, prediction_stall}
    compliance: ComplianceCfg # optional orthogonal null-space / soft-axis compliance for articulated parts
) -> ContactOutcome
```

* **Omnidirectional contact stop (`src/atrace/guard.py`):**
  * Replace $F_z$-only termination checks with a frame-invariant stop predicate combining:
    1. One-step controller prediction residual (already computed by `ContactDetector`), and
    2. Wrench projected onto the motion twist: $\mathbf{w}^\top \xi > w_{\text{th}}$ (calibrated per payload mass/inertia to prevent false stops during horizontal carries).
* **SDF-adaptive gripper release (`src/atrace/manip.py`, `src/atrace/body.py`):**
  * Instead of hardcoding "open fully" (top-down) vs. "open partway" (side insertion), compute the release opening `gripper_to` directly from the 3D SDF around the finger spheres: select the largest opening above `object_width + release_margin` that keeps all finger collision spheres clear of the scene SDF.
* **Unified stage composition (`src/atrace/harness.py`, `src/atrace/articulate.py`):**
  * **Top-down place:** `TRACE(to standoff)` $\to$ `GuardedScrew(dir=-e_z, stop=contact)` $\to$ `SDF-Adaptive Release` $\to$ `GuardedScrew(dir=+e_z, dist=retreat)`.
  * **Side / angled insertion:** `TRACE(to standoff)` $\to$ `GuardedScrew(dir=a_in, stop=dist|contact)` $\to$ short `GuardedScrew(dir=-e_z, max_travel=1.5 cm, stop=contact)` $\to$ `SDF-Adaptive Release` $\to$ `GuardedScrew(dir=-a_in, dist=exit)`.
  * **Articulated actuation (doors, drawers, knobs):** `TRACE(to standoff)` $\to$ `GuardedScrew` along the estimated joint screw $\xi_{\text{joint}}$ with orthogonal compliance.

### C. Cross-Stage Affordance Intersection ($\mathcal{G}_{\text{pick}} \cap \mathcal{G}_{\text{place}}$)

**Replace task-dependent grasp branches with joint pick-and-place feasibility filtering.**

* **Isotropic grasp candidate generation (`src/atrace/manip.py`):**
  * Sample antipodal grasp poses $g = T^{\text{obj}}_{\text{hand}} \in SE(3)$ across the object's observed 3D geometry (handles, rims, opposing faces) without enforcing a hard top-down $35^\circ$ ceiling. Keep a mild preference score for top-down grasps only as a tie-breaker when multiple candidates are equally feasible.
* **Joint feasibility intersection (`src/atrace/harness.py`):**
  * Pass the planned destination pose and discovered approach corridor(s) into `plan_grasp`.
  * Filter every candidate relative grasp $g$ through the same kinematics and whole-body collision checks at both ends:
    $$g \in \mathcal{G}_{\text{pick}}(T_{\text{start}}) \;\cap\; \mathcal{G}_{\text{place}}(T_{\text{dest}}, \hat{a})$$
    * $\mathcal{G}_{\text{pick}}$: collision-free approach/grasp and valid arm IK at the object's current pose.
    * $\mathcal{G}_{\text{place}}$: at the destination and along the insertion corridor, the palm, fingers, arm links, and carried payload clear the 3D scene SDF with valid arm IK (automatically rejecting top-grasps whose palm collides with a microwave roof, and favoring pitched handle grasps that keep the palm outside the cavity).
* **Universal regrasp trigger:**
  * If $\mathcal{G}_{\text{pick}} \neq \emptyset$ and $\mathcal{G}_{\text{place}} \neq \emptyset$ but $\mathcal{G}_{\text{pick}} \cap \mathcal{G}_{\text{place}} = \emptyset$, automatically invoke a generic **staging regrasp** (place on a nearby collision-free surface in an orientation exposing a grasp in $\mathcal{G}_{\text{place}}$, then regrasp). No microwave-specific logic is required.

### D. Delegating Continuous Approach & Posture Optimization to TRACE

**Stop enumerating fixed yaw/pitch grids in the harness; formulate approach flexibility inside TRACE's residual program.**

* **Continuous terminal & corridor solving (`src/atrace/tracemotion.py`, `src/traceplan/service.py`, `src/traceplan/primitives.py`):**
  * Keep TRACE strictly bounded to contact-free segments ending at the computed standoff $p_c - d\hat{a}$ (where $d$ accounts for axial perception uncertainty $\sigma_a$, tracking tube, and finger guard).
  * When a nominal straight approach fails screening or certification, relax the terminal specification inside `TraceRequest` rather than running outer discrete angle loops:
    * Specify the target standoff region via `reach_point` / `distance_band`,
    * Constrain the entry direction via `approach_cone(axis=a_hat, half_angle=cone_rad)`,
    * Keep payload orientation via `tilt_limit` (mapped from `src/atrace/levels.py`),
    * Enforce full palm/payload/scene `clearance` and `reach_limit`.
  * TRACE's coupled annealing (MPPI exploration + L-BFGS gradient refinement with augmented Lagrangian multipliers) continuously finds angled entries and pitched wrist postures that thread narrow passages.
* **System 1 semantic contract (`src/atrace/program.py`):**
  * Add an optional high-level `approach_hint` (e.g., `above`, `from_front`, `from_left`, `from_right`) to stage placement schemas. The harness translates the hint into a world vector via camera calibration and cross-checks it against the 3D corridor geometry from Section 2A; geometry resolves conflicts $> 45^\circ$ and logs both.
* **Generic obstacle clearing (articulated parts):**
  * If TRACE returns `conflict` or `no_path` and diagnosis attributes the primary collision volume to an articulated entity (e.g., a half-open door blocking the corridor), surface that entity ID in the structured failure diagnosis so System 2 / REPAIR can schedule a generic actuation step (push/pull along its joint axis to its limit) before retrying.

---

## 3. Physical Boundary: Interface for Contact-Rich Skills

This geometric architecture unifies quasi-static rigid-body manipulation (pick, place, insert, shelf, drawer, door, knob). It intentionally does **not** attempt to model contact dynamics such as pouring, wiping, compliant peg-in-hole jamming recovery, or in-hand slip control.

To keep the architecture extensible without polluting the geometric harness:
* Expose a thin `LocalContactSkill` protocol alongside `GuardedScrew` in `src/atrace/motion.py`: `run(scene_snapshot, target_spec, guard) -> ContactOutcome`.
* TRACE delivers the end-effector to the standoff; `GuardedScrew` handles rigid quasi-static contact; future force/visual-servoing or learned closed-loop skills plug into the exact same standoff handoff slot for dynamic contact phases.

---

## 4. File-Level Change Map

| Component | Target Files | Key Changes |
| :--- | :--- | :--- |
| **3D Volumetric Scene** | `src/atrace/perception.py`, `src/atrace/snapshot.py` | Remove 2.5D solid-fill beneath surfaces; preserve ray-carved 3D `free`/`occupied`/`unknown` states in `ObjectModel` and `SceneSnapshot`. |
| **Corridor & Grasp Gen** | `src/atrace/manip.py` | Add ray-swept corridor discovery; fix aim-snap locality/support check; generalize grasp sampling beyond $35^\circ$ top-down; add SDF-adaptive release width computation. |
| **Unified Guarded Motion** | `src/atrace/guard.py`, `src/atrace/motion.py`, `src/atrace/articulate.py` | Implement omnidirectional `GuardedScrew` primitive with twist-projected wrench + prediction-error stop; route placement, insertion, and articulation strokes through it. |
| **Cross-Stage Orchestration** | `src/atrace/harness.py`, `src/atrace/program.py`, `src/atrace/levels.py` | Implement $\mathcal{G}_{\text{pick}} \cap \mathcal{G}_{\text{place}}$ filtering and staging regrasp trigger; add `approach_hint` schema field; replace hardcoded `_v2_put` descent with corridor-parameterized `GuardedScrew`. |
| **TRACE Continuous Entry** | `src/atrace/tracemotion.py`, `src/traceplan/service.py`, `src/traceplan/diagnose.py` | Pass `approach_cone` + relaxed terminal standoff specs to TRACE on tight entries; expose blocking entity attribution in `no_path`/`conflict` diagnostics. |
| **Tests & Replay** | `tests/test_atrace_offline.py`, `tests/test_placement_soft.py`, `tests/test_traceplan.py`, `scripts/trace_replay.py`, `scripts/trace_bench.py` | Add synthetic cavity/corridor scenes, omnidirectional guard unit tests, and pick-place intersection tests. |

---

## 5. Phased Milestones & Verification Gates

Every phase ends at a hard verification gate to prevent regressions on `t00`–`t08` and catch kinematic infeasibility early.

| Phase | Deliverables | Verification Gate |
| :--- | :--- | :--- |
| **M0: Feasibility Probe** | Evaluator-side kinematic & corridor check across `t09` (all 10 episodes) plus 2–3 held-out LIBERO-90 side-opening/articulated scenes (cabinet, oven, drawer). Test continuous pitched/angled entries and door-limit clearance using true geometry. | Confirm that collision-free solutions exist in the combined pick-place-corridor space (including staging regrasp / door actuation where needed). Quantify how much of `t09` is robot-kinematic-limited vs. harness-limited. |
| **M1: 3D Map & Aim-Snap Cleanup** | Remove 2.5D downward solid-fill in `perception.py`; enforce local support-backed aim snapping; add unit tests on recorded `t09` `ep000`/`ep007` frames and synthetic cavity fixtures. | Zero regression on offline replay (`scripts/trace_replay.py`) and spot-checked `t00`, `t05`, `t07` episodes; recorded `t09` aims no longer snap to microwave roof or open door. |
| **M2: `GuardedScrew` & SDF Release** | Implement omnidirectional `GuardedScrew` and SDF-adaptive release; calibrate `ContactDetector` on horizontal and angled payload carries; migrate `_v2_put` and `articulate.py` onto `GuardedScrew`. | All existing top-down tests (`tests/test_atrace_offline.py`, `tests/test_placement_soft.py`) pass using `GuardedScrew(dir=-e_z)`; false-stop rate on horizontal payload moves measured and bounded. |
| **M3: Corridor Discovery & $\mathcal{G}_{\text{pick}} \cap \mathcal{G}_{\text{place}}$** | Implement ray-swept corridor search + automatic `look_toward` carving; enable isotropic grasp sampling and joint pick-place intersection filtering; wire staging regrasp on empty intersection. | Unit tests confirm top-down grasps remain selected on open plates/baskets, while pitched handle grasps (palm outside cavity) are automatically selected on side-opening cavities without any container-type `if/else`. |
| **M4: TRACE Continuous Corridor & Obstacle Attribution** | Wire `approach_cone` terminal relaxation in `tracemotion.py`/`service.py`; add blocking-entity attribution to `diagnose.py` to trigger generic articulated clearing; add `approach_hint` to System 1 schema. | Synthetic benchmark (`scripts/trace_bench.py`) adds a narrow side-cavity scene (S7) achieving $\ge 90\%$ certified success; end-to-end execution succeeds on feasible M0 scenes. |
| **M5: Full Evaluation & Held-Out Generalization** | Run full `t00`–`t09` batch (v2, high effort, 600 s / 50 calls) plus held-out LIBERO-90 side-opening and articulated tasks never tuned during M1–M4. | No regression on `t00`–`t08`; significant lift on `t09` up to M0's physical ceiling; held-out side-opening tasks succeed without task-specific prompt or code changes. |

---

## 6. Risks, Guardrails & Open Decisions

### Technical Risks & Mitigations
* **Phantom free space after removing 2.5D solid-fill:** Removing downward solid-fill means occluded regions under overhanging objects become `unknown` instead of `occupied`. *Mitigation:* Keep TRACE's existing rule that traversing `unknown` space is a `require` term (triggering a wrist look or relaxing only to slow guarded motion), never treated as guaranteed free space.
* **Panda kinematic limits on horizontal poses:** As prior probes showed, horizontal grasps/insertions push the Panda arm near joint limits. *Mitigation:* Prioritize pitched handle grasps in $\mathcal{G}_{\text{pick}} \cap \mathcal{G}_{\text{place}}$ (which keep the wrist in a high-manipulability posture while holding vertical handles upright) and allow TRACE to optimize continuous entry angles.
* **Omnidirectional `ContactDetector` false positives:** Carrying a payload horizontally introduces different inertial/controller lag than vertical descent. *Mitigation:* Calibrate twist-projected wrench and prediction-error thresholds explicitly during M2 before enabling M5 batch sweeps.
* **Relapse into task-specific rules (Guardrail):** Enforce the existing no-task-token lint across all new corridor, grasp-intersection, and `GuardedScrew` code; forbid object nouns (`microwave`, `mug`, `drawer`) and hard-coded task dimensions anywhere in `src/atrace/` and `src/traceplan/`.

### Decisions Required Before Starting
1. **M0 Gate Scope:** Run M0 across `t09` plus 2–3 held-out LIBERO-90 side-opening tasks (e.g., cabinet shelf / oven) so M1–M4 are developed on unseen geometry rather than overfitting to `t09`?
2. **Staging Regrasp Budget:** Approve automatic staging regrasp when $\mathcal{G}_{\text{pick}} \cap \mathcal{G}_{\text{place}} = \emptyset$ (adds ~30–60 s wall time when triggered, zero extra LLM calls if handled locally in System 2)?
3. **Articulated Clearing Policy:** When TRACE attributes a blocked corridor to an articulated fixture (e.g., a $90^\circ$-open door), should System 2 autonomously attempt a guarded push toward its joint limit, or always escalate to System 1 `REPAIR` first?
4. **Migration Scope:** Migrate `v2`, `v1`, and `articulate.py` onto `GuardedScrew` together in M2 (cleanest long-term), or migrate `v2` first and deprecate `v1` afterward?