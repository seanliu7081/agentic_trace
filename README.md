# agentic_trace: the integrated harness (CP-5 + TRACE)

This repository implements the combined loop of `TRACE integration.md`: TRACE as a numeric trajectory service
behind the harness's middle layer. The build follows its three rules:

- The harness owns the scene and every model call.
- Each TRACE result is bound to one scene snapshot.
- TRACE stops at a standoff short of contact; the harness owns the last centimetre, the gripper event and the
  check after it.

TRACE follows `TRACE_method.md`, as changed by the integration doc. The harness side is a fork of CP-5 build
`13686994` (`/workspace/astra_harness_v5_cp5`, see PROVENANCE.md). CP-5 already has the improvement plan's
measured scene state (RGB-D voxel map with unknown space, SAM 3.1 entities, robot model, whole-arm checks) and
its single metric-free plan call. `DESIGN.md` is the implementation spec. It maps every rule of the integration
doc to code, and the table below summarizes that mapping.

## Status (2026-11-22)

| Check | Result |
|---|---|
| Unit tests (`tests/`) | 54 passed (traceplan 20, solvers and retime 24, harness offline 10), each suite run once |
| Bench: synthetic S1–S6, 3 seeds (`runs/bench_20261122_115619`) | 10 of 12 explore/exploit pairs certify 18/18; the default MPPI + L-BFGS has a median of 0.71 s. The bench saturates and cannot rank pairs, so the method note's default was kept. Program tests: 20 of 20 injected errors caught for each of the 5 types, 0 false alarms in 20 correct programs |
| TRACE alone on cuda, budget 2 s | S1: fast path in 11–16 ms. S2–S5: certified by search in 0.65–0.97 s. S6: conflict with quote "upright: tilt max 5° → needs 40°" in about 1.0 s |
| Offline dev episode, no model calls (`runs/dev/c_run3`: LIBERO-object, ketchup + BBQ sauce into the basket) | See below |
| Real end-to-end run with Astra (Libero10-Collision t00 ep000) | 4 attempts; the last two succeeded. See below |

Offline dev episode (`runs/dev/c_run3`):

- Success at step 685, 77.8 s policy wall time.
- 26 TRACE requests: 14 certified (all executed to the end), 12 no_path.
- 0 guard stops and 0 tube exits on certified paths. Tracking used at most 0.31 of the tube.
- Fast path: p50 0.51 s, p95 0.69 s. Search: p50 1.68 s.
- No robot contact over 1 mm.
- Evaluator-side re-check of all 14 certified paths on the true scene: 0 violations.

Real end-to-end run with Astra, Libero10-Collision t00 ep000, which is a seen episode. Astra ran at medium effort,
on GPU 0. Each attempt after the first includes the fixes the previous one showed:

| Run | Result | Steps | Policy wall time | Model calls | TRACE requests | Guard stops / tube exits |
|---|---|---|---|---|---|---|
| CP-5 7bc1aebd (baseline) | success | 1118 | 186 s (67 s of RRT planning) | 1 | – | – |
| `e2e_t00_ep000_trace` | failed: grasps never certified | 675 | 250 s | 3 | 46 (25 certified) | 1 / 10 |
| `e2e_t00_ep000_trace2` | failed: grasp ends rejected | 392 | 175 s | 3 | 26 (14 certified) | 0 / 0 |
| `e2e_t00_ep000_trace3` | success | 965 | 257 s | 3 | 37 (22 certified) | 0 / 0 |
| `e2e_t00_ep000_trace4` | **success** | **815** | **175.5 s** | 2 | 29 (19 certified) | 0 / 0 |

Details for `trace4`:

- **Motion:** 27% fewer control steps than CP-5. The fast path answered 52% of all requests and 79% of the
  certified ones. Fast path p50 0.71 s, p95 0.80 s; search p50 1.93 s, p95 2.12 s.
- **Certificates:** 0 certified executions contained a forbidden contact onset. The 19 certified paths were
  re-checked on the true scene with 0 violations.
- **Contacts:** the benchmark monitor counts only can-on-can and can-on-basket contacts (at most 0.4 mm). Your
  collision rule does not count these. The added obstacle was never touched.
- **REPAIR:** one call was spent because the sauce's pre-place approaches failed the arm check (wrist limit, see
  Known gaps).

## Run

The Python environment is the CP-5 venv (symlinked as `.venv`). The harness uses GPU 7 by default when it is
free; use `--gpu 0` when GPU 7 is held.

```bash
# one Libero10-Collision episode (real Astra calls), TRACE motion
.venv/bin/python scripts/run_episode.py --task t00 --episode 0 --gpu 0 --motion trace [--true-scene]
#   --motion trace | v0 | cp5     cp5 = CP-5's own planner (A/B); v0 = straight line, else CP-5's planner path, certified
#   --trace-budget-ms 2000        per-request TRACE budget
#   --trace-device cuda:0
#   --compiles N                  N-1 extra low-effort PROGRAM calls for term confidence (off by default)
#   --true-scene                  evaluator-only: save the true scene at every request (certificate honesty)
#   --placement ladder | v1 | v2 | unified  how a held object is put down (see "Placement modes"); default ladder

# dev scene, no model calls (S1 program from the stub)
.venv/bin/python scripts/dev_episode.py --gpu 0 --motion trace --suite libero_object \
    --task pick_up_the_ketchup_and_place_it_in_the_basket --init 3 --s1 offline \
    --goal in:ketchup_1:basket_1_contain_region,in:bbq_sauce_1:basket_1_contain_region \
    --instruction "put both the ketchup and the bbq sauce in the basket"

# measures and replay of a run (no model, no simulator)
.venv/bin/python scripts/trace_measures.py RUN_DIR        # -> RUN_DIR/trace/measures.json
.venv/bin/python scripts/trace_replay.py RUN_DIR          # -> RUN_DIR/trace/replay.{json,txt}

# TRACE bench (stages 1-4 of TRACE_method.md section 8.2 on the synthetic scenes)
.venv/bin/python scripts/trace_bench.py --seeds 3

# tests
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/
```

Each run writes `trace/trace.jsonl` (one line per request and response), `trace/req_<id>.pkl` (the full request
and response), `trace/snapshots/v<version>.npz` and, with `--true-scene`, `trace/true/<id>.npz`. It also writes
`trace_*` and `guard_defect` events in `events.jsonl` and a `trace` block in `summary.json`.

## Placement modes (`--placement`, 2026-11-22)

All three share the grasp, the transit (TRACE) and the release; they differ in how the place is chosen.

- **`ladder`** (default; the behaviour of build 6f795ffc). The careful planner (`Harness._plan_place`) with the
  margin ladder, steps 0-2. It returns nothing (`no_feasible_placement`) when no candidate passes every test.
- **`v1`: rank, don't reject.** The same planner plus a step 3 that cannot fail on the destination's side:
  - Every cell of the region is a candidate (`manip.soft_candidates`; `rest_candidates(soft=True)` for stacking).
    The share of the footprint over the opening, the distance to the destination's walls and the stability of the
    rest order the candidates and rule none out.
  - Hand and payload need no margin at step 3, only no predicted contact, and only against what the scoring rule
    counts: other objects, the table, surface that is not part of the destination. The destination, what is in it
    (payload only) and the fixture it belongs to do not rule a candidate out. Candidates are ordered by how little
    of them they are predicted to touch (`_soft_touch`).
  - The fixture (`Harness._fixture`) is the objects System 1 tied to the destination with `part_of`, plus the
    unlabeled surface connected to them above the table.
  - The arm keeps its normal margin against everything except the destination, at every step.
  - A step-3 candidate is aimed at the destination's floor (the lowest release height that touches nothing that
    counts); the 4 N guard ends the descent on whatever the object meets first.
  - TRACE certifies a contact approach only up to a standoff where nothing touches, so it turns down a candidate
    whose payload is predicted to touch the destination. The hand then goes to the pose above the destination
    (`to_above_place`, contact-free) and the guarded descent covers the whole way down.
  - On the last attempt the look before an "in" placement (`landing_check`) no longer rules a spot out.
  - Still a refusal: no reachable arm posture, or every candidate touches something that counts.
- **`v2`: r1's mechanism, no placement planner** (`Harness.place_object_v2`). System 1 points at a pixel of the
  head image where the held object must come down and says how its long side must lie (`place_pose` call). System 2
  carries it above that point with a collision-checked transit, lowers it straight down until contact (a "drop"
  stage: lets go 3 cm above what lies under it) and opens the gripper. System 1 then judges fresh head and wrist
  images (`place_look` call); when it is not right, the object is picked up again and put where System 1 points
  now. At most 4 put-downs per stage call, then REPAIR. Nothing is refused on geometry.
  - The pixel is read on the horizontal plane at the destination's top near its ray. For a container whose
    opening the model shows, the pixel selects the opening (compartment) the ray passes through, and the aim is
    the spot of that opening nearest to the pointed one where the object has room (`Harness._v2_aim`).
  - When the object is not where it was let go, it is searched for again before the re-grasp; if the head camera
    does not find it, the stage fails over to REPAIR (`object_lost_after_release`).
  - It costs two model calls per placement and one more per retry, so run it with `--max-calls 16`.
  - Without an answer (call budget, model error) it aims at the program's place point or the middle of the
    destination, and accepts the release unjudged (VERIFY decides at the end).
  - The descent is vertical only, so a destination that must be entered from the side cannot be reached.
  - Parking a held object on the table still uses the careful planner.
- **`unified`: the contact stage of `CONTACT_STAGE_PLAN.md` (M1-M3, 2026-11-23)** (`contact.py`,
  `ContactMixin.place_object_unified`). System 1 points at the surface where the object's centre should rest
  (`place_pose`, unified wording). System 2 searches rest poses around that point from all observed geometry
  (`rest_poses`: on what is under the footprint, nothing standing inside it, walls around it for "in", no overhang
  on the way down, the open hand clear, an arm posture it can leave), hands TRACE the set as ends that slide back to
  where its end test passes, lowers the object with an expected contact window, and opens the gripper only when the
  object meets its support inside that window (a "drop": over the opening, within a 12 cm fall). An early contact is
  marked as geometry and the next rest pose is tried without a model call. After the release: the hand-empty test and
  re-localisation of the object's known shape (no new segmentation), then System 1 judges (`place_look`).
  M4 (direction, 2026-11-23):
  - the support is read at the pointed level (surfaces higher than the pointed surface + the object's height + 3 cm
    are not supports: a roof, a shelf or a handle above the spot); "in" is decided by the destination's walls (in 6 of
    8 directions the first surface rising above the object's bottom, at any distance up to 30 cm, is the
    destination's: its label, its parts, its fixture, unlabelled surface connected to it, or inside its footprint
    hull; a roof counts); "on" by the footprint share: at least 60 % of the object's footprint cells over the
    destination's own surface (inside its outline, standing out from the table, not another object's surface; its
    labels decide only without an outline or for a destination too flat to stand out); a footprint whose support
    varies by more than half the object's
    height (a wall top, a rim edge) is rejected before the 3D checks, which are spread over the hand's turns (12 when
    the long side does not matter);
  - the release opens as wide as the surroundings allow (as `release_width`), and a "place" may let go up to 3 cm
    above the rest when the hand is not clear lower;
  - the entry is searched (`entry_search`): straight down, a 45-degree line, or a slide (down in front, in at 2-3.5 cm
    above the rest, then the set-down; only through the spot's open sides, the hint and the arm's side), for hand
    and object together on observed geometry; the arm needs a posture
    along the whole entry with 6 mm clearance (`_entry_posture`, which also names what the arm would meet). The
    put-down runs the entry as guarded legs (no contact expected) and the last one with its contact window; after the
    release the hand leaves along the reverse of its approach (a side grasp) and back out along the entry;
  - System 1 may say from which side the object comes in (`entry`, unified place-pose answer); it is tried first;
  - grasps from any direction (`manip.side_grasp_candidates`: the cloud turned so that a horizontal or 45-degree
    direction is up, the top-down generator there, picks spread over the object) when no top-down grasp fits or has
    a posture (every placement mode);
  - look-ahead before the grasp of a transfer (`lookahead_grasp`): System 1's place-pose call is made before the
    grasp (no extra call); a spot not seen well enough is looked at first (level with it from its open side when
    something is above it); when no top-down grasp has a put-down there, a side grasp that has one is taken (the
    hand lowest first, never from below);
  - regrasp (user decision D3, `plan_regrasp`): when the grasp held has no way in, the object is set down at a staging
    spot (where it was taken from) turned so that a grasp with a put-down can be taken, and taken again;
  - making room (user decision D4, `make_room`): when the rest-pose search or TRACE names a part of a fixture (a
    door, a drawer) as what is in the way, the object is set down, the part is moved to its open end stop with the
    general actuation, and the object is taken again (once per stage).

Shared by every mode since the M1-M3 build (CONTACT_STAGE_PLAN.md):
- the hand-off is a set (contact pose, its 180-degree twin, alternative grasps or turns) whose ends TRACE slides back
  to where its end test passes (`search.end_test`, one rule for TRACE and the harness); refusals carry a typed reason
  (`last_move["why"]`: end blocked by what, no arm posture, which joint, where the path binds);
- unseen space is not a safety obstacle: a request blocked by it looks there first, then crosses it slowly
  (user decision D2); the guard and the arm check use observed geometry, unseen space only slows the arm;
- departures (lift, release lift, back-offs) are guarded lines until the hand clears the scene by the transit
  margins, TRACE when the line has no posture; no unguarded fallback;
- free space seen once is remembered across map rebuilds; the inferred fill under an object never overwrites seen
  free space; a released object's solid is inserted whole; a model never absorbs another tracked object;
- grasps: up to three distinct feasible grasps per attempt, postures away from joint limits with a feasible lift, the
  closed width must match the predicted one (-6..+10 mm) before lifting, and a failed contact region and approach are
  excluded on the next attempt;
- strokes (push / pull): continuation in contact (re-seat and push on, no new transit), a measured state that
  contradicts the hand's travel is not accepted, two strokes without progress end the face, a press replaces a grip
  that closed on nothing, what lies in a moving part moves with it, and a handle is operated as the part it moves with;
- System 1 may point at an object in the wrist image (`image: 2`) when the head camera cannot see it (user decision
  D6); failure reports carry `cause` and `levers`.

## Where each part of the integration doc lives

| Integration doc | Code |
|---|---|
| The contract: request, response with four results, `valid_while`, quote, look_at | `src/traceplan/contract.py`; entry point `traceplan.service.TraceService.plan(request, snapshot, arm_check, ik_check)` |
| Trajectory function (§3) and search (§6): fast path first, then seeds (straight, diverse A*, provided planner path), coupled continuation, MPPI then L-BFGS per level | `spline.py`, `se3.py`, `search.py`, `solvers.py` |
| Scene model built by the harness; TRACE reads one versioned snapshot and derives its own blurred fields | `atrace/snapshot.py` (export, per-entity covariance); `traceplan/scene.py` (signed fields, blurs and distance transforms on the GPU, trilinear lookup) |
| Shared primitive library | `traceplan/primitives.py` |
| Goal pose: entity, part and relation; ranked end candidates; computed standoff | `EndCandidate` (harness candidates, `standoff: "auto"`, lateral `window`); `search.score_candidates`; `free_end_candidates` (TRACE's own end-pose solve) |
| Safety terms written by the harness only; named levels; allowed relaxations | `atrace/levels.py` (level table, `compile_terms`, `safety_terms`, `can_relax`); the PROGRAM schema has per-stage `terms` (`atrace/program.py`) |
| Uncertainty fusion, and "not certifiable with this perception" leading to a look | `traceplan/program.py: fuse`; the auto standoff and lateral-window rule in `service.py` |
| Five compiles (off by default) | `--compiles`, `atrace/program.py: merge_compiles` |
| Arm check: the controller model rolled along the path | `atrace/body.py: ControllerModel.rollout`, `make_arm_check` (whole-arm clearance and joint limits on CP-5's exact geometry) |
| Repair: the sequencer applies relaxations the plan allowed; otherwise one model call carries the quote | `atrace/harness.py: trace_move` reactions (relax, look, next candidate, `StageFailure("trace_conflict")` to REPAIR, interface repair) |
| Program tests before any search | `traceplan/program.py: run_program_tests` (lint, range, theta-dependence, invariance, start feasibility) |
| Certificate (dense exact check, arm, retiming) and conflict diagnosis (leave-one-out quote) | `certify.py`, `diagnose.py` |
| Margin ladder: TRACE plans with guard + tube + perception; the guard stops at the guard distance alone | `atrace/levels.py` (ladder), `atrace/guard.py` (`StepGuard`) |
| Certificate validity: scene version, holding, start q; voided when the scene changes inside the tube | `atrace/tracemotion.py`; `traceplan/scene.changed_inside` |
| "Do not get closer" near both ends | the `zone` rule in the clearance primitives, the arm check and the guard |
| Contact detector firing within 1–2 steps; guarded moves end on contact | `atrace/guard.py: ContactDetector` (coupled one-step model plus directional force), used in the grasp, place and press descents |
| Retiming from certified clearance; tracking tube grows with speed | `traceplan/retime.py`; `Executor.run_timed` (spatial tube check, defects logged) |
| Seeded and logged, regression set at no model cost | service `log_dir`; `scripts/trace_replay.py` |
| Measures: fast-path share, solve-time p95, certificate honesty, guard stops per certified path | `scripts/trace_measures.py`; `scripts/true_scene.py` (evaluator-only true scene) |

## Defaults

None of these values was tuned on test episodes. All are in one config per module.

- **Margin ladder** (`atrace/levels.py`):
  - guard: arm 10, palm 8, finger 5, link 10, payload 5 mm (against the table plane: 15/8/4/15/3 mm);
  - perception bound: 5 mm;
  - tracking tube: 4 mm + 0.022 s × speed (15 mm at 0.5 m/s, as recorded) + 3 mm per rad/s of hand rotation.
  - TRACE therefore plans with guard + tube(0.05 m/s) + perception, for example 2.0 cm for the arm.
- **Speed limits:** 0.40 m/s free and 0.20 m/s holding; 0.5 and 0.25 m/s².
- **Named levels:**
  - `tilt_limit`: upright 10°, tilted 25°, loose 45°;
  - `clearance` from a named object: close 1 cm, normal 3 cm, wide 8 cm;
  - `approach`: vertical 10°, steep 25°.
  - Unknown space is a `require` term that can always be relaxed to slow guarded motion.
- **Auto standoff:** `1.64·σ_along + tube(v_min) + finger guard + 5 mm`, clipped to 2–10 cm. CP-5 always used 10 cm.

## Deviations from DESIGN.md

- **Execution.** robosuite's OSC couples position and orientation through the arm's mass matrix (a slow,
  lightly damped mode). The per-axis lead command of CP-5 therefore ended moves 8–12 mm off the path.
  `run_timed` now commands the model-inverse feedforward instead. On straight dev-scene moves the error dropped
  to 0.4–1.0 mm; it was 3 mm on arcs. The tube check measures distance to the path, not time lag. The contact
  detector uses the same coupled model and a directional force signal. On dev descents it fired at the first
  contact step; before, it fired 3 steps late.
- **TRACE.**
  - L-BFGS iterations per level are (5, 5, 10, 20).
  - A search cut short by the budget answers `no_path`, never a conflict or a look.
  - Fields are built on the GPU.
- **Not implemented.** Levenberg–Marquardt, SLSQP and SVGD, and representation factor A (only the
  tangent-offset spline exists).
- **Harness.**
  - TRACE requests first clear the held object's leftover map space.
  - Looks while holding raise the view by the payload's hang. This applies in trace mode only, so `--motion cp5`
    stays CP-5.
  - The map version moves only when the geometry changes, which keeps certificates and exact-geometry caches
    valid.
- **Fixes from the real t00 run.** These are general, and none is fitted to the episode:
  - **The tube grows with angular rate** (`tube_w` 3 mm per rad/s). In-place wrist turns drifted the hand 5–9 mm
    through the OSC coupling, against a tube that only grew with linear speed. The retimer now slows rotation
    where the slack is tight.
  - **Posture feedforward.** The posture command is `q + (q'' + kd_null q') / kp_null`, the inverse of
    robosuite's null-space loop (kp 10, a lag of about 0.6 s). Without it, the elbow trailed the arm that the
    certificate checked by about 3 cm, and the guard stopped it.
  - **Closed by contact.** A lateral end window flagged at the perception floor is closed by contact instead of
    sending the robot to look again. A look cannot lower the 3 mm floor, so the rule had blocked every can
    grasp: `1.64·σ + tube` > 7–9 mm. The integration doc says such an axis is closed by contact. The response
    carries `stats.closed_by_contact`.
  - **Finger exemption at contact approaches.** At the end of a contact approach, finger spheres against the
    step's named target are exempt from TRACE's end-pose penetration check, as in the doc's contact exemption.
    The path keeps the "do not get closer" rule against the target. The conservative finger spheres (about
    9 mm) otherwise reject every grasp standoff.
  - **"on" placement may stack (2.5D rest test, user 2026-11-22).** `manip.rest_candidates` replaces the 2D
  footprint-vs-walls test for "on": the payload rests where its bottom profile first meets the destination or anything
  already on it; parts that sit higher may overhang lower surfaces (a mug handle over a plate rim); every surface within
  the clearance level sideways must stay below the payload's local bottom (6 mm tolerance); stable = centroid inside
  the contact hull by 5 mm; stacking is scored down so a free spot wins. Stacked candidates get a footprint ignore box
  and a post-release height check at the planned rest height. t04 ep000 (second mug: 0 candidates before) succeeded:
  786 steps, 178 s, 1 call, monitor collision-free (`runs/e2e_stack_t04_ep000`, build 84d1d99d).
- **Pose requirement per transfer stage (user 2026-11-22).** New stage field `pose` ("any" / "upright", null =
  "any" for in, "upright" for on). "any": the drop over the reference (`plan_drop_in`, now also for "on" targets,
  whose top-view footprint is the opening) comes first and the careful placement is the fallback; "upright": the
  reverse. After a transit failure the other planner goes first. A dropped "on" object counts when it ends over the
  target at or below the surface it was let go over, in any pose. Runs: t00 ep000 success 142 s, 2 calls (drop
  poses near the rim failed TRACE transit 3x, then a repair); t08 ep000 failed as before (transit to the second
  pot's placements); all user-rule collision-free. The planner swap after a transit failure came after these runs.
- **Release verb from the instruction (user 2026-11-22; replaces `pose`).** The user writes the verb into the task
  instruction (`run_episode.py --instruction`, success check unchanged); S1 copies it per transfer stage into
  `release`: "drop" = `plan_drop_in` first, the careful placement only when no drop exists; "place" = the careful
  placement only, never a drop; null = drop for "in", place for "on". No planner swap. A drop now lets go with the
  payload's bottom 3 cm above the rim (hand in free space, so TRACE's end-pose check passes); 2 cm above what lies
  under it only where the hand cannot be above the rim. "drop" for a beside target (left of / right of ...): the
  careful planner's spot, released 3 cm above the surface first (lower levels only where the hand cannot be there),
  any resulting pose accepted. After the first verbs batch (build 44872e8a, stopped at 10/30): drop hand turns are
  tried smallest first (above the rim every turn passed, so -90 deg was always used and TRACE found no path, t07);
  a drop whose move fails is followed by the careful placement; CP-5's beside-target region crashed on a 3D
  direction (`region_for`, every left/right/front/behind target) and now uses the top-view direction.
  Verbs batch `runs/eval_1a0c0363_libero10_verbs_ep000_002_20261122T194943Z` (t00-t09 x ep000-002, user-approved
  prompts in its manifest): LIBERO success 13/30 (t00 3, t01 2, t02 3, t04 1, t07 3, t08 1; t03/t05/t06/t09 0), every
  success collision-free under the user's rule; `no_feasible_placement` is 29 of 42 stage failures.
- **Careful placement planner: contact classes, unseen space, margin ladder (user 2026-11-22).** The planner keeps
  margins only from what the user's collision rule counts. Hand and payload may touch the destination; unlabeled surface
  and unobserved space within 1 cm of it count as it (`checker.free_near`). At tier 2 the payload may also touch what
  is in/on it and the other task objects (`_place_excl_pay`). Unobserved space elsewhere is no obstacle while
  planning (`point_clearance(..., unknown=False)`, `esdf(..., include_unknown=False)`); a plan that relies on that is
  marked `unknown_near`, and `place_object` looks at the spot once and plans again. Ladder (`TIER_SCALE`): tier 0 =
  NEAR and the tolerance's footprint levels, tier 1 = half, tier 2 = half from everything else and touching for the
  allowed contacts; the arm keeps NEAR. Each candidate walks the ladder in the precheck; ranking is tier, release
  height, score. The "in" footprint prefilter is now exact (narrowest reach of the payload from its centre) instead of
  0.6 x its radius, which had rejected every cell for a 4.5 x 11.5 cm book in a 5.6 cm slot (t05).
  Rerun `runs/eval_6f795ffc_placement_ladder_t03-t08_ep000_002_20261122T211428Z` (t03 t04 t05 t06 t08 x ep000-002, same
  prompts): 3/15 vs 2/15 for build 1a0c0363 (t05 ep000 gained); placement stage failures 21 -> 16; tiers 1-2 used in 3
  plans, no collision from them.
- **Seed fallback.** When CP-5's planner finds no seed, the seed is the joint-space straight line.

## Known gaps

- **Wrist-limit approaches cost REPAIR calls.** This is the main remaining cost. TRACE orients the hand along
  the geodesic, with offsets up to 2.5 rad. When the start and end configurations sit near joint 7's limit
  (q7 2.3–2.7 against 2.90) and the base turns by about 1 rad, the arm check's rollout clips the wrist at the
  limit. The approach is then answered `no_path`, even with the joint-space seed. This cost one REPAIR in
  `trace4` and two in `trace3`. Candidate general fixes:
  - also offer each end's twin rotated 180° about the approach axis (the Panda gripper is symmetric);
  - add joint-limit avoidance to the rollout's null-space term.
- **Fast path is slower than the target.** Its live p95 is 0.69–0.80 s against the 0.5 s target. TRACE's own share
  is about 0.2 s; the rest is harness callbacks (IK about 70 ms per candidate, arm check about 125 ms per
  certification).
- **Replay is not exact.** The saved snapshot does not carry the live exact geometry, and 2 of 26 requests in
  the dev run differ.
- **Evaluator re-check is partial.** It covers hand and payload spheres, not the arm, and true distances are
  unsigned.
- **Deadline overshoot.** After the deadline, the remaining certification and the leave-one-out diagnosis are
  not bounded.
- **Payload weight reads as contact.** A payload heavier than about 0.6 kg reads as contact at the start of a
  lift (6 N force rule).
- **Posture lag.** The posture command keeps CP-5's lead, while the OSC null space lags by about 0.6 s, so the
  real elbow can depart from the rolled-out arm near obstacles.
- **Bench saturates.** It needs harder scenes or smaller budgets before it can rank solver pairs.
- **No gate measurement yet.** Gates from the merged build order (certificate honesty of at least 95%, guard
  stops near zero, at least half of conflicts resolved without a model call) need the batch evaluation of the
  improvement plan. The tooling exists; the sweeps have not been run.
