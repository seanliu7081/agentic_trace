# DESIGN: the integrated harness (CP-5 + TRACE)

Implementation spec for the build of 2026-11-22. It implements `TRACE integration.md` (read it first; it is the
spec), using the math of `TRACE_method.md` and the harness plan in `Agentic Manipulation Harness Improvement
Plan.md`. Section numbers like §5.5 refer to `TRACE_method.md`.

## 0. What is being built

TRACE is a numeric trajectory service behind the harness's middle layer. Three rules from the integration doc
govern everything:

1. The harness owns the scene and every model call. TRACE makes no model calls and has no perception or robot
   model. It reads one versioned scene snapshot and calls back into the harness for anything about the arm.
2. Each TRACE result is bound to one scene snapshot (`valid_while`).
3. TRACE stops at a standoff short of contact. The harness owns the last centimetre into contact, the gripper
   event and the check after it.

**Base.** The harness is a fork of CP-5 build `13686994` (`src/atrace/`, copied from
`/workspace/astra_harness_v5_cp5/src/cp5/`; see PROVENANCE.md). CP-5 already provides the improvement plan's
scene state: an RGB-D voxel map with free, occupied and unknown space, SAM 3.1 entities, a robot model with
batched FK and IK, and whole-body checks. It also provides one metric-free plan call (PROGRAM), a local retry
ladder, and REPAIR and VERIFY calls. **Never edit `/workspace/astra_harness_v5_cp5`**: another session is
developing it. Read it as much as you like.

**Changes from CP-5:**

1. Every contact-free segment goes through the TRACE contract (§2): reach to the pre-grasp, pre-place or part
   pose; carry; look poses; retreats; home. CP-5's joint-space planner (direct, lift, RRT) remains only as the
   `v0` seed and as `--motion cp5` for A/B comparison.
2. Segment ends are computed standoffs (`standoff: "auto"`), not CP-5's fixed 10 cm pre-poses. The guarded move
   from the standoff ends on contact, detected within one or two control steps.
3. The plan call emits per-stage `terms`: named levels plus allowed relaxations. A harness level table turns
   levels into tolerances. The model can neither create nor weaken safety terms.
4. A per-step guard runs with smaller margins than TRACE (the margin ladder). It adds a tube check, certificate
   validity rules and defect logging.
5. Every request and response is logged with its seed, so replay and the measures work at no model cost.

**Non-goals.** No change to CP-5's grasp, place or articulation solvers beyond the standoff and approach hooks.
No learned components. No edits to `/workspace/Libero10-Collision`. No task-specific constants or branches: a
general primitive parameterized by the plan and by run-time measurements is fine; a value fitted to one task or
episode is forbidden.

## 1. Layout

```
agentic_trace/
  DESIGN.md  PROVENANCE.md  README.md (written last)
  .venv -> CP-5 venv (Python 3.10, torch 2.10+cu128, numpy 1.26, scipy 1.15, mujoco 3.2.6, pytest 9)
  third_party, artifacts -> CP-5 (LIBERO assets, SAM 3.1 weights)
  src/traceplan/            TRACE (no imports from atrace)
    contract.py             DONE: shared data types (Spheres, Entity, SceneSnapshot, Term, EndCandidate,
                            TraceRequest, ArmResult, Timed, TraceResponse, to_jsonable)
    se3.py spline.py scene.py primitives.py program.py objective.py synthetic.py
    search.py certify.py diagnose.py service.py        (agent T-core)
    solvers.py retime.py                               (agent T-algo)
  src/atrace/               the harness (fork of cp5; relative imports)
    snapshot.py body.py levels.py guard.py tracemotion.py   (new, agent H)
    harness.py motion.py program.py                         (edited, agent H)
  scripts/run_episode.py dev_episode.py s1_stub.py          (forked, agent H)
  scripts/trace_bench.py trace_replay.py trace_measures.py true_scene.py   (agent C)
  tests/test_traceplan.py (T-core)  tests/test_solvers_retime.py (T-algo)  tests/test_atrace_offline.py (H)
```

Run Python as `/workspace/agentic_trace/.venv/bin/python`, with `src/` on `sys.path` (scripts insert it; tests
use `tests/conftest.py` with `sys.path.insert(0, <repo>/src)`).

## 2. TRACE (`src/traceplan`)

### 2.1 Contract

`contract.py` is written and fixed. Add fields only if a need cannot be met otherwise, and say so in your
result. `TraceService.plan(request, snapshot, arm_check, ik_check=None) -> TraceResponse` is the single entry
point the harness calls.

### 2.2 Trajectory (§3): `se3.py`, `spline.py`

- **`se3.py`**: batched torch `so3_exp`, `so3_log` (stable near 0 and near pi), `angle_between(u, v)`
  (atan2 form), and numpy float64 twins for certification.
- **`spline.py`**: clamped uniform cubic B-spline with K control poses. Knot vector:
  `[0,0,0,0, interior (K-4 uniform), 1,1,1,1]`. Provide:
  - `basis_matrix(ts, K, deriv)`, using Cox–de Boor, with derivatives 0, 1 and 2;
  - `greville(K)`;
  - `insert_knots(ctrl, K_from, K_to)`: exact Boehm insertion. The schedule `K_SCHEDULE = (4, 5, 7, 11)` has
    nested knot vectors.
- **`SE3Spline(R0, p0, Rg, pg, K, pin_end=True, phi_max=2.5, device, dtype)`**:
  - Position: `p(t) = sum_k B_k(t) p_k`.
  - Rotation: `R(t) = R_ref(t) Exp(sum_k B_k(t) phi_k)`, with `R_ref(t) = R0 Exp(t Log(R0^T Rg))`.
  - Pins: the start is pinned (`p_1 = p0`, `phi_1 = 0`). The end is pinned when `pin_end` is set; otherwise
    `p_K` and `phi_K` are free.
  - `theta` is the flat vector of free entries.
  - Methods: `dim`, `straight_theta()`, `theta_from_path(p_path (L,3), R_path=None)` and `ctrl(theta)`.
    `theta_from_path` is a least-squares fit at arc-length parameters with pins respected; with `R_path`, it
    also fits `phi` to `Log(R_ref^T R)`. `ctrl(theta)` returns `(P (M,K,3), Phi (M,K,3))`.
  - More methods: `eval(theta (M,D), ts)` returns a dict with `R (M,N,3,3)`, `p`, `v = dp/dt` and
    `acc = d2p/dt2`, all differentiable torch; `refine(theta, K_new)` returns `(SE3Spline, theta')` exactly;
    `sample_np(theta_1, ts)` returns `(R, p)` as float64 numpy.
  - Serialization: `to_dict(theta_1)` and `from_dict(d)`.
  - Clamp the `phi` control norms to `phi_max` (box bound, §3.2).
- **Properties that must hold** (unit-test them, §3.2):
  - The straight seed equals the straight geodesic, with error at most 1e-12 in float64.
  - `refine` leaves T(t) unchanged, with error at most 1e-12.
  - `||phi(t)|| <= max_k ||phi_k||`.

### 2.3 Scene fields (§4, as changed): `scene.py`, `synthetic.py`

The harness builds the `SceneSnapshot` (contract). TRACE derives its own fields from it and caches them on the
snapshot object (key by arguments):

- **`sdf_grid(snap, exclude=(), only=None, unknown=False)`** returns a float32 grid.
  - Input mask:
    - `occ` minus the voxels whose label is in `exclude`;
    - with `only`, just the voxels of that entity;
    - with `unknown`, the field is built from `unknown & ~shadow-of-excluded` instead.
  - Signed distance: `where(mask, -edt(mask), edt(~mask) - 1) * res`. This is conservative like CP-5's
    `SceneMap.clearance`. `edt` is `scipy.ndimage.distance_transform_edt`.
- **`blurred(grid, beta)`** applies `gaussian_filter` with `sigma = beta / res` voxels. Use it for the coarse
  continuation levels.
- **`GridField(grid, origin, res, device)`**: torch trilinear lookup `phi(x (...,3)) -> (...)`, differentiable in
  x. Use `grid_sample(align_corners=True, padding_mode="border")`, and mind the (z, y, x) order of the grid's
  last dimension. Above the grid's top it returns +1 m, so open air is free.
- **`coarse_occupancy(snap, inflate, res=0.02)`** returns a bool grid of occupied voxels dilated by `inflate`,
  for A* seeding. Unknown space counts as occupied here.
- **`exact_distance(snap, pts, exclude)`**: `snap.exact` if set, else the trilinear grid value. Both exclude the
  table plane, which is a separate primitive.
- **`changed_inside(snap_old, snap_new, pts, radius)`**: True if the occupancy or unknown state differs in a
  voxel within `radius` of any point. It implements the rule that the scene changed inside the tube.
- **`synthetic.py`** holds analytic scenes for tests and the bench:
  - `box_scene(boxes, spheres, table_z, bounds, res, cov)`. Boxes may carry a yaw. `exact` is the analytic SDF,
    `labels` and entities come from the shapes, and `unknown` is empty unless given.
  - `default_hand()`: Panda-like spheres in the grip-site frame. The finger-closing axis is x and the approach
    axis is +z. The palm box spans x ±0.10 m, y ±0.03 m and z from -0.11 to -0.031. The fingers are boxes at
    x = ±(opening/2 + 0.01), spanning z from -0.045 to +0.009. Link 7 is a cylinder of radius 0.045 along -z,
    from z = -0.11 to -0.25. Mark groups palm, finger and link.
  - `standard_scenes()`: S1–S6 of §8.3. Each is a dict with the snapshot, a start pose (R, p, q=zeros(7)), end
    candidates, bodies, hand-written terms (contract `Term`s) and a `holding` flag. Contents:
    - S1: free-space reach.
    - S2: one box obstacle on the straight line.
    - S3: a narrow passage; a shelf with an opening slightly wider than the hand.
    - S4: a keep-upright carry through clutter, with a payload sphere set and a `tilt_limit`.
    - S5: carry with attached payload, as a stand-in for multi-stage.
    - S6: a contradictory program. Two require terms cannot both hold. Example: `tilt_limit` of 5 degrees while
      the end orientation differs from the start by 40 degrees of tilt.

### 2.4 Primitives (§5.3, shared): `primitives.py`

Batched torch residuals on an `EvalContext` built from `SE3Spline.eval` output. The context holds R, p, v, acc
and omega (finite difference of R), sphere world positions per group `x = R c + p`, the radii, the snapshot's
fields and entities as tensors, and the start pose. Every primitive returns residuals `(M, N, d)` in physical
units. The registry `PRIMITIVES[name] = Spec(shape, fn, param, unit)` lists:

- `reach_point(a, b, offset)` (eq).
  - `a` is one of `"eef"`, `"payload.base"` (lowest payload sphere) or `"payload.centroid"`.
  - `b` is an entity keypoint `"<id>.<keypoint>"` or a world point `[x, y, z]`.
  - `perp` (optional unit axis) projects the residual perpendicular to that axis.
- `reach_rot(R_g)` (eq): `Log(R_g^T R)`.
- `align_axis(u, v)` (eq).
  - Axis names: `"eef.x|y|z"`, `"world.up"` and `"payload.up"`.
  - `payload.up(t) = R(t) R0^T u0`, where `u0` (`args["u0"]`, a world vector) is the object's up direction at
    the segment start, supplied by the harness.
- `tilt_limit(u, v, max)` (ineq): `angle(u, v) - max`. Param `max`, unit rad.
- `above(a, b, h)` (ineq): `(b_z + h) - a_z`. Param `h`.
- `clearance(body, field, margin)` (ineq), per sphere: `margin + r_s - phi_field(x_s)`. Param `margin`.
  - `body` is a group name or `"all"`.
  - `field = {"exclude": [ids], "only": id|None, "unknown": bool}`.
  - **Zone relaxation ("do not get closer"):** with `args["zone"]` set (default 0.05 m), samples within `zone`
    of the start position use `min(margin, c_start - 0.001)`. Samples within `zone` of the end position use
    `min(margin, c_end - 0.001)`.
  - `c_start` and `c_end` are the group's minimum clearance at the start and end poses. The service computes
    them from `exact_distance` and stores them in args.
- `plane_clearance(body, z, margin)` (ineq): `margin + r_s - (x_z - z)` against the table plane. Same zone rule.
  Param `margin`.
- `distance_band(a, b, lo, hi)` (ineq).
- `half_space(a, n, d)` (ineq).
- `approach_cone(axis, half_angle)` (ineq): `angle(v(t), axis) - half_angle`. It applies only to samples within
  `args["zone"]` of the end position with `|v| > 1e-6`, and is 0 elsewhere. Param `half_angle`.
- `value_map(grid_key)` (eq): a scalar GridField at p.
- `smooth_pos` (eq): `acc`. `smooth_rot` (eq): finite-difference angular acceleration.
- `reach_limit(center, r_max)` (ineq): `||p - center|| - r_max`.

### 2.5 Program, fusion, program tests (§5.1, §5.5, §7.1): `program.py`

- `Program(terms, holding, allow_contact)`.
  - `allow_contact` entries are `{body: "finger", entity, zone}`. Inside `zone` of the end, the named entity is
    excluded from that body's clearance field and a 1 mm floor applies. The harness only ever names finger vs
    the step's target.
- **`fuse(program, snap, z=1.64) -> (fused_program, flags)`.**
  - A term with an `entity` and `effect == "attract"` gets `sigma~^2 = sigma^2 + lambda_max(Sigma)`.
  - A `"repel"` term gets `margin~ = margin + z sqrt(lambda_max(Sigma))`.
  - Global safety clearance terms already include the ladder's perception bound; leave them as they are.
  - Flags: `{term_id: {"certifiable": sqrt(lambda_max(Sigma)) <= sigma, "uncertainty": ...}}`. The original
    sigma is used for acceptance.
  - When the term has `args["perp"]`, use the covariance projected perpendicular to that axis.
- **`run_program_tests(program, snap, make_spline, bodies, rng) -> findings`** returns
  `[{test, term, detail}]`. All tests are numeric, use no model and take milliseconds:
  - **lint**: known prim, `shape` matches the spec, entity references exist, scope inside [0, 1], role valid,
    and no `safety` term whose `src` is `"language"`.
  - **range**: sigma within sane bounds per unit (m: 1e-4..0.5; rad: 1e-3..1.6), and violations on the straight
    seed below 1e3. This catches cm-for-m and degree-for-radian mistakes.
  - **theta-dependence**: `max ||dr/dtheta||` over 8 random trajectories must be greater than 0. This catches a
    term between static entities, or a payload term while not holding.
  - **invariance**: apply a random yaw about z plus a translation to the snapshot queries and the trajectory
    together; residuals must stay equal within 1e-5. Implement it by transforming query points with the inverse
    transform, entity keypoints with the forward transform, and the start and end poses. This catches
    hard-coded world coordinates.
  - **start feasibility**: a path-scope `require` term must hold at the start pose.

### 2.6 Objective (§5.2): `objective.py`

- `Objective(program, snap, bodies, spline, start, ends, device, n_eval=32)` implements `J_l` exactly as §5.2.
  - Prefer terms use the robust kernel: Huber, or identity for `src == "regularizer"`, weighted by kappa.
  - Safety and require terms use the augmented Lagrangian psi, with multipliers per (seed, term, sample).
  - `set_level(level, spline)` switches the tolerance inflation `s_l`, the blurred fields (`beta_l`), `mu_l` and
    K. The default levels are `LEVELS = [(s=8, beta=0.08, K=4, mu=1), (4, 0.04, 5, 4), (2, 0.02, 7, 16),
    (1, 0, 11, 64)]`.
  - `__call__(theta (M,D)) -> J (M,)`.
  - `update_multipliers(theta)` and `demote(term_id)`. `demote` returns a copy with that term as prefer, for
    leave-one-out.
- **`dense_report(theta_1, n=128)`** is the certificate's term check.
  - It uses the original sigma and exact geometry for clearance terms: `exact_distance` on the sphere centres
    at the dense samples.
  - It returns per term `max_violation`, `at_t`, `role`, `src`, `kappa`, `level`, plus per-sample clearance
    slack per group, which retiming uses.

### 2.7 Solvers (§6.3): `solvers.py` (agent T-algo)

Generic, with no knowledge of robots. `J: Tensor (M,D) -> Tensor (M,)`.

- **`explore(name, J, theta (M,D), budget_evals, gen, cfg) -> theta`**.
  - `name` is one of `none`, `random`, `cem`, `mppi` or `cmaes`; cmaes may be a diagonal (sep-CMA) variant.
  - Each of the M seeds is optimized independently. Draw samples per seed and evaluate all M x n_samples in one
    call of J.
  - Noise is smooth in control-point space: `eps ~ N(0, alpha^2 (D^T D + eps I)^-1)`, where D is the
    second-difference operator over the K control points, applied per coordinate. `cfg["layout"]` gives
    `(n_ctrl, n_pos_dims, n_rot_dims)` so position and rotation get separate scales `alpha_pos` and `alpha_rot`.
    The caller shrinks those scales with `s_l`.
  - MPPI: `theta <- sum_m w_m theta_m`, with `w ∝ exp(-(J_m - min J)/tau)`.
- **`exploit(name, J, theta, budget_iters, cfg) -> theta`**: `none`, `adam` or `lbfgs` (torch autograd; J summed
  over seeds is separable). Levenberg–Marquardt and SLSQP are not needed; say so in the docstring.
- **`portfolio(arms, J, budget_evals)`**: successive halving over (seed, explore, exploit) arms, keeping the
  better half by J and doubling the budget (§6.3 portfolio mode).
- Everything is driven by a `torch.Generator`, so a seed reproduces the result on CPU.

### 2.8 Search (§6.1, §6.2, §6.4): `search.py`

- **Terminal candidates.**
  - With explicit `ends`, apply the standoff to each candidate (§2.10). Then score it with the terminal-scope
    terms, clearance of the hand spheres at the end pose, and `ik_check` if given. Drop infeasible candidates
    and keep the best `G = 2`, in the harness's order when scores tie.
  - With `free_end`, sample a few hundred poses around the entity: above or at the entity at the given height,
    with random yaw. Refine each with Adam on a 6-D pose vector under the terminal terms, keep the best G
    distinct ones, and verify each with `ik_check`.
- **Path seeds.** For each terminal candidate:
  - the straight seed;
  - `H - 1` A* paths on `coarse_occupancy` with inflation equal to the largest hand-sphere radius plus 1 cm.
    After each path, penalize voxels within 4 cm of it, so later paths take different corridors. Fit each path
    with K=4 control points by least squares;
  - the candidate's provided seed path (`p_path`, `R_path`), fitted at K=11.
  - Defaults: H = 4. Seeds are spatially diverse but not topologically complete.
- **Coupled continuation.** At each level of `LEVELS`: for every seed run explore then exploit, then update the
  multipliers. Keep the best half of the seeds (at least 1), then refine knots exactly to the next K. Starting
  budgets: 256 samples per explore iteration, 10 explore and 20 exploit iterations per level, rounds R = 1.
  The explore/exploit pair defaults to `("mppi", "lbfgs")` unless the bench says otherwise.
  - Respect `request.budget_ms`: check the clock between seeds and levels. When time runs out, stop and go to
    certification with what exists.
- For closed-loop reuse, `search(..., warm_theta=)` reruns only the last level from a previous solution.

### 2.9 Certificate, arm check, retime (§7.2 as changed): `certify.py`, `retime.py`

- **`certify(...)`** returns `(accepted, cert)`. It runs, in order:
  1. `dense_report` at 4N = 128 samples. Every safety and require term needs `max_violation <= 1`.
  2. The arm check. Call `arm_check(R, p, s, q_ref)` with the dense samples. The default `q_ref` is the linear
     interpolation by arc length from `start_q` to the end candidate's `q` (or `start_q` when the candidate has
     none), or the candidate's `q_path` resampled. The check must return `ok`.
  3. Retiming must succeed.
  - `cert` holds the report rows, the flags, the rolled-out `q`, the arm clearance and the timed result.
- **`retime(p, R, q, s, slack, cfg) -> Timed`** sets speed from certified clearance: fast where there is room,
  slow in tight passages. The tube grows with speed.
  - Per dense sample, the speed cap is `v_cap = clip((slack - tube0) / tube_k, v_min, v_max)`. Here `slack` is
    the minimum over groups of (clearance - guard - perception), from `dense_report`, combined with the arm
    check's clearance when given.
  - Where a relaxed known-space term is violated, cap at `limits["v_slow"]`.
  - Then a forward and backward pass with `a_max` and the angular limit `w_max`.
  - Resample everything on the control grid `dt`: p by linear interpolation, R by slerp, q linearly.
  - `tube_k = tube0 + tube_k * speed_k`. `valid_while.tube_mm` is the maximum tube over the path, in mm.

### 2.10 Standoff, uncertainty, results (§7.3 plus the integration doc): `diagnose.py`, `service.py`

- **Auto standoff.** For a candidate with an `approach` axis a and an entity whose covariance is Sigma:
  - `s = clip(z sqrt(a^T Sigma a) + tube(v_min) + guard["finger"] + 0.005, s_min=0.02, s_max=0.10)`.
  - The end pose is `(R, p - s a)`.
  - Add a terminal require term `end_window`: `reach_point(eef, p_end)` with `perp=a`, sigma = window, effect
    attract, entity = the candidate's entity. If `z sqrt(lambda_max(P Sigma P)) + tube(v_min) > window`, with
    `P = I - a a^T`, the candidate is flagged *not certifiable with this perception*.
  - This applies the integration doc's rule: an axis whose workable window is smaller than perception error
    plus tracking error must be closed by contact (along a, through the standoff) or by a closer look
    (laterally).
- **The flow of `TraceService.plan`:**
  1. **Compile.** Add the start/end clearances to clearance terms and the auto standoffs, run `fuse`, then run
     `run_program_tests`. Any finding returns `program_error` with `findings`, before any search.
  2. **Terminal candidates.** Skip candidates flagged not certifiable. If every candidate is flagged, return
     `not_certifiable` with `look_at = {entity, term, uncertainty, tolerance, axis: "lateral"}`.
  3. **Fast path.** For each kept candidate in order, certify the straight line. If one is accepted, return
     `certified` with `stats.stage = "fast_path"`.
  4. **Seeds and search.**
     - In mode `"v0"`: certify the provided seed paths only, without search.
     - In mode `"trace"`: run the coupled continuation from all seeds within the budget. Certify the results in
       order of J, then any unrefined provided seed path.
     - The first accepted path returns `certified` with stage `search` or `seed`.
     - If the budget runs out, certify what exists.
  5. **Diagnose the best non-accepted path.**
     - **Conflict.** Rank require terms by violation and multiplier. For the top 3, leave one out: demote it to
       prefer and rerun the last level from the best theta. If the rest becomes feasible, the term is in the
       conflict. Return `conflict` with a `quote` entry per conflicting term:
       `{term, prim, level, relax_to, param, value, needed, unit}`. Here
       `needed = value + max_t r(t) * sign`, the parameter value the re-solved path would need. Examples:
       "tilt max 10 deg -> needs 23 deg"; "clearance 8 cm -> feasible at 3.1 cm".
     - **Look.** If only safety clearance terms are violated, and each violation is at most the perception part
       of the margin near a labelled entity whose `sqrt(lambda_max)` exceeds 5 mm, return `not_certifiable` with
       look_at that entity. The quote names which remedy would open the passage, a closer look or slower
       motion.
     - **No path.** Otherwise return `no_path`: a search failure, not proof that no path exists.
  6. **Response.** A certified response carries `timed`, `spline`, `end_index`, `end_R`, `end_p`, `standoff`,
     `valid_while = {scene_version, tube_mm, holding, start_q}`, the full `report` and `stats` (stage, solve_ms,
     certify_ms, evals, seeds, levels_run, candidates_tried).
- **Seeded and logged.** Seed numpy and torch from `request.seed`. `TraceService(log_dir=...)`, when given a
  directory, does two things for every plan call:
  - appends one JSON line (request summary, response summary, timing) to `log_dir/trace.jsonl`;
  - pickles `(request, response)` to `log_dir/req_<request_id>.pkl` after stripping callables, and saves the
    snapshot once per version to `log_dir/snapshots/v<version>.npz`.
- **Device.** Search runs in torch float32 on `cfg["device"]` (CUDA if available, else CPU). Certification
  runs in float64 numpy.

## 3. The harness (`src/atrace`)

### 3.1 Fork hygiene

- The package is `atrace`; it uses relative imports. Scripts import `atrace.*`.
- The build hash covers `src/atrace/*.py`, `src/traceplan/*.py` and `scripts/run_episode.py`.
- Output runs go to `agentic_trace/runs/`.
- CLI flags:
  - `--motion trace|v0|cp5` (default `trace`);
  - `--trace-device` (default `cuda:0`);
  - `--trace-budget-ms` (default 2000);
  - `--compiles N` (default 1);
  - `--true-scene` (evaluator side, §4).

### 3.2 Snapshot export: `snapshot.py`

`build_snapshot(h, version_key=None) -> SceneSnapshot` reads the harness's `SceneMap` and objects and copies:

- `occ = (state == OCC)` with the held object's labelled voxels removed (the held object travels as payload
  spheres);
- `unknown = (state == UNKNOWN)`, `labels`, `shadow`, `table_z = scene.table_z`, origin, res;
- entities from the `ObjectModel`s: id = the plan's object id (the reverse of `h.id_map`, else `f"oid{oid}"`),
  label = oid, points subsampled to at most 600, cov from `entity_cov(obj)`, and keypoints `top_center`, `base`
  and `centroid` from CP-5's model;
- `exact = lambda pts, exclude: h.checker.point_clearance(pts, oids(exclude) + held)`, CP-5's exact near field.

**Version.** Use `scene.version` and bump it on any object-model change, so a new version means new geometry.
Cache by version.

**`entity_cov(obj)` (perception model, configurable, not tuned).** It returns `diag(s^2, s^2, s^2)` with
`s = sqrt(0.003^2 + (0.006 if the object has no close view else 0)^2)`. A close view is an entry in `obj.views`
from the wrist camera. This exports the per-entity covariance the integration doc requires.

### 3.3 Bodies and controller model: `body.py`

- **`hand_spheres(rm, finger)`**:
  - CP-5 robot-model collision points of the palm (group 1) and finger (group 2) bodies at that opening, in the
    grip-site frame, voxel-downsampled at 1.2 cm for the palm and 0.8 cm for the fingers. Each radius is the
    sample radius plus the downsampling cover radius.
  - **link** spheres from `robot0_link7`, which is rigid relative to the hand, at 2 cm.
  - The result is a `Spheres` with groups palm, finger and link.
- **`payload_spheres(attached)`**: `Attached.pts_site` voxel-downsampled at 1.5 cm, radius 0.9 cm, group
  payload.
- **`ControllerModel(rm, kp=150, damping=1, dt=0.05)`**: the one-step model of the absolute-pose OSC.
  - **`predict(p, v, p_cmd)`**: the exact discretization over dt of `x'' = kp (x_cmd - x) - kd x'`, per axis,
    with `kd = 2 damping sqrt(kp)`. This is the guard's prediction. Its error is the contact signal.
  - **`rollout(q0, R (N,3,3), p (N,3), q_ref (N,7))`** returns `(q (N,7), err_p (N,), err_r (N,))`. It is the
    controller's redundancy resolution: per sample, a few damped least-squares steps (lambda 0.03) warm-started
    from the previous q, with null-space attraction to `q_ref[k]`, clipped to joint limits minus 0.02.
    Batched-friendly numpy.
- **`make_arm_check(h, margins, zone)`** builds the `arm_check` callback:
  - Run `rollout`. Fail with `unreachable` if `err_p > 3 mm` or `err_r > 0.03 rad`, `joint_limit` if within 0.01
    of a limit, `discontinuous` if `|dq|_inf > 0.35` between samples.
  - Compute arm-group clearance (CP-5 `Checker.clearance` for groups arm and plane_arm, exact) against
    `guard["arm"] + tube(v_min) + perception`. The zone relaxation of §2.4 applies near both ends.
  - Return `ArmResult(ok, q, reason, at_t, min_clearance, clearance - margin)`.
  - `ik_check` wraps `rm.solve_ik`.

### 3.4 Level table and term compiler: `levels.py`

Model-facing vocabulary. The model writes levels; this table owns the numbers:

| prim | levels (param, sigma) | applies to |
|---|---|---|
| `tilt_limit` | upright (max 10°, σ 3°), tilted (25°, 5°), loose (45°, 8°) | payload.up vs world.up, carry segments only |
| `clearance` | close (margin 1 cm, σ 0.5 cm), normal (3 cm, 1 cm), wide (8 cm, 2 cm) | all bodies vs the named entity only (`field.only`), effect repel |
| `approach` | vertical (half-angle 10°, σ 4°), steep (25°, 6°) | approach_cone along the end candidate's approach axis, in the end zone |

Harness-only levels:

- `known_space`: `strict` (clearance from unknown space, margin = guard, require) or `slow`. `slow` demotes the
  term to prefer and sets `args["slow"] = v_slow` so retiming caps speed there.
- Goal tolerance for standoff ends: 1.5 cm and 6°. The note's 5 mm default equals what the arm achieves under
  load.

Functions:

- `compile_terms(stage_terms, h, phase, holding)` maps plan terms to contract `Term`s (src `"language"`, with
  `level` and `relax_to` copied).
  - A `tilt_limit` of the held object applies only in `phase == "carry"`; `u0` is the object's up at grasp
    carried into the current hand orientation.
  - Unknown entity ids are kept as they are, so the program tests report them (interface repair).
- `safety_terms(h, phase, ladder, zone)` writes the harness's safety terms:
  - `clearance` per group (palm, finger, link, payload when holding) against occupied space, excluding the held
    entity, with margin `guard[g] + tube(v_min) + perception` and zone relaxation;
  - `plane_clearance` per group with margin `guard_plane[g] + tube(v_min)`;
  - `reach_limit` (Panda shoulder = base + (0, 0, 0.333), r_max 0.855);
  - regularizers `smooth_pos` and `smooth_rot` (prefer);
  - `known_space` as require with `relax_to = "slow"`.
- `can_relax(quote_entry, term)` says whether `LEVELS[prim][relax_to]` admits `needed`.

**Default margin ladder.** All values are configurable and none was tuned on test episodes:

- `guard = {arm: 0.010, palm: 0.008, finger: 0.005, link: 0.010, payload: 0.005}`
- `guard_plane = {arm: 0.015, palm: 0.008, finger: 0.004, link: 0.015, payload: 0.003}`
- `perception = 0.005`
- `tube0 = 0.004 m`, `tube_k = 0.022 s` (15 mm at 0.5 m/s, matching the recorded tracking), `v_min = 0.05`
- `limits = {v_max: 0.40 (0.20 holding), a_max: 0.5 (0.25 holding), w_max: 1.2, dt: 0.05, v_slow: 0.05}`
- `zone = 0.05`

### 3.5 `trace_move` and the four reactions: `tracemotion.py` and `harness.py`

`go_configs(...)` keeps its signature. With `--motion cp5` it is unchanged. Otherwise it calls
`self.trace_move(ends=[EndCandidate(site_pose(g), q=g) for g in goals], tag=...)`, with standoff 0 for these
pose-goal moves. The reach call sites use `self.reach(...)`, which passes contact poses (§3.6).

`trace_move` builds a `TraceRequest`:

- start = measured pose, q and finger opening; `holding` = the held entity id;
- ends; terms = `compile_terms(stage terms of the current stage, phase)` plus `safety_terms`;
- bodies = `hand_spheres(finger)` plus payload spheres when holding;
- margins and limits from §3.4; `mode` from `--motion`;
- seed = a per-episode counter;
- for mode v0, or as the last seed in trace mode, the candidate's seed path from CP-5's planner. Use
  `planner.plan(q0, [g], TRANSIT, ..., allow_rrt=True, time_budget=2.0)`, densified and converted with FK. Only
  compute it if the fast path fails: call TRACE once in `"straight"` mode first.

It then calls `TraceService.plan` and reacts:

| result | reaction (logged as `trace_reaction`) |
|---|---|
| certified | Execute `run_timed` (§3.7). On `done`, carve the swept robot as CP-5 does. On `contact`, `tube_exit` or `guard_stop`, void the certificate, log the defect, `reobserve()`, re-request once, then fail the move. On `dropped`, raise `StageFailure("dropped_in_transit")` as CP-5 does. |
| conflict | If every quoted term has a `relax_to` and `can_relax`, replace those terms' levels and re-request, with no model call. Otherwise raise `StageFailure("trace_conflict", quote=..., terms=...)`, so one REPAIR call carries the quote. |
| not_certifiable | Look: `look_from_above` the `look_at` entity (CP-5's grasp-time look and refresh), `rebuild_map`, new snapshot, re-request. At most 2 looks per move; then treat as no_path. |
| no_path | Next candidate: return False with `self.last_move = {status: "no_path", ...}`. The caller's ladder tries its next grasp, place or look candidate. When the ladder is exhausted, CP-5's StageFailure leads to REPAIR, the one model call. |
| program_error | Interface repair: drop the offending language terms, log them in `self.dropped_terms` (included in the next REPAIR report), re-request. |

A `known_space` conflict is always relaxable (to `slow`): crossing unknown space becomes slow guarded motion.

### 3.6 Standoffs and guarded moves

- **`reach(contact_R, contact_p, q_contact, approach, entity_oid, window, tag, grip, finger, q_pre)`.** In trace
  mode it requests `EndCandidate(R, p, q=q_contact, approach, standoff="auto", window, entity=id)`. After a
  certified segment the hand is at the standoff. The caller's existing guarded descent (`move_line` to the
  contact pose) covers the rest, now from the standoff. In cp5 mode it is `go_configs([q_pre])`. Wire these
  call sites:
  - `grasp_object` `to_pregrasp`: approach `g.R[:,2]`; window = finger clearance per side,
    `(opening - width) / 2`;
  - `place_object` `to_preplace`: approach = the unit vector from the pre-place to the place position; window =
    the placement tolerance; use `0.01` if unknown;
  - the articulation `to_part` and `to_press` calls: the approach from their line;
  - keep each caller's later logic unchanged.
- **Retreats.** Route the contact-free lifts and back-offs (post-grasp lift, post-release lift, back-offs after
  blocked descents or empty closes) through `self.retreat(R, p_goal, ...)`: a `trace_move` with one standoff-0
  candidate. If TRACE does not certify (near contact), fall back to CP-5's `move_line`. Departures from contact
  remain the harness's.
- **Contact detector (`guard.py`).** `ContactDetector(controller_model, f_thr=6.0, e_min=0.0015)` fires when
  either condition holds:
  - the force change exceeds `f_thr`;
  - the prediction error along the commanded direction exceeds `max(e_min, 4 sigma_free)` for 2 consecutive
    steps.
  `sigma_free` is the running std of the error over the first free-motion steps of the move. Expose
  `detector.guard(reason)`, a callable with CP-5's guard signature `(robot, F, F0, k)` that keeps CP-5's force
  rule as well. Use it in the grasp descent (reason `early_contact`), the place descent (`support_contact`) and
  the press. The descent then ends on contact, as the integration doc requires (CP-5's `close_at_contact` logic
  stays).

### 3.7 Execution, tube, guard: `motion.py` and `guard.py`

**`Executor.run_timed(timed, grip, finger, guard, detector, drop_check)`.** For each control step k it does the
following:

- Command pose `T(t_k + TAU_LEAD)` (interpolated) with posture `q(t_k + TAU_LEAD)`, as CP-5's `run_path` does.
- Tracking error = `||p_meas - p(t_{k+1})||`. If it exceeds `tube[k+1]`, stop with `"tube_exit"`.
- `guard.check(q_meas)`: the per-step guard. It checks robot and payload clearance on the snapshot's exact
  geometry against `guard` and `guard_plane` alone, with the same zone relaxation and allowed contacts as the
  certificate, plus joint limits. On a violation, stop with `"guard_stop"`.
- `detector`: stop with `"contact"`.
- `drop_check`: stop with `"dropped"`, the object-follows-gripper check.
- At the end, settle like CP-5: stop when error < 4 mm and speed is small, or after 3 stable steps with error <
  1.5 cm (the standoff tolerance), or after 40 steps.

Any stop on a certified path is a **defect**. Log it as `guard_defect` with `{request_id, step, kind, group,
clearance, tracking_err, tube, scene_version, explanation}`. The explanation is one of three:

- `scene_changed`: the snapshot version moved since certification;
- `tracking_left_tube`: the error exceeded the tube before the stop;
- `geometry_disagreement`: anything else.

**Certificate validity.** Before executing, check that the response's `valid_while` matches the current
snapshot version, held entity and start q (within 0.02 rad). If only the version differs, run
`changed_inside(old, new, swept sphere centres, tube)`: no change means the certificate is still valid; a change
means void and re-request.

### 3.8 Plan call: terms (`program.py`)

- Add `terms` to `STAGE_SCHEMA`: an array of `{prim: enum[tilt_limit, clearance, approach], of: string,
  level: enum[all levels], role: enum[require, prefer], relax_to: enum[levels] | null}` (strict schema, all
  required).
- Add 8–12 lines to `PROGRAM_INSTRUCTIONS` (see the vocabulary table). Use terms only when the task or the
  objects call for it: keep an open container upright, stay well clear of a fragile or precarious object, or
  arrive vertically over a narrow opening. Collision avoidance is automatic, so no terms for it. `relax_to` is
  a looser level System 2 may switch to by itself, or null.
- REPAIR instructions: after a `trace_conflict` the report lists the terms that cannot all hold, and the
  smallest relaxation (physical units) that would make the motion feasible. Relax, rescope or drop them, or
  change the plan.
- `validate_program` checks the level against the prim.
- `--compiles N > 1`: N-1 extra PROGRAM calls at effort low, run in parallel. kappa = the fraction of compiles
  containing the term (matched by stage index, prim and of). Take the most common level. A require term with
  kappa < 0.4 becomes prefer. Off by default.
- `s1_stub.py` adds to every transfer stage `{prim: tilt_limit, of: <object>, level: upright, role: require,
  relax_to: tilted}`, so offline runs exercise the term path. Offline runs have no model.

### 3.9 Logging

Write `run_dir/trace/` through the service's `log_dir`. Add these entries to `events.jsonl`:

- `trace_request`: id, tag, n_ends, mode;
- `trace_result`: result, stage, solve_ms, tube_mm, standoff, end_index;
- `trace_reaction`;
- `guard_defect`;
- `exec_timed`: request_id, status, steps, first_step, last_step.

The step range lets the measures match collision onsets to certified executions. The summary gets a `trace`
block with counts per result and reaction, the fast-path share, solve-time p50 and p95, and guard stops.

## 4. Tools, bench, tests (agent C, after the others)

- **`scripts/trace_bench.py`**: Stages 1–4 of §8.2 on `standard_scenes()`, reduced to 3 seeds.
  - Stage 1: explore x exploit over {none, random, cem, mppi} x {none, adam, lbfgs} at equal evaluation budgets.
  - Stage 2: seeding (straight / random / A*), continuation (off / all) and hard terms (penalty / AL).
  - Stage 3: the K schedule.
  - Stage 4: program checks on 5 injected error types x 20 programs: wrong entity, unit error, not attached
    (theta-independent), hard-coded world coordinate, wrong scope. Report detection rate and false alarms.
  - Write JSON plus a markdown table to `runs/bench_<stamp>/`. If one pair clearly wins, set it as the default
    in `search.py`.
- **`scripts/trace_replay.py RUN_DIR`**: re-run every logged request on its saved snapshot (grid geometry, since
  `exact` is not saved). Compare result category, end index and max position difference. Write a table.
- **`scripts/trace_measures.py RUN_DIR`** computes:
  - requests by result; fast-path share; solve time p50 and p95 for the fast path and the search;
  - conflicts resolved without a model call versus all conflicts;
  - looks; guard stops per certified path; tube exits;
  - certificate honesty: certified executions whose step range contains a forbidden contact onset, from
    `collision.json` in eval runs or `contacts.json` in dev runs.
  - With `true/` snapshots it also re-checks each certified trajectory on the true scene, which splits defects
    into perception error and tracking error.
- **`scripts/true_scene.py`** (evaluator only, never a policy input): `true_snapshot(env, res=0.01)` built from
  the MuJoCo collision geoms of non-robot bodies at the current state. Sample the surfaces of boxes, cylinders,
  spheres, capsules and meshes at 3 mm, voxelize, and use a KD-tree distance as `exact`. `--true-scene` in the
  runners saves `run_dir/trace/true/<request_id>.npz` from an evaluator hook the harness calls with the request
  id only.
- Run every test once. Do one offline dev episode with no model calls on a non-test scene, GPU 0:
  `--suite libero_object --task pick_up_the_ketchup_and_place_it_in_the_basket --init 3 --s1 offline --goal
  in:ketchup_1:basket_1_contain_region,in:bbq_sauce_1:basket_1_contain_region`. Fix what it shows.

## 5. Rules for implementers

- **Necessary checks only.** No defensive code: no try/except around your own logic, no input re-validation
  inside TRACE (the contract is trusted; the program tests are the checks), no fallbacks for impossible states.
  Let bugs raise.
- **Lean code.** Vectorize in torch and numpy. Comments say why, not what. No dead options: every option listed
  here exists because a doc asks for it.
- **No task-specific values.** Constants are general and listed in one config dict per module.
- **GPUs and data.**
  - Use GPU 0 only, and only for the simulator smoke runs. Use `CUDA_VISIBLE_DEVICES=0` for torch tests.
  - Set `OMP_NUM_THREADS=4` for numpy work; the machine is shared.
  - Never run, replay or read simulator truth in Libero10-Collision test scenes (KITCHEN_SCENE3, 4, 6, 8;
    LIVING_ROOM_SCENE1, 2, 5, 6; STUDY_SCENE1) during development.
- **Tests run once.** Report what was and was not tested.
