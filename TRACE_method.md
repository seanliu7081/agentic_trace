# TRACE: Tolerance-typed Residual Programs with Annealed, Certified Search

*Working name. Method note v0.1, 2026-11-20.*

**Status.** This is a design, not a tested system. Nothing here has been benchmarked. The three spline properties in §3 were checked numerically; every hyperparameter is a starting value, not a tuned one.

---

## 1. Summary

TRACE finds a robot end-effector (eef) trajectory in SE(3) from three kinds of input (a goal pose, a language instruction, and perception) without training anything.

- **Representation.** The trajectory is a function $T(t;\theta)$ with 18 to 60 parameters: a cubic B-spline for position and a tangent-space B-spline offset on a geodesic reference for rotation.
- **Loss.** Every input source is compiled into the same object: a residual in physical units plus a tolerance. Dividing one by the other puts metres, radians and language constraints on one dimensionless scale, where "1" always means "at the edge of acceptable".
- **Search.** Diverse seeds are refined coarse-to-fine by a pluggable pair of solvers (one sampling, one gradient), with tolerance, scene detail and spline resolution tightened together.
- **Checks.** The loss is tested numerically before the search runs, and the result ships with a per-term certificate. When constraints conflict, the method reports which ones and by how much.

```
goal pose ───────────────┐
language ──(LLM/VLM)─────┼──► residual program ──► tests on the program
RGB-D ──(frozen models)──┘          │                      │
        │                           ▼                      ▼
        └──► scene model ──► seeds ──► annealed search ──► certificate
                                       (explore → exploit)     │
                                              ▲                │ conflict?
                                              └── repair ◄─────┘
```

---

## 2. Problem and scope

**Given**

- the current eef pose $T_0=(R_0,p_0)\in SE(3)$;
- one or more RGB-D observations;
- a language instruction $\ell$;
- optionally a goal pose $T_g$, or a set of acceptable goal poses.

**Find** a trajectory $T(t)=(R(t),p(t))$, $t\in[0,1]$, plus gripper open/close events, that satisfies every requirement from all three sources, or a report of which requirements cannot be met together.

**"Training-free" means** no gradient update to any network, no demonstrations and no task data. The solver has no learned parameters. Frozen pretrained models are used only as tools: perception models to build the scene, and an LLM/VLM to translate language into the residual program. If no neural model at all is allowed, the goal-pose and geometric channels still work and the language channel needs a hand-written program.

**Assumptions**

- Motion is kinematic and quasi-static. Contact dynamics are not modelled.
- A grasped object moves rigidly with the gripper.
- The search is over the geometric path. Timing is assigned afterwards by retiming, so speed requirements are handled there.

---

## 3. Trajectory function

### 3.1 Definition

Use $K$ control poses $(p_k,\phi_k)$ and a clamped cubic B-spline basis $B_k(t)$.

$$p(t)=\sum_{k=1}^{K}B_k(t)\,p_k$$

$$R(t)=R_{\mathrm{ref}}(t)\,\mathrm{Exp}\Big(\sum_{k=1}^{K}B_k(t)\,\phi_k\Big),\qquad R_{\mathrm{ref}}(t)=R_0\,\mathrm{Exp}\big(t\,\mathrm{Log}(R_0^{\top}R_g)\big)$$

- $p_k\in\mathbb{R}^3$ are position control points and $\phi_k\in\mathbb{R}^3$ are rotation offsets in the tangent space.
- $R_{\mathrm{ref}}$ is the shortest rotation from the start to the candidate goal orientation. The spline only has to encode the deviation from it.
- The start is pinned: $p_1=p_0$, $\phi_1=0$. The end is pinned when an exact goal pose is given and free otherwise.
- The search variable is $\theta=(p_2,\dots,p_K,\phi_2,\dots,\phi_K)\in\mathbb{R}^{6(K-1)}$, a plain unconstrained vector.

### 3.2 Properties the search relies on

| Property | Statement | Why it matters |
|---|---|---|
| Straight-line start | Control points placed on the start-goal line at the Greville abscissae, with all $\phi_k=0$, give exactly the straight geodesic path. | A cheap, valid initial guess at any $K$. |
| Exact refinement | Inserting knots produces a larger control set that traces the identical $p(t)$ and identical offset, hence the identical $T(t)$. | Resolution can grow mid-search without losing progress. |
| Bounded offset | $\lVert\phi(t)\rVert\le\max_k\lVert\phi_k\rVert$. | A box bound $\lVert\phi_k\rVert\le\phi_{\max}<\pi$ keeps $\mathrm{Exp}$ away from its singularity. |
| Batched evaluation | With a precomputed basis matrix $B\in\mathbb{R}^{N\times K}$, positions for $M$ candidates are one matrix product. | Thousands of candidates per iteration on a GPU. |

Numerical check of the first three: line error $2\times10^{-16}$ for $K\in\{4,5,7,11\}$; curve change under refinement $4\times10^{-16}$ along $4\to5\to7\to11$; offset norm 2.29 against a control bound of 2.5.

### 3.3 Derived quantities

- Gripper collision spheres: $x_s(t)=R(t)\,c_s+p(t)$ for sphere centres $c_s$ in the eef frame.
- Points on a grasped object: $x(t)=T(t)\,T(t_{\mathrm{grasp}})^{-1}\,x(t_{\mathrm{grasp}})$.
- Velocity and acceleration of $p$: analytic from the basis derivatives. Angular velocity: $\omega_j=\mathrm{Log}(R_j^{\top}R_{j+1})/\Delta t$.
- Cost evaluation uses $N$ time samples (start with $N=32$); certification uses $4N$.

---

## 4. Scene model

Built once per observation from frozen models and geometry.

| Element | How it is obtained | Used for |
|---|---|---|
| Signed distance field $\Phi(x)$ | Depth to point cloud to TSDF, then a distance transform; trilinear lookup | Clearance |
| Blurred copies $\Phi^{(l)}=G_{\beta_l}*\Phi$ | Gaussian blur of width $\beta_l$ | Coarse levels of the search |
| Coarse occupancy grid | Obstacles inflated by the gripper radius | Seeding |
| Entities | Open-vocabulary detection, segmentation, back-projection; centroid, principal axes or 6D pose, named keypoints | Arguments of residuals |
| Covariance $\Sigma$ per keypoint or object | Depth noise model, mask-boundary spread, or variance across frames | Uncertainty-aware fusion (§5.5) |
| Gripper spheres | Fixed per gripper | Clearance |

Unobserved space is treated as occupied by default. This is conservative and configurable.

---

## 5. The loss: a tolerance-typed residual program

### 5.1 One term

Every requirement, whatever its source, is a tuple

$$c_i=\big(r_i,\ \sigma_i,\ \text{shape}_i,\ \text{role}_i,\ \text{scope}_i,\ \kappa_i,\ \text{src}_i\big)$$

| Field | Values | Meaning |
|---|---|---|
| $r_i$ | function from the primitive library | Residual in physical units (m or rad) |
| $\sigma_i$ | positive number, same units | Tolerance: how much error is acceptable |
| shape | `eq`, `ineq` | `eq` wants $r=0$; `ineq` wants $r\le0$ |
| role | `safety`, `require`, `prefer` | See below |
| scope | `terminal`, `path`, interval $[a,b]$, stage | When the term applies |
| $\kappa_i$ | $(0,1]$ | Confidence in the term |
| src | goal, perception, language, regularizer | For attribution in the certificate |

Roles:

- **`safety`**: hard and never relaxed. Collision, joint limits, workspace. Only the perception and robot compilers can create these.
- **`require`**: hard, but open to repair (§7.3). Goal pose and "must" language.
- **`prefer`**: soft. "Try to" language and regularizers. Never blocks acceptance.

The normalized violation at time sample $t_j$ is

$$v_{ij}=\frac{\lVert r_i(t_j;\theta)\rVert}{\sigma_i}\ \ (\texttt{eq}),\qquad v_{ij}=\frac{\max(0,\ r_i(t_j;\theta))}{\sigma_i}\ \ (\texttt{ineq})$$

$v\le1$ means "within tolerance" for every term from every source. For clearance, set the margin to the true safety distance plus $\sigma$, so that $v\le1$ still guarantees the safety distance.

### 5.2 Objective

At annealing level $l$ with tolerance inflation $s_l\ge1$ and penalty weight $\mu_l$:

$$J_l(\theta)=\sum_{i\in\text{prefer}}\frac{\kappa_i}{|T_i|}\sum_{j\in T_i}\rho_i\Big(\frac{v_{ij}^2}{s_l^2}\Big)\ +\sum_{i\in\text{safety}\,\cup\,\text{require}}\frac{1}{|T_i|}\sum_{j\in T_i}\psi\Big(\frac{u_{ij}}{s_l};\ \lambda_{ij},\ \mu_l\Big)$$

where $u_{ij}=r_i(t_j;\theta)/\sigma_i$ is the signed normalized residual and $T_i$ is the set of time samples in the term's scope.

- $\rho_i$ is the identity for regularizers and a robust kernel (Huber by default) for other `prefer` terms, so one unreachable preference cannot dominate.
- $\psi$ is the augmented-Lagrangian term with multiplier $\lambda$:

$$\psi_{\texttt{eq}}(u;\lambda,\mu)=\lambda^{\top}u+\tfrac{\mu}{2}\lVert u\rVert^2,\qquad \lambda\leftarrow\lambda+\mu\,u$$

$$\psi_{\texttt{ineq}}(u;\lambda,\mu)=\tfrac{1}{2\mu}\big(\max(0,\lambda+\mu u)^2-\lambda^2\big),\qquad \lambda\leftarrow\max(0,\lambda+\mu u)$$

Two consequences of this form:

- $J_l$ is a robustified sum of squares up to constants, so Gauss-Newton and Levenberg-Marquardt (with reweighting for the robust kernel) apply in addition to black-box and first-order solvers.
- Read as a probability, $\exp(-J)$ is a product of experts with $\sigma_i$ as standard deviations. Minimizing $J$ is MAP inference over trajectories.

### 5.3 Primitive library

All primitives take entities from the scene model, eef-frame axes, world axes and gravity. All are differentiable almost everywhere.

| Primitive | Shape | Residual $r$ |
|---|---|---|
| `reach_point(a, b, offset)` | eq | $a(t)-b-\text{offset}$ |
| `reach_rot(R_g)` | eq | $\mathrm{Log}(R_g^{\top}R(t))$ |
| `align_axis(u, v)` | eq | $\mathrm{atan2}(\lVert u\times v\rVert,\ u\cdot v)$ |
| `tilt_limit(u, v, max)` | ineq | $\angle(u,v)-\text{max}$ |
| `above(a, b, h)` | ineq | $(b_z+h)-a_z$ |
| `clearance(body, obstacle, margin)` | ineq | $\text{margin}+r_s-\Phi(x_s(t))$ per sphere |
| `distance_band(a, b, lo, hi)` | ineq | $\max(\text{lo}-d,\ d-\text{hi})$ with $d=\lVert a-b\rVert$ |
| `half_space(a, n, d)` | ineq | $d-n^{\top}a(t)$ |
| `approach_cone(axis, half_angle)` | ineq | $\angle(\dot p(t),\ \text{axis})-\text{half\_angle}$ |
| `value_map(V)` | eq | $V(p(t))$, for fuzzy spatial language |
| `smooth_pos`, `smooth_rot` | eq | $\ddot p(t)$, $\dot\omega(t)$ |
| `reach_limit` | ineq | workspace or IK residual, joint-limit hinge |

Program-level statements: `attached(obj)` for grasped objects, `allow_contact(a, b, scope)` to exempt intended contacts from clearance, and stage boundaries at gripper events.

### 5.4 Three compilers into one program

**Goal pose.** `reach_point` and `reach_rot` at `terminal` scope, role `require`, with default tolerances of 5 mm and 3°. For symmetric objects, take a soft minimum over the symmetry set.

**Perception.** `clearance` for the gripper and any attached object against the scene over `path` scope, role `safety`, plus `reach_limit`. It also supplies entities and covariances to the other compilers.

**Language.** The LLM/VLM receives the instruction, an image with numbered entities, the entity table and the primitive signatures as a JSON schema. It returns a program, not free-form code. It may only emit `require` and `prefer` terms.

Example for *"Put the mug on the shelf. Keep it upright and stay away from the laptop."* (units: m, rad):

```json
{
  "stages": [{
    "name": "carry_and_place",
    "attached": ["mug"],
    "allow_contact": [{"a": "mug", "b": "shelf", "scope": {"interval": [0.9, 1.0]}}],
    "terms": [
      {"id": "place", "prim": "reach_point",
       "args": {"a": "mug.base", "b": "shelf.top_center", "offset": [0, 0, 0.01]},
       "shape": "eq", "role": "require", "scope": "terminal", "sigma": 0.01},
      {"id": "upright", "prim": "tilt_limit",
       "args": {"u": "mug.axis", "v": "world.up", "max": 0.17},
       "shape": "ineq", "role": "require", "scope": "path", "sigma": 0.05},
      {"id": "laptop", "prim": "clearance",
       "args": {"body": "mug+gripper", "obstacle": "laptop", "margin": 0.10},
       "shape": "ineq", "role": "prefer", "scope": "path", "sigma": 0.03},
      {"id": "descend", "prim": "approach_cone",
       "args": {"axis": "-world.up", "half_angle": 0.35},
       "shape": "ineq", "role": "prefer", "scope": {"interval": [0.85, 1.0]}, "sigma": 0.10}
    ]
  }]
}
```

### 5.5 Uncertainty-aware fusion

Sources differ in reliability, and the right response depends on what the term does.

**Perception noise is handled asymmetrically.**

- A term that pulls toward a perceived target is loosened in the objective, so a noisy target is not over-fitted:

$$\tilde\sigma_i^2=\sigma_i^2+\lambda_{\max}(\Sigma_i)$$

- A term that pushes away from a perceived obstacle is tightened, so the path is more cautious where the scene is uncertain ($z_\delta=1.64$ for a 95% one-sided bound under a Gaussian assumption):

$$\tilde m_i=m_i+z_\delta\sqrt{\lambda_{\max}(\Sigma_i)}$$

- Acceptance still uses the original $\sigma_i$. When $\sqrt{\lambda_{\max}(\Sigma_i)}>\sigma_i$ the certificate flags the term as *not certifiable with this perception*, which is a cue to take a closer view.

**Language confidence comes from self-consistency.** Compile the instruction $n$ times (start with $n=5$). Match terms across compiles by primitive, arguments and scope.

- $\kappa_i$ is the fraction of compiles that contain the term.
- Numeric parameters take the median. The square of their spread (scaled MAD) is added to $\tilde\sigma_i^2$.
- A `require` term with $\kappa_i<0.4$ is demoted to `prefer`.

---

## 6. Search

### 6.1 Seeding

1. **Terminal candidates.** If no goal pose is given, solve the 6-D problem for the end pose alone using only terminal-scope terms, clearance at that pose and reachability. Use a few hundred random starts around the referenced entities, refine each, and keep the best $G$ distinct poses.
2. **Path seeds.** For each terminal candidate, take the straight-line seed plus $H-1$ paths from A* on the coarse grid. After each path, penalize voxels within a radius of it and plan again, so later paths take different corridors. Fit each path with $K=4$ control points by least squares.

These seeds are spatially diverse. They are not guaranteed to be topologically distinct.

### 6.2 Coupled continuation

Four schedules move together across levels $l=0,\dots,L$:

| Level | Tolerance inflation $s_l$ | Scene blur $\beta_l$ | Control poses $K_l$ | Penalty $\mu_l$ |
|---|---|---|---|---|
| 0 | 8 | 8 cm | 4 | 1 |
| 1 | 4 | 4 cm | 5 | 4 |
| 2 | 2 | 2 cm | 7 | 16 |
| 3 | 1 | 0 (exact) | 11 | 64 |

Early levels see a smooth problem with wide basins and few parameters. Later levels see the true problem at full resolution. The trajectory carried into the next level is unchanged, because refinement of the spline is exact (§3.2); only the objective gets stricter.

### 6.3 Solver slots

Each level runs an **explore** solver, then an **exploit** solver, on every surviving seed. Both see the same $J_l$.

| Slot | Options |
|---|---|
| Explore | none, random shooting, CEM, MPPI, CMA-ES, SVGD, annealed diffusion-style sampling |
| Exploit | none, Adam, L-BFGS, Levenberg-Marquardt, SLSQP |
| Hard terms | augmented Lagrangian (default), fixed penalty, rejection |

Sampling noise is smooth in control-point space, with separate scales for position and rotation that shrink with $s_l$:

$$\theta_m=\theta+\epsilon_m,\qquad \epsilon_m\sim\mathcal{N}\big(0,\ \alpha_l^2\,(D^{\top}D+\varepsilon I)^{-1}\big)$$

where $D$ is the second-difference operator over control points. MPPI, for example, then updates $\theta\leftarrow\sum_m w_m\theta_m$ with $w_m\propto\exp\big(-(J_m-\min_m J_m)/\tau\big)$.

**Portfolio mode.** Treat each (seed, solver pair) as an arm. Give every arm a small budget of cost evaluations, keep the better half by current $J_l$, double the budget, repeat. This hedges against picking the wrong solver for a given scene and needs no training.

### 6.4 Algorithm

```
build scene model
program  ← compile(goal pose, perception, language × n)          # §5.4, §5.5
program  ← run_program_tests(program)                             # §7.1
for round in 0..R:
    goals ← terminal_candidates(program)                          # §6.1
    seeds ← path_seeds(goals)                                     # §6.1
    for level l in 0..L:
        for seed in seeds:
            seed ← explore(J_l, seed, budget_E)                   # §6.3
            seed ← exploit(J_l, seed, budget_F)
            update multipliers λ
        seeds ← keep_best_half(seeds)
        seeds ← refine_knots(seeds, K_{l+1})                      # exact
    for seed in seeds sorted by J_L:
        cert ← certify(seed)                                      # dense check + IK, §7.2
        if cert.accepted: return retime(seed), cert
    conflict ← diagnose(best seed)                                # §7.3
    program  ← repair(program, conflict)       # LLM or user; safety terms untouched
return failure, cert, conflict
```

Starting budgets: $G=2$, $H=4$, 256 samples per explore iteration, 10 explore and 20 exploit iterations per level, $R=2$.

For closed-loop use, rerun only the last level from the previous solution whenever perception updates.

---

## 7. Checking the loss and the result

A training-free method has no validation set, so the loss itself has to be tested.

### 7.1 Before the search: tests on the program

All tests are numeric, need no model call, and run in milliseconds.

| Test | What it does | Error it catches |
|---|---|---|
| Lint | Schema, entity references, scope ranges | Malformed output, hallucinated entities |
| Range | Tolerances inside sane bounds; violations on the seeds not astronomically large | Unit mistakes (cm for m, degrees for radians) |
| $\theta$-dependence | $\max_m\lVert\partial r_i/\partial\theta\rVert$ over random trajectories must be non-zero | A term between two static entities, or an object not marked `attached`: the trajectory cannot affect it |
| Invariance | Move scene and trajectory together by a random yaw and translation; residuals must not change | Hard-coded world coordinates |
| Start feasibility | A `path`-scope `require` term must hold at the fixed start pose | Wrong scope ("stay above 30 cm" when the start is at 10 cm) |

A failed test goes back to the language compiler with the specific finding, or to the user.

### 7.2 After the search: certificate

A trajectory is **accepted** when every `safety` and `require` term has $\max_j v_{ij}\le1$ on the dense time grid, IK succeeds along the path with continuous joint motion, and retiming meets the velocity and acceleration limits.

The certificate is returned either way:

| Term | Role | Source | Max violation | At $t$ | $\kappa$ | Certifiable |
|---|---|---|---|---|---|---|
| scene clearance | safety | perception | 0.00 | – | 1.0 | yes |
| place | require | language | 0.62 | 1.00 | 1.0 | yes |
| upright | require | language | 0.91 | 0.47 | 0.8 | yes |
| laptop | prefer | language | 1.80 | 0.55 | 0.6 | yes |

(Illustrative values.) Because every number is in tolerance units, the table is readable without knowing the underlying physical units.

### 7.3 Conflict diagnosis and repair

When no seed is accepted:

1. **Rank** the `require` terms by violation and multiplier size.
2. **Leave one out.** For each top-ranked term, demote it to `prefer` and rerun the last level from the current solution. If the rest become feasible, the term is part of the conflict.
3. **Quote the minimum relaxation.** The violation the demoted term ends at is the price of feasibility, in physical units: for example, "feasible if the tilt limit is relaxed from 10° to 23°".
4. **Repair.** The quote goes to the LLM or the user, who may relax, rescope or drop `require` and `prefer` terms. `safety` terms cannot be touched. Every repair is logged in the certificate.

An optional last check renders the path onto the camera image and asks a VLM whether each clause of the instruction is met.

---

## 8. Combinations to test

### 8.1 Factors

Defaults in bold.

| Factor | Levels |
|---|---|
| A. Rotation representation | waypoints + SLERP; **tangent-offset B-spline**; cumulative SO(3) B-spline |
| B. Language compiler | **primitive-library JSON**; free-form cost code; 3D value map |
| C. Fusion | hand-set weights; tolerance-normalized; **tolerance-normalized + uncertainty** |
| D. Seeding | straight line; random multi-start; **diverse A\*** |
| E. Explore | none; random shooting; CEM; **MPPI**; CMA-ES; SVGD |
| F. Exploit | none; Adam; **L-BFGS**; Levenberg-Marquardt; SLSQP |
| G. Continuation | off; tolerance only; scene only; resolution only; **all** |
| H. Hard terms | fixed penalty; **augmented Lagrangian**; rejection |
| I. Program checks | off; tests only; **tests + repair**; tests + repair + VLM check |

### 8.2 Staged plan

The full grid is too large, so vary one group at a time.

| Stage | Vary | Hold | Needs | Runs |
|---|---|---|---|---|
| 1. Solver grid | E × F (30 pairs) | defaults | Synthetic scenes, hand-written programs, no LLM | 30 × 6 scenes × 10 seeds |
| 2. Search structure | D, G, H | best two pairs from stage 1 | Same | about 45 settings |
| 3. Representation | A, and $K$ schedule | best from stage 2 | Same | about 10 settings |
| 4. Program checks | I | best from stage 3 | Correct programs with injected errors | 5 error types × 20 programs |
| 5. Language and fusion | B, C | best from stage 3 | LLM/VLM, simulated perception noise | about 9 settings |
| 6. Real perception | best overall vs. baselines | – | RGB-D, robot or simulator | – |

Stages 1 to 4 need no language model and isolate the search from the compiler. Compare solvers at equal numbers of cost evaluations and report wall time separately.

### 8.3 Scenes

| ID | Scene | Tests |
|---|---|---|
| S1 | Free-space reach | Precision, speed |
| S2 | Single obstacle | Basic avoidance |
| S3 | Narrow passage, shelf insertion | Local minima |
| S4 | Keep-upright carry through clutter | Coupled position and rotation constraints |
| S5 | Pick, pour, place | Multi-stage, attached objects |
| S6 | Contradictory instruction | Conflict diagnosis |

### 8.4 Metrics

- Success rate: accepted by the certificate.
- Maximum normalized violation over hard terms.
- Soft cost at the solution.
- Cost evaluations and wall time to the first accepted trajectory.
- Variance across 10 or more random seeds.
- Number of distinct corridors among the final seeds.
- For program checks: detection rate per injected error type, and false alarms on correct programs.
- For the language compiler: term precision and recall against hand-written programs.

### 8.5 Hypotheses

| | Hypothesis | Evidence |
|---|---|---|
| H1 | Explore then exploit beats either alone at equal budget on S3 and S4. | Stage 1 |
| H2 | Continuation raises success on S3 and S4 and changes little on S1. | Stage 2 |
| H3 | Levenberg-Marquardt needs fewer evaluations than L-BFGS or Adam once inside a basin. | Stage 1 |
| H4 | Seeding choice affects success more than the choice of explore solver. | Stages 1 and 2 |
| H5 | The $\theta$-dependence and invariance tests catch most wrong-entity and wrong-frame errors before any search. | Stage 4 |
| H6 | Asymmetric uncertainty handling cuts collisions under perception noise at a small cost in goal accuracy. | Stage 5 |

### 8.6 Baselines

- Straight line plus IK.
- A sampling-based geometric planner (RRT-Connect or RRT*) with the goal pose only.
- cuRobo-style multi-seed particle plus L-BFGS optimization.
- VoxPoser-style value map with greedy search.
- ReKep-style keypoint constraints with SciPy solvers.

---

## 9. What is new and what is not

**Not new.** Each of these is established, and TRACE builds on them:

| Idea | Prior work |
|---|---|
| A VLM writes constraints or costs; an optimizer finds eef poses; no training | ReKep, VoxPoser, OmniManip, EmboAlign |
| An LLM/VLM designs the cost for a sampling-based MPC, in some cases from a fixed residual library and with revision from feedback | Language to Rewards, CoRAL, Objectives as Policies |
| Language to a formal temporal specification with automatic checking | AutoTAMP |
| Many parallel seeds, particle search then L-BFGS | cuRobo |
| Training-free annealed sampling for trajectory optimization | Model-Based Diffusion, DIAL-MPC |
| Coarse-to-fine relaxation of a non-convex robust objective | Graduated non-convexity |
| Chance-constrained clearance under perception noise | Dawson et al. |
| VLM visual self-check of a plan | OmniManip |

**Candidate contributions.** These are where TRACE appears to differ, based on a short literature check rather than a full review. Each needs a dedicated search before being claimed in a paper.

1. **Tolerance as the common unit.** Goal pose, perception and language all compile to (residual, tolerance, role). The cost-library methods above combine terms through weights, with thresholds used for stage completion. Tolerances make the fused loss, the hard/soft split and the acceptance test share one scale, and make every solver family applicable to the same objective.
2. **Asymmetric uncertainty fusion with self-consistency confidence.** Perception noise loosens attracting terms and tightens repelling ones; language confidence comes from repeated compilation. Chance-constrained clearance is established; the loosen/tighten split applied to model-compiled task terms did not turn up in the check.
3. **Coupled continuation with exact refinement.** Tolerance, scene blur, spline resolution and penalty are tightened together, and the tangent-offset SE(3) spline makes level changes lossless.
4. **Numeric tests on the cost program before search.** $\theta$-dependence, invariance, range and start-feasibility tests. The revision loops found in the check rely on the model reviewing its own program, on rollout or interaction feedback, or on a rendered visual check.
5. **Conflict set and minimum-relaxation quote.** Leave-one-out re-solves identify which requirements clash and quote the smallest relaxation in physical units.
6. **Solver portfolio with successive halving** over (seed, solver) arms.

**Scope difference from the closest recent work.** CoRAL and Objectives as Policies target contact-rich control: they run sampling-based MPC with a physics model in the loop. TRACE searches a kinematic eef path with no simulator. That makes it cheaper and gradient-friendly, and it cannot do what they do with contact.

---

## 10. Limitations

- **No contact dynamics.** Pushing, sliding and force-critical tasks are out of scope without a simulator in the loop.
- **Rigid-grasp assumption.** Slip and in-hand motion break the attached-object model.
- **Systematic compiler errors.** Self-consistency catches unstable outputs, not a mistake the model makes every time. The program tests catch only the error classes listed in §7.1.
- **Tolerances are model-chosen.** A wrong $\sigma$ from the LLM passes every test if it is inside the sane range.
- **Perception quality bounds everything.** Occlusion from a single view leaves the SDF incomplete.
- **Eef space only.** Arm-level collisions and redundancy are checked at certification, not optimized, unless an IK residual is added to the loop.
- **Seeds are not topologically complete.** A needed corridor can be missed.

---

## 11. Build order

1. Spline trajectory with batched evaluation and knot refinement.
2. Analytic SDF scenes (boxes, spheres), sphere gripper, and the primitives `reach_point`, `reach_rot`, `clearance`, `tilt_limit`, `smooth_*`.
3. Objective with roles, tolerances and augmented Lagrangian; written in PyTorch or JAX so one function serves all solvers.
4. Solvers behind one interface, `solve(J, theta0, budget) -> theta`; run stage 1.
5. Seeding and continuation; run stages 2 and 3.
6. Program tests and conflict diagnosis; run stage 4.
7. Language compiler and perception front end; run stages 5 and 6.

---

## References

Checked online for this note:

- Huang et al., *ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation*, 2024. [arXiv:2409.01652](https://arxiv.org/abs/2409.01652)
- Pan et al., *OmniManip: Towards General Robotic Manipulation via Object-Centric Interaction Primitives as Spatial Constraints*, CVPR 2025. [arXiv:2501.03841](https://arxiv.org/abs/2501.03841)
- Zhang et al., *EmboAlign: Aligning Video Generation with Compositional Constraints for Zero-Shot Manipulation*, 2026. [arXiv:2603.05757](https://arxiv.org/abs/2603.05757)
- Cicek, Er, Oguz, *CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation*, RSS 2026. [Proceedings](https://www.roboticsproceedings.org/rss22/p054.html)
- Lu et al., *Objectives as Policies: Vision-Language Model Cost Programs for Contact-Rich Manipulation*, Nov 2026. [arXiv:2611.18298](https://arxiv.org/abs/2611.18298) (abstract, introduction, related work and method read; experiments not read)
- Chen et al., *AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers*, 2023. [arXiv:2306.06531](https://arxiv.org/abs/2306.06531)
- Sundaralingam et al., *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*, 2023. [arXiv:2310.17274](https://arxiv.org/abs/2310.17274)
- Pan et al., *Model-Based Diffusion for Trajectory Optimization*, 2024. [arXiv:2407.01573](https://arxiv.org/abs/2407.01573)
- Xue et al., *Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing* (DIAL-MPC), 2024. [arXiv:2409.15610](https://arxiv.org/abs/2409.15610)
- Dawson et al., *Chance-Constrained Trajectory Optimization for High-DOF Robots in Uncertain Environments*, 2023. [arXiv:2302.00122](https://arxiv.org/abs/2302.00122)

Cited from memory, not re-checked:

- Huang et al., *VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models*, CoRL 2023.
- Yu et al., *Language to Rewards for Robotic Skill Synthesis*, CoRL 2023.
- Yang, Antonante, Tzoumas, Carlone, *Graduated Non-Convexity for Robust Spatial Perception*, RA-L 2020.
- Classical trajectory optimization and sampling: CHOMP, STOMP, TrajOpt, GPMP2, CEM, MPPI, CMA-ES, SVGD.
