# General Contact Stage Policy Plan (`agentic_trace` — Rev 3)

**Date:** 2026-11-28  
**Target Path:** Validate in **Panda MuJoCo (`robosuite 1.4.1`)** $\to$ Deploy on **standard 6-DoF AgileX PiPER-X** (zero wrist F/T dependency).  
**Minimal Viable Approach (MVA):** Build **one single-stage `UniversalContactPolicy` network and runtime interface once** covering all **5 contact stage families** (`pick`, `place / insert`, `push / pull`, `turn`, `planar_push`), enforce **5 hard bug-prevention locks**, validate causal behavior on a **2-hour dual-angle + slip ignition test**, and then flip the config switch to **full 5-family generalist training**.

---

## 1. Grounded Facts & Hardware Boundary

### 1.1 Historical Denominators & Codebase Traps Fixed by Design

1. **Both Entry and Execution Matter:** In `v2` (`115` failed stages, `CONTACT_STAGE_PLAN.md:38`), `50` failed at the hand-off, `26` on corrupted object models, and `~14` in contact motion. In `G0` (`eval_ucs_gate0.py:361`, `gate0_results.json:31`), `29/30 (96.7%)` was conditional on `WholeStrokeSampler` finding a candidate (`30/60` among layouts where at least one sampler succeeded; layouts where both failed were skipped). Every new evaluation fixes $N$ **before** any sampler runs.
2. **Four Silent Code Bugs Eliminated Before Training:**
   - **Hidden `wrench()` leak in `TRACE` guard:** `tracemotion.py:267` calls `guard.py:83`, which reads wrist force `F` (`push > self.f_thr`). Sensorless mode disables that force clause (keeping `ContactDetector`'s pure kinematic residual `x > strong * thr`) and makes `robot.wrench()` raise `RuntimeError` on any policy-side call.
   - **Missing held-payload collisions:** `ucs_env.py:603` (`r1 ^ r2`) missed held-cup-vs-ceiling collisions; contact scanning now unions `is_robot | is_held_payload` against non-allowed geoms.
   - **Mismatched reward/eval thresholds & missing slip labels:** `ucs_env.py:589` rewarded `0.80 * true_span` (`60 deg` on a `75 deg` door), and `ucs_env.py:693` only emitted `slipped_or_missed` on dropped free payloads (never on door/drawer handle slips or air closes). Both are unified to exact relative-object predicates.
   - **Hand-travel false completion & premature benchmark exit:** Hand travel (`45.1 cm` while the microwave door barely moved in `KITCHEN_SCENE7` `init1`) never declares `reached_goal`; and `run_episode.py:306` logs `first_libero_done_step` instead of raising `GoalReached` mid-air so release and `10`-step settle are always tested.

### 1.2 Standard PiPER-X vs. Panda Simulation Contract

| Dimension | Panda Sim (`robosuite 1.4.1`) | Standard AgileX PiPER-X | Unified Design Rule |
|---|---|---|---|
| **Kinematics** | 7-DoF (`855 mm` reach, 1-D null space) | **6-DoF** (`669.6 mm` reach, **0-D null space**, `J4/J5 +-90 deg`) | No 7-D joint vectors in shared policy contract; kinematic health passed as embodiment-agnostic scalars. |
| **Wrist F/T vs. Joint Torque Residual** | Has ideal wrist F/T (`sim_iface.py:128`, **forbidden**) AND joint actuator torques `d.qfrc_actuator` | **No wrist F/T**; `pyAgxArm` `get_motor_states(1..6)` provides current-converted joint `torque` ($\tau_{\text{meas}}$) | **Lock out wrist `wrench()` (`RuntimeError`), KEEP joint-torque residual $\Delta\tau = \tau_{\text{meas}} - \hat{\tau}_{\text{free}}(q,\dot{q})$** (tared at hand-off, projected via damped Jacobian + link-residual norm, with reversal stiction noise in sim). |
| **Transfer Path** | `500 Hz` `OSC_POSE` (`20 Hz` outer goal) | `pyAgxArm` `move_mit` (`kp, kd, t_ff`) / `move_p` | Validate generalist architecture in Panda sim; calibrate real PiPER-X force resolution (N) across postures and retrain native 6-DoF weights on `piper_x_description` URDF. |

---

## 2. Unified Architecture: One Policy Across All 5 Contact Stage Families

No two-stage policy split (`Arm C` deferred unless single-stage saturates), no hand-written `3 mm / 1.8 N` phase-switch threshold, and no per-object `if/else` rules (`p92` handle slicing, `CLOSE_WINDOW`, `part = parent`, `tilts_deg`, and `slot_z` are retired on `--placement gcp`).

### 2.1 How One `ContactContract` Covers Every Contact Stage

Every stage specifies **target vs. static `reference_entity`** (`p*` is never conflated between contact site and destination):

| Stage Family | Examples | `payload_pts` (`128` pts) | `p_contact_hint` (Gaussian in scene cloud) | Goal in Static `reference_entity` Frame $R_{\text{ref}}$ | `want_held_end` |
|---|---|---|---|---|---|
| **1. `pick`** | Thin book, mug handle, bowl, bottle | All `0` (empty hand) | Graspable region on target object | Target object lifted `+0.06 m` above support (`Δp_z >= +0.06 m`) | `+1.0` |
| **2. `place / insert`** | Microwave side cavity (`t09`), bottom drawer (`t03`), caddy slot, stove (`t08`) | **Frozen held-object cloud in $E_t$** | `0` (already holding payload) | Held object resting stably inside `p_goal_region` on `reference_entity` | `-1.0` |
| **3. `push / pull` part** | Close/open low & top drawers, open hinged door `40 deg` or `75 deg` | All `0` | Handle bar or push face on part | Part displaced by `target_delta` (`m` or `rad`, **final opening relative to closed `reference_entity` or signed `Δ`**) | `-1.0` |
| **4. `turn` part** | Stove knob (`t02`), offset faucet lever | All `0` | Knob / lever body | Part rotated by signed `target_delta` (`rad`) about mount relative to `reference_entity` | `-1.0` |
| **5. `planar_push`** | Push flat plate to front of stove | All `0` | Rear rim/face of object | Object centroid translated into `p_goal_region` on support | `-1.0` |

### 2.2 `UniversalContactPolicy` (`~0.9M` Params, Fixed Normalization Constants)

Runs uninterrupted at `20 Hz` from `TRACE`'s `<= 8 cm` hand-off pose until stage stop.

- **Inputs (all expressed in live EEF frame $E_t$, while goal progress is tracked against static $R_{\text{ref}}$):**
  1. **Dual-Group Local PointNet (`96x96` wrist + head depth in `[-0.22, +0.22] m` around fingertips):**
     - `scene_pts` (`384 x 5`): `[x, y, z, contact_heat, is_wrist]`, where `contact_heat = exp(-||p - p_contact_hint||^2 / (2 * 0.03^2))` (no `SAM 3` mask needed at `20 Hz`; zeroed when `holding_now = 1`);
     - `payload_pts` (`128 x 4`): `[x, y, z, valid_mask]` of the frozen held object transformed by FK into $E_t$ (`0` when empty-handed).
  2. **Explicit Goal & Kinematic Header (`16-D`, directly visible to Actor):**
     - `goal_dir_eef (3)`: unit desired target motion vector transformed from $R_{\text{ref}}$ into $E_t$;
     - `goal_axis_eef (3)`: unit rotation/reference axis in $E_t$;
     - `rem_trans_m (1)`, `rem_rot_rad (1)`: **remaining** linear (`m`) and angular (`rad`) displacement to goal measured relative to $R_{\text{ref}}$ (**persisted across retries** so partial progress is never lost when $E_0$ resets);
     - `p_goal_rel_eef (3)`: vector from current payload/target center to `p_goal_region` in $E_t$ (`0` for pure joint goals);
     - `want_held_end (1)` (`+1` hold / `-1` release), `holding_now (1)` (`0` or `1`), `time_left (1)`;
     - `min_joint_margin (1)`, `manip_pos (1)`, `manip_ori (1)`.
  3. **12-Step Proprioceptive, Compliance & Joint-Torque Residual Stack (`12 x 36` in attempt frame $E_0$, zero separate estimator net):**
     - Kinematics & compliance: `p_rel (3)`, `R_rel_6d (6)`, `v_rel (3)`, `w_rel (3)`, **`e_track_pos = p_des - p_meas (3)`**, **`e_track_rot = Log(R_des R_meas^T) (3)`**, `grip_open (1)`, `grip_rate (1)`, `prev_action (8)`;
     - **Joint-current torque residual channel (`8-D`, from $J(q)^\top \hat{w} \approx \tau_{\text{meas}} - \hat{\tau}_{\text{free}}$, tared by $\Delta\tau_0$ at hand-off to cancel static tool/payload gravity error):**
       1. `w_hat_dls (6)`: singularity-safe Damped Least-Squares wrench in $E_t$, $\hat{w}_{\text{ext}} = \mathrm{Ad}_{E_t}^\top (J J^\top + \lambda^2 I_6)^{-1} J (\Delta\tau - \Delta\tau_0)$ (fixed `6-D` across 7-DoF Panda & 6-DoF PiPER-X; decays smoothly near `J5 -> 0`);
       2. `tau_res_max (1)` = $\max_j |(\Delta\tau_j - \Delta\tau_{0,j}) / \tau_{\max,j}|$ and **`tau_res_link (1)`** = $\|(\Delta\tau - \Delta\tau_0) - J^\top \hat{w}_{\text{ext}}\|_2$ — lights up specifically when an **upstream forearm/wrist link hits an upper drawer** (where a wrist F/T sensor would read zero!), letting the policy pair "effort rose" with visual depth ("did the target drawer actually move?"). *(On native 6-DoF PiPER-X export, raw signed `6-D` $\Delta\tau / \tau_{\max}$ is also directly attachable.)*

- **Continuous Action (`8-D` in `[-1, 1]`, Zero `done` Action):**
  - `a[0:3]`: 3D position increment in $E_t$ (`+-8 mm/step`);
  - `a[3:6]`: 3D orientation increment in $E_t$ (`+-4.5 deg/step`);
  - `a[6]`: compliance scale $\alpha \in [0.30, 1.00]$ setting `Leaky-PLAI` lead cap $e_{\max}(\alpha) \in [4, 14]\text{ mm}$ and contact stiffness;
  - `a[7]`: **continuous target gripper width** mapped to `[0.0, 0.08] m` — active at every step so the single policy naturally learns pre-shaping, closing, **re-opening & re-grasping after a slip or air-close**, wrist re-indexing on `J6`-limited knobs, and clearance-safe partial opening inside narrow cavities.

### 2.3 `20 Hz / 500 Hz` Compliant Execution, Joint-Residual Simulation & Outcome Verification

1. **Entry (`MultiHypothesisEntrySampler` -> `TRACE`):** Samples `<= 8 cm` spherical-shell poses + $180^\circ$ twins keeping $p_{\text{contact}}^\star$ in wrist view; requires collision-free IK over the `<= 8 cm` approach + `4 cm` probe AND feasibility of $\ge 1$ hypothesis in $\mathcal{H} = \{\text{prismatic}, \text{hinge-L}, \text{hinge-R}, \text{hinge-H}\}$ (or insertion corridor for `place`). Detaches `target_entity`'s `t = 0` static voxels from `StepGuard` during contact.
2. **Control Core & Realistic $\Delta\tau$ Corruption:** `20 Hz` Leaky-PLAI integrates $(\Delta p, \Delta R)$ on desired pose over the `500 Hz` inner loop and clamps $\|p_{\text{des}} - p_{\text{meas}}\| \le e_{\max}(\alpha)$. In simulation, $\Delta\tau = \tau_{\text{act}} - \hat{\tau}_{\text{free}}(q, \dot{q})$ is injected with **velocity-reversal Coulomb stiction jumps ($\mathrm{sign}(\dot{q})$ hysteresis) + current noise** so the policy never relies on clean sub-Newton simulation torques and learns to fuse effort residuals with visual target motion.
3. **Stop Trigger vs. Completion Verdict (Strictly Separated):**
   - **Motion burst ends** on time budget (`6.5 s`) OR steady stall ((`||e_track|| >= 0.8 e_max` or elevated `tau_res_max`) with `||v_meas|| < 4 mm/s` for `6` steps) OR post-release retreat. **Hand stall never declares `reached_goal`.**
   - **Outcome measurement (in static $R_{\text{ref}}$):** Compares before/after `target_entity` pose relative to `reference_entity` from `HEAD_CAMERA` + `WRIST_CAMERA` depth registration (if arm occludes target, retreats clear and captures one fresh view first -> `uncertain_occluded` resolved).
   - **Transitions:** `reached_goal` (within `goal_spec.tolerance` AND `held` matches `want_held_end`) $\to$ update `ObjectPermanenceStore` & finish stage; `progress_short` $\to$ update `rem_trans_m / rem_rot_rad` in $R_{\text{ref}}$ and continue in place (`S1` `done_effect` cannot override); `missed_or_slipped` $\to$ next candidate; `blocked_external` $\to$ guarded back-off, mark contact voxel, try next entry direction (escalate to `S1` after `3` blocks).

---

## 3. Training & Verification Sequence (Config Switch, Zero Architectural Rewrite)

### Step 0: Bug Locks, Label Audit & Full-Loop Throughput Check (`~0.5 day`)
1. Implement wrist `wrench()` runtime assertion lock, joint-torque residual $\Delta\tau$ observer (with hand-off tare + reversal stiction randomization), `is_robot | is_held_payload` penetration scanner, exact `1.0x` goal predicates (`ucs_env.py`), full-state Staged Reset (`qpos, qvel, p_des, R_des, grip, 12-step hist, Δτ_0, goal_spec`), and `run_episode.py` post-done settle logging.
2. **Offline Label Sanity Check (`50` scripted rollouts):** Verify ground-truth outcome labeler cleanly separates `{firm handle grasp, single-finger graze, cabinet-frame pinch, push-face contact, mid-pull slip, air close}`.
3. **Visual PPO Throughput Benchmark:** Measure end-to-end steps/s (`FastSimRobot` + `96x96` dual-camera EGL depth with `D9` noise + `512`-pt cloud + PPO update) on GPUs `0–3` + `96–128` workers to lock exact wall-clock step budgets.

### Step 1A: Ignition Smoke Gate — Dual-Angle Door + Slip + Protruding Drawer (`~2 hours`)
Run `train_gcp.py --preset smoke_door_drawer` (same `UniversalContactPolicy` code, activating only procedural hinged doors and multi-drawer stacks):
- **Test 1 (Goal Conditioning):** On the **exact same door and start snapshot**, command `target_delta = 40 deg` vs. `75 deg` (`+-5 deg` tolerance). Policy must stop near `40 deg` when asked for `40 deg` and pull past `75 deg` when asked for `75 deg` (`>= 85%` on each).
- **Test 2 (Autonomous Slip Recovery):** Inject a forced gripper release/perturbation at `15 deg` opening on `50` trials; verify single-stage policy re-approaches, re-grasps, and completes to `75 deg` (`>= 60%` recovery) while preserving progress in $R_{\text{ref}}$.
- **Test 3 (Neighbor Clearance):** Close bottom drawer beneath an open upper drawer (`>= 85%` unconditional, `P95` penetration `< 1.0 mm`).

### Step 1B: Full 5-Family Generalist Training & Unseen Simulator Gate (`~1.5–2 days`)
Flip config to `train_gcp.py --preset all_5_families` (`pick`, `place/insert`, `push/pull`, `turn`, `planar_push` on `100%` procedural assets; **zero LIBERO fixtures in training**; Asymmetric PPO + `50%` Staged Reset Buffer + SAPU `< 1.0 mm` return weighting; checkpoint selection on procedural `VAL_SEED` only).
- **One-Shot Unseen Simulator Gate (`N = 400` pre-drawn starts per family on unseen LIBERO fixtures `short_cabinet`, `short_fridge`, `slide_cabinet`, `window`, `faucet`, `basin_faucet` + unseen arch cavities & overhang bins; zero starts skipped):**
  - Report full 4-stage funnel (`Entry Coverage`, `Conditional Success`, `Unconditional Success` with `95% CI`, `P95` robot+payload penetration, and `> 1.0 mm` trial fraction);
  - Unconditional paired comparison against `BaseController` on identical starts: `UniversalContactPolicy` achieves **`>= 80%` unconditional (`>= 85%` conditional)** on `pick`, `push/pull`, `planar_push`, and `place/insert` (`>= 70%` on unseen offset `faucet` levers), beating `BaseController` by **`>= +15` pts** on hinged doors and constrained cavity insertions;
  - Goal sign-flip / shuffled-goal falsification control **`<= 2%`**.

### Step 2: End-to-End Harness Integration, Regression & Blind Evaluation (`~1–2 days`)
Wire frozen `UniversalContactPolicy` into `agentic_trace` (`--placement gcp`, replacing `grasp_object`, `place_object_v2`, `_actuate`, and adding `planar_push`, with `wrench()` lock active):
1. **Development Regression Set (previously inspected tasks — used for integration verification):**
   - `17`-episode dev coverage matrix (including `KITCHEN_SCENE7` `init 0, 1`, `open_bottom_drawer`, `push_the_plate`, `caddy` thin book, `open_top_drawer + bowl`) + `t03` (`ep0–9`, reclassified as regression since its trajectories were previously analyzed). Target: **`>= 14/17`** on dev matrix and **`>= 8/10`** strict success (`<= 200 s`) on `t03`.
2. **Blind Generalization Set (never inspected during development, run once on frozen build + weight SHA256):**
   - Untouched Held-Out LIBERO-90 scenes (`33` remaining tasks $\times$ `3` inits) + remaining sealed `Libero10-Collision` tasks (`t00–t02, t04–t09` $\times$ `10` eps), reporting **both** stock LIBERO score and **Strict Post-Retreat Collision-Free Completion** (`<= 200 s`, `0` forbidden `> 1.0 mm` robot+payload contacts).
3. **PiPER-X Hardware Export (Post-Gate):** Measure real PiPER-X $\Delta\tau$ deadband & resolution (in Newtons under known push/pull loads across workspace postures, including near `J5 -> 0`), then load `piper_x_description` URDF + 6-DoF `MIT PD` model + calibrated stiction/compliance profile into `train_gcp.py --preset all_5_families --robot piper_x` to export native 6-DoF deployment weights.
