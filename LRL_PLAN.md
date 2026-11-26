# LRL: a learned contact layer for agentic_trace (implementation plan)

*2026-11-25. Plan only: nothing below is implemented. Base: the v2 path of the live tree (build 27c3f19c, the build
of the current v2 runs). Reader notes, probe scripts and their outputs: `runs/plan_lrl_20261125/`. Your decisions of
2026-11-25 are recorded in section 10 and worked into the text.*

`lrl` is a run-wide version selected like v2 (`--placement lrl`). I read the name as "lightweight RL".

Contents: 0 Status · 1 Summary · 2 Evidence · 3 Rules · 4 Stage flow · 5 Parts · 6 Training and data ·
7 Evaluation · 8 Milestones and gates · 9 Code map · 10 Decisions for you · 11 Risks and limits ·
12 Facts behind this plan

## 0. Status

**M0 done, 2026-11-25.** Built: `src/atrace/lrl_contract.py`, `src/atrace/lrl_skill.py`, `src/lrl_train/sim.py`,
`scenes.py`, `eval_stroke.py`, and their tests (26 pass). Report: `runs/lrl_m0_20261125/reference/REFERENCE.md`.

Gate G0, measured in the stage simulator on 25,749 trials (245 of 400 planned layouts; the measuring agent was cut
off by the usage limit):

- *Fidelity: yes.* On the white cabinet, today's stroke from `side` poses closes the bottom and middle drawers 0
  times in 150, with the hand body or wrist link 7 on the drawer above at the stop in 145 of them: the block of the
  recorded t03 runs.
- *Need: yes.* Today's stroke, first family with a pose, full error: white cabinet 72 %, wooden cabinet 71 %,
  microwave door 23 %, generated door 35 %. So M1 goes ahead.
- **On drawers the hand pose decides.** `tip` poses close 95 to 100 % on the task cabinets, `side` and `tip_steep`
  17 to 33 %. The press search tries `tip_lean` (81 to 92 %) last. Entry order is therefore the first thing lrl
  changes in the harness (M1 step 5), before any trained policy.
- **On doors the motion law decides.** Microwave: today's stroke 17 to 25 %, the base law 43 to 61 %. This is where
  the trained policy has the most to gain.
- Found and fixed: the base law had no way in from the standoff to the touch. `Skill.run(..., way_in=...)` now
  approaches along the entry's axis until the wrist feels the part, then re-anchors the contact frame.

**M1 first half done, 2026-11-25.** Reports: `runs/lrl_m1_20261125/harness/NOTES.md` (first harness pass),
`harness2/HANDOFF_ERROR.md` and `harness2/NOTES.md` (second pass, 33 dev episodes), `train/REPORT.md` (training).

- *Harness.* `--placement lrl` with `lrl_push` (ranked press candidates, arrival on early contact, grip-and-push,
  skill loop with the base law, no policy yet), the stock-task runner (`run_episode.py --suite --suite-root --init
  --stage-truth`, `suites/`, `scripts/stage_truth.py`, `scripts/handoff_error.py`). t03 ep0-2: 1 of 3 on the first
  build, ep000 then succeeded after the fixes. 33 dev episodes (11 part-moving tasks x 3): 24 LIBERO successes;
  first attempt of the actuation reached the goal by truth in 13 of 32 (push 8/14, pull 4/15, turn 1/3).
- *Hand-off error (measured).* Direction is bimodal: 34 of 52 attempts within 7.3 degrees of the true axis (25
  reached the goal), 18 at 46-90 degrees (none reached): a wrong face for a push (upward face), the knob's side
  normal for a turn, one grip on the wrong drawer. Travel within 3 cm in 36 of 40. So the prior covers the good mode;
  the bad mode is a wrong face or axis, which no skill absorbs: it has to be another candidate.
- *Pulls and turns are still v2's code*, and they lose there: the grip search finds no hold on low handles (about
  100 s per search, its vetoes count the part's own fixture), and the turn axis comes from one face guess.
- *Training.* PPO in Tier A (`src/lrl_train/env.py`, `vec.py`, `ppo.py`, `train_move_part.py`, `eval_policy.py`),
  1.0-1.7 k steps/s on 96 processes, so at most 4 M steps per policy. M1 policy so far: `weights/lrl/
  move_part_push_20261125_c3fae1aa.pt` (run 2, 2.0 M steps). Full prior error, task fixtures: stroke 44 %,
  base_touch 51 %, policy 59 % (pose families pooled; measured family order: stroke 60 %, policy 78 %); microwave
  door 46 -> 66 %. Unseen fixtures: stroke 35 %, base_touch 40 %, policy 35 % (short fridge 16 -> 1 %: the policy
  pushes harder early and trips the 40 N stop on heavy or stiff parts; damping and mass were randomized only 0.5-2
  times). Gate G1 simulator side not met.

**M1 done, 2026-11-25 21:20 UTC; gate G1 fails on every criterion. Paused before M1E (your instruction).**
Reports: `runs/lrl_m1_20261125/train2/NOTES.md` (training, detector, simulator gate), `final/M1_RESULTS.md` and
`final/analysis.md` (51 harness episodes, policy on), `final_base/analysis.md` (the same 51 with `--lrl-parts
skill,detector`, added by the main session).

- *Simulator gate.* Policy `move_part_20261125_2fad24a7` (18.6 M steps) below the base law almost everywhere (full
  error, stroke / base / policy: push task fixtures 40 / 50 / 36 %, turn stove 59 / 93 / 63 %, unseen turn
  19 / 25 / 5 %; pull outside contacts 2 / 4 / 28 %): its mean residuals are large, biased and accumulate. Detector
  `c2801d14`: "reached" precision 97.6 % on validation (98 % needed), recall 57 %.
- *Harness* (build 9e240d3e, gpt-6.2-astra high, caps 600 s / 50 calls):

| set | policy + detector | base law + detector | first half (harness2: base-law push, v2 pull and turn, no detector) |
|---|---|---|---|
| dev, 33 | 18 / 33 | 20 / 33 | 24 / 33 |
| held-out, 15 | 9 / 15 | 11 / 15 | not run |
| t03 ep0-2 | 0 / 3 | 0 / 3 | 1 / 3 (ep000 later succeeded) |

- First attempt of the first actuation reached the goal (truth): dev 38 % / 42 % / 41 %, held-out 27 % / 53 %; the
  bar was 80 %.
- *Main causes.* (1) the policy (above). (2) The detector does not transfer from the stage simulator to the harness:
  of 23 dev runs that truly reached the goal (base-law set) it called 3 reached and 17 no_effect or no_contact, and
  a no_effect crosses off the whole face, so multi-stage tasks lost the face they had just completed (dev close top
  + bowl on top 3 -> 0 against the first half). (3) Candidate generation unchanged: no hold on low bar handles (open
  bottom drawer 0 / 3 in every configuration), the wrong-face mode (19 of 68 attempts beyond 45 degrees). (4) Goals
  from constants: the contract asks 15 cm or 90 degrees, LIBERO's microwave needs 74 degrees (door left at 54
  degrees, 0 / 3 everywhere). (5) t03: in the close stage the first no_effect on the true front removed it, then the
  side-face candidates were refused, until the 600 s cap.

**M1E started, 2026-11-25 22:50 UTC (your instruction "start the M1E").** Output root `runs/lrl_m1e_20261125/`.
First workflow, three tracks in parallel: `e0/` (E0, the base law switched off, M1's recipe and step count),
`e1/` (E1, the privileged-state teacher: `src/atrace/lrl_e2e.py`, new handle kinds, `teacher_env.py`, short runs,
then the long run) and `e2/` (E2 preparation: rendering throughput, the student's point-cloud input with the D9
sensor model, student network and DAgger loop, measured with a stand-in teacher). The student's DAgger training, E3
and gate G1E follow once a teacher is good. Decisions taken at the start (main session; you can change them):

- D9 as recommended (section 10).
- Goals are general, with no per-fixture constant: open = towards the open end of the joint's range, close = to the
  closed end, turn on / off = towards that end of a knob's range, an optional amount when S1 gives one. LIBERO's
  predicates are met on the way (M1's cause 4).
- Teacher, student and harness share one action space, one frame and one safety wrapper, in `src/atrace/lrl_e2e.py`.
- The teacher sees only geometry a depth camera could show (plus robot state and force); hidden physics goes to the
  critic, so that the student can copy what the teacher acts on.
- Starts include the wrong-face mode measured in M1 (a third of the hand-offs, 46-90 degrees off): the teacher must
  choose the right contact from wherever TRACE leaves the hand.

*State at 2026-11-26 00:20 UTC.* E2 preparation done (`e2/REPORT.md`): 96 px depth of both cameras costs under 1 ms
a frame, the physics step 12.5 ms; the whole DAgger loop gives about 700 labelled steps/s on 30 processes (5 M in
2 h, 20 M in 8 h); SAM3 re-run on the frames S1 saw in 99 dev episodes: open parts IoU median 0.91, closed parts
mostly the handle alone (31 of 54) or nothing (14, all 9 stove knobs). E0 at 8 M of 18.6 M steps learns far slower
than M1 with its base law. The teacher's seven short runs (stopped at about 1 M steps each) learned some closing
and no opening or turning; a scripted expert that knows the truth failed through the interface on doors and
generated knobs and drawers.

**Your decision A (2026-11-26 00:20 UTC, "A, follow your suggestion").** The teacher track was restarted
(workflow `lrl-m1e-teacher-a`) in this order: (1) interface acceptance: a privileged scripted expert
(`src/lrl_train/expert.py`, training time only) must succeed through `lrl_e2e` on every training kind (90 % from
grasped or near starts, 80 % from the start region), the interface is fixed where it is the cause; (2) behaviour
cloning of the expert's perturbed rollouts as the warm start, then PPO fine-tuning with a decaying pull towards
the clone (the deployed student stays learned; fine-tuning may leave the script where it can do better);
(3) runs judged at about 10 M steps, not 1 M; (4) the running pure-RL run s7 is kept as the "longer pure RL"
data point; (5) fronts also turned about 90 degrees from the robot (E2's request: the head camera saw backs);
then the long run. E0 runs to its end (about 02:10 UTC) and evaluates itself.

**E0 done, 2026-11-26 02:33 UTC (`e0/REPORT.md`): the base law helps in every mode.** Turning the base law off cost
success in every gate cell at the full prior error, after M1's 18.6 M transitions. Free policy against the base law:
push on the task fixtures 31 vs 50 %, pull on generated mechanisms 21 vs 54 %, turn on the task fixtures 17 vs 93 %.
The free policy is also at or below M1's residual policy in most cells. Its training curve was still rising at the
end (push / pull / turn 18 / 9 / 6 %). Without a motion prior, exploration is the bottleneck on this budget, which
supports the expert warm start of decision A.

**M1E paused, 2026-11-26 04:14 UTC (your instruction: "stop the current first, and let me rethink about this").**
The teacher workflow, its long run L4 (stopped at 7.4 M steps) and the watchdog are stopped; nothing is running.
Teacher state at the stop (`e1/NOTES.md`):

| item | result |
|---|---|
| scripted expert with the truth (accept/v9; 24 starts per kind and start type) | grasped start 78 %, near 73 %, start region 39 % (bars 90 / 90 / 80 %); most region failures: no feasible contact from the hand-off posture (joint limits or collisions; low parts near the robot, folded arm); worst on the microwave, doors without a handle to pinch, low bars |
| behaviour clone of the expert (bc2) | 22 %: close 31, open 5, turn on 33, turn off 23 % |
| pure RL from scratch (s7, 6.8 M steps) | about 10 % in training: close 29 %, open and turn 0-6 %, never left the easiest curriculum level |
| PPO fine-tuning (f1, f2, L3) | at or below the clone; without the pull towards the clone (L3) it fell to 13 % |
| long run L4 (clone anchor kept) | 19.5 % at 4.2 M (close 37, open 0, turn 8-10 %); outside contacts on task fixtures 34 % (clone 12 %) |

Kept for any next step: `src/atrace/lrl_e2e.py` (shared action space, frame A, start region, wrapper with the fixed
force cap), the E2E scenes (low bars, recesses, overhangs, fronts up to 90 degrees off the robot),
`teacher_env.py`, `expert.py`, `bc.py` with 971 k labelled expert steps (`e1/bc/data1`), the clone `e1/bc/bc2.pt`,
the student side (`render.py`, `student_obs.py`, `student.py`, `dagger.py`) and E0's report. The reading at the
stop: the expert caps the clone, the expert's failures from the start region are mostly kinematic (no feasible
contact from the hand-off posture), and opening is nearly absent from the expert data.

**M1 second half (decided 2026-11-25 from these results; no user decision needed, all within sections 3-6):**

1. One `move_part` network for push, pull and turn (mode as an input, 5.2). Training recipe changes: damping and mass
   randomized 0.25-6 times, heavy hinged doors among the generated mechanisms, a force penalty from about 15 N so
   that effort follows the felt resistance, white exploration noise (run 2), kinds sampled by low success, injected
   failures (wrong face or axis, blocked part, no contact, slip) in a share of episodes.
2. Outcome detector v0 from the same rollouts (5.3), calibrated; on the lrl path it replaces the in-place part
   reading and the verdict thresholds.
3. Candidates in the harness: every face hypothesis of a part for push and pull, every axis hypothesis for a turn;
   a candidate the detector calls `no_effect` or `no_contact` is marked and the next one is taken, with no model
   call. The grip search on the lrl path does not veto contact with the part's own fixture (5.1: `allowed`).
4. Pull and turn through the skill loop, with the policy and the detector loaded from `weights/lrl/`.
5. Evaluation: the 33 dev episodes again, the 15 held-out part-moving episodes, t03 ep0-2; gate G1; then the M1
   report and a pause (your instruction).

## 1. Summary

- **What LRL is.** It keeps v2's task program, grounding and TRACE transit, and rebuilds the contact stage (grasp,
  put down or insert, move a part) from six parts:

| Part | Kind | What it does | What it replaces in v2 |
|---|---|---|---|
| Skill policy | learned (residual RL) | Closes the loop inside a contact at 20 Hz from wrist force, finger gap and hand motion | Strokes and descents that end for good on the first stall or force spike |
| Outcome detector | learned (labels from simulation) | Says what a contact did: reached, short, no effect, no contact, touched something else; held or not; released or not | Hand-set thresholds (2 mm empty close, −6..+10 mm close window, 1 cm "moved", 2 cm "open") and v2's unconditional release |
| Candidate scorer | learned (labels from simulation) | Ranks grasp, put-down and press candidates by predicted success on the map the harness really has | Hand-weighted scores and chains of geometric vetoes on things that are meant to be touched |
| Contract hand-off | hand-written, exact | TRACE brings the hand to an entry pose it can certify; the skill owns the rest of the stage | Certified contact poses, pre-pose line checks, a new approach between strokes |
| Object store | hand-written, exact (a pretrained tracker is optional) | Keeps each object's identity and pose through grasp, carry and release | Re-segmenting from scratch; a map rebuilt from the last 8 frames |
| Facts, next candidate, choice | hand-written reporting; S1 decides | A failure is reported as measurements; S2 takes the next-ranked candidate; S1 picks among options S2 can execute | Cause-and-lever sentences, re-pointing after "could not reach", REPAIR restarting the program |

- **The rule that sorts them** (section 3). Code written by hand holds only what is exact: kinematics, robot limits,
  bookkeeping, interfaces. Anything that guesses a physical outcome or a perception error is learned in simulation or
  decided by S1.
- **Why** (section 2). In the v2 full test every episode that needed no repair succeeded (50 of 50) and episodes that
  needed one or two succeeded 15 times in 50. The failures are spread over many mechanisms, so no single skill fixes
  them, and adding geometric checks made things worse (unified 6/30 against v2 10/30 on the same episodes).
- **Training** uses no demonstrations. By your rule it may use LIBERO-90, LIBERO-goal, LIBERO-object and
  LIBERO-spatial, and never the Libero10-Collision test tasks. Two sets of layouts stay out as well: the 9 test
  scenes, including the 35 LIBERO-90 tasks located in them, and the 7 held-out scenes. LIBERO's fixtures and objects
  are placed in the remaining scenes and in new random layouts, next to generated mechanisms. Everything is driven
  through the harness's own robot interface, which runs at 14 ms per control step without cameras (measured).
- **Order (D2: pilot first; D11: M1E after M1).** M0 foundations and two measurements · M1 pilot: move a part
  (policy and detector) · M1E end-to-end contact policy for moving parts: the policy chooses the contact itself ·
  M2 facts, next candidate, S1 choice · M3 object store · M4 scorer and entry hand-off for put-down and grasp ·
  M5 place and insert skill · M6 full evaluation. Each learned part has a go / no-go gate measured on fixtures and
  tasks it never trained on.
- **First gate (G0).** Before any training: does today's stroke fail in the stage simulator the way it failed in
  the recorded t03 runs? If the stroke already works there, the pilot has little to gain and the plan moves on to
  M2–M4.
- **v2 is left aside for now** (your decision of 2026-11-25). No v2 run is planned; lrl is built and measured on
  its own, and the comparison with v2 comes when you ask for it.
- **Not in it.** No change to TRACE's planner, the S1 program schema, or the transit path. No real-robot run. No
  demonstrations, no VLA.
- **Decisions** (section 10). D1–D4, D8 and D11 are settled, D5 is dropped, D7 is put off with v2. D9 is taken as
  recommended at the start of M1E (you can still change it before the student is trained). D6 and D10 are open,
  needed by M4 and M2.

## 2. Evidence

| Finding | Numbers | Source |
|---|---|---|
| Recovery barely works | Episodes with 0 repairs 50/50; 1 repair 11/12; 2 repairs (the cap) 4/38, or 4/28 without t09 | `results.json` of the v2 full test (`runs/eval_fe630da9_v2_...`) |
| Failures are spread out | Failed episodes: t09 10 (side opening), t08 6 (4 start with "object lost after release"), t03 6, t06 5 (4 in episodes where the target is hidden), t04 4, t01 2, t00 1, t02 1 | same |
| Most failed stages fail before the contact motion | Of 115 failed stages, a refused or missing hand-off pose is the direct cause of 50 and the wrong object model of 26; about 14 are the contact motion itself (my tally) | CONTACT_STAGE_PLAN.md section 2 |
| More geometry did not help | unified (M1–M4) 6/30 against v2 10/30 on t03, t04, t09 ep0–9; t03 0/10 against 4/10 | batch d1337c7d against the v2 full test |
| S1 calls go into re-pointing | `place_pose` 311 of 604 calls; pointing again after "could not reach" failed 113 of 121 times | `results.json`; CONTACT_STAGE_PLAN.md |
| t03 under the current v2 fails only at the drawer close | v2, gpt-6.2-astra, max effort: 6/10; the bowl went in on the first `place_pose` in 10/10; all 4 failures are the close | runs `v2_27c3f19c_gpt62_max_..._t03_*` |
| What stops the close | In ep006 and ep008 the fingers are on the bottom drawer as planned, and each stalled stroke coincides with the hand body or wrist link 7 touching the **top** drawer (11, 9 and 18 contact onsets in the step ranges where strokes 2–4 of ep006 stalled). The hand then stands still pushing with a steady 17–20 N. The drawer joint has damping only (50 N·s/m), so it is not the drawer resisting, and the bowl touches the upper drawers only during the put-down | `collision.json`, `rec/telemetry.jsonl`, `white_cabinet.xml` |
| A stalled stroke is expensive | Each further press costs about 40 s: 15–30 s pose search (the grip search fails first every time), about 5 s hand-off, 3 s stroke, 5–7 s observation. ep004 made no stroke in its last 330 s | `events.jsonl` |
| The stroke has no feedback on force | It leads the hand 1.5 cm, re-aims along the hand's own displacement, and ends for good on a 3 mm stall or 40 N | [articulate.py:362-401](src/atrace/articulate.py#L362-L401) |
| Outcomes are read from thresholds or not at all | v2 opens the gripper whatever the descent returned; an unanswered look counts as success; hold is finger opening only | [harness.py:2300](src/atrace/harness.py#L2300), reader notes `stages.md` |

What the t03 finding means for the pilot: the missing piece is not push force. It is choosing and keeping a hand
pose whose body clears the fixture over the whole travel, and noticing from force and torque when it does not. That
is what the training scenes must contain (the white cabinet itself in other layouts, and generated drawer stacks
with protruding handles) and what the entry ranking must learn.

## 3. Rules

1. **The line.** Hand-written code may hold: kinematics, robot and controller limits, state bookkeeping, interfaces,
   candidate generators that only propose, and TRACE's guarantee for things that must not be touched. It may not
   hold: margins against things that are meant to be touched, fill rules, done thresholds, cause-to-fix tables.
   Those are learned, or S1 decides.
2. **Run-time inputs are what a real robot has**: joint and finger encoders, wrist force and torque, RGB-D from the
   head and wrist cameras with calibration. Simulator truth is used for rewards, labels and scoring only.
3. **General skills.** One policy per contact class, conditioned on the contract. No task, object or fixture name
   reaches a model or a constant.
4. **What may be trained on** (your rule of 2026-11-25 and the two limits you added to it).
   - Allowed: LIBERO-90, LIBERO-goal, LIBERO-object, LIBERO-spatial, and every LIBERO fixture and object as an asset.
   - Never: training or tuning on the Libero10-Collision test tasks.
   - Also left out: the layouts of the 9 test scenes, including the 35 LIBERO-90 tasks located in them, and the
     layouts of the 7 held-out scenes (7.1), which are for evaluation only.
   - A scene means a layout: which fixtures and objects stand where. The room arenas (kitchen table, living-room
     table, study table, floor) are shared by all scenes and are used.
   - For lrl this replaces the line on test scenes in DESIGN.md section 5. Evaluator outputs of finished test runs
     may be read, as before.
5. **Scoring is unchanged**: LIBERO's original success rule, your collision rule, the caps of 600 s control time and
   50 model calls, infrastructure errors rerun from the start.
6. **Checks.** Each milestone ends with the tests once, the named runs, and a short report: the numbers against the
   gate and what was not tested. The gates in section 8 are measurements a decision needs, not reviews.
7. **Constants ledger.** Every number on the lrl path carries one of four tags in its config dict: `R` robot or
   controller specification; `M` measured in the stage simulator or on dev runs, with the script; `C` an operating
   point read from a calibration curve on held-out generated data; `T` a training setting that only shapes weights.
   A number fitted to a test or held-out episode has no tag and is not allowed.

## 4. Stage flow in lrl

```
S1 program stage (transfer | actuate)                       schema unchanged
 └ candidates   generators propose: grasps; put-down spots x entry directions; part faces x hand poses
    └ scorer ranks them                                      learned (until M4: the generator's own order)
       └ contract: entry poses, allowed contacts, goal, limits
          └ TRACE certifies transit to one entry pose        unchanged; a refusal means "next candidate"
             └ skill: base motion + residual policy          learned; runs until done or a limit
                └ detector: what happened                    learned
                   ├ reached            -> object store update -> next stage
                   ├ not reached        -> facts recorded -> next candidate, no model call
                   └ list used up, or the state is unclear -> S1 `choose` among marked options
```

With a part switched off (`--lrl-parts`), lrl runs v2's own code for that step, so every part can be compared with
v2 inside one build.

## 5. Parts

### 5.1 Contract hand-off (`src/atrace/lrl_contract.py`)

`Contract` is the first data type for the "one record for every contact" of CONTACT_STAGE_PLAN.md 4.1, which exists
only as a table today.

| Field | Content |
|---|---|
| `kind` | `move_part`, `put`, `grasp` |
| `toucher` | closed fingertips, open pads, or the held object |
| `allowed` | ids of what may be touched: the part and its fixture, the destination, the object |
| `entries` | ordered entry poses, each with its approach axis, a standoff interval, the 180° twin flag and the candidate it came from |
| `goal` | `move_part`: push, pull or turn, the direction or axis believed, the travel wanted (from the depth measurement taken before). `put`: destination, relation, set down or drop. `grasp`: the object |
| `limits` | force, speed, a tube around the entry pose and the believed path, time |
| `done` | the detector classes that end the stage |

- **To TRACE** each entry is an `EndCandidate` with `approach` and `standoff=(s_min, s_max)`: the slide that exists
  today ([contract.py:153-177](src/traceplan/contract.py#L153-L177)). TRACE stops where its end test passes and an
  arm posture exists. Nothing past that point is certified, as today. M1 needs no TRACE change.
- **What goes on the harness side.** No veto of an entry for predicted contact with something in `allowed`: the
  press search's line checks and the rigid-hand prefilters against the target are not run on the lrl path.
- **`s_max` is measured.** It is the skill's capture range along the approach, read from the gate curves, tag `M`.
- **Things that must not be touched keep their geometric guarantee.** During the skill the per-step guard checks
  hand, arm and held object against observed geometry outside `allowed`, with the guard callback that exists.
- **The stage is not left between sub-strokes.** No new transit and no new pose search until the detector ends the
  stage or a limit is hit.

### 5.2 Skills (`src/atrace/lrl_skill.py`)

One loop for every contact, at the control rate:

1. read hand pose and velocity, wrist force and torque (tared at entry, low-passed over 3 steps), finger gap, the
   last command;
2. compute the base command from a minimal law for the kind;
3. add the policy's residual;
4. wrapper: clip to the contract's tube and speed; stop and back off on the force limit or a guard hit; stop on time;
5. `rb.step(R, p, grip)` and append the step to the trace.

| Kind | Base law (no stall rule, no re-aim rule, no re-seat rule: the policy and detector replace those) |
|---|---|
| `move_part`, push | closed fingers; lead the hand 1.5 cm along the believed direction |
| `move_part`, pull | close on the handle; lead 1.5 cm along the believed direction |
| `move_part`, turn | close on the part; turn about the believed axis at 30°/s |
| `put` | move along the entry axis; open at contact; back out along the entry axis |
| `grasp` | move along the approach to the planned depth; close; lift 10 cm |

- **Residual action per step**, all bounded, in the contact frame:
  - position ±1 cm and orientation ±3°;
  - effort 0.3–2 times the lead (sets the push force through the controller's stiffness);
  - steer: turns the base direction by at most 5° per step, accumulated (a door's arc, a misjudged axis);
  - gripper keep, open or close (not for push);
  - done flag.
- **Observation**: hand pose relative to the entry pose, linear and angular velocity, wrench in the hand frame,
  finger gap and its rate, commanded minus measured pose, and the goal (mode, believed direction in the hand frame,
  travel wanted, travel so far, time left), stacked over 8 steps. No vision in M1.
- **Network**: MLP 3×256 per kind, the mode as an input, about 0.3 M parameters.
- **Real robot.** The policy sets equilibrium poses at 20 Hz on top of an impedance loop. Here that loop is
  robosuite's OSC at kp 150; on a Franka it would be the Cartesian impedance controller. Not validated in this plan.

### 5.3 Outcome detector (`src/atrace/lrl_models.py`)

- **Input**: the skill trace of the contact (up to 6 s at 20 Hz: pose in the contact frame, velocity, wrench, finger
  gap, command error) and the contract goal. From M4 also two wrist depth crops, before and after.
- **Output**: class probabilities (reached, short, no effect, no contact, touched something outside `allowed`, lost
  grip) and the travel done with its spread. For `grasp`: held or not. For `put`: released and clear, or still on
  the hand.
- **Model**: 4 temporal convolution layers and pooling, about 0.2 M parameters, calibrated on a held-out split.
- **Labels**: simulator truth at the end of every training rollout (joint travel, object pose, contact pairs),
  including injected failures: a blocked part, a missing contact, the wrong face, slip.
- **Use**: ends the skill together with the policy's done flag (class by argmax), fills the fact report, triggers
  the next candidate. S1's `effect_check` stays for stages whose effect S1 named (the stove is on).
- **Replaces**: the verdict thresholds of [harness.py:3739-3763](src/atrace/harness.py#L3739-L3763), the 2 mm and
  close-window tests of the grasp, v2's unconditional release.

### 5.4 Candidate scorer (`src/atrace/lrl_models.py`)

- **Input**: a 40³ voxel crop at 5 mm around the candidate's contact point, in the candidate's frame. Channels:
  target, destination, other observed geometry, unknown, the toucher at the entry pose, the toucher at the contact
  pose. Plus a short vector: approach direction relative to gravity, gripper width or held-object size, distance to
  the nearest joint limit at the entry posture, release height.
- **Output**: P(the stage succeeds with this candidate) and P(something outside `allowed` is touched).
- **Model**: 3D CNN, 4 blocks, about 1 M parameters, one head per kind.
- **Labels**: each sampled candidate is executed in simulation by the real lrl path, on a map built by the
  harness's own perception from rendered depth. Simulator truth gives success and contacts.
- **Use**: orders the candidates. The first one TRACE can certify is tried; a failed one is marked and the next is
  taken. A candidate is never dropped for predicted contact with something in `allowed`.
- **Generators stay hand-written and only propose**: the antipodal grasp generator; put-down spots around S1's
  pixel with entry directions over the upper hemisphere and the destination's open sides; press and grip poses on a
  part's faces in several hand orientations.
- **Replaces**: the hand-weighted grasp score ([manip.py:199-201](src/atrace/manip.py#L199-L201)) and its veto chain
  for allowed contacts, v2's compartment snap, the press search's rejections, `_face_implausible`.

### 5.5 Object store (`src/atrace/lrl_store.py`)

- **Record per entity**: id, S1's name, shape (points in the object's own frame, frozen at the grasp or at the first
  full view), pose, status (seen, held, released and unverified, verified, lost), parent (support, container or the
  part it lies in), last frame seen, grasp transform while held.
- **Exact updates**:
  - held: pose = hand pose ∘ grasp transform at every step (today the model stays at its pre-grasp place);
  - released: pose = the last held pose, then one look aimed there and registration of the frozen shape
    (`relocalize`, which exists) before the status becomes verified;
  - part moved: the part and what lies in it shift by the travel the detector reports.
- **Map**: every record is re-inserted at its current pose in `rebuild_map`
  ([harness.py:1192-1227](src/atrace/harness.py#L1192-L1227)), so nothing depends on the last 8 frames.
- **When registration fails**: S1 gets a `choose` call among the detections marked in the image.
- **Tracker (optional, gate G3)**: SAM3's video predictor is in `third_party/sam3` but not loaded. It is tried
  offline on the 66 recorded episodes before any integration.

### 5.6 Facts, next candidate, S1 choice (`src/atrace/lrl_report.py`, `program.py`)

- **Facts.** One record per attempt: the candidate, what was commanded, what the detector measured (classes with
  probabilities, travel, force), the thing nearest the contact point in the map, and TRACE's refusal as it is typed
  today (which body, which entity or unseen space, by how much; which joint). No interpretation: the sentences built
  by `_grasp_why` and `_why_words` are not used on the lrl path.
- **Next candidate.** The ranked list lives for the whole stage. A failed candidate and its near-duplicates (within
  the capture range) are marked; the next one is taken. There is no table from cause to fix.
- **S1 `choose`.** A new call kind, used when the list is used up or the store reports an unclear state. S1 sees the
  facts and the head and wrist images with the remaining options drawn as numbered marks (put-down spots with their
  entry direction, grasp regions, faces or handles of a part, detections of a lost object, "look from there
  first"). It answers with a number, or with a new pixel when none fits. Execution resumes inside the stage.
- **REPAIR** stays for changes to the program itself; its report becomes the list of facts.
- **Budget.** `choose` counts as a model call under the 50-call cap.

## 6. Training and data

### 6.1 Stage simulator, two tiers

| | Tier A: blind | Tier B: with perception |
|---|---|---|
| Used for | skill policies without vision, detector v0 | scorer, detector with depth, place and insert policy |
| Scene | a training layout of 6.2, with the Panda | same |
| Robot interface | `SimRobot` unchanged, environment built without cameras | `SimRobot` with head and wrist RGB-D |
| Start of an episode | arm set to an entry pose by IK, then 10 hold steps | the same, after the harness's own perception has built the map |
| Perception | none: the believed contact frame is the true one plus a sampled hand-off error | harness perception; S1 replaced by the stub; SAM3 replaced by simulator masks degraded to mimic it |
| Speed | 14 ms per control step per process, measured | to be measured first in M4 |
| Hardware | CPU processes only | EGL rendering on GPUs (D8) |

- **Why the harness's own interface.** A policy trained through `SimRobot` and the same OSC sees at run time the
  dynamics it was trained on. The controller gain, the 20 Hz rate and the command format are identical by
  construction.
- **Hand-off error model.** M1 measures, on the first lrl runs on dev tasks, the transform between the contact frame
  the harness believed and the true one at hand-off time: position, normal or axis angle, travel estimate. Tier A
  then resamples that distribution, widened by 1.5. Until then it uses the prior range of M0.
- **No test layout and no held-out layout** is used by either tier (rule 4 of section 3).

### 6.2 Training scenes

Layouts come from two sources, both in LIBERO's room arenas:

- **Native layouts** of LIBERO-goal, LIBERO-object, LIBERO-spatial and the four LIBERO-90 dev scenes
  (KITCHEN_SCENE1, KITCHEN_SCENE10, LIVING_ROOM_SCENE3, STUDY_SCENE2).
- **New random layouts.** LIBERO fixtures and objects are placed at random through generated BDDL files, which is
  LIBERO's own scene path. Generated content is written as MJCF and added through the model XML
  (`reset_from_xml_string`).

What the layouts contain:

- **LIBERO fixtures with a part to move**: `wooden_cabinet`, `white_cabinet`, `microwave`, `flat_stove`. The white
  cabinet and the microwave occur only in test and held-out scenes, so they are trained on in other layouts only.
- **LIBERO containers, supports and objects**: basket, tray, caddy, plates, the two-layer shelf, drawers standing
  open, the microwave cavity, and the object meshes of the suites.
- **Generated mechanisms**, for variety beyond four fixtures:
  - drawer stacks: 1–4 drawers; front 15–45 cm by 6–20 cm; handle as bar, knob, lip or none, protruding 0–4 cm;
    neighbours above and below with their own handles; carcass overhang 0–3 cm; travel 8–35 cm; loose contents that
    can jam;
  - doors: vertical hinge left or right, horizontal hinge top or bottom; panel 15–45 cm; range 60–150°; a cavity or
    a solid body behind;
  - knobs: turn about the mount normal; 2–6 cm across; range 60–300°;
  - sliding panels: travel parallel to the face.
- **Physics randomization**, on generated and LIBERO joints alike: damping, friction loss, mass, stops; finger
  friction; controller gain ±20 %.
- **Sensor randomization**: wrench noise 0.3–1.5 N and bias, delay 0–2 steps (starting values from the near-contact
  design panel, which marked them as assumptions).
- **Placement**: on the table within reach, yaw ±60°, random height.
- **Generated containers, supports and objects (M4, M5)**: open-top boxes and trays, cavities with a side opening,
  shelves, drawers standing open under an overhang, supports that already hold something; boxes, cylinders, bowls,
  mugs with handles, bottles, flat packs, all from primitives.
- All ranges are training settings (tag `T`), set from ordinary household sizes.
- **Kept out of training**: the six LIBERO fixtures no task uses (`short_cabinet`, `short_fridge`, `slide_cabinet`,
  `window`, `faucet`, `basin_faucet`) and robosuite's `door`. They are the unseen fixtures of the gates (7.1).

### 6.3 Policy training

- **Algorithm**: PPO written in torch (about 400 lines; the shared environment has torch and no RL library), one
  actor on the observation of 5.2 and a critic that also sees simulator truth (joint state, true axis, contact
  flags, physics parameters).
- **Reward**: progress of the part toward the wanted travel; penalties for force above the soft limit, for contact
  outside `allowed`, for time and for residual size; a terminal bonus when done is raised with the goal reached, a
  penalty when it is raised wrongly.
- **Curriculum**: entry error and axis error grow from zero to the measured distribution times 1.5.
- **Budget**: about 25 M control steps per hour on 100 processes at the measured step time. A run of 20–50 M steps
  is 1–2 h. The step count is my estimate from comparable contact-RL work, not a measurement.
- **Known traps handled up front**: observation normalization stored as persistent buffers and checked by a
  save-and-load round trip; rewards and labels from joint values and contact pairs, not from `mj_geomDistance` on
  boxes (bogus in MuJoCo 3.2.6); every process goes through `libero10_collision.runtime.setup()` so LIBERO's config
  stays per process.

### 6.4 Detector and scorer data

- **Detector v0** trains on the same Tier A rollouts as the policy: all policies seen during training, the base law
  alone, and injected failures, so that it does not learn "done means success".
- **Scorer** needs Tier B. Per sample: render, build the map, sample candidates, set the arm to the entry pose,
  run the skill, label. Target about 100,000 labelled candidates per kind.
- **Detector with depth** (M4) adds the wrist crops from the same Tier B rollouts.
- **Splits**: layouts are split by seed into train and validation; calibration curves come from validation.

### 6.5 Checkpoints

- `weights/lrl/<part>_<kind>_<date>_<sha8>.pt` with `weights/lrl/manifest.json` (sha256, training commit, data
  seeds). The directory is not tracked by git.
- The build hash covers `src/atrace/*.py`, `src/traceplan/*.py` and `run_episode.py` only. The lrl modules are flat
  files in `src/atrace`, so they are covered; the weights' sha256 values are added to the hash input in
  `run_episode.py` and in `freeze.sh`.

## 7. Evaluation

### 7.1 Sets

| Set | Content | Used | Simulator truth |
|---|---|---|---|
| Training layouts | the layouts of 6.2, seeds A | training | rewards, labels |
| Validation layouts | the same recipe, seeds B | model selection, calibration, gates | labels |
| Unseen fixtures | Parts to move: LIBERO `short_cabinet`, `short_fridge`, `slide_cabinet`, `window`, `faucet`, `basin_faucet` (no LIBERO task uses them) and robosuite `door`. Containers: the `short_fridge` cavity, a `short_cabinet` drawer standing open. Never trained on | gates: the only measure of a new kind of drawer or door | scoring |
| Dev tasks | 30 tasks: the 20 LIBERO-90 tasks of KITCHEN_SCENE1 (5), KITCHEN_SCENE10 (6), LIVING_ROOM_SCENE3 (5), STUDY_SCENE2 (4), and the 10 LIBERO-goal tasks | full episodes during development; the hand-off error measurement | scoring, and the M0 measurement |
| Held-out tasks | 35 LIBERO-90 tasks: KITCHEN_SCENE2 (7), 5 (5), 7 (3), 9 (6); LIVING_ROOM_SCENE4 (5); STUDY_SCENE3 (5), 4 (4). Their layouts are never used for training or development | gates only | scoring |
| Test | Libero10-Collision t00–t09, ep0–9. The layouts of its 9 scenes are not trained on, and the 35 LIBERO-90 tasks located in them are not used | M6, and the named sanity episodes | never |

- LIBERO-90 has 55 tasks in the 11 scenes LIBERO-10 does not use; they split by scene into 20 dev and 35 held-out.
  Its other 35 tasks sit in the 9 test scenes and are not used at all.
- Dev covers opening wooden drawers, closing one, putting things into one and closing it, a tray, the caddy, and
  from LIBERO-goal the stove knob, the wine rack and pushing a plate. Held-out covers the white cabinet, opening the
  microwave, the stove knob, stacking, a two-layer shelf (including "under the shelf", a side entry) and mugs beside
  the caddy.
- What each number says:
  - held-out tasks: layouts never seen; their fixture classes were trained on in other layouts;
  - unseen fixtures: fixture classes never seen. This answers "if we change to another kind of drawer, can we
    adapt?";
  - test: layouts not trained on, fixture classes trained on, plus the added obstacle.
- Limit: object classes are shared between the training suites and the test (bowls, mugs, books).

### 7.2 Measures

- Per episode: LIBERO success, control time, model calls, collisions by your rule.
- Per contact kind: first-attempt success, attempts, time in the stage.
- Recovery: success rate of episodes in which at least one attempt failed. This is the 15-in-50 number of section 2.
- Per learned part, in the stage simulator: success against entry error (the capture-range curve); detector
  precision and recall per class; scorer top-1 success against the generator's own order.

### 7.3 Comparison with v2: later

- Every lrl run uses gpt-6.2-astra at effort high (D3), the caps of section 3, and a frozen build.
- No v2 run is planned (your decision of 2026-11-25: "we put v2 later"). lrl is reported on its own.
- What exists for v2, none of it at the D3 setting: the full test at gpt-6-astra high, 65/100, from a build that was
  not frozen; and t03, t04, t09 at gpt-6.2-astra max, 6/10, 5/10, 0/10, from build 27c3f19c.
- When the comparison is wanted: v2 once per set at the D3 setting. With every part off, lrl runs the same code.

### 7.4 Runner changes

- `run_episode.py` accepts only t00–t09, and the collision monitor needs the added obstacle. It gains `--suite`,
  `--suite-root` and `--init` for stock LIBERO tasks, with the evaluator contact log that `dev_episode.py` has.
- The 55 LIBERO-90 bddl files and their initial states are copied from
  `/workspace/ysk/past2next_bug_fixed/third_party/LIBERO` into `suites/libero_90/` (the harness's own LIBERO
  checkout has no libero_90; it does have LIBERO-goal and LIBERO-object).
- `user_rule.py` is hard-coded for t00–t09. For other tasks the exemptions are derived from the program events it
  already reads; until that exists, the raw robot and held-object contact onsets are reported.
- `--stage-truth` (evaluator side, dev tasks only): at every stage start, hand-off and contact end, save the true
  joint values and object poses.
- Skill traces are written per contact to `contacts/` at 20 Hz. Today a per-step trace exists only in the
  evaluator's recorder, at 10 Hz and without torque.

## 8. Milestones and gates

| | Work | Check | Gate | Size (rough) |
|---|---|---|---|---|
| **M0** | Tier A; skill loop; reference curves of today's stroke. No harness run, no model call | Tests once; today's stroke run in Tier A | **G0** | 1–2 days |
| **M1** | PPO; `move_part` policy (push, then pull, turn); detector v0; flag and runner; lrl actuation; hand-off error measured | Tests once; dev drawer and knob tasks; t03 ep0–2 | **G1** | 4–6 days |
| **M1E** | End-to-end contact policy for moving parts: the hand is handed over 10–20 cm from the part and the policy chooses face, push or hold and hold type from depth; teacher by RL on simulator state, student on RGB-D by distillation | Tests once; stage simulator on unseen fixtures and low handles; dev and held-out part-moving episodes, t03 ep0–2 | **G1E** | 1–2 weeks (estimate) |
| **M2** | Facts; candidate list per stage; `choose`; REPAIR report | Tests once; dev tray and caddy tasks | none (measured: recovery rate, calls) | 2 days |
| **M3** | Object store; map re-insertion; tracker tried offline | Tests once; dev tasks with a release | **G3** (tracker only) | 2–3 days |
| **M4** | Tier B; scorer; entry hand-off and detector for put-down and grasp | Tests once; all 60 dev episodes | **G4** | 5–7 days |
| **M5** | Place and insert policy where the base falls short; grasp policy only if needed | Tests once; dev tasks | **G5** | 4–6 days |
| **M6** | Freeze; lrl on held-out and test; ablations | — | your approval before it starts | about 8 h compute |

### M0 — foundations

1. Tier A: environment without cameras, the scene builders of 6.2 (generated BDDL files, XML insertion), entry pose
   by IK, vector of processes, loaders for the unseen fixtures.
2. Skill loop, base laws and wrapper, unit-tested with a simulated hand.
3. Reference curves in Tier A: success against entry error for today's stroke and for the base law, on generated
   drawer stacks, on the wooden and white cabinets in random layouts, and on the unseen fixtures. Today's stroke is
   `articulate.translate` itself, called through a shim from the hand poses the press search uses.
4. Entry error: a prior range until it is measured in M1: position ±3 cm, normal or axis ±20°, travel estimate
   ±3 cm (tag `T`).

Moved to M1, where lrl first runs in the harness: the flag, the runner for stock tasks, stage truth, per-contact
traces.

**G0, two questions.**
- *Fidelity*: does today's stroke on the white cabinet, in a random layout, show the block recorded in the t03 runs
  of section 2 (hand body or wrist link on the drawer above, a steady force with the hand standing still)? If the
  simulator never shows it, the reason is found before any training.
- *Need*: is today's stroke below 90 % over the prior range? If not, skip M1 and go to M2–M4.

### M1 — pilot: move a part

1. Train push, then pull and turn, on the layouts of 6.2 with the prior entry-error range.
2. Train detector v0 on the same rollouts.
3. Harness side: `--placement lrl` (every part off behaves as v2), `--lrl-parts`, `--lrl-weights`, weights in the
   build hash and the summary; runner for stock LIBERO tasks, `suites/`, `--stage-truth`, per-contact traces.
4. `lrl_actuate`: the contract is built from the measurements that exist (mount frame, part face, S1's motion and
   goal). Entries are poses a few centimetres in front of the face in several hand orientations, or at the handle
   with open pads. TRACE goes to one of them; the skill runs until the detector ends the stage. Not on this path:
   the press search, `_linear_verdict`, the re-seat rule, a new approach between strokes.
5. Entry order before the scorer exists: by the success rate each hand orientation reached on validation layouts
   (tag `M`).
6. Hand-off error: the first lrl runs on the dev tasks that move a part (11 tasks × 3 initial states, 33 episodes)
   record stage truth. If the measured error is wider than the prior, the policy is retrained on it.

**G1.**
- Stage simulator, at the measured entry error (the prior range until step 6 has run):
  - validation layouts with the task fixtures: policy success at least 90 % and at least 15 points above today's
    stroke;
  - unseen fixtures: at least 80 % and at least 15 points above today's stroke;
  - contacts outside `allowed` not more frequent than with today's stroke;
  - detector precision for "reached" at least 98 % on validation layouts.
- Harness: on the 33 dev episodes of step 6, the actuation succeeds first time in at least 80 % of the stages and
  within 15 points of what the simulator gives at the measured entry error. Then the 5 held-out part-moving tasks ×
  3 initial states (15 episodes), and t03 ep0–2.
- If the simulator gate passes and the harness falls short, the gap is in the hand-off or in perception; that
  finding orders M3 and M4.

### M1E — end-to-end contact policy for moving parts (your decision D11, 2026-11-25)

**What changes.** In M1 the contact is chosen by hand-written generators (press pose families, the antipodal grip
search, face and axis hypotheses) and the policy only corrects a base law at a pose it is given. In M1E the policy
chooses the contact itself:

- TRACE certifies transit to a start region 10–20 cm from the part S1 named (its pointer, segmented as today), not
  to a contact pose. Long-range motion and collision safety stay with TRACE.
- From there the policy decides which face to touch, whether to push or to hold, and how to hold (pads across a
  handle, fingertips hooked behind a low bar, pads around a knob), and moves the part until it reports done.
- No base law. The action per step (10–20 Hz) is a bounded change of the hand pose, the gripper and the done flag.
- What stays: the safety wrapper (force limit, a workspace box around the start region, speed, time), the outcome
  detector as the judge, S1's effect check where S1 named an effect. M1's candidate path is the fallback when the
  detector says the policy did not move the part.

**Why.** The failures measured in M1 sit in the hand-written choice of contact, not in the motion: 18 of 52
attempts on the dev tasks were on a wrong face or axis and none succeeded; on low bar handles the grip generator
proposes no hold at all; on drawers the entry pose decided success in the stage simulator (M0). A residual on a
given pose cannot fix any of these.

**Run-time inputs (real-robot rule).** Depth from the wrist and head cameras turned into a point cloud of the region
around the part S1 pointed at, robot state, wrist force and torque, finger gap, the goal (mode, wanted effect and
amount, turn direction). Simulator truth only in the teacher, the rewards and the labels.

**Steps.**

1. **E0, cheap test (about half a day).** In Tier A, the M1 policy with the base law off and wider action bounds,
   trained for the same number of steps, against the M1 residual. Answers whether the base law helps or limits.
2. **E1, teacher.** PPO on simulator state (the part's true geometry: faces, handle position, size, protrusion and
   height above the table, joint type, axis, range and value), starting 10–20 cm out with the start error measured
   in M1. Reward as in 6.3, plus approach shaping early in a curriculum. Training scenes: the generated mechanisms
   with more handle kinds (low bars, recessed handles, knobs, lips, no handle) and the LIBERO fixtures in random
   layouts; physics randomization and injected failures as in M1. My estimate: 50–200 M steps, 5–18 h per run at
   the measured 3.1 k steps/s.
3. **E2, student.** Distilled from the teacher by DAgger (student acts, teacher labels) onto the run-time inputs
   above: depth rendered with noise and a calibration error (D9), a segmentation mask degraded towards SAM3's.
   Rendering throughput is measured first; it decides the student's resolution and the data budget.
4. **E3, harness.** `lrl_move` and `lrl_turn` get the start-region hand-off and the student; the detector judges;
   after two detector calls of no_effect or no_contact the stage falls back to M1's candidates.

**Gate G1E.**

- Stage simulator, unseen fixtures and validation layouts: the student's first-attempt success at least that of
  M1's path (candidates + residual + detector) and at least 80 %; it solves the low-bar handle class, where M1's
  generator has no hold.
- Harness: the 33 dev and 15 held-out part-moving episodes and t03 ep0–2, against M1's numbers on the same
  episodes: higher first-attempt success of the actuation and less time in the stage.
- At G1E: whether the same teacher-student route replaces the scorer and base skills of M4 and M5 for grasp and
  put-down.

**Reused from M1.** Tier A and its scene generators, injected failures, the outcome detector, the evaluation
protocol, the stock-task runner with stage truth, the measured hand-off error.

**Risks.** Exploration without a base law (shaping and curriculum needed); sample cost on the CPU simulator (no GPU
physics on this machine without a port); rendering throughput not yet measured; generalization (the blind M1 policy
already lost on unseen fixtures, so training diversity is the crux); real depth is noisier than simulated depth.

### M2 — facts, next candidate, choice

1. `Facts`, the per-stage candidate list, resume inside a stage.
2. `choose` call, mark drawing, `scripts/s1_stub.py` extended so the offline path covers it.
3. REPAIR report built from facts.

Measured on the dev tasks: recovery rate and model calls per episode.

### M3 — object store

1. Store, kinematic pose while held, release prediction, aimed look and registration, map re-insertion.
2. Offline trial of the SAM3 video predictor on the recorded episodes in which the harness lost an object.

**G3 (tracker only).** Integrate it if it keeps the object through the release in at least 80 % of those cases and
its memory fits beside the image detector on one GPU. Otherwise the store stays with prediction and registration.

### M4 — scorer and entry hand-off for put-down and grasp

1. Tier B and its throughput; generated containers, supports and objects.
2. Data, scorer, detector with depth.
3. lrl put-down: candidates around S1's pixel including non-vertical entries, ranked; TRACE to an entry pose; base
   skill; the detector decides released or not. lrl grasp: ranked candidates kept for the stage; held or not from
   the detector.

**G4.** On validation layouts and on the unseen containers (500 layouts or more per kind): success of the
top-ranked candidate at least 8 points above the generator's own first choice. Below that, the scorer is not
switched on for that kind.

### M5 — place and insert policy

1. Capture-range curves of the base `put` skill with ranked candidates in Tier B, per class: open top, side
   cavity, drawer under an overhang, occupied support.
2. For classes below 90 %: a residual policy that also sees a local crop of the map and a wrist depth crop.
3. Grasp policy only if, after M4, more than 10 % of grasp attempts still fail in the contact itself.

**G5.** The form of G1, for `put`: validation layouts and the unseen containers.

### M6 — full evaluation

- Freeze the build and the weights.
- lrl on the held-out tasks (105 episodes) and the test (100 episodes). The comparison with v2 comes later (7.3).
- Ablations on 30 episodes: each part off in turn.
- Report per task: success, control time, model calls, collisions; per part: what it changed.

## 9. Code map

| File | Change |
|---|---|
| `scripts/run_episode.py` | `lrl` in `--placement` (line 51); `--lrl-parts`, `--lrl-weights`; stock-task loading; `--stage-truth`; weights in the build hash (lines 141–143) and the summary |
| `src/atrace/harness.py` | accept `lrl` (assert at line 93); route the put-down at line 1825 (exact match on `"v2"` today: a new value falls through to the ladder planner); branches at the hook points below; `Harness(ContactMixin, LrlMixin)` |
| `src/atrace/lrl.py` (new) | `LrlMixin`: `lrl_actuate`, `lrl_put`, `lrl_grasp`, candidate lists |
| `src/atrace/lrl_contract.py` (new) | `Contract`, entries to `EndCandidate`, limits with their tags |
| `src/atrace/lrl_skill.py` (new) | loop, base laws, wrapper, trace; depends only on the robot interface, so Tier A imports it |
| `src/atrace/lrl_models.py` (new) | policy, detector and scorer networks; loading; inference |
| `src/atrace/lrl_store.py` (new) | object store |
| `src/atrace/lrl_report.py` (new) | facts, marks |
| `src/atrace/program.py` | `CHOOSE_SCHEMA`, `CHOOSE_INSTRUCTIONS`; facts in the REPAIR report |
| `src/lrl_train/` (new) | Tier A and B environments; scene builders (generated BDDL files for LIBERO assets, MJCF for generated content); PPO; datasets; training and evaluation scripts. M1E adds: the teacher environment (start region, privileged state, approach shaping), depth rendering for the student, DAgger distillation |
| `src/atrace/lrl_models.py` (M1E) | the student policy (point-cloud encoder and MLP), loading and inference |
| `scripts/s1_stub.py` | `choose` |
| `tests/test_lrl_*.py` (new) | contract to `EndCandidate`; loop equivalence with `translate`; model save and load; store kinematics; report and `choose` schemas |
| `runs/contact_stage_dev_20261123/freeze.sh` | weights manifest in the hash; `suites/` linked |
| `README.md` | an LRL section: flags, parts, sets, the constants ledger |

Hook points in `harness.py` (line numbers of the live tree):

| Stage | Where lrl branches |
|---|---|
| Actuate | dispatch 3390, 3411; hand-offs 3448, 3918, 4070; strokes 3944, 4087, 4099; continuation 4092–4102; outcome 4167, 3739 |
| Put-down | dispatch 1825; aim and move above 2104–2244; descent and release 2286–2300; look 2319–2356 |
| Grasp | 1387–1546; planner 1601–1730; hold check 1474–1515 |
| Repair | 4784–4796 (the program restarts at stage 0 today); report built at 4791–4792 |
| Map and objects | `rebuild_map` 1192–1227; grasp 1505–1510; release 2358–2382; `relocalize` in `contact.py:197` |

## 10. Decisions

Settled on 2026-11-25:

| | Decision | Your answer |
|---|---|---|
| **D1** | How `lrl` is selected | `--placement lrl`, like v2 |
| **D2** | Order | Pilot first: M1 before M2 and M3 |
| **D3** | S1 model and effort | gpt-6.2-astra at effort high, for every lrl run and every baseline |
| **D4** | Held-out design and training data | The split of 7.1 is accepted. Training may use LIBERO-90, LIBERO-goal, LIBERO-object and the like, never the test tasks. The 7 held-out scenes are for evaluation only. The 35 LIBERO-90 tasks inside the 9 test scenes are not used |
| **D5** | Fixture split | Dropped. Under D4 the white cabinet, the microwave and the stove are trained on in other layouts; the unseen fixtures of 7.1 take their place at the gates |
| **D7** | Test protocol at M6 | Put off: "we don't need to care about the v2 ... we put v2 later". M6 runs lrl only (7.3) |
| **D11** | End-to-end RL for the contact stage | After M1, as milestone M1E: the policy chooses the contact itself from a start region 10–20 cm from the part (option B of the 2026-11-25 discussion), not a removal of the base law alone (A) and not the whole stage from S1's pointer (C) |
| **D8** | Compute and environment | Training simulator processes: as many as speed training up, up to about 240 of the 256 cores, bounded only by memory (about 450 GB); no cores are kept back for others ("I own this server", 2026-11-25; first answer was up to 96); GPUs 0 to 3 for anything lrl needs: episodes, rendering, training, experiments ("gpu 0-3 are all yours", 2026-11-25; first answer was GPUs 2 and 3); the shared `.venv` read-only, PPO written in torch, nothing installed |
| **D9** | Sensor model in training data | Taken as recommended when you said "start the M1E" (no explicit answer; change it before the student is trained if you want otherwise): depth noise and a calibration error of a few millimetres in what the student, the scorer and the detector see, and a part mask degraded towards SAM3's. The harness's own depth is noise-free today |

My readings inside D4, to correct if they are wrong:

- LIBERO-spatial is counted under "and the like".
- Objects are treated as fixtures are: any LIBERO object mesh may be used as an asset in a training layout, also one
  whose tasks are all in test or held-out scenes.
- The rest of D4 as proposed: 2 initial states per dev task and 3 per held-out task, LIBERO's original
  instructions, no added obstacle.

Open:

| | Decision | Recommendation | Needed by |
|---|---|---|---|
| **D6** | May a confident detector skip S1's look after a put-down? | Yes, at 98 % precision on the calibration curve. It saves up to 4 calls per put-down stage | M4 |
| **D10** | S1 text for `choose` and the fact report | General text with no task content. Shown to you as a draft before the first run that uses it | M2 |

## 11. Risks and limits

- **The stage simulator may not match the harness.** G0 tests it before any training. The robot side is identical
  by construction, and the task fixtures are LIBERO's own.
- **A new kind of drawer or door is measured on a small set.** The test's fixture classes are trained on in other
  layouts, so the test and the held-out tasks measure transfer to new layouts, not to new fixtures. The unseen
  fixtures at the gates are the only number for that, and they are six LIBERO fixtures and one door. If the policy
  falls short there, the generated mechanisms get more variety, or fixture meshes from RoboCasa are added (on disk,
  built for a newer MuJoCo; loading untested).
- **A geometric check could fix the t03 stroke alone.** Sweeping the hand body along the planned travel would catch
  the contact with the drawer above. It is not planned: it is one more gate of the kind section 3 rules out, and the
  simulator will show whether the learned route covers the same case.
- **A false "reached" moves the program on.** Hence the precision gate and S1's check where an effect was named.
- **The scorer is the largest piece.** Tier B speed is unmeasured, and simulator masks are not SAM3 masks.
- **A bounded residual may be too weak** for re-posing the hand after a block. If the M1 curves show it, the bounds
  widen or the policy outputs the pose target directly inside the same wrapper.
- **Model calls.** At about 6 calls per episode: M0 none, M1 about 300 (51 episodes), M6 about 2,100 including
  ablations. Usage limits may pause batches; the runner resumes.
- **Code size.** lrl adds modules; v2's code stays for the other modes.
- **Rules can creep back.** The ledger of section 3 is the check; the count of tagged constants is reported at every
  milestone.
- **Not covered**: the real robot (inputs are kept to what it has; depth is noise-free and calibration exact in the
  harness today); objects shared between LIBERO-90 and the test; S1's own errors in grounding beyond what `choose`
  and the store catch; contact dynamics such as pouring or wiping.
- **Another session runs batches from `builds/`.** All work sits behind the flag, and every run uses a frozen build.

## 12. Facts behind this plan

Measured while writing it (scripts and outputs in `runs/plan_lrl_20261125/`):

- Repair statistics and failed episodes per task: computed from the v2 full test's `results.json`.
- Stage simulator speed: a scene built with LIBERO's `ControlEnv` without cameras steps at 14.0 ms per control step
  in one process (72 steps/s); building the environment takes 1.8 s; restoring a state 0.1 ms. For comparison, a
  reader found a median of about 44 ms per control step in the harness's certified executions, where both cameras
  are rendered at every step (from the v2 batch logs; not re-measured by me).
- XML insertion: a box drawer on a slide joint was added to a scene, the environment reloaded, and the closed
  gripper pushed it 15 cm through `SimRobot` with about 3 N during travel and 58 N at the stop.
- These two probes used the t03 and t00 scenes as carriers. They read nothing about those scenes' own objects, but
  they did run the simulator in test scenes, which DESIGN.md section 5 rules out for development. You accepted this
  on 2026-11-25. The stage simulator uses no test layout (rule 4 of section 3).
- t03 drawer close (run `v2_27c3f19c_gpt62_max_..._t03_ep003_009`, ep004, ep006, ep008): stroke ends and drawer
  readings from `events.jsonl`; contacts by body from `collision.json`; force and hand motion from
  `rec/telemetry.jsonl`; joint parameters from the fixture XML. The t03 environment was built only to map geom names
  to bodies; no test episode was run or replayed. Numbers in `runs/plan_lrl_20261125/RESULTS.md`.
- Hardware on 2026-11-25: 256 cores (load about 35), 503 GB RAM, 8 RTX 4090 with GPUs 2 and 3 free.
- Suites, from the bddl files on disk: LIBERO-goal uses `wooden_cabinet`, `flat_stove` and `wine_rack`;
  LIBERO-spatial `wooden_cabinet` and `flat_stove`; LIBERO-object the basket. The white cabinet and the microwave
  occur only in LIBERO-90, each in one test scene and one held-out scene. 35 LIBERO-90 tasks are in the 9 test
  scenes; KITCHEN_SCENE4 there has `close_the_bottom_drawer_of_the_cabinet` and
  `put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet`, the two stages of t03.

From six read-only readers (notes in `runs/plan_lrl_20261125/notes/`), not re-read by me except where named:

- Mode wiring, S1 call kinds, v2 put-down and grasp flow, constants (`stages.md`). I read `run_episode.py`,
  `harness.py` lines 52–140, `sim_iface.py`, `body.py` and `articulate.py` lines 1–60 and 340–401 myself.
- Actuation flow and hook points (`actuation.md`).
- TRACE request and margins, robot interface, simulator construction (`handoff_sim.md`).
- Object model lifecycle, scene map, SAM3 entry points (`perception_state.md`).
- Episode files, task definitions, the LIBERO-90 inventory, fixtures, environments (`data_eval_assets.md`).
- Earlier plans, your recorded decisions, the near-contact design panel (`prior_designs.md`).

Not checked:

- Whether the libero_90 initial states load under the pinned LIBERO revision.
- Whether LIBERO fixtures and objects can be placed at random through generated BDDL files (LIBERO's own scene
  path; only the XML insertion of generated content was probed).
- Whether SAM3's video predictor takes frames one at a time, and its memory.
- Tier B throughput.
- Whether RoboCasa's fixture XML loads in MuJoCo 3.2.6.
- The number of training steps a policy needs.
- That an environment built without cameras gives the same contact behaviour as the harness's (same physics and
  controller; assumed).
