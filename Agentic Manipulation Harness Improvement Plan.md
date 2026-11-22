# Agentic Manipulation Harness: Improvement Plan

Nov 16, 2026 · @Zilin Dai

## Bottom line

Both harnesses are slow, imprecise and collision-prone for one shared reason: the frontier model is used as the robot's ruler and as its feedback loop. Nothing else in the system holds a measured, persistent picture of the scene, so every coordinate, every closer look and every "did that work?" costs a model call.

The larger problem sits above the harness. The two repos together record one live episode on one frozen scene, so no change can yet be shown to make the system faster, safer or more general. Fix the measurement first; without it the three areas below cannot be judged either.

What the one recorded episode shows (v2 harness, task T04 with the added can):

- 27 s of robot motion sits inside 305 s of wall time. Most of the rest is waiting on 14 model calls.
- 12 of the 13 execution calls were triggered by things a local layer can do: 2 interface rejections, 4 "move closer to see where it is", 6 "check what happened".
- Scored against simulator truth, the model's position estimates are off by a median of 4 cm. Its box for the test obstacle covered 36% of the can's footprint.
- All 3,372 forbidden contact samples fell in grasp and release phases, where the obstacle router is switched off by design.

Three areas for improvement, in order of dependence:

1. **Measure geometry instead of guessing it.** The harness owns a scene state built from perception. The model points at things; code computes the metres and checks the whole robot against them.
2. **Put the model on the right time-scale.** Plans carry their own success tests and fallbacks, checked locally. The model is called for decisions, not for measurements.
3. **Make generality measurable.** Batch evaluation over randomized scenes with held-out tasks, one thin robot interface, and a pruned code tree.

Projected effect on the recorded task: about 3 to 4 model calls instead of 14, and roughly 80 to 100 s instead of 305 s. This is a projection from the trace, not a result; the Phase 2 and Phase 3 gates test it.

## Reframing the goal

The need behind the goal is a robot that does a new manipulation task on day one, with no task-specific data or code, at a speed and contact safety a person would accept. The agentic harness is one means to that end. In the code, each of the five stated properties has drifted into a narrower meaning than the need requires.

| Stated property | How the code interprets it | What the evidence says | Reframed target |
| --- | --- | --- | --- |
| Fast | Faster actuation and batching: transit command cap raised from 20 mm to 100 mm, 120-action fragments | Motion is 27 s of 287 s of control time. Motion five times faster would save under 8% | Episode time = calls × latency + motion. Budget calls per subtask first |
| Zero-shot | Stateless: nothing calibrated, remembered or reused; RGB images only | The model must estimate metres from two 512 px images, and is off by a median of 4 cm against simulator truth | No task-specific data or code. Pretrained perception, per-robot calibration and cross-task memory are allowed |
| Collision avoidance | One point (the grip site) routed around 0 to 4 model-guessed boxes, in transit only. Contacts are recorded, never prevented | Every recorded forbidden contact was the hand body touching the object during grasp or release | Hand, arm and carried object checked against measured geometry in every phase |
| Few attempts | No automatic retries; one manually approved episode per launch | Success rate is unknown. 2 of 13 calls repaired interface rejections, not physical failures | First-attempt success rate and recoveries per subtask, over many randomized episodes |
| Not task- or benchmark-specific | No task nouns or task-tuned numbers in control code, policed by a lint | The design was selected by repeated runs on T04/T08-style scenes (the reference run is labelled uc71, apparently the 71st user-check variant). Every layer assumes Panda, robosuite OSC and LIBERO | Results hold on held-out tasks and on a second robot or simulator |

Three confusions matter most:

- **Symptom for cause.** Slowness looks like a slow arm. The cause is the number of model round-trips, and the cause of those is the missing local source of geometry and verification.
- **Means for end.** Zero-shot exists to avoid per-task cost. Applied as "no state, no depth, no calibration", it raises per-task cost, because the model re-derives the scene on every call.
- **Syntactic for statistical generality.** A lint proves the code names no task. Only held-out results prove the behaviour is not fitted to one.

The most expensive inherited constraint is RGB-only sensing. The prompts state that no depth is supplied, and both extremes the code has tried are responses to that rule: first a hand-built stack recovering height maps from camera parallax, then a model asked to estimate metres directly. If the target is real robots, a depth camera is a design choice. If the target is parity with RGB-only policies on LIBERO, it is a benchmark rule, and it conflicts with the fifth property. This report assumes depth may be admitted and gives an RGB-only fallback.

## Where the two harnesses stand today

The two repos are two answers to one question, how much the model should remember, and neither holds a scene state. The v2 harness keeps the whole transcript and V3 r1 keeps almost nothing. In both, geometry exists only as numbers the model writes.

|  | v2 obstacle harness, 100 mm | V3 r1 (astra\_tools\_r1) |
| --- | --- | --- |
| Model's job | Choose objects, grasps, world-frame positions and orientations, and obstacle boxes, from two RGB views | Same |
| Memory between calls | Full chronological transcript plus the last 6 image pairs | 6 one-line contract summaries; current image pair only |
| Plan format | One reviewed plan call, then 1 to 6 segments per call | 1 to 24 segments per call, up to 3 alternatives each |
| Perception tools | None used. Native macros are offered and were never chosen in the reference run | SAM 3.1 and Depth Anything 3 on request; results are advisory only |
| Obstacles | 0 to 4 model-estimated boxes; grip-site route on a 20 mm grid; transit only | Same files, same behaviour |
| Robot guard | None | Joint limits and sampled self-collision; no scene geometry |
| Episode end | Evaluator stops at raw LIBERO success | Model must claim finish; strict physical check plus a 10-step hold |
| Live result in the repo | One episode: raw success, 549 actions, 14 calls, 304.8 s | None |

### What the one recorded episode shows

The v2 reference run is saved as a replay fixture, so every call and action can be read back. Of its 13 execution calls, one was needed.

| Why control returned to the model | Calls | What would remove it |
| --- | --- | --- |
| First decision of the episode | 1 | Needed |
| Interface rejection: a 90° turn over the 1.2 rad stage limit; a start pose inside its own inflated obstacle box | 2 | The harness repairs mechanical violations itself (Area 2) |
| Checkpoint to re-locate a target from closer up | 4 | A measured scene state (Area 1) |
| Checkpoint to judge an outcome: aligned, held, lowered, placed | 6 | Local outcome checks (Areas 1 and 2) |

- **Time.** 549 actions at 20 Hz are 27.5 s of robot motion. Control time was 287.3 s, plus 17.5 s for the plan call. Simulation pauses during inference, so the remainder is model waiting plus simulator overhead; the fixture does not split the two.
- **Geometry.** Scored against simulator truth in the next subsection: a median position error of 4 cm.
- **Contacts.** 3,372 forbidden contact samples over 106 control steps, in three bursts (steps 114–132, 261–300, 420–466). All are palm or link geometry against the object being grasped or released. None fall in a routed transit.
- **Tokens.** 641,669 input tokens for 5,030 output tokens; 2.5% of input came from cache. Each turn carries 5,500 to 7,700 tokens of harness text and about 620 tokens of image.
- **Stalls.** 3 of the 13 handoffs were stalls. Two were within 5 mm of target while carrying a mug; the 4 mm arrival tolerance appears tighter than the arm settles under load.
- **Ending.** The evaluator ended the episode at action 549 with the gripper still closed on the second mug. The 305 s excludes release, retreat and the model deciding it is done. V3's stricter ending will cost more calls on the same task.

### Scored against simulator truth

The saved initial state holds the true object positions, so the model's stated geometry can be scored. Its estimates are off by a median of 4 cm, mostly along the external camera's viewing axis.

| Model estimate | Made at | Position error | Along camera axis | Sideways |
| --- | --- | --- | --- | --- |
| Can, centre of avoid box | Call 0, initial views | 4.8 cm | 4.8 cm | 0.0 cm |
| Yellow mug, centre of avoid box | Call 0, initial views | 5.6 cm | 5.2 cm | 1.8 cm |
| Red mug, centre of avoid box | Call 0, initial views | 2.7 cm | 2.7 cm | 0.5 cm |
| Left plate centre | Call 4, before moving over it | 5.7 cm | 5.0 cm | 2.8 cm |
| Left plate centre | Call 6, wrist camera above it | 2.5 cm | 1.0 cm | 2.3 cm |
| Right plate centre | Call 11, before moving over it | 3.3 cm | 0.0 cm | 3.3 cm |

- **Size right, depth wrong.** The model's boxes are within 3 cm of each object's true size. The position error is the single-view depth ambiguity, which is a measurement problem, not a reasoning problem.
- **The obstacle map.** The model's box covered 36% of the can's true footprint. The 4 cm clearance used for the carry was the smallest setting that covered the whole can: 3.9 cm of it was consumed by the position error.
- **How the can was actually avoided.** The carried mug's footprint passed over the can's edge with 4 to 6 cm of vertical clearance, which came from the lift height the model chose. The router tracks only the grip point and could not have checked it.
- **Closer looks help in one axis.** The wrist view from above cut the left plate's depth error from 5.0 cm to 1.0 cm and left 2.3 cm sideways.

Truth here is the horizontal bounding box of each object's collision geometry at the saved reset pose. Settling may shift it by millimetres, and heights are not scored.

### State of the code

- V3's nine own modules are about 2,000 lines. Their static import closure is 86 files and 31,000 lines. A further 41,000 lines in the tree are not reachable from the V3 entry script.
- 107 of V3's 114 top-level modules are byte-identical to the v2 repo, including the fragment executor and the router. V3 changes the protocol and adds a guard; the motion layer is the same.
- `GeometryEvidence.assess_path`, a depth-based swept-tube check, exists in V3 and is never called.

Four things are worth keeping exactly as they are: the evaluator's isolation from policy inputs, strict physical success, the generic contact monitor, and saved-response replay that reproduces all 549 actions exactly.

## Root causes from first principles

Any manipulation system has four jobs, each with a natural tool and a natural rate. The harness gives three of the four to the slowest and least metric component it has.

| Job | Natural tool and rate | Who does it today |
| --- | --- | --- |
| Estimate where things are | Sensors and geometry, on every observation, in millimetres | The model, once per call, from two RGB images, by reasoning about camera rays |
| Decide what to do | A reasoning model, once per semantic decision | The model, which is the right owner |
| Move without unintended contact | A planner and a guard that know the robot's full shape, on every control step | A point router over model-guessed boxes in transit; nothing in contact phases |
| Confirm what happened | A local comparison of expected and observed, right after each action | The model, on a fresh image pair, roughly 18 s later |

Three root causes follow.

**1. There is no measured scene state.** Geometry lives only in the model's outputs and is re-derived on each call. The v2 prompt ends with a tutorial on pinhole projection: the harness is asking a language model to intersect camera rays in its head. Routing, grasping and verification all inherit the resulting error, measured at a 4 cm median.

**2. The model is the only thing that closes a loop.** A plan is a straight queue. Alternatives are tried only before a segment has moved, and every checkpoint returns to the model. Each uncertainty therefore costs one call of roughly 18 to 20 s (inferred from totals; the plan call measured 17.5 s). Call count grows with the number of motions, not the number of decisions: 6.5 calls per pick-and-place in the trace.

**3. Nothing measures the goal, and the code only grows.** One manually approved episode per launch, a required frozen reset hash and one initial state per custom scene mean that success rate, attempts and generality have never been observed. Without that signal the design has swung between extremes: a hand-built geometry stack of about 24,000 lines with numbered rules and a 190-step calibration, then a model that owns every coordinate, then foundation vision that is present but given no authority. Each swing wrapped the previous layer instead of replacing it. Module notes say the legacy code is "deliberately not modified", and versions sit side by side as files (visual\_field\_v4 to v8, speed\_first\_v2 to v4).

The causes are ordered. Without a scene state there is nothing for a local check to compare against, so loops cannot be closed locally. Without measurement, neither fix can be shown to work. That is why the evaluation work in Area 3 starts first in the plan, although it is listed third.

## Area 1: Measure geometry instead of guessing it

Give the harness a scene state built from measurements, and change the model's job from writing coordinates to pointing at things. This removes the cause of the localization calls and makes real collision checking possible for the first time.

### What changes

1. **A scene state the harness owns.** Entities with label, masks, 3-D points, a bounding shape, pose, uncertainty and last-seen step; support planes; and occupancy with three values: free, occupied, unknown.
2. **Filled from measurement.** With depth admitted: sensor depth from both cameras, fused through the known camera poses. RGB-only fallback: Depth Anything 3 scaled against things of known size and position (the robot's own fingers, the table plane), accepted only where the existing cross-view agreement test passes, plus triangulation across wrist-camera poses. SAM 3.1 supplies instance masks either way.
3. **A pointing contract.** The model names an entity and a part, marks a pixel or picks a numbered mark, and states the relation it wants ("on the left plate", "handle, from above"). The harness returns metric grasp and place candidates and the model chooses. Raw coordinates remain as a fallback.
4. **Collision checks for the whole robot in every phase.** Extend `RobotGuard`, which already holds convex shapes for every link, from self-collision to the scene state, and include the carried object. Wire in the unused `assess_path`.
5. **An approach rule for contact phases.** Before descending, check the hand's known outline (already in `robot_facts_v2.json`) against the target's measured points along the approach axis.

### Consequences

**First order.** Target localization should move from the measured 4 cm median toward the 1 cm range; the gate is a 10 mm median against simulator truth. The 4 "move closer" calls should go. Grasp-phase contacts, which are all of the recorded forbidden contacts, become checkable before they happen.

**Second order, on other modules.**

- *Protocol and prompt.* The schema changes and the camera-math text leaves the prompt. Old replays stop matching, so the v2 fixture becomes a baseline, not a regression test.
- *Router.* A point on a 20 mm grid gives way to a shape-aware check. Route planning takes about 1 ms today against an 18 s model call, so there are four orders of magnitude of compute headroom.
- *Guard.* It gains the power to veto, which creates a new failure class: false vetoes from phantom surfaces, from the held object seen as an obstacle, and from unknown space. This is probably why V3 strips vision of authority. The answer is graded authority: full where two views agree, slow guarded motion where space is unknown, and a logged model override.
- *Verification.* "Held" becomes "the object's points move with the gripper". "Placed" becomes "the object's points rest on the target plane after release". Area 2 depends on this.
- *Deployment.* GPU perception moves from optional to core, and each snapshot adds perception time. Run it during motion, not in series with it.
- *Evaluator.* Perception can be scored directly against simulator truth, which gives the first per-layer metric.

**Third order.**

- A compact state table in the prompt gives V3's short context a memory without bringing back v2's transcript. Memory becomes state, not history.
- Failures become attributable to perception, decision or control, which is what the failure taxonomy in Area 3 needs.
- A metric scene allows rehearsing a candidate in simulation before executing it. That is an option for later, not part of this plan.
- The main risk is relapse into the abandoned 24,000-line stack. Guardrail: the scene layer holds no object-specific or relation-specific rules and stays under a fixed size budget. Anything that needs a rule goes to the model as a question.
- Confident wrong geometry is worse than none. Reflective and transparent objects must widen uncertainty, not return a clean surface.

### Across task types

| Task family (LIBERO examples) | Effect | Remaining gap |
| --- | --- | --- |
| Rigid pick-and-place (mugs, bowls, boxes) | Largest gain: grasp and place poses come from measured shape | Little |
| Containers (basket, drawer interior, microwave) | Free volume becomes explicit | Interiors stay unknown until the wrist camera looks in |
| Articulated parts (drawer, microwave door, stove knob) | Handle location improves | The joint axis must be estimated from the first few centimetres of motion, or left to the model |
| Thin, shiny or transparent objects (wine bottle, moka pot, plate rims) | Masks still help | Depth is unreliable; rely on multi-view rays and wider margins |
| Pushing (plate to the front of the stove) | Object extent and contact side become known | Needs a push primitive (Area 2) |

**Complexity.** This adds a scene layer and a GPU dependency. It removes the 4-box map, the camera tutorial and most "look closer" behaviour. The net is positive only if the legacy tree is pruned at the same time (Area 3).

## Area 2: Put the model on the right time-scale

Call the model once per decision that needs semantics, and let a local layer handle everything that is a measurement, a check or a mechanical repair. On the recorded task this targets 12 of the 13 execution calls.

### What changes

1. **Plans that carry their own tests.** Each step states its goal, the outcome the harness should observe, and what to do if it does not: try the next candidate, back off and re-observe, or escalate. Today a segment offers up to 3 alternatives that are tried only before any motion, and a checkpoint always returns to the model.
2. **Local outcome checks.** Finger gap plus object-follows-gripper for "held"; object-at-rest-on-support for "placed". These run right after the action, at control rate.
3. **Guarded moves.** Lower until contact, then release. The stall detector already exists. This removes the need to know support height, which the model guessed in the trace and then approached in two cautious stages.
4. **Mechanical repairs inside the harness.** Split a rotation that exceeds the stage envelope. Step out of an inflated obstacle box that contains the start pose. Accept arrival under load when the error is small and stable. These cost 2 calls and 2 of 3 stalls in the trace.
5. **Thinking while moving.** Send the next request during execution, on the predicted post-motion state, and discard it if the outcome check disagrees. Today the arm waits while the model thinks, and the model waits while the arm moves.
6. **Effort and context matched to the question.** The V3 client refuses any reasoning effort except high, although the [model](https://developers.openai.com/api/docs/models/gpt-6-astra.md) offers low through max. Use low effort for confirmations. Keep the request prefix byte-stable so caching works (2.5% of input was cached in the trace). Move static capability text out of every turn (about 4,800 characters per turn in v2).

One step of such a plan could look like this. The names are placeholders, not a schema proposal.

```json
{
  "step": "grasp",
  "target": {"entity": "white mug", "part": "handle", "mark": 3},
  "approach": "from above",
  "expect": {"held": "white mug", "stays_upright": true},
  "on_fail": ["next_candidate", "back_off_and_look", "escalate"]
}
```

The model supplies names, a mark and intent. The harness supplies every number.

### Consequences

**First order.** About 3 to 4 calls instead of 14 on the recorded task, and roughly 80 to 100 s instead of 305 s. The projection is 27 s of motion plus 3 to 4 calls at about 18 s each.

**Second order, on other modules.**

- *Model awareness.* The model sees fewer frames, so it loses incidental observations such as a neighbouring object being nudged. Change detection on the scene state must raise those as exceptions.
- *Check errors.* A false "held" is silent in a way a model review is not, and it propagates. Measure each check's false-accept and false-escalate rates against simulator truth, and bias toward escalation.
- *Guard.* Longer stretches run without a model look, so the collision check from Area 1 carries more weight. Area 2 without Area 1 raises collision risk.
- *Schema.* Branching plans are harder to validate, and the model can write wrong fallbacks. Keep the fallback vocabulary to the three verbs above.
- *Cost and rate limits.* Speculative requests that are discarded still consume tokens and subscription quota.
- *Replay.* Deterministic replay must capture local check decisions. The checks must be deterministic to keep that asset.

**Third order.**

- Once calls drop, motion becomes the larger share of time. Only then do actuator limits matter: the 5 mm contact-profile cap, and the 22% of control steps in the trace that moved under 0.5 mm. Tuning motion before that point optimizes under a tenth of the time.
- Call count scales with decisions, not motions, so long tasks become practical. At today's 6.5 calls per pick-and-place, a ten-object task needs about 65 calls and 20 minutes of waiting.
- Attempts become cheap, which changes the best policy. A quick try with a local check can beat a long deliberation, so "few attempts" becomes a cost trade the harness can make per step.
- The sequencer is new code where task-specific shortcuts can hide. The existing no-task-token lint should cover it from day one.

### Across task types

| Task family | Effect |
| --- | --- |
| Pick-and-place | Largest reduction in calls; the checks are simple |
| Articulated parts | Guarded pull or push along an estimated axis; end stops detected by stall, not by image |
| Insertion and tight placement (bowl in drawer, mug in microwave, bottle on rack) | Contact-triggered transitions matter more than planning |
| Continuous visual judgment (pouring, wiping, deformables) | Little gain. These need a learned closed-loop skill the model can call, not finer plans |
| Long multi-object tasks | The gain compounds with task length |

**Complexity.** This adds a sequencer and a small set of checks. It removes most checkpoint traffic and two classes of protocol repair. The plan schema grows, and that is the cost to watch.

## Area 3: Make generality measurable

Build the instrument the goal implies: many randomized episodes, tasks the design was never tuned on, and a robot interface thin enough to swap. Until it exists, "few attempts" and "not benchmark-specific" are intentions, not properties.

### What changes

1. **Batch evaluation.** At least 10 episodes per task over randomized initial states and perturbed scenes (object pose jitter, distractors, camera jitter, lighting), run in parallel without per-episode approval. Today a launch needs `--approved` and a frozen reset hash, and each custom scene has one initial state.
2. **Fixed endpoints.** Success with a confidence interval, first-attempt success, calls per subtask, wall time split into model, motion and perception, forbidden-contact steps, and localization error.
3. **A development set and a held-out set.** Tune on the first. Run the second only at gates, and include one different robot or simulator in it.
4. **Failure attribution from traces.** Label each failed or slow episode by layer: perception, decision, interface, control, verification.
5. **One thin robot interface.** Observe, camera models, link shapes, servo to pose, gripper, limits. Panda and OSC specifics move behind it, such as the plant coefficients written into `robot_guard.py` and the grip-site offsets written into the prompts.
6. **A pruned tree.** Frozen baselines become git tags. The working tree keeps only what V3 imports, then shrinks from 31,000 lines toward 10,000.
7. **Replay as a regression gate.** Saved-response replay already reproduces a run exactly. Use it in CI for every local-layer change, at no model cost.

### Consequences

**First order.** Every later change gets a number. The properties that have never been observed, few attempts and generality, become measurable.

**Second order, on other modules.**

- *Cost.* At the size of the v2 run, 20 tasks × 10 episodes is about 17 hours of episode time and 128 million input tokens per sweep. That is about $1,300 at the [API list price](https://developers.openai.com/api/docs/models/gpt-6-astra.md) of $10 per million; the subscription changes the bill, not the scale. Area 2's call reduction is what makes sweeps affordable, and replay covers local-layer changes for free.
- *Statistics.* The model is stochastic. Ten episodes on one task give a 95% interval of about ±31 points; 200 pooled episodes give about ±7. Compare variants on paired seeds and report pooled numbers.
- *Reproducibility.* Randomized resets end the bit-exact reset guarantee. Log seeds and hashes instead of requiring them.
- *Spend control.* Dropping per-episode approval needs per-sweep caps on calls and time. The harness has none, by design.
- *Frozen modules.* The interface refactor ends byte-identical r1 files. Baselines live in tags from then on.

**Third order.**

- The team's loop changes from watching one video to reading a table. That is faster and less biased, but qualitative insight is lost unless failure clips are surfaced by category.
- Held-out results will probably be lower than results on the scenes the design grew up on. Expect the headline number to fall before it rises.
- A second robot forces honesty in the scene and grasp layers, which assume a parallel-jaw hand throughout.
- With attribution, effort follows data. If most failures are perception, prompt tuning stops.
- The suite becomes the new thing the design can be fitted to. Rotate the held-out set.

**Complexity.** This is the only area that removes more than it adds. The 41,000 lines unreachable from V3 leave the tree, and the seven-class client inheritance chain under the V3 client can be flattened.

## Cross-disciplinary analogies and the target design

Fields that pair a slow, capable decision-maker with a fast physical process have converged on one shape: slow deliberation sets intent, a middle layer holds state and checks outcomes, and a fast layer enforces limits. The table gives the mechanism each field uses and what it implies here.

| Field | Mechanism there | What it implies here |
| --- | --- | --- |
| Motor neuroscience | Cortex plans. The cerebellum predicts the sensory result of each command and flags mismatches. Spinal reflexes act within tens of milliseconds | Each plan step ships with its expected observation. Only mismatches go up to the model (Area 2) |
| Aviation, fly-by-wire | The pilot commands intent. Envelope protection enforces limits. Authority steps down through defined laws when sensors degrade | A guard independent of the planner, with graded authority for perception in place of all or nothing (Area 1) |
| Image-guided surgery | Register the patient to the image once, then navigate against it. Virtual fixtures hold the tool in a safe corridor near tissue | Register the scene, then act. Add approach corridors in the contact phase, where every recorded contact occurred (Area 1) |
| Precision assembly | Compliance and guarded moves absorb position error where measuring more would cost more | Lowering until contact removes the need to know support height (Area 2) |
| Computer architecture | Amdahl's law, caches, speculative execution | Motion is under a tenth of the time, so speeding it up is capped near 8%. The scene state is a cache of expensive perception. Thinking ahead is speculation (Area 2) |
| Compilers and network protocols | A stable intermediate form between many front ends and back ends. Mechanical errors are fixed or reported before running. A narrow waist | The plan as intermediate form, auto-repair of envelope violations, one thin robot interface (Areas 2 and 3) |
| Military mission command | Orders state intent and constraints, not movements. Subordinates adapt and report exceptions | The model gives the goal and the success test. The harness picks the waypoints (Area 2) |
| Distributed systems | A snapshot plus recent events, in place of replaying the whole log | v2 replays the full log on every call (7,945 input tokens growing to 87,425). V3 keeps almost no log. Neither keeps a snapshot (Areas 1 and 2) |
| Clinical research | A case report is not a trial. Endpoints are fixed in advance. Validation is held out | One episode is a case report. Fix endpoints and sample size before changing the design (Area 3) |

### Why this design, and not one more swing

No design can be proven globally optimal, but this one beats both extremes the tree has tried on each stated property. The all-classical extreme, the native stack, was precise where its rules applied and brittle outside them, at about 24,000 lines. The all-model extreme is simple and general in principle, but slow and centimetre-accurate. An end-to-end learned policy would be a third extreme, and it breaks the zero-shot rule.

The layered design keeps the model's generality for decisions and takes precision and speed from measurement, with no task rules in either. Robotics has long called this split a three-layer architecture. What is new is a top layer capable enough to need no task programming.

&#91;embedded content: target design · three layers and an offline evaluation loop\]

Read it top down. The model is called once per decision. The accented layer measures, checks and repairs between calls. The control layer and guard run on every step, and the offline loop scores all of them.

## Sequenced plan, metrics and decision gates

Measurement comes first, the cheap repairs run alongside it, and the scene state lands before the plan format changes. Each phase ends at a gate, and a phase that misses its gate is not built upon.

&#91;embedded content: roadmap · 5 phases, 5 gates\]

Phase 1 can run alongside Phase 0; every other phase waits for the gate before it. Gate 2 is the one decision gate: if the RGB-only fallback misses 10 mm, admit sensor depth instead of extending the phase.

Two decisions belong in Phase 0: whether depth is admitted, and what counts as an attempt. No durations are given because team size is unknown. In relative terms Phase 2 is the largest, and Phases 0 and 1 are the smallest.

| Metric | Recorded today (v2, T04, one episode) | Target | Gate |
| --- | --- | --- | --- |
| Success rate with interval, development and held-out sets | Unknown | Measured at Gate 0; held-out within 10 points of development | 0, 4 |
| Model calls per pick-and-place subtask | 6.5 | 5 or fewer, then 2 or fewer | 1, 2 |
| Cached share of input tokens | 2.5% | 70% or more | 1 |
| Control steps with forbidden contact | 106 | Down 80% or more | 2 |
| Target localization error against simulator truth | Median 4 cm over 6 estimates (2.5 to 5.7 cm) | 10 mm median or better | 2 |
| Episode wall time, T04 with the can | 304.8 s, ended by the evaluator before release | 100 s or less, including release and finish | 3 |
| Share of wall time that is robot motion | 9% | 30% or more | 3 |
| Source lines in the working tree | 72,000 | Under 10,000 | 4 |

The targets are proposals. Revise them once the Phase 0 baseline exists.

## Risks, open questions and what not to do

The plan fails in two likely ways: the scene layer grows back into a rule engine, or perception gets blunt veto power and deadlocks the robot. The code's history shows both tendencies, so the guardrails below are part of the plan.

### Risks

| Risk | Early sign | Guardrail |
| --- | --- | --- |
| The scene layer relapses into the abandoned rule stack | Numbered special cases; object nouns in scene code | Size budget; no object- or relation-specific rules; lint extended to new modules |
| False vetoes deadlock the robot | "Blocked" outcomes rise while contacts do not fall | Graded authority; slow guarded motion in unknown space; logged model override |
| Local checks accept failures silently | First-attempt success looks high while task success falls | Score each check against simulator truth; escalate when uncertain |
| Evaluation is too expensive to run | Sweeps skipped before merges | Replay tier for local changes; small development set; Area 2 before large sweeps |
| The suite becomes the new overfit target | The held-out gap widens | Rotate held-out tasks; keep a second robot or simulator in the set |
| The RGB-only fallback cannot reach 10 mm | The localization gate is missed | Decision gate: admit sensor depth |

### Open questions

- [ ] Is RGB-only sensing a requirement, or an inherited benchmark convention?
- [ ] What counts as an attempt: a retry inside an episode, or a whole episode? This report assumes recoveries inside an episode, plus first-attempt success.
- [ ] Is the ChatGPT-subscription transport fixed? It may limit control over caching and latency tiers; the repos do not say.
- [ ] Which real robot, if any, is the target? It decides the interface and whether force sensing is available for guarded moves.
- [ ] Are live V3 results available for r1 to r4? The V3 repo contains none, and the Phase 0 baseline needs them.

### What not to do

- Do not raise motion speed further now. It addresses under a tenth of the time.
- Do not keep tuning prompts or router parameters before Phase 0. Without a sweep, their effect cannot be told from noise.
- Do not add a second model as planner or critic. It adds calls, and calls are the scarce resource.
- Do not make vision a binary gate. V3 appears to have backed away from that design for good reason.
- Do not rebuild the native geometry stack. Use pretrained perception and keep rules out of the scene layer.
