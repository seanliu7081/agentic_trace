# Contact stage: general plan (agentic_trace)

2026-11-23. Written on build fe630da9. M1-M3 implemented the same day (section 8).

Evidence:
- the full test `runs/eval_fe630da9_v2_high_cap600s50c_t00-t09_ep000_009_20261123T084931Z` (100 episodes, 65 succeeded,
  115 failed stages);
- reader results, notes and scripts in `runs/plan_contact_stage_20261123/` (`results/`, `notes/`, `scratch/`).

Words used below:
- **S1** is the model that writes the stage program and points at pixels. **S2** is the harness. **TRACE** plans
  contact-free motion.
- **Hand-off pose**: the last pose TRACE certifies. From there S2 moves into contact with a guarded motion.
- **Contact stage**: grasp, put down or insert, press or push, grip and pull or turn.

## 1. Summary

- **The failures are not tied to one kind of contact.** 115 stages failed: grasp 64, put-down 44, actuation 7. Their
  causes sit in four places that every kind of contact shares:
  1. the hand-off between TRACE and the guarded motion;
  2. how the result of a guarded motion or gripper action is read;
  3. what the scene model looks like after a contact;
  4. the direction of approach, which is fixed.
- **Side entry is one case of item 4.** t09 is 10 of the 35 failed episodes. The other 25 fail in vertical contacts,
  most of them for reasons a new approach direction does not touch.
- **The unified plan has the right aim and about half the right content.** Its premise about solid cavities is true but
  narrow, its contact primitive lacks the piece the data asks for (an expected contact), and it does not cover grasp
  transit, strokes, or recovery. Section 3 lists what to keep.
- **Proposal.** One record describes any contact. One hand-off, one executor and a few scene rules serve all kinds.
  The kind of contact only supplies the candidate list and the check.
- **Order.** M1 hand-off and scene rules, M2 executor on grasp and put-down, M3 strokes, M4 approach direction, M5 full
  test. M1–M3 act on all ten tasks; M4 is what t09 and the t03 bowl need.

## 2. What the last run shows

### Failed stages by kind and root cause

| Kind | Failed stages | Root cause (count) |
|---|---|---|
| Grasp | 64 | The model handed to the planner is not the object: 26 (another object or background 13, an occluded sliver 5, a part only after S1 re-pointed 4, the object still hanging on the hand 4). The move to the pre-grasp pose failed: 22. No hand or arm solution on a correct model: 9. Dropped at lift: 6. Closed empty: 1. |
| Put-down | 44 | "Could not reach above the point": 34 (25 no arm posture straight above, 6 no path, 3 mixed). Released and judged not done: 2. Object lost: 8 (7 after a release, 1 after a drop in transit). |
| Actuation | 7 | t03 drawer 5: the stroke stops short and the continuation is refused, the wrong face is pushed, or the part is re-grounded as the whole cabinet front. t02 knob 2: the move to the part failed (both episodes recovered). |

Of the 35 failed episodes, 21 start with a put-down problem (t09 10, t08 6, others 5), 11 with a grasp of an object
that had not been moved, 2 with actuation, and 1 (t03 ep001) reports the drawer closed when it is open.

### Shared mechanisms

| # | Mechanism | Measured | Kinds hit |
|---|---|---|---|
| 1 | One fixed pose is handed to TRACE | 447 of 447 pre-grasp requests carry one end pose. A put-down tries one pose at three heights. 59 first pre-grasp requests ask the hand to turn 120° or more where the gripper's 180° twin needs a median 36°; 78 % of them fail on the straight line against 48 % of the rest. | all |
| 2 | S2 and TRACE judge the same pose by different rules | All 152 end-pose rejections above a place point are "pose overlaps the scene"; none is a missing arm posture, and S2 had passed each one on. For a grasp or a press S2 tests with the object, or the part and its fixture, excluded and 4–8 mm margins; for a put-down it tests nothing. TRACE exempts only the fingers against the target and uses 17.5–22.5 mm. All 6 refused press approaches are this mismatch. | all |
| 3 | Unseen space counts as solid at the full margin | About half of the 152 overlap unseen space (65 unseen only, 10 both); with unseen space taken out of the safety term, 42–47 of them pass every hand and held-object term offline. Unseen space binds in 25 of 82 failed pre-grasp moves. In 2,961 requests the "look" reaction fired 3 times and the "slow" relaxation 3 times. | all, and camera looks |
| 4 | The scene map forgets | It is rebuilt from the last 8 camera frames at 16 of 17 call sites. In t04 ep009 a refused point certified at once after a re-grasp, when the unseen voxels above the plate fell from 856 to 197. | all |
| 5 | The gripper acts on any stop | 24 of 60 set-downs were released 3.8–26 cm above the support, most on a first touch of something else (t03 10 of 10 on the handle of the drawer above, t08 second pot 6 of 6 on the first pot). All 15 "not done" verdicts on set-downs are among them; all 36 releases within 1.7 cm of the support were done. After a close, a width within −6 to +4 mm of the predicted one held 151 of 158 times; more than 6 mm narrower held 4 of 15. Nothing tests that the hand is empty: the bowl stayed hooked on a finger in 4 stages. | put-down, grasp, press |
| 6 | A stroke is not closed on its effect | The first drawer push on the correct face was planned with a predicted 9 mm overlap in all 6 episodes. In 5 of them a push stopped at 40 N on a protrusion 0.6–3.8 cm before flush. The next push is planned as a fresh approach; in 3 of the 5 it was refused by the 4–8 mm hand margins against the part's own surroundings. In t03 ep001 "closed" was accepted after 0.1 mm of hand travel. | push, pull, turn |
| 7 | Object models degrade after a contact | A released object is found again by segmentation: a pot model 37 cm wide, the stove absorbing the pots, two pots swapping identity (inferred). 0 of 21 "not done" releases were recovered inside the stage. | put-down, then grasp |
| 8 | Failure reports carry no cause | S1 is told the object's size and the gripper opening, or "no path". 19 of 29 repairs after a grasp failure blamed width, which was never the cause. Pointing again after "could not reach" failed 113 of 121 times: 157 calls, 26 % of all model calls. | all |
| 9 | The direction is fixed | Every put-down comes straight down; every grasp approaches within 35° of vertical. | t09 (side cavity), t03 (the bowl must go down, then in under the handle above), grasps near or far from the base (held 14 of 81 under 0.35 m or over 0.80 m from the shoulder, 158 of 271 between) |

A refused or missing hand-off pose is the direct cause of 50 failed stages and a contributing cause of 19 more. About
half of the 50 are the pose straight above t09's roof (row 9); the rest come from rows 1–4. Outside t09, none of the
58 "could not reach" put-down attempts showed that the spot was out of reach.

### What the 65 % does not show

- In at least 58 of the 65 successes the episode ended while the last object was still in the hand above its
  destination, because LIBERO's check fires there. The last release ran in 3 successes.
- Releases that did run: on an empty support 38 of 38 done; into a basket (drop) 28 of 29; into a container, set down,
  7 of 16 (all 7 are falls off the drawer handle); onto a support that already holds an object 0 of 6; beside 0 of 5.

### Costs inside contact stages

- About 1,900 s of 25,231 s control time went into moves that TRACE refused.
- Grasp planning took 4,135 s (11.5 s per call). A press pose search took 60–123 s, 53 % of all t03 actuation time.

### How these numbers were obtained

- Four readers returned full results (hand-off, grasp, put-down, code inventory). Two were cut off by a session
  restart (actuation, premise check of the unified plan); their notes were complete and I finished the open items.
- Reproduced by me from the logs and the code: the 152 rejections and what they overlap, the 8-frame rebuild, the
  fill that overwrites free space, unseen space inside the safety term, the closed widths (same counts), the twin
  statistic, and the count of successes that ended with an object in the hand. For the release heights I reproduced
  the pattern with a cruder reference height: 20 high releases hold 11 of the 15 "not done" verdicts, and the other 4
  are t09 releases whose aim was already on the roof.
- Taken from the readers without a second pass: the root-cause split of the 64 grasp stages, the actuation timelines,
  the offline count of poses that would pass without unseen space (42–47 of 152), and one kinematic probe (t04 ep007:
  the query found no posture where all 12 turns about the vertical are reachable).

## 3. The two existing plans

**SIDE_ENTRY_PLAN.md.** Its t09 facts stand. Its design adds a side-insertion routine next to the existing ones, which
is the per-type route. Each of its pieces is the side-opening case of a general item below: the opening model
(4.5), the grasp that keeps the hand outside (4.5), the guarded insertion (4.3), making room (4.2, typed reasons), and
the aim-snap fix (4.5, rest region from all geometry). Its feasibility probe stays as the first step of M4.

**unified_frame_invariant_contact_placement_plan.md.**

| Item | Verdict | Reason |
|---|---|---|
| A. Replace the 2.5D solid fill with ray-carved 3D occupancy | Shrink to two map rules (4.4) | The map is already a ray-carved 3D grid with free, occupied and unseen states. The fill is one step, [perception.py:389](src/atrace/perception.py#L389), run after carving on every rebuild, and it overwrites space a camera saw as free. Roofed fixtures become solid (microwave 50–65 % of the volume under the roof, cabinet 77 %). Baskets and the open drawer are already free inside. |
| A. Approach corridors found by sweeping along exit rays | Keep and widen | Use it for every toucher (hand for grasp and press, held object for put-down) and allow two segments. |
| B. One guarded primitive with a stop in any direction | Keep the idea, change the content | The stop is already direction-free. The only directional rule, `(F − F0)[2] > 4`, has the wrong sign in 3 recorded descents of an earlier run, so the 61 support stops of this run were the contact detector's (inferred). What is missing is an expected contact (row 5). A stop in any direction without one makes early releases more frequent. |
| B. Release width from clearance around the fingers | Keep | The ladder path already does it; v2 opens fully. |
| C. Grasp chosen from pick-feasible ∩ place-feasible, staging regrasp | Keep, as a look-ahead check | [plan_grasp](src/atrace/harness.py#L1299) takes no destination today. |
| D. Continuous cone and band optimisation inside TRACE | Change | TRACE can only choose among given ends (the spline end is pinned, at most 2 ends are kept). The failures come from one pose per request and from rule mismatch, not from a missing optimiser. A small set of valid ends from the contact logic is enough. |
| D. Approach hint from S1, blocker attribution | Keep and extend | Reasons are computed and then dropped: the worst sphere position is stripped and the arm-check reason never leaves TRACE. Extend to typed reasons for every refusal (row 8). |
| Phase gates, held-out probe before starting, task-token lint | Trim | The probe is needed only before M4. The "existing" lint does not exist in this repository. |
| Not in the plan | Add | Grasp transit and object models (64 failed stages), strokes (t03), the release gate, recovery after a failed put-down, the unseen-space rule, map memory. |

## 4. Design

### 4.1 One record for every contact

What stays: S1's pointing and look loop, TRACE's path search, the candidate generators, the contact detector, the
compliant stroke. What changes is how they are joined.

A contact action is data. The kind of contact fills in the fields; no kind has its own motion code.

| Field | Grasp | Put down, insert | Press, push | Grip and pull or turn |
|---|---|---|---|---|
| Toucher | open pads | held object | closed fingertips | open pads |
| May touch | the object | the support or cavity floor | the part | the part |
| Candidates: contact pose and approach axis | opposing faces within the stroke | rest pose near S1's pixel, with its entry path | a patch on the part's face | opposing faces, in the face frame |
| Expected contact on the approach | none | the support, at a known travel | the part, at a known travel | none |
| Gripper action and expected reading | close; predicted width | open to a width; hand empty | none | close; predicted width |
| Motion in contact and expected effect | short test lift; the object follows | none | compliant translation; part state | compliant translation or rotation; part state or visible effect |
| Depart | lift with the object | back along the entry | back along the approach | open, back along the approach |
| Check | object follows the hand | rest pose measured, then S1's look | part state or effect | part state or effect |

Today the same 13 steps are written out in seven routines, with 4 rigid-hand tests, 5 pre-pose and line checks, 6 force
rules and separate literals for the same quantities (approach speed 0.04, 0.05, 0.06 m/s; force 8, 10, 12 N).

### 4.2 One hand-off

1. **One rulebook.** The record's "may touch" pairs are passed to every check: S2's sweep, TRACE's end test and
   end-zone terms, the per-step guard, and the reading of a stop. S2 stops re-implementing the end test and calls
   TRACE's own function on the same snapshot. A pose S2 proposes is then certifiable by construction, or both refuse it
   for the same stated reason.
2. **A set, not a point.** The candidate provider returns several different contact poses. For each, S2 adds the
   members the task leaves free:
   - the 180° twin about the approach axis (pads and fingertips);
   - any turn about the support normal when S1 says the long side does not matter (236 of 252 answers);
   - an interval along the approach.

   Each member is solved with its arm posture, with enough seeds that existing solutions are not missed. TRACE receives
   them as different ends and picks one.
3. **The hand-off sits where certification ends.** S2 slides each candidate back along its approach until TRACE's end
   test passes. The guarded line covers exactly the rest and is swept with contact margins against everything but the
   allowed pairs. Departures mirror it: leave contact guarded along the reverse approach until the end test passes,
   then TRACE. This replaces the fixed 4, 10, 16 cm heights, the fixed 6, 8, 10 cm pre-poses and the 44 retreats that
   fell back to an unchecked straight line.
4. **Unseen space is looked at, then crossed slowly.** The safety clearance measures against observed geometry only
   ([snapshot.py:73](src/atrace/snapshot.py#L73) passes a default that counts unseen space). When unseen space blocks
   a hand-off, the end test says so, S2 aims one look at that volume along the approach, and what stays unseen is
   crossed at guarded speed. The mechanism for "slow" already exists in TRACE; it fired 3 times in the run.
5. **Every refusal has a typed reason, and S2 acts on it first.** End not admissible (which body, which entity or
   unseen space, by how many millimetres), arm (which joint, where), path (where). S2 answers with the twin, another
   candidate, a look or a slower motion. S1 is called only when the reason is one a new pixel can change.
6. **Ordered tolerances and bounded search.** Thresholds in candidate generation, hand test, kinematics and
   certification are ordered so that a pose accepted upstream cannot fail downstream on residue (today's thresholds
   leave a flat box 0.3 mm of slack). The search stops at the first few feasible candidates or returns its main
   rejection.

### 4.3 One executor

- **One guarded motion for every leg that touches or may touch**: approach, motion in contact, depart. It always runs
  the contact detector, uses one force convention (along the motion), and returns one outcome record: cause, travel,
  distance short of the goal, force, finger opening, nearest thing. Twists, strokes and retreats run outside it today.
- **Every leg carries its expectation** (4.1). The outcome is compared with it, and the reaction is the same for every
  kind of contact:

| Outcome | Reaction |
|---|---|
| Contact inside the expected window, reading as predicted | Next leg. |
| Contact before the window, or where none was expected | Leave the gripper as it is. Back off along the approach, add the contact point to the map, take the next candidate. No model call. |
| No contact by the end of the window | The target is not where the model says: look and re-localise. |
| Closed width outside −6 to +4 mm of the predicted one | Open, back off, exclude that contact region, take the next candidate. No lift. |
| The object still follows the hand after opening | Not released. Treat it as held and release again. |
| Stall or force limit before the effect | If the hand is still on the part, re-seat a few millimetres and continue from there, with no new transit. Otherwise take the next contact point. |
| The arm cannot leave its pose | Guarded back-out along the approach before anything else is tried. |

- **Release gate.** A set-down opens the gripper only when contact arrived inside the window around the support
  height. A drop opens it only when the footprint is over the opening and the fall height is bounded. The opening
  width comes from the clearance around the fingers.
- **Strokes close on the effect.** The stroke repeats from the current contact until the part's state or the visible
  effect is reached, or two strokes make no progress. A measured part displacement must agree with the hand's travel.

### 4.4 Scene rules

1. **Free space is remembered.** A voxel seen free stays free across rebuilds until something is seen or placed there.
2. **Seen-free beats inferred fill.** The fill under an object's top never overwrites space a camera saw as free. A
   cavity under a roof becomes free once a camera has looked into it.
3. **One object, one model.** The held object's shape is fixed at the grasp. After a release or a failure it is
   re-localised by registering that shape to fresh depth, not by segmenting again. A destination does not absorb what
   is put on it. What lies in a part moves with the part. A part S1 names is a preferred contact region on the whole
   model.
4. **Look where the plan needs it.** The contact regions and the volume the hand and held object sweep must have been
   observed before a candidate is used. If not, the next action is a look aimed there.

### 4.5 The direction is a variable

- **Candidates in any frame.** The grasp generator already runs in any frame for parts
  ([articulate.py:283](src/atrace/articulate.py#L283) turns the cloud so that a chosen axis is "up"). Objects get the
  same: reference directions are up, the horizontal ring, and S1's hint, kept when the toucher's sweep from the
  hand-off to the contact is free of everything but the target.
- **Rest region from all observed geometry.** The support patch or cavity floor, minus what stands on it, minus
  overhangs of its fixture and of neighbours, with the footprint and the open hand fitted in. S1's pixel ranks the
  candidates; geometry decides the pose. This replaces the snap to a top-view "opening" that put the aim on the
  microwave roof and door.
- **The entry is searched.** A free sweep of held object and hand from the hand-off to the rest pose: vertical,
  oblique, horizontal, or two segments (down, then in). A side opening and a drawer under an overhang are the same
  case.
- **Look-ahead.** A grasp is chosen only if the put-down that follows has a feasible entry with that grasp, tested
  with the same solve at the destination. When no single grasp serves both, a regrasp at a staging spot is planned.

### 4.6 What S1 sees

- An optional `entry` hint on a transfer stage and in the place-pose answer (above, image left, image right, toward
  the camera, away). Geometry decides; a disagreement is logged.
- Typed failure reports with matching levers, in place of "size and gripper opening" and "no path".
- No further pointing call when the reason is one a pixel cannot change.

### 4.7 Where the code changes

| Area | Files | Change |
|---|---|---|
| Contact record, solve, executor, reactions | new `src/atrace/contact.py` | Holds 4.1–4.3. |
| Callers | `src/atrace/harness.py` | `grasp_object`, `_v2_put`, `_grip_stroke`, `_press_stroke`, `_actuate`, `_actuate_linear` keep their candidate providers and checks and lose their motion code. Four functions without callers go (one here, three in `motion.py`). |
| Candidates | `src/atrace/manip.py`, `articulate.py` | Generator in a given frame for objects; width limit from the real opening; rest region from all geometry. |
| Scene rules | `src/atrace/perception.py`, `snapshot.py`, `motion.py` | 4.4 rules 1–2; `exact` honours the unseen flag; exact distances up to the largest sphere plus margin. |
| Guard | `src/atrace/guard.py` | Known geometry only; speed cap in unseen space; one force convention. |
| TRACE | `src/traceplan/search.py`, `service.py`, `contract.py` | End test shared and driven by the allowed pairs; number of kept ends from the request; typed reasons in every refusal. |
| S1 contract | `src/atrace/program.py` | 4.6. |

## 5. Milestones

Each milestone ends with the tests once and the real episodes named. No other checks are planned.

| | Work | Addresses | Check | Size |
|---|---|---|---|---|
| **M1** | Hand-off (4.2) and scene rules 1, 2, 4 (4.4). | Refused hand-offs outside roofed fixtures: about half of the 50 failed stages they cause directly, the 19 they contribute to, and the 57 pointing calls outside t09 that could not help. | Replay of the logged pre-grasp and above-place requests of the full test (no simulator, no model). Then t04 ep007 and t01 ep001. | 1–1.5 days |
| **M2** | Executor (4.3) on grasp and put-down; rest region from all geometry near S1's pixel; scene rule 3; typed reports (4.6). | Early releases (24), lifts after a wrong close, the object still on the hand, 0 of 21 recoveries, palm on the obstacle during descents. | t08 ep006 (second pot) and t00 ep004 (can on the basket rim, then recovery). | about 2 days |
| **M3** | Strokes through the executor: continuation in contact, effect loop, travel consistency; press when a grip closes empty. | t03's stroke failures and false "closed"; the t09 door. | t03 ep008 and t03 ep001. | about 1 day |
| **M4** | Direction (4.5): entry search, any-frame grasp candidates, look-ahead, `entry` hint. Starts with the t09 probe from SIDE_ENTRY_PLAN.md (30 min, evaluator side). | t09, the t03 bowl under the handle, grasps near and far from the base. | t03 ep006 and one t09 episode from the probe's feasible set. This run also measures the contact detector on a horizontal move with a payload. | about 2 days |
| **M5** | Full test, same settings as the last one. | — | — | about 2 h compute |

Not planned: review or verification workflows, a new perception stack, learned contact skills.

## 6. Decisions for you

- **D1. Order.** Shared layers first, direction last (recommended): M1–M3 act on all ten tasks. Or direction first,
  which targets t09 and t03 only.
- **D2. Unseen space.** Look first, then cross what stays unseen at guarded speed (recommended). The cost is a possible
  touch of something unseen at about 5 cm/s, which the monitor counts. The alternative keeps unseen space solid and
  keeps row 3.
- **D3. Regrasp at a staging spot** when no single grasp serves pick and put-down. Allowed? It costs about 30–60 s.
  t09 probably needs it: the first probe found no posture for a horizontal handle grasp where the mug starts.
- **D4. Making room.** When the reason names an articulated part as the blocker, may S2 move it to its limit on its
  own, or does S1 decide in a repair?
- **D5. Placement modes.** After M5, keep one put-down path (recommended) or keep the ladder, v1 and v2 flags.
- **D6. Objects the head camera cannot see.** 13 failed grasp stages (t06 9, t00 2, t03 2) start from a model of the
  obstacle or the background, because S1 can only point in the head image. This is outside the contact stage. Add
  "S1 may point in a wrist image" now, or leave it.

## 7. Risks and limits

- **The arm.** A wider set of approach directions helps near and far grasps, but the Panda sets the limit. For t09 the
  first probe found about three postures for a straight side entry, and a path through the door and the obstacle.
- **Touches while probing.** An early contact is now answered by backing off, but the touch itself has happened. The
  same holds for slow motion through unseen space.
- **Registration after a release** can fail when the object is hidden; the stage then asks S1 to point, as today.
- **The contact detector** was measured on vertical descents only. It sees translation along the commanded step, not
  rotation or sideways sliding.
- **Time.** Looks and back-offs add seconds per stage. The refused moves (about 1,900 s) and long searches are where
  time should come back.
- **Not covered.** Contact dynamics such as pouring, wiping and in-hand slip. An action's "motion in contact" field can
  later name an external skill.

## 8. Decisions and implementation (2026-11-23)

Decisions (user): D1 shared layers first (M1-M3 now). D2 look first, then cross unseen space at guarded speed.
D3 regrasp at a staging spot allowed. D4 the harness may move an articulated blocker on its own. D5 keep ladder, v1
and v2 and add the unified put-down as a fourth choice (`--placement unified`). D6 System 1 may point in the wrist
image, now.

Implemented (M1-M3): see README "Placement modes" (unified, and what every mode shares). D3 and D4 belong to M4
(look-ahead regrasp; making room) and are not implemented yet. Changes against the design above, from the check
episodes:
- 4.3: the closed-width window is -6..+10 mm (the full test held 14 of 16 closes at +4..+10 mm).
- 4.2: TRACE gives its search the whole budget after screening the set of ends (with 4-6 ends the screening had
  starved the search: `levels_run` 0-1); at most 4 straight lines are certified per request.
- 4.4: a grasp posture keeps 0.08 rad from every joint limit and its 10 cm lift must have a posture: an object close
  to the base (t04 ep007, 0.32 m from the shoulder) is now reported as "no arm posture" instead of being grasped in a
  pose the arm cannot leave. Approach directions other than near-vertical (M4) are what reaches it.
- Strokes: a part of a part (a handle on a drawer) is operated as the outermost moving part, measured against its
  fixture (a handle "flush with its drawer front" read as closed in t03).
- Unified aim: the surface under System 1's pixel (the unified prompt asks for the floor of the opening), not the
  ray at the destination's top (a basket's handle made that height wrong).
- Unified put-down "in": no flat-rest requirement (any pose counts inside; a narrow basket's sloped liner made every
  spot "uneven"), what already lies in the container counts as its floor, walls must surround the spot, and a drop
  opens once the object is below the rim (on contact with the container or at the planned pose).
- Departures and re-seat back-offs are guarded by the contact detector only (a released press force looked like a
  force along the motion); a re-seat backs off along the reverse of the press approach and repeats while each one
  still moves the part (at most 6).
- A measured part shift larger than the hand's travel (by 2 cm or 15 %) is not accepted; if the hand did not move,
  the earlier measurement stands.

Checks (the plan's named episodes, full-test settings: effort high, 600 s / 50 calls, the full test's instructions;
runs in `runs/contact_stage_dev_20261123/`, builds frozen in `builds/<hash>/`):

| Milestone | Episode | Full test (fe630da9) | After M1-M3 (build of the run) |
|---|---|---|---|
| M1 | t01 ep001, v2 | fail (pre-grasp moves refused) | success, 214 s, 8 calls (2a971151) |
| M1 | t04 ep007, v2 | fail (arm stuck at the grasp pose, 20 calls) | fail, 265 s, 5 calls (616b49e5): the second mug sits 0.32 m from the shoulder; typed "no arm posture" (needs M4's approach directions) |
| M2 | t08 ep006, unified | fail (second pot released on the first) | success, 162 s, 5 calls (ee80eef7) |
| M2 | t00 ep004, unified | fail (600 s cap) | success, 174 s, 7 calls (f4de7c00) |
| M3 | t03 ep001, v2 | fail (drawer reported closed while open) | success, 350 s, 5 calls (f4de7c00) |
| M3 | t03 ep008, v2 | fail (drawer 0.6 cm short) | success, 317 s, 4 calls (2c9ff702) |
| sanity | t00 ep000, unified | success | success, 220 s, 12 calls (f4de7c00) |
| sanity | t04 ep000, unified | success, 216 s, 4 calls | success, 262 s, 11 calls (cf233cb9) |

No run touched the added obstacle. The latest build is 2c9ff702; runs on earlier builds were not repeated on it.

Offline (no simulator): the full test's 152 refused place hand-offs replayed with the new end test are all admissible,
55 as logged (unseen space was the only blocker) and 97 after sliding up a median 8 cm (`replay_m1_ends.py`).

### M4 (user: "do M4", 2026-11-23)

**Probe first (evaluator side, `runs/plan_side_entry_20261123/probe/RESULTS.md`).** With exact clearance, carrying the
t09 mug in by its handle is feasible in all 10 episodes (hand pitch 0-45 degrees). At the episode's own door angle it
is feasible in 8 of 10; ep3 and ep9 need the door opened further (D4). Margins to the microwave are 0.4-1.2 cm. A
handle grasp at 45 degrees is reachable at the start in every episode. The earlier side-entry probe numbers were
wrong: `mj_geomDistance` (MuJoCo 3.2.6) returns bogus negative box-box distances, so the probes now use
`sdf_clear.py`.

**Implemented** (placement `unified`; ladder, v1 and v2 unchanged except the side-grasp fallback, which every mode
gets when no top-down grasp fits or has a posture):
- 4.5 rest region: the support is read at the pointed level (surfaces higher than the pointed surface + the object's
  height + 3 cm are not supports). "In" is decided by the destination's walls: in at least 6 of 8 directions, the first
  surface that rises above the object's bottom belongs to the destination. That means its label, one of its parts, its
  fixture, or a cell inside its footprint hull, at any distance up to 30 cm; a roof over a cavity counts. "On" is
  decided by the footprint share (next point). Spots whose support varies by more than half the object's height
  across the footprint (on a wall top or rim edge) are rejected before the 3D checks.
- (user, same day) "On" is a footprint share: at least 60 % of all the object's footprint cells lie over the
  destination's own surface, meaning inside its outline (its model and the unlabelled surface connected to it), above
  the table by more than 4 mm, and not another object's surface. Its labels decide only where geometry cannot: the
  destination has no outline, or its top is less than 8 mm above the table (`ContactMixin._footprint_share`). This
  replaces "half of the supporting cells labelled or inside the hull", which let an object hanging over a plate's edge
  pass. Check: t04 ep000 (build 48cdf28c), success, 461 s, 11 calls, no contacts. Both plates were found by the share,
  with about 300 spots per search rejected as not on the plate. The time over the batch's 258 s came from a failed
  first grasp (arm at a joint limit, about 110 s with the repair) and four hand-offs blocked in transit by unsegmented
  surface. The look-ahead before each grasp costs 19-29 s even when the top-down grasp is kept.
- 4.5 entry search (`entry_search`): straight down, a 45-degree line, or a slide (in at 2-3.5 cm above the rest, from
  where hand and object can rise, then the set-down). Slides leave only through the spot's open sides (rays without a
  wall), System 1's hint and the arm's side, at most 4 directions. The arm needs a posture along the whole entry with
  6 mm clearance (`_entry_posture`), and the thing it would meet is named. The put-down runs the entry as guarded legs
  (no contact expected) and the last one with its contact window. After the release the hand leaves along the reverse
  of its approach (side grasps), then back out along the entry.
- Rest-pose details: the release opening follows `release_width`'s ladder; a "place" may let go up to 3 cm above the
  rest when the open hand is not clear lower. The costly 3D checks (20) are spread over the hand's turns: before this,
  all of them went to the nearest spots at one turn. When the long side does not matter, or the hand holds the object
  from the side, there are 12 turns.
- Any-frame grasps (`manip.side_grasp_candidates`): the cloud is turned so that a horizontal or 45-degree direction
  is up, the top-down generator runs there, and the picks are spread over the object (one per 2 cm), so a handle is
  kept next to the rim.
- Look-ahead (`lookahead_grasp`): System 1's place-pose call is made before the grasp (no extra call). The usual
  grasps are judged by the same rest-pose solve for the payload they would hold. When none works and the hand was in
  the way, side grasps are tried, the hand lowest first and never from below. The spot is looked at first while the
  hand is free when it is not seen well enough: from its open side, level with it, when something is above.
- D3 regrasp (`plan_regrasp`): grasps of the held object whose put-down works (same solve), and a staging spot
  (where the object was taken from) where one of them can be taken; executed as put-down + `grasp_object(want=...)`.
  It is triggered only when at least 8 spots reached the 3D stage and hand/entry/arm reasons dominate.
- D4 making room (`make_room`): a part of a fixture named for most 3D-stage rejections, or by a TRACE refusal, and not
  the destination itself, is moved to its open end stop. The object is set down first and taken again afterwards.
  Once per stage.
- 4.6 the `entry` hint in the unified place-pose answer (`from_above`, `from_image_left/right`, `from_camera_side`,
  `from_far_side`, `any`): it is tried first, and agreement is logged. The program-level hint was not added; the
  place-pose call comes before every put-down.

**Checks (2026-11-23).** t03 ep006 and t09 ep008 (the probe's widest feasible set), unified placement, full-test
settings, runs `runs/contact_stage_dev_20261123/m4{a..m}_*` (one build per round). Neither episode succeeded. Each round
exposed one gap, fixed before the next (the list above includes the fixes):
- rest-region rules: the walls test looked only 4 cm out (a 21 cm drawer has its far wall 7 cm away); a partial
  segment of the destination made its roof and walls "not the destination's" (now: its connected unlabelled surface
  counts); the 3D budget went to one turn of the hand; the release check tried only the fully open hand.
- look-ahead and regrasp: side grasps were rim grasps from above (now: spread over the object, hand lowest first,
  never from below); the regrasp spent its budget generating alternatives; looks into a roofed spot came from the
  arm's side (now: from the open side), and one failed look blocked all later ones in the stage.
- D4 fired on the destination's own drawer and on a handle named by 4 of 25 spots (now: not the destination; the part
  must be named for most 3D rejections).

Where the two stand on the last builds:
- **t03 ep006 (bowl into the open bottom drawer):** every spot inside the drawer reaches the 3D stage and fails on the
  hand at the release. With a rim grasp the bowl's rim is 1.5 cm below the drawer walls' tops, so the palm meets them.
  On the side toward the cabinet, the palm meets the cabinet face and the middle drawer's handle. The one fitting
  orientation (closing axis along the drawer's width, bowl centre 12-13 cm from its front) is a band of about 1 cm. The
  3 cm "place" release and 12 turns did not find it. System 1's repair then sets the bowl aside and tries to pull the
  drawer further (no grip found). Unsolved; candidates: a grasp chosen for this put-down (the look-ahead judged only
  3 top-down grasps), or the drop release for "in" when the hand is the only blocker.
- **t09 ep008 (mug into the microwave):** the cavity is recognised as "in" (roof and walls as the microwave's) once
  System 1 points at the cavity floor and the look from the opening succeeds. Before that, System 1 pointed at the
  microwave's front panel in about half of its answers. The remaining blocker is perception: the microwave's model gets
  the inferred solid fill under its observed roof (M1 rule: fill below observed tops except where a camera saw free
  space). The cavity was seen only partly (from the head camera at a grazing angle, one wrist look from the front), so
  most of its interior stays occupied in the map (1,474 occupied voxels labelled microwave inside the cavity in the last
  snapshot). The mug "overlaps" it at every deep spot, so no rest pose reaches the entry search and the side-grasp
  regrasp was never reached with a valid put-down. Next: the inferred fill must not close a space that the
  destination's own walls enclose from the side (a roofed object can be hollow), or the cavity must be looked into
  until it is seen free.
- Not exercised by a run: a slide entry end to end (so the contact detector on a horizontal move with a payload was
  not measured), a D3 staging regrasp end to end, D4 on a real blocker (t09 ep3/ep9 need the door opened further). A
  look-ahead side grasp was never chosen.
- Unit tests: 68 pass (4 new: slide entry under a roof, handle pinch from the side generator, entry-hint words, the
  "on" footprint share).

### M4 fixes before M5 (user's order, 2026-11-23/24)

Implemented (all in `unified`; tests: 73 pass):
1. "On" footprint share: the destination's support region is its own points above the table (plus connected
   unlabelled surface), closed by 1.5 cm with holes filled; labels decide only when it is too sparse (< 30 points) or
   flat. Per-spot share diagnostics are logged. Look-ahead pruning: rest spots of the object alone once
   (`object_spots`, early exit when none), then per grasp the open hand at the release first and the entry and IK
   only after it (`_grasp_puts`), on a coarse preview of the top-down grasps (0.7 s, against ~11 s for the full set)
   and up to 16 side grasps.
2. Cavity: destinations of "in" stages are hollow in the map (`insert_object(hollow=True)`: a 2 cm shell below
   observed surfaces, the rest stays unknown). The prompt asks for the floor inside a side opening. An "in" pixel on a
   vertical face moves to the lowest destination surface behind it (`_inward_support`). A destination's moving parts
   (a door) are not enclosing walls.
3. Bounded drop for "in" only: release up to 4.5 cm above the rest with the tightest clear width, when the payload is
   clear at the floor, only the hand blocks there and raising it helps (`_blocked_below`), and the footprint is
   at least 1 cm inside the walls at the object's height. The 3 cm "place" drop was removed ("on" sets down only).
4. Found during the checks, all general: an object's unlabelled leftover at its pick location is cleared after
   every map rebuild (`_vacated`; it blocked the t04 transits). Pick checks use observed geometry (D2) and treat
   unsegmented surface hugging the object as the object (`free_near`, now also in the voxel test). A side grasp is
   checked against the object seen along its own direction. The 3D budget (36) is spread over turns and area. The
   `_look_at_rest` method lost in the refactor was restored, with a test that every called method exists.

Checks (`runs/contact_stage_dev_20261123/m4fix*`):
- t04 ep01 (m4fix2): **success**, 223 s, 4 calls, no added-obstacle contact, clean under the user's rule. Spots on
  the plate found (none before); look-ahead geometry 2.2 s and 2.5 s (plus 3.5-4 s for the place-pose call).
- t09 ep008 (m4fix12): the **first end-to-end side entry**: look-ahead chose the 45-degree handle pinch with a
  slide, but `plan_grasp` fell back to a rim grasp; with it the put-down still found a slide: down 2.6 cm, a 11.8 cm
  horizontal slide with the mug and the set-down on contact inside its window. **The contact detector made no false
  stop on the horizontal payload legs.** The mug was let go 2-3 cm too far out (front of the opening), tipped, and
  the recovery failed: 600 s cap, 10 calls, **2 added-obstacle onsets** (forearm, during the re-grasp of the tipped
  mug, not during the side entry).
- t03 ep006: not passing. The bounded drop clears the hand at 10 spots, but those have no free way in (the middle
  drawer's handle overhangs them). The pose that fits everything is the bowl within about 5 mm of the drawer's
  front wall, which the 5 mm grid counts as resting on the wall.
