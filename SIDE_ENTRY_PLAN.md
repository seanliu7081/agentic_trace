# Side-entry placement plan (agentic_trace)

2026-11-23. Current build fe630da9 (its source equals d2d469a5). Evidence:
- full test `runs/eval_fe630da9_v2_high_cap600s50c_t00-t09_ep000_009_20261123T084931Z`;
- reader notes and the reachability probe in `runs/plan_side_entry_20261123/`.

## 1. Summary

- **Why objects only come down from above.** Every placement mode (ladder, v1, v2, and the drop planner) puts the
  object down vertically. This is how the place step is written, not a limit of the robot, LIBERO or TRACE.
- **What that cost.** t09 failed 10/10. Almost every put-down aimed at the microwave roof, and the arm had no posture
  above it. A spurious "opening" snap made things worse: in ep007 it put the mug on the open door three times.
- **What a real side entry needs.** Four pieces the harness lacks:
  1. a model of an opening that faces sideways;
  2. a grasp that keeps the hand outside the opening;
  3. a horizontal guarded insertion, with release and back-out;
  4. a way for System 1 to say "from the side".
- **Feasibility in t09 is not given.** A true-geometry probe on ep000 found two problems:
  - A horizontal grasp on the mug handle where the mug starts has no IK solution.
  - The straight horizontal insertion is reachable by only about 3 arm postures. The best of them passes 4.4 cm
    through the open door and 2.9 cm through the added obstacle.

  Angled entries, a hand pitched above the handle, and opening the door to its 120° limit are untested.
- **So the plan starts with a cheap gate (M0).** If no collision-free side entry exists in most t09 scenes, side entry
  will not lift t09. The aim-snap fix (§4 F) is worth doing either way.
- **Priority context.** t09 is 10 of the 35 failed episodes in the full test. Grasp failures cost a similar number
  (stage failures: no_feasible_grasp 41, grasp_failed 23), spread over t01, t04, t06 and t08.

## 2. What is known

**Code: every path is top-down.**
- `_v2_aim` ([harness.py:1673](src/atrace/harness.py#L1673)) reads System 1's pixel at the destination's top height.
- `_v2_put` ([harness.py:1752](src/atrace/harness.py#L1752)):
  - tries only poses straight above the aim (line 1776), with the hand turned about the vertical (1774);
  - moves straight down (1809) with a force guard on Fz only (1803);
  - opens the gripper fully (1813) and lifts +z (1819).
- `_plan_place` ([harness.py:2036](src/atrace/harness.py#L2036)) builds its pre-pose straight above the spot (2244).
- Object grasps are top-down, plus tilts of up to 35° (`manip.py:87`), and tilt is penalised (`manip.py:197`).
- `plan_grasp` ignores the stage and its destination (`harness.py:1303`).

**System 1 already asked for a side entry, with no effect.**
- In t09 ep000, REPAIR wrote "approach through the open front rather than requiring an overhead approach", then asked
  for a "grasp compatible with front insertion".
- No field carries either request, so the regrasp was top-down again.

**Aim-snap defect** (`_v2_aim` compartment snap via `container_region`, `manip.py:232`):
- In ep000, four different pixels became the same roof point.
- In ep007, the aim was snapped 31 cm onto the open door's top edge. The mug was released there three times and ended
  upside down on the table.

**Geometry** (true MuJoCo model, `probe/RESULTS.md`):

| What | Facts |
|---|---|
| Opening | Faces world -y, sideways to the robot (which faces +x). 20.7 W × 14.4 H × 16.9 D cm; floor 4.4 cm above the table. |
| Door | Open about 90°; a slab at x ≈ -0.18, y -0.006 to 0.25, z 0.92 to 1.11. It walls off the robot's side of the space in front of the opening. |
| Obstacle | On the line straight out of the opening, about 40 cm in front of it; 0.17–0.24 m tall. |
| Mug | 10.4 cm tall, with 4 cm of headroom in the cavity. A top grasp needs about 10 cm of hand above the rim, so a top-grasped mug cannot go in. |
| Mug body | About 9.4 cm across, more than the 6.4 cm widest slab the gripper can hold, so the body cannot be grasped from the side. |
| Mug handle | The only side-graspable feature: a bar 0.8 × 1.8 × 5.8 cm, 8.3 cm from the axis. |
| Palm | ±10.4 cm along the closing axis, wider than the opening, so only the fingers can enter. A handle-held mug reaches about 17 cm ahead of the palm, about the cavity's depth. |

**Probe** (ep000; `probe/RESULTS.md`):
- No IK solution for a horizontal handle grasp at the start pose.
- The straight insertion collides with the door and the obstacle (numbers above).

**Prior art.**
- No harness has ever placed an object into a side opening, and r1 has never run t09.
- The joint key-pose study (`/workspace/joint_planning_20261118/order_experiment`) let System 1 choose `entry_from`,
  but its side branches never reached a placement.

**Collision rule.**
- Under your rule, robot or mug contact with the microwave and its door is a destination contact and does not count;
  the benchmark monitor counts it as minor. Obstacle contact is major.
- Pushing the door from the robot side swings it toward closed, into the path. The plan therefore avoids door contact
  for physical reasons, not because of the metric.

## 3. Goal and constraints

- **Generality.** One general "in, from the side" placement for any destination whose opening faces sideways:
  microwave, oven, cabinet shelf, cubby. No task-specific code and no thresholds tuned on t09; the System 1 prompt
  text may stay task-specific.
- **Real robot.** At run time only RGB-D and model calls. Simulator truth is for evaluation and the M0 probe only.
- **No regression.** Top-down paths stay unchanged when the opening faces up (t00–t08 behaviour).
- **Same test settings.** v2, effort high, 600 s / 50 calls.

## 4. Design

### A. Opening model (perception)

Add `side_opening(dest)` next to `container_region` ([manip.py:232](src/atrace/manip.py#L232)). It finds a mouth: a
roughly vertical face of the destination with a rectangle where depth lies behind the face plane. The plane fits work
as in `articulate.mount_faces` ([articulate.py:67](src/atrace/articulate.py#L67)).

It returns:
- the mouth plane (a point and the outward normal);
- the width × height rectangle;
- the floor height (lowest supporting interior point, else the mouth's bottom edge);
- the ceiling height.

The head camera cannot see inside (the roof hides it). One wrist look through the mouth (`look_toward`,
[harness.py:368](src/atrace/harness.py#L368)) carves the interior free before planning, because unknown space is an
obstacle by default. When a destination has a side mouth, top-view "openings" on its roof are ignored.

### B. System 1 interface

- `place_pose` gets an `entry` field: above / from image-left / image-right / toward the camera / away. It uses the
  same vocabulary as `long_side` (`PLACE_POSE_SCHEMA`, program.py:263; `PLACE_LOOK_SCHEMA`, :289).
- For a side entry the pixel means "the cavity floor seen through the opening" (`_POINT_RULES`, :235). The stage
  echo in `_v2_stage` ([harness.py:1645](src/atrace/harness.py#L1645)) carries the field.
- The harness turns `entry` into a horizontal world direction with the head-camera calibration, as `_v2_yaws` already
  does (harness.py:1742). It then checks that direction against the mouth normal from A. If they differ by more than
  45°, the geometry wins and both are logged.
- REPAIR gets an entry lever (`REPAIR_INSTRUCTIONS`, program.py:148), and failure notes name the entry that was tried.

### C. Grasp for insertion

`plan_grasp` ([harness.py:1299](src/atrace/harness.py#L1299)) learns the planned placement.

When the destination has a side mouth, it adds grasps whose hand stays outside the mouth:
- horizontal grasps, and grasps pitched from above-behind, with a horizontal closing axis;
- generated with tilts up to ±90° about the closing axis, or with the mount-frame trick of
  `articulate.part_grasps` ([articulate.py:283](src/atrace/articulate.py#L283)) on horizontal normals.

The pitched variant matters. A pinch on a vertical handle bar holds the mug upright at any hand pitch about the closing
axis. So the hand can come from above-behind, stay above the mouth's lower edge and outside the roof's front edge, and
still carry the mug in level.

Each candidate is scored on insertion fit:
- the payload fits the mouth with margins along the whole insertion;
- at the end pose the palm is outside the mouth plane (`_hand_local` / `_hand_block`, harness.py:2650/2659).

If no compatible grasp can be reached where the object is, it regrasps via a staging spot. The object is put down
(careful planner) at a reachable spot in front of the destination, with its graspable side toward the robot, and
grasped again.

### D. Insertion motion (new `insert_object`)

Used by v2, and by ladder/v1 if D5 says so, when the entry is sideways.

1. **Transit.** TRACE moves the hand (`reach`, [harness.py:672](src/atrace/harness.py#L672)) to a staging pose in front
   of the mouth: payload fully outside with about 2 cm margin, approach = the entry axis, entity = the destination.
2. **Line check.** The whole insertion line is checked before moving, following the `_press_stroke` pattern
   ([harness.py:3405](src/atrace/harness.py#L3405)) but without excluding the destination. Arm, palm and payload must
   clear the mouth frame, the floor lip, the ceiling and the door.
3. **Guarded insertion.** A straight line in, with the payload 1.5 cm above the floor, at 4–5 cm/s. The guard is the
   ContactDetector (direction-agnostic, guard.py:31) plus force along the motion. This replaces the Fz-only rule at
   harness.py:1803 for this path.
4. **Release.** Lower to the floor with an Fz guard (at most 1.5 cm), then open the gripper only partway. The opening
   is the `gripper_to` value checked by `_hand_block(release=True)`, as `_plan_place` does, instead of a full open.
5. **Back out.** Move out along -entry, guarded, then the usual retreat. Also give `retreat`'s move_line fallback a
   guard (harness.py:689–698).
6. **Check.** v2's `place_look` with a side-view variant: the wrist camera looks into the mouth after backing out. The
   PLACE_LOOK wording is adapted for side openings.

### E. Making room (searched, not scripted)

If no collision-free insertion exists:
1. try angled entries: the entry turned ±20–35° in yaw, and a hand pitch per C;
2. move an articulated part that limits the approach to the end of its range, as an actuation step. In t09 that means
   opening the door from about 90° toward its 120° limit;
3. tell System 1 which constraint blocks (door, obstacle or reach) so REPAIR can choose.

### F. Aim-snap fix (independent, small, general)

- Only snap to an opening within a few cm of System 1's point.
- Ignore "openings" that have no floor below their rim, such as the tops of a roof or door.

This removes both t09 failure modes above: different pixels merging into one roof point, and the mug released on the
door. It can also change put-downs in t03, t05 and t08, which use the same snap.

## 5. Milestones

| | Work | Checks (only the necessary ones) | Size |
|---|---|---|---|
| **M0** | **Feasibility gate**, evaluator-side, no policy. Extend the probe to: entry yaw 0/±20/±35°, hand pitch 0/15/30/45/60° (handle pinch, mug upright), door at its episode angle and at 120°, all 10 t09 episodes (the obstacle varies), handle grasps at the start pose and at staging spots. | Per episode: does any insertion line clear the door, obstacle, microwave frame and table by ≥ 1 cm, and which variant? **Gate:** fewer than about 3/10 → stop t09 work after M1. | ~30 min |
| M1 | F (aim-snap fix). | Unit test on the recorded ep000/ep007 frames (aims stay at the pointed spot); tests once. | ~1 h |
| M2 | A + B (opening model, entry field, interior look). | Offline on recorded t09 frames: mouth plane, normal and rectangle against the true geometry, as an evaluator-side comparison. | ~half day |
| M3 | C (grasp for insertion, staging regrasp). | Geometric unit tests (the palm stays outside the mouth; payload fit); tests once. | ~half day |
| M4 | D, plus the E options M0 found necessary. | Tests once, then one real end-to-end t09 episode chosen from M0's feasible set. This also measures the ContactDetector on horizontal guarded moves carrying a payload, which has never been calibrated. | ~1 day |
| M5 | Batch t09 ep00–09, same settings. | One spot-check episode each of t00, t05 and t07, because the guard, release and grasp code is shared. | ~1.5 h compute |

Not planned: review or verification workflows, or re-measuring facts already measured here.

## 6. Decisions for you

- **D1. M0 first?** Run M0 first and decide from its result (recommended), or build directly.
- **D2. Who decides the entry direction?** System 1 proposes and geometry checks (recommended), geometry only, or
  System 1 only.
- **D3. Making room (E).** May the search open an articulated part further, like the microwave door, and use angled or
  pitched entries? Robot–door contact is not counted by your rule but is a minor collision for the monitor.
- **D4. Regrasp via a staging spot.** Allowed? It costs about 30–60 s and 1–2 model calls.
- **D5. Scope.** v2 only, or also ladder/v1 when the entry is sideways?
- **D6. Dev and test split.** Every t09 episode has now been tested, so new t09 numbers are seen-scene results. Develop
  on a LIBERO-90 microwave scene (same scene family as t09) or on a separate side-opening scene (cabinet shelf)?

## 7. Risks

- **Kinematics is the bottleneck.** The Panda is near its limits pointing sideways at microwave height: about 3 IK
  postures for the straight entry in ep000. M0 decides whether this is fatal.
- **Guard calibration.** The ContactDetector thresholds were measured on vertical descents and certified moves only
  (guard.py:3–12). False stops are possible on horizontal moves with a payload; M4's run measures this.
- **Handle grasp on a real robot.** The simulated mug weighs about 0.03 kg; a real one about 0.3 kg on an 8 cm lever,
  so it can twist in the grip. The tilt_limit term (levels.py:79–83) keeps the planned pose upright but cannot prevent
  slip.
- **Interior perception.** The cavity is dark and narrow, and one wrist look may not carve it fully. Unknown space then
  allows only slow guarded motion, as TRACE already permits.
- **Time.** An insertion adds a look, a staging pose and a back-out (about 40–80 s), more with a regrasp, and t09 still
  has to close the door afterwards.
