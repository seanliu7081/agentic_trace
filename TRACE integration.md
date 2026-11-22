# TRACE integration

Nov 21, 2026

TRACE fits the harness as a numeric trajectory service behind the middle layer, and the split you describe is right in outline. Three changes make it work: the harness owns the scene and every model call, each TRACE result is bound to one scene snapshot, and TRACE stops at a standoff short of contact. On the one recorded run, that standoff rule matters more than mid-air obstacle avoidance.

Based on the TRACE method note v0.1 (2026-11-20) and the recorded v2 run on task T04. TRACE is a design with no benchmark yet, so every TRACE number here is a target.

## What moves where

TRACE keeps its trajectory function, search, program tests and certificate. Everything that touches perception, the model or the robot's body moves to the harness, so each fact has one owner.

One difference from your wording: every contact-free segment should go through TRACE's contract, not only the complex ones. The fast path keeps simple moves cheap, and a second motion path would need its own margins and its own checks.

| TRACE note | In the combined loop | Reason |
| --- | --- | --- |
| §3 trajectory, §6 search | TRACE keeps both. A fast path certifies the straight line before any search | 3 of the 4 trajectories in the recorded task pass as straight lines |
| §4 scene model | The harness builds it. TRACE reads one versioned snapshot and derives its own blurred copies | TRACE and the guard must check the same geometry |
| §5.3 primitive library | Shared. TRACE plans with it, the sequencer monitors with it, outcome checks verify with it | One vocabulary for what should hold before, during and after a motion |
| §5.4 goal pose | The model names an entity, a part and a relation. TRACE's end-pose solve returns ranked candidates | The model no longer has to choose among candidates |
| §5.4 safety terms | The harness writes them from the scene state and the robot model | The model cannot create or weaken them |
| §5.4 language compiler | Merged into the harness plan call. TRACE makes no model calls | Model calls are the scarce resource |
| §5.5 five compiles per instruction | Off by default. Parallel low-effort copies of the plan call, only for steps with unusual terms | At about 18 s a call, five serial compiles for each of four trajectories is 360 s. The whole recorded episode took 305 s |
| §5.5 perception uncertainty | Kept. The scene state must export a covariance per entity | A new required output of perception |
| §7.2 arm check | Roll the guard's one-step controller model along the path, in place of a separate IK solve | The arm follows the controller's own redundancy resolution, not TRACE's |
| §7.3 repair | The sequencer applies relaxations the plan allowed in advance. Otherwise one model call carries the quote | Replaces trial and error. In the recording the model tried clearances of 2.5 cm, 6 cm (rejected) and 4 cm |
| Stages and gripper events | The sequencer owns them. One request covers one contact-free segment | The carried object's pose in the hand is known only after the grasp is checked |

## One trajectory request, four results

A request has four possible results, not two, and each maps to one reaction in the sequencer. "Certified" and "conflict with a quote" are the two you named. The other two exist because perception can be too uncertain to certify anything, and because a failed search is not proof that no path exists.

&#91;embedded content: trajectory request · 4 results, 4 reactions\]

Look, relax and next candidate all lead to a new request. Only two reactions can cost a model call: a conflict with no allowed relaxation, and a search with no candidates left. A failed program test comes back before any search and is handled like an interface repair.

## The contract

The harness should depend on this contract, not on TRACE's internals, so a simpler solver can stand in until TRACE is proven. A request and a certified response could look like this. The names are placeholders.

```json
{
  "scene_version": 41,
  "start": {"pose": "measured", "joints": "measured", "holding": "white mug"},
  "end": {"entity": "left plate", "relation": "above", "standoff": "auto"},
  "terms": [
    {"prim": "tilt_limit", "of": "white mug", "level": "upright",
     "role": "require", "relax_to": "tilted"},
    {"prim": "clearance", "from": "can", "level": "wide", "role": "prefer"}
  ],
  "budget_ms": 500
}
```

```json
{
  "result": "certified",
  "trajectory": "spline plus timed samples",
  "valid_while": {"scene_version": 41, "tube_mm": 15, "holding": "white mug"},
  "report": [
    {"term": "scene clearance", "role": "safety", "max_violation": 0.0},
    {"term": "upright", "role": "require", "max_violation": 0.4}
  ],
  "quote": null,
  "look_at": null
}
```

Five rules sit behind those fields:

- **Named levels, not numbers.** The model picks a level such as "upright" or "wide". A harness table turns it into a tolerance. This closes the note's own limitation that tolerances are model-chosen, and it removes unit mistakes at the source.
- **Allowed relaxations travel with the plan.** `relax_to` lets the sequencer accept a quote without a model call.
- **A certificate is valid for one snapshot.** It holds for one scene version, one start state, one grasp and one tracking tube. The sequencer voids it when the scene changes inside the tube, when the object-follows-gripper check fails, or when tracking leaves the tube.
- **A margin ladder.** TRACE plans with the guard distance plus the tracking tube plus the perception bound. The guard stops at the guard distance alone. A certified path tracked inside its tube then cannot stop the guard unless the world changed.
- **Seeded and logged.** TRACE draws its random samples from a seed in the request, and its response is saved in the trace like a model response. Saved-response replay then stays exact.

## Where the boundary sits

TRACE should own every contact-free segment (reach, carry, retreat) and end each one at a standoff. The harness owns the last centimetre into contact, the gripper event and the check after it. The recorded run shows why, and how far back the standoff has to be.

| Measured in the recorded run | Value | What it means for the split |
| --- | --- | --- |
| Trajectories needed before the recording ends | 4. A straight line clears every object by 20 mm or more in 3 of them. In the white-mug carry it cuts 19 mm into the can | Certify the straight line first and search only when it fails |
| Gap between a grasped mug and its neighbour at pickup | About 3 mm to the can and 21 mm to the red mug | No planning margin can hold at departure or arrival. Near both ends the rule has to be "do not get closer", not "stay beyond the margin" |
| Where the hand met the mug in the two grasps | 8 and 9 mm above the commanded depth. The hand then pressed on for 6 and 5 mm | TRACE's end pose should aim for palm clearance, and a guarded move should end the descent on contact |
| Time to notice that contact | 13 control steps of pressing before the stall detector returned control on the second grasp. It never fired on the first | Guarded moves need a contact detector that fires within a step or two. The V3 guard already predicts each step, so its prediction error is a ready signal |
| Tracking of straight transits at about 0.5 m/s | Up to 13 mm sideways and 8 mm of overshoot. Final error of 4.8 and 4.9 mm when carrying. Tilt within 2.5° of vertical | The certificate needs a tracking tube of about 15 mm at this speed. The note's 5 mm default goal tolerance equals what the arm achieves under load, so standoff goals need a looser level |
| Control steps spent in the slow contact profile | 305 of 549, covering 36 cm at about 2.5 cm/s | Most of that distance is contact-free. A certified approach and retreat would save about 8 s of the 27 s of motion |

All three bursts of forbidden contact in the recording began in the last few millimetres of a vertical descent. In both grasps the onset height matches the palm reaching the top of the mug handle, to within about 3 mm. The palm sits 31 mm behind the grip point. The harness stores that figure, but the v2 request never passes it to the model.

So the error was a bias, not noise. The workable depth window on these handles is roughly 1 to 2 cm, and the commands aimed below it by under a centimetre. A hand-shape clearance term in TRACE's end-pose solve removes the bias. Ending the descent on contact covers the noise that remains.

The general rule: an axis whose workable window is smaller than perception error plus tracking error must be closed by contact, not by planning. TRACE's "not certifiable with this perception" flag detects exactly that case, so the certificate can tell the sequencer where to end the segment. The standoff becomes a computed distance, not a constant.

The path search earns its place on the carry. The old router tracked one point and could not see the carried mug, which a straight line would have driven 19 mm into the can.

Method: truth is each object's collision geometry at the saved reset pose, with resting heights inferred from the settled can. The hand is a palm box and two finger boxes built from the harness's stored hand dimensions. This is one episode.

## Second- and third-order consequences

The integration removes a router and adds a solver, but its wider effect is to make margins, tolerances and unknown space explicit quantities that several modules must agree on. Most of the risks below are disagreements between modules.

### On other modules

- **Guard.** It stays independent and uses a smaller margin than TRACE. A guard stop on a certified path becomes a defect to log and explain: the scene changed, tracking left the tube, or the two disagree about geometry. Without the shared snapshot and the margin ladder, the two can deadlock, each rejecting what the other accepts.
- **Perception.** A covariance per entity becomes a required output. Unknown space should reach TRACE as a `require` term the sequencer may relax to slow guarded motion, not as occupied space. With one external camera every object casts an unseen region behind it, and the note's default of treating unseen space as occupied would close those regions to planning.
- **Model contract.** The plan call now emits terms, levels and allowed relaxations. The model-written obstacle map, its 4-box limit and the per-stage limits of 0.5 m and 1.2 rad go. Removing the stage limits also removes the rejected 90° turn seen in the recording.
- **Speed.** Retiming should set speed from certified clearance: fast where there is room and slow in tight passages, because the tracking tube grows with speed. This is where the 8 s of slow contact-free motion is recovered.
- **Replay and evaluation.** Every request and response is logged, which builds a regression set at no model cost. The contact monitor can score certificates: a certified path that touches something in simulation is a measured defect. Re-checking the same path on the true scene splits that defect into perception error and tracking error.

### On task types

| Task family | Fit | Note |
| --- | --- | --- |
| Pick-and-place, containers, insertion | Full | Approach cones and standoffs do the work. The arm check matters inside cabinets and the microwave |
| Articulated parts (drawers, doors) | Partial | A guarded pull estimates the joint axis. TRACE could then plan the rest of the arc as a path-equality term. Untested |
| Pushing, pouring, wiping | None | TRACE has no contact dynamics. These stay with harness primitives or a learned skill |
| Thin, shiny or transparent objects | Reduced | The "not certifiable" flag will fire often, which sends more axes to contact and asks for more looks |

### On complexity and risk

- **Net code.** TRACE adds a package and a contract. It removes the grid router and its planner chain, the model-written obstacle map, the stage limits and the unused `assess_path`.
- **Research risk.** TRACE has no benchmark yet. Version 0 behind the contract can be: certify the straight line, else seed from the existing grid planner, then certify. TRACE's search replaces that only when it wins on the logged requests.
- **Margin stacking.** Guard distance, tracking tube and perception bound add up to roughly 3 to 4 cm at the Gate 2 perception target. Passages narrower than the hand plus twice that will read as blocked. The remedies are slower motion and a closer look, and the quote should name which one would open the passage.
- **False infeasibility.** "No path found" is not proof that none exists, since the note says its seeds are not topologically complete. The sequencer must treat it as a search failure, not as a fact about the scene.
- **Contact exemptions.** `allow_contact` is the one place a model statement can weaken a safety term. Limit it to finger geometry against the step's named target part, inside the final standoff zone.

## Merged build order and gates

TRACE's first four test stages need no model and no perception, so they can run beside harness Phases 0 and 1. The two meet in Phase 2, once the scene state exists.

| Harness phase | TRACE work alongside | Added to the gate |
| --- | --- | --- |
| 0 Instrument | Build steps 1 to 4 and Stage 1 (solver grid) on synthetic scenes. Log every transit of the baseline runs as a request. Export true scenes from the simulator | A request set from real runs exists |
| 1 Quick repairs | Stages 2 and 3 (search structure, representation), on synthetic scenes plus the logged requests | None |
| 2 Scene state | Contract version 0 behind the sequencer, replacing the grid router. Then TRACE's search replaces the internals. Stage 4 (program tests) | No guard stop on a certified path when planning on the true scene |
| 3 Guarded plans | Plan steps carry terms, levels and allowed relaxations. Quotes drive the fallbacks. Stage 5 (language and fusion) becomes the plan-call experiments | At least half of conflicts resolved without a model call |
| 4 Generality | Stage 6. A second robot through the thin interface: hand shape, controller model, tracking tube | Certificates hold on held-out tasks |

Four measures join the plan's table. The targets are proposals.

| Measure | Target |
| --- | --- |
| Share of requests answered by the fast path | Report; 3 of 4 in the recorded task |
| Solve time, 95th percentile | 0.5 s for the fast path, 2 s for a full search |
| Certificate honesty: certified paths that run with no forbidden contact and every hard term within tolerance on the true scene | 95% or more |
| Guard stops per certified path | Near zero; each one explained |

At the note's starting budgets a full search is about 39,000 trajectory evaluations, before any leave-one-out re-solves. With 20 collision spheres per pose, the distance lookups alone took about 4 s on one CPU core in a NumPy timing. A full search therefore needs the perception GPU or a smaller budget, and where it still does not fit it should run during model latency or the previous motion.

## Open questions

- [ ] TRACE takes RGB-D input. Does that settle the depth question in the main plan in favour of sensor depth?
- [ ] Is a brief, deliberate touch acceptable as the way to find grasp depth? The contact monitor counts palm contact as forbidden today, however short.
- [ ] Who builds TRACE, and does it share the perception GPU? The time targets assume it does.
- [ ] Should TRACE look one segment ahead, so a grasp is chosen only if the placement that follows is reachable with it?
