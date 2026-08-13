# Fresh-seed confirmation of locomotion-zero hold

## Status and motivation

The lifecycle-invalid brake/handoff panel produced one strong but non-promotable hypothesis: do not switch to the stationary actor. Keep the frozen Stage219/Stage264 locomotion actor active and give it a zero velocity command. In the viewed panel this completed hold in 16/16 screening lanes and 250/256 validation lanes that reached the transition, while direct stationary handoff caused 218 hold terminations.

This document does not reinterpret v5b as a pass. It defines the minimum fresh-data confirmation needed before the controller can enter an official MuJoCo panel.

## Confirmatory design

- Three new reset seeds: `786101`, `786102`, `786103`; none may be replaced.
- Three independent Isaac launches, each 256 environments and the same 820-step, 16.4-second event.
- Within every launch, 128 lanes use `direct_mix` and 128 use `locomotion_zero`. Assignment is checkerboard-balanced over environment id and gait-offset stratum; direct/candidate parity flips between seeds.
- Both treatments are definitionally identical through the end of deceleration. They differ only at terminal hold: direct switches to the frozen mixed stationary backend; candidate keeps the locomotion actor under zero command.
- The random reset panel, source actor, action template, ideal actuator endpoint, fixed upper body, and all measurement definitions remain those of v5b.
- No posture residual, stationary actor in the candidate, reward inspection, optimizer, backward pass, checkpoint, export, or Baidu access.

The analysis unit is first the environment trajectory within seed and finally the three seed aggregates. The 384 candidate lanes are not called 384 independent experiments.

## Validity gates

Every launch must have finite complete rows, exact code/input hashes, zero timeout, unchanged source model, zero optimizer/backward/checkpoint, and autonomous lifecycle exit. Success evidence must be fsynced and the process must call `os._exit(0)` without `SimulationApp.close()`; a 300-second supervisor deadline remains fail-closed.

Before terminal hold, treatment cohorts must be balanced and safe:

- cruise/deceleration candidate termination count no larger than direct plus one lane per seed;
- velocity RMSE difference at most 0.010/0.015 m/s;
- lateral/yaw difference at most 0.015/0.020;
- support difference at most 0.002/0.003 m;
- slip p95 difference at most 0.030 m/s;
- root height and tilt retain the frozen absolute safety bounds.

Candidate hold must satisfy in every seed:

- zero hold termination and timeout;
- at least 98% of candidate lanes reach hold (pre-hold baseline failures are reported separately);
- speed p95 at most 0.12 m/s;
- double-support fraction at least 0.98;
- root height at least 0.60 m and tilt at most 0.35 rad;
- support outside at most 0.015 m, slip p95 at most 0.15 m/s, flight at most 0.01;
- action slew at most the frozen source-relative bound;
- the hold metrics above pass both pooled summaries and every seed summary.

The candidate must also reduce pooled hold terminations by at least 90% relative to `direct_mix`. Confidence intervals may be reported, but with only three top-level seeds they are descriptive and cannot substitute for the all-seed gates.

## Decision tree

- All validity and efficacy gates pass: `PASS_LOCOMOTION_ZERO_OFFICIAL_PANEL_PREREG_ONLY`. This only permits writing and reviewing an official MuJoCo start/cruise/stop panel preregistration; launch, deployment, and training remain locked.
- Candidate retains hold in all seeds but misses a secondary locomotion metric: `PARTIAL_LOCOMOTION_ZERO_BRAKE_SKILL_PREREG_ONLY`. This permits only a separately preregistered brake/hold skill design.
- Any candidate hold termination, unstable seed, or major safety regression: `FAIL_LOCOMOTION_ZERO_NEW_ACTOR_PREREG_ONLY`. Stop handoff-state-machine work and move to a new full-capacity actor/training design.

No result directly unlocks training, whole-body control, export, or deployment.
