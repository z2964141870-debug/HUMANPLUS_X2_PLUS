# X2 state-feedback posture-and-stop teacher feasibility

## Decision

The Phase77p LoRA campaign is a valid negative result.  It is not extended to
update 10 and the preregistered-but-unrun feedback search v3 is not executed.
V3 still scales one frozen 15-D direction and disables the residual in the
terminal hold, so it cannot answer the current stop/hold question.

This campaign preserves the Stage219/Stage264 locomotion actor and the frozen
stationary actor.  It asks a narrower question before any policy training:

> Can privileged, state-feedback lower-body corrections improve backward
> pitch during 0.35 m/s cruise and deceleration, then hand off safely into an
> 8.8 second stationary hold?

Stage264 is a semantic deployment identity (Stage219 iteration 2600 plus the
Stage250 action contract), not a separate trainable checkpoint.

## Three treatments

All treatments replay the same reset state, gait clock and command schedule.

1. `source_direct`: frozen locomotion actor, followed by the historical direct
   blend to the frozen stationary actor.
2. `fixed_half_direct`: the frozen v2b direction at scale 0.5 while moving,
   followed by the same direct handoff.  This is a diagnostic control, not the
   proposed solution.
3. `feedback_gated`: a bounded 15-D state-feedback teacher during cruise,
   deceleration and braking; handoff occurs only after a dwell-qualified
   low-speed, double-support, low-tilt and support-safe gate.  A smaller
   feedback correction remains available around the stationary actor in hold.

The episode is 16.4 seconds at 50 Hz: 1.0 s stand, 1.0 s acceleration, 4.2 s
cruise, 2.0 s deceleration and at least 8.2 s hold.  Reset perturbations and a stratified command-clock
offset vary the initial state and the gait/contact phase at deceleration.

## Teacher

The privileged feature vector includes current/future stop intent, root
pitch/roll and angular velocity, body-frame velocity error, support outside
distance, realized left/right contact and the gait clock.  A searched linear
feedback matrix maps these features directly to all 15 lower/waist actions;
there is no single fixed direction or scalar dose.  Output is bounded, slew
limited and suppressed in non-finite/flight/unsafe states.

Search is gradient-free and does not use PPO, a critic or a scalar reward.
Candidates are ordered lexicographically:

1. finite execution, survival, root height, tilt and action bounds;
2. terminal speed/double support and regressions in support, slip, velocity,
   lateral motion and yaw;
3. cruise/deceleration pitch mean and p05 improvement;
4. residual magnitude and slew.

## Feasibility gates

The feedback treatment must satisfy all gates on reset-identical validation:

- zero termination and timeout for the candidate;
- cruise pitch mean improvement at least 0.020 rad and p05 at least 0.010 rad;
- deceleration pitch mean improvement at least 0.015 rad and p05 non-regression;
- hold root height at least 0.60 m, tilt at most 0.35 rad, terminal speed p95 at
  most 0.12 m/s and double-support fraction at least 0.95;
- velocity RMSE no more than source +0.010 m/s, lateral/yaw no more than source
  +0.015, support outside no more than source +0.001 m and slip p95 no more
  than source +0.030 m/s;
- normalized residual absolute value at most 0.12 and per-step slew at most
  0.015; final action clipping and non-finite values are fail-closed;
- reset fingerprints are exact across all treatments and the stationary/main
  source weights remain immutable.

## Routing

- Pass: freeze teacher traces, then separately preregister BC followed by one
  DAgger collection round.  PPO remains locked and any later critic starts
  from the new task rather than reusing the old terminal value.
- Cruise/deceleration pass but hold fail: build a separate brake/hold skill;
  do not alter the cruise actor.
- Teacher fail: stop Stage219/Stage264 adapter work and preregister a new full
  actor or from-scratch native X2 locomotion policy.

No result from this feasibility campaign directly unlocks deployment, whole-
body training, large motion datasets or robot execution.
