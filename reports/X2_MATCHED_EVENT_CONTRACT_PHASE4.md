# X2 matched-event contract audit — Phase 4

Date: 2026-08-09
Scope: official AimDK v1.0 MuJoCo, no real robot, no long training

## Outcome

The previous 4/5 result for Stage347 remains a valid improvement under the shared
legacy deployment gate, but it is not evidence that the actor passes the event
used during training.  After matching the official evaluation to the training
clock (1.0 s acceleration, 4.2 s cruise, 2.0 s deceleration), both the frozen
Stage306 source and the one-update Stage347 branch fail the strict gate.

Long training therefore remains locked.  The current failure is a reproducible
long-horizon yaw/lateral-stability defect in the BASE contract, not evidence
that the one-update branch uniquely damaged the source actor.

## Hypothesis

The apparent Stage347 improvement might depend on a train/eval mismatch: its
training event used a smooth start/cruise/stop clock, while the legacy official
gate used an abrupt 0.3 m/s command, a four-second move and a deployment brake
supervisor.

## Intervention

The official adapter now exposes the same C1 speed event as training:

```text
stand -> accelerate 1.0 s -> cruise 4.2 s -> decelerate 2.0 s -> hold
```

The schedule is implemented as a pure tested function and is also used for the
future-intent preview.  Gate summaries record the event and supervisor
parameters.  The runner rejects CycloneDDS domains outside 0..232; three
initial Stage350 attempts above that range were infrastructure-invalid and
were rerun at valid domain IDs.

## Controls and results

| Stage | Moving actor | Deployment correction | Heading gain | Strict pass | Startup | Mean heading max | Mean stop drift | Mean stop root-z |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 349 | Stage347 aligned-u1 | legacy lateral supervisor | 0 | 0/5 | 5/5 | 0.537 rad | 0.327 m | 0.135 m |
| 350 | frozen Stage306 | legacy lateral supervisor | 0 | 0/5 | 4/5 | 0.518 rad | 0.321 m | 0.135 m |
| 351 | frozen Stage306 | none | 0 | 0/1 | 0/1 | 0.438 rad | 0.448 m | 0.619 m |
| 352 | frozen Stage306 | none | 1 | 0/1 | 1/1 | 0.407 rad | 0.328 m | 0.616 m |
| 353 | frozen Stage306 | none; preserve episode heading | 1 | 0/1 | 1/1 | 0.404 rad | 0.312 m | 0.610 m |

Strict thresholds relevant here are 0.30 rad maximum move heading deviation,
0.30 m move lateral displacement and 0.15 m post-stop XY drift.

## Interpretation

1. Stage349 and Stage350 are almost identical.  The matched-event failure
   predates the aligned one-update branch; it is not primarily caused by that
   update.
2. Removing the legacy lateral supervisor avoids the terminal fall in the
   single Stage351/352 probes.  The supervisor was harmful under the new event,
   but it was not the only problem.
3. Restoring the training-time heading gain improves heading and drift and
   restores startup, but Stage352 remains outside every horizontal gate.  It is
   a partial mechanism check, not a promotable controller.
4. Preserving the episode-initial heading instead of rebasing after the stand
   actor's approximately -0.14 rad yaw changes little: Stage353 remains outside
   move and stop gates.  This cleanly rejects heading-target ownership as the
   primary remaining cause, so the one-repeat pre-screen is not expanded.
5. Training return and episode length cannot be used as promotion evidence:
   the Phase3 five-update branch improved both while official performance fell.

## Decision

- Do not promote Stage349--353.
- Do not run a heading-gain sweep.
- Do not unlock 25-update or long training.
- Keep Stage347 only as a diagnostic legacy-gate checkpoint; keep Stage306 as
  the frozen BASE source.
- Next BASE work must restore the complete event/controller state at reset
  (clock, previous action, gait phase and command state) and pass a zero-update
  official matched-control probe before any PPO update.
- WBT remains independent: repair initial reference qvel and official-contact
  event alignment before expanding dynamic-teacher search.

## Evidence boundary

Stages351--353 each use one deterministic official-physics repeat.  They are
pre-screens used to reject or retain a mechanism, not robustness evidence.  No
claim is made about real X2 hardware.
