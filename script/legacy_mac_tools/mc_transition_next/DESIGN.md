# Next handoff state machine

## Safety boundary

The pure machine decides which command contract must remain continuous and
whether exit is permitted.  It does not perform migration or publishing.  A
future adapter must treat every directive as a requested action that still
needs independent authorization and validation.

The central invariant is: after a `Develop_MC` intent has been emitted, an
uncertain result is candidate ownership.  A timeout, rejection, process exit,
feedback loss, Sonic loss, output gap, tracking fault, or vibration fault must
not silently remove the last command.  Only verified official overlap or a
confirmed physical E-stop permits release.

## State flow

```text
OBSERVE_OFFICIAL
  -> OVERLAP_OFFICIAL
  -> WAIT_DEVELOP                 (Develop_MC intent emitted)
  -> RAMP_CANDIDATE               (C2 contract blend, anchor target)
  -> CANDIDATE_HOLD
  -> ACTIVE                       (operator command, fresh Sonic required)
  -> RETURN_RAMP                  (C2 blend back to captured anchor/contract)
  -> WAIT_READY                   (Ready intent emitted only after settle)
  -> VERIFY_OFFICIAL              (fresh post-marker live official overlap)
  -> COMPLETE                     (candidate publisher may be released)

Any post-intent fault -> LOCKED_RECOVERY -> explicit handback path or ESTOP
Any pre-intent fault  -> ABORTED
```

## Freshness contract

Freshness is not callback recency alone.  Each joint group must provide:

1. A strictly newer uint32 sequence number.  No automatic reset is accepted.
2. Local receive age within its deadline.
3. Source-derived age within its deadline.
4. A bounded cross-group receive skew, so a mixed-time body pose is rejected.
5. A bounded observed inter-sample gap and a continuous stability dwell.

The adapter must derive source age from a monotonic source timestamp or a
validated clock-offset estimator.  If it cannot, the source-age field is not
valid and the preflight must fail.

## Command continuity

All leg, waist, arm, and head commands form one cycle and one ownership unit.
The adapter must report an independent output sequence, last-publish age,
maximum publish gap, and whether all groups were emitted from the same cycle.
A ROS timer callback merely being scheduled is not proof of output continuity.

Before migration, an output-continuity failure aborts while official control is
still verified.  After migration intent, the same failure latches recovery and
forbids normal exit.

## Contract ramp

`contract_blend` uses a quintic smoothstep with zero velocity and acceleration
at both ends.  The blend applies atomically to target, stiffness, damping,
feed-forward effort, velocity, limits, and joint ordering.  Target-only or
gain-only blending is prohibited.

The ramp first keeps the candidate target at the captured anchor.  Sonic motion
requires a later local operator command after a vibration-free hold dwell.  A
normal return ramps the same complete contract to the captured official
anchor, settles tracking, and only then emits a `Ready` intent.

These rules do not prove the interpolated controller is stable.  Every point
on the contract ramp and the anchor behavior still need closed-loop simulation
and suspended validation before any hardware run.

## Fault semantics

`LOCKED_RECOVERY` freezes the last fully accepted output and its current
contract blend.  Fresh packets never auto-resume motion.  This preserves
command continuity, but it is not a universal physical recovery controller.
For a balancing robot, a frozen target can still be unsafe.  Hardware remains
blocked until a vendor-supported, independently supervised fallback controller
is demonstrated to survive the specified faults.
