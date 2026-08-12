# Stage264 new-machine replay diagnosis

## Decision

The historical `24/24` claim is not reproduced. The complete one-attempt matrix
is `22/24`: medium is `12/12`; low-speed straight is `5/6`, right turn is `2/3`,
and left turn is `3/3`. Both failures are valid physics episodes, are retained
byte-for-byte, and must not be retried or overwritten for baseline accounting.

## Failure signature

Both failures pass startup and locomotion, then fail only after the direct
moving-actor to stationary-actor handoff used by `STOP_CONTROLLER=policy`:

| case | first stop violation | z min | tilt max | stop drift | handoff action jump L2 |
|---|---:|---:|---:|---:|---:|
| `stage264_low_straight_r4` | 1.60 s | 0.117 m | 1.547 rad | 0.680 m | 1.985 |
| `stage264_low_turn_right_r3` | 1.90 s | 0.117 m | 1.544 rad | 0.654 m | 1.763 |

The failing runs entered stop at 0.368–0.393 m/s. Their move gates had already
passed, so they are not failures to initiate walking or turn. The common mode is
a fragile zero-command policy handoff from a still-dynamic state.

This is not proof that action jump alone causes the fall: one passing right-turn
episode has a jump of 1.821. It is evidence that the current stop handoff has too
little state-conditioned margin and should be fixed jointly with posture/contact
training, not hidden by relaxing gates.

## Contract audit

- Frozen weights and gait template match their expected SHA-256 values.
- Current 93-D baseline behavior leaves later default-off features inactive:
  no upper motion, symmetry projection, future preview, acceleration curriculum,
  recovery model, or PD multiplier is enabled. With no recovery model,
  `stop_policy_slot(None)` still resolves to the stationary actor.
- Current adapter SHA-256 is
  `91a67ab6c583d48bbf09fea1d02de33f0b77961a404d8e67b7fab7764662d733`.
  The Stage250 commit (`bea845f`) adapter SHA-256 was
  `75189c48d77eaee68f8afd7c18ecb0e53b18ff4ffd7d7a2636992ad5eaf8a221`;
  later code adds default-off features and evidence capture, so source identity is
  not falsely claimed even though active 93-D control semantics were audited.
- Historical straight-repeat orchestration retained passed files but reran a
  failed file on a later invocation. The checked-in Stage250 report contains
  final passing files but no attempt history, so its `24/24` cannot establish the
  stricter one-attempt probability measured here.

## Task 2 consequence

Group mean signed walking pitch remains between -0.197 and -0.160 rad. Task 2
must therefore optimize signed posture without reducing command speed, while
also evaluating the moving-to-stop transition. A/B/C selection must use one-shot
nominal gates; a candidate is not accepted by replacing failed samples.
