# X2 response-state clone audit (2026-08-14)

## Decision

`FAIL_RESPONSE_STATE_CLONE_BLOCK_PARALLEL_MPC`

Parallel MPC is not unlocked. Student training and deployment remain locked.
The next teacher feasibility route is response-domain RL, which does not depend
on cloning a live contact state into parallel simulator lanes.

## Frozen audit

- Domain: identified Session03+Session04 actuator response, no randomization.
- Contract: direct normalized 15-D lower-body joint-position targets.
- Environments: 128.
- Warm-up: 25 control steps.
- Identical-action rollout: 50 control steps.
- Immediate maximum gate: `2e-5`.
- Rollout maximum gate: `2e-4`.
- Termination gate: zero terminated or truncated lanes.
- Policy weights, optimizer steps, and checkpoint writes: zero.

## Result

Immediately after cloning, all observed robot state and all initialized
actuator-response histories matched exactly (`max_abs = 0`). During the
identical-action rollout, every audited actuator filter and delay history still
matched exactly, but simulator state diverged:

| Quantity | Maximum absolute deviation |
| --- | ---: |
| Joint velocity | `0.4305532873` |
| Root state (origin-relative) | `0.0217052363` |
| Root angular velocity (body) | `0.0217383243` |
| Joint position | `0.0011865944` |
| Projected gravity | `0.0013168333` |
| Root linear velocity (body) | `0.0030812435` |

No lane terminated. Because the explicit actuator-response state remains
identical while PhysX state separates after an exact visible-state clone, the
result is consistent with non-cloned contact-solver warm-start/cache state or
equivalent simulator-hidden state. This is an inference from the audit, not a
claim that the hidden PhysX structure itself was directly inspected.

## Interpretation

The result does not say that X2 cannot be controlled. It says that exact
fork-in-time, many-lane MPC cannot be justified by copying only exposed root,
joint, action, and actuator-response tensors. Candidate rollouts would start
from observably equal but dynamically non-equivalent contact states, so their
ranking could be contaminated by the fork rather than by candidate actions.

## Evidence and recovery

The report, checksum sidecar, initial guarded-abort log, successful retry log,
sizes, and SHA256 values are recorded in
`manifests/x2_response_clone_audit_20260814.json`. Artifacts live in the
`BFM-Zero` data namespace under the `2026-08-14/x2_clone_audit` processed/log
directories. No artifact is stored in Git.

