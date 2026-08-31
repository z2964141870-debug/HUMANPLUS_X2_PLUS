# Future integration boundary

This is an interface note, not a procedure for connecting to a robot.  It must
not be used to unlock the existing v1.0 scripts.

## Adapter inputs

A future adapter would translate read-only telemetry into `Observation`:

- system state plus transition status;
- per-group feedback and official-command `StreamHealth` from source stamps,
  sequences, local reception, group skew, and gap history;
- the same health record for Sonic targets;
- official contract completeness, powered/stationary checks, and target error;
- independent all-group output sequence/gap/age evidence;
- tracking and vibration metrics computed over documented windows;
- whether official command samples were produced after the handback marker.

The adapter must pass one monotonic `now_s` domain to the machine.  Clock
translation used for source age must expose uncertainty and fail closed when
that uncertainty exceeds the age budget.

## Directive interpretation

- `publish=false` means candidate output is unnecessary only when `can_exit`
  is also true.
- `MIRROR_CAPTURED_OFFICIAL` reproduces the captured complete official contract.
- `CANDIDATE_ANCHOR` applies `contract_blend` atomically to all fields while
  keeping the effective target at the validated anchor.
- `SONIC_RATE_LIMITED` uses the complete Sonic contract with adapter-enforced
  position, velocity, acceleration, jerk, torque/current, and joint limits.
- `FREEZE_LAST_ACCEPTED` means keep the last complete all-group cycle and latch;
  it does not assert that a frozen controller is physically safe.
- `MIRROR_LIVE_OFFICIAL` reproduces fresh post-marker official commands during
  the final overlap dwell.  Until the first such command is verified, the
  directive remains `FREEZE_LAST_ACCEPTED`.
- `transition_request` is intent data.  A separate, explicitly authorized
  migration component would have to validate and act on it.

## Required separation

Use three independently supervised components if this design ever advances:

1. A deterministic real-time command owner that can outlive policy and ROS
   orchestration failures.
2. A read-only telemetry/health monitor with source-time and downstream command
   evidence.
3. A state-transition orchestrator that cannot release command ownership by
   process exit or exception cleanup.

The pure machine should remain importable and testable without any of those
components.  Robot-specific message construction, DDS QoS, migration clients,
publisher lifecycle, network parsing, and operator UI do not belong in
`state_machine.py`.
