# X2 Sonic Decision Log

## D-001: Keep the v0.9 direct-HAL architecture

**Status:** accepted and frozen.

Use firmware `v0.9.0-rc7`, `aimdk_msgs 0.8.18`, Sonic on SoC1, official
MC/HAL on SoC0, and direct HAL command topics. The MC pause, ownership proof,
feedback, custom PD, and cleanup path have already been demonstrated. The v1.0
`Develop_MC` branch belongs to a different robot/software line and is not an
alternative entry here.

## D-002: Fix the HAL writer at 250 Hz

**Status:** accepted and frozen.

The 500 Hz regression produced grouped message rates near 2000 messages per
second and bursty state delivery. At 250 Hz, state freshness was reproduced.
Normal runs must leave `X2_SONIC_WRITER_HZ` unset.

## D-003: Use `neutral_damped` as the sole policy parent

**Status:** accepted.

It is the only preset with a five-minute bounded supported run. All other
launchers are frozen experimental history. `run_x2_ground_load_hold.sh` may be
used only for policy-disabled static diagnostics.

## D-004: Do not tune gains until entry is controlled

**Status:** accepted.

`waist2` changed pitch and roll together and also entered policy from a
different load condition. The result was confounded. The exact baseline later
failed under another entry, so physical entry posture/support/contact must be
isolated before any gain change.

## D-005: Use reconstructed pelvis IMU

**Status:** accepted for the current baseline.

The policy/MJCF root is the pelvis while hardware supplies torso IMU plus
3-DOF waist state. The deployed path reconstructs pelvis orientation. Raw
torso mode is diagnostic only and must not be mixed into a baseline run.

## D-006: Treat the 300-second result narrowly

**Status:** accepted.

`170450` proves bounded gantry-supported neutral behavior for one physical
entry. It does not prove unsupported standing, walking, garment teleoperation,
or safe entry from arbitrary orientations.

## D-007: Entry posture is a candidate, not a root cause

**Status:** accepted.

Stable and failed entries had opposite pelvis pitch signs and different roll,
but gantry load and foot contact were not measured. The next test may target
the known-good orientation as a single-variable reproduction. Its matching
window is an experimental inclusion criterion, not a universal safety limit.

## D-008: Keep garment/ZMQ out of the powered loop

**Status:** retained for powered tests; superseded for offline integration by
D-010 through D-012.

Changing the reference source before the balance/contact gate closes would
confound controller stability with teleoperation transport and garment data.
Offline implementation, loopback, and dry-run validation are now active, but
garment input remains blocked from powered HAL control until its explicit
acceptance gates close.

## D-009: Preserve the dirty checkout and deploy by hashes

**Status:** accepted.

Local `main` contains existing modified and untracked user work. SoC1 is a
synchronized non-Git runtime copy. Do not clean either tree or define runtime
identity by `HEAD` alone; use the manifest hashes in
`docs/BASELINE_170450.md`.

## D-010: Run the final garment path entirely onboard the target X2

**Status:** accepted.

The target X2 SoC1 owns AX210 BLE reception, TIC/LFP, Fast SMPL, GMR, HMCP,
G1-to-X2 conversion, LiveMotion, localhost ZMQ, and Sonic. The 5060 laptop is a
proven shadow/source environment only. The Mac is an operator console only.
Neither belongs in the runtime data path.

## D-011: Preserve the proven G1-reference garment semantics

**Status:** accepted and frozen.

Keep `V2 -> TIC/LFP -> Fast SMPL -> GMR to G1 qpos36 -> G1-to-X2 incremental
mapping -> LiveMotion` with `pose-scale=0.7`. Do not replace it with direct-X2
GMR, an HP3090 PKL, a fixed gait, a new planner, or a future predictor.

## D-012: Require reference readiness and supported return on ZMQ loss

**Status:** dry-run implementation verified; powered validation pending.

Powered policy entry must require a warmed and fresh body-bearing ZMQ v5
reference. During `SUPPORTED_POLICY`, reference starvation must take the
bounded return to the captured static hold instead of jumping directly to a
default-angle `SAFE_IDLE`. The 0.5-second starvation threshold remains the
fail-closed trigger. The onboard disconnect test stopped HMCP after deliberate
3588S loss and entered `SAFE_IDLE` in about `0.518 s` with no local fallback.

## D-013: Stage corrected standing before changing target slew

**Status:** completed and superseded by D-014/D-015.

The corrected `0.12 rad/s` and staged `0.30 rad/s` powered probes are complete.
Increasing slew did not restore tracking and worsened peak ankle error/speed,
so further slew increases are rejected. Keep gains, target envelopes, garment
input, and writer rate frozen while support transfer is isolated.

## D-014: Separate fixed-support acceptance from support transfer

**Status:** accepted on 2026-08-31.

Changing sling tension while policy is on invalidates a fixed-support
acceptance run. Conversely, a long taut-sling hold does not establish
autonomous load-bearing balance. The two questions require separate protocols.

The 2026-08-31 M2 attempt stayed bounded for more than 170 seconds under heavy
fixed support while retaining approximately `0.49 rad` left ankle-pitch and
`0.42 rad` waist-pitch tracking errors. Freehand sling release then changed
the external wrench and produced a guarded return. Therefore no parent
manifest may treat this run as a 300-second pass or an unsupported-balance
failure.

The next powered branch must first reduce the loaded tracking mismatch through
a policy-off support-geometry diagnostic. Support transfer, when authorized,
uses discrete marked hoist increments and hold plateaus with explicit event
timestamps. Continuous unmeasured loosening is rejected. Garment arming stays
closed until the supported-only scope is explicitly accepted or the staged
support-transfer gate passes.

## D-015: Gate and mark the first support-transfer step

**Status:** accepted for one bounded powered probe; no physical result yet.

The first support-transfer probe uses the unchanged `neutral_damped` parent,
50 Hz policy, 250 Hz writer, reconstructed pelvis, `0.12 rad/s` target slew,
gains, LPFs, clamps, joint mapping, and model. The new mode changes
orchestration only:

- support remains immutable for the first 20 policy seconds;
- both ankle-pitch errors must be `<=0.20 rad`, waist-pitch error
  `<=0.20 rad`, maximum joint speed `<=0.05 rad/s`, pelvis tilt `<=3 deg`, and
  one-second tilt growth `<=0.10 deg` continuously for five seconds;
- one accepted `support_step` command records exactly one telemetry marker but
  does not actuate the hoist;
- the operator makes one small repeatable hoist adjustment, then holds support
  fixed for 20 seconds;
- the existing bounded return restores captured static PD automatically.

The M2 loaded equilibrium (`0.494/0.120 rad` ankle-pitch error and `0.420 rad`
waist-pitch error) must be rejected. A state-machine dry-run may shorten times
and relax measured thresholds only to test transitions; it is not physical
acceptance evidence. A 100-second staged release and garment arming remain
blocked until the single-step powered probe passes.

## D-016: Do not gate support transfer on absolute sling-loaded waist error

**Status:** accepted offline on 2026-08-31; ARM64 build and powered validation
pending. This decision supersedes only the waist-error clause of D-015.

The X2 upper-torso sling applies an unavoidable waist moment. Official-MC
snapshots measured approximately `+0.349 rad` waist pitch both suspended and
at contact, the supported 300-second `170450` run entered at `+0.332 rad`, and
the visually upright policy-off pose in `134829` measured `+0.281 rad` while
remaining quiet at `0.007 rad/s`. These observations make an absolute
`<=0.20 rad` waist tracking requirement neither reproducible nor a valid
precondition for starting the fixed StandStill policy.

The support-step gate will continue to record waist target, position, error,
effort, and trend, but the dedicated candidate disables waist absolute error
as an accept/reject condition. It retains both ankle-pitch errors
`<=0.20 rad`, pelvis tilt `<=3 deg`, maximum joint speed `<=0.05 rad/s`,
one-second tilt growth `<=0.10 deg`, five continuous stable seconds, the first
20 seconds of immutable support, one marked support adjustment, and a bounded
return. The M2 loaded equilibrium remains rejected because its left
ankle-pitch error was `0.494 rad`.

Do not adjust the sling after `load` merely to drive waist pitch toward zero.
The relative policy-entry gate is evaluated against the initially captured
loaded pose and does not include waist error. Controller gains, LPF, target
slew, model, mapping, pelvis reconstruction, writer rate, and policy ramp
remain frozen.

## D-017: Reduce support between fixed-support runs, not during policy

**Status:** accepted as the next bounded standing route. This supersedes D-015
as the next powered experiment; the support-step implementation remains
preserved as offline experimental history.

Two physical A/B observations now point in the same direction. On 2026-08-30,
loosening support during policy caused an immediate tilt disturbance, while
establishing reduced support under policy-off static PD and then keeping it
fixed produced a complete 30-second StandStill run. M2 again remained quiet
under fixed heavy support and failed only after support changed during policy.
Repeating the same transition with a marker does not remove the external-wrench
step.

For the next run, final foot contact and sling tension are established before
`load`. `load` captures that exact loaded pose as the relative entry reference.
After policy begins, feet and sling are immutable for the complete run. Lower
support is tested in a later independent run, again established before
`load`; it is never introduced as an unmeasured on-policy disturbance.

The first prepared duration is 100 seconds using the frozen M1 binary and a
new hash-gated wrapper, `run_x2_supported_preloaded_fixed100.sh`. It changes
only policy duration and does not enable `support_step`. A pass proves only a
bounded fixed-support equilibrium at that qualitative sling level; without a
load cell or verified slack it does not prove unsupported standing.

The wrapper also pins the powered runner and MC/HAL handoff launcher, clears
stale adoption/debug environment overrides, and provides `--verify-only` for
artifact checks that start no process. An offline argv regression test proves
the 100-second and frozen 300-second profiles differ only in duration.
