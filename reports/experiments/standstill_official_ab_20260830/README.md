# X2 Sonic StandStill Stability Ledger - 2026-08-30

## Scope

This stage isolates the fixed `StandStill` reference and supported balance path.
It does not change or test clothing, HMCP, offload, or ZMQ. The target slew limit
remains `0.12 rad/s`. No command from this stage has been sent to robot HAL.

## Inputs and domains

- Policy: `x2_sonic_frozen_g1core_lora_v2.onnx`.
- Hardware-entry replay: `hardware_success_300s_entry_q.json`.
- Current Sonic domain: the Sonic-X2 `x2_ultra.xml` and its meshes.
- Official domain: untouched AgiBot `x2_rl_deploy_mujoco` XML and meshes.
- Both domains load as `nq=38`, `nv=37`, `nu=31`, timestep `0.001 s`.
- Diagnostic script SHA-256 after torso-harness instrumentation:
  `b136e47d1802211a0391f7c17390752aabd55423d95151042d1e1b3c34bd6b1f`.

## Evidence reconciled

The real 300-second run entered policy with waist pitch about `+0.332 rad` and
waist roll about `+0.158 rad`. Waist pitch stayed near that value and waist roll
only moved to about `+0.126 rad`. This run used strong, persistent gantry support.

The failed hardware probes entered with materially different waist pitch:

| Run | Entry waist pitch | Result |
| --- | ---: | --- |
| 300 s historical run | `+0.332 rad` | Completed under strong support |
| 0.12 slew probe | `+0.187 rad` | Failed |
| 0.30 slew probe | `+0.184 rad` | Failed |

Default-entry simulation with 25% pelvis support failed in both dynamics domains:

| Domain | Policy time before gate | Trigger |
| --- | ---: | --- |
| Current Sonic XML | `7.34 s` | Waist roll `0.802 rad/s` |
| Official AgiBot XML | `6.58 s` | Waist roll `0.843 rad/s` |

This rules out the modified Sonic MJCF as the sole cause. The baseline simulation
also pinned waist pitch at the `-0.315 rad` mechanical limit while the policy/HAL
target moved toward `+0.20 rad`, producing about `0.515 rad` tracking error.

Doubling waist stiffness moved static pitch from `-3.83 deg` to `+10.25 deg` and
failed sooner (`4.06 s`). Raising only waist-roll damping to about
`2.0 Nms/rad` extended failure only to `7.90 s`. Neither tuning is a hardware
candidate.

Injecting the successful hardware joint snapshot with 100% pelvis support passed
30 seconds in both domains, but it pinned simulated waist pitch/roll near their
limits and left almost no foot contact. Therefore the historical 300-second run
proves supported policy execution, not independent standing.

## Harness-model correction

Robot photos show the rope attached around the rear upper torso/neck, not at the
pelvis centre of mass. The earlier virtual gantry applied body-weight support and
a direct attitude-restoring torque at the pelvis, which can create false stability.

The simulator now supports:

- `--gantry-body pelvis|torso_link`;
- `--gantry-attachment-offset X Y Z` in body-local metres;
- `--gantry-attitude-scale`, with `0` representing a single-point rope;
- CSV telemetry for attachment position, applied force/moment, bilateral foot
  contact normal force, and measured foot load fraction.

Legacy `pelvis`, zero-offset, attitude-scale `1` remains the backward-compatible
mode. The first physical-harness approximation is `torso_link`, local offset
`[-0.05, 0.0, 0.30] m`, attitude-scale `0`. The offset is a modeling hypothesis
and must be checked with a small fore/aft sensitivity scan.

## Completed acceptance work

- The 25/50/75/100% torso-support scan was completed in both dynamics domains.
- A single-point upper-torso attachment with no artificial attitude torque was
  selected; the stable fore/aft region is approximately `-0.025..-0.05 m`.
- The frozen candidate uses `torso_link`, offset `[-0.04, 0, 0.30] m`, and 50%
  nominal support with unchanged `neutral_damped` gains and `0.12 rad/s` slew.
- All nine `+-2 deg` roll/pitch perturbations and all historical hardware entry
  snapshots passed 30 policy seconds in both domains.
- Both domains passed 300 policy seconds from the default entry. Final tilt was
  `1.65/1.63 deg`; foot load settled near `42.6%`; no waist/ankle limit pinning
  or growing oscillation was observed.

These are supported-dynamics results, not unsupported-standing proof. They point
to physical support/contact setup as the next controlled variable and do not
justify another gain or slew sweep.

## Current implementation stage

The deploy source now records pre-safety policy target, final HAL target, q/dq
and tracking error, measured effort, effective Kp/Kd, estimated PD torque, motor
temperatures/voltage, and four group protection states. Control parameters and
writer behavior are unchanged.

The isolated SoC1 build passed `2/2` tests and has binary SHA-256
`7fa51a4df2553fcd52158dfb62fe6bb7dfa71e79ea51b718bd4f086e94efe40c`.
A read-only three-second StandStill dry-run produced 150 aligned rows in all 16
CSVs while command publishers were disabled. All domain states were zero and,
after exit, the four HAL command topics still had one publisher each. No powered
probe has used this binary yet.

The isolated 30-second wrapper is
`run_x2_supported_neutral_damped_telemetry30.sh` (SHA-256
`a147673e0553ff733c3aa2551405b6a30a891cbbc4813d7cf0cdbcd3de4b89f8`).
It pins fixed StandStill, `neutral_damped`, `0.12 rad/s` slew, and a 250 Hz
writer against `runtime_suspended/ws_telemetry_20260830`. It has been staged and
hash-checked but has not yet been used for a powered probe.

## Next acceptance sequence

1. Confirm exactly one HAL writer for the isolated telemetry runtime.
2. Run three independent 30-second fixed-StandStill probes under reproducible
   strong torso support, rejecting growth, toe rise, or any protection state.
3. Reproduce one 300-second run only after all three short probes pass.

Garment/HMCP remains out of this powered stage. At all stages there must be
exactly one HAL writer.

## 2026-08-30 15:23 CST: post-battery checkpoint

After the battery restart, no previous deploy/handoff process remained. All
four HAL command topics had one official-MC publisher and all four joint-state
groups were live. The wrapper, shared runner, and isolated telemetry binary
matched their frozen SHA-256 values (`a147673e...`, `ca67c3ca...`, and
`7fa51a4d...`, respectively).

The hash-gated 30-second wrapper is currently loaded at
`runtime_suspended/logs/suspended_sonic_20260830_152344`. It has cached a fresh
pose and is waiting in `STANDBY`; its writer is suppressed and MC is still
running. No powered transition has occurred. The next action is permitted only
after confirming firm suspension, a clear exclusion zone, and a ready E-stop.

## 2026-08-30 15:25-15:36 CST: powered takeover checkpoint

The suspended gate was accepted. SoC0 `mc_app_main` PID 2239 was paused, MC
wire silence and exclusive ownership were proven, and the 250 Hz custom writer
completed cached-pose PD acquisition plus the fixed-StandStill pose ramp. The
robot is now in `READY_FOR_GROUND` under custom static PD; policy is off and MC
has not been restored.

Feedback initially arrived in 1-2 second bursts (peak observed age about 3.24
seconds), so the policy gate correctly stayed `WAITING`. The frozen 250 Hz
writer and state/control/writer callback isolation were all present. A separate
read-only subscriber received leg state continuously at 999-1004 Hz, isolating
the transient to the deploy reader's DDS settling rather than HAL or the cable.
The deploy reader subsequently recovered without restart and now remains
`ARMED`: body state is near 1 kHz, head near 333 Hz, IMU near 500 Hz, and ages
are about 0-3 ms. No `load` or `policy` command has yet been sent.

## 2026-08-30 15:42-15:56 CST: powered telemetry probe 1

The first fixed StandStill probe used the frozen `neutral_damped`, `0.12 rad/s`,
250 Hz configuration and exactly one HAL writer. It did not pass: the bounded
return triggered after `3.740 s`, then static PD remained active and MC remained
paused.

The temporary `+5..+9 deg` absolute pelvis-pitch entry gate was invalid. The
robot's visually natural supported posture was about `-8 deg`; forcing it to
`+8 deg` produced the severe whole-body lean preserved in
`evidence/powered1_forced_positive_pitch.jpg`. The 300-second historical
`+7 deg` entry was therefore a harness-configuration artifact, not a reusable
neutral-standing target.

Aligned 16-channel telemetry shows:

- pelvis roll/pitch/tilt grew from `-1.29/+8.00/8.10 deg` to
  `-6.21/+11.76/13.28 deg`;
- ankle-pitch tracking error entered at about `0.296 rad` on both sides and
  peaked at `0.582 rad` left / `0.542 rad` right;
- right ankle speed peaked at `0.531 rad/s`;
- ankle Kp/Kd stayed at `32.064/3.003`;
- requested versus measured peak effort matched within about `0.2 Nm`;
- all four DomainErrorState channels stayed zero, voltage stayed `51-52 V`,
  and temperatures were normal.

The run rules out writer loss, gain loss, voltage collapse, and an exposed
domain-protection trip as the immediate cause. It instead started from a
badly harness-loaded pose with a large ankle error. No second probe should use
that entry. Replace the absolute pitch gate with a relative captured-pose gate
and add a pre-policy ankle tracking-error gate before another powered run.

Local analysis:

```text
/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/
  analysis_logs/suspended_sonic_20260830_152344_powered1
  analysis_reports/standstill_20260830_powered1/full_telemetry
  scripts/analyze_powered_telemetry_probe.py
```

## 2026-08-30 16:39-16:42 CST: relative-entry runtime active

The old absolute-entry writer was replaced without restoring official MC. The
new binary was first loaded in silent `STANDBY` on debug port `5559`; the old
launcher cleanup was frozen, its C++ writer exited through `lifted`, and the
new launcher did not trigger until the old writer PID was absent and command
wire silence had been proven. No two active HAL writers overlapped.

The robot is now held by the only active deploy PID `58822` at 250 Hz. SoC0
`mc_app_main` PID `2239` remains paused (`Tl`). The relative-entry runtime log
is `runtime_suspended/logs/suspended_sonic_20260830_163947`; it completed the
3-second PD acquire and 5-second StandStill pose ramp and is in
`READY_FOR_GROUND`, with fresh feedback armed and policy off. Next: light foot
contact under a still-load-bearing gantry, then `load` to capture the relative
entry baseline before any policy pulse.

## 2026-08-30 16:45-16:47 CST: relative-entry probe 1

`load` captured pitch/roll `-11.65/-0.60 deg`; the entry remained stable with
`0.099 rad` ankle-pitch error and `0.007 rad/s` maximum speed. The subsequent
fixed-StandStill policy probe completed 1500 aligned samples / 29.979 seconds.
It brought pelvis tilt from `11.66 deg` to `2.26 deg` and settled near
pitch/roll `-0.43/+2.22 deg` without a protection state or action clip. Peak
joint speed was `0.336 rad/s`; bus voltage was `47-48 V`.

Steady load-bearing tracking errors remained `0.425/0.314 rad` at left/right
ankle pitch and `0.438 rad` at waist pitch. Corresponding measured effort
matched estimated PD effort, so this is sustained load torque rather than a
missing writer. Timeout froze the final policy command; the old relative entry
reference and `0.15 rad` ankle gate therefore block an immediate repeat. For a
fresh second probe, re-suspend, use `lifted`, and relaunch the same runtime with
MC still paused. Probe 1 remains telemetry-pass / physical-observation-pending
until the operator confirms no toe rise, oscillation, or abnormal sound.

The operator subsequently reported no clear abnormal motion, toe rise,
oscillation, or abnormal sound; only a possibly imperceptible small movement.
Probe 1 is now a telemetry pass and tentative physical pass. The final static
PD command then held for about 460 seconds. A `joint speed >= 1.0 rad/s` guard
occurred only while the robot was being lifted and manually moved, well after
the bounded policy interval, so it does not invalidate the probe. `lifted`
cleanly stopped the writer, no custom HAL writer remains, and SoC0 MC PID 2239
is still paused (`Tl`).

## 2026-08-30 17:00-17:04 CST: relative-entry probe 2

An independent relaunch with the same frozen 250 Hz configuration completed a
second full 1500-sample / 29.979-second probe. The entry was stable at
roll/pitch `-0.52/-9.81 deg`, with `0.115 rad` ankle-pitch error and fresh
feedback. Policy brought roll/pitch/tilt to `+1.89/-0.32/1.92 deg`. Roll had a
single `+3.09 deg` peak at 8.92 seconds, then stayed within
`+1.84..+1.90 deg` from 12 seconds through timeout; pitch stayed within
`-0.33..-0.26 deg`. The operator noticed only a small sway and no major
problem, matching a bounded settling transient rather than growth.

All domain states were zero, no action clip or runtime fault occurred, and
peak measured speed was `0.275 rad/s`. Left/right ankle-pitch measured effort
peaked at `16.38/17.03 Nm` and matched estimated PD effort. The later tilt
guard was caused by manually lifting the robot after policy completion.
`lifted` exited cleanly, no writer remains, and MC PID 2239 is still paused.
Probe 2 passes both telemetry and physical observation.

Full report:

```text
/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/
  analysis_logs/suspended_sonic_20260830_165839_powered2
  analysis_reports/standstill_20260830_relative_powered2/full_telemetry
```

## 2026-08-30 17:12-17:14 CST: relative-entry probe 3 and battery stop

The third independent 30-second run used the same frozen configuration. Its
entry was stable at roll/pitch `-1.14/-9.21 deg`, with fresh feedback,
`0.007 rad/s` maximum speed, and `0.130 rad` ankle-pitch error. All 1500 policy
ticks completed and converged to about roll/pitch/tilt
`+2.54/-0.41/2.57 deg`; the final approximately 19 seconds showed no growth,
action clip, or runtime fault. The operator reported no abnormal motion, toe
rise, or sound, so probe 3 passes runtime telemetry and physical observation.

Offline analysis subsequently confirmed 1500 aligned samples / 29.979 seconds,
zero malformed rows, and zero nonzero domain-state samples. Pelvis
roll/pitch/tilt changed from `-1.13/-9.22/9.29 deg` to
`+2.54/-0.41/2.57 deg`; peak joint speed was `0.263 rad/s`. Left/right
ankle-pitch effort peaks `17.03/17.26 Nm` matched estimated PD effort
`17.10/17.35 Nm`.

The later tilt guard came from lifting after policy completion. `lifted`
returned the launcher to its shell. The robot battery then ran out and is being
replaced; no 300-second run was started.

After reboot, no custom writer remained, MC restarted normally as PID 2129,
`mc_ros2_node` was the sole HAL command publisher, and leg feedback was about
1000 Hz. A dedicated relative-entry 300-second wrapper was then added and
synced. It preserves every accepted 30-second setting and changes only the
policy timeout from 30 to 300 seconds. Wrapper SHA-256 is
`580f8b26a1c94a2257ca11c7e1c81f1c82730f3fa0dc0c6450f74ea52ae87f04`.

Full report:

```text
/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/
  analysis_logs/suspended_sonic_20260830_171018_powered3
  analysis_reports/standstill_20260830_relative_powered3/full_telemetry
```

## 2026-08-30 20:18-20:39 CST: fixed-harness relative-entry probe

After another battery restart, the command graph was clean and the only custom
deploy was loaded in writer-suppressed `STANDBY`. The 250 Hz cold start then
paused MC, acquired the current suspended pose, and reached
`READY_FOR_GROUND` with fresh feedback armed and policy off.

The fixed upper-torso harness left suspended waist pitch at `+0.3436 rad` while
the static HAL target was `0.0000 rad`. Measured waist effort was about
`-4.84 Nm`, consistent with the effective `14.25 Nm/rad` PD stiffness and the
tracking error. This is close to the historical 300-second entry
(`+0.332 rad`), so it is not by itself a failed entry and does not justify
forcing waist pitch to zero or increasing waist gain.

At `load`, the relative-entry baseline was:

- pelvis pitch/roll/tilt `-8.27/-0.68/8.30 deg`;
- maximum joint speed `0.007 rad/s`;
- ankle-pitch tracking error `0.124 rad`;
- waist pitch target/measured/error `0.000/+0.303/-0.303 rad`;
- fresh feedback, zero stale ticks, and a ready pose gate.

The fixed-StandStill policy then completed all `1500` samples / `29.979 s`.
Pelvis roll/pitch/tilt changed from `-0.69/-8.27/8.30 deg` to
`+1.85/-0.31/1.88 deg`; peak joint speed was `0.372 rad/s` at the right elbow.
Peak left/right ankle-pitch tracking errors were `0.480/0.340 rad`, and peak
waist-pitch tracking error was `0.449 rad`. Measured versus estimated PD effort
peaks were `18.90/18.97 Nm` left ankle, `13.39/13.60 Nm` right ankle, and
`6.45/6.44 Nm` waist pitch. Raw action peaked at `1.295`, all four domain-state
channels stayed zero, no action clip or runtime safety trip occurred, and valid
motor temperatures stayed `25-43 C` (coil `33-50 C`).

The two-second bounded return restored the captured static hold; policy is off,
custom PD remains active, and MC has not been restored. Telemetry passes. Final
physical acceptance remains pending the operator's report of oscillation, toe
rise, and abnormal sound.

Copied evidence:

```text
/Users/yu/Documents/ChatGPT/X2/standstill_official_ab_20260830/evidence/
  suspended_sonic_20260830_201850/
```

## 2026-08-30 20:43-20:58 CST: support-change A/B

The operator reported no visible oscillation, toe rise, or abnormal sound in
the preceding 30-second run, so that run passes physical acceptance as well as
telemetry.

An A/B test then separated support change timing from support amount:

- **A, change support during policy:** the same policy initially converged,
  but loosening the rope during the run produced a fast disturbance. Tilt rose
  from about `3.05 deg` at `15.0 s` to `7.94 deg` at `15.5 s`; the envelope
  gate returned to static PD. The post-return transient peaked near `22 deg`
  before settling. Static PD remained active throughout. After the robot was
  re-suspended, the dedicated `clear-fault` command cleared the software latch
  without exiting the writer or restoring MC.
- **B, establish reduced support before policy:** with policy off, the robot
  was lowered, the rope was loosened to the intended supported load, and the
  static state was allowed to settle. Entry was stable for about `50 s` at
  pitch/roll `-8.79/-0.03 deg`, ankle-pitch tracking error `0.123 rad`, and
  maximum speed `0.007 rad/s`. The subsequent fixed-StandStill policy completed
  all 30 seconds. Final pitch/roll/tilt was `-0.69/+0.22/0.72 deg`; the late-run
  maximum speed was about `0.007 rad/s`, with no action clip or safety fault.
  The operator again reported no visible oscillation, toe rise, or abnormal
  sound.

This closes the supported fixed-StandStill entry gate for the present harness:
support amount can be reduced, but support must be established and allowed to
settle before policy activation. Changing gantry load during policy is a large
external disturbance and is not a valid startup procedure. Do not spend more
time repeating identical 30-second probes. Freeze this runtime/configuration as
the supported baseline and move the project to the first bounded dynamic
reference / garment-integration gate. Unsupported autonomous standing remains
unproven and is a separate acceptance target.

## 2026-08-31 final-parent interface candidate

The minimal policy-off telemetry gap is closed in a new SoC1 scratch build.
`GROUND_LOAD_HOLD` now publishes measured `x2_debug` only while all robot state
groups are fresh. Direct CTest passed `2/2`; the exact candidate binary is:

```text
3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f
```

An isolated `--dry-run` process test created no HAL command publisher. Official
MC remained the sole owner on all four command topics while the exact candidate
fed robot debug to the live-reference proxy. The proxy reached
`STANDSTILL_READY`, published exact anchored StandStill with at most
`2.9564e-08 rad` position error and zero velocity, and then locked out as
designed when source/debug was removed. All test processes and ports were
clean afterward. Startup timing was shortened only for this non-commanding
interface test and is not a powered-parent configuration result.

This result changes the next required StandStill run: use this same candidate
binary for one supported 300-second final-parent acceptance, preserving the
accepted rule that support is established while policy is OFF and never
changed after policy starts. Historical `170450` and the 2026-08-30 30-second
runs cannot substitute for the same-binary 300-second evidence. No additional
identical 30-second probe is required.

## 2026-08-31 support-transfer correction and candidate

The attempted final-parent 300-second run remained bounded for more than 170
seconds only under heavy fixed sling support. Left ankle-pitch and waist-pitch
tracking error remained about `0.49/0.42 rad`. Loosening the sling while policy
was active changed the external support wrench, invalidated the fixed-support
acceptance, and triggered a guarded return. It was neither a 300-second pass nor
a clean autonomous-standing failure.

Support transfer is now isolated in a dedicated candidate. The frozen
controller is unchanged; the new mode requires 20 seconds of immutable support
and five continuously gated seconds before accepting one `support_step` marker.
The production gates are both ankle-pitch errors `<=0.20 rad`, waist-pitch
error `<=0.20 rad`, maximum joint speed `<=0.05 rad/s`, pelvis tilt `<=3 deg`,
and one-second tilt growth `<=0.10 deg`. The marker does not move the hoist. It
authorizes one small repeatable operator adjustment followed by a fixed
20-second plateau and automatic return to static PD.

```text
SoC1 workspace: runtime_suspended/ws_support_step_20260831_v3
candidate binary: 9d4916a8e5fd64e4ac3ddb0f0a32903ccc90b9e86b4e146cc6cabefe63695483
support-step wrapper: 19148f68824f608a91e05ba181ce96cffb7a7487644594987ba5a98f9616667c
```

The scratch build passed `2/2` CTests. A final read-only process test waited for
`feedback_gate=ARMED`, emitted exactly one support-step marker, completed its
accelerated plateau, returned to `GROUND_LOAD_HOLD`, and exited through
`lifted`. All 449 CSV rows were `dry_run=1`; no custom HAL publisher existed,
and final ownership remained one official `mc_ros2_node` publisher per command
group. Evidence is retained under
`gear_sonic_deploy/analysis_logs/x2_support_step_process_test_20260831_v3_marker4/`.

Next: one powered single-step support-transfer probe after fresh confirmation
that the robot is fully suspended and the emergency stop is ready. A 100-second
release and garment arming remain blocked.

## 2026-08-31 14:01-14:19 CST: support-step entry correction

The first powered run of the support-step candidate did not start policy. Its
initial loaded state already met the real relative entry gate, but support was
then adjusted in an attempt to reduce the later waist-error gate. Tilt rose to
`17.84 deg` and the policy-off static guard latched. A subsequent visually
upright state was quiet at tilt `5.42 deg`, ankle-pitch tracking `0.169 rad`,
waist-pitch tracking `0.281 rad`, and maximum speed `0.007 rad/s`.

The run proves a procedural mismatch, not a Sonic failure. The fixed upper-
torso sling produces approximately `0.28-0.35 rad` waist pitch even in visually
reasonable poses; `170450` entered its supported 300-second run at
`+0.332 rad`. The dedicated support-step candidate therefore keeps waist error
as telemetry but disables it as an absolute gate. Both ankle errors
`<=0.20 rad`, policy-settled tilt `<=3 deg`, speed `<=0.05 rad/s`, non-growing
tilt, five-second continuity, and the bounded one-step plateau remain.

The local change passes CTest `2/2` and has not been built or synchronized on
SoC1. The remote raw log is
`runtime_suspended/logs/suspended_sonic_20260831_134829` and must be copied
after network recovery. `lifted` ended the custom writer, but official MC
publisher recovery could not be completed before the robot went offline; keep
the robot suspended until one official publisher is verified on all four HAL
command groups.

## Next bounded route: preloaded fixed-support 100 seconds

The next run will not loosen the sling during policy. Final support and foot
contact are established before `load`; the relative entry is captured there,
and support remains fixed for 100 seconds. Any further support reduction is a
new independent run. This follows the successful 2026-08-30 B condition and
avoids repeating the failed A/M2 on-policy support change.

```text
run_x2_supported_preloaded_fixed100.sh
X2_SUPPORTED_PRELOADED_FIXED100_START
wrapper sha256 73bbd6122356d605802c1ac2e62467627b7bae26ea79c476906f7f1fda1513a1
```

The wrapper is local-only while the robot is offline. It must not be copied or
run until official MC is restored and all four command topics have one
`mc_ros2_node` publisher.
