# X2 Sonic Balance Stage

Last updated: 2026-08-30

## Fixed architecture

- Robot firmware: `test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7`.
- Sonic runs on SoC1. SoC0 remains the HAL and official MC host.
- Use the existing v0.9 suspended takeover in `run_x2_suspended_sonic.sh`.
- The takeover and custom PD path are already proven: the robot has accepted
  custom PD, borne weight, and stood without the official Standing policy.
- Do not restart the V1.0 `Develop_MC` investigation. It belongs to the other
  robot and is not an entry point for this firmware.

## Proven baseline

The only grounded Sonic baseline is:

```text
launcher: run_x2_supported_neutral_damped.sh
log: runtime_suspended/logs/suspended_sonic_20260828_170450
policy: StandStill neutral reference
policy duration: 299.97 s / 15000 control frames
```

The runtime uses separate callback groups for state ingestion, 50 Hz policy
inference, and the HAL writer. The normal HAL writer rate is 250 Hz.

During the 300-second run, reconstructed pelvis tilt remained between 4.13 and
7.08 degrees and ended at 4.46 degrees. Policy started at log time 3122.78 s;
the gantry remained load-bearing until approximately 3340 s and was loosened
only for the final approximately 80 seconds. Sonic therefore settled under
gantry support before body weight was transferred. This proves bounded
steady-state supported neutral standing, not full-load policy entry,
unsupported balance, or walking.

### What this baseline proves

- v0.9 MC pause, HAL ownership, custom PD, feedback, and cleanup can work.
- Sonic can run continuously on SoC1 without a feedback or ownership fault.
- The neutral-damped gain/filter profile is the last known non-divergent policy
  profile and must be the parent of subsequent tests.

### Known defects

- The posture is not upright and has shown forward/lateral lean.
- Waist pitch measured about +0.331 to +0.337 rad while its command reached the
  -0.20 rad probe limit. Waist roll measured +0.120 to +0.158 rad while its
  command also reached -0.20 rad. The waist therefore had about 0.5 rad of
  command-to-measurement mismatch under load.
- The run did not validate push recovery, unsupported balance, stepping, or
  garment teleoperation.
- It did not validate enabling Sonic after the robot was already fully loaded.
  The successful order was policy first under gantry support, then gradual
  weight transfer.
- Earlier observations included fore-aft sway and toe-rise tendency. A long
  process lifetime alone must not be reported as a solved standing controller.

## Regressions that are not baselines

`waist2` doubled waist pitch/roll stiffness after the 300-second run. Session
`suspended_sonic_20260828_212937` had fresh feedback and stable static PD, but
Sonic exceeded the relative tilt envelope after 4.14 seconds. Tilt first fell
from 4.36 to 2.51 degrees, then reversed and reached 8.50 degrees. That run
changed both the waist response and the physical entry condition: policy was
enabled after the robot was already bearing weight. It is therefore a
confounded failure, not evidence that either change alone is the root cause.
It does not reopen MC/HAL handoff or feedback work.

Do not use `run_x2_supported_neutral_waist2.sh` as the next test baseline. Do
not add entry-residual compensation based on the 4-second failure: that idea
was never compiled or powered and has been removed.

The HAL writer is fixed at 250 Hz by default. A 2026-08-28 regression restored
500 Hz, producing about 2000 grouped command messages per second. Session
`suspended_sonic_20260828_203843` then showed all HAL state streams arriving in
bursts with roughly 0.1-2 second gaps, so the policy gate stayed in `WAITING`.
That run did not enter policy and cannot be used to judge Sonic balance.

The 250 Hz baseline was verified twice on 2026-08-28:

- `suspended_sonic_20260828_211439`: the first run inherited DDS settling from
  the earlier 500 Hz run, then armed after about 33 seconds and remained fresh.
- `suspended_sonic_20260828_211808`: armed 2.04 seconds after handoff and stayed
  continuously fresh for about 73 seconds. Leg, waist, and arm state ran at
  about 1 kHz; head at about 333 Hz; IMU at about 500 Hz; reported ages stayed
  near 0-3 ms.

MC was resumed normally after both tests. Feedback continuity at 250 Hz is now
a completed gate, not the active tuning problem.

For a deliberate rate A/B only, set `X2_SONIC_WRITER_HZ`; accepted values are
100 through 500. Normal powered tests must leave it unset.

## 2026-08-28 supported-entry reproduction

Session `suspended_sonic_20260828_220655` reproduced the exact
`neutral_damped` software profile at 250 Hz. The robot was lightly touching the
floor while the gantry remained load-bearing. Static PD was stable at 9.67
degrees tilt and 0.007 rad/s peak joint speed before policy entry.

Sonic initially corrected tilt from 9.67 to 0.70 degrees in 6.5 seconds, then
entered a large low-frequency oscillation without any feedback or ownership
fault. Tilt repeatedly crossed both pitch and roll, reached 14.11 degrees at
29.5 seconds, and tripped the relative-tilt return. Left/right ankle-pitch
targets swept as far as approximately +0.24 rad while measured ankle positions
lagged by up to about 0.67 rad. The automatic 2-second return completed and
static PD settled near 6.89 degrees.

This disproves the assumption that policy-first under merely light gantry load
is sufficient. The 300-second run depended on a materially stronger/longer
support condition or a different contact geometry during policy settling. The
active blocker is now the ankle-driven contact-boundary limit cycle, not waist
stiffness, MC/HAL handoff, DDS feedback, or policy process lifetime.

The first-30-second comparison is now complete. At policy entry, the stable run
had reconstructed pelvis roll/pitch `+0.41/+7.00 deg` while the failed run had
`-5.92/-7.67 deg`. Their projected-gravity x/y components were
`+0.1219/-0.0070` and `-0.1334/+0.1021`, respectively. Initial ankle pitch
positions differed by only about 0.02-0.03 rad, but measured ankle speed during
the first 30 seconds grew to about 0.76 rad/s in the failed run versus about
0.15 rad/s in the stable run.

This identifies entry orientation/support/contact as the next bounded variable.
It does not prove that orientation was the root cause: gantry force and foot
contact geometry were not instrumented and also differed.

## Current gate

1. Keep `neutral_damped`, the model, gains, filters, 250 Hz writer, pelvis
   reconstruction, and safety limits fixed.
2. Use the fixed launcher to acquire static PD first. Do not send `policy` from
   the current negative-pitch suspended posture.
3. Under load-bearing gantry support, record a three-second entry snapshot and
   hold static PD for at least ten seconds.
4. For the first single-variable reproduction, target the known-good entry
   orientation (approximately pitch `+5..+9 deg`, roll `-2..+2 deg`) without
   forcing the robot. This is a diagnostic matching window from one run, not a
   universal safety envelope.
5. If the matching state cannot be obtained, end from full suspension and
   record a blocked run. Do not change gains or try another launcher.
6. Only after a matched supported entry succeeds may waist mismatch be tested
   one axis at a time. Garment/ZMQ remains out of the powered loop.

## Powered-test entry

Read `../docs/OPERATIONS.md` before using this entry. All other policy wrappers
are frozen experimental history.

```bash
cd ~/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy
read -rsp 'SoC0 password: ' X2_SOC0_PASSWORD
echo
export X2_SOC0_PASSWORD
unset X2_SONIC_WRITER_HZ
./run_x2_supported_neutral_damped.sh X2_SUPPORTED_NEUTRAL_DAMPED_START
```

Start fully suspended with a person holding the physical emergency stop. After
`READY_FOR_GROUND`, use `load` and collect the static entry measurements. Do
not type `policy` unless the current experiment's inclusion criteria are met.
Use `stop` to return to static custom PD; use `lifted` only after the robot is
fully suspended and the session should end.

## 2026-08-29 corrected standing A/B

The two powered logs above used the old deploy binary. The corrected binary
`64d2663431f7531ce907bc006f6979ed34d05528ee335c3b71e8f4a48cc98bf6`
postdates and contains both checkpoint action-scale correction and supported
published-action feedback. C++ observation/gate tests pass on Mac and SoC1.

A MuJoCo supported-entry harness now reproduces the real sequence: ten seconds
of static PD under a virtual load-bearing gantry, four-second policy ramp,
`neutral_damped` gains/envelopes/LPF, gradual 80-second weight transfer, then
policy-only standing. Results with corrected action feedback were:

| Candidate | Result after virtual gantry reached zero |
| --- | --- |
| original `0.12 rad/s` target slew | failed before full transfer; target authority was rate-limited |
| `0.20 rad/s` | stood, but settled near 4 deg tilt |
| `0.30 rad/s` | stood about 66.6 s unsupported near 1.5-2.1 deg, then lost height |
| `0.40 rad/s`, `0.60 rad/s`, or disabled | same short-horizon posture as `0.30 rad/s` |
| leg envelope `0.60 -> 1.20 rad` | remained standing but posture worsened to about 4.5 deg |

This identifies `0.12 rad/s` as an over-restrictive deploy guard for balance.
It does not prove long-term real standing: the `0.30` simulation eventually
lost height, virtual gantry/contact dynamics are approximate, and tilted-entry
A/B did not establish a universal orientation window.

Two isolated powered launchers are staged, but neither is authorized merely by
this document:

```text
run_x2_supported_neutral_damped_corrected.sh   corrected binary, old 0.12 rate
run_x2_supported_neutral_damped_slew030.sh     corrected binary, 0.30 candidate
```

On SoC1 both launchers now use the isolated workspace
`runtime_suspended/ws_entrygate_8ee01b_20260829`. Its `install` link resolves
to `../ws/install_entrygate_20260829`, whose deploy binary SHA-256 is
`8ee01bc8f78d843895e5718627f36fdfd50771df1824ba06f288a0ed05b0a6a0`.
Each launcher verifies this hash before entering the internal powered launcher.
This build adds a signed pelvis pitch/roll, joint-speed, and continuous-stable
entry gate. The configured `+5..+9 deg` pitch and `+-2 deg` roll window is a
temporary reproduction gate for this A/B, not a permanent Sonic requirement.
The older `ws_corrected_*` directories remain historical runtimes.

Staged launcher SHA-256 values are:

```text
922ad88911e4dd6576efdd0ebbc60cd6c2f0322d21a507913932dd27e8dd6c5e  run_x2_supported_neutral_damped_corrected.sh
93307620e8d5a50ed521fcf035e55f6edc9b4f048fa404760e63ee166d83e366  run_x2_supported_neutral_damped_slew030.sh
```

The corrected `0.12 rad/s` probe has now been powered under support. Session
`suspended_sonic_20260829_210922` returned after about 2.96 seconds on tilt,
with no clipping, ownership, or feedback fault. Right and left ankle pitch were
slew-limited for about 71% and 64% of effective policy cycles; the right ankle
desired-to-published target backlog exceeded 0.20 rad. This confirms that the
deploy guard, rather than the checkpoint action clip, was withholding much of
the immediate ankle correction in this run.

That proposed `0.30 rad/s` experiment was subsequently run and failed. It is
historical evidence, not the next action; see the offline audit below.

## 2026-08-30 fixed-StandStill offline audit

The three relevant hardware logs were aligned and replayed with
`scripts/analyze_supported_standstill.py`:

```text
success_300s:    suspended_sonic_20260828_170450
failure_slew012: suspended_sonic_20260829_210922
failure_slew030: suspended_sonic_pdhot_20260829_222159
```

For both ankle-pitch joints the replay follows the deployed command-shaping
order exactly:

```text
clipped policy action -> raw target -> 3 s safety ramp
-> 4 s supported ramp/envelope -> 8 Hz LPF -> target slew
-> logged SafeCommand/HAL target -> measured q/dq
```

The reconstructed and logged ankle HAL targets agree within `1e-6 rad` in all
three runs. The early-return terminal row in each short failure contains the
previous applied action rather than a fresh ONNX action; the analyzer excludes
that row from raw-action statistics.

| Run | Entry ankle error L/R | Slew-limited L/R | Peak tracking error L/R | Peak ankle speed L/R |
| --- | ---: | ---: | ---: | ---: |
| 300 s success | 0.116/0.118 rad | 4.1%/6.2% | 0.419/0.471 rad | 0.140/0.153 rad/s |
| 0.12 failure | 0.254/0.322 rad | 62.8%/70.3% | 0.419/0.540 rad | 0.128/0.336 rad/s |
| 0.30 failure | 0.271/0.275 rad | 29.4%/26.2% | 0.594/0.673 rad | 0.519/0.861 rad/s |

The growth check distinguishes a bounded offset from divergence. Pelvis tilt
maximum changed from first to last one-second windows as follows: `7.04 ->
4.46 deg` in the 300-second pass, `6.26 -> 11.29 deg` in the 0.12 failure, and
`5.22 -> 10.38 deg` in the 0.30 failure. Both failed runs also had increasing
left and right ankle tracking error.

This closes two questions. The `0.12 rad/s` limiter withheld substantial ankle
correction in its failed run, but increasing it to `0.30 rad/s` did not restore
physical tracking and made peak ankle speed/error materially worse. No further
target-slew increase is authorized. The 0.30 run entered within the temporary
roll/pitch matching window and still failed, so pelvis entry angle alone is not
a sufficient explanation.

The remaining attribution is deliberately open between realized PD/torque
authority, joint zero/reference alignment, foot/support contact, and hardware
protection. Historical logs contain targets, q/dq, and IMU, but no measured
effort/current, motor voltage, per-tick gains, foot or gantry force, or
protection state. The reported `13-21 Nm` PD values are computed from
`kp*(target-q)-kd*dq`; they are not measured torque.

Artifacts:

```text
analysis_reports/standstill_20260830/README.md
analysis_reports/standstill_20260830/summary.json
analysis_reports/standstill_20260830/*_ankle_trace.csv
```

The next stage is offline simulation/replay of the unchanged `0.12 rad/s`
parent with exact deploy ordering and effective ankle gains. Compare unloaded
support with loaded contact; do not run another slew sweep. Before any powered
probe, add logging for measured effort, motor voltage/temperature, published
gains, available PMU/protection state, and an explicit operator support/contact
annotation. Only then freeze one configuration for three consecutive 30-second
supported probes, followed by one 300-second reproduction.

## 2026-08-30 dual-domain torso-harness replay

The new official RL simulation bundle and the external-disk Sonic assets were
audited on the x64 host. The physical photos show that the rope attaches behind
the upper torso/neck rather than at the pelvis centre of mass, so the earlier
pelvis-force plus attitude-torque gantry was rejected as an inaccurate support
model. The evaluator now supports a point-force attachment on `torso_link`,
with no artificial attitude-restoring torque, and logs bilateral foot force and
the fraction of body weight carried by the feet.

The frozen candidate is:

```text
reference: fixed StandStill
gains/profile: neutral_damped (unchanged)
target slew: 0.12 rad/s (unchanged)
gantry body: torso_link
attachment offset: approximately [-0.04, 0.0, 0.30] m in torso coordinates
attitude scale: 0 (single-point rope)
support fraction: 0.50
```

The useful fore/aft attachment region was approximately `-0.025` to `-0.05 m`.
At the selected `-0.04 m` point, all nine `roll={-2,0,+2} deg` by
`pitch={-2,0,+2} deg` perturbations passed 30 policy seconds. The historical
300-second entry, all three failed hardware entry snapshots, and an all-default
joint entry also passed 30 seconds in both the current Sonic and untouched
official AgiBot MuJoCo domains. The default entry was the cleanest short replay:
tilt fell from about `6.14 deg` to `0.26 deg`, with an ankle-speed peak near
`0.159 rad/s`.

Both dynamics domains then passed 300 policy seconds from the default entry.
Final tilt was `1.65 deg` in the current domain and `1.63 deg` in the official
domain. Late ankle velocity was effectively zero, foot load settled near
`42.6%` body weight, and neither waist nor ankle was pinned at a mechanical
limit. This is evidence for a bounded controller under a realistic supported
contact setup; it is not evidence for unsupported standing because the virtual
rope still carries half the nominal weight.

Decision: do not tune gains or target slew again yet. Support/contact setup is
the active experimental variable. The next powered run must preserve this
controller and collect enough HAL telemetry to distinguish realized torque,
motor protection, zero/reference mismatch, and contact geometry.

Remote replay artifacts:

```text
/home/yu/projects/ZHY/x2_official_rl_deploy_v1/sonic_standstill_20260830/
  current/results/current_torso_m004_gantry050_default_300policy.csv
  official/results/official_torso_m004_gantry050_default_300policy.csv
```

## 2026-08-30 HAL telemetry stage

The source tree now ingests the v0.9 `JointState` fields `effort`, `coil_temp`,
`motor_temp`, and `motor_vol`, plus `JointStateArray.state.value` for leg,
waist, arm, and head. The logger is being extended to persist:

```text
policy_target_pos.csv   pre-safety policy target in MuJoCo order
target_pos.csv          final target republished to HAL
tracking_error.csv      target_pos - measured q
joint_effort.csv        measured HAL effort (N m)
stiffness.csv           final Kp
damping.csv             final Kd
pd_torque.csv           Kp*(target-q) - Kd*dq estimate
coil_temp.csv
motor_temp.csv
motor_voltage.csv
domain_state.csv        leg/waist/arm/head protection state
```

This stage changes observation only. It does not change policy output, gains,
slew, safety gates, writer frequency, or MC/HAL ownership.

The telemetry source was built successfully against the SoC1 v0.9 ABI in an
isolated install. Existing C++ tests passed `2/2` on Mac and `2/2` on SoC1. The
isolated binary is:

```text
path: runtime_suspended/ws/install_telemetry_20260830/
      agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref
sha256: 7fa51a4df2553fcd52158dfb62fe6bb7dfa71e79ea51b718bd4f086e94efe40c
```

A three-second fixed-StandStill dry-run then completed 150 policy rows while
`AimdkIo` explicitly reported `command publishers disabled`. All 16 CSV files
had constant row width and 150 aligned data rows. Measured effort ranged from
`-1.905` to `+0.908 N m`, reported motor voltage reached `42`, and all four
domain states remained `0`. After exit, every HAL command topic still had
exactly one publisher, the official MC. Dry-run evidence is on SoC1 at:

```text
/tmp/x2_telemetry_dryrun_150517
```

The v0.9 head-yaw thermal field remains known-invalid (`coil_temp=0`, rapidly
changing motor values above 100); the existing `ThermalMonitor` ignores only
that unsupported head row. Body-joint telemetry is retained unchanged.

No powered probe has used this telemetry build yet. The prior source files are
backed up on SoC1 at:

```text
source_backups/pre_telemetry_20260830_145956
```

The first powered candidate has a dedicated hash-gated wrapper and does not
replace the historical entrygate runtime:

```text
wrapper: run_x2_supported_neutral_damped_telemetry30.sh
wrapper sha256: a147673e0553ff733c3aa2551405b6a30a891cbbc4813d7cf0cdbcd3de4b89f8
shared runner sha256: ca67c3caba7671cf72778df1ba86f99c5147eb7f643d81ff99d65a3b0becc088
workspace: runtime_suspended/ws_telemetry_20260830
confirm token: X2_SUPPORTED_NEUTRAL_DAMPED_TELEMETRY30_START
writer: 250 Hz
reference: fixed StandStill
supported target slew: 0.12 rad/s
bounded policy duration: 30 s
```

The shared runner's only new behavior is the explicit
`--supported-neutral-damped-30` case. Existing 5-second, 0.30-slew, garment,
and all historical modes retain their previous arguments.

### Resume point

1. Freeze the exact `neutral_damped`, `0.12 rad/s` candidate and verify one HAL
   writer before power is enabled.
2. Run three separate 30-second fixed-reference probes with reproducible strong
   torso support; reject growth, toe rise, or any non-zero domain state.
3. Only after three passes, reproduce 300 seconds. Garment input remains out of
   the powered loop until this gate is complete.

### 2026-08-30 15:23 CST: post-battery powered-probe preparation

The battery restart cleared the previous SSH/deploy session. Read-only checks
found no residual `x2_deploy_onnx_ref`, suspended-start, or custom handoff
process. The four HAL command topics each had exactly one publisher (official
MC), and leg, waist, arm, and head feedback each produced a fresh sample.

The staged files were re-hashed before launch and matched the frozen record:

```text
run_x2_supported_neutral_damped_telemetry30.sh
  a147673e0553ff733c3aa2551405b6a30a891cbbc4813d7cf0cdbcd3de4b89f8
run_x2_suspended_sonic.sh
  ca67c3caba7671cf72778df1ba86f99c5147eb7f643d81ff99d65a3b0becc088
x2_deploy_onnx_ref
  7fa51a4df2553fcd52158dfb62fe6bb7dfa71e79ea51b718bd4f086e94efe40c
```

The first launch attempt exited before deploy because the fresh SSH environment
did not contain `X2_SOC0_PASSWORD`; it did not stop MC or publish HAL commands.
The same hash-gated wrapper was relaunched with the existing SoC0 credential and
is now waiting at the suspended cold-start confirmation gate:

```text
runtime log:
runtime_suspended/logs/suspended_sonic_20260830_152344
state: STANDBY, fresh pose cached
writer: suppressed
MC: unchanged/running
next physical gate: robot firmly suspended, exclusion zone clear, E-stop ready
```

No powered transition has occurred at this checkpoint.

### 2026-08-30 15:25-15:36 CST: takeover and feedback settling

After physical suspension and E-stop readiness were reconfirmed, the operator
gate was accepted. SoC0 `mc_app_main` PID 2239 was paused while HAL remained
running. The launcher proved the MC command stream silent and found no unknown
command publisher before enabling the custom writer. The telemetry runtime then
completed cached-pose PD acquisition and the reference-pose ramp:

```text
writer: 250 Hz
reference: fixed StandStill
PD acquire: 3.0 s at cached suspended pose
reference ramp: 5.0 s, max delta 0.655 rad, limit 0.15 rad/s
terminal startup state: READY_FOR_GROUND, policy OFF, custom static PD active
```

The post-pause state callbacks initially arrived in DDS bursts, with reported
ages typically 1-2 s and a maximum observed leg/waist age near 3.24 s. The
policy gate correctly remained `WAITING`; no `load` or `policy` command was
sent. This was not a lost writer-rate or callback-isolation configuration:

- launcher output confirmed `HAL writer=250Hz`;
- state subscriptions retained `SensorDataQoS().keep_last(1)` and their
  reentrant state callback group;
- control inference and writer timers retained separate callback groups;
- an independent read-only SoC1 subscriber simultaneously received the same
  leg state continuously at approximately 999-1004 Hz, proving HAL and the
  inter-PC transport were live.

After DDS discovery/endpoint settling, the deploy reader recovered without a
restart. The gate became and remained `ARMED`, with leg/waist/arm near 1000 Hz,
head near 333 Hz, IMU near 500 Hz, and reported ages around 0-3 ms. MC remains
paused and custom static PD remains active. The next physical action is to
position the feet just above the floor while keeping the gantry load-bearing,
then enter `load`; policy remains off until a separate accepted `policy` gate.

### 2026-08-30 15:42-15:56 CST: first 16-channel powered probe

`load` entered static PD with fresh feedback, one 250 Hz HAL writer, MC still
paused, and all four domain states clear. The temporary absolute entry gate
required pelvis pitch `+5..+9 deg`. The natural supported pose measured about
`-8 deg`, so the operator had to pull the upper body through roughly 16 degrees
to satisfy the gate. A side photograph confirms that the accepted `+8 deg`
pose was visibly and materially leaned under the rear upper-torso rope. The
absolute pitch gate therefore encoded a historical harness pose, not neutral
standing, and must not be reused.

The fixed StandStill probe entered at roll/pitch/tilt
`-1.29/+8.00/8.10 deg` and returned automatically after `3.740 s` when the
tilt envelope was exceeded. Pelvis tilt grew to `13.28 deg`; no DomainErrorState
was raised. Both ankle-pitch joints already entered with approximately
`0.296 rad` command-to-position error. Final HAL targets moved from `-0.363`
toward zero at the frozen `0.12 rad/s` slew, while peak tracking errors reached
`0.582 rad` left and `0.542 rad` right. Right ankle speed peaked at
`0.531 rad/s`.

The new telemetry separates requested control from hardware response. Ankle
Kp/Kd remained fixed at `32.064/3.003`; estimated versus HAL-reported peak
effort matched closely (`18.40/18.37 Nm` left and `17.11/16.91 Nm` right).
Leg, waist, arm, and head domain states remained zero, leg bus voltage remained
`51-52 V`, and temperatures were normal. This is evidence against a dropped
writer, gain loss, voltage collapse, or an active motor protection limit. The
failed probe was dominated by the artificial harness/contact entry and its
large pre-existing ankle load error; it is not a valid test of neutral Sonic
balance.

After policy returned to static PD, the operator restored the visually natural
pose and measured pelvis pitch returned to about `-8 deg` without a fault. The
next candidate must gate relative to a freshly captured natural supported pose
and reject large pre-policy ankle tracking error. Do not retry by forcing the
robot into the old positive-pitch window.

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_152344_powered1
analysis_reports/standstill_20260830_powered1/full_telemetry/README.md
analysis_reports/standstill_20260830_powered1/full_telemetry/summary.json
analysis_reports/standstill_20260830_powered1/full_telemetry/selected_trace.csv
scripts/analyze_powered_telemetry_probe.py
```

### 2026-08-30 16:39-16:42 CST: relative-entry writer adoption

The old absolute-entry runtime remained stable in static PD for more than
3500 seconds, but its launcher owned the `SIGSTOP` lifecycle of SoC0 MC and
would resume MC on ordinary exit. A narrow `--adopt-paused-mc` launcher mode
was added for a suspended writer-to-writer replacement. It requires the unique
`mc_app_main` process to already be paused, does not signal it, and deliberately
leaves it paused during cleanup.

The relative-entry binary first reached `STANDBY` with its writer suppressed
and a separate debug port (`5559`). The old launcher was then frozen to disable
its cleanup trap, its old C++ writer accepted `lifted` and exited normally, and
the old process tree was removed. Only after the old writer PID had disappeared
did the new launcher prove 0.25 seconds of command-wire silence and trigger the
new writer. There was no interval with two active HAL writers and MC was never
resumed.

Current verified state:

```text
runtime log: runtime_suspended/logs/suspended_sonic_20260830_163947
deploy PID: 58822 (only active x2_deploy_onnx_ref binary)
MC worker: PID 2239, state Tl (still SIGSTOP'd)
writer: 250 Hz
reference: fixed StandStill
startup: PD_ACQUIRE 3.0 s -> POSE_RAMP 5.0 s -> READY_FOR_GROUND
feedback gate: ARMED, reported ages approximately 0-3 ms
policy: OFF
```

The next gate is physical: lower the feet to light contact while the gantry
still carries load, then enter `load`. That command captures the relative
pelvis baseline; no `policy` command is permitted until the new relative
pitch/roll and `0.15 rad` ankle-pitch tracking-error gates report stable.

### 2026-08-30 16:45-16:47 CST: relative-entry powered probe 1

At light foot contact, `load` captured pelvis pitch/roll
`-11.65/-0.60 deg`. The relative pose stayed within about `0.02 deg`, maximum
joint speed stayed `0.007 rad/s`, and ankle-pitch tracking error stayed
`0.099 rad`; the entry gate became ready after the required two-second window
and remained ready for more than 30 seconds before policy entry.

The first relative-entry fixed-StandStill probe then completed its full bounded
duration: 1500 aligned samples over 29.979 seconds at a 20.000 ms median sample
period. Pelvis tilt fell from `11.66 deg` to `2.26 deg`; after the initial
uprighting transient it remained near pitch/roll `-0.43/+2.22 deg` from about
9 seconds through timeout. Peak measured joint speed was `0.336 rad/s` at the
right elbow; left/right ankle-pitch peaks were `0.104/0.055 rad/s`. No action
clip fired and all leg, waist, arm, and head domain states remained zero.

The final load-bearing equilibrium retained position error at the left/right
ankle pitch (`0.425/0.314 rad`) and waist pitch (`0.438 rad`). This is not a
writer or torque-path failure: measured versus estimated PD effort remained
consistent, with ankle peaks about `15.85/11.69 Nm`, while bus voltage remained
`47-48 V` and temperatures were normal. The position error is the mechanism by
which the position-only PD controller generates sustained gravity/contact
torque under this supported condition.

At the 30-second timeout the controller intentionally froze the final powered
policy command instead of returning to the original harness-loaded pose.
Consequently the original load-relative entry reference is now about 11 degrees
away and the ankle entry error is above `0.15 rad`; the gate correctly refuses
an immediate second pulse. This is not a probe fault. A genuinely independent
second probe should re-suspend the robot, exit through `lifted` while leaving MC
paused, and relaunch the same relative-entry runtime through
`--adopt-paused-mc`. Physical confirmation of no toe rise, oscillation, or
abnormal sound is still required before marking probe 1 as accepted.

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_163947_powered1
analysis_reports/standstill_20260830_relative_powered1/full_telemetry/README.md
analysis_reports/standstill_20260830_relative_powered1/full_telemetry/summary.json
analysis_reports/standstill_20260830_relative_powered1/full_telemetry/selected_trace.csv
```

Operator observation after the bounded policy interval: no clear abnormal
motion, toe rise, oscillation, or abnormal sound was noticed; at most there was
a very small ambiguous movement. Probe 1 is therefore recorded as a tentative
physical pass in addition to its telemetry pass.

Static PD then held the final policy command for approximately 460 seconds
without a stability fault. While the operator lifted the robot, the resulting
manual joint motion crossed the `1.0 rad/s` ground-load speed guard. This was
long after the 30-second policy interval and is classified as an expected
lift-induced guard event, not probe instability. The `lifted` command then
terminated the writer normally. No custom HAL writer remains, and SoC0
`mc_app_main` PID 2239 remains paused (`Tl`); official MC was not restored.

### 2026-08-30 17:00-17:04 CST: relative-entry powered probe 2

The same frozen configuration was relaunched independently through
`--adopt-paused-mc`; exactly one 250 Hz HAL writer was active and official MC
remained paused. `load` captured pelvis pitch/roll `-9.81/-0.53 deg`. The
relative entry stayed effectively fixed, maximum measured speed was
`0.007 rad/s`, ankle-pitch entry error was `0.115 rad`, and the two-second gate
became and remained ready before policy entry.

The second bounded StandStill probe produced 1500 aligned samples over 29.979
seconds at a 20.000 ms median period. Pelvis roll/pitch/tilt changed from
`-0.52/-9.81/9.82 deg` to `+1.89/-0.32/1.92 deg`. Roll reached one bounded
`+3.09 deg` peak at 8.92 seconds; from 12 seconds through timeout it stayed in
`+1.84..+1.90 deg`, while pitch stayed in `-0.33..-0.26 deg`. This confirms the
operator's observation of a small sway as a settling transient, not a growing
oscillation.

Peak joint speed was `0.275 rad/s` at the right elbow; left/right ankle-pitch
peaks were `0.140/0.067 rad/s`. All domain states remained zero and no action
clip or runtime fault occurred. Left/right ankle-pitch tracking error peaked at
`0.520/0.536 rad` and ended at `0.469/0.325 rad`; measured effort peaks
`16.38/17.03 Nm` matched estimated PD peaks `16.33/17.15 Nm`. This again shows
intentional sustained load torque rather than a missing or limited command
path.

After timeout, static PD held the final command until the robot was lifted.
The changing harness pose then exceeded the load-relative tilt guard, after
the policy interval; it does not invalidate the probe. `lifted` stopped the
writer normally. No custom HAL writer remains and SoC0 MC PID 2239 remains
paused (`Tl`). Probe 2 is accepted as a telemetry and physical pass with one
minor bounded settling sway.

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_165839_powered2
analysis_reports/standstill_20260830_relative_powered2/full_telemetry/README.md
analysis_reports/standstill_20260830_relative_powered2/full_telemetry/summary.json
analysis_reports/standstill_20260830_relative_powered2/full_telemetry/selected_trace.csv
```

### 2026-08-30 17:12-17:14 CST: relative-entry powered probe 3

The third independent launch used the same hash-gated binary and frozen 250 Hz
StandStill configuration. `load` captured pelvis pitch/roll
`-9.21/-1.14 deg`; the relative entry changed by only about `0.01 deg`, maximum
speed was `0.007 rad/s`, feedback remained fresh, and ankle-pitch entry error
was `0.130 rad`, below the `0.15 rad` gate.

The policy completed all 1500 control ticks / 30 seconds. It converged to
approximately pitch/roll/tilt `-0.41/+2.54/2.57 deg`; from about 11 seconds
through timeout the pose and `0.007 rad/s` maximum speed remained essentially
fixed. Runtime reported zero action-clip ticks and no policy fault. The
operator reported no abnormal motion, toe rise, or sound. Probe 3 therefore
passes its runtime telemetry and physical observation. Full 16-file offline
analysis confirmed 1500 aligned samples over 29.979 seconds with a 20.000 ms
median sample period, zero malformed rows, and zero nonzero domain-state
samples. Pelvis roll/pitch/tilt changed from `-1.13/-9.22/9.29 deg` to
`+2.54/-0.41/2.57 deg`; peak roll was `2.81 deg`. Peak joint speed was
`0.263 rad/s` at the right elbow. Left/right ankle-pitch tracking error peaked
at `0.541/0.543 rad`, while measured effort peaks `17.03/17.26 Nm` matched
estimated PD peaks `17.10/17.35 Nm`.

After the policy interval, static PD held the final command. The later
load-relative tilt fault occurred while the robot was being lifted, not during
the bounded probe. `lifted` returned the launcher to its shell before the
battery change. The battery then ran out and the operator began replacing it.
No 300-second probe was started. After reboot, first verify that no writer is
alive and reassess MC state rather than assuming the pre-power-cycle paused
state survived.

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_171018_powered3
analysis_reports/standstill_20260830_relative_powered3/full_telemetry/README.md
analysis_reports/standstill_20260830_relative_powered3/full_telemetry/summary.json
analysis_reports/standstill_20260830_relative_powered3/full_telemetry/selected_trace.csv
```

### 2026-08-30 17:21 CST: post-battery baseline and relative 300 preparation

After reboot, no custom deploy or launcher process was present. SoC0
`mc_app_main` restarted normally as PID 2129 (`Sl`), the only HAL leg-command
publisher was `mc_ros2_node`, and leg-state feedback measured approximately
1000 Hz. No pre-power-cycle paused-state assumption will be reused.

A dedicated relative-entry 300-second mode and hash-gated wrapper were added.
They share the exact accepted relative entry gates, neutral_damped gains,
`0.12 rad/s` target slew, LPFs, clamps, and 250 Hz writer with the 30-second
wrapper. The only policy behavior change is
`--supported-policy-seconds 30.0 -> 300.0`.

Verified local and robot hashes:

```text
run_x2_suspended_sonic.sh
57d4bd3de534486bfe043b54b9b0d860a742a2fd55e3a2fb7534b0489e9816cd

run_x2_supported_relative_neutral_damped_telemetry300.sh
580f8b26a1c94a2257ca11c7e1c81f1c82730f3fa0dc0c6450f74ea52ae87f04

relative-entry deploy binary (unchanged)
be5ce11100c8aa9d1f7923e89e43ea02734d5b9d61ed8a172665879b9a9215d0
```

### 2026-08-30 18:23 CST: ankle-pitch Kp=40 supported probe 1

The next powered experiment changed one control variable from the accepted
relative-entry parent: left and right ankle-pitch Kp increased from `32.064`
to `40.000`. Ankle-pitch Kd remained `3.003`; the 250 Hz writer, `0.12 rad/s`
target slew, leg/waist LPFs, four-second policy ramp, fixed StandStill
reference, entry gates, and protection limits were unchanged.

```text
launcher: run_x2_supported_relative_neutral_damped_anklep40_30.sh
runtime log: runtime_suspended/logs/suspended_sonic_20260830_182345
entry pelvis roll/pitch/tilt: +0.39/-8.88/8.89 deg
entry ankle-pitch tracking error: 0.122 rad
policy interval: 1500 samples / 29.979 s
```

The complete policy interval passed. Pelvis roll/pitch/tilt converged to
`+0.03/-0.52/0.52 deg`; after approximately 14 seconds, tilt remained near
`0.5 deg` through timeout. Peak measured joint speed was `0.250 rad/s` at the
right elbow. No action clip, stale-feedback event, runtime fault, or nonzero
leg/waist/arm/head domain state occurred during policy. The operator reported
the robot stable.

The higher Kp did not eliminate loaded tracking offset. Left ankle-pitch error
was `0.122 rad` at entry, peaked at `0.554 rad`, and ended at `0.413 rad`;
measured effort peaked at `22.07 N m` and matched the `22.08 N m` requested PD
estimate. Right ankle-pitch error ended at only `0.071 rad`, so a substantial
left/right asymmetry remains. Waist pitch independently ended with target
`-0.200 rad`, measured position `+0.236 rad`, error `-0.436 rad`, and measured
effort near the `6.26 N m` PD estimate. The control path and published gains
were therefore active; this run does not distinguish joint reference/contact
asymmetry from insufficient effective load authority.

After the bounded policy interval, the runtime intentionally held the final
policy command with policy OFF. The operator then manually changed the robot's
pose, producing the later approximately `4.9 deg` pelvis tilt and the visibly
leaning photograph. Those post-policy values are not evidence of policy
divergence and are excluded from the pass decision. The robot was subsequently
power-cycled rather than exited through `lifted`; after reboot there was no
custom writer, official `mc_ros2_node` was the sole HAL command publisher, and
leg feedback was fresh at approximately 1000 Hz.

This is Kp=40 reproduction `1/3`, not authorization for a 300-second run. Two
more independent 30-second probes must use the same hashes and support/contact
procedure, with no manual pose adjustment during the policy interval. Do not
raise target slew. If all three remain bounded, compare the repeated
left/right ankle and waist offsets before deciding between a 300-second
reproduction and a one-variable reference/gain diagnostic.

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_182345_anklep40_complete
analysis_reports/standstill_20260830_anklep40_powered1/
  full_telemetry_complete/README.md
  full_telemetry_complete/summary.json
  full_telemetry_complete/selected_trace.csv
```

### 2026-08-30 19:20 CST: ankle-pitch Kp=40 probe 2 failure

The second independent Kp=40 launch used the same binary and controller
settings as probe 1. It captured a stable supported entry at pelvis
roll/pitch `-1.28/-8.11 deg`; feedback was fresh, maximum static speed was
`0.007 rad/s`, and the existing entry gate admitted an ankle-pitch tracking
error of `0.144 rad` because its limit was `0.150 rad`.

Policy returned automatically after `9.14 s / 458` aligned samples when left
ankle-pitch speed reached `0.910 rad/s`, exceeding the frozen `0.8 rad/s`
joint-speed guard. Pelvis pitch crossed from `-8.11` to `+10.69 deg` and peak
tilt reached `10.75 deg`. Left/right ankle-pitch speed reached
`0.910/0.604 rad/s`; tracking error peaked at `0.524/0.485 rad`. Requested PD
effort again matched HAL effort, bus voltage remained `48 V`, and all four
domain-state channels remained zero. The operator observed the corresponding
physical failure as forward lean and toe rise. The two-second automatic return
completed and the robot was re-suspended before `lifted` ended the custom
writer.

This failure is not unique to Kp=40. The earlier Kp=32 session
`suspended_sonic_20260830_172543_powered4` entered with almost identical
left/right ankle-pitch error `0.144/0.134 rad` and pelvis roll `-1.30 deg`, then
failed after `8.78 s` with forward/lateral growth. Across the six comparable
relative-entry probes, the measured maximum ankle-pitch entry error separates
the observed outcomes:

| Probe | Kp | Entry ankle error L/R | Duration | Physical result |
| --- | ---: | ---: | ---: | --- |
| relative 1 | 32.064 | 0.096 / 0.099 rad | 30 s | pass |
| relative 2 | 32.064 | 0.106 / 0.115 rad | 30 s | pass |
| relative 3 | 32.064 | 0.121 / 0.130 rad | 30 s | pass |
| relative 4 | 32.064 | 0.144 / 0.134 rad | 8.78 s | forward/lateral failure |
| Kp=40 probe 1 | 40.000 | 0.122 / 0.117 rad | 30 s | pass |
| Kp=40 probe 2 | 40.000 | 0.144 / 0.133 rad | 9.14 s | forward toe-rise failure |

The old `0.150 rad` entry gate therefore admits two repeated bad contact/load
states. A new isolated mode tightens only this measured gate to `0.135 rad`;
it does not alter policy output, Kp/Kd, target slew, LPF, writer rate, tilt or
speed protection. This threshold retains all four recorded passes and rejects
both matching failures. It is an empirical supported-contact inclusion gate,
not proof that ankle error is the policy's underlying training-domain cause.

```text
shared runner:
  run_x2_suspended_sonic.sh
  sha256 7b6edcb1e23f2e7507335622ae3f7c57dce0754ddeba289d22bb17582a268ff6

isolated wrapper:
  run_x2_supported_relative_neutral_damped_anklep40_entry135_30.sh
  sha256 40a71e8e468dbc07b3a5f028782e8eb29833e81bd5ece43b2aa02f91a004fb01

confirm token:
  X2_SUPPORTED_RELATIVE_NEUTRAL_DAMPED_ANKLEP40_ENTRY135_30_START
```

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_192035_anklep40_powered2
analysis_reports/standstill_20260830_anklep40_powered2/full_telemetry/
```

### 2026-08-30 19:42-19:53 CST: entry-0.135 probe 1 and support diagnosis

The first Kp=40 run with the ankle-pitch entry gate tightened from `0.150` to
`0.135 rad` entered at left/right ankle error `0.092/0.086 rad` and completed
all `1500` policy samples / `29.979 s`. Feedback stayed aligned at a
`20.000 ms` median period, every domain-state channel remained zero, and peak
joint speed was `0.360 rad/s`. The runtime therefore did not trip a safety
guard.

This run was not a physical stability pass. Pelvis roll/pitch changed from
`-0.40/-11.29 deg` to `+4.12/-0.10 deg`, with visible forward-load tendency
and the left foot unloading against the gantry. Raw policy-to-HAL backlog
peaked at `0.992/0.606 rad` for left/right ankle pitch. At timeout the left
ankle HAL target and measurement were `+0.095/-0.308 rad`; the right values
were `-0.179/-0.418 rad`. Waist pitch stayed limited at a `-0.200 rad` HAL
target while measured position ended at `+0.218 rad`. Measured effort matched
the requested PD estimate, peaked at `17.08/15.50 N m` at the ankles and
`6.81 N m` at waist pitch, and bus voltage remained `46-47 V`.

The old timeout path then froze that arbitrary final policy command with the
policy OFF. This made the asymmetric ankle phase persist instead of returning
to the pre-policy static PD hold. Timeout now follows the existing two-second
`SUPPORTED_POLICY_RETURN` path, identical to an operator `stop`; it does not
resume official MC and does not change policy, gains, clamps, filters, slew,
writer rate, or protection limits.

The read-only official-MC snapshots add a stronger support-geometry result.
With MC publishing zero position/gain/effort commands, waist pitch measured
exactly `+0.3491 rad` both fully suspended and at foot contact. Pelvis/torso
pitch was `-12.87/+7.11 deg` fully suspended and `-11.00/+8.98 deg` at contact.
The upper-torso rope therefore folds the passive waist to approximately
`+20 deg` before custom PD or Sonic starts. The latest policy entry still had
waist pitch `+0.325 rad`. This real constraint is absent from the point-force
virtual gantry and explains why all recorded hardware entries pass the
dual-domain replay yet real outcomes remain support-sensitive.

Decision: do not raise Kp, target slew, or policy clamps next. First run a
policy-OFF supported static-PD diagnostic while changing only the support
geometry. The entry target is a waist pitch materially below the passive
`+0.3491 rad` stop, with the gantry still carrying load. Only then repeat a
bounded fixed-StandStill policy probe using the corrected timeout return.

After the robot rebooted, read-only ownership checks found no custom deploy or
launcher process. Every HAL command topic had exactly one publisher,
`mc_ros2_node`, and SoC0 `mc_app_main` was running. The corrected relative-entry
source built successfully on SoC1 and both package tests passed via direct
`ctest` (`test_obs_builder`, `test_supported_policy`). The normal `colcon test`
discovery command was not used because the migration tree contains duplicate
backup copies of the same package name.

```text
source x2_deploy_onnx_ref.cpp:
  58417e5e01aafbb273e8825c2639c41466a25805c8eb7522740a0e980cbd578d
installed x2_deploy_onnx_ref:
  e9bfe4d5fcb330802a9f1ccbef85a0fc27370605653bac04a12aeb2159886aec
entry-0.135 wrapper:
  dfc93e0ac8c5cedb9365d121f3b7e446a6ae3b540dce3d19d33930ca18c31bb4
```

Artifacts:

```text
analysis_logs/suspended_sonic_20260830_194201_entry135_complete/
analysis_reports/standstill_20260830_anklep40_entry135_powered1/full_telemetry/
analysis_logs/static_load_fully_suspended_mc_20260830_175354.json
analysis_logs/static_load_feet_contact_mc_20260830_175707.json
```

### 2026-08-31 M1 policy-off debug interface verification

This was an isolated dry-run interface test, not another powered StandStill
probe. The `GROUND_LOAD_HOLD` branch now publishes the existing `x2_debug`
schema only while the measured RobotState is fresh. The SoC1 scratch build
passed `2/2` CTests; its exact binary SHA-256 is
`3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f`.

With HAL command publishers disabled by `--dry-run`, official `mc_ros2_node`
remained the sole publisher on leg/waist/arm/head. The exact candidate entered
simulated `GROUND_LOAD_HOLD`, emitted live debug on `:5557`, and drove the
safety proxy through `WAIT_ROBOT -> WARMUP -> STANDSTILL_READY`. Guarded
StandStill error was at most `2.9564e-08 rad`, velocity was zero, and debug
timestamps advanced by about `20 ms`. Source and debug loss each produced the
expected terminal LOCKOUT. Cleanup left no test process or listener.

Evidence is retained in
`runtime_suspended/logs/m1_ground_debug_dryrun_20260831_104650/`. This closes
`M1_PREPOLICY_DEBUG_READY`; the next balance action is one 300-second supported
StandStill acceptance using this same binary and frozen configuration. It does
not authorize garment input, unsupported standing, or a parameter change.

### 2026-08-31 M2 final-parent preflight

The robot was fully suspended and remained under official MC during a fresh
read-only preflight. No custom controller or garment/proxy process was present,
ports `5555/5556/5557/51234/51237` were free, and each leg/waist/arm/head HAL
command topic had one official `mc_ros2_node` publisher. Each corresponding
state topic had one publisher.

`run_x2_supported_final_parent_telemetry300.sh` now hash-gates the exact M1
candidate (`3c4dee77...c3559f`), frozen model (`8ccc42a8...271e9`), current
powered launcher (`7b6edcb1...68ff6`), and 250 Hz
`--supported-neutral-damped-relative-300` profile. The wrapper SHA-256 is
`1b73418811619fd701ff3fa3195dfe3ea0e40edf8f755b42b0d515ae92b184d8`.
Local and SoC1 syntax/hash checks passed; no MC or HAL mutation occurred. M2 is
prepared but remains incomplete until the full 300-second policy interval,
normal lifted cleanup, and final publisher-ownership verification pass.

### 2026-08-31 11:13-11:31 CST: M2 attempt invalidated by sling release

The exact M1 candidate entered the final-parent 300-second profile from a
fresh entry at pelvis roll/pitch `-0.73/-9.93 deg`, ankle-pitch tracking error
`0.127/0.100 rad`, and maximum joint speed `0.007 rad/s`. With the original
heavy sling support unchanged, policy remained bounded for more than 170
seconds. From seconds 20-170, pelvis tilt stayed within `1.19-1.30 deg` and
maximum measured speed was `0.018 rad/s`.

This was not autonomous standing. During the same quiet interval, left ankle
pitch remained approximately `0.494 rad` away from its HAL target and waist
pitch remained `0.420 rad` away. Their median measured efforts were
`15.91 N m` and `5.90 N m`. The photograph taken during the run also showed
the upper-body sling taut and the torso held by a rearward support moment.

At about policy second 174 the operator loosened the sling. That changed the
physical test condition while policy was on, so this run is INVALID as the
required fixed-support 300-second acceptance. It nevertheless showed that the
current loaded equilibrium does not transfer directly to self-support: within
about 4.4 seconds ankle-pitch errors reached `0.686/0.615 rad`, pelvis tilt
reached `13.34 deg` during policy, and waist-pitch speed reached `0.859 rad/s`.
The joint-speed guard returned to static PD. Motion during return/load hold
briefly reached about `40 deg` tilt before the sling caught and the system
settled.

There was no feedback, domain-state, voltage, or ownership fault. After full
suspension, `lifted` ended the writer. Because `SIGCONT` resumed SoC0 PID 2055
without restoring its ROS publisher, official MC was fully restarted with EM.
Final ownership was clean: no custom process, and `mc_ros2_node` was the sole
publisher for leg, waist, arm, and head commands; all state groups were live.

Decision: do not count this as M2 pass/fail and do not approve the garment
parent. Fixed-support stillness is already demonstrated; the next missing
fact is controlled support transfer. First use policy-off static PD to correct
the upper-body sling geometry and reduce both ankle-pitch and waist-pitch
tracking errors. Then run one explicitly marked, single-increment support
release with a hold plateau. Do not continuously loosen by hand. A 100-second
test with a fixed first 20 seconds is permitted only after the single-step
probe closes its tracking, velocity, and tilt gates.

Artifacts:

```text
analysis_logs/suspended_sonic_20260831_111359_m2_intervened/
analysis_reports/standstill_20260831_m2_intervened/full_telemetry/
```

### 2026-08-31 13:22-13:40 CST: controlled support-step candidate

The M2 freehand sling release is replaced by an explicit single-step protocol.
The isolated SoC1 candidate adds no control tuning: model, `neutral_damped`,
reconstructed pelvis, 50/250 Hz rates, `0.12 rad/s` slew, gains, LPFs, clamps,
joint mapping, and safety return remain frozen. Its production mode requires
20 seconds of unchanged support followed by five continuously gated seconds:
ankle-pitch errors `<=0.20 rad`, waist-pitch error `<=0.20 rad`, maximum joint
speed `<=0.05 rad/s`, pelvis tilt `<=3 deg`, and one-second tilt growth
`<=0.10 deg`. An accepted `support_step` writes one marker, never actuates the
hoist, and starts a 20-second fixed-support plateau before automatic return to
static PD.

```text
workspace: runtime_suspended/ws_support_step_20260831_v3
binary: 9d4916a8e5fd64e4ac3ddb0f0a32903ccc90b9e86b4e146cc6cabefe63695483
source: d13205445b1a04d25745aad2d3c8e564ff9ecda9acc0a85f59041f2990679722
wrapper: 19148f68824f608a91e05ba181ce96cffb7a7487644594987ba5a98f9616667c
powered launcher: 247e02c28fe4fee8d5746f5b2c1bef758b10c64f8598e5d23fe803c21deee6a6
handoff launcher: d16d52db6763edec280238bfad83b91f04b76f4fa7d66831307b5bd9ea9c63b4
model: 8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
```

SoC1 CTest passed `2/2`. The final process dry-run waited for an explicit
`feedback_gate=ARMED`, then completed the full state sequence with command
publishers disabled. Its `tick.csv` has 449 rows, all `dry_run=1`, exactly one
`supported_policy_support_step_marker`, and a normal return to
`GROUND_LOAD_HOLD`. Cleanup left no test process and one official
`mc_ros2_node` publisher on each command group. The accelerated dry-run timing
and relaxed thresholds prove orchestration only, not support transfer.

At this historical checkpoint the proposed next action was one support-step
probe. The later `134829` result and D-017 supersede that plan; do not use this
paragraph as current authorization.

### 2026-08-31 14:01: first powered support-step attempt stopped before policy

The run at `runtime_suspended/logs/suspended_sonic_20260831_134829` never
started Sonic policy. At first `load`, the relative entry state was already
quiet and eligible: pitch/roll/tilt `-10.71/-0.78/10.74 deg`, ankle-pitch
tracking maximum `0.101 rad`, and maximum velocity `0.007 rad/s`. The
`0.322 rad` waist tracking error was not an entry-gate field.

Asking the operator to change the sling before policy in order to satisfy the
later `0.20 rad` support-step waist gate was a protocol error. The adjustment
increased tilt to `17.84 deg` and latched the static ground-load guard. A later
visually upright state measured tilt `5.42 deg`, ankle-pitch tracking
`0.169 rad`, waist-pitch tracking `0.281 rad`, and speed `0.007 rad/s`; the
latched fault correctly remained closed.

This agrees with the earlier official-MC snapshots (`+0.349 rad` passive waist
pitch) and `170450` entry (`+0.332 rad`). The support rig cannot make absolute
waist tracking approach zero, so waist error remains telemetry but is removed
from the dedicated support-step decision. The following gates remain:

```text
both ankle-pitch tracking errors <= 0.20 rad
pelvis tilt <= 3 deg after policy settling
maximum joint velocity <= 0.05 rad/s
one-second tilt rise <= 0.10 deg
all conditions continuous for 5 s
```

The relative pre-policy entry gate is unchanged, as are model, gains, LPF,
`0.12 rad/s` target slew, 4-second policy ramp, reconstructed pelvis, 50/250 Hz
rates, clamps, mapping, and bounded return. Local offline build and CTest pass
`2/2`; the candidate has not been ARM64-built or deployed.

After full suspension, `lifted` ended the custom writer. The resumed MC worker
then exited and command publisher counts were zero; network loss prevented the
normal official `stop-app mc` / `start-app mc` recovery. Do not lower or run
the robot until connectivity returns and all four command topics again show
one official `mc_ros2_node` publisher.

### Next route: preloaded fixed support, not on-policy sling adjustment

The 2026-08-30 support-change A/B and M2 both show that changing sling tension
during policy is a large external-wrench disturbance. The next run therefore
sets the final support level before `load`, captures that pose, and keeps both
sling and feet fixed throughout policy. Support is reduced only between
independent runs.

```text
launcher: run_x2_supported_preloaded_fixed100.sh
token: X2_SUPPORTED_PRELOADED_FIXED100_START
duration: 100 s
binary: existing M1 candidate 3c4dee77...c3559f
runner: f14d4054101e66a0de291223f04e9c4470e4d69063c3a6ee8a49200f69c8d982
wrapper: 62cb405d2b6077b1a4ae06ced8c7f1e46a07eed85c74688dfe0b5cdf5c8cbc82
support_step: disabled
```

This launcher is local-only until network recovery, remote hash verification,
and official MC ownership recovery. `--verify-only` checks the M1 binary,
model, powered runner, and `deploy_x2.sh` without starting any process. A
disposable offline runner test proves that relative-100 and relative-300 have
identical generated arguments except for policy duration, and that fixed-100
contains no support-step or ZMQ mode. It is a fixed-support test, not proof of
unsupported balance.
