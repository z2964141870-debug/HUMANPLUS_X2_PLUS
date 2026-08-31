# X2 Sonic Experiment Ledger

## 2026-08-31 M0 post-reboot cleanup verification

**Type:** read-only recovery/pre-build gate; no powered experiment.

The operator confirmed the X2 was fully suspended. The robot had rebooted
about 21 minutes earlier, so the previous `x2_waist_static` tmux session and
custom deploy/writer were absent. No `lifted`, `Ctrl-C`, kill, MC mutation, or
HAL command was issued.

All four HAL command topics had one publisher and identified the official
`mc_ros2_node` publisher endpoint. Measured state rates were approximately
`1001.9 Hz` leg, `1001.0 Hz` waist, `1000.4 Hz` arm, and `333.5 Hz` head. Ports
`5555`, `5556`, `5557`, `51234`, and `51237` were unused, with no garment,
offload, proxy, or custom deploy process present. Result:
`M0_CURRENT_SESSION_CLEAN` passed; isolated source staging/build may begin.

## 2026-08-31 M1 Policy-OFF x2_debug candidate verification

**Type:** isolated build plus dry-run process/network test; no powered policy
experiment.

The three patched sources were synchronized to SoC1 only after M0 passed and
built in the new scratch workspace
`runtime_suspended/ws_ground_debug_20260831`. The existing runtime install was
not overwritten. The exact source hashes were:

```text
x2_deploy_onnx_ref.cpp       ea8595a5eb0573e45b501647ee623e50e636911f0ca5501070216eae39d208e9
supported_policy_gates.hpp   26532959bfaa435f3abde55ca8e2bab8bad141aa36e0e330b1cc4268d855ed62
test_supported_policy.cpp    bc8532008fe1d9d027b66f35dea35c65a5e0590c6637830335bae9252e69bdd0
candidate binary             3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f
```

Direct CTest in the SoC1 build directory passed `2/2`. The proxy and parent
manifest suites passed `18/18` both locally and in the isolated SoC1 test
bundle; the v5.1 byte-level wire self-check passed. The retained 1379-frame
garment replay remained in `WARMUP`, never reached `STANDSTILL_READY`, and
never emitted live output, as expected for that moving/non-neutral capture.

For the process test, the exact candidate ran with `--dry-run`, which disables
all four HAL command publishers in `AimdkIo`. Official MC stayed active. A
zero-amplitude StandStill source published raw v5.1 frames on `:5555`; the
safety proxy consumed raw reference plus candidate `x2_debug` on `:5557` and
published guarded output on `:5556`. After the dry-run candidate entered
`GROUND_LOAD_HOLD`, the proxy traversed
`WAIT_ROBOT -> WARMUP -> STANDSTILL_READY` without ARM or policy entry.
Startup ramp durations were shortened only to exercise the interface quickly;
this run is not evidence for the frozen powered parent configuration.

The READY snapshot reported `output_active=true`, `debug_dry_run=true`, robot
age `0.008710 s`, and a valid unit quaternion. A guarded frame matched trained
StandStill within `2.9564e-08 rad`, had zero velocity, and had quaternion norm
`1.0`. Two direct debug samples had ROS timestamp delta `0.019989 s`; the
policy-off control tick correctly remained zero while timestamps advanced.
When the first synthetic source expired, the proxy entered terminal LOCKOUT at
source age `0.153 s`. In a second run, ending the candidate caused terminal
LOCKOUT at debug age `0.102 s`. Both are expected fail-closed results.

The candidate exited through its normal dry-run `lifted` path. The source and
proxy were then stopped. Ports `5555/5556/5557` were free, no custom process
remained, and leg/waist/arm/head command topics each still had one official
`mc_ros2_node` publisher. Full candidate CSVs and copied process evidence are
retained at:

```text
runtime_suspended/logs/m1_ground_debug_dryrun_20260831_104650/
runtime_suspended/logs/m1_ground_debug_dryrun_20260831_104650/process_evidence/
```

Result: `M1_PREPOLICY_DEBUG_READY` passed. This proves the missing policy-off
debug interface and proxy ordering, not powered StandStill or garment control.

## 2026-08-31 M2 final-parent 300-second preflight

**Type:** powered StandStill acceptance preparation; controller handoff and
policy interval not yet started at the time of this entry.

The operator confirmed that the X2 was fully suspended. A fresh SoC1 read-only
preflight found no custom deploy, garment, offload, or proxy process and no
listeners on ports `5555`, `5556`, `5557`, `51234`, or `51237`. Leg, waist,
arm, and head command topics each had exactly one publisher, identified as the
official `mc_ros2_node`; all four state topics each had one publisher. The
three-second pelvis probe received 4481 samples. Official MC remained active
throughout this preparation.

The final-parent profile wrapper was added without changing controller
parameters. It pins the M1 candidate, frozen model, current powered launcher,
250 Hz writer, and existing `--supported-neutral-damped-relative-300` mode:

```text
candidate binary
3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f

model
8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9

powered launcher: run_x2_suspended_sonic.sh
7b6edcb1e23f2e7507335622ae3f7c57dce0754ddeba289d22bb17582a268ff6

profile launcher: run_x2_supported_final_parent_telemetry300.sh
1b73418811619fd701ff3fa3195dfe3ea0e40edf8f755b42b0d515ae92b184d8
```

The new wrapper passed local and SoC1 `bash -n`, matched hashes after transfer,
and exited with code 2 when invoked without its confirmation token. Result:
`M2_FINAL_PARENT_PREFLIGHT_READY`. This was not 300-second evidence; the first
powered attempt is recorded below.

## 2026-08-31 M2 attempt 1: support changed during policy

**Classification:** INVALID for fixed-support 300-second acceptance. It is a
failed support-transfer observation and must not be entered into the approved
parent manifest.

The exact M1 candidate and frozen M2 wrapper entered policy from a fresh,
stable `GROUND_LOAD_HOLD` state:

```text
runtime log: runtime_suspended/logs/suspended_sonic_20260831_111359
policy samples / duration: 8920 / 178.373033 s
entry pelvis roll/pitch/tilt: -0.731/-9.932/9.959 deg
entry ankle-pitch error L/R: 0.127/0.100 rad
entry maximum joint speed: 0.007 rad/s
```

With the sling kept at the original heavy-support setting, policy was bounded
for more than 170 seconds. During policy seconds 20-170, pelvis tilt had a
`1.239 deg` median and `1.297 deg` maximum, and maximum measured joint speed
was `0.018 rad/s`. This apparent stillness hid a load-dependent equilibrium:

| Joint | Median tracking error | Median measured effort |
| --- | ---: | ---: |
| left ankle pitch | `0.494 rad` | `15.91 N m` |
| right ankle pitch | `0.120 rad` | `3.78 N m` |
| waist pitch | `0.420 rad` | `5.90 N m` |

The operator then loosened the sling while policy remained on. That violated
the fixed-support acceptance condition and changed the external support
wrench, so the remaining interval cannot judge spontaneous policy stability.
Within about 4.4 seconds, left/right ankle-pitch tracking error reached
`0.686/0.615 rad`, policy-interval pelvis tilt reached `13.34 deg`, and maximum
joint speed reached `0.859 rad/s` at waist pitch. The joint-speed guard
returned to static PD over two seconds. Continued physical motion during the
return/load-hold interval reached about `40 deg` tilt before settling.

Feedback remained fresh, all four domain-state channels stayed zero, motor
voltage stayed near `47-49 V`, and measured effort followed the requested PD
estimate. The event is therefore not an MC/HAL ownership, stale-feedback, or
power-bus failure. The proved near-term cause is removal of a large external
support moment while ankle/waist targets were still substantially unrealized.
The remaining root-cause branch is support geometry/load asymmetry versus
joint-zero/realized-authority mismatch versus policy transfer mismatch.

After the robot was fully suspended, `lifted` ended the custom writer. The
original SoC0 worker resumed but did not recreate its DDS publishers, so the
official `stop-app mc` / `start-app mc` recovery was used while suspended.
Final verification found no custom process, one `mc_ros2_node` publisher on
each leg/waist/arm/head command topic, and one publisher on each state topic.

Raw logs and generated analysis are retained at:

```text
gear_sonic_deploy/analysis_logs/suspended_sonic_20260831_111359_m2_intervened/
gear_sonic_deploy/analysis_reports/standstill_20260831_m2_intervened/full_telemetry/
```

**Gate:** do not repeat an unmeasured freehand release or approve the parent.
First change only support geometry under policy-off static PD so the sling is
fall arrest rather than a torso moment. Before any policy-on release, require
provisionally for at least five seconds: both ankle-pitch errors `<=0.20 rad`,
waist-pitch error `<=0.20 rad`, maximum joint speed `<=0.05 rad/s`, pelvis tilt
`<=3 deg`, and no increasing tilt. These are conservative experiment gates,
not a proof of balance. The first release probe gets one marked hoist step and
one hold plateau; a 100-second staged release follows only after that passes.

## 2026-08-31 support-step candidate and process dry-run

**Type:** isolated build, unit test, and read-only process test; no MC mutation,
custom HAL publisher, or powered policy experiment.

The candidate mode `--supported-neutral-damped-relative-support-step` preserves
the frozen model and all controller parameters. It adds a 20-second immutable
support interval, five continuously gated seconds, one operator event marker,
a 20-second fixed-support plateau, and automatic bounded return to the captured
static PD hold. The wrapper does not control the hoist.

```text
SoC1 scratch workspace
runtime_suspended/ws_support_step_20260831_v3

candidate binary
9d4916a8e5fd64e4ac3ddb0f0a32903ccc90b9e86b4e146cc6cabefe63695483

x2_deploy_onnx_ref.cpp
d13205445b1a04d25745aad2d3c8e564ff9ecda9acc0a85f59041f2990679722

run_x2_suspended_sonic.sh
247e02c28fe4fee8d5746f5b2c1bef758b10c64f8598e5d23fe803c21deee6a6

run_x2_supported_neutral_damped_support_step.sh
19148f68824f608a91e05ba181ce96cffb7a7487644594987ba5a98f9616667c

deploy_x2.sh
d16d52db6763edec280238bfad83b91f04b76f4fa7d66831307b5bd9ea9c63b4

model
8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
```

The SoC1 scratch build passed `2/2` CTests. A process-level dry-run waited until
`READY_FOR_GROUND feedback_gate=ARMED`, then traversed
`load -> policy -> support_step -> plateau -> return -> GROUND_LOAD_HOLD ->
lifted`. Accelerated test timing used a 3-second immutable interval and
2-second plateau; physical thresholds were deliberately relaxed to exercise
the state machine only. The 449-row `tick.csv` contained exactly one
`supported_policy_support_step_marker`, 349 policy rows, 11 return rows, and
`dry_run=1` on every row. `AimdkIo` reported command publishers disabled.

Evidence is retained at:

```text
gear_sonic_deploy/analysis_logs/
  x2_support_step_process_test_20260831_v3_marker4/
```

After cleanup, no custom, garment, proxy, or offload process remained and each
HAL command topic still had one official `mc_ros2_node` publisher. Result:
`SUPPORT_STEP_PROCESS_READY`. This closes the software-state-machine gate only.
At this historical checkpoint the proposed next experiment was one powered
support-step probe. D-017 and the later `134829` evidence supersede that plan;
do not run it as the current next action.

Update this file before and after every powered experiment. Failed runs are
evidence and must remain in the ledger and on disk.

## Completed runs

| ID / log | Primary change or purpose | Result | What it means |
| --- | --- | --- | --- |
| `170450` | `neutral_damped`, 250 Hz, policy settled under stronger/longer gantry support | PASS (supported only): 299.968 s / 15000 policy frames | Proven parent configuration; does not prove arbitrary entry or unsupported balance |
| `203843` | writer accidentally restored to 500 Hz | INVALID for balance: bursty feedback, policy stayed `WAITING` | 500 Hz overloads this path; keep normal writer at 250 Hz |
| `211439` | 250 Hz feedback recovery after 500 Hz run | PASS for feedback only after DDS settling | Infrastructure gate, not a balance test |
| `211808` | clean 250 Hz feedback reproduction | PASS for feedback: armed in 2.04 s and remained fresh about 73 s | Feedback continuity is closed at 250 Hz |
| `212937` | `waist2` plus a different already-loaded policy entry | FAIL after 4.14 s; relative tilt return | Confounded; cannot assign cause to waist gains or entry condition |
| `220655` | exact `neutral_damped` software reproduction at 250 Hz, light-contact/load-bearing gantry entry | FAIL after 29.699 s; low-frequency ankle/contact oscillation, automatic return | Same software is not sufficient across physical entries; entry/contact state is active variable |
| `20260829_210922` | corrected applied-action feedback, original `0.12 rad/s` slew, supported five-second probe | FAIL after about 2.96 s; tilt return; no ownership, feedback, or action-clipping fault | Ankle correction was predominantly slew-limited, but this alone did not establish root cause |
| `20260829_222159` | same fixed StandStill family with `0.30 rad/s` slew | FAIL after about 3.20 s; tilt return; peak ankle error/speed increased | Reducing slew backlog did not restore physical tracking; reject further blind slew increases |

## 170450 versus 220655

Both used the same frozen launcher, model, 250 Hz writer, filters, gains,
limits, and reconstructed pelvis state. Their policy-entry states differed:

| Metric at policy entry | `170450` stable | `220655` failed |
| --- | ---: | ---: |
| pelvis roll | +0.406 deg | -5.915 deg |
| pelvis pitch | +7.003 deg | -7.666 deg |
| projected gravity x | +0.12193 | -0.13340 |
| projected gravity y | -0.00703 | +0.10213 |
| left ankle pitch | -0.4795 rad | -0.5015 rad |
| right ankle pitch | -0.4810 rad | -0.5076 rad |
| waist pitch | +0.3320 rad | +0.2952 rad |
| waist roll | +0.1578 rad | +0.1096 rad |
| max measured ankle speed, first 30 s | about 0.153 rad/s | about 0.763 rad/s |

In `220655`, Sonic initially reduced tilt from about 9.67 to 0.70 degrees in
6.5 seconds, then crossed pitch and roll repeatedly. Quaternion tilt reached
about 14.11 degrees near 29.5 seconds and the relative-tilt gate returned the
controller to static PD. Ankle pitch targets swept to roughly +0.24 rad while
measured positions lagged by as much as about 0.67 rad.

The orientation and joint-state differences are correlations. The unmeasured
gantry load and foot contact geometry also changed, so no root cause is
declared.

## Planned experiment E-ENTRY-01

**State:** ready for static acquisition; policy request not yet authorized by
the measured entry state.

**Hypothesis:** matching the known-good pelvis orientation/contact setup before
policy entry removes the early ankle/contact limit cycle without changing the
controller.

**Fixed variables:** firmware, message ABI, SoC split, binary, ONNX model,
`neutral_damped` launcher, reconstructed pelvis, 50/250 Hz rates, gains,
filters, target slew, safety envelopes, and policy duration.

**Only primary variable:** physical entry orientation/support/contact setup.

**Procedure:**

1. Begin fully suspended with official MC as the sole command publisher and a
   person on the physical emergency stop.
2. Launch `run_x2_supported_neutral_damped.sh`; accept the powered handoff only
   after the read-only preflight passes.
3. Stop at `READY_FOR_GROUND`. Position both feet consistently, keep the
   gantry load-bearing, then type `load`.
4. Hold static PD for at least ten seconds. Record pelvis roll/pitch, maximum
   joint velocity, tracking error, foot placement, and qualitative gantry
   tension.
5. The diagnostic reproduction target is pitch `+5..+9 deg`, roll `-2..+2
   deg`, and maximum joint velocity below `0.1 rad/s`. Do not force the robot
   to reach it. This target matches one successful entry and is not a general
   safety envelope.
6. If the target is not met, re-tension, fully suspend, type `lifted`, and mark
   the run BLOCKED without requesting policy.
7. If the target is met, obtain a fresh operator confirmation before one
   policy pulse. Keep gantry tension and foot contact fixed. Type `stop` at the
   first growing oscillation, toe rise, noise, or unexpected motion.
8. After static return, fully suspend before typing `lifted`. Confirm official
   MC is again the sole publisher.

**Success criterion:** no growing pitch/roll or ankle oscillation, no toe rise,
fresh feedback, maximum measured joint speed below the 0.8 rad/s trip, and
clean return to static PD. A pass remains supported-only.

**Failure criterion:** any increasing low-frequency oscillation, relative or
absolute tilt return, joint-speed return, tracking/contact instability, stale
feedback, ownership anomaly, or emergency-stop use.

**Required record after run:** timestamped log path, pre-policy probe, exact
entry measurements, operator description of sling/feet, duration, peak tilt,
peak ankle speed/tracking error, return reason, final publisher counts, and
hardware state.

## Offline experiment E-SLEW-01

**State:** complete in MuJoCo; not a powered result.

The simulator reproduced static PD under virtual gantry support, a four-second
policy ramp, corrected applied-action feedback, `neutral_damped` gains/LPF and
an 80-second gradual support release. The original `0.12 rad/s` target slew
lost height before full transfer. Rates `0.30`, `0.40`, `0.60`, and disabled
all completed transfer near 1.5-2.1 degrees tilt; `0.30` then remained fully
unsupported for about 66.6 seconds before a later height loss. `0.20` remained
standing but settled near 4 degrees. Widening only the leg envelope from 0.60
to 1.20 rad worsened posture and is rejected.

**Superseded decision:** this section originally selected `0.30 rad/s` as the
next powered A/B. That run is now complete and failed; do not repeat the sweep.

## Offline experiment E-ANKLE-CHAIN-01

**State:** complete on 2026-08-30; fixed StandStill only.

The 300-second pass, corrected `0.12` failure, and `0.30` failure were replayed
through policy action, raw target, safety and supported ramps, envelope, LPF,
slew, final HAL target, measured ankle q/dq, tracking error, and pelvis
roll/pitch. Replayed ankle HAL targets match the logged targets within
`1e-6 rad` in every run.

| Run | Tilt max first -> last 1 s | Left tracking max first -> last 1 s | Right tracking max first -> last 1 s |
| --- | ---: | ---: | ---: |
| `170450` | 7.04 -> 4.46 deg | 0.159 -> 0.204 rad | 0.181 -> 0.167 rad |
| `210922` | 6.26 -> 11.29 deg | 0.279 -> 0.419 rad | 0.374 -> 0.540 rad |
| `222159` | 5.22 -> 10.38 deg | 0.296 -> 0.594 rad | 0.315 -> 0.673 rad |

**Attribution:** output limiting is proven for the 0.12 failure. The 0.30 run
reduced LPF-to-HAL backlog to about `0.04 rad`, but peak left/right tracking
error rose to `0.594/0.673 rad` and speed to `0.519/0.861 rad/s`. Therefore the
unresolved branch is realized PD/torque authority versus zero alignment versus
support/contact versus hardware protection, not target slew alone. The 0.30
entry also passed the temporary roll/pitch window, so that window is not a
sufficient predictor.

**Telemetry limitation:** historical logs do not contain measured
effort/current, motor voltage, per-tick gains, foot/gantry force, or protection
state. Implied PD torque is an estimate and must not be reported as measured.

**Artifacts:** `gear_sonic_deploy/analysis_reports/standstill_20260830/`.

**Gate:** reproduce the exact real-deploy `0.12 rad/s` ordering and effective
gains in simulation/replay, compare unloaded and loaded support without a slew
sweep, then add missing telemetry. A powered configuration must pass three
consecutive 30-second supported probes before one 300-second reproduction.

## 2026-08-31 13:48-14:01 CST: policy-off support adjustment blocked

**Type:** powered static-PD diagnostic. **Policy was never started.**

```text
remote log: runtime_suspended/logs/suspended_sonic_20260831_134829
mode: --supported-neutral-damped-relative-support-step
writer: 250 Hz
policy: OFF for the entire run
model: 8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
```

The first fresh state after `load` was pelvis pitch/roll/tilt
`-10.71/-0.78/10.74 deg`, ankle-pitch tracking maximum `0.101 rad`, waist
pitch target/measured/error `0.000/+0.322/-0.322 rad`, and maximum joint speed
`0.007 rad/s`. This already satisfied the actual relative policy-entry gate:
the reference delta was zero, ankle error was below `0.15 rad`, and state was
quiet. No waist threshold participates in that entry decision.

The operator was nevertheless asked to adjust support to reduce waist error
for the later support-step gate. That mixed two protocol phases. The waist
offset did not materially close, while pelvis tilt grew to `17.84 deg`; the
static guard latched `torso tilt exceeded baseline limit`. Static PD remained
active, feedback stayed fresh, and policy remained off.

The operator then supplied a side photograph of a visually upright pose. The
corresponding stable telemetry was tilt/pitch/roll `5.42/-5.08/-1.88 deg`,
ankle-pitch tracking `0.169 rad`, waist-pitch tracking `0.281 rad`, and maximum
speed `0.007 rad/s`. This is useful calibration evidence: visual uprightness
and low dynamics coexist with a waist error above `0.20 rad` because the
upper-body sling preloads the waist. The latched earlier fault correctly
prevented reuse of the same run.

After the operator fully suspended the robot, `lifted` stopped the custom
writer and the custom process check was empty. Cleanup reported that SoC0 MC
PID 2169 resumed, but that PID then disappeared and all four HAL command topics
reported zero publishers. The robot network disconnected before official MC
could be restarted. Recovery therefore remains pending and the robot must stay
suspended.

**Result:** `BLOCKED_POLICY_OFF_PROTOCOL_ERROR`, not a Sonic failure and not a
support-step powered probe. Preserve and copy the remote log when connectivity
returns.

## 2026-08-31 14:10-14:19 CST: sling-aware waist gate candidate

**Type:** local offline code/test only; no robot connection or HAL publisher.

The support-step waist tracking ceiling now supports a negative value meaning
"record but do not gate". Only the dedicated support-step mode sets `-1.0`.
Both ankle-pitch ceilings, pelvis tilt, velocity, tilt trend, five-second
continuity, marker, plateau, and bounded return remain unchanged. A unit test
proves a `0.420 rad` waist offset is accepted when the waist gate is disabled;
the M2 state still fails because its left ankle error is `0.494 rad`.

```text
local powered launcher sha256:
  f14d4054101e66a0de291223f04e9c4470e4d69063c3a6ee8a49200f69c8d982
local deploy source sha256:
  c567f104e49d6a040a7b1ba58308c1d4790ecb4e131a4b8c0ebd4b5a6d1f403a
local gate header sha256:
  8070bbc97f987c9ecc2a4a23bd6f0ef003d938e0ba5840c2a04ea7cc423a8d72
local gate test sha256:
  dd06bbb3f6b3ce459edd28b0aa65bd8ecf92c22fdd6b64e2ce75aece7ca66a02
offline build: /tmp/x2_support_waist_gate.ywGsSu
CTest: 2/2 passed
```

This is not deploy-ready. Full ARM64 build, exact binary hash, wrapper hash
update, process-level dry-run, official-MC ownership recovery, and fresh
powered authorization remain pending.

## 2026-08-31 14:20-14:28 CST: preloaded fixed-support 100-second wrapper

**Type:** local launcher preparation only; no robot connection or HAL writer.

The next physical experiment no longer changes the external support wrench
while policy is active. The operator establishes the intended foot contact and
sling tension before `load`; `load` captures that pose, and support remains
immutable through the complete 100-second policy interval. A lower support
level, if warranted, is a separate run with a separately captured entry.

The wrapper reuses the already-built M1 policy-off-debug candidate and frozen
model. The shared runner adds only a `100.0 s` duration branch to the existing
relative-entry profile. It does not pass `--support-step-probe` and changes no
gain, clamp, LPF, target slew, mapping, writer rate, pelvis reconstruction, or
policy ramp.

```text
wrapper:
  run_x2_supported_preloaded_fixed100.sh
confirm token:
  X2_SUPPORTED_PRELOADED_FIXED100_START
M1 binary expected sha256:
  3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f
model expected sha256:
  8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
powered launcher sha256:
  f14d4054101e66a0de291223f04e9c4470e4d69063c3a6ee8a49200f69c8d982
wrapper sha256:
  73bbd6122356d605802c1ac2e62467627b7bae26ea79c476906f7f1fda1513a1
shell syntax: PASS
```

The wrapper is not synchronized or authorized while the robot is offline.
Before any powered use: restore official MC, verify one official publisher on
all four command groups, compare remote hashes, and repeat the normal physical
suspension/emergency-stop preflight.

## 2026-08-31 14:29-14:50 CST: fixed-100 launcher offline hardening

**Type:** local shell/profile verification only; no robot connection, MC
mutation, process start, or HAL publisher.

The fixed-100 wrapper now pins `deploy_x2.sh` in addition to the M1 binary,
model, and powered runner. It clears the old writer-adoption/debug environment
overrides before launch and accepts `--verify-only`, which exits after artifact
hash checks with `FIXED100_ARTIFACTS_VERIFIED: no process started`.

An offline regression test executes the real shared runner against a
disposable fake `deploy_x2.sh` and compares the complete generated argv for
`--supported-neutral-damped-relative-100` and the frozen relative-300 parent.
After normalizing runtime paths, their only difference is policy duration
`100.0` versus `300.0` seconds. The fixed-100 argv retains 250 Hz writer,
8.0 Hz target LPF, `0.12 rad/s` target slew, and relative-to-load entry; it
contains neither `--support-step-probe` nor `--vla`.

```text
run_x2_supported_preloaded_fixed100.sh
  62cb405d2b6077b1a4ae06ced8c7f1e46a07eed85c74688dfe0b5cdf5c8cbc82
run_x2_suspended_sonic.sh
  f14d4054101e66a0de291223f04e9c4470e4d69063c3a6ee8a49200f69c8d982
deploy_x2.sh
  d16d52db6763edec280238bfad83b91f04b76f4fa7d66831307b5bd9ea9c63b4
scripts/test_fixed100_launcher.py
  dafc470ecd86c02570327475cbdf484947fa6b7be8eed8ddfc0bec39bb27da7b
bash -n: PASS
offline launcher profile test: FIXED100_LAUNCHER_TEST_PASS
offline CTest: 2/2 PASS
fresh offline build: /tmp/x2_fixed100_offline.JGDrD6
```

This does not build ARM64 code and does not authorize a powered run. The next
online operation remains official-MC ownership recovery and log collection,
followed by remote artifact verification before any writer can start.
