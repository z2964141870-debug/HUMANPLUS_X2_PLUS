# X2 Sonic Current State

Updated: 2026-08-31 14:50 CST

## 2026-08-31 superseding state

This section supersedes the older live snapshots and next-action text below.
The older sections remain as historical context only.

### 14:01 policy-off support-geometry run and offline correction

- Powered log `runtime_suspended/logs/suspended_sonic_20260831_134829`
  never entered policy. It remained in `GROUND_LOAD_HOLD` with the 250 Hz
  static-PD writer and therefore does not evaluate Sonic balance.
- The first loaded state was already eligible for the actual relative policy
  entry gate: pelvis pitch/roll/tilt `-10.71/-0.78/10.74 deg`, ankle-pitch
  tracking maximum `0.101 rad`, and maximum speed `0.007 rad/s`.
- The operator was incorrectly asked to reduce the `0.322 rad` static waist
  error before policy. That `0.20 rad` threshold belongs to the later,
  policy-on support-step gate, not the policy-entry gate. Sling adjustment
  moved pelvis tilt to `17.84 deg` and latched the static ground-load fault.
- A subsequent visually upright pose was objectively quiet and much closer to
  level: tilt/pitch/roll `5.42/-5.08/-1.88 deg`, ankle-pitch tracking
  `0.169 rad`, waist-pitch tracking `0.281 rad`, and maximum speed
  `0.007 rad/s`. The prior fault remained latched as designed.
- Historical evidence already shows that upper-torso suspension folds the
  passive waist to about `+0.349 rad`; the supported 300-second run entered at
  `+0.332 rad`. Requiring absolute waist error `<=0.20 rad` is therefore not
  a physically reproducible support-transfer condition for this sling.
- After full suspension, `lifted` ended the custom writer normally and the
  custom deploy/launcher process check was empty. The resumed SoC0 MC PID then
  disappeared and all four command topics showed zero publishers. The robot
  network disconnected before the documented official `stop-app mc` /
  `start-app mc` recovery could run. Keep the robot suspended; official MC
  ownership is **not yet verified**.
- A local-only candidate now permits a negative waist support-step threshold
  to mean telemetry-only. The support-step launcher uses that mode while
  retaining both ankle `<=0.20 rad`, pelvis tilt `<=3 deg`, maximum speed
  `<=0.05 rad/s`, one-second tilt rise `<=0.10 deg`, and five continuous
  stable seconds. Offline CMake build and CTest pass `2/2`.
- This candidate has not been ARM64-built, synchronized, hash-frozen, or run
  on the robot. Its current wrapper intentionally still expects the previous
  hashes and cannot authorize a powered run.
- The next recommended powered route no longer changes sling tension during
  policy. A local `run_x2_supported_preloaded_fixed100.sh` wrapper pins the
  existing M1 binary, establishes final support before `load`, and runs a
  fixed-support 100-second relative-entry probe. It is local-only until the
  robot reconnects and official MC ownership is restored. Its offline-hardened
  SHA-256 is `62cb405d2b6077b1a4ae06ced8c7f1e46a07eed85c74688dfe0b5cdf5c8cbc82`;
  `--verify-only` checks all artifacts without starting a process.

### Earlier 2026-08-31 milestones (historical, not current robot state)

- The robot has rebooted and is online under official MC. No custom controller,
  garment, proxy, or offload process is running. Leg, waist, arm, and head
  command topics each have exactly one publisher, `mc_ros2_node`. Physical
  suspension and emergency-stop readiness have not yet been re-confirmed for a
  powered run.
- The onboard garment/offload dry-run has passed at `35.02 Hz`, but powered
  garment arming remains blocked.
- M1 policy-off `x2_debug` is complete. The exact candidate binary is
  `3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f`.
- The 2026-08-31 M2 300-second attempt is INVALID as fixed-support acceptance
  evidence because the operator loosened the sling while policy was on. The
  controller had been bounded for about 174 seconds with fixed heavy support.
- The intervention is still useful evidence: before release, left ankle-pitch
  and waist-pitch tracking errors were about `0.49` and `0.42 rad`; after the
  support wrench changed, the robot tipped forward and the joint-speed guard
  returned policy after `178.373 s`. This configuration did not demonstrate
  autonomous load-bearing balance.
- No fixed-parent manifest is approved. Do not start the garment source,
  proxy, or a powered live-reference run.
- A support-transfer orchestration candidate is staged in the isolated SoC1
  workspace `runtime_suspended/ws_support_step_20260831_v3`. It preserves the
  frozen `neutral_damped` controller and adds only a measured request gate, one
  event marker, one fixed observation plateau, and automatic return to static
  PD. Candidate binary SHA-256 is
  `9d4916a8e5fd64e4ac3ddb0f0a32903ccc90b9e86b4e146cc6cabefe63695483`.
- SoC1 CTest passes `2/2`. The completed process-level dry-run contained 449
  rows with `dry_run=1`, exactly one
  `supported_policy_support_step_marker`, a 2-second accelerated plateau, and
  a normal return to `GROUND_LOAD_HOLD`. It created no HAL command publisher;
  final ownership remained one official `mc_ros2_node` publisher per group.
- A post-test read-only preflight found one publisher on all four joint-state
  topics and the torso IMU topic. The three-second pelvis probe received 4004
  samples and reported reconstructed pelvis roll/pitch `-1.26/-12.78 deg`.
  This is an official-MC/harness snapshot, not a policy-entry target.

The support-step candidate above was the planned action at that historical
checkpoint. D-017 supersedes it: changing support during policy is no longer
the next experiment.

## Historical status snapshot (2026-08-29)

At that historical snapshot the robot was under official control. No garment,
HMCP/ZMQ adapter, Sonic, or custom HAL/PD process was running; ports `51234`
and `5556` were free. This paragraph does not describe the current disconnected
state above.

The complete live garment-to-Sonic compute path has passed twice on SoC1 in
read-only dry-run mode. The donor-compatible run delivered `13.85 Hz`; native
GMR plus optimized CPU Fast-SMPL raised this to `23.04 Hz`. Sonic itself stayed
at 50 Hz and created no HAL publisher in both runs. The current launcher adds
a numerically identical one-thread planner setting and matches the proven
MuJoCo GMR iteration count; it awaits one live throughput run. Powered garment
tracking remains closed.

## Goal and pipeline

Target pipeline:

```text
V2 garment -> X2 AX210 BLE -> onboard TIC/LFP -> Fast SMPL -> GMR G1 qpos36
-> localhost HMCP -> G1-to-X2 LiveMotion -> localhost ZMQ v5
-> Sonic 50 Hz -> safety gates -> HAL writer 250 Hz -> the same X2
```

The 5060 laptop at `100.101.30.68` is the proven shadow/source environment,
not part of the final runtime topology. The Mac is an SSH/log console only.
See `gear_sonic_deploy/README_X2_GARMENT_LIVE.md` for the frozen contract and
acceptance gates.

## Frozen software identity

| Item | Value |
| --- | --- |
| Local repository | `/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit` |
| Local branch/commit | `main` / `70bed4539efae431621013b3df71e6f81df1ab1c` |
| SoC1 deployment | `/agibot/data/home/agi/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy` |
| Firmware | `test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7` |
| ROS/message ABI | ROS Humble / `aimdk_msgs 0.8.18` |
| Model | `x2_sonic_frozen_g1core_lora_v2.onnx` |
| Model SHA-256 | `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9` |
| Corrected dry-run deploy SHA-256 | `64d2663431f7531ce907bc006f6979ed34d05528ee335c3b71e8f4a48cc98bf6` |
| Optimized dry-run launcher SHA-256 | `7d84009d0a73258a5e3fb130b20f76b6b9211520a17b5ee129def27a57bbb5a2` |
| Policy rate / writer rate | 50 Hz / 250 Hz |
| IMU mode | reconstructed pelvis from torso IMU plus 3-DOF waist |
| Active preset | `neutral_damped` |

The local launcher, source, model, motion file, and compiled SoC1 runtime were
hash-checked before this status was written. See `docs/BASELINE_170450.md` for
the full manifest.

## Live snapshot

At 2026-08-29 20:10 CST:

- SoC1 hostname is `agi`; it is NVIDIA Orin/aarch64.
- AX210 Wi-Fi and Bluetooth are present and unblocked.
- No garment/TIC/LFP/GMR/HMCP/ZMQ pose process is running; ports 51234 and 5556
  are unused.
- The existing target
  `/agibot/data/home/agi/miniconda3/envs/teleop` Python 3.10 environment has
  NumPy 2.2.6 and pyzmq 27.2.0 and passes the garment adapter tests.
- The 19:27 donor-path live dry-run accepted `595/595` HMCP frames at
  `13.85 Hz` and completed 1601 C++ ticks without a custom HAL publisher.
- The 20:01 optimized live dry-run accepted `943/943` HMCP frames at
  `23.04 Hz`; p95/max gaps were `50.2/78.0 ms`, and the C++ side again
  completed 1601 ticks without a custom HAL publisher.
- The isolated optimized runtime is
  `~/projects/smartwear_v2/runtime_g1_gmr_fast_20260829`. Exact import checks
  load its ARM64 `libgmr_native.so` and optimized `general_motion_retargeting`.
- The formal launcher now defaults to CPU Fast-SMPL, native GMR, two GMR
  iterations, and one PyTorch CPU thread. The exact 23.04 Hz parent and original
  baseline launcher are retained next to it.

This is a stable static diagnostic, not a garment or policy result.

## What is established

- SoC0 MC pause, command-silence proof, direct HAL ownership, custom PD,
  feedback continuity, cleanup, and MC restoration can work on v0.9.
- At 250 Hz the HAL state chain stays fresh while the custom writer owns the
  command topics.
- The `neutral_damped` profile completed 299.968 seconds in one supported
  entry and is the only non-divergent parent configuration.
- The same profile is not reproducible under arbitrary suspended/contact
  entry conditions.
- The most recent failure returned to static PD over two seconds and official
  MC was restored. There was no feedback or ownership fault.
- The real V2 jacket/pants path through TIC/LFP, Fast SMPL, native GMR,
  `qpos36`, G1-to-X2 mapping, `LiveMotion`, Sonic, and official X2 MuJoCo ran at
  about 33-35 Hz on the 5060 laptop shadow environment with `pose-scale=0.7`.
- The real HMCP-to-ZMQ v5.1 adapter is now implemented and staged on SoC1. Its
  strict parsing, named-joint mapping, WXYZ-to-XYZW conversion, fixed-50-Hz
  velocity contract, live-edge future window, capture/replay, and wire format
  pass on both Mac and SoC1. A 100-frame source-level comparison against the
  retained HumanPlus mapping was within `7.1e-08` maximum absolute error.
- The final onboard garment/offload dry-run passed at `35.02 Hz` with timestamp
  velocity on every HMCP frame. Deliberate 3588S loss stopped HMCP and drove
  the watchdog to `SAFE_IDLE` without local fallback.
- The support-step process state machine has a passing one-marker dry-run and
  a hash-gated isolated candidate; it has not yet been powered.

## What is not established

- A universal safe pelvis roll/pitch entry range.
- Whether posture, gantry force, foot contact geometry, or their interaction
  caused the different outcomes.
- Unsupported balance, push recovery, stepping, walking, or garment control.
- Correctness of `waist2`, 500 Hz, or v1.0 MC.
- A 100-second fixed-support run at a support level selected before `load`.
- A same-binary 300-second final-parent acceptance after the fixed-support
  sequence is understood.
- Powered garment tracking.

## Next bounded action

While the robot is offline, do not attempt network recovery or any powered
action. After connectivity returns, in order:

1. Keep the robot fully suspended and restore official MC using the documented
   `stop-app mc` / `start-app mc` flow.
2. Prove no custom process exists and all four command topics have exactly one
   `mc_ros2_node` publisher; copy the `134829` log.
3. Synchronize the fixed-100 runner/wrapper only after source/destination hash
   comparison, then run the wrapper with `--verify-only`.
4. Obtain a fresh suspension/emergency-stop confirmation before one fixed-
   support 100-second run. Establish support before `load`; do not move feet or
   sling after `load`.

Do not start garment, offload, proxy, support-step mode, or a second HAL
writer.
