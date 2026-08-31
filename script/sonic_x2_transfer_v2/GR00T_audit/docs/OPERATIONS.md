# X2 Sonic Operations

This is a real-hardware runbook for the current v0.9 supported-neutral stage.
It is not an unsupported-standing or walking procedure.

## Current hard stop

The robot is currently offline after the custom writer ended normally but the
resumed official MC worker disappeared. Keep it fully suspended. On reconnect,
the first operation is read-only process/topic inspection followed, if still
needed, by the documented official `stop-app mc` / `start-app mc` recovery.
No launcher is allowed until every command topic has exactly one official
`mc_ros2_node` publisher.

## Roles and stop paths

- One person remains at the robot with the physical emergency stop in hand.
- One operator controls the SSH terminal and reads every transition aloud.
- The robot begins fully suspended and the gantry remains attached throughout.
- `stop` requests a two-second policy-to-static-PD return.
- `lifted` is valid only after policy is off and the robot is fully suspended;
  it ends the custom writer and restores official MC.
- Use the physical emergency stop immediately for dangerous motion. Software
  gates are not a substitute for it.

Do not close SSH, use `Ctrl-C`, or type `lifted` while the robot is partly
loaded. If Wi-Fi is lost, the person at the robot keeps/supports the robot and
uses the emergency stop if motion is unsafe.

## Read-only preflight

Connect to SoC1 and source the correct runtime:

```bash
ssh agi@192.168.43.21
source /opt/ros/humble/setup.bash
source ~/aimdk_ws_0_8_18/install/setup.bash
cd /agibot/data/home/agi/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy
```

Confirm no custom controller is running:

```bash
pgrep -af 'x2_deploy_onnx_ref|run_x2_supported|run_x2_suspended|deploy_x2.sh'
```

The expected output is empty. Then confirm each command topic has exactly one
official publisher:

```bash
for group in leg waist arm head; do
  printf '%s ' "$group"
  timeout 3 ros2 topic info "/aima/hal/joint/$group/command" 2>/dev/null \
    | awk '/Publisher count:/ {print $3}'
done
```

Confirm all state groups and torso IMU have publishers, then record the pelvis
probe. Always use the system Python:

```bash
/usr/bin/python3 -u scripts/x2_pelvis_imu_probe.py --seconds 3
```

Abort before handoff if a custom process exists, a command publisher count is
not exactly one, a state stream is missing, hashes differ from the manifest,
or the physical suspension/emergency-stop conditions are not current.

## Allowed launchers

| Launcher | Current use |
| --- | --- |
| `run_x2_supported_preloaded_fixed100.sh` | next bounded candidate only after recovery and `--verify-only` passes |
| `run_x2_ground_load_hold.sh` | static-PD diagnostics only; policy hard-disabled |
| `run_x2_supported_neutral_damped_support_step.sh` | superseded experimental history; do not run |
| all other `run_x2_*.sh` | frozen experimental history; do not run |

`run_x2_suspended_sonic.sh` is internal. Do not invoke it directly. The
fixed-100 wrapper pins the writer to 250 Hz and clears stale adoption/debug
environment overrides.

## Static entry acquisition

After recovery, synchronization, and hash comparison, verify the complete
fixed-100 artifact chain without starting a process:

```bash
./run_x2_supported_preloaded_fixed100.sh --verify-only
```

It must print `FIXED100_ARTIFACTS_VERIFIED: no process started`. Then supply
the SoC0 password interactively without placing it in a script or document and
launch only after a fresh physical suspension/emergency-stop confirmation:

```bash
read -rsp 'SoC0 password: ' X2_SOC0_PASSWORD
echo
export X2_SOC0_PASSWORD
./run_x2_supported_preloaded_fixed100.sh X2_SUPPORTED_PRELOADED_FIXED100_START
```

At the launcher's safety prompt, re-confirm full suspension and the physical
emergency-stop operator before answering `y`. The script must prove MC command
silence before enabling the custom writer. It captures the current pose,
ramps PD, moves to the trained default pose, and stops at
`READY_FOR_GROUND` with policy off.

While policy is off, establish the final feet and sling setting intended for
the entire run. Do not chase the sling-induced waist offset. Then type:

```text
load
```

Hold at least ten seconds. Record tilt, maximum velocity, ankle-pitch tracking
error, freshness, foot contact, and qualitative sling tension. The relative
entry gate is captured from this loaded pose. Waist pitch remains telemetry;
it is not an absolute entry target. Do not alter feet or sling after `load`.

## Bounded policy pulse

Only after the relative entry gate reports ready and the physical conditions
remain unchanged:

```text
policy
```

Keep support and feet unchanged for the complete 100-second interval. Do not
send `support_step`. Watch the 2 Hz status lines for tilt,
pitch/roll, maximum joint velocity, tracking error, and ankle target/position.
At the first growing oscillation, toe rise, unexpected sound, or operator
concern, type:

```text
stop
```

Wait until `GROUND_LOAD_HOLD` confirms policy is off and static PD is stable.
Re-tension and fully lift the robot. Only then type:

```text
lifted
```

The launcher must resume official MC. Re-run publisher counts and verify no
custom process remains.

## Recovery rules

- If policy trips an envelope, let the two-second return complete unless the
  physical emergency stop is needed.
- If static PD reports a ground-load fault, re-tension the gantry; do not
  request policy. Fully suspend before `lifted`.
- If the launcher exits unexpectedly, inspect the exact PID/process state and
  publisher counts before starting anything. Do not kill unknown processes.
- If official MC does not return, keep the robot suspended and do read-only
  diagnosis. Do not launch another writer.
- Preserve the timestamped runtime log and append the outcome to
  `docs/EXPERIMENTS.md` before another powered attempt.
