# X2 support-step process dry-run

Date: 2026-08-31 13:32-13:40 CST

This directory contains the 16 aligned CSV outputs from the final process-level
dry-run of the isolated support-step candidate. The robot remained under
official MC and `AimdkIo` explicitly disabled every HAL command publisher.

## Candidate identity

```text
workspace: runtime_suspended/ws_support_step_20260831_v3
binary: 9d4916a8e5fd64e4ac3ddb0f0a32903ccc90b9e86b4e146cc6cabefe63695483
source: d13205445b1a04d25745aad2d3c8e564ff9ecda9acc0a85f59041f2990679722
support-step wrapper: 19148f68824f608a91e05ba181ce96cffb7a7487644594987ba5a98f9616667c
model: 8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
```

## Result

- `tick.csv`: 449 rows spanning 8.979633 seconds.
- Every row has `dry_run=1`.
- `suspended_reference_pose_ramp`: 51 rows.
- `ground_load_static`: 37 rows.
- `supported_policy_probe`: 349 rows.
- `supported_policy_support_step_marker`: exactly 1 row.
- `supported_policy_return`: 11 rows.
- The marker was accepted at policy time 5.000 seconds.
- The accelerated 2-second plateau completed and returned to
  `GROUND_LOAD_HOLD` before `lifted` exited the process.
- CTest passed `2/2`; afterward no custom process remained and every command
  topic still had one official `mc_ros2_node` publisher.

The test deliberately shortened orchestration timings and relaxed measured
gates. It proves state transitions, one-shot marker consumption, and cleanup;
it does not prove powered support transfer or balance.
