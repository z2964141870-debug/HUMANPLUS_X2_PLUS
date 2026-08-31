# X2 P0 passive logging result

- Date: 2026-08-14 13:33 CST
- Entry host: PC2 / SoC1 (`192.168.43.21`, internal `10.0.1.41`)
- Robot mode before recording: `PASSIVE_DEFAULT`
- Current input source before recording: empty, priority 0, timeout 0
- Safety boundary: read-only topic/service inspection and rosbag recording only; no publish, mode switch, controller start, or motion command
- Duration: 14.382 s
- Bag size: about 6.0 MiB

## Recorded data

| Topic | Messages | Measured rate |
|---|---:|---:|
| `/aima/mc/joint/retargeting` | 719 | 49.936 Hz |
| `/aima/mc/retarget_frame` | 720 | 50.000 Hz |
| `/aima/hal/imu/torso/state` | 7191 | 500.047 Hz |
| `/aima/hal/imu/chest/state` | 7192 | 500.043 Hz |

`/aima/hal/odom/state` produced no messages while the robot was passive and therefore was not written into the bag.

## Structural checks

- Every `/aima/mc/joint/retargeting` message has 62 values.
- Every `/aima/mc/retarget_frame` message has 93 values.
- Across 719 nearest-timestamp pairs, `retargeting[7:62]` matches `retarget_frame[38:93]`:
  - mean absolute difference: `2e-9`
  - maximum absolute difference: `4.8e-8`
  - mean nearest timestamp separation: `4.9667 ms`
- The 62-value command was constant for the whole passive recording. This proves the bridge was publishing a stationary default target; it does **not** prove closed-loop joint tracking.
- The semantics of `retarget_frame[0:38]` remain undocumented and are intentionally not assigned by this analysis.

## Passive IMU baseline

| Sensor | Quaternion norm mean | Angular-speed norm mean | Acceleration norm mean |
|---|---:|---:|---:|
| torso | 1.00000000 | 0.003252 rad/s | 9.798364 m/s² |
| chest | 1.00000002 | 0.003193 rad/s | 9.826896 m/s² |

The sensor streams and quaternion normalization look healthy for a stationary baseline. The torso/chest gravity magnitudes differ by about `0.0285 m/s²`; retain this as a calibration/bias baseline rather than correcting it from one short sample.

## Result

P0 passes: cross-board ROS data can be recorded centrally from PC2, the two retargeting streams are structurally consistent, and both IMUs provide stable high-rate data. P0 does not evaluate balance or tracking performance because the robot remained in passive mode.

The next meaningful stage is P1: supported static stand, no mocap motion, recording joint feedback plus the same topics. P1 is an active robot test and requires an on-site operator, clear space/support, and an explicit go-ahead.
