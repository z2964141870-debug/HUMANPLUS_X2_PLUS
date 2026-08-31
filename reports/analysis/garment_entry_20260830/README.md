# X2 Garment Entry Analysis

Offline only; no ROS, ZMQ, MC, or HAL process was started.

## Source

- Frames: `1379` over `39.355 s`
- Rate: `35.01 Hz`
- Gap p95/max: `30.5/46.2 ms`
- Timestamp derivative resets: `0`

## Candidate Entry

- Selection: last HMCP frame at or before deploy CONTROL entry
- HMCP sequence: `225`
- Reference root roll/pitch/yaw: `+1.80/-5.46/-83.29 deg`
- Robot yaw from deploy log: `+32.62 deg`
- Uncorrected reference-minus-robot yaw: `-115.91 deg`
- Largest StandStill joint offsets: left_shoulder_roll_joint=+1.075, left_shoulder_pitch_joint=-0.503, left_knee_joint=-0.447, right_knee_joint=-0.434, right_elbow_joint=+0.393, right_ankle_pitch_joint=+0.338
- Largest instantaneous reference velocities: left_elbow_joint=+0.345, left_shoulder_roll_joint=+0.333, left_hip_yaw_joint=-0.272, left_shoulder_pitch_joint=-0.259, left_shoulder_yaw_joint=-0.258, right_elbow_joint=-0.201

## Pre-entry Window (1.968 s / 69 frames)

| Group | Offset p95/max rad | Velocity p95/max rad/s | Range max rad | Largest-range joint |
| --- | ---: | ---: | ---: | --- |
| leg | 0.430/0.447 | 0.171/0.369 | 0.159 | right_hip_yaw_joint |
| waist | 0.108/0.111 | 0.161/0.224 | 0.111 | waist_roll_joint |
| arm | 0.495/1.075 | 1.189/3.189 | 1.078 | left_shoulder_roll_joint |
| head | 0.000/0.000 | 0.000/0.000 | 0.000 | head_yaw_joint |

Root roll/pitch/yaw ranges in the window: `1.83/1.98/8.79 deg`.

## Lowest-motion Observed Window

Sequences `52..120` over `1.988 s` were the lowest-motion complete window in this capture. This is a ranking result, not a powered-safe label.

| Group | Offset p95/max rad | Velocity p95/max rad/s | Range max rad |
| --- | ---: | ---: | ---: |
| leg | 0.440/0.443 | 0.098/0.110 | 0.009 |
| waist | 0.055/0.055 | 0.066/0.074 | 0.006 |
| arm | 1.008/1.016 | 0.194/0.235 | 0.011 |
| head | 0.000/0.000 | 0.000/0.000 | 0.000 |

Lowest-motion root roll/pitch/yaw ranges: `0.31/0.08/0.47 deg`.

Largest StandStill offsets at the end of that window: left_shoulder_roll_joint=+1.007, right_shoulder_roll_joint=-0.968, left_elbow_joint=+0.555, right_elbow_joint=+0.524, right_knee_joint=-0.439, left_knee_joint=-0.433

## Lowest-motion Neutral-envelope Window

Sequences `313..383` over `1.990 s` were the lowest-motion window that also rejected the calibration T-Pose.

| Group | Offset p95/max rad | Velocity p95/max rad/s | Range max rad |
| --- | ---: | ---: | ---: |
| leg | 0.436/0.450 | 0.448/1.246 | 0.256 |
| waist | 0.133/0.150 | 0.315/0.623 | 0.142 |
| arm | 0.359/0.381 | 0.372/1.009 | 0.263 |
| head | 0.000/0.000 | 0.000/0.000 | 0.000 |

Neutral-window root roll/pitch/yaw ranges: `3.65/2.84/79.61 deg`.

Largest StandStill offsets at the end of that window: right_knee_joint=-0.420, left_ankle_pitch_joint=+0.386, right_shoulder_yaw_joint=+0.342, right_elbow_joint=+0.333, right_ankle_pitch_joint=+0.330, left_knee_joint=-0.322

## Interpretation

- The live root yaw is not in the robot entry frame. A single captured yaw rebase is required before powered use.
- Entry must be gated on a continuous stationary-reference window; frame freshness alone is insufficient.
- StandStill-to-live must use a bounded blend. The report's induced blend velocity is joint offset divided by the configured blend duration.
- Dry-run robot joints remained under official MC; action/target statistics are interface evidence, not powered tracking evidence.

## Dry-run Policy Evidence

- Logged action max/p99: `20.000/15.041`
- Ticks at the logged `20` action clip: `38`
- First-5-second target delta max: `3.162 rad`
