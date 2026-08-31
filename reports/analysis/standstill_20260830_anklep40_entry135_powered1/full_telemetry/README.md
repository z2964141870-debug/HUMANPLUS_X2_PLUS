# Powered StandStill Telemetry Probe

Source: `analysis_logs/suspended_sonic_20260830_194201_entry135_complete`

## Outcome

- Aligned policy data: `1500` samples / `29.979 s`.
- Pelvis roll/pitch/tilt: `-0.40/-11.29/11.30 deg` -> `+4.12/-0.10/4.12 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.725 -> +0.677 | +0.656 -> +0.677 | +0.013 / 0.060 / +0.001 | 0.226 | 5.42 Nm | 5.35 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.103 -> +0.095 | -0.455 -> -0.308 | +0.092 / 0.437 / +0.404 | 0.214 | 17.08 Nm | 17.18 Nm | 40.000 / 3.003 |
| left_ankle_roll_joint | +0.108 -> +0.110 | -0.000 -> +0.005 | +0.000 / 0.175 / +0.104 | 0.201 | 5.66 Nm | 5.58 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.462 -> +0.463 | +0.665 -> +0.473 | +0.004 / 0.056 / -0.010 | 0.189 | 5.36 Nm | 5.53 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | +0.151 -> -0.179 | -0.449 -> -0.418 | +0.086 / 0.394 / +0.239 | 0.165 | 15.50 Nm | 15.69 Nm | 40.000 / 3.003 |
| right_ankle_roll_joint | +0.038 -> +0.024 | -0.010 -> +0.005 | +0.010 / 0.080 / +0.019 | 0.226 | 2.67 Nm | 2.54 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.390 -> -0.200 | +0.325 -> +0.218 | -0.325 / 0.475 / -0.418 | 0.192 | 6.81 Nm | 6.96 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.284 -> -0.200 | +0.006 -> -0.046 | -0.006 / 0.207 / -0.154 | 0.243 | 2.87 Nm | 2.96 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `waist_pitch_joint`: `0.475 rad`.
- `left_ankle_pitch_joint`: `0.437 rad`.
- `right_ankle_pitch_joint`: `0.394 rad`.
- `right_shoulder_pitch_joint`: `0.257 rad`.
- `left_hip_roll_joint`: `0.253 rad`.

## Largest measured speeds

- `right_elbow_joint`: `0.360 rad/s`.
- `left_shoulder_pitch_joint`: `0.263 rad/s`.
- `waist_roll_joint`: `0.243 rad/s`.
- `left_elbow_joint`: `0.238 rad/s`.
- `left_knee_joint`: `0.226 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
