# Powered StandStill Telemetry Probe

Source: `analysis_logs/suspended_sonic_20260830_172543_powered4`

## Outcome

- Aligned policy data: `440` samples / `8.780 s`.
- Pelvis roll/pitch/tilt: `-1.30/-6.83/6.95 deg` -> `+9.29/+7.73/12.06 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.807 -> +0.807 | +0.663 -> +0.851 | +0.006 / 0.044 / -0.044 | 0.226 | 5.60 Nm | 5.22 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.099 -> +0.099 | -0.507 -> -0.206 | +0.144 / 0.583 / +0.305 | 0.201 | 18.43 Nm | 18.39 Nm | 32.064 / 3.003 |
| left_ankle_roll_joint | +0.092 -> +0.092 | +0.002 -> +0.053 | -0.002 / 0.192 / +0.039 | 0.079 | 6.07 Nm | 6.04 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.533 -> +0.533 | +0.663 -> +0.489 | +0.006 / 0.060 / +0.044 | 0.177 | 5.83 Nm | 5.51 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | +0.167 -> +0.167 | -0.497 -> -0.406 | +0.134 / 0.576 / +0.573 | 0.153 | 18.02 Nm | 18.20 Nm | 32.064 / 3.003 |
| right_ankle_roll_joint | +0.089 -> +0.089 | -0.001 -> +0.037 | +0.001 / 0.054 / +0.052 | 0.177 | 1.55 Nm | 1.49 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.200 -> -0.200 | +0.296 -> +0.161 | -0.296 / 0.441 / -0.361 | 0.161 | 6.30 Nm | 6.36 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.198 -> -0.198 | +0.017 -> -0.119 | -0.017 / 0.211 / -0.079 | 0.121 | 3.04 Nm | 3.00 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `left_ankle_pitch_joint`: `0.583 rad`.
- `right_ankle_pitch_joint`: `0.576 rad`.
- `waist_pitch_joint`: `0.441 rad`.
- `left_shoulder_pitch_joint`: `0.258 rad`.
- `right_shoulder_pitch_joint`: `0.255 rad`.

## Largest measured speeds

- `right_elbow_joint`: `0.311 rad/s`.
- `left_elbow_joint`: `0.238 rad/s`.
- `left_knee_joint`: `0.226 rad/s`.
- `left_ankle_pitch_joint`: `0.201 rad/s`.
- `waist_yaw_joint`: `0.201 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
