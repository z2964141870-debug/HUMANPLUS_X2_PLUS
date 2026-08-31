# Powered StandStill Telemetry Probe

Source: `analysis_logs/suspended_sonic_20260830_152344_powered1`

## Outcome

- Aligned policy data: `188` samples / `3.740 s`.
- Pelvis roll/pitch/tilt: `-1.29/+8.00/8.10 deg` -> `-6.21/+11.76/13.28 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.625 -> +0.625 | +0.687 -> +0.670 | -0.018 / 0.052 / -0.046 | 0.177 | 4.48 Nm | 4.57 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | -0.001 -> -0.001 | -0.659 -> -0.581 | +0.296 / 0.582 / +0.581 | 0.128 | 18.37 Nm | 18.40 Nm | 32.064 / 3.003 |
| left_ankle_roll_joint | +0.235 -> +0.235 | +0.003 -> +0.068 | -0.003 / 0.177 / +0.167 | 0.128 | 5.48 Nm | 5.50 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.859 -> +0.859 | +0.695 -> +0.871 | -0.026 / 0.027 / -0.012 | 0.214 | 2.67 Nm | 2.63 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | +0.014 -> +0.014 | -0.659 -> -0.421 | +0.296 / 0.542 / +0.435 | 0.531 | 16.91 Nm | 17.11 Nm | 32.064 / 3.003 |
| right_ankle_roll_joint | -0.165 -> -0.165 | -0.005 -> -0.108 | +0.005 / 0.101 / -0.057 | 0.153 | 2.96 Nm | 3.09 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.194 -> -0.194 | +0.163 -> +0.105 | -0.163 / 0.343 / -0.298 | 0.201 | 4.78 Nm | 4.84 Nm | 14.251 / 2.722 |
| waist_roll_joint | +0.121 -> +0.121 | +0.021 -> +0.078 | -0.021 / 0.052 / +0.043 | 0.163 | 0.80 Nm | 0.64 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `left_ankle_pitch_joint`: `0.582 rad`.
- `right_ankle_pitch_joint`: `0.542 rad`.
- `waist_pitch_joint`: `0.343 rad`.
- `right_shoulder_pitch_joint`: `0.246 rad`.
- `left_shoulder_pitch_joint`: `0.245 rad`.

## Largest measured speeds

- `right_ankle_pitch_joint`: `0.531 rad/s`.
- `right_shoulder_yaw_joint`: `0.275 rad/s`.
- `right_elbow_joint`: `0.275 rad/s`.
- `right_shoulder_pitch_joint`: `0.250 rad/s`.
- `right_knee_joint`: `0.214 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
