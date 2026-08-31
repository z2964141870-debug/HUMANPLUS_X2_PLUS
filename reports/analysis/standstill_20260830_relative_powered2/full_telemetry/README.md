# Powered StandStill Telemetry Probe

Source: `analysis_logs/suspended_sonic_20260830_165839_powered2`

## Outcome

- Aligned policy data: `1500` samples / `29.979 s`.
- Pelvis roll/pitch/tilt: `-0.52/-9.81/9.82 deg` -> `+1.89/-0.32/1.92 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.658 -> +0.658 | +0.650 -> +0.659 | +0.019 / 0.039 / -0.001 | 0.189 | 3.60 Nm | 3.40 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.123 -> +0.123 | -0.469 -> -0.346 | +0.106 / 0.520 / +0.469 | 0.140 | 16.38 Nm | 16.33 Nm | 32.064 / 3.003 |
| left_ankle_roll_joint | +0.170 -> +0.170 | -0.016 -> +0.038 | +0.016 / 0.181 / +0.132 | 0.092 | 5.83 Nm | 5.75 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.487 -> +0.487 | +0.668 -> +0.488 | +0.001 / 0.048 / -0.001 | 0.116 | 4.19 Nm | 4.47 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | -0.098 -> -0.098 | -0.478 -> -0.424 | +0.115 / 0.536 / +0.325 | 0.067 | 17.03 Nm | 17.15 Nm | 32.064 / 3.003 |
| right_ankle_roll_joint | -0.037 -> -0.037 | -0.006 -> -0.049 | +0.006 / 0.029 / +0.012 | 0.165 | 0.79 Nm | 0.72 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.487 -> -0.200 | +0.319 -> +0.238 | -0.319 / 0.440 / -0.438 | 0.150 | 6.45 Nm | 6.30 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.242 -> -0.200 | +0.012 -> -0.016 | -0.012 / 0.207 / -0.184 | 0.053 | 2.86 Nm | 2.96 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `right_ankle_pitch_joint`: `0.536 rad`.
- `left_ankle_pitch_joint`: `0.520 rad`.
- `waist_pitch_joint`: `0.440 rad`.
- `left_shoulder_pitch_joint`: `0.247 rad`.
- `right_shoulder_pitch_joint`: `0.243 rad`.

## Largest measured speeds

- `right_elbow_joint`: `0.275 rad/s`.
- `right_shoulder_pitch_joint`: `0.238 rad/s`.
- `left_shoulder_pitch_joint`: `0.226 rad/s`.
- `left_elbow_joint`: `0.201 rad/s`.
- `left_knee_joint`: `0.189 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
