# Powered StandStill Telemetry Probe

Source: `GR00T_audit/gear_sonic_deploy/analysis_logs/suspended_sonic_20260830_182345_anklep40`

## Outcome

- Aligned policy data: `1500` samples / `29.979 s`.
- Pelvis roll/pitch/tilt: `+0.39/-8.88/8.89 deg` -> `+0.03/-0.52/0.52 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.591 -> +0.591 | +0.685 -> +0.595 | -0.016 / 0.069 / -0.004 | 0.153 | 6.71 Nm | 6.49 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | -0.009 -> -0.010 | -0.485 -> -0.422 | +0.122 / 0.554 / +0.413 | 0.067 | 22.07 Nm | 22.08 Nm | 40.000 / 3.003 |
| left_ankle_roll_joint | +0.133 -> +0.133 | +0.019 -> +0.022 | -0.019 / 0.171 / +0.112 | 0.079 | 5.54 Nm | 5.46 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.496 -> +0.496 | +0.664 -> +0.516 | +0.005 / 0.025 / -0.019 | 0.128 | 2.14 Nm | 2.31 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | -0.355 -> -0.356 | -0.480 -> -0.427 | +0.117 / 0.253 / +0.071 | 0.055 | 9.93 Nm | 10.09 Nm | 40.000 / 3.003 |
| right_ankle_roll_joint | -0.075 -> -0.075 | +0.003 -> -0.046 | -0.003 / 0.055 / -0.029 | 0.140 | 1.73 Nm | 1.78 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.489 -> -0.200 | +0.305 -> +0.236 | -0.305 / 0.437 / -0.436 | 0.140 | 6.36 Nm | 6.26 Nm | 14.251 / 2.722 |
| waist_roll_joint | +0.138 -> +0.138 | -0.016 -> -0.010 | +0.016 / 0.175 / +0.148 | 0.070 | 2.66 Nm | 2.48 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `left_ankle_pitch_joint`: `0.554 rad`.
- `waist_pitch_joint`: `0.437 rad`.
- `left_hip_roll_joint`: `0.259 rad`.
- `left_shoulder_pitch_joint`: `0.254 rad`.
- `right_ankle_pitch_joint`: `0.253 rad`.

## Largest measured speeds

- `right_elbow_joint`: `0.250 rad/s`.
- `left_elbow_joint`: `0.214 rad/s`.
- `right_shoulder_pitch_joint`: `0.201 rad/s`.
- `left_shoulder_pitch_joint`: `0.177 rad/s`.
- `left_shoulder_roll_joint`: `0.177 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
