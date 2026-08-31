# Powered StandStill Telemetry Probe

Source: `GR00T_audit/gear_sonic_deploy/analysis_logs/suspended_sonic_20260831_111359_m2_intervened`

## Outcome

- Aligned policy data: `8920` samples / `178.373 s`.
- Pelvis roll/pitch/tilt: `-0.73/-9.93/9.96 deg` -> `+5.29/+13.25/14.25 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.799 -> +0.799 | +0.682 -> +0.881 | -0.013 / 0.082 / -0.082 | 0.299 | 10.11 Nm | 9.91 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.237 -> +0.237 | -0.490 -> -0.336 | +0.127 / 0.686 / +0.573 | 0.470 | 22.12 Nm | 22.01 Nm | 32.064 / 3.003 |
| left_ankle_roll_joint | +0.291 -> +0.291 | +0.018 -> +0.123 | -0.018 / 0.228 / +0.168 | 0.263 | 7.30 Nm | 7.23 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.719 -> +0.719 | +0.665 -> +0.711 | +0.004 / 0.038 / +0.008 | 0.226 | 3.31 Nm | 3.43 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | +0.179 -> +0.179 | -0.463 -> -0.423 | +0.100 / 0.615 / +0.601 | 0.360 | 19.43 Nm | 19.55 Nm | 32.064 / 3.003 |
| right_ankle_roll_joint | -0.121 -> -0.121 | +0.004 -> -0.076 | -0.004 / 0.107 / -0.045 | 0.214 | 3.19 Nm | 3.27 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.190 -> -0.190 | +0.309 -> +0.057 | -0.309 / 0.436 / -0.247 | 0.859 | 6.21 Nm | 6.31 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.084 -> -0.084 | -0.003 -> -0.083 | +0.003 / 0.190 / -0.001 | 0.181 | 2.50 Nm | 2.68 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `left_ankle_pitch_joint`: `0.686 rad`.
- `right_ankle_pitch_joint`: `0.615 rad`.
- `waist_pitch_joint`: `0.436 rad`.
- `left_hip_roll_joint`: `0.301 rad`.
- `left_shoulder_pitch_joint`: `0.257 rad`.

## Largest measured speeds

- `waist_pitch_joint`: `0.859 rad/s`.
- `right_elbow_joint`: `0.495 rad/s`.
- `left_ankle_pitch_joint`: `0.470 rad/s`.
- `right_shoulder_pitch_joint`: `0.409 rad/s`.
- `right_ankle_pitch_joint`: `0.360 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
