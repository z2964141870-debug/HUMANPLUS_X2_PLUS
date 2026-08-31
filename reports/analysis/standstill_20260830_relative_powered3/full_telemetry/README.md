# Powered StandStill Telemetry Probe

Source: `analysis_logs/suspended_sonic_20260830_171018_powered3`

## Outcome

- Aligned policy data: `1500` samples / `29.979 s`.
- Pelvis roll/pitch/tilt: `-1.13/-9.22/9.29 deg` -> `+2.54/-0.41/2.57 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.671 -> +0.671 | +0.661 -> +0.692 | +0.008 / 0.040 / -0.021 | 0.189 | 3.72 Nm | 3.66 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.081 -> +0.081 | -0.484 -> -0.335 | +0.121 / 0.541 / +0.416 | 0.116 | 17.03 Nm | 17.10 Nm | 32.064 / 3.003 |
| left_ankle_roll_joint | +0.183 -> +0.182 | -0.012 -> +0.046 | +0.012 / 0.185 / +0.136 | 0.104 | 6.01 Nm | 5.85 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.507 -> +0.507 | +0.679 -> +0.496 | -0.010 / 0.042 / +0.011 | 0.104 | 3.72 Nm | 3.93 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | +0.009 -> +0.009 | -0.493 -> -0.435 | +0.130 / 0.543 / +0.444 | 0.055 | 17.26 Nm | 17.35 Nm | 32.064 / 3.003 |
| right_ankle_roll_joint | -0.047 -> -0.047 | -0.012 -> -0.036 | +0.012 / 0.028 / -0.011 | 0.116 | 0.79 Nm | 0.73 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.484 -> -0.200 | +0.312 -> +0.241 | -0.312 / 0.441 / -0.441 | 0.122 | 6.45 Nm | 6.32 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.298 -> -0.200 | +0.021 -> -0.020 | -0.021 / 0.210 / -0.180 | 0.047 | 2.92 Nm | 3.00 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `right_ankle_pitch_joint`: `0.543 rad`.
- `left_ankle_pitch_joint`: `0.541 rad`.
- `waist_pitch_joint`: `0.441 rad`.
- `left_shoulder_pitch_joint`: `0.250 rad`.
- `right_shoulder_pitch_joint`: `0.241 rad`.

## Largest measured speeds

- `right_elbow_joint`: `0.263 rad/s`.
- `left_elbow_joint`: `0.214 rad/s`.
- `left_shoulder_roll_joint`: `0.201 rad/s`.
- `right_shoulder_pitch_joint`: `0.201 rad/s`.
- `left_knee_joint`: `0.189 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
