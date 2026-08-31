# Powered StandStill Telemetry Probe

Source: `analysis_logs/suspended_sonic_20260830_163947_powered1`

## Outcome

- Aligned policy data: `1500` samples / `29.979 s`.
- Pelvis roll/pitch/tilt: `-0.59/-11.65/11.66 deg` -> `+2.22/-0.43/2.26 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.673 -> +0.673 | +0.641 -> +0.669 | +0.028 / 0.031 / +0.004 | 0.140 | 2.90 Nm | 2.92 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.088 -> +0.088 | -0.459 -> -0.338 | +0.096 / 0.497 / +0.425 | 0.104 | 15.85 Nm | 15.73 Nm | 32.064 / 3.003 |
| left_ankle_roll_joint | +0.157 -> +0.158 | +0.002 -> +0.023 | -0.002 / 0.160 / +0.134 | 0.067 | 5.19 Nm | 5.08 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.483 -> +0.483 | +0.646 -> +0.480 | +0.023 / 0.027 / +0.003 | 0.104 | 2.49 Nm | 2.31 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | -0.107 -> -0.106 | -0.462 -> -0.420 | +0.099 / 0.368 / +0.314 | 0.055 | 11.69 Nm | 11.81 Nm | 32.064 / 3.003 |
| right_ankle_roll_joint | -0.024 -> -0.024 | -0.002 -> -0.031 | +0.002 / 0.021 / +0.007 | 0.177 | 0.67 Nm | 0.55 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.463 -> -0.200 | +0.349 -> +0.238 | -0.349 / 0.438 / -0.438 | 0.154 | 6.37 Nm | 6.32 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.238 -> -0.200 | +0.009 -> -0.021 | -0.009 / 0.203 / -0.179 | 0.058 | 2.74 Nm | 2.88 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `left_ankle_pitch_joint`: `0.497 rad`.
- `waist_pitch_joint`: `0.438 rad`.
- `right_ankle_pitch_joint`: `0.368 rad`.
- `left_shoulder_pitch_joint`: `0.253 rad`.
- `right_shoulder_pitch_joint`: `0.248 rad`.

## Largest measured speeds

- `right_elbow_joint`: `0.336 rad/s`.
- `left_shoulder_pitch_joint`: `0.214 rad/s`.
- `left_shoulder_roll_joint`: `0.201 rad/s`.
- `left_elbow_joint`: `0.201 rad/s`.
- `right_shoulder_pitch_joint`: `0.201 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
