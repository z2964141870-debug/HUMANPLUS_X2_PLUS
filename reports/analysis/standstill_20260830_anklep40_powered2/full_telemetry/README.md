# Powered StandStill Telemetry Probe

Source: `GR00T_audit/gear_sonic_deploy/analysis_logs/suspended_sonic_20260830_192035_anklep40_powered2`

## Outcome

- Aligned policy data: `458` samples / `9.140 s`.
- Pelvis roll/pitch/tilt: `-1.28/-8.11/8.21 deg` -> `+1.10/+10.69/10.75 deg`.
- Domain protection states: leg=[0], waist=[0], arm=[0], head=[0].

## Selected joints

| Joint | Policy -> HAL end | q entry -> end | Tracking entry / peak / end | dq peak | Measured effort peak | Estimated PD peak | Kp / Kd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| left_knee_joint | +0.742 -> +0.742 | +0.671 -> +0.804 | -0.002 / 0.071 / -0.062 | 0.238 | 7.47 Nm | 7.56 Nm | 99.098 / 6.309 |
| left_ankle_pitch_joint | +0.237 -> +0.237 | -0.507 -> -0.090 | +0.144 / 0.524 / +0.327 | 0.910 | 20.66 Nm | 20.71 Nm | 40.000 / 3.003 |
| left_ankle_roll_joint | +0.197 -> +0.197 | +0.005 -> +0.093 | -0.005 / 0.196 / +0.104 | 0.238 | 6.24 Nm | 6.16 Nm | 32.064 / 1.996 |
| right_knee_joint | +0.583 -> +0.583 | +0.668 -> +0.570 | +0.001 / 0.056 / +0.013 | 0.238 | 5.25 Nm | 5.22 Nm | 99.098 / 6.309 |
| right_ankle_pitch_joint | +0.218 -> +0.218 | -0.496 -> -0.156 | +0.133 / 0.485 / +0.375 | 0.604 | 18.78 Nm | 19.11 Nm | 40.000 / 3.003 |
| right_ankle_roll_joint | -0.043 -> -0.043 | -0.000 -> +0.003 | +0.000 / 0.053 / -0.046 | 0.177 | 1.67 Nm | 1.72 Nm | 32.064 / 1.996 |
| waist_pitch_joint | -0.188 -> -0.188 | +0.311 -> +0.103 | -0.311 / 0.463 / -0.291 | 0.577 | 6.61 Nm | 6.70 Nm | 14.251 / 2.722 |
| waist_roll_joint | -0.091 -> -0.091 | +0.013 -> -0.021 | -0.013 / 0.164 / -0.070 | 0.123 | 2.09 Nm | 2.30 Nm | 14.251 / 0.907 |

## Largest tracking errors

- `left_ankle_pitch_joint`: `0.524 rad`.
- `right_ankle_pitch_joint`: `0.485 rad`.
- `waist_pitch_joint`: `0.463 rad`.
- `right_shoulder_pitch_joint`: `0.262 rad`.
- `left_shoulder_pitch_joint`: `0.261 rad`.

## Largest measured speeds

- `left_ankle_pitch_joint`: `0.910 rad/s`.
- `right_ankle_pitch_joint`: `0.604 rad/s`.
- `waist_pitch_joint`: `0.577 rad/s`.
- `right_elbow_joint`: `0.336 rad/s`.
- `right_shoulder_pitch_joint`: `0.263 rad/s`.

`policy_target_pos.csv` is pre-safety/ramp/slew. `target_pos.csv` is the final SafeCommand consumed unchanged by the 250 Hz writer. `pd_torque.csv` is requested PD torque before motor-side limits; `joint_effort.csv` is HAL-reported measured effort.
