# X2 P1/P2 standing-mode comparison

Date: 2026-08-14

## Test boundary

- P1: official controller `L2+X`, confirmed `JOINT_DEFAULT`, 19.47 s.
- P2: official controller `R2+X`, confirmed `STAND_DEFAULT`, 19.12 s.
- The operator kept the sticks centered. Codex only queried read-only services and recorded DDS topics from PC2; no mode, joint, or velocity commands were sent over SSH.
- P2 ended in `STAND_DEFAULT`, status 100.

## Data integrity

| Stream | P1 | P2 |
|---|---:|---:|
| Arm/leg/waist state | ~1000 Hz | ~998 Hz |
| Arm/leg/waist command | ~500 Hz | ~500 Hz |
| Head state / command | ~333 / 100 Hz | ~332 / 100 Hz |
| Torso/chest IMU | ~500 Hz | ~500 Hz |
| Retarget command/frame | ~50 Hz | ~50 Hz |

`odom` and `rl_debug` did not publish in either mode. The 62-value retarget command remained a constant default target in both recordings and was not the source of the standing-mode transition.

## Main result

The operator's observation is correct: P1 and P2 are fundamentally different control regimes.

- Mean per-joint absolute tracking RMS fell from `0.03959 rad` in P1 to `0.02794 rad` in P2.
- P2 did not merely stiffen the P1 pose. It generated a different load-bearing lower-body command:
  - left/right hip pitch targets moved from about `-0.05 rad` to `-0.259/-0.275 rad`;
  - left/right knee targets moved from `0.10 rad` to `0.464/0.419 rad`;
  - ankle pitch targets moved from `-0.05 rad` to `-0.097/-0.148 rad`;
  - hip stiffness rose from `30–40` to about `100`, and knee stiffness from `80` to `150`.
- Large P1 offsets that were misleading as balance metrics improved in P2:
  - left hip pitch error: `0.184 -> 0.063 rad`;
  - left shoulder roll error: `0.134 -> 0.024 rad`;
  - right ankle pitch error: `0.202 -> 0.130 rad`.
- P2 residuals remain concentrated in the sagittal balance chain:
  - right ankle pitch RMS error: `0.130 rad`;
  - left ankle pitch RMS error: `0.117 rad`;
  - waist pitch RMS error: `0.085 rad`;
  - left hip pitch RMS error: `0.063 rad`;
  - right knee RMS error: `0.062 rad`.

These residuals should not be treated as independent position-control failures. In `STAND_DEFAULT`, the controller is using load-bearing joint targets and torques to maintain whole-body balance. The research-relevant contract is therefore state-dependent and concentrated in the hip–knee–ankle–waist sagittal chain.

## Consequence for garment teleoperation

The garment system must not overwrite the entire 62-value target as if every joint were freely position controlled. A defensible first interface is:

1. keep official `STAND_DEFAULT` responsible for lower-body balance;
2. inject only bounded upper-body targets at first;
3. measure how upper-body amplitude changes ankle/waist/hip residuals and IMU motion;
4. expand control authority only after the balance margin is quantified.

The next safe experiment is P3 shadow mode: run the garment inference and retargeting pipeline without UDP/ROS publication, record its proposed 62-value commands alongside P2 robot feedback, and evaluate latency, discontinuities, joint-limit violations, and conflict with the official standing targets before enabling any motion.
