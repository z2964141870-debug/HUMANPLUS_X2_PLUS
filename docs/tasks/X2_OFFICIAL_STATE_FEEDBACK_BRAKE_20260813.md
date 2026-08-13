# X2 official state-feedback brake feasibility

## Question

Can the already implemented odometry-feedback brake remove the first-second
momentum that remains in the current right-turn failure, then hand control to
the newly qualified Stage219 zero-command locomotion hold?

## Fixed intervention

- Frozen Stage219 actor, Stage250 action contract, official AimDK MuJoCo.
- During stop, body-frame horizontal command is `clip(-1.5 * v_xy, -0.30, 0.30)`.
- Gait remains active while braking; template amplitude follows measured speed
  with the existing `0.30 m/s` scale and `0.25` floor.
- At or after `0.5 s`, once horizontal speed is at most `0.05 m/s` and the
  predicted gait phase is double support, latch permanently to the same main
  actor under the exact zero-command mode qualified by the preceding campaign.
- No emergency latch, no stationary actor, no parameter sweep.

The gain is not selected from this campaign. `K=1.5` is the pre-existing best
short-horizon drift setting recorded in
`OFFICIAL_X2_FOOT_CONTACT_AND_BRAKE_ARBITRATION_20260808.md`. The only new
question is whether the stable Stage219 locomotion-zero hold closes the old
post-brake stability gap.

## Cases and stopping rule

Run exactly once on each current Stage264 new-machine failure:

1. low straight r4;
2. low right-turn r3.

Both use their frozen movement supervisors and the unchanged official gates.
If both fully pass, only writing a fresh 24-case panel preregistration is
unlocked. If movement survives but a stop metric fails, retain this as a brake
skill result and stop controller tuning. Any movement/start failure routes to a
new-actor preregistration only.

## Prohibitions

Zero training, backward, optimizer, checkpoint, export, deployment, Baidu or
GitHub-remote operation. Raw traces and resources are immutable and each case
has one physical attempt.
