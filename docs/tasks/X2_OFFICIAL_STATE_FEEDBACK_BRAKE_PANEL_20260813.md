# X2 official state-feedback brake panel

## Question

Does the fixed odometry-feedback brake followed by the Stage219 main actor at
zero command preserve the complete current 24-case official locomotion panel?

This is an integration test, not training and not a controller search.  The
controller was fixed before this panel from earlier evidence:

- body-frame brake command: `clip(-1.5 * v_xy, -0.30, 0.30)`;
- gait-template speed reference `0.30 m/s`, floor `0.25`;
- latch no earlier than `0.5 s`, at speed at most `0.05 m/s`, and only at the
  predicted double-support phase;
- after the latch, use the same Stage219 locomotion actor with a zero command;
- no stationary actor and no emergency latch.

## Frozen matrix

The matrix is reconstructed in the exact order of the current Stage264
new-machine replay:

- speeds: `0.30`, then `0.25 m/s`;
- at each speed: six straight repetitions, three right turns, three left
  turns;
- movement duration `4.0 s`, stop duration `8.0 s`;
- exactly one attempt per case.

The movement controller, gait template, action supervisors, mirror settings,
PD gains, actor, model XML, and official pass thresholds remain unchanged.
Only the stop controller changes from the stationary-policy handoff to the
fixed state-feedback brake and locomotion-zero hold.

## Decisions

- `24/24`, with a real hold latch in every case: freeze a local integration
  candidate for a later independently reviewed deployment change.
- `23/24` with all movement gates intact: retain the evidence but stop; do not
  tune on this panel.
- otherwise: reject the integration and stop.

Every route keeps training, export, and deployment locked.  The campaign has
zero optimizer steps, zero backward calls, and zero checkpoint writes.  It
uses neither Baidu Netdisk nor a GitHub remote.
