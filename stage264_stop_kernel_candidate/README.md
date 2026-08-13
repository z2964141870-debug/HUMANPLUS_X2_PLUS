# Stage264 state-feedback stop kernel candidate

This directory freezes a local integration candidate for the **stop phase
only**.  It is not a replacement locomotion checkpoint and it is not approved
for deployment.

The kernel preserves the Stage219 actor and Stage250 action contract.  During
the stop event it commands body-frame planar velocity
`clip(-1.5 * v_xy, -0.30, 0.30)` through the same locomotion actor and gait
template.  Once speed is at most `0.05 m/s`, at least `0.5 s` has elapsed, and
the predicted gait state is double support, it latches to the same actor at a
zero command.  It never hands control to the stationary actor.

Evidence boundary:

- the corrected 24-case matrix passed the stop gate 24/24;
- stop drift mean/median/max was 0.0564/0.0510/0.1367 m;
- minimum stop height was 0.6033 m and maximum stop tilt was 0.2324 rad;
- the complete locomotion gate was only 20/24 because four turns missed the
  yaw-progress gate before the stop phase.

Therefore this manifest may be used to reproduce or integrate the stop kernel
behind a separate review, but it may not be cited as a 24/24 locomotion result.
Training, export, hardware deployment, and changing the default adapter
behavior remain locked.
