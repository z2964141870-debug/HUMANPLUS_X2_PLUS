# X2 official locomotion-zero failure-case feasibility

## Purpose

Translate the IsaacLab `locomotion_zero` hold signal into the current official
MuJoCo execution contract without training or changing any moving-policy
behavior.  This is the brake-skill feasibility stage unlocked by the partial
three-seed confirmation; it is not the 24-case official promotion panel.

## Frozen comparison

Run exactly the two current Stage264 new-machine failures once each with their
original speed, turn, action-bias, PD, upper, model, template and timing
configuration.  The only change is `STOP_CONTROLLER=locomotion_zero`:

- `stage264_low_straight_r4` (`vx=0.25`);
- `stage264_low_turn_right_r3` (`vx=0.25`, `wz=0.15`).

The new controller keeps the Stage219/Stage264 locomotion actor in the `main`
policy slot, sets command velocity to zero, uses the deployable standing gait
suffix, disables gait-template motion through the existing `moving=false`
contract, and preserves the main actor's executed previous-action history.
It never calls the stationary actor during the stop interval.

Each case receives one physical attempt in a fresh, non-overwriting result
namespace.  Infrastructure failure stops the campaign; a physics gate failure
is preserved and the other preregistered case is still run.

## Decision

- Both cases pass startup, move, stop, and full gates: unlock writing (not
  launching) a fresh 24-case official-panel preregistration.
- Startup/move pass but either stop/full gate fails: keep the candidate as an
  Isaac-only brake/hold hypothesis and preregister a dedicated brake skill or
  new actor; do not sweep controller weights.
- Any startup/move regression: reject official integration and revert to the
  frozen adapter default.

All outcomes keep optimizer, training, export, deployment and the 24-case
launch locked.  No Baidu or GitHub operation is part of this experiment.
