# X2 native full-actor capability pilot

## Decision

The small-adapter line is closed. Phase60 and Phase77p changed the deployed
action too little to produce a repeatable posture benefit, while the existing
Stage219/Stage264 identity already supplies useful locomotion. The next policy
experiment will therefore use:

- the complete 15-dimensional actor, warm-started from Stage219 iteration
  2600;
- a freshly initialized and fully trainable critic;
- the exact 93D observation, 15D action and gait-template scale `0.15` used by
  the recorded Stage219 training contract;
- frozen exploration standard deviation;
- no additive residual, LoRA, old critic bootstrap, or stationary actor;
- the separately validated state-feedback brake as the product stop kernel,
  outside the learned locomotion actor.

This is deliberately not a literal random policy start. Throwing away a
current 22/24 locomotion backend would make the experiment answer two questions
at once. Full actor fine-tuning tests whether the old locomotion representation
can be reshaped when all actor weights are free and the critic is valid for the
new objective. A from-scratch actor is the sole fallback if this full-capacity
pilot remains ineffective.

A CPU-only load of the real checkpoint has verified this transformation. The
actor remains bit-identical, the critic state changes, the source 15D standard
deviation remains frozen (`min=0.0882390`, `max=0.7206377`), and the resulting
trainable set is the complete 75,407-parameter actor plus a fresh
73,601-parameter critic (149,008 parameters total).

## Pilot budget

Two independent training seeds are required. Each seed uses 512 environments,
48 transitions per environment and 25 PPO updates: 614,400 transitions and 200
minibatch optimizer steps. The optimizer uses two fixed parameter groups:

- full actor: `3e-5`;
- fresh critic: `2e-4`.

PPO uses two epochs, four minibatches, gamma `0.99`, lambda `0.95`, clip `0.2`
and gradient norm `1.0`. Installed RSL-RL must evaluate `V(s_T)` from the new
critic. No Stage219 critic value may be loaded or used.

## Command curriculum

Every training batch contains eight equal, scene-index-interleaved roles:

1. straight 0.20 m/s;
2. straight 0.35 m/s;
3. straight 0.50 m/s;
4. left turn at 0.35 m/s and +0.15 rad/s;
5. right turn at 0.35 m/s and -0.15 rad/s;
6. deceleration from 0.20 m/s;
7. deceleration from 0.35 m/s;
8. deceleration from 0.50 m/s.

Terminal stationary hold is not assigned to this actor. The product controller
hands the terminal state to the already validated state-feedback brake and then
the zero-command locomotion actor. This prevents the known stationary-policy
handoff defect from contaminating locomotion learning.

## Objective curriculum

All recorded Stage219 locomotion terms remain present. The one-sided backward
pitch and actual-contact COM-support costs ramp together:

- updates 1–5: `-0.5 / -0.5`;
- updates 6–15: `-1.0 / -1.0`;
- updates 16–25: `-1.5 / -1.5`.

The schedule is fixed before training. It is not changed in response to an
intermediate plot. Per-role and per-term reward accounting must be recorded so
that another conflict with velocity/yaw tracking is visible rather than hidden
inside total reward.

## Trust and stopping gates

Before every optimizer update, retain a rollback copy. After the update,
recompute the current actor against the immutable source on the rollout states.
Rollback and stop the seed if any of these occurs:

- non-finite observation, return, loss, gradient or parameter;
- source Gaussian KL mean above `0.02` or max above `0.20`;
- deterministic normalized action drift above `0.10`;
- timeout, or termination fraction more than `0.02`;
- minimum root height below `0.60 m` or maximum tilt above `0.35 rad`.

Evaluate held-out paired source/candidate panels after updates 5, 10, 15, 20
and 25. Source failures remain paired reference evidence; unlike Phase77p they
do not make candidate success mathematically impossible. Candidate absolute
safety still gates promotion.

The update-10 continuation gate requires both train seeds to improve moving
pitch mean by at least `0.005 rad`, with p05 non-regression and no material
velocity, yaw, support, slip or termination regression. The final local gate
requires both seeds to improve mean pitch by at least `0.015 rad` and p05 by at
least `0.010 rad`. Only then may an official 24-case panel be run. Nothing in
this pilot authorizes ONNX export, deployment, upper-body integration or long
robust training.

## Current implementation state

`src/cwi_x2/native_full_actor_pilot.py` contains the simulator-independent
contract and tests. A GPU runner is intentionally not launched until it also
implements deterministic fresh-critic initialization, per-update rollback,
post-update rollout KL, complete term accounting, paired evaluation and
unreliable-source-safe finalization.
