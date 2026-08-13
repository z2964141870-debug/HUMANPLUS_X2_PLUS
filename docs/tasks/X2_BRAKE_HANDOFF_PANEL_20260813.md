# X2 brake/hold handoff panel

## Purpose

The state-feedback posture/stop v4 run showed that posture correction and stationary-policy acquisition were still coupled. Its source direct handoff terminated 251/256 validation environments during hold; fixed 0.5× and the searched feedback matrix also failed hold. The next experiment therefore removes all posture residuals and all learning.

This panel asks only whether a deterministic transition controller can safely move from the frozen Stage219/Stage264 locomotion actor to either its zero-command behavior or the frozen stationary actor.

## Fixed experiment

- One Isaac process, 256 environments, 820 control steps at 50 Hz.
- Event: 1 s stand, 1 s accelerate, 4.2 s cruise at 0.35 m/s, 2 s decelerate, at least 8.2 s hold.
- Sixteen fixed handoff strategies, 16 environments per strategy, with all 16 schedule-to-gait offsets represented in each strategy.
- The top three screening strategies plus `direct_mix` and `locomotion_zero` are replayed sequentially on the same 256 reset states.
- Frozen Stage219 actor, frozen stationary ONNX, fixed gait template scale 0.15, ideal actuator endpoint, fixed upper body, no push or observation corruption.
- No posture residual, reward inspection, optimizer, backward pass, checkpoint, or model export.

The strategy list is immutable in `src/cwi_x2/brake_handoff_panel.py`. It covers direct stationary acquisition, locomotion-only zero-command hold, stationary-backend variants, contact/speed-gated acquisition, negative-velocity braking commands, time blends, speed blends, and a delayed direct handoff.

## Decision boundary

`PASS_HANDOFF_CONTROLLER_OFFICIAL_PANEL_PREREG_ONLY` requires zero termination/timeout, safe cruise and deceleration relative to direct handoff, terminal speed p95 at most 0.12 m/s, double support at least 0.98, root height at least 0.60 m, tilt at most 0.35 rad, and the fixed support/slip/action continuity bounds. It only permits a separately preregistered official MuJoCo panel.

`PARTIAL_HANDOFF_BRAKE_SKILL_PREREG_ONLY` requires no moving-phase termination and at least a 50% reduction in hold terminations. It only permits designing a separately preregistered brake/hold skill.

Otherwise the result is `FAIL_HANDOFF_NEW_ACTOR_PREREG_ONLY`: stop trying to repair the handoff with a state machine and design a new full-capacity actor/training campaign.

No branch unlocks training, deployment, whole-body control, or model export directly.

## Resource accounting

The machine concurrently runs unrelated archive jobs, so the filesystem-wide delta from the external ledger is provenance only. The experiment gates its own screen/resource evidence size, free space, GPU memory, runtime, and the change under `/tmp/IsaacLab`. No Baidu operation is permitted.
