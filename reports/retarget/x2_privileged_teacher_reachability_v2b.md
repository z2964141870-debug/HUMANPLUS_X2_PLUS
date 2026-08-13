# Privileged teacher reachability v2b

Formal decision: `FAIL_NO_TEACHER_REACHABILITY_STOP`.

The result-blind repair was technically valid. The attempt0 failure occurred after its first in-memory population because `torch.inference_mode()` made Isaac state buffers immutable across reset; v2b changed only that context to `torch.no_grad()`. The repaired run completed 8 CEM updates (504 coordinated 15D candidates), two parity-swapped validation passes, 2,000 control steps, zero policy optimizer steps, zero backward calls, and zero checkpoints.

The best direction contains a large posture signal: candidate versus source signed-pitch mean improved by `+0.121636 rad` and p05 by `+0.056284 rad`. Support outside improved by `-0.024539 m`, stance-slip p95 by `-0.102489 m/s`, and yaw RMSE by `-0.012741 rad/s`. This is the first evidence in this route that a coordinated 15D correction can materially change the target posture without retraining from scratch.

The full correction is not a safe teacher. Velocity RMSE regressed by `+0.098917 m/s`, lateral RMS by `+0.016753 m/s`, residual step max was `0.175010`, and source/candidate each recorded three termination events across the swapped validation. Candidate lane A failed while lane B survived, so the correction is also environment-index sensitive. The source itself did not meet the absolute survival/tilt gate, which must be treated as a paired baseline limitation rather than hidden.

No teacher panel, BC/DAgger, PPO, export, or deployment is unlocked. The next admissible experiment is a separately preregistered zero-training cyclic scale sweep of the frozen best direction. Its purpose is only to test whether a smaller, smoother dose retains at least `+0.015 rad` pitch improvement without worsening paired termination, velocity, lateral/yaw, support, slip, height, or tilt.
