# Privileged teacher dose v2c

Formal decision: `PASS_SAFE_TEACHER_DOSE_LOCAL_ONLY`.

The exact v2b `15 × 5` teacher direction was frozen and evaluated at eight preregistered scales using eight cyclic reset passes. Every one of the 64 environment indices ran every scale exactly once; no new search or fitting occurred. All technical replay, assignment, finite, model-immutability, and resource checks passed. There were zero policy optimizer steps, zero backward calls, and zero checkpoints.

Scale `0.5` was the only nonzero scale to pass every pooled and even/odd parity gate. Relative to the same-index source, it improved signed-pitch mean/p05 by `+0.046054/+0.024129 rad`, reduced velocity RMSE by `0.016939 m/s`, yaw RMSE by `0.095725 rad/s`, and stance-slip p95 by `0.073108 m/s`. Lateral RMS increased only `0.004580 m/s` and support outside only `0.000242 m`, within the frozen limits. Source had four terminations; the selected candidate had zero, a `0.626536 m` minimum root height, and `0.226353 rad` maximum tilt. The realized action residual max/slew were both `0.087505`, below `0.10`.

Smaller doses `0.10–0.25` were stable and often improved speed/yaw/slip, but did not clear both material pitch gates and regressed support beyond `0.001 m`. Full scale `1.0` retained the large pitch effect but failed termination, speed, lateral, height, tilt, and slew gates.

This pass proves a safe local coordinated-action window at fixed `0.35 m/s` in the ideal/fixed-upper IsaacLab domain. It does not yet prove command, turn, transition, actuator, upper-body, MuJoCo, or hardware generalization. It unlocks only a separately preregistered multi-command teacher panel; BC/DAgger, PPO, export, and deployment remain locked.
