# Privileged teacher reachability v1

Decision: `FAIL_NO_TEACHER_REACHABILITY_STOP`.

This CPU-only run preserved the Stage219/Stage250 action contract: 700 historical
rows replayed within `3.58e-7`, the fresh 10-tick fork was exact, and the two
zero-residual source rollouts were exact. No optimizer, backward call, policy
change, checkpoint, or GPU launch occurred.

The CEM oracle found a coordinated 15-D residual with large apparent pitch
improvement (`mean +0.092905 rad`, `p05 +0.037608 rad`) and lower velocity,
support, and slip metrics. Those values are not valid performance evidence:
the direct-MuJoCo source and candidate both fell (`root z min ~0.073 m`, tilt
~1.73 rad), and the candidate also failed lateral safety.

The failure agrees with the pre-existing Phase28 evidence, whose direct-MuJoCo
Stage250 seed was itself `qualified_native_dynamic_seed=false`. Exact action
replay therefore does not make this direct dynamics runner a valid source
performance domain. This v1 result does not unlock teacher labels, BC/DAgger,
PPO, long training, export, or deployment.

Next: repeat the same reachability question in the already validated IsaacLab
Stage219 4-second source domain. Official/AimDK remains the final external
panel, not the inner-loop teacher search model.
