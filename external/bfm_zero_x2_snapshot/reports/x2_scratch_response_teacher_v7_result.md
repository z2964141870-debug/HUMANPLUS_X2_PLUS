# X2 scratch response-teacher v7

Decision: `PASS_FIXED_COMMAND_SCRATCH_KERNEL_LOCAL_ONLY`.

The fresh BFM student now survives all 512 lanes for 400 control steps in both
the response domain and the ideal domain.  Mean forward velocity is
`0.366578 m/s` in response and
`0.412366 m/s` in ideal.  Stage219 weights were
not loaded into the student; 204,800 response-domain teacher frames were used
as labels.

This supports the dataset-coverage hypothesis for fixed 0.35 m/s locomotion.
It does not prove posture correction, stopping, turns, low speed, deployment,
or long-training readiness.  The teacher command screen failed for
`turn_right, vx_0p20`, so those roles are not admitted as training data.
