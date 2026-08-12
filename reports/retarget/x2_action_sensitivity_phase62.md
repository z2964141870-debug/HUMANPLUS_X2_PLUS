# X2 Phase62 sagittal action sensitivity

- Decision: `PASS_DIAGNOSTIC_DIRECTION_ONLY`; no checkpoint, ONNX, training, or deployment promotion.
- Contract: one Isaac run, 64 ideal-actuator/fixed-upper envs, 200 steps (4.0 s), Stage219 source, base plus bilateral hip/knee/ankle-pitch and waist-pitch `±0.02` action probes.
- Integrity: all 9 groups survived 4.0 s with zero termination; result SHA-256 `d166e8ecb1658a70745bb56207f546cc6bae27b6d63a407962dbd2cae3fa7f1e`.
- Strongest local direction: negative bilateral hip-pitch action, central pitch slope `-0.277599 rad/action`. At `-0.02`, mean pitch improved `+0.005962 rad` and p05 improved `+0.002043 rad`; support and velocity were non-worse versus base.
- Rejection boundary: the same `-0.02` group increased lateral/yaw absolute metrics from `0.081068/0.090550` to `0.124020/0.137780`. It is therefore causal direction evidence only, not an actor intervention.
- Next allowed experiment: preregistered small-dose bilateral hip-pitch screen; no further reward-only tuning or long training is unlocked by this result.
- Resource: `26.846 s`, peak GPU memory `3252 MiB`, disk used delta `135,966,720 bytes`, no generated weight.
- Provenance note: the first report retained the reused evaluator's legacy `mode=eval` tag. The Phase62/action-sensitivity payload is complete, its SHA is frozen, and the finalizer records this compatibility path; code now emits `mode=screen` for future runs.
