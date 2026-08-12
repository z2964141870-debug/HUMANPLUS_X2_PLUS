# Stage264 frozen baseline

This directory freezes the Task50 baseline by content hash. `Stage264` denotes
the readiness decision; its policy weights are Stage219-s2600 and its nominal
deployment behavior is the Stage250 contract.

The authoritative machine-readable inventory is `manifest.json`. Assets remain
in their established locations to avoid duplicate checkpoints and SDK trees.
The baseline is not replay-unlocked until every required asset is a regular file
with the expected SHA-256.

Run:

```bash
python3 tools/official_x2/audit_native_locomotion_baseline_preflight.py
```

Current limitation: the exact 40-frame gait-template NPZ has not been restored.
Do not replace it with an approximate reconstruction.
