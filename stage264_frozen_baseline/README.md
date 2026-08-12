# Stage264 frozen baseline

This directory freezes the Task50 baseline by content hash. `Stage264` denotes
the readiness decision; its policy weights are Stage219-s2600 and its nominal
deployment behavior is the Stage250 contract.

The authoritative machine-readable inventory is `manifest.json`. Assets remain
in their established locations to avoid duplicate checkpoints and SDK trees.
The baseline is replay-unlocked only when every required asset is a regular file
with the expected SHA-256.

Run:

```bash
python3 tools/official_x2/audit_native_locomotion_baseline_preflight.py
```

The exact 40-frame gait template was rebuilt byte-for-byte from the preserved,
hash-verified Session04 canonical dataset and the original generator. The
recovery candidate matched the frozen SHA-256 before installation; see
`gait_template_recovery_report.json`.

The first new-machine matrix stopped at 15/16 after low-speed straight repeat 4
passed walking but fell during stopping. `new_machine_replay_partial.json`
hash-binds all completed traces. Do not claim the historical 24/24 gate has been
reproduced until the full no-failure matrix passes.
