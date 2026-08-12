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

The strict no-retry new-machine matrix is complete at **22/24**, not 24/24.
Low-speed straight repeat 4 and low-speed right-turn repeat 3 both passed their
move gates and then fell during the direct locomotion-to-stand policy handoff.
`new_machine_replay.json` hash-binds all 24 external traces; the earlier
`new_machine_replay_partial.json` remains as the independently backed-up 16-case
checkpoint. Do not describe the historical 24/24 result as reproduced, and do
not rerun or overwrite either failed trace to select a passing outcome.

The final eight episodes used an isolated AimRT HTTP listener on port 31822
because the host browser occupied the vendor default 51822. This listener is
not on the ROS/control/physics path; all policy, MuJoCo, PD, timing, gate and ROS
domain settings remained frozen.
