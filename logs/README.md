# Logs

This directory tracks compact, readable execution records that are useful in
code review.

Current contents include the 35.02 Hz garment dry-run terminal logs, static
MC snapshots, and small legacy metadata. Full telemetry CSV sets remain
outside Git and are indexed by `data/manifests/external_artifacts_20260831.tsv`.

Use one timestamped subdirectory per run. Never rewrite failed-run evidence.
If a log set is large, commit a summary and external manifest instead of the
raw payload.
