# HumanPlus Sonic X2 Agent Rules

## Read order

Before changing control code or touching the robot, read:

1. `README.md`
2. `reports/README.md`
3. `script/sonic_x2_transfer_v2/GR00T_audit/AGENTS.md`
4. `script/sonic_x2_transfer_v2/GR00T_audit/docs/CURRENT_STATE.md`
5. `script/sonic_x2_transfer_v2/GR00T_audit/docs/DECISIONS.md`
6. `script/sonic_x2_transfer_v2/GR00T_audit/docs/OPERATIONS.md`
7. `script/sonic_x2_transfer_v2/GR00T_audit/docs/EXPERIMENTS.md`

The nested `AGENTS.md` contains the real-hardware safety contract and remains
binding.

## Project hygiene

- Keep all project content under `data/`, `logs/`, `script/`, or `reports/`.
- Commit code, compact text logs, manifests, and Markdown to GitHub.
- Put weights, raw telemetry, rosbag, videos, archives, and large datasets in
  Baidu Netdisk. Commit their byte size, SHA-256, remote path, and status.
- Never commit passwords, OAuth tokens, cookies, private keys, or local
  credential configuration.
- Do not delete a local large artifact until remote byte size and a downloaded
  SHA-256 have both been verified.
- Preserve failed experiment logs. Summarize evidence narrowly.
- The live deployment workspace is intentionally dirty. Do not reset, clean,
  rename, or reformat it while taking snapshots into this branch.
- After each completed stage, update the relevant README or experiment ledger
  immediately.

## Robot boundary

This repository organization does not authorize robot commands. Before any
custom HAL publisher, re-check current MC ownership, prove exactly one writer,
obtain current physical suspension confirmation, and follow the nested
operations guide. Never infer current robot state from this filesystem alone.
