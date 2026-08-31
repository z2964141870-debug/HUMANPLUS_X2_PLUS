# Cleanup Ledger

Date: 2026-08-31

## Removed or consolidated

- Removed the clean duplicate linked worktree
  `HUMANPLUS_X2_PLUS_latest`; its contents remain recoverable from Git commit
  `fafe16c`.
- Renamed the primary Git worktree from `HUMANPLUS_X2_PLUS` to
  `humanplus_sonic_x2` after creating the new branch.
- Moved 157 MB of untracked legacy HUMANPLUS training artifacts, logs,
  runtime files, and videos out of the new branch into:

  ```text
  /Users/yu/Documents/ChatGPT/X2/_pending_baidu/
    HUMANPLUS_X2_PLUS_main_legacy_20260831/
  ```

  These files were not deleted and remain pending Baidu verification.

- Consolidated 1.9 GB of migration snapshots, raw X2 logs, StandStill
  telemetry, machine archives, AIMDK version data, motion assets, and
  compressed handoff bundles under:

  ```text
  /Users/yu/Documents/ChatGPT/X2/_pending_baidu/
    humanplus_sonic_x2_20260831/
  ```

- Moved the independent `egolocate_x2_bridge` checkout to
  `/Users/yu/Documents/ChatGPT/X2/_other_projects/` so it is no longer mixed
  with Sonic project data.
- Imported the loose top-level Python, shell, Markdown, GMR, MC-transition,
  firmware-package, and audit sources into `script/` or `reports/`, verified
  the copies, and moved the original 3.5 MB source bundle to macOS Trash.
  Compatibility symlinks remain for `.remote_work` and
  `gmr_cpp_prototype`.
- Moved a redundant 2.8 MB partial Git clone and 6.2 MB regenerated public
  PDF cache to macOS Trash. The partial clone's only commit was already an
  ancestor of `main`; it contained no unique working file.
- Removed `.DS_Store`, `.pyc`, and empty `__pycache__` entries from the outer
  X2 workspace.

## Intentionally retained

The following large items remain locally under `_pending_baidu` because no
verified Baidu CLI upload exists on this Mac yet:

- Sonic ONNX weights and active 877 MB telemetry tree.
- Existing X2 logs and rosbag files.
- Machine archives and migration tarballs.
- StandStill raw evidence.
- AIMDK review package, motion assets, and historical transfer snapshots.

They are candidates for later cleanup only after the storage policy's remote
size and downloaded-SHA verification gates pass.

## Regenerable cleanup

The outer workspace now has only the Git-managed project, `_pending_baidu`,
`_other_projects`, two compatibility symlinks, and Codex's `.team` state.
