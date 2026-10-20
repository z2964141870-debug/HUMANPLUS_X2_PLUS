# HumanPlus Sonic X2

Branch: `humanplus_sonic_x2`

**Current entry point:** [2026-10-20 successor task card](reports/HANDOFF_TASK_CARD_20261020.md).
The user approved an operator-performed reboot after the failed recovery;
reboot completion and current robot state have not been checked. The keyboard
v0.2 task supersedes the historical garment objective below. Credentials and
raw binaries/telemetry are not included in the GitHub handoff.

Last observed checkpoint: [2026-10-20 SSH takeover](reports/X2_SSH_TAKEOVER_20261020.md), 22:41 CST, before the approved onsite reboot. Two suspended powered attempts hit the upstream native stale protection. At that checkpoint no custom writer remained; MC existed but actual command/feedback recovery was unverified. Vendor IMU logs show approximately 2000 ms blocked Publish calls and FastDDS SHM port errors. Reboot completion and present ownership must be checked by the successor before powered work. Earlier no-publish computation passed; neither ground balance nor keyboard walking has passed. The August snapshot below is historical.

This branch is the project-management and version-control home for the
garment-to-Sonic-to-X2 integration. It was reorganized on 2026-08-31 from the
active deployment workspace without changing the robot or the deployment
copy.

## Directory contract

| Directory | Purpose | Storage rule |
| --- | --- | --- |
| `data/` | Small generated data, model metadata, motion fixtures, manifests | Small reproducibility inputs may be committed. Large data is indexed here and stored in Baidu Netdisk. |
| `logs/` | Terminal output and compact text records | Compact text logs are committed. Raw telemetry, rosbag, large CSV/JSONL, and videos go to Baidu Netdisk. |
| `script/` | Source, launchers, tests, analysis, and project tools | GitHub is authoritative. No credentials or generated runtimes. |
| `reports/` | Current state, decisions, experiment conclusions, and READMEs | GitHub is authoritative. Every completed stage updates a report immediately. |

Do not add new top-level project directories. Root-level governance files are
the only exception.

## Historical August objective

The end goal is a supported first powered closed-loop segment:

```text
V2 garment BLE
  -> SoC1 AX210 + CUDA LFP
  -> 3588S postprocess + Fast-SMPL + native GMR
  -> timestamp HMCP qpos36
  -> raw ZMQ :5555
  -> safety proxy :5557/:5556
  -> Sonic policy 50 Hz
  -> single HAL writer 250 Hz
  -> X2
```

As of the 2026-08-31 snapshot, the garment dry-run has passed at 35.02 Hz and the
fixed StandStill policy has passed only under gantry support. Unsupported
standing is not established. The robot was last reported fully suspended and
offline after a normal `lifted` exit; official MC publisher ownership must be
re-established before any powered work.

## Source identity

The versioned source snapshot is under
`script/sonic_x2_transfer_v2/`. It was copied from the intentionally dirty
deployment workspace:

```text
/Users/yu/projects/sonic_x2_transfer_v2
```

That deployment workspace remains untouched so existing robot paths continue
to work. Its nested upstream Git checkout was at `70bed45`, with local Sonic
changes and untracked integration files. Runtime identity therefore remains
file SHA-256, not upstream `HEAD` alone.

The frozen transfer model is not stored in Git:

```text
x2_sonic_frozen_g1core_lora_v2.onnx
sha256 8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9
```

Its provenance and action-scale sidecars are committed in
`data/model_metadata/`.

## Start here

1. Read `AGENTS.md` first. For v0.2 keyboard walking, read the [2026-10-19 deployment and operation-card review](reports/SONIC_V02_KEYBOARD_WALK_REVIEW_20261019.md), then the [Planner-first successor handoff](reports/SONIC_V02_PLANNER_FIRST_HANDOFF_20261008.md). For the frozen old runtime, read the [2026-09-29 handoff](reports/HANDOFF_20260929.md).
2. Read `reports/README.md` and the last recorded operational documents linked
   there.
3. Run `script/project_tools/audit_git_payload.sh` before every commit.
4. Regenerate `data/manifests/external_artifacts_20260831.tsv` after changing
   any external model, telemetry, or archive.

Storage and cleanup decisions are recorded in
`reports/STORAGE_POLICY.md` and
`reports/PROJECT_FILE_ORGANIZATION_20260831.md`.
