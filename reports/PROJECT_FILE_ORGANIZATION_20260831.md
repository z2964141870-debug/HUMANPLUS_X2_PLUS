# Project File Organization Report

Date: 2026-08-31

## Outcome

A new `humanplus_sonic_x2` branch was created from the current remote `main`
at `8fe53ba8d58e6732f52edae13ecbf5008a5de611`. The branch was reduced to the
four-directory project contract and populated from the active Sonic deployment
workspace.

The active deployment workspace was not moved or cleaned:

```text
/Users/yu/projects/sonic_x2_transfer_v2
```

This avoids breaking existing local-to-SoC1 paths while the Git-managed copy
is established.

## Migration map

| Previous source | New location | Treatment |
| --- | --- | --- |
| Active Sonic source and dirty integration work | `script/sonic_x2_transfer_v2/` | Copied without `.git`, virtualenv, weights, assets, raw telemetry, caches, or backup files |
| Native GMR prototype | `script/gmr_cpp_prototype/` | Copied as source |
| Garment/offload migration files | `script/smartwear_runtime/` | Copied as source |
| Compact analysis summaries | `reports/analysis/` | Markdown and JSON only |
| Garment dry-run output | `logs/garment_dryrun/` | Compact log/JSON/README files only |
| Model provenance | `data/model_metadata/` | JSON sidecars only |
| Raw telemetry, weights, archives, rosbag, assets | External local paths | Kept in place and indexed for Baidu Netdisk |

## Important interpretation

This is a project-management reorganization, not a controller change. No
robot process was started or stopped, no HAL topic was published, and no
runtime was uploaded or compiled.

The outer workspace was reduced to:

```text
X2/
  humanplus_sonic_x2/   # this Git branch
  _pending_baidu/       # large local artifacts awaiting verified upload
  _other_projects/      # independent checkouts
  .remote_work          # compatibility symlink into script/
  gmr_cpp_prototype     # compatibility symlink into script/
  .team                 # Codex runtime state, not project content
```

## Git remote state

The branch is published at:

```text
https://github.com/z2964141870-debug/HUMANPLUS_X2_PLUS/tree/humanplus_sonic_x2
```

The initial organization commit was
`f50f83228afd6f12827fda162c13379a95072270`, based on remote `main` commit
`8fe53ba8d58e6732f52edae13ecbf5008a5de611`. Its local and remote branch
hashes matched after push.

The default SSH key and historical local proxy were unavailable. Push was
completed through GitHub's SSH endpoint on port 443 with the existing
project-specific key. That key path is stored only in local `.git/config` and
is not part of this repository.
