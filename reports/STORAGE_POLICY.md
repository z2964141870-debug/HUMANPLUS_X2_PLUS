# GitHub and Baidu Netdisk Storage Policy

Updated: 2026-08-31

## GitHub

GitHub tracks this branch's code, launchers, tests, compact terminal logs,
small fixtures, manifests, and Markdown reports. The hard repository limit is
50 MB per new file even though GitHub's technical rejection threshold is
higher.

Run before committing:

```bash
script/project_tools/audit_git_payload.sh
```

## Baidu Netdisk

The planned remote root is relative to the official `bdpan` application root:

```text
HUMAN+/HUMANPLUS_X2_PLUS/humanplus_sonic_x2/2026-08-31/
  models/
  data/
  logs/
  archives/
  manifests/
```

Every uploaded artifact must have local path, byte size, SHA-256, remote path,
upload status, and verification status in `data/manifests/`.

Local deletion is allowed only after:

1. Upload completed.
2. Remote byte size matched.
3. A downloaded copy matched the local SHA-256.
4. The verification manifest was committed and pushed.

## Current Mac capability

- Baidu Netdisk desktop client 8.7.9 is installed at
  `/Applications/BaiduNetdisk.app` and can be used for manual upload.
- No `bdpan`, `BaiduPCS-Go`, `bypy`, or `rclone` command is currently in
  `PATH`.
- The `bdpan` binary preserved in historical `main` is Linux x86-64 ELF and
  cannot run on this Apple Silicon Mac.
- No OAuth token, cookie, or account configuration was read during this
  audit.

Therefore automated upload is not yet available on this computer. Large
artifacts remain local and are marked `pending`, rather than being deleted or
falsely reported as backed up.
