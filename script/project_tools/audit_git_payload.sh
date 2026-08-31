#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

allowed_root='^(AGENTS\.md|README\.md|\.gitignore|\.gitattributes|data/|logs/|script/|reports/)'
bad_root=0
while IFS= read -r -d '' item; do
  if [[ ! "$item" =~ $allowed_root ]]; then
    echo "unexpected top-level path: $item" >&2
    bad_root=1
  fi
done < <(git ls-files -z --cached --others --exclude-standard)

if ((bad_root)); then
  exit 1
fi

large_file=0
while IFS= read -r -d '' item; do
  bytes="$(stat -f '%z' "$item" 2>/dev/null || stat -c '%s' "$item")"
  if ((bytes > 52428800)); then
    echo "file exceeds 50 MiB Git limit: $bytes $item" >&2
    large_file=1
  fi
done < <(find data logs script reports -type f -print0)

if ((large_file)); then
  exit 1
fi

if find . -path './.git' -prune -o -type f \
  \( -name '*.onnx' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \
     -o -name '*.db3' -o -name '*.mcap' -o -name '*.pem' -o -name '*.key' \
     -o -name '.env' -o -name '.env.*' \) -print | grep -q .; then
  echo "forbidden artifact or credential file found" >&2
  exit 1
fi

if rg -l --hidden --glob '!.git/**' \
  --glob '!script/project_tools/audit_git_payload.sh' \
  'BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|(^|[^A-Za-z])(BDUSS|STOKEN)=|sshpass[[:space:]]+-p|密码.{0,16}(为|是|:|：|=|[[:space:]])[[:space:]]*[0-9]{1,}' \
  . >/dev/null; then
  echo "possible credential material found" >&2
  exit 1
fi

git diff --check
git diff --cached --check -- \
  .gitattributes .gitignore AGENTS.md README.md \
  data/README.md logs/README.md script/README.md \
  reports/BAIDU_UPLOAD_STATUS.md reports/CLEANUP_LEDGER_20260831.md \
  reports/PROJECT_FILE_ORGANIZATION_20260831.md reports/README.md \
  reports/STORAGE_POLICY.md script/project_tools
echo "Git payload audit passed."
