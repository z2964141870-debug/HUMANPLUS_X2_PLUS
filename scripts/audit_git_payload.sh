#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIMIT_BYTES="${GIT_PAYLOAD_LIMIT_BYTES:-10485760}"

cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a git work tree: $ROOT" >&2
  exit 2
fi

violations=0
while IFS= read -r -d '' path; do
  size="$(stat -c '%s' "$path")"
  if (( size > LIMIT_BYTES )); then
    printf 'oversize tracked file: %s bytes\t%s\n' "$size" "$path" >&2
    violations=1
  fi
done < <(git ls-files -z)

for forbidden in logs checkpoints videos; do
  if git ls-files --error-unmatch "$forbidden" >/dev/null 2>&1; then
    echo "forbidden generated directory is tracked: $forbidden" >&2
    violations=1
  fi
done

if (( violations != 0 )); then
  exit 1
fi

tracked_count="$(git ls-files | wc -l)"
tracked_bytes="$(
  git ls-files -z \
    | xargs -0 -r stat -c '%s' \
    | awk '{sum += $1} END {print sum + 0}'
)"
printf 'git payload audit passed: %s files, %s bytes, per-file limit %s bytes\n' \
  "$tracked_count" "$tracked_bytes" "$LIMIT_BYTES"

