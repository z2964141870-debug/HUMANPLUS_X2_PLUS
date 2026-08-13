#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${X2_MIGRATION_ROOT:-${HOME}/x2_migration_20260812}"
MODE="${1:---full}"
FAILED=0
COUNT=0
while IFS=$'\t' read -r tier remote relative bytes sha; do
  [[ -z "${tier}" || "${tier}" == \#* ]] && continue
  [[ "${MODE}" == "--full" || "${tier}" == "core" ]] || continue
  file="${ROOT}/downloads/${relative}"
  COUNT=$((COUNT + 1))
  if [[ ! -f "${file}" ]] \
    || [[ "$(stat -c '%s' "${file}" 2>/dev/null || echo -1)" != "${bytes}" ]] \
    || [[ "$(sha256sum "${file}" 2>/dev/null | awk '{print $1}')" != "${sha}" ]]; then
    echo "FAIL ${relative}"
    FAILED=$((FAILED + 1))
  else
    echo "PASS ${relative}"
  fi
done < "${SCRIPT_DIR}/migration_manifest.tsv"
[[ "${FAILED}" == 0 ]] || { echo "verification failed: ${FAILED}/${COUNT}" >&2; exit 1; }
echo "verification passed: ${COUNT}/${COUNT}"
