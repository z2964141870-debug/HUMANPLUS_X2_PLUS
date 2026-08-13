#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/migration_manifest.tsv"
ROOT="${X2_MIGRATION_ROOT:-${HOME}/x2_migration_20260812}"
BDPAN="${BDPAN:-${HOME}/.local/bin/bdpan}"
MODE="full"

case "${1:---full}" in
  --core) MODE="core" ;;
  --full) MODE="full" ;;
  *) echo "usage: $0 [--core|--full]" >&2; exit 64 ;;
esac
[[ -x "${BDPAN}" ]] || BDPAN="$(command -v bdpan || true)"
[[ -n "${BDPAN}" && -x "${BDPAN}" ]] || { echo "bdpan unavailable" >&2; exit 2; }
"${BDPAN}" whoami 2>/dev/null | grep -q '已登录' || {
  echo "bdpan is not logged in; run login_bdpan.sh" >&2
  exit 3
}

mkdir -p "${ROOT}/downloads"
TOTAL=0
DONE=0
while IFS=$'\t' read -r tier remote relative bytes sha; do
  [[ -z "${tier}" || "${tier}" == \#* ]] && continue
  [[ "${MODE}" == "full" || "${tier}" == "core" ]] || continue
  TOTAL=$((TOTAL + 1))
  target="${ROOT}/downloads/${relative}"
  mkdir -p "$(dirname "${target}")"
  if [[ -f "${target}" ]] \
    && [[ "$(stat -c '%s' "${target}")" == "${bytes}" ]] \
    && [[ "$(sha256sum "${target}" | awk '{print $1}')" == "${sha}" ]]; then
    echo "[${TOTAL}] verified, skip: ${relative}"
    DONE=$((DONE + 1))
    continue
  fi
  if [[ -e "${target}" ]]; then
    echo "existing target does not match manifest; refusing overwrite: ${target}" >&2
    exit 4
  fi
  echo "[${TOTAL}] download: ${remote}"
  "${BDPAN}" download "${remote}" "${target}"
  [[ -f "${target}" ]] || { echo "download did not create ${target}" >&2; exit 5; }
  [[ "$(stat -c '%s' "${target}")" == "${bytes}" ]] || {
    echo "size mismatch: ${target}" >&2; exit 6;
  }
  actual="$(sha256sum "${target}" | awk '{print $1}')"
  [[ "${actual}" == "${sha}" ]] || { echo "SHA mismatch: ${target}" >&2; exit 7; }
  DONE=$((DONE + 1))
done < "${MANIFEST}"
echo "download gate passed: ${DONE}/${TOTAL} files"
