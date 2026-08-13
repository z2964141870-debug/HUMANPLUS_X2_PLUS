#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_BIN="${SCRIPT_DIR}/bin/bdpan"
EXPECTED_SHA="ccb24ba1da59c1ecba75d93f94dd67fbded53d3961ba83b4e4c4a87534123dfe"
TARGET_DIR="${HOME}/.local/bin"
TARGET_BIN="${TARGET_DIR}/bdpan"

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || {
  echo "This bundled bdpan is only for Linux x86_64." >&2
  exit 2
}
[[ -f "${SOURCE_BIN}" ]] || { echo "missing ${SOURCE_BIN}" >&2; exit 3; }
printf '%s  %s\n' "${EXPECTED_SHA}" "${SOURCE_BIN}" | sha256sum -c -
mkdir -p "${TARGET_DIR}"
install -m 0755 "${SOURCE_BIN}" "${TARGET_BIN}"
case ":${PATH}:" in
  *":${TARGET_DIR}:"*) ;;
  *) echo "Add this line to your shell profile: export PATH=\"${TARGET_DIR}:\$PATH\"" ;;
esac
"${TARGET_BIN}" version
