#!/usr/bin/env bash
set -euo pipefail

BDPAN="${BDPAN:-${HOME}/.local/bin/bdpan}"
[[ -x "${BDPAN}" ]] || BDPAN="$(command -v bdpan || true)"
[[ -n "${BDPAN}" && -x "${BDPAN}" ]] || {
  echo "bdpan is not installed; run install_local_bdpan.sh first." >&2
  exit 2
}

if "${BDPAN}" whoami 2>/dev/null | grep -q '已登录'; then
  echo "bdpan is already logged in."
  "${BDPAN}" whoami
  exit 0
fi

AUTH_URL="$("${BDPAN}" login --get-auth-url --accept-disclaimer)"
echo "Open this URL in your own browser:"
echo "${AUTH_URL}"
echo
read -r -s -p "Paste the 32-character authorization code: " AUTH_CODE
echo
[[ "${AUTH_CODE}" =~ ^[0-9a-fA-F]{32}$ ]] || {
  unset AUTH_CODE
  echo "invalid authorization code format" >&2
  exit 3
}
printf '%s\n' "${AUTH_CODE}" \
  | "${BDPAN}" login --set-code-stdin --accept-disclaimer >/dev/null
unset AUTH_CODE
"${BDPAN}" whoami
