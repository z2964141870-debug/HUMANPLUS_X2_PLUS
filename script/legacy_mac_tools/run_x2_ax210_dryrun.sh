#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${X2_SONIC_REAL_DIR:-$HOME/projects/X2_sonic_real}"
PYTHON_BIN="${X2_SONIC_PYTHON:-$HOME/venvs/x2_sonic_cuda/bin/python}"
AX210_ADDRESS="${X2_AX210_ADDRESS:-E4:4A:E0:A4:69:E1}"

cd "$PROJECT_DIR"
mkdir -p logs

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_LOG="logs/ax210_dryrun_${STAMP}.log"
exec > >(tee -a "$SESSION_LOG") 2>&1

echo "[x2-ax210] READ-ONLY + LOCAL DRY-RUN; HAL/MC will not be touched"
echo "[x2-ax210] session log: $SESSION_LOG"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[x2-ax210] Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

controller_line="$(bluetoothctl list 2>/dev/null | grep -F "Controller $AX210_ADDRESS " || true)"
if [[ -z "$controller_line" ]]; then
  echo "[x2-ax210] AX210 controller $AX210_ADDRESS is unavailable" >&2
  bluetoothctl list 2>/dev/null || true
  exit 3
fi

controller_info="$(bluetoothctl show "$AX210_ADDRESS" 2>/dev/null || true)"
if ! grep -q $'\tPowered: yes' <<<"$controller_info"; then
  echo "[x2-ax210] AX210 is present but not powered" >&2
  exit 4
fi

hci_line="$(hciconfig 2>/dev/null | awk -v address="$AX210_ADDRESS" '
  /^hci[0-9]+:/ { device=$1; sub(":", "", device) }
  $0 ~ "BD Address: " address { print device; exit }
')"
if [[ -z "$hci_line" ]]; then
  echo "[x2-ax210] no HCI device exposes AX210 address $AX210_ADDRESS" >&2
  exit 5
fi

hci_info="$(hciconfig "$hci_line" 2>/dev/null || true)"
if grep -q 'ACL MTU: 0:0' <<<"$hci_info"; then
  echo "[x2-ax210] AX210 firmware did not initialize (ACL MTU is 0:0)" >&2
  exit 6
fi

if ! "$PYTHON_BIN" -c 'import bleak' >/dev/null 2>&1; then
  echo "[x2-ax210] bleak is unavailable in $PYTHON_BIN" >&2
  exit 7
fi

echo "[x2-ax210] controller=$AX210_ADDRESS hci=$hci_line powered=yes"
echo "[x2-ax210] jacket=${X2_BLE_UP:-FB:08:4D:3B:D6:06}"
echo "[x2-ax210] pants=${X2_BLE_DOWN:-CC:D1:FA:DD:6D:B8}"
echo "[x2-ax210] wait for both garments to remain stable for 5 seconds"
echo "[x2-ax210] press Enter only after the T-POSE READY prompt"

exec ./run_x2_live_dryrun.sh
