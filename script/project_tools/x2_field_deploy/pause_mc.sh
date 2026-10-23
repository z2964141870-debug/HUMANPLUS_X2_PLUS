#!/bin/bash
# pause_mc.sh <seconds>|--print-identity  (SoC0, run@)
# Auto-discovers the running MC, prints its identity, then hands the
# identity-checked bounded SIGSTOP to x2_mc_bounded_pause_probe.py.
# Auto-CONT applies ALWAYS (Ctrl-C or window end) - MC can never stay frozen.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PID="$(pgrep -x mc_main | head -1 || true)"
if [ -z "$PID" ]; then echo "FATAL: no mc_main process on this host" >&2; exit 1; fi
BOOT="$(cat /proc/sys/kernel/random/boot_id)"
ASHA="$(sha256sum "/proc/$PID/cmdline" | cut -d' ' -f1)"
TICKS="$(sed 's/^[0-9]* (.*) //' "/proc/$PID/stat" | awk '{print $20}')"
echo "MC identity: pid=$PID start_ticks=$TICKS argv_sha256=${ASHA:0:16}... boot=${BOOT:0:8}..."
if [ "${1:-}" = "--print-identity" ]; then echo "(dry -- nothing paused)"; exit 0; fi
SECS="${1:-7200}"
case "$SECS" in (*[!0-9]*|'') echo "usage: pause_mc.sh <seconds>|--print-identity" >&2; exit 1;; esac
exec python3 "$DIR/x2_mc_bounded_pause_probe.py" \
  --pid "$PID" --boot-id "$BOOT" --start-ticks "$TICKS" --argv-sha256 "$ASHA" \
  --execute SUSPENDED_MC_PAUSE_2S --pause-seconds "$SECS"
