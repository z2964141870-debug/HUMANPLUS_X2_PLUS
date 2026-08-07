#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
PREFIX="${PREFIX:?PREFIX is required}"
REPEATS="${REPEATS:-6}"
BASE_DOMAIN_ID="${BASE_DOMAIN_ID:-190}"
RESUME="${RESUME:-true}"
COMMAND_VX="${COMMAND_VX:-0.30}"

if (( BASE_DOMAIN_ID < 0 || BASE_DOMAIN_ID + REPEATS - 1 > 232 )); then
  echo "invalid ROS domain range" >&2
  exit 2
fi

failures=0
for repeat in $(seq 1 "$REPEATS"); do
  result="$RESULT_ROOT/${PREFIX}_straight_r${repeat}.json"
  if [[ "$RESUME" == "true" && -f "$result" ]] && python3 - "$result" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["summary"].get("full_gate_pass") else 1)
PY
  then
    echo "resume: keeping passed result $result"
    continue
  fi
  if ! env CASE_NAME="${PREFIX}_straight_r${repeat}" \
    ROS_DOMAIN_ID="$((BASE_DOMAIN_ID + repeat - 1))" RESULT_ROOT="$RESULT_ROOT" \
    MODEL_PATH=/models/stage219_s2600_actor.onnx COMMAND_VX="$COMMAND_VX" \
    MOVE_SECONDS=4.0 STOP_SECONDS=8.0 \
    ACTION_BIAS_MODE=lateral_recovery_supervisor ACTION_BIAS=0.50 \
    ANKLE_ROLL_COMMON_BIAS=0.20 RECOVERY_ENTER_M=0.12 RECOVERY_EXIT_M=0.04 \
    RECOVERY_SLEW_RATE_PER_S=1.0 STOP_CONTROLLER=policy \
    bash "$SCRIPT_DIR/run_official_gate_case.sh"; then
    failures=$((failures + 1))
  fi
done

python3 - "$RESULT_ROOT" "$PREFIX" "$REPEATS" "$failures" <<'PY'
import glob, json, sys
root, prefix, repeats, shell_failures = sys.argv[1:]
paths = sorted(glob.glob(f"{root}/{prefix}_straight_r*.json"))
runs = [json.load(open(path, encoding="utf-8"))["summary"] for path in paths]
result = {
    "prefix": prefix,
    "expected_runs": int(repeats),
    "found_runs": len(runs),
    "passes": sum(bool(run.get("full_gate_pass")) for run in runs),
    "shell_failures": int(shell_failures),
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["found_runs"] == result["expected_runs"] and result["passes"] == result["expected_runs"] else 2)
PY
