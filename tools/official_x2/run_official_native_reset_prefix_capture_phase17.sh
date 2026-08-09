#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
WORKSPACE="${WORKSPACE:-$OFFICIAL_ROOT/worktree/x2_rl_deploy}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_reset_prefix_phase17_20260809}"
DEFAULT_YAML="${DEFAULT_YAML:-$OFFICIAL_ROOT/vendor/aimdk-aarch64-a424add7-artifacts/extra/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml}"
DOCKER_IMAGE="${DOCKER_IMAGE:-x2-aimdk-humble:1.0}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-216}"
RL_SECONDS="${RL_SECONDS:-20}"
CAPTURE_NAME="${CAPTURE_NAME:-official_native_event_v2_reset_ready_phase17}"

[[ "$ROS_DOMAIN_ID" == "216" ]] || { echo "Phase17 is frozen to isolated domain 216" >&2; exit 2; }
[[ "$ROS_DOMAIN_ID" != "232" ]] || { echo "domain 232 is reserved by current A3" >&2; exit 2; }
mkdir -p "$RESULT_ROOT"

timeout 120s sg docker -c "docker run --rm --init \
  --name x2-native-reset-phase17-$ROS_DOMAIN_ID --network host --ipc host \
  -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -e RL_SECONDS=$RL_SECONDS \
  -e CAPTURE_NAME=$CAPTURE_NAME -e OUTPUT_ROOT=/results \
  -v $SCRIPT_DIR/run_official_native_reset_prefix_capture_v2_inner.sh:/run_capture_inner.sh:ro \
  -v $WORKSPACE:/official:ro -v $REPO_ROOT:/repo:ro -v $RESULT_ROOT:/results \
  -v $DEFAULT_YAML:/official_default.yaml:ro \
  $DOCKER_IMAGE bash -lc 'cp -a /official/. /workspace/; cp /official_default.yaml /workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml; source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; bash /run_capture_inner.sh'"

python3 - "$RESULT_ROOT/$CAPTURE_NAME.manifest.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "manifest": sys.argv[1], "snapshots": r["snapshots"],
    "command_events": r["command_events"], "mode_events": r["mode_events"],
    "control_mode_sequence": r["control_mode_sequence"],
    "ros_domain_id": r["ros_domain_id"],
}, indent=2))
PY
