#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
WORKSPACE="${WORKSPACE:-$OFFICIAL_ROOT/worktree/x2_rl_deploy}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_activation_prefix_phase21_20260809}"
DEFAULT_YAML="${DEFAULT_YAML:-$OFFICIAL_ROOT/vendor/aimdk-aarch64-a424add7-artifacts/extra/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml}"
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-220}"
CAPTURE_NAME="${CAPTURE_NAME:-official_native_event_v2_activation_prefix_phase21}"
[[ "$ROS_DOMAIN_ID" == "220" ]] || { echo "Phase21 frozen to domain220" >&2; exit 2; }
[[ "$ROS_DOMAIN_ID" != "232" ]] || exit 2
mkdir -p "$RESULT_ROOT"
timeout 150s sg docker -c "docker run --rm --init --name x2-native-phase21-$ROS_DOMAIN_ID --network host --ipc host \
  -e ROS_DOMAIN_ID=$ROS_DOMAIN_ID -e RL_SECONDS=20 -e CAPTURE_NAME=$CAPTURE_NAME -e OUTPUT_ROOT=/results \
  -v $SCRIPT_DIR/run_official_native_activation_prefix_phase20_inner.sh:/run_phase21.sh:ro \
  -v $WORKSPACE:/official:ro -v $REPO_ROOT:/repo:ro -v $RESULT_ROOT:/results \
  -v $DEFAULT_YAML:/official_default.yaml:ro x2-aimdk-humble:1.0 bash -lc \
  'cp -a /official/. /workspace/; cp /official_default.yaml /workspace/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/default.yaml; source /opt/ros/humble/setup.bash; source /workspace/install/setup.bash; bash /run_phase21.sh'"
python3 - "$RESULT_ROOT/$CAPTURE_NAME.manifest.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print(json.dumps({k:r[k] for k in ('snapshots','command_events','mode_events','control_mode_sequence','ros_domain_id','subscription_ready_elapsed_ns')},indent=2))
PY
