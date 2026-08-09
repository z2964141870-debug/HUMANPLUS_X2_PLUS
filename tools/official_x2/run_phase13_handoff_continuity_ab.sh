#!/usr/bin/env bash
set -uo pipefail

# BASE Phase13: one-variable official AimDK handoff physical-target continuity A/B.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1}"
MODEL_ROOT="${MODEL_ROOT:-$OFFICIAL_ROOT/models}"
RESULT_ROOT="${RESULT_ROOT:-$OFFICIAL_ROOT/results/official_native_strict_20260807}"
SUMMARY_PATH="${SUMMARY_PATH:-$REPO_ROOT/reports/official_x2/phase13_handoff_continuity_ab.json}"
REPEATS=5

ADAPTER_SHA="a929ccec61f91af9d8492b11c4e50c7a61379d27a69b02327b984b2a9dd6759f"
HANDOFF_SHA="cd4df30310205a0001f52d1b5e44aae63ccd5ed6f5bd13367a59e09afe2c1a32"
INNER_SHA="c832c3aea34edf38d72f9938e0bc6aad2a26b2bb1a0ca648a7882166ca47cf01"
OUTER_SHA="16007d8cc576ba0fbdcbcc60f20819195b1f56d3c1f8bba2e9713f9a61797801"
MOVING_SHA="da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c"
SOURCE_STAND_SHA="edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565"

check_sha() {
  local expected="$1" path="$2" actual
  [[ -f "$path" ]] || { echo "missing frozen input: $path" >&2; exit 2; }
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA mismatch: $path actual=$actual expected=$expected" >&2
    exit 2
  }
}
check_sha "$ADAPTER_SHA" "$SCRIPT_DIR/stage208_official_mujoco_adapter.py"
check_sha "$HANDOFF_SHA" "$SCRIPT_DIR/skill_handoff_contract.py"
check_sha "$INNER_SHA" "$SCRIPT_DIR/run_official_gate_inner.sh"
check_sha "$OUTER_SHA" "$SCRIPT_DIR/run_official_gate_case.sh"
check_sha "$MOVING_SHA" "$MODEL_ROOT/stage306_s2652_transition_head_actor.onnx"
check_sha "$SOURCE_STAND_SHA" "$MODEL_ROOT/stand_backend_scratch_i150_actor.onnx"

labels=(blend0p0 blend0p5)
blend_seconds=(0.0 0.5)
for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  blend="${blend_seconds[$index]}"
  for repeat in $(seq 1 "$REPEATS"); do
    case_name="phase13_source_recovery_${label}_stiff1p2_fixed_r${repeat}"
    result="$RESULT_ROOT/${case_name}.json"
    if [[ -f "$result" ]]; then
      echo "resume: keeping $result"
      continue
    fi
    env \
      CASE_NAME="$case_name" ROS_DOMAIN_ID="$((150 + index * 10 + repeat))" \
      RESULT_ROOT="$RESULT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
      MODEL_PATH=/models/stage306_s2652_transition_head_actor.onnx \
      STATIONARY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
      RECOVERY_MODEL_PATH=/models/stand_backend_scratch_i150_actor.onnx \
      COMMAND_VX=0.30 POLICY_VX_FLOOR=0 PHASE_OFFSET=0 CLOCK_MODE=step \
      CONTROL_MODE=full DEFAULT_POSE_PROFILE=stage208 \
      PREPARE_SECONDS=0.2 STAND_SECONDS=2.0 STATIONARY_CONTROLLER=policy \
      STATIONARY_WARMUP_SECONDS=0 STATIONARY_BLEND=0.5 \
      MOVE_SECONDS=5.2 MOVE_ACCELERATE_SECONDS=1.0 MOVE_TEMPLATE_MULTIPLIER=1.0 \
      STOP_SECONDS=8.0 STOP_CONTROLLER=curriculum_then_policy \
      STOP_TRANSITION_SECONDS=2.0 STOP_INTENT_DECELERATE_SECONDS=2.0 \
      CURRICULUM_RECOVERY_HANDOFF_BLEND_SECONDS="$blend" \
      FUTURE_STOP_PREVIEW_SECONDS=0.5 \
      ACTION_BIAS_MODE=none ACTION_BIAS=0 ANKLE_ROLL_COMMON_BIAS=0 \
      LEFT_HIP_YAW_BIAS=0 RIGHT_HIP_YAW_BIAS=0 YAW_ACTION_GAIN=0 \
      LATERAL_POSITION_GAIN=0 LATERAL_VELOCITY_GAIN=0 HEADING_GAIN=0 \
      CROSS_TRACK_HEADING_GAIN=0 STATE_PREDICTION_SECONDS=0 STATE_QOS_DEPTH=10 \
      ACTION_EMA_ALPHA=1 WAIST_TILT_ACTION_MULTIPLIER=1 \
      PD_PROFILE=official_kp_ankle PD_KP_MULTIPLIER=1.2 PD_KD_MULTIPLIER=1.2 \
      MAX_ATTEMPTS=2 \
      bash "$SCRIPT_DIR/run_official_gate_case.sh" || true
  done
done

python3 - "$RESULT_ROOT" "$SUMMARY_PATH" <<'PY'
import json, math, pathlib, statistics, sys

root = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
groups = {}
expected_counts = {"main": 360, "stationary": 100, "recovery": 300}
scales = [0.4,0.4,0.4,0.4,0.12,0.08,0.4,0.4,0.4,0.4,0.12,0.08,0.4,0.16,0.16]
defaults = [-0.248,0,0,0.5303,-0.2823,0,-0.248,0,0,0.5303,-0.2823,0,0,0,0]

def l2(values):
    return math.sqrt(sum(value * value for value in values))

def first(rows, predicate):
    for row in rows:
        relative = float(row["elapsed_s"]) - 2.0
        if relative >= -1e-9 and predicate(row):
            return max(0.0, relative)
    return None

def episode(label, repeat):
    path = root / f"phase13_source_recovery_{label}_stiff1p2_fixed_r{repeat}.json"
    if not path.exists():
        return {"status":"missing", "path":str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    stop = sorted((r for r in payload["trace"] if r["stage"] == "stop"), key=lambda r:r["elapsed_s"])
    pre = max((r for r in stop if r["elapsed_s"] < 2.0), key=lambda r:r["elapsed_s"])
    post = min((r for r in stop if r["elapsed_s"] >= 2.0), key=lambda r:r["elapsed_s"])
    actor_delta = [b-a for a,b in zip(pre["action"], post["action"])]
    proposal = post["unblended_policy_action"] or post["action"]
    proposal_delta = [b-a for a,b in zip(pre["action"], proposal)]
    physical_delta = [b-a for a,b in zip(pre["physical_lower_target_rad"], post["physical_lower_target_rad"])]
    blend_rows = [r for r in stop if 1.98 <= r["elapsed_s"] <= 2.5 + 1e-9]
    physical_steps = [
        l2([b-a for a,b in zip(left["physical_lower_target_rad"], right["physical_lower_target_rad"])])
        for left,right in zip(blend_rows, blend_rows[1:])
    ]
    endpoint = min((r for r in stop if r["elapsed_s"] >= 2.5 - 1e-9), key=lambda r:r["elapsed_s"])
    expected_endpoint = [default + action*scale for default,action,scale in zip(defaults,endpoint["action"],scales)]
    endpoint_error = max(abs(a-b) for a,b in zip(endpoint["physical_lower_target_rad"], expected_endpoint))
    recovery_rows = [r for r in stop if r["elapsed_s"] >= 2.0]
    history_errors = []
    for previous,current in zip(recovery_rows,recovery_rows[1:]):
        observed_previous = current["obs"][74:89]
        history_errors.append(max(abs(a-b) for a,b in zip(observed_previous,previous["action"])))
    speed = lambda r: math.hypot(r["root_vx_w_mps"], r["root_vy_w_mps"])
    return {
        "status":"pass" if summary.get("full_gate_pass") else "fail",
        "path":str(path),
        "stand":bool(summary.get("stand_gate_pass")),
        "startup":bool(summary.get("startup_gate_pass")),
        "move":bool(summary.get("move_gate_pass")),
        "stop":bool(summary.get("stop_gate_pass")),
        "heading_max_rad":summary.get("move_heading_max_deviation_rad"),
        "lateral_m":summary.get("move_lateral_displacement_m"),
        "stop_drift_m":summary.get("stop_root_xy_drift_m"),
        "stop_settle_s":summary.get("stop_settle_time_s"),
        "slot_counts":summary.get("policy_slot_inference_counts"),
        "actor_handoff_delta_l2":l2(actor_delta),
        "unblended_policy_handoff_delta_l2":l2(proposal_delta),
        "actor_scaled_handoff_delta_l2_rad":l2([a*s for a,s in zip(actor_delta,scales)]),
        "physical_target_handoff_delta_l2_rad":l2(physical_delta),
        "physical_target_max_step_l2_rad_through_0p5s":max(physical_steps),
        "physical_target_endpoint_error_linf_rad":endpoint_error,
        "next_previous_action_sync_linf_max":max(history_errors),
        "tilt_violation_s_after_handoff":first(stop,lambda r:r["root_tilt_rad"]>0.30),
        "height_violation_s_after_handoff":first(stop,lambda r:r["root_z_m"]<0.45),
        "speed_reference_s_after_handoff":first(stop,lambda r:speed(r)>0.03),
        "stop_pitch_mean_rad":summary.get("stop_root_pitch_mean_rad"),
    }

def aggregate(rows):
    keys=("actor_handoff_delta_l2","unblended_policy_handoff_delta_l2","actor_scaled_handoff_delta_l2_rad",
          "physical_target_handoff_delta_l2_rad","physical_target_max_step_l2_rad_through_0p5s",
          "physical_target_endpoint_error_linf_rad","next_previous_action_sync_linf_max","tilt_violation_s_after_handoff",
          "height_violation_s_after_handoff","heading_max_rad","lateral_m","stop_drift_m",
          "stop_settle_s","stop_pitch_mean_rad")
    return {
        "valid":sum(r["status"] in {"pass","fail"} for r in rows),
        "full":sum(r["status"]=="pass" for r in rows),
        "subgates":{k:sum(bool(r.get(k)) for r in rows) for k in ("stand","startup","move","stop")},
        "metrics":{
            key:{"median":statistics.median([r[key] for r in rows]),
                 "population_variance":statistics.pvariance([r[key] for r in rows]),
                 "min":min(r[key] for r in rows),
                 "max":max(r[key] for r in rows)}
            for key in keys
        },
    }

for label,seconds in (("blend0p0",0.0),("blend0p5",0.5)):
    rows=[episode(label,r) for r in range(1,6)]
    groups[label]={"blend_seconds":seconds,"rows":rows,"aggregate":aggregate(rows)}

all_rows=[r for g in groups.values() for r in g["rows"]]
result={
    "stage":"BASE Phase13 handoff continuity A/B",
    "single_variable":"curriculum_recovery_handoff_blend_seconds 0.0 vs 0.5",
    "groups":groups,
    "decision":{
        "all_10_valid":all(r["status"] in {"pass","fail"} for r in all_rows),
        "all_slot_counts_exact":all(r.get("slot_counts")==expected_counts for r in all_rows),
        "candidate_physical_first_tick_continuous":groups["blend0p5"]["aggregate"]["metrics"]["physical_target_handoff_delta_l2_rad"]["median"] < 1e-9,
        "candidate_history_synchronized":groups["blend0p5"]["aggregate"]["metrics"]["next_previous_action_sync_linf_max"]["max"] < 1e-6,
        "candidate_improves_full_gate":groups["blend0p5"]["aggregate"]["full"] > groups["blend0p0"]["aggregate"]["full"],
        "training_unlocked":False,
    },
}
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))
PY
