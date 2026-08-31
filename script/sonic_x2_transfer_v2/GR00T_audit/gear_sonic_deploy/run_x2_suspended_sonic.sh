#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$SCRIPT_DIR/runtime_suspended"
MODEL="$SCRIPT_DIR/../../models/x2_sonic_frozen_g1core_lora_v2.onnx"
MOTION="$SCRIPT_DIR/../../motions/x2_idle_stand.x2m2"
AIMDK_PREFIX="${HOME}/aimdk_ws_0_8_18/install/aimdk_msgs"
CONFIRM_TOKEN="X2_SUSPENDED_SONIC_START"
GROUND_LOAD_TEST_ONLY=false
SUPPORTED_POLICY_PROBE=false
MAX_TARGET_DEV=0.05
TARGET_LPF_HZ=3.0
WRITER_HZ="${X2_SONIC_WRITER_HZ:-250}"
SUPPORTED_POLICY_ARGS=()
DEPLOY_TUNING_ARGS=()
SUPPORTED_ENTRY_ANKLE_PITCH_ERROR_MAX_RAD=0.15
# Live-reference (garment -> localhost ZMQ v5) transport + safety flags.
# Populated ONLY by the --supported-neutral-garment-zmq mode; empty for every
# gantry/static mode so the exec below is byte-identical to neutral_damped
# unless the garment path is explicitly requested.
ZMQ_ARGS=()
# Workspace override for validating a scratch colcon install (e.g. the
# install_gatecheck build) WITHOUT touching the live install space. Point
# X2_ONBOT_WS at a directory containing install/setup.bash; deploy_x2.sh only
# ever reads "$ONBOT_WS/install/setup.bash" and "$ONBOT_WS/src".
ONBOT_WS_DIR="${X2_ONBOT_WS:-$RUNTIME/ws}"
if [[ ! -f "$ONBOT_WS_DIR/install/setup.bash" ]]; then
    echo "X2_ONBOT_WS resolved to '$ONBOT_WS_DIR' but $ONBOT_WS_DIR/install/setup.bash is missing" >&2
    exit 2
fi
if [[ "$ONBOT_WS_DIR" != "$RUNTIME/ws" ]]; then
    echo "NOTE: using overridden workspace $ONBOT_WS_DIR (live install space untouched)" >&2
fi

if [[ ! "$WRITER_HZ" =~ ^[0-9]+$ ]] || (( WRITER_HZ < 100 || WRITER_HZ > 500 )); then
    echo "X2_SONIC_WRITER_HZ must be an integer in [100, 500]" >&2
    exit 2
fi

if [[ "${1:-}" != "$CONFIRM_TOKEN" ]]; then
    cat <<EOF
Usage:
  $0 $CONFIRM_TOKEN

Sequence:
  suspended -> stop MC -> verify silence -> capture current pose
  -> PD gain ramp -> Sonic default-pose ramp -> wait for 'go'

This route never requests official Standing. In supported-policy modes,
'policy' and 'stop' may be repeated while custom static PD remains active.
Only type 'lifted' after re-suspending the robot to exit and restore MC.
EOF
    exit 2
fi

case "${2:-}" in
    "") ;;
    --ground-load-test-only) GROUND_LOAD_TEST_ONLY=true ;;
    --supported-policy-probe) SUPPORTED_POLICY_PROBE=true ;;
    --supported-stand-stage1)
        SUPPORTED_POLICY_PROBE=true
        MAX_TARGET_DEV=0.10
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-seconds 60.0
            --supported-policy-ramp-seconds 3.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.08
            --supported-policy-max-dev-waist 0.06
            --supported-policy-max-dev-arm 0.10
            --supported-policy-max-dev-head 0.05
            --supported-policy-tilt-delta-deg 2.0
            --supported-policy-abs-tilt-deg 20.0
            --supported-policy-joint-vel-trip 0.5
            --supported-policy-target-rate 0.15
        )
        ;;
    --supported-stand-stage2)
        SUPPORTED_POLICY_PROBE=true
        MAX_TARGET_DEV=0.10
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-seconds 60.0
            --supported-policy-ramp-seconds 3.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.10
            --supported-policy-max-dev-waist 0.08
            --supported-policy-max-dev-arm 0.10
            --supported-policy-max-dev-head 0.05
            --supported-policy-tilt-delta-deg 2.0
            --supported-policy-abs-tilt-deg 20.0
            --supported-policy-joint-vel-trip 0.5
            --supported-policy-target-rate 0.20
        )
        ;;
    --supported-stand-stage3)
        SUPPORTED_POLICY_PROBE=true
        MAX_TARGET_DEV=2.0
        TARGET_LPF_HZ=0.0
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-seconds 15.0
            --supported-policy-ramp-seconds 3.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.25
            --supported-policy-max-dev-waist 0.08
            --supported-policy-max-dev-arm 0.10
            --supported-policy-max-dev-head 0.05
            --supported-policy-tilt-delta-deg 5.0
            --supported-policy-abs-tilt-deg 25.0
            --supported-policy-joint-vel-trip 0.8
            --supported-policy-target-rate 0.20
        )
        ;;
    --supported-stand-stage4)
        SUPPORTED_POLICY_PROBE=true
        MAX_TARGET_DEV=2.0
        TARGET_LPF_HZ=0.0
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-anchor-default
            --supported-policy-seconds 30.0
            --supported-policy-ramp-seconds 3.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.60
            --supported-policy-max-dev-waist 0.20
            --supported-policy-max-dev-arm 0.25
            --supported-policy-max-dev-head 0.08
            --supported-policy-tilt-delta-deg 5.0
            --supported-policy-abs-tilt-deg 25.0
            --supported-policy-joint-vel-trip 0.8
            --supported-policy-target-rate 0.20
        )
        ;;
    --supported-neutral-stand|--supported-neutral-long)
        SUPPORTED_POLICY_PROBE=true
        # Stage5 removes the pitched x2_idle_stand reference entirely. Keep
        # the envelope centred on the trained default pose so ankle and waist
        # corrections are not clipped around a gantry-dependent captured pose.
        MOTION=""
        MAX_TARGET_DEV=2.0
        TARGET_LPF_HZ=0.0
        if [[ "${2}" == "--supported-neutral-long" ]]; then
            NEUTRAL_POLICY_SECONDS=300.0
        else
            NEUTRAL_POLICY_SECONDS=30.0
        fi
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-anchor-default
            --supported-policy-seconds "$NEUTRAL_POLICY_SECONDS"
            --supported-policy-ramp-seconds 3.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.60
            --supported-policy-max-dev-waist 0.20
            --supported-policy-max-dev-arm 0.25
            --supported-policy-max-dev-head 0.08
            --supported-policy-tilt-delta-deg 5.0
            --supported-policy-abs-tilt-deg 25.0
            --supported-policy-joint-vel-trip 0.8
            --supported-policy-target-rate 0.20
        )
        ;;
    --supported-upright-idle)
        SUPPORTED_POLICY_PROBE=true
        MOTION="$SCRIPT_DIR/../../motions/x2_idle_stand_upright.x2m2"
        MAX_TARGET_DEV=2.0
        TARGET_LPF_HZ=8.0
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-seconds 300.0
            --supported-policy-ramp-seconds 3.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.60
            --supported-policy-max-dev-waist 0.20
            --supported-policy-max-dev-arm 0.25
            --supported-policy-max-dev-head 0.08
            --supported-policy-tilt-delta-deg 5.0
            --supported-policy-abs-tilt-deg 25.0
            --supported-policy-joint-vel-trip 0.8
            --supported-policy-target-rate 0.20
        )
        ;;
    --supported-neutral-damped|--supported-neutral-damped-30|--supported-neutral-damped-relative-30|--supported-neutral-damped-relative-100|--supported-neutral-damped-relative-300|--supported-neutral-damped-relative-support-step|--supported-neutral-damped-anklep40-relative-30|--supported-neutral-damped-anklep40-entry135-relative-30|--supported-neutral-damped-slew030|--supported-neutral-garment-zmq)
        SUPPORTED_POLICY_PROBE=true
        MOTION=""
        MAX_TARGET_DEV=2.0
        TARGET_LPF_HZ=8.0
        DEPLOY_TUNING_ARGS+=(
            --target-lpf-hz-waist 2.5
            --kd-scale-ankle-pitch 3.31
            --kd-scale-ankle-roll 2.20
            --kd-scale-waist-pitch 3.0
        )
        if [[ "${2}" == "--supported-neutral-damped-anklep40-relative-30" \
              || "${2}" == "--supported-neutral-damped-anklep40-entry135-relative-30" ]]; then
            # The shared ankle scale yields pitch kp=32.064. This additional
            # factor matches the official X2 MC ankle-pitch kp=40 while
            # leaving ankle roll, waist, damping, filtering and slew unchanged.
            DEPLOY_TUNING_ARGS+=(--kp-scale-ankle-pitch 1.2475)
        fi
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-anchor-default
            --supported-policy-seconds "$(
                if [[ "${2}" == "--supported-neutral-damped-30" \
                      || "${2}" == "--supported-neutral-damped-relative-30" \
                      || "${2}" == "--supported-neutral-damped-anklep40-relative-30" \
                      || "${2}" == "--supported-neutral-damped-anklep40-entry135-relative-30" ]]; then
                    printf '30.0'
                elif [[ "${2}" == "--supported-neutral-damped-relative-support-step" ]]; then
                    printf '60.0'
                elif [[ "${2}" == "--supported-neutral-damped-relative-100" ]]; then
                    printf '100.0'
                elif [[ "${2}" == "--supported-neutral-damped-relative-300" ]]; then
                    printf '300.0'
                else
                    printf '5.0'
                fi
            )"
            --supported-policy-ramp-seconds 4.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.60
            --supported-policy-max-dev-waist 0.20
            --supported-policy-max-dev-arm 0.25
            --supported-policy-max-dev-head 0.08
            --supported-policy-tilt-delta-deg 5.0
            --supported-policy-abs-tilt-deg 25.0
            --supported-policy-joint-vel-trip 0.8
            --supported-policy-target-rate "$(
                if [[ "${2}" == "--supported-neutral-damped-slew030" ]]; then
                    printf '0.30'
                else
                    printf '0.12'
                fi
            )"
        )
        if [[ "${2}" == "--supported-neutral-damped-relative-30" \
              || "${2}" == "--supported-neutral-damped-relative-100" \
              || "${2}" == "--supported-neutral-damped-anklep40-relative-30" \
              || "${2}" == "--supported-neutral-damped-anklep40-entry135-relative-30" \
              || "${2}" == "--supported-neutral-damped-relative-support-step" \
              || "${2}" == "--supported-neutral-damped-relative-300" ]]; then
            if [[ "${2}" == "--supported-neutral-damped-anklep40-entry135-relative-30" ]]; then
                SUPPORTED_ENTRY_ANKLE_PITCH_ERROR_MAX_RAD=0.135
            fi
            SUPPORTED_POLICY_ARGS+=(
                --deploy-extra-arg --supported-policy-entry-relative-to-load
                --deploy-extra-arg --supported-policy-entry-pitch-min-deg
                --deploy-extra-arg -3.0
                --deploy-extra-arg --supported-policy-entry-pitch-max-deg
                --deploy-extra-arg 3.0
                --deploy-extra-arg --supported-policy-entry-roll-abs-max-deg
                --deploy-extra-arg 2.5
                --deploy-extra-arg --supported-policy-entry-ankle-pitch-error-max-rad
                --deploy-extra-arg "$SUPPORTED_ENTRY_ANKLE_PITCH_ERROR_MAX_RAD"
                --deploy-extra-arg --supported-policy-entry-stable-seconds
                --deploy-extra-arg 2.0
            )
        else
            SUPPORTED_POLICY_ARGS+=(
                --deploy-extra-arg --supported-policy-entry-pitch-min-deg
                --deploy-extra-arg 5.0
                --deploy-extra-arg --supported-policy-entry-pitch-max-deg
                --deploy-extra-arg 9.0
                --deploy-extra-arg --supported-policy-entry-roll-abs-max-deg
                --deploy-extra-arg 2.0
                --deploy-extra-arg --supported-policy-entry-stable-seconds
                --deploy-extra-arg 2.0
            )
        fi
        if [[ "${2}" == "--supported-neutral-damped-relative-support-step" ]]; then
            # Experimental orchestration only. The controller remains the
            # exact neutral_damped 0.12 rad/s parent above. 'support_step'
            # timestamps one manual hoist adjustment after a 20 s immutable
            # support interval and five continuously gated seconds, then the
            # existing bounded return runs after a 20 s fixed plateau.
            SUPPORTED_POLICY_ARGS+=(
                --deploy-extra-arg --support-step-probe
                --deploy-extra-arg --support-step-min-policy-seconds
                --deploy-extra-arg 20.0
                --deploy-extra-arg --support-step-plateau-seconds
                --deploy-extra-arg 20.0
                --deploy-extra-arg --support-step-pre-stable-seconds
                --deploy-extra-arg 5.0
                --deploy-extra-arg --support-step-max-ankle-pitch-error-rad
                --deploy-extra-arg 0.20
                --deploy-extra-arg --support-step-max-waist-pitch-error-rad
                # The upper-body sling imposes a repeatable waist preload.
                # Keep logging it, but use ankles/base dynamics for the
                # support-transfer decision instead of requiring zero load.
                --deploy-extra-arg -1.0
                --deploy-extra-arg --support-step-max-joint-velocity-rad-s
                --deploy-extra-arg 0.05
                --deploy-extra-arg --support-step-max-tilt-deg
                --deploy-extra-arg 3.0
                --deploy-extra-arg --support-step-tilt-trend-window-seconds
                --deploy-extra-arg 1.0
                --deploy-extra-arg --support-step-max-tilt-rise-deg
                --deploy-extra-arg 0.10
            )
        fi
        # Garment path ONLY: swap StandStillReference for the live localhost
        # ZMQ v5 pose stream and arm the powered live-reference safety gates.
        # Everything above is shared verbatim with --supported-neutral-damped,
        # so the sole difference on this mode is the reference source + gates.
        if [[ "${2}" == "--supported-neutral-garment-zmq" ]]; then
            ZMQ_ARGS+=(
                # Transport: garment publisher -> localhost ZMQ v5 :5556.
                --vla
                --vla-zmq-host 127.0.0.1
                --vla-zmq-port 5556
                --vla-zmq-topic pose
                # Powered live-reference safety gates (forwarded raw to the
                # deploy binary; deploy_x2.sh does not model these knobs).
                --deploy-extra-arg --pose-ref-stale-s
                --deploy-extra-arg 0.5
                --deploy-extra-arg --strict-reference-velocity
                --deploy-extra-arg --zmq-warmup-body-frames
                --deploy-extra-arg 40
                --deploy-extra-arg --zmq-entry-max-age-s
                --deploy-extra-arg 0.5
                --deploy-extra-arg --zmq-entry-min-fresh-s
                --deploy-extra-arg 0.8
            )
        fi
        ;;
    --supported-neutral-waist2|--supported-neutral-waist2-anklemc)
        SUPPORTED_POLICY_PROBE=true
        MOTION=""
        MAX_TARGET_DEV=2.0
        TARGET_LPF_HZ=8.0
        DEPLOY_TUNING_ARGS+=(
            --target-lpf-hz-waist 2.5
            --kp-scale-waist-pr 2.0
            --kd-scale-ankle-pitch 3.31
            --kd-scale-ankle-roll 2.20
            --kd-scale-waist-pitch 4.24
            --kd-scale-waist-roll 1.41
        )
        if [[ "${2}" == "--supported-neutral-waist2-anklemc" ]]; then
            # The shared ankle scale below gives pitch kp=32.064. Match the
            # official X2 MC sagittal ankle gain (kp=40, kd=3) exactly; the
            # existing kd_scale_ankle_pitch already yields kd=3.
            DEPLOY_TUNING_ARGS+=(--kp-scale-ankle-pitch 1.2475)
        fi
        SUPPORTED_POLICY_ARGS+=(
            --supported-policy-anchor-default
            --supported-policy-seconds 300.0
            --supported-policy-ramp-seconds 4.0
            --supported-policy-return-seconds 2.0
            --supported-policy-max-dev-leg 0.60
            --supported-policy-max-dev-waist 0.20
            --supported-policy-max-dev-arm 0.25
            --supported-policy-max-dev-head 0.08
            --supported-policy-tilt-delta-deg 5.0
            --supported-policy-abs-tilt-deg 25.0
            --supported-policy-joint-vel-trip 0.8
            --supported-policy-target-rate 0.12
        )
        ;;
    *) echo "Unknown mode: ${2}" >&2; exit 2 ;;
esac

GROUND_LOAD_ARGS=()
if $GROUND_LOAD_TEST_ONLY; then
    GROUND_LOAD_ARGS+=(--ground-load-test-only)
fi
if $SUPPORTED_POLICY_PROBE; then
    GROUND_LOAD_ARGS+=(--supported-policy-probe)
fi

REQUIRED=(
    "$SCRIPT_DIR/deploy_x2.sh" \
    "$RUNTIME/ws/install/setup.bash" \
    "$RUNTIME/venv/bin/python3" \
    "$RUNTIME/onnxruntime/lib/libonnxruntime.so" \
    "$AIMDK_PREFIX/local/lib/python3.10/dist-packages" \
    "$MODEL"
)
if [[ -n "$MOTION" ]]; then
    REQUIRED+=("$MOTION")
fi
for required in "${REQUIRED[@]}"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required runtime asset: $required" >&2
        exit 1
    fi
done

MOTION_ARGS=()
if [[ -n "$MOTION" ]]; then
    MOTION_ARGS+=(--motion "$MOTION")
fi

DEBUG_ARGS=()
if [[ -n "${X2_OBS_DUMP_PATH:-}" ]]; then
    DEBUG_ARGS+=(--obs-dump "$X2_OBS_DUMP_PATH")
fi

MC_HANDOFF_ARGS=(--pause-mc-worker)
if [[ "${X2_ADOPT_PAUSED_MC:-0}" == "1" ]]; then
    MC_HANDOFF_ARGS=(
        --adopt-paused-mc
        --vla-debug-port "${X2_ADOPT_DEBUG_PORT:-5559}"
    )
    echo "[x2-sonic] adopting an already-paused MC worker; cleanup will leave MC paused"
fi

RUN_LOG="$RUNTIME/logs/suspended_sonic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_LOG"
echo "[x2-sonic] HAL writer=${WRITER_HZ}Hz; runtime log=$RUN_LOG"

exec "$SCRIPT_DIR/deploy_x2.sh" onbot \
    --model "$MODEL" \
    "${MOTION_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    --suspended-start \
    "${MC_HANDOFF_ARGS[@]}" \
    "${GROUND_LOAD_ARGS[@]}" \
    "${SUPPORTED_POLICY_ARGS[@]}" \
    "${ZMQ_ARGS[@]}" \
    --pd-acquire-seconds 3.0 \
    --default-pose-seconds 5.0 \
    --default-pose-max-rate 0.15 \
    --writer-hz "$WRITER_HZ" \
    --max-target-dev "$MAX_TARGET_DEV" \
    --target-lpf-hz "$TARGET_LPF_HZ" \
    "${DEPLOY_TUNING_ARGS[@]}" \
    --mc-mode-poll-s 0 \
    --wrist-bypass freeze \
    --kp-scale-ankle 1.5 \
    --ramp-seconds 3.0 \
    --hold-for-mc-timeout-s 0 \
    --no-wire-probe \
    --no-hand-bridge \
    --log-dir "$RUN_LOG" \
    --onbot-prefix "$RUNTIME" \
    --onbot-ws "$ONBOT_WS_DIR" \
    --onbot-venv "$RUNTIME/venv" \
    --onbot-onnxruntime "$RUNTIME/onnxruntime" \
    --onbot-aimdk-prefix "$AIMDK_PREFIX"
