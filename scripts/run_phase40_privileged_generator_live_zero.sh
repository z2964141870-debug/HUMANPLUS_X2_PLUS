#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim"
STAGE152="${ROOT}/scripts/run_dcpeft_stage152.sh"
SEED_FILE="${ROOT}/research/dynamic_retargeting_20260811/phase36_native_generator_seed.npz"
SEED_SHA="4bd8bc428cc47fea3c5a783af9277babb45cc5cfdbdf2ce451628e81be5e7ad2"
MOTION="${ROOT}/artifacts/retarget/x2_phase45_kinematic_bronze_train.pkl"
OUTPUT="${PHASE40_OUTPUT:-${ROOT}/research/dynamic_retargeting_20260811/phase40_privileged_generator_live_zero.json}"

X2_PRIVILEGED_GENERATOR_NATIVE_RESET=true \
X2_PRIVILEGED_GENERATOR_SEED_PATH="${SEED_FILE}" \
X2_PRIVILEGED_GENERATOR_SEED_SHA256="${SEED_SHA}" \
X2_PRIVILEGED_GENERATOR_RESET_FRACTION=1.0 \
X2_PRIVILEGED_GENERATOR_SAMPLING=evenly_spaced \
X2_PRIVILEGED_GENERATOR_FIXED_FRAMES='[0]' \
X2_PRIVILEGED_GENERATOR_LIVE_ZERO_OUTPUT="${OUTPUT}" \
FAITHFUL_WBT29=true \
FAITHFUL_WBT29_ENV_TARGET=x2_privileged_generator_live_env.PrivilegedGeneratorTrackingEnvCfg \
FAITHFUL_WBT29_TRAINER_TARGET=x2_privileged_generator_live_zero.PrivilegedGeneratorLiveZeroTrainer \
TRAIN_AGENT_ENTRYPOINT="${ROOT}/scripts/train_agent_trl_privileged_generator.py" \
CHECKPOINT=/home/humanplus/humanoid-GPT/A/sonic_release/last.pt \
MOTION_FILE="${MOTION}" \
RUN_NAME=x2_privileged_generator_live_zero_phase40 \
RUN_KIND=contract_zero \
SEED=0 NUM_ENVS=1 STEPS_PER_ENV=2 ITERS=0 MOTIONS=3 \
ADAPTIVE_SAMPLING_ENABLED=false START_FROM_FIRST_FRAME=true \
FREEZE_FRAME_AUG=true CLEAN_EVAL=true CLEAN_COMMAND_INIT=true \
ENCODER_SAMPLE_PROBS='{g1:1.0,teleop:0.0,smpl:0.0}' \
X2_COLLISION_PROFILE=sole12 X2_PD_PROFILE=natural_frequency \
X2_ACTION_SCALE_PROFILE=torque_normalized ACTION_SCALE_MULT=1.0 \
SIM_DT=0.005 DECIMATION=4 ACTUATOR_RESPONSE_MODE=off \
TRAIN_BOUNDARY_KEYS='[]' SAVE_FREQUENCY=-1 SAVE_LAST_FREQUENCY=-1 \
TIMEOUT_SECONDS=900 bash "${STAGE152}"

# Isaac Kit can occasionally terminate after printing a Python traceback while
# its launcher still returns zero.  Treat the trainer's explicit report as the
# authoritative postcondition so a missing or failed live-zero can never be
# reported as success.
python3 - "${OUTPUT}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Phase40 report was not created: {path}")
report = json.loads(path.read_text(encoding="utf-8"))
decision = report.get("decision", {})
runtime = report.get("runtime", {})
if decision.get("result") != "PASS_LIVE_ZERO_ONLY":
    raise SystemExit(f"Phase40 did not pass: {decision}")
if runtime.get("optimizer_steps") != 0 or runtime.get("checkpoints_created") != 0:
    raise SystemExit(f"Phase40 zero-update boundary violated: {runtime}")
print(f"phase40_live_zero_verified={path}")
PY
