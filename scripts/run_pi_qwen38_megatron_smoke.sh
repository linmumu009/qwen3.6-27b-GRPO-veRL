#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-/models/Qwen3.6-27B}"
DATA_FILE="${DATA_FILE:-${PROJECT_ROOT}/data/pi_verified_smoke.parquet}"
RUN_NAME="${RUN_NAME:-llin-qwen38-27b-megatron-grpo-one-step-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}"

mkdir -p "${OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/scripts/check_qwen38_smoke_ray_cluster.py" \
  --address "${RAY_ADDRESS}" \
  > "${OUTPUT_DIR}/ray_cluster.safe.json"
python3 "${PROJECT_ROOT}/scripts/check_qwen38_model_compat.py" \
  --reference-model "${REFERENCE_MODEL_PATH}" \
  --candidate-model "${MODEL_PATH}" \
  --output "${OUTPUT_DIR}/qwen38_compat.safe.json"

cat > "${OUTPUT_DIR}/engineering_smoke_contract.txt" <<EOF
contract=llin-qwen38-27b-megatron-grpo-engineering-smoke-v1
initialization=qwen38_hf_base
qwen36_step120_checkpoint_reused=false
trainer_npus=16
trainer_topology=tp4_pp2_cp2
rollout_npus=16
rollout_topology=tp8_dp2
optimizer_steps=1
checkpoint_saved=false
EOF

MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${DATA_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=8 ROLLOUT_NPUS=16 \
TOTAL_TRAINING_STEPS=1 SAVE_FREQ=-1 \
OPTIMIZER_CPU_OFFLOAD=false ENGINE_OPTIMIZER_OFFLOAD=false \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_megatron_tp4_pp2_cp2.sh"
