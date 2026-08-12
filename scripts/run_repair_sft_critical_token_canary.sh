#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_critical_token_20260812}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/critical_token_repair_sft_train.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/contract.json}"
TOKENIZATION_GATE="${TOKENIZATION_GATE:-${DATA_DIR}/critical_token_mask_gate.json}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-critical-token-step120-1step-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
CRITICAL_TOKEN_WEIGHT="${CRITICAL_TOKEN_WEIGHT:-32.0}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python3 -m scripts.check_critical_token_sft_dataset \
  --data-file "${TRAIN_FILE}" \
  --model-path "${MODEL_PATH}" \
  --max-length 8192 \
  --critical-token-weight "${CRITICAL_TOKEN_WEIGHT}" \
  --output "${TOKENIZATION_GATE}"

cat > "${OUTPUT_DIR}/causal_canary_contract.txt" <<EOF
purpose=repair_sft_semantic_critical_token_canary
source_checkpoint=step120
same_state_conditioned_rows=16
base_weights=0.25_8_1
only_causal_change=first_semantic_nongreedy_sql_token_weight_8_to_${CRITICAL_TOKEN_WEIGHT}
semantic_query_plan_or_aggregation_tasks=12_of_16
training_steps=1
checkpoint_policy=final_model_and_extra_only
promotion_allowed=false
next_gate=semantic_critical_token_rank_then_full_correction_sql_probability
EOF

PROJECT_ROOT="${PROJECT_ROOT}" \
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
DATA_CONTRACT="${DATA_CONTRACT}" \
TOKENIZATION_GATE="${TOKENIZATION_GATE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TOTAL_STEPS=1 \
TOTAL_EPOCHS=1 \
MAX_LENGTH=8192 \
SFT_RECIPE=semantic_critical_token_recovery \
DATASET_PATH="${PROJECT_ROOT}/scripts/qwen36_critical_token_sft_dataset.py" \
DATASET_NAME=Qwen36CriticalTokenSFTDataset \
TOOL_STRUCTURE_WEIGHT=0.25 \
SQL_PAYLOAD_WEIGHT=8.0 \
FINAL_ANSWER_WEIGHT=1.0 \
CRITICAL_TOKEN_WEIGHT="${CRITICAL_TOKEN_WEIGHT}" \
bash "${PROJECT_ROOT}/scripts/run_repair_sft_train236_overfit.sh" \
  "+data.critical_token_weight=${CRITICAL_TOKEN_WEIGHT}" \
  "$@"
