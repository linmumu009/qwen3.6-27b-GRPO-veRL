#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_train236_20260811}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/repair_sft_train.parquet}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-sql-weighted-step120-1step-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TOOL_STRUCTURE_WEIGHT="${TOOL_STRUCTURE_WEIGHT:-0.25}"
SQL_PAYLOAD_WEIGHT="${SQL_PAYLOAD_WEIGHT:-8.0}"
FINAL_ANSWER_WEIGHT="${FINAL_ANSWER_WEIGHT:-1.0}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python3 -m scripts.check_sql_weighted_sft_dataset \
  --data-file "${TRAIN_FILE}" \
  --model-path "${MODEL_PATH}" \
  --max-length 2048 \
  --tool-structure-weight "${TOOL_STRUCTURE_WEIGHT}" \
  --sql-payload-weight "${SQL_PAYLOAD_WEIGHT}" \
  --final-answer-weight "${FINAL_ANSWER_WEIGHT}" \
  --output "${OUTPUT_DIR}/sql_weighted_mask_gate.json"

cat > "${OUTPUT_DIR}/causal_canary_contract.txt" <<EOF
purpose=repair_sft_sql_weighted_one_variable_canary
source_checkpoint=step120
semantic_gate_verified_first_query_support=0_of_16
intervention=sql_payload_weight_only
model_state_correction_examples=0
tool_structure_weight=${TOOL_STRUCTURE_WEIGHT}
sql_payload_weight=${SQL_PAYLOAD_WEIGHT}
final_answer_weight=${FINAL_ANSWER_WEIGHT}
training_steps=1
checkpoint_policy=final_model_and_extra_only
promotion_allowed=false
next_gate=forward_only_sql_token_rank_then_first_query_semantic_replay
EOF

PROJECT_ROOT="${PROJECT_ROOT}" \
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TOTAL_STEPS=1 \
TOTAL_EPOCHS=1 \
SFT_RECIPE=sql_payload_weight_only \
DATASET_PATH="${PROJECT_ROOT}/scripts/qwen36_sql_weighted_sft_dataset.py" \
DATASET_NAME=Qwen36SQLWeightedSFTDataset \
TOOL_STRUCTURE_WEIGHT="${TOOL_STRUCTURE_WEIGHT}" \
SQL_PAYLOAD_WEIGHT="${SQL_PAYLOAD_WEIGHT}" \
FINAL_ANSWER_WEIGHT="${FINAL_ANSWER_WEIGHT}" \
bash "${PROJECT_ROOT}/scripts/run_repair_sft_train236_overfit.sh" "$@"
