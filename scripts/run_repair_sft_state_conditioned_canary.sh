#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/repair_sft_state_conditioned_20260812}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/state_conditioned_repair_sft_train.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/contract.json}"
TOKENIZATION_GATE="${TOKENIZATION_GATE:-${DATA_DIR}/state_conditioned_mask_gate.json}"
RUN_NAME="${RUN_NAME:-llin-repair-sft-state-conditioned-step120-1step-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TOOL_STRUCTURE_WEIGHT="${TOOL_STRUCTURE_WEIGHT:-0.25}"
SQL_PAYLOAD_WEIGHT="${SQL_PAYLOAD_WEIGHT:-8.0}"
FINAL_ANSWER_WEIGHT="${FINAL_ANSWER_WEIGHT:-1.0}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python3 -m scripts.check_state_conditioned_sft_dataset \
  --data-file "${TRAIN_FILE}" \
  --model-path "${MODEL_PATH}" \
  --max-length 8192 \
  --tool-structure-weight "${TOOL_STRUCTURE_WEIGHT}" \
  --sql-payload-weight "${SQL_PAYLOAD_WEIGHT}" \
  --final-answer-weight "${FINAL_ANSWER_WEIGHT}" \
  --output "${TOKENIZATION_GATE}"

cat > "${OUTPUT_DIR}/causal_canary_contract.txt" <<EOF
purpose=repair_sft_state_conditioned_sql_recovery_canary
source_checkpoint=step120
same_tasks_as_sql_weighted_canary=16
same_target_weights_as_sql_weighted_canary=0.25_8_1
only_causal_change=step120_first_wrong_sql_and_observed_tool_result_as_zero_loss_context
error_context_assistant_loss_weight=0
supervised_assistant_turn_indices=1,2
training_steps=1
checkpoint_policy=final_model_and_extra_only
promotion_allowed=false
next_gate=state_conditioned_rank_and_direct_prompt_non_regression
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
SFT_RECIPE=state_conditioned_sql_recovery \
DATASET_PATH="${PROJECT_ROOT}/scripts/qwen36_sql_weighted_sft_dataset.py" \
DATASET_NAME=Qwen36SQLWeightedSFTDataset \
TOOL_STRUCTURE_WEIGHT="${TOOL_STRUCTURE_WEIGHT}" \
SQL_PAYLOAD_WEIGHT="${SQL_PAYLOAD_WEIGHT}" \
FINAL_ANSWER_WEIGHT="${FINAL_ANSWER_WEIGHT}" \
bash "${PROJECT_ROOT}/scripts/run_repair_sft_train236_overfit.sh" "$@"
