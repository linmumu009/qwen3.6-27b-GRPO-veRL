#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/semantic_plan_sufficiency_gate_20260812}"
STATE_PARQUET="${STATE_PARQUET:-${PROJECT_ROOT}/data/repair_sft_state_conditioned_20260812/state_conditioned_repair_sft_train.parquet}"
REPLAY_PARQUET="${REPLAY_PARQUET:-${PROJECT_ROOT}/data/repair_sft_train236_20260811/repair_sft_replay.parquet}"
SEMANTIC_AUDIT="${SEMANTIC_AUDIT:-${PROJECT_ROOT}/runs/llin-repair-sft-critical-token-step120-1step-20260812-01/state_recovery_semantics.json}"
DATABASE="${DATABASE:-/pi_sandbox/sft/20260628_v15/logistics.sqlite}"
GATE_FILE="${GATE_FILE:-${DATA_DIR}/semantic_plan_sufficiency_gate.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/contract.json}"
CPU_AUDIT="${CPU_AUDIT:-${DATA_DIR}/cpu_audit.json}"
RUN_NAME="${RUN_NAME:-llin-semantic-plan-sufficiency-step120-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

for path in "${MODEL_DIST_CKPT}/.metadata" "${STATE_PARQUET}" "${REPLAY_PARQUET}" "${SEMANTIC_AUDIT}" "${DATABASE}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'semantic-plan gate input missing: %s\n' "${path}" >&2
    exit 2
  fi
done

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:/verl:${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
python3 -m scripts.prepare_semantic_plan_sufficiency_gate \
  --state-parquet "${STATE_PARQUET}" \
  --replay-parquet "${REPLAY_PARQUET}" \
  --semantic-audit "${SEMANTIC_AUDIT}" \
  --database "${DATABASE}" \
  --output-dir "${DATA_DIR}"
python3 -m scripts.check_semantic_plan_sufficiency_gate \
  --data-file "${GATE_FILE}" \
  --contract "${DATA_CONTRACT}" \
  --output "${CPU_AUDIT}"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=semantic_plan_sufficiency_no_training_gate
source_checkpoint=step120
tasks=16
arms=control_operator_oracle_full_plan_oracle
rows=48
sampling=greedy_n1
max_assistant_turns=1
max_user_turns=1
generated_tool_execution=false_due_to_assistant_turn_limit_before_tool_parse
only_generated_tool=bash
optimizer_initialized=false
checkpoint_saved=false
promotion_allowed=false
EOF

export LLIN_VAL_ONLY_FORCE_DIST_SYNC=1
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_FILE="${GATE_FILE}" \
EVAL_FILE="${GATE_FILE}" \
MAX_ASSISTANT_TURNS=1 \
MAX_USER_TURNS=1 \
MAX_PROMPT_TOKENS=8192 \
MAX_RESPONSE_TOKENS=4096 \
MAX_CONTEXT_TOKENS=12288 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=16 \
bash "${PROJECT_ROOT}/scripts/run_pi_frozen_baseline.sh" \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  actor_rollout_ref.actor.megatron.dist_checkpointing_path="${MODEL_DIST_CKPT}" \
  'actor_rollout_ref.actor.checkpoint.load_contents=[]' \
  trainer.resume_mode=disable \
  trainer.val_only=True \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  data.filter_overlong_prompts=False \
  trainer.rollout_data_dir=null \
  trainer.save_freq=-1 \
  "$@"

validation_file="$(find "${OUTPUT_DIR}/validation" -maxdepth 1 -type f -name '*.jsonl' | sort | tail -1)"
if [[ -z "${validation_file}" || ! -f "${validation_file}" ]]; then
  printf 'semantic-plan validation JSONL missing under %s\n' "${OUTPUT_DIR}/validation" >&2
  exit 2
fi
python3 -m scripts.prepare_semantic_plan_gate_outputs \
  --validation "${validation_file}" \
  --output "${OUTPUT_DIR}/generated_one_turn.jsonl" \
  --summary-output "${OUTPUT_DIR}/output_adapter_summary.json"
python3 -m scripts.analyze_semantic_plan_sufficiency_gate \
  --replay-parquet "${REPLAY_PARQUET}" \
  --generated-jsonl "${OUTPUT_DIR}/generated_one_turn.jsonl" \
  --dataset-contract "${DATA_CONTRACT}" \
  --database "${DATABASE}" \
  --output "${OUTPUT_DIR}/semantic_plan_sufficiency_result.json"
