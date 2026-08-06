#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/boss_v15_dwh_full276_20260806/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/boss_pi_train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/boss_pi_val.parquet}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-v15-dwh-bossreward-12groups-100step-20260805-03/checkpoints/global_step_100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${PROJECT_ROOT}/runs/resume-views/llin-v15-step100-train236/global_step_100}"
RUN_NAME="${RUN_NAME:-llin-pi-formal-grpo-step100-to-step200-12groups-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

# Cumulative targets: veRL restores policy version 100 and rollout counter 401,
# so 200 policy steps and 800 groups yield exactly 100 new updates / 400 groups.
START_POLICY_STEP=100
FINAL_POLICY_STEP=200
TOTAL_TRAINING_STEPS="${FINAL_POLICY_STEP}"
GROUPS_PER_STEP=4
TOTAL_ROLLOUT_GROUPS="$((FINAL_POLICY_STEP * GROUPS_PER_STEP))"
FINAL_EVAL_STEP="${FINAL_POLICY_STEP}"
FINAL_SAVE_STEP="${FINAL_POLICY_STEP}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
PREWARM_GROUPS=8
MAX_QUEUE_GROUPS=8
STALENESS_THRESHOLD=2.0
TARGET_CONCURRENT_GROUPS=12
MAX_CONTEXT_TOKENS=49152
MAX_QUEUE_TOKENS="$((MAX_QUEUE_GROUPS * 4 * MAX_CONTEXT_TOKENS))"
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80
ROLLOUT_MAX_BATCHED_TOKENS=16384
ROLLOUT_MAX_SEQS=24
AGENT_WORKERS=12
WEIGHT_BUCKET_MB=2560
ROLLOUT_DP=2
RESPONSES_PER_GROUP=4

if (( ROLLOUT_DP * ROLLOUT_MAX_SEQS < TARGET_CONCURRENT_GROUPS * RESPONSES_PER_GROUP )); then
  printf 'Rollout capacity cannot hold %s groups x %s responses\n' \
    "${TARGET_CONCURRENT_GROUPS}" "${RESPONSES_PER_GROUP}" >&2
  exit 2
fi

python3 "${PROJECT_ROOT}/scripts/check_boss_alignment_contract.py" \
  --data-dir "${DATA_DIR}"

for path in "${TRAIN_FILE}" "${VAL_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Formal PI data file not found: %s\n' "${path}" >&2
    exit 2
  fi
done

if [[ ! -f "${RESUME_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
  printf 'Prepared resume actor checkpoint not found: %s\n' "${RESUME_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/scripts/verify_checkpoint_integrity.py" \
  --checkpoint-dir "${SOURCE_CHECKPOINT}" \
  --base-model-dir "${BASE_MODEL_DIR:-/models/Qwen3.6-27B}" \
  --output "${OUTPUT_DIR}/source_checkpoint_integrity.json"

cat > "${OUTPUT_DIR}/resume_contract.txt" <<EOF
source_policy_step=${START_POLICY_STEP}
final_policy_step=${FINAL_POLICY_STEP}
new_optimizer_updates=$((FINAL_POLICY_STEP - START_POLICY_STEP))
source_checkpoint=${SOURCE_CHECKPOINT}
resume_checkpoint_view=${RESUME_CHECKPOINT}
checkpoint_load_contents=model,extra
optimizer_state=reset_missing_from_source
dataloader_state=reset_for_corrected_train236
EOF

RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_FILE="${TRAIN_FILE}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
SAVE_FREQ="${FINAL_SAVE_STEP}" \
FASTEST_K=4 \
OVERSAMPLE_CANDIDATES=4 \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD="${STALENESS_THRESHOLD}" \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=45056 \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS}" \
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS}" \
AGENT_WORKERS="${AGENT_WORKERS}" \
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]' \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  trainer.val_before_train=False \
  trainer.test_freq="${FINAL_EVAL_STEP}" \
  trainer.log_val_generations=20 \
  trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
  trainer.save_freq="${FINAL_SAVE_STEP}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_CHECKPOINT}" \
  trainer.del_local_ckpt_after_load=False \
  async_training.use_trainer_do_validate=False \
  "$@"
