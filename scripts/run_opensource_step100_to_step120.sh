#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
export PYTHONPATH="${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:${PYTHONPATH:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/step120_opensource_20260824}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/opensource_step120_train.parquet}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-v15-dwh-bossreward-12groups-100step-20260805-03/checkpoints/global_step_100}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${PROJECT_ROOT}/runs/resume-views/llin-step100-opensource/global_step_100}"
RUN_NAME="${RUN_NAME:-llin-step120-opensource-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

START_POLICY_STEP=100
FINAL_POLICY_STEP=120
TOTAL_TRAINING_STEPS=120
GROUPS_PER_STEP=4
TOTAL_ROLLOUT_GROUPS=480
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
if [[ ! -f "${TRAIN_FILE}" ]]; then
  printf 'Step-120-open-source training file not found: %s\n' "${TRAIN_FILE}" >&2
  exit 2
fi
if [[ ! -f "${DATA_DIR}/opensource_step120_quality_report.json" ]]; then
  printf 'Step-120-open-source quality report not found: %s\n' "${DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RESUME_CHECKPOINT}/actor/ckpt_contents.json" ]]; then
  printf 'Prepared Step-100 resume checkpoint not found: %s\n' "${RESUME_CHECKPOINT}" >&2
  exit 2
fi

python3 - "${TRAIN_FILE}" <<'PY'
import sys
from datasets import Dataset

path = sys.argv[1]
dataset = Dataset.from_parquet(path)
if len(dataset) != 80:
    raise SystemExit(f"expected 80 open-source training prompts, got {len(dataset)}")
counts = {}
for name in dataset["ability"]:
    counts[name] = counts.get(name, 0) + 1
expected = {"MATH": 48, "PHYBench": 16, "C-Eval-dev": 8, "GSM8K": 8}
if counts != expected:
    raise SystemExit(f"unexpected open-source mixture: {counts!r}")
PY

mkdir -p "${OUTPUT_DIR}"
python3 "${PROJECT_ROOT}/scripts/verify_checkpoint_integrity.py" \
  --checkpoint-dir "${SOURCE_CHECKPOINT}" \
  --base-model-dir "${BASE_MODEL_DIR:-/models/Qwen3.6-27B}" \
  --output "${OUTPUT_DIR}/source_checkpoint_integrity.json"

cat > "${OUTPUT_DIR}/resume_contract.txt" <<EOF
run_contract=step120-opensource-recovery-v1
source_policy_step=${START_POLICY_STEP}
final_policy_step=${FINAL_POLICY_STEP}
new_optimizer_updates=$((FINAL_POLICY_STEP - START_POLICY_STEP))
new_rollout_groups=80
responses_per_group=${RESPONSES_PER_GROUP}
source_checkpoint=${SOURCE_CHECKPOINT}
resume_checkpoint_view=${RESUME_CHECKPOINT}
checkpoint_load_contents=model,extra
optimizer_state=reset_missing_from_source
dataloader_state=reset_for_opensource80
reward=strict_binary_final_answer
multi_turn_tools=disabled
EOF
cp "${DATA_DIR}/opensource_step120_quality_report.json" "${OUTPUT_DIR}/data_quality_report.json"

RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_FILE="${TRAIN_FILE}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
SAVE_FREQ="${FINAL_POLICY_STEP}" \
FASTEST_K=4 \
OVERSAMPLE_CANDIDATES=4 \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD="${STALENESS_THRESHOLD}" \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=45056 \
MAX_ASSISTANT_TURNS=1 \
MAX_USER_TURNS=1 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS}" \
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS}" \
AGENT_WORKERS="${AGENT_WORKERS}" \
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.megatron.optimizer_offload=False \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,extra]' \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/opensource_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq="${FINAL_POLICY_STEP}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${RESUME_CHECKPOINT}" \
  trainer.del_local_ckpt_after_load=False \
  async_training.use_trainer_do_validate=False \
  "$@"
