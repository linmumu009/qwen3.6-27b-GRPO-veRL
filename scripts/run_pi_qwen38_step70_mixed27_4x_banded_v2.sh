#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/exports/llin-qwen38-grpo-step70-hf-20260819-02}"
POOL_DIR="${POOL_DIR:-${PROJECT_ROOT}/runs/llin-qwen38-step70-mixed27-4x-20260820-01/data}"
CANONICAL_FILE="${CANONICAL_FILE:-${POOL_DIR}/train27.sensitive.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${POOL_DIR}/train27x4.sensitive.parquet}"
SEALED_FILE="${SEALED_FILE:-${POOL_DIR}/sealed6.sensitive.parquet}"
SAFE_SUMMARY="${SAFE_SUMMARY:-${POOL_DIR}/train27x4.safe.json}"
RUN_NAME="${RUN_NAME:-llin-qwen38-step70-mixed27-4x-banded-v2-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"

TRAIN_TASKS=27
SEALED_TASKS=6
EXPOSURES_PER_TASK=4
GROUPS_PER_STEP=2
RESPONSES_PER_GROUP=8
TOTAL_ROLLOUT_GROUPS="$((TRAIN_TASKS * EXPOSURES_PER_TASK))"
TOTAL_TRAINING_STEPS="$((TOTAL_ROLLOUT_GROUPS / GROUPS_PER_STEP))"
MAX_PROMPT_TOKENS=4096
MAX_RESPONSE_TOKENS=49152
MAX_CONTEXT_TOKENS=53248
AGENT_TIMEOUT_SECONDS=1800
PREWARM_GROUPS=4
MAX_QUEUE_GROUPS=6
MAX_QUEUE_TOKENS="$((MAX_QUEUE_GROUPS * RESPONSES_PER_GROUP * MAX_CONTEXT_TOKENS))"
EMBEDDING_WEIGHT_MIB=2425
MIN_WEIGHT_BUCKET_MB=2560
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB:-2560}"

if (( TOTAL_ROLLOUT_GROUPS != 108 || TOTAL_TRAINING_STEPS != 54 )); then
  printf 'unexpected Qwen3.8 Step70 mixed27 shape\n' >&2
  exit 2
fi
if (( WEIGHT_BUCKET_MB < MIN_WEIGHT_BUCKET_MB )); then
  printf 'weight sync bucket %s MiB cannot safely hold the %s MiB Qwen3.8 embedding tensor; require at least %s MiB\n' \
    "${WEIGHT_BUCKET_MB}" "${EMBEDDING_WEIGHT_MIB}" "${MIN_WEIGHT_BUCKET_MB}" >&2
  exit 2
fi
python3 "${PROJECT_ROOT}/scripts/assemble_qwen38_step70_mixed27.py" validate \
  --canonical "${CANONICAL_FILE}" \
  --schedule "${TRAIN_FILE}" \
  --sealed "${SEALED_FILE}" \
  --safe-summary "${SAFE_SUMMARY}"

python3 - "${MODEL_PATH}/llin_export_manifest.json" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (manifest.get("verification") or {}).get("valid") is not True:
    raise SystemExit("Step70 HF export verification is not valid")
if "global_step_70" not in str(manifest.get("actor_checkpoint") or ""):
    raise SystemExit("training initialization is not the verified Step70 checkpoint")
PY

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/training_contract.txt" <<EOF
contract=llin-qwen38-step70-mixed27-four-exposure-formal-v1
model=qwen38-27b-grpo-step70
initialization=verified_step70_hf_export
source_policy_step=70
reward=banded-v2-strict-table-v1
owner_authorized_mixed27=true
train_tasks=${TRAIN_TASKS}
sealed_tasks=${SEALED_TASKS}
exposures_per_task=${EXPOSURES_PER_TASK}
rollout_groups=${TOTAL_ROLLOUT_GROUPS}
responses_per_group=${RESPONSES_PER_GROUP}
groups_per_optimizer_step=${GROUPS_PER_STEP}
optimizer_steps=${TOTAL_TRAINING_STEPS}
learning_rate=1e-7
reasoning_effort=medium
max_prompt_tokens=${MAX_PROMPT_TOKENS}
max_response_tokens=${MAX_RESPONSE_TOKENS}
max_context_tokens=${MAX_CONTEXT_TOKENS}
trajectory_timeout_seconds=${AGENT_TIMEOUT_SECONDS}
timeout_scope=entire_pi_agent_run_including_vllm_admission_generation_and_tools
temperature=1.0
top_p=0.95
top_k=20
trainer_npus=16
trainer_topology=tp4_pp2_cp2
rollout_npus=16
rollout_topology=tp4_dp4
rollout_max_num_seqs_per_replica=16
weight_sync_bucket_mib=${WEIGHT_BUCKET_MB}
checkpoint_frequency=${TOTAL_TRAINING_STEPS}
kept_checkpoints=1
checkpoint_payload=model,extra
optimizer_checkpoint_saved=false
validation=disabled_sealed6_not_loaded_by_trainer
promotion_allowed=false
EOF

PI_REWARD_MODE=banded_v2 \
MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${TRAIN_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
RAY_ADDRESS="${RAY_ADDRESS:-192.168.202.5:36379}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP}" \
SAVE_FREQ="${TOTAL_TRAINING_STEPS}" \
FASTEST_K="${RESPONSES_PER_GROUP}" \
OVERSAMPLE_CANDIDATES="${RESPONSES_PER_GROUP}" \
PREWARM_GROUPS="${PREWARM_GROUPS}" \
STALENESS_THRESHOLD=2.0 \
MAX_QUEUE_TOKENS="${MAX_QUEUE_TOKENS}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS}" \
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS}" \
MAX_ASSISTANT_TURNS=26 \
MAX_USER_TURNS=25 \
AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS}" \
MAX_PARALLEL_TOOL_CALLS=4 \
MAX_TOOL_RESPONSE_CHARS=32768 \
ROLLOUT_GPU_MEMORY_UTILIZATION=0.80 \
ROLLOUT_MAX_BATCHED_TOKENS=16384 \
ROLLOUT_MAX_SEQS=16 \
AGENT_WORKERS=12 \
CONCURRENT_SAMPLES_PER_REPLICA=6 \
WEIGHT_BUCKET_MB="${WEIGHT_BUCKET_MB}" \
OPTIMIZER_CPU_OFFLOAD=false \
ENGINE_OPTIMIZER_OFFLOAD=false \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.shuffle=false \
  data.seed=20260820 \
  +data.apply_chat_template_kwargs.reasoning_effort=medium \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${PROJECT_ROOT}/configs/pi_workspace_tools.yaml" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/configs/pi_agent_loops.yaml" \
  actor_rollout_ref.rollout.agent.default_agent_loop=pi_agent \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  actor_rollout_ref.actor.optim.lr=1e-7 \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
  reward.custom_reward_function.name=compute_score_banded_v2 \
  trainer.project_name=llin-qwen38-verl-grpo \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq="${TOTAL_TRAINING_STEPS}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  async_training.use_trainer_do_validate=false \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'
