#!/usr/bin/env bash
set -euo pipefail

# Frozen formal contract.  This file is intentionally inert until the shadow
# audit owner supplies the explicit approval token below.
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
VERL_ROOT="${VERL_ROOT:-/verl}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/runs/llin-v15-codex-model2-100-step120-8x-20260821-01/grpo_readiness_audit_20260822-05}"
APPROVED43="${APPROVED43:-${PACKAGE_ROOT}/private/grpo_approved43.sensitive.parquet}"
APPROVED43_MANIFEST="${APPROVED43_MANIFEST:-${PACKAGE_ROOT}/private/grpo_approved43_manifest.sensitive.jsonl}"
TASKS_FILE="${TASKS_FILE:?TASKS_FILE must point to the frozen private 100-task JSONL used only by manifest index}"
RUN_NAME="${RUN_NAME:-llin-qwen38-approved43-4x-v5-nominal172-optpending}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${OUTPUT_DIR}/private/approved43x4.sensitive.parquet}"
TRAIN_SUMMARY="${TRAIN_SUMMARY:-${OUTPUT_DIR}/approved43x4.safe.json}"

EXPECTED_CONFIG_SHA256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
EXPECTED_MODEL_COMPOUND_SHA256="e94e58ab1b25c6bbbff809b8af9f57a5d42eeff7db7b077a8248d359a20b7325"
TRAIN_TASKS=43
EXPOSURES_PER_TASK=4
TOTAL_ROLLOUT_GROUPS=172
RESPONSES_PER_GROUP=8
OVERSAMPLE_CANDIDATES=16
GROUPS_PER_STEP=2
TOTAL_NOMINAL_STEPS=86
PI_PROCESS_BONUS_ALPHA="${PI_PROCESS_BONUS_ALPHA:-0}"

if [[ "${FORMAL_TRAINING_APPROVED:-}" != "approved-after-shadow-v2" ]]; then
  printf 'formal training remains paused: missing post-shadow approval token\n' >&2
  exit 3
fi
if [[ "${PI_PROCESS_BONUS_ALPHA}" != "0" && "${PI_PROCESS_BONUS_ALPHA}" != "0.10" ]]; then
  printf 'PI_PROCESS_BONUS_ALPHA must be 0 or 0.10\n' >&2
  exit 2
fi
if [[ "${PI_PROCESS_BONUS_ALPHA}" == "0.10" && "${VERIFIED_PROCESS_AUDIT_APPROVED:-}" != "coverage-and-precision-pass" ]]; then
  printf 'verified process bonus is not approved; use binary H*C fallback\n' >&2
  exit 3
fi
if (( TRAIN_TASKS * EXPOSURES_PER_TASK != TOTAL_ROLLOUT_GROUPS || TOTAL_ROLLOUT_GROUPS / GROUPS_PER_STEP != TOTAL_NOMINAL_STEPS )); then
  printf 'frozen group geometry changed\n' >&2
  exit 2
fi

observed_config_sha256="$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')"
if [[ "${observed_config_sha256}" != "${EXPECTED_CONFIG_SHA256}" ]]; then
  printf 'Qwen3.8 config hash mismatch\n' >&2
  exit 2
fi
if [[ "${QWEN38_MODEL_COMPOUND_SHA256_ATTESTED:-}" != "${EXPECTED_MODEL_COMPOUND_SHA256}" ]]; then
  printf 'Qwen3.8 18-shard compound hash attestation missing or mismatched\n' >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/private"
chmod 700 "${OUTPUT_DIR}/private"
python3 "${PROJECT_ROOT}/scripts/prepare_qwen38_approved43_outcome_training.py" \
  --approved43 "${APPROVED43}" \
  --manifest "${APPROVED43_MANIFEST}" \
  --tasks "${TASKS_FILE}" \
  --output "${TRAIN_FILE}" \
  --safe-summary "${TRAIN_SUMMARY}"

# Patch a CPU-visible veRL source tree before any model is loaded.  Re-running
# these patches is idempotent.
python3 "${PROJECT_ROOT}/scripts/patch_verl_fastest_k_oversampling.py" \
  --rollouter "${VERL_ROOT}/verl/experimental/fully_async_policy/fully_async_rollouter.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py" \
  --tool-agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/tool_agent_loop.py" \
  --llm-server "${VERL_ROOT}/verl/workers/rollout/llm_server.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_hard_gate_resampling.py" \
  --agent-loop "${VERL_ROOT}/verl/experimental/agent_loop/agent_loop.py"
python3 "${PROJECT_ROOT}/scripts/patch_verl_grpo_strict_variance_gate.py" \
  --trainer "${VERL_ROOT}/verl/experimental/separation/ray_trainer.py"

cat > "${OUTPUT_DIR}/training_contract.txt" <<EOF
contract=qwen38-approved43-outcome-gated-trajectory-v5
training_status=authorized_only_after_shadow_v2
actor_initialization=${MODEL_PATH}
reference_initialization=${MODEL_PATH}
qwen36_or_historical_checkpoint_reused=false
model_config_sha256=${EXPECTED_CONFIG_SHA256}
model_18shard_compound_sha256=${EXPECTED_MODEL_COMPOUND_SHA256}
reward=H*C*(1+${PI_PROCESS_BONUS_ALPHA}*P_verified)
incorrect_reward=0
observed_table_field_fit_efficiency_in_reward=false
reward_scope=trajectory_scalar_after_complete_multiturn
turn_or_token_credit_assignment=false
nominal_groups=${TOTAL_ROLLOUT_GROUPS}
accepted_responses_per_group=${RESPONSES_PER_GROUP}
physical_attempt_cap_per_group=${OVERSAMPLE_CANDIDATES}
nominal_steps=${TOTAL_NOMINAL_STEPS}
optimizer_steps=dynamic_strict_mixed_only
algorithm_use_kl_in_reward=false
actor_use_kl_loss=true
actor_kl_loss_coef=0.001
actor_kl_loss_type=low_var_kl
hard_staleness=0_exact_policy_version
learning_rate=5e-8_constant
entropy=0
save_policy=final_only_model_and_extra
EOF

export PI_PROCESS_BONUS_ALPHA
MODEL_PATH="${MODEL_PATH}" \
DATA_FILE="${TRAIN_FILE}" \
RUN_NAME="${RUN_NAME}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
TRAIN_TP=4 TRAIN_PP=2 TRAIN_CP=2 TRAIN_NPUS=16 \
ROLLOUT_TP=4 ROLLOUT_NPUS=16 ROLLOUT_NNODES=1 \
TOTAL_TRAINING_STEPS="${TOTAL_NOMINAL_STEPS}" \
TOTAL_ROLLOUT_GROUPS="${TOTAL_ROLLOUT_GROUPS}" \
GROUPS_PER_STEP="${GROUPS_PER_STEP}" \
RESPONSES_PER_GROUP="${RESPONSES_PER_GROUP}" \
FASTEST_K="${RESPONSES_PER_GROUP}" \
OVERSAMPLE_CANDIDATES="${OVERSAMPLE_CANDIDATES}" \
PREWARM_GROUPS=2 \
STALENESS_THRESHOLD=0 \
MAX_PROMPT_TOKENS=4096 \
MAX_RESPONSE_TOKENS=90112 \
MAX_CONTEXT_TOKENS=94208 \
MAX_QUEUE_TOKENS="$((GROUPS_PER_STEP * RESPONSES_PER_GROUP * 94208))" \
SAVE_FREQ="${TOTAL_NOMINAL_STEPS}" \
bash "${PROJECT_ROOT}/scripts/run_pi_grpo_fully_async_tp4_pp2_cp2.sh" \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TRAIN_FILE}" \
  data.shuffle=false \
  data.seed=20260822 \
  actor_rollout_ref.actor.optim.lr=5e-8 \
  actor_rollout_ref.actor.optim.lr_decay_style=constant \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.bypass_mode=True \
  reward.custom_reward_function.path="${PROJECT_ROOT}/llin_verl/pi_reward.py" \
  reward.custom_reward_function.name=compute_score_correctness_gated_process_v5 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq="${TOTAL_NOMINAL_STEPS}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]'
