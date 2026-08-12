#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/semantic_delta_margin_gate_20260812}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/semantic_delta_margin_gate.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/contract.json}"
TOKEN_GATE="${TOKEN_GATE:-${DATA_DIR}/token_gate.json}"
RUN_NAME="${RUN_NAME:-llin-semantic-delta-pairwise-step120-1step-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PAIRWISE_BETA="${PAIRWISE_BETA:-1.0}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid pairwise topology\n' >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${MODEL_DIST_CKPT}/.metadata" "${TRAIN_FILE}" "${DATA_CONTRACT}" "${TOKEN_GATE}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'pairwise canary input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

python3 - "${DATA_CONTRACT}" "${TOKEN_GATE}" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
gate = json.load(open(sys.argv[2], encoding="utf-8"))
errors = []
if contract.get("contract") != "semantic-delta-margin-gate-dataset-v1":
    errors.append("unexpected pairwise data contract")
if contract.get("pairs") != 16 or contract.get("rows") != 32:
    errors.append("pairwise canary requires 16 pairs and 32 rows")
for key in (
    "chosen_queries_mechanically_verified_by_source_contract",
    "rejected_queries_are_actual_step120_first_errors",
    "pair_prefix_is_identical_through_observed_error_result",
    "post_candidate_tail_is_fixed_non_scored_stub",
):
    if contract.get(key) is not True:
        errors.append(f"pairwise data gate failed: {key}")
if gate.get("contract") != "semantic-delta-margin-token-gate-v1":
    errors.append("unexpected pairwise token gate")
for key in (
    "all_delta_masks_nonempty",
    "all_delta_masks_subset_of_sql",
    "all_chosen_critical_targets_match_frozen_step120",
    "all_pairs_adjacent_chosen_then_rejected",
    "all_candidate_signs_and_pair_indices_match",
):
    if gate.get(key) is not True:
        errors.append(f"pairwise token gate failed: {key}")
if errors:
    raise SystemExit("; ".join(errors))
PY

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=semantic_delta_reference_free_pairwise_logistic_canary
source_checkpoint=step120
pairs=16
rows=32
optimizer_steps=1
global_batch_rows=32
micro_batch_rows=2
pair_order=chosen_then_rejected_no_shuffle
loss_scope=semantic_delta_tokens_only
pairwise_beta=${PAIRWISE_BETA}
learning_rate=${LEARNING_RATE}
optimizer=fresh_cpu_offloaded_adam
checkpoint_policy=final_model_and_extra_only
optimizer_checkpoint_saved=false
topology=tp${TP}_pp${PP}_cp${CP}
post_gate=chosen_preferred_12_margin_improved_12_no_earlier_regressions
full_replay_before_probability_gate=false
promotion_allowed=false
EOF

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m scripts.run_semantic_delta_pairwise_training \
  "data.train_files=${TRAIN_FILE}" \
  data.val_files=null \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=2 \
  data.use_dynamic_bsz=false \
  "data.max_token_len_per_gpu=${MAX_LENGTH}" \
  "data.max_length=${MAX_LENGTH}" \
  data.pad_mode=no_padding \
  data.truncation=error \
  data.ignore_input_ids_mismatch=false \
  "data.custom_cls.path=${PROJECT_ROOT}/scripts/qwen36_semantic_delta_margin_dataset.py" \
  data.custom_cls.name=Qwen36SemanticDeltaMarginDataset \
  model=hf_model \
  "model.path=${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  model.lora_rank=0 \
  optim=megatron \
  "optim.lr=${LEARNING_RATE}" \
  optim.lr_warmup_steps_ratio=0 \
  optim.weight_decay=0 \
  'optim.betas=[0.9,0.95]' \
  optim.clip_grad=1.0 \
  optim.lr_decay_style=constant \
  +optim.override_optimizer_config.optimizer_cpu_offload=true \
  +optim.override_optimizer_config.optimizer_offload_fraction=1 \
  +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=true \
  engine=megatron \
  "engine.tensor_model_parallel_size=${TP}" \
  "engine.pipeline_model_parallel_size=${PP}" \
  "engine.context_parallel_size=${CP}" \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.use_megatron_fsdp=false \
  engine.use_remove_padding=false \
  engine.forward_only=false \
  engine.param_offload=false \
  engine.optimizer_offload=true \
  engine.grad_offload=true \
  engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=true \
  engine.use_dist_checkpointing=true \
  "engine.dist_checkpointing_path=${MODEL_DIST_CKPT}" \
  ++engine.override_transformer_config.attention_backend=auto \
  ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  ++engine.override_transformer_config.recompute_method=uniform \
  ++engine.override_transformer_config.recompute_granularity=full \
  ++engine.override_transformer_config.recompute_num_layers=1 \
  ++engine.override_transformer_config.use_flash_attn=true \
  ++engine.override_transformer_config.sequence_parallel=true \
  'checkpoint.load_contents=[]' \
  'checkpoint.save_contents=[model,extra]' \
  "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints" \
  trainer.project_name=llin-semantic-delta-pairwise \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.max_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}" \
  "+pairwise.beta=${PAIRWISE_BETA}"
