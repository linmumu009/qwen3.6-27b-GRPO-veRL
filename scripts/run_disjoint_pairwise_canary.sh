#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:-${SOURCE_CHECKPOINT}/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/disjoint_first_error_pairs_20260812}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/disjoint_first_error_pairs.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/first_error_pair_contract.json}"
TOKEN_GATE="${TOKEN_GATE:?TOKEN_GATE is required}"
MARGIN_RESULT="${MARGIN_RESULT:?MARGIN_RESULT is required}"
RUN_NAME="${RUN_NAME:-llin-disjoint-pairwise-step120-1step-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
PAIRWISE_BETA="${PAIRWISE_BETA:-1.0}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid disjoint pairwise topology\n' >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${MODEL_DIST_CKPT}/.metadata" "${TRAIN_FILE}" "${DATA_CONTRACT}" "${TOKEN_GATE}" "${MARGIN_RESULT}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'disjoint pairwise canary input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

read -r PAIRS ROWS < <(python3 - "${DATA_CONTRACT}" "${TOKEN_GATE}" "${MARGIN_RESULT}" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
gate = json.load(open(sys.argv[2], encoding="utf-8"))
margin = json.load(open(sys.argv[3], encoding="utf-8"))
errors = []
pairs = int(contract.get("pairs") or 0)
rows = int(contract.get("rows") or 0)
if contract.get("contract") != "current-definition-disjoint-first-error-pairs-v1":
    errors.append("unexpected disjoint pair data contract")
if not 48 <= pairs <= 64 or rows != 2 * pairs or contract.get("pair_count_gate_passed") is not True:
    errors.append("disjoint pair count must be 48-64 with exactly two rows per pair")
for key in (
    "chosen_queries_mechanically_verified",
    "rejected_queries_are_actual_step120_first_errors",
    "all_first_error_tool_results_observed",
    "pair_prefix_identical_through_observed_error_result",
):
    if contract.get(key) is not True:
        errors.append(f"disjoint pair data gate failed: {key}")
if gate.get("contract") != "current-definition-disjoint-pair-token-gate-v1":
    errors.append("unexpected disjoint pair token gate")
if gate.get("pairs") != pairs or gate.get("rows") != rows:
    errors.append("disjoint pair token gate count differs")
for key in (
    "all_delta_masks_nonempty",
    "all_delta_masks_subset_of_sql",
    "all_pairs_adjacent_chosen_then_rejected",
    "all_candidate_signs_and_pair_indices_match",
):
    if gate.get(key) is not True:
        errors.append(f"disjoint pair token gate failed: {key}")
if margin.get("contract") != "current-definition-disjoint-pair-margin-result-v1":
    errors.append("unexpected disjoint pair margin result")
if margin.get("task_count") != pairs:
    errors.append("disjoint pair margin count differs")
if margin.get("training_allowed") is not True:
    errors.append("disjoint pair margin gate does not authorize one step")
if (margin.get("decision") or {}).get("one_step_training_allowed") is not True:
    errors.append("disjoint pair decision does not authorize one step")
if errors:
    raise SystemExit("; ".join(errors))
print(pairs, rows)
PY
)

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=disjoint_semantic_delta_reference_free_pairwise_logistic_canary
source_checkpoint=step120
pairs=${PAIRS}
rows=${ROWS}
optimizer_steps=1
global_batch_rows=${ROWS}
micro_batch_rows=2
pair_order=chosen_then_rejected_no_shuffle
loss_scope=semantic_delta_tokens_only
pairwise_beta=${PAIRWISE_BETA}
learning_rate=${LEARNING_RATE}
optimizer=fresh_cpu_offloaded_adam
checkpoint_policy=final_model_and_extra_only
optimizer_checkpoint_saved=false
topology=tp${TP}_pp${PP}_cp${CP}
post_gate=original_frozen16_chosen_preferred_12_margin_improved_12_no_earlier_regressions
full_replay_before_frozen16_probability_gate=false
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
  "data.train_batch_size=${ROWS}" \
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
  trainer.project_name=llin-disjoint-pairwise \
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
