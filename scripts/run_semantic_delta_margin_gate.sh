#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120/actor/model/dist_ckpt}"
CRITICAL_DIR="${CRITICAL_DIR:-${PROJECT_ROOT}/data/repair_sft_critical_token_20260812}"
CRITICAL_DATA="${CRITICAL_DATA:-${CRITICAL_DIR}/critical_token_repair_sft_train.parquet}"
CRITICAL_CONTRACT="${CRITICAL_CONTRACT:-${CRITICAL_DIR}/contract.json}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/semantic_delta_margin_gate_20260812}"
DATA_FILE="${DATA_FILE:-${DATA_DIR}/semantic_delta_margin_gate.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/contract.json}"
RUN_NAME="${RUN_NAME:-llin-semantic-delta-margin-step120-20260812-01}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid topology for semantic-delta margin gate\n' >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${MODEL_DIST_CKPT}/.metadata" "${CRITICAL_DATA}" "${CRITICAL_CONTRACT}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'semantic-delta margin input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
python3 -m scripts.prepare_semantic_delta_margin_gate \
  --critical-data "${CRITICAL_DATA}" \
  --critical-contract "${CRITICAL_CONTRACT}" \
  --output-dir "${DATA_DIR}"
python3 -m scripts.check_semantic_delta_margin_gate \
  --data-file "${DATA_FILE}" \
  --contract "${DATA_CONTRACT}" \
  --model-path "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --output "${OUTPUT_DIR}/token_gate.json"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=step120_correct_vs_actual_wrong_sql_semantic_delta_margin
pairs=16
rows=32
forward_only=true
optimizer_initialized=false
checkpoint_saved=false
topology=tp${TP}_pp${PP}_cp${CP}
semantic_delta_margin=rejected_mean_nll_minus_chosen_mean_nll
baseline_non_regression=frozen_first_nongreedy_token_exact_reconstruction
promotion_allowed=false
EOF

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m scripts.run_teacher_forced_component_diagnostic \
  "data.train_files=${DATA_FILE}" \
  "data.val_files=${DATA_FILE}" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=false \
  data.train_batch_size=32 \
  data.micro_batch_size_per_gpu=1 \
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
  engine=megatron \
  "engine.tensor_model_parallel_size=${TP}" \
  "engine.pipeline_model_parallel_size=${PP}" \
  "engine.context_parallel_size=${CP}" \
  engine.use_mbridge=true \
  engine.vanilla_mbridge=false \
  engine.use_megatron_fsdp=false \
  engine.use_remove_padding=false \
  engine.forward_only=true \
  engine.param_offload=false \
  engine.optimizer_offload=false \
  engine.grad_offload=false \
  engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=true \
  engine.use_dist_checkpointing=true \
  "engine.dist_checkpointing_path=${MODEL_DIST_CKPT}" \
  ++engine.override_transformer_config.attention_backend=auto \
  ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  ++engine.override_transformer_config.use_flash_attn=true \
  ++engine.override_transformer_config.sequence_parallel=true \
  'checkpoint.load_contents=[]' \
  'checkpoint.save_contents=[]' \
  "trainer.default_local_dir=${OUTPUT_DIR}/unused_checkpoints" \
  trainer.project_name=llin-semantic-delta-margin \
  "trainer.experiment_name=${RUN_NAME}" \
  'trainer.logger=["console"]' \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.nnodes=1 \
  "trainer.n_gpus_per_node=${NPROC}" \
  "+diagnostic.output_path=${OUTPUT_DIR}/diagnostic.json" \
  +diagnostic.model_label=step120_semantic_delta_margin

python3 -m scripts.analyze_semantic_delta_margin_gate \
  --diagnostic "${OUTPUT_DIR}/diagnostic.json" \
  --dataset-contract "${DATA_CONTRACT}" \
  --output "${OUTPUT_DIR}/semantic_delta_margin_result.json"
