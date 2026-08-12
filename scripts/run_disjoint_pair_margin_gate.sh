#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/llin-verl-grpo}"
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MEGATRON_BRIDGE_ROOT="${MEGATRON_BRIDGE_ROOT:-${PROJECT_ROOT}/reference/Megatron-Bridge-de93536e/src}"
MODEL_DIST_CKPT="${MODEL_DIST_CKPT:-${PROJECT_ROOT}/runs/llin-pi-dense-correctness-step100-to-step120-20260810-01/checkpoints/global_step_120/actor/model/dist_ckpt}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/disjoint_first_error_pairs_20260812}"
DATA_FILE="${DATA_FILE:-${DATA_DIR}/disjoint_first_error_pairs.parquet}"
DATA_CONTRACT="${DATA_CONTRACT:-${DATA_DIR}/first_error_pair_contract.json}"
RUN_NAME="${RUN_NAME:-llin-disjoint-pair-margin-step120-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/${RUN_NAME}}"
MODEL_LABEL="${MODEL_LABEL:-step120_disjoint_pair_margin}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
NPROC="${NPROC:-16}"
TP="${TP:-4}"
PP="${PP:-2}"
CP="${CP:-2}"

if (( TP * PP * CP != NPROC )); then
  printf 'invalid topology for disjoint pair margin gate\n' >&2
  exit 2
fi
for path in "${MODEL_PATH}/config.json" "${MODEL_DIST_CKPT}/.metadata" "${DATA_FILE}" "${DATA_CONTRACT}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'disjoint pair margin input missing: %s\n' "${path}" >&2
    exit 2
  fi
done
if [[ ! -d "${MEGATRON_BRIDGE_ROOT}/megatron/bridge" ]]; then
  printf 'pinned Megatron-Bridge source missing: %s\n' "${MEGATRON_BRIDGE_ROOT}" >&2
  exit 2
fi

read -r PAIRS ROWS < <(python3 - "${DATA_CONTRACT}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(int(value["pairs"]), int(value["rows"]))
PY
)
if (( PAIRS < 48 || ROWS != 2 * PAIRS )); then
  printf 'disjoint pair count gate failed: pairs=%s rows=%s\n' "${PAIRS}" "${ROWS}" >&2
  exit 2
fi

export PYTHONPATH="${MEGATRON_BRIDGE_ROOT}:${PROJECT_ROOT}/runtime:${PROJECT_ROOT}:/verl:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
python3 -m scripts.check_disjoint_first_error_pairs \
  --data-file "${DATA_FILE}" \
  --contract "${DATA_CONTRACT}" \
  --model-path "${MODEL_PATH}" \
  --max-length "${MAX_LENGTH}" \
  --output "${OUTPUT_DIR}/token_gate.json"

cat > "${OUTPUT_DIR}/experiment_contract.txt" <<EOF
purpose=disjoint_correct_vs_actual_wrong_sql_semantic_delta_margin
source_checkpoint=step120
pairs=${PAIRS}
rows=${ROWS}
forward_only=true
optimizer_initialized=false
checkpoint_saved=false
global_batch_rows=${ROWS}
topology=tp${TP}_pp${PP}_cp${CP}
semantic_delta_margin=rejected_mean_nll_minus_chosen_mean_nll
training_authority=only_after_pair_count_token_and_margin_gates
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
  "data.train_batch_size=${ROWS}" \
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
  trainer.project_name=llin-disjoint-pair-margin \
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
  "+diagnostic.model_label=${MODEL_LABEL}"

python3 -m scripts.analyze_disjoint_pair_margin \
  --diagnostic "${OUTPUT_DIR}/diagnostic.json" \
  --dataset-contract "${DATA_CONTRACT}" \
  --output "${OUTPUT_DIR}/margin_result.json"
