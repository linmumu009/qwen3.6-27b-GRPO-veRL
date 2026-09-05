#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
DATA=${ROOT}/runs/logistics-mcq-fit-20260905/data
OUT=${ROOT}/runs/logistics-mcq-fit-20260905/train
BASE=${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
SOURCE=${ROOT}/runs/llin-step120-opensource-20260825-02/checkpoints/global_step_120/actor/model/dist_ckpt
BRIDGE=${ROOT}/reference/Megatron-Bridge-de93536e/src
[[ ! -e "${OUT}" ]] || { printf 'refusing overwrite\n' >&2; exit 2; }
# Two model-only checkpoints plus two HF exports, with a free-space reserve.
available_bytes=$(df -B1 --output=avail "${ROOT}/runs" | tail -n 1 | tr -d ' ')
[[ "${available_bytes}" -ge 260000000000 ]] || { printf 'need 260GB free before training\n' >&2; exit 2; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="${ROOT}:${BRIDGE}:${ROOT}/runtime:/verl:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1 HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=true
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
python3 "${ROOT}/scripts/check_logistics_mcq_fit.py" --data "${DATA}" --model "${BASE}"
mkdir -p "${OUT}/torchrun_logs"
torchrun --standalone --nnodes=1 --nproc_per_node=16 --log-dir="${OUT}/torchrun_logs" --redirects=3 --tee=0 \
  -m verl.trainer.sft_trainer \
  "data.train_files=${DATA}/train.parquet" "data.val_files=${DATA}/train.parquet" \
  data.train_max_samples=-1 data.train_batch_size=4 data.micro_batch_size_per_gpu=1 \
  data.use_dynamic_bsz=false data.max_token_len_per_gpu=4096 data.max_length=4096 \
  data.pad_mode=no_padding data.truncation=error data.num_workers=0 \
  "data.custom_cls.path=${ROOT}/scripts/qwen36_mcq_answer_dataset.py" data.custom_cls.name=Qwen36MCQAnswerDataset \
  model=hf_model "model.path=${BASE}" model.trust_remote_code=true model.use_remove_padding=false model.lora_rank=0 \
  model.mtp.enable=false model.mtp.enable_train=false model.mtp.enable_rollout=false \
  optim=megatron optim.lr=2e-6 optim.min_lr=2e-6 optim.lr_warmup_steps_ratio=0.01 \
  optim.weight_decay=0 'optim.betas=[0.9,0.999]' optim.clip_grad=1.0 optim.lr_decay_style=cosine \
  +optim.override_optimizer_config.adam_beta1=0.9 +optim.override_optimizer_config.adam_beta2=0.999 \
  +optim.override_optimizer_config.optimizer_cpu_offload=false \
  engine=megatron engine.tensor_model_parallel_size=4 engine.pipeline_model_parallel_size=2 engine.context_parallel_size=2 \
  engine.use_mbridge=true engine.vanilla_mbridge=false engine.use_megatron_fsdp=false engine.use_remove_padding=false \
  engine.param_offload=false engine.optimizer_offload=false engine.grad_offload=true engine.dtype=bfloat16 \
  engine.use_distributed_optimizer=true engine.use_dist_checkpointing=true "engine.dist_checkpointing_path=${SOURCE}" \
  ++engine.override_transformer_config.attention_backend=auto ++engine.override_transformer_config.context_parallel_algo=kvallgather_cp_algo \
  ++engine.override_transformer_config.recompute_method=uniform ++engine.override_transformer_config.recompute_granularity=full \
  ++engine.override_transformer_config.recompute_num_layers=1 ++engine.override_transformer_config.use_flash_attn=true \
  ++engine.override_transformer_config.sequence_parallel=true 'checkpoint.load_contents=[]' 'checkpoint.save_contents=[model,extra]' \
  "trainer.default_local_dir=${OUT}/checkpoints" trainer.project_name=llin-logistics-mcq-fit \
  trainer.experiment_name=same-item-answer-sft-1x-2x 'trainer.logger=["console"]' \
  trainer.total_epochs=2 trainer.total_training_steps=836 trainer.save_freq=418 trainer.test_freq=418 \
  trainer.resume_mode=disable trainer.max_ckpt_to_keep=2 trainer.nnodes=1 trainer.n_gpus_per_node=16
TOKENS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sequence_tokens_per_epoch"])' "${DATA}/manifest.safe.json")
python3 "${ROOT}/scripts/summarize_logistics_cpt_exposure_curve.py" --run-dir "${OUT}" \
  --experiment logistics_same_item_mcq_answer_sft --steps-per-exposure 418 --total-exposures 2 \
  --sequence-tokens-per-exposure "${TOKENS}" --checkpoint-exposure 1 --checkpoint-exposure 2 \
  --output "${OUT}/training_summary.safe.json"
