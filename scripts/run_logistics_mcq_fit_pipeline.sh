#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
RUN=${ROOT}/runs/logistics-mcq-fit-20260905
BASE=${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
CASES=${ROOT}/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl
BRIDGE=${ROOT}/reference/Megatron-Bridge-de93536e/src
POSTTRAIN_ONLY=${POSTTRAIN_ONLY:-false}
umask 077
if [[ "${POSTTRAIN_ONLY}" == true ]]; then
  [[ -f "${RUN}/train/training_summary.safe.json" && ! -e "${RUN}/recovery.log" ]] || { printf 'missing completed training or recovery already attempted\n' >&2; exit 2; }
elif [[ "${POSTTRAIN_ONLY}" != false || -e "${RUN}" ]]; then
  printf 'invalid mode or refusing overwrite\n' >&2; exit 2
fi
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || { printf 'another logistics training pipeline holds the lock\n' >&2; exit 3; }
mkdir -p "${RUN}/private" "${RUN}/safe"
if [[ "${POSTTRAIN_ONLY}" == true ]]; then
  exec >"${RUN}/recovery.log" 2>&1
else
  exec >"${RUN}/pipeline.log" 2>&1
fi
trap 'printf "failed at line %s\n" "${LINENO}" > "${RUN}/status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}:${BRIDGE}:${ROOT}/runtime:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
python3 -c 'import importlib.util, pathlib, sys; origin=pathlib.Path(importlib.util.find_spec("megatron.bridge").origin).resolve(); expected=pathlib.Path(sys.argv[1]).resolve(); assert origin.is_relative_to(expected), f"unexpected bridge: {origin}"; print(f"verified bridge: {origin}")' "${BRIDGE}"
if [[ "${POSTTRAIN_ONLY}" == false ]]; then
  printf 'preparing_data\n' > "${RUN}/status.txt"
  python3 "${ROOT}/scripts/prepare_logistics_mcq_fit.py" --source "${CASES}" --model "${BASE}" --output-dir "${RUN}/data"
  python3 "${ROOT}/scripts/check_logistics_mcq_fit.py" --data "${RUN}/data" --model "${BASE}"
fi
evaluate_model() {
  local label=$1 model=$2
  printf 'evaluating_%s\n' "${label}" > "${RUN}/status.txt"
  MODEL_PATH="${model}" MODEL_LABEL="${label}" CASES_PATH="${CASES}" REPEATS=3 \
    MAX_MODEL_LEN=4096 MAX_OUTPUT_TOKENS=96 PRIVATE_OUTPUT="${RUN}/private/${label}.jsonl" \
    SAFE_OUTPUT="${RUN}/safe/${label}.json" \
    bash "${ROOT}/scripts/run_logistics_mcq_on_m05.sh" > "${RUN}/${label}.eval.log" 2>&1
  python3 "${ROOT}/scripts/summarize_mcq_repeats.py" \
    --repeat "${RUN}/private/${label}.jsonl" --repeat "${RUN}/private/${label}.repeat2.jsonl" \
    --repeat "${RUN}/private/${label}.repeat3.jsonl" --cases "${CASES}" --model-label "${label}" \
    --private-output "${RUN}/private/${label}.majority.jsonl" --safe-output "${RUN}/safe/${label}.majority.json"
}
if [[ "${POSTTRAIN_ONLY}" == false ]]; then
for arm in all answer; do
  evaluate_model "mask8_${arm}" "${ROOT}/runs/logistics-mask8-${arm}-20260905/hf_export_step_64"
done
python3 "${ROOT}/scripts/summarize_logistics_mcq_fit.py" --run "${RUN}" --phase masks
printf 'training_full_1672_items\n' > "${RUN}/status.txt"
bash "${ROOT}/scripts/run_logistics_mcq_fit_train.sh" > "${RUN}/train.log" 2>&1
else
  printf 'validating_saved_training\n' > "${RUN}/status.txt"
  TOKENS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sequence_tokens_per_epoch"])' "${RUN}/data/manifest.safe.json")
  python3 "${ROOT}/scripts/summarize_logistics_cpt_exposure_curve.py" --run-dir "${RUN}/train" \
    --experiment logistics_same_item_mcq_answer_sft --steps-per-exposure 418 --total-exposures 2 \
    --sequence-tokens-per-exposure "${TOKENS}" --checkpoint-exposure 1 --checkpoint-exposure 2 \
    --output "${RUN}/train/recovery_validation.safe.json"
fi
for step in 418 836; do
  printf 'exporting_step_%s\n' "${step}" > "${RUN}/status.txt"
  available_bytes=$(df -B1 --output=avail "${ROOT}/runs" | tail -n 1 | tr -d ' ')
  [[ "${available_bytes}" -ge 95000000000 ]] || { printf 'need 95GB free before export\n' >&2; exit 2; }
  python3 "${ROOT}/scripts/export_megatron_dist_to_hf.py" \
    --actor-checkpoint "${RUN}/train/checkpoints/global_step_${step}" --base-model "${BASE}" \
    --output-dir "${RUN}/hf_export_step_${step}" > "${RUN}/export_${step}.${POSTTRAIN_ONLY}.log" 2>&1
  evaluate_model "sft_step${step}" "${RUN}/hf_export_step_${step}"
done
python3 "${ROOT}/scripts/summarize_logistics_mcq_fit.py" --run "${RUN}" --phase final
printf 'complete\n' > "${RUN}/status.txt"
