#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
RUN=${ROOT}/runs/logistics-mcq-fit-20260905
BASE=${ROOT}/runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource
CASES=${ROOT}/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl
umask 077
[[ ! -e "${RUN}" ]] || { printf 'refusing overwrite\n' >&2; exit 2; }
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || { printf 'another logistics training pipeline holds the lock\n' >&2; exit 3; }
mkdir -p "${RUN}/private" "${RUN}/safe"
exec >"${RUN}/pipeline.log" 2>&1
trap 'printf "failed at line %s\n" "${LINENO}" > "${RUN}/status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
printf 'preparing_data\n' > "${RUN}/status.txt"
python3 "${ROOT}/scripts/prepare_logistics_mcq_fit.py" --source "${CASES}" --model "${BASE}" --output-dir "${RUN}/data"
python3 "${ROOT}/scripts/check_logistics_mcq_fit.py" --data "${RUN}/data" --model "${BASE}"
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
for arm in all answer; do
  evaluate_model "mask8_${arm}" "${ROOT}/runs/logistics-mask8-${arm}-20260905/hf_export_step_64"
done
python3 "${ROOT}/scripts/summarize_logistics_mcq_fit.py" --run "${RUN}" --phase masks
printf 'training_full_1672_items\n' > "${RUN}/status.txt"
bash "${ROOT}/scripts/run_logistics_mcq_fit_train.sh" > "${RUN}/train.log" 2>&1
for step in 418 836; do
  printf 'exporting_step_%s\n' "${step}" > "${RUN}/status.txt"
  available_bytes=$(df -B1 --output=avail "${ROOT}/runs" | tail -n 1 | tr -d ' ')
  [[ "${available_bytes}" -ge 95000000000 ]] || { printf 'need 95GB free before export\n' >&2; exit 2; }
  python3 "${ROOT}/scripts/export_megatron_dist_to_hf.py" \
    --actor-checkpoint "${RUN}/train/checkpoints/global_step_${step}" --base-model "${BASE}" \
    --output-dir "${RUN}/hf_export_step_${step}" > "${RUN}/export_${step}.log" 2>&1
  evaluate_model "sft_step${step}" "${RUN}/hf_export_step_${step}"
done
python3 "${ROOT}/scripts/summarize_logistics_mcq_fit.py" --run "${RUN}" --phase final
printf 'complete\n' > "${RUN}/status.txt"
