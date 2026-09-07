#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=${ROOT}/runs/book-sft-diagnosis-20260907
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
[[ ! -e "${OUT}" ]] || exit 2
umask 077
mkdir -p "${OUT}"
exec >"${OUT}/pipeline.log" 2>&1
trap 'printf "failed_at_line_%s\n" "$LINENO" > "${OUT}/status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
for label in cpt sft; do
 if [[ "$label" == cpt ]]; then
  MODEL=${ROOT}/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116
 else
  MODEL=/opt/llin-book-sft-2000-20260907-run2/hf_export
 fi
 printf 'evaluating_%s\n' "$label" > "${OUT}/status.txt"
 python3 "${ROOT}/scripts/diagnose_book_sft_inference.py" --model "${MODEL}" --label "$label" \
  --dev "${ROOT}/runs/book-sft-production-2000-20260907/dev.messages.private.jsonl" --output "${OUT}/${label}"
done
printf 'inference_complete\n' > "${OUT}/status.txt"
