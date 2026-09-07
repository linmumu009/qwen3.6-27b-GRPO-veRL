#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=${ROOT}/runs/book-sft-diagnosis-20260907
# Run only after the original inference stage releases its device lock.
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -w 1800 9 || exit 3
[[ "$(cat "${OUT}/status.txt")" == inference_complete ]] || exit 2
umask 077
exec >"${OUT}/concise.log" 2>&1
trap 'printf "failed_at_line_%s\n" "$LINENO" > "${OUT}/concise_status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
for label in cpt sft; do
 if [[ "$label" == cpt ]]; then MODEL=${ROOT}/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116
 else MODEL=/opt/llin-book-sft-2000-20260907-run2/hf_export; fi
 printf 'evaluating_%s\n' "$label" > "${OUT}/concise_status.txt"
 python3 "${ROOT}/scripts/diagnose_book_sft_inference.py" --model "${MODEL}" --label "$label" --concise \
 --dev "${ROOT}/runs/book-sft-production-2000-20260907/dev.messages.private.jsonl" --output "${OUT}/${label}_concise"
done
printf 'inference_complete\n' > "${OUT}/concise_status.txt"
