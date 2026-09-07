#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=${ROOT}/runs/cpt-targeted-diagnosis-20260907
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
[[ ! -e "$OUT" ]] || exit 2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
trap 'test ! -d "$OUT" || printf "failed_at_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
python3 "$ROOT/scripts/run_cpt_targeted_diagnosis.py" --root "$ROOT" --output "$OUT"
