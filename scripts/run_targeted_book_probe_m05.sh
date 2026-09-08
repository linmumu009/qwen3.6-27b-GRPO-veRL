#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
DATA=${DATA_DIR:-${ROOT}/runs/targeted-book-groups-20260908-v2}
OUT=${OUT_DIR:-${ROOT}/runs/targeted-book-probe-20260908}
[[ -f "$DATA/summary.safe.json" && ! -e "$OUT" ]] || exit 2
exec 9>"${ROOT}/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
exec >"${OUT}.launch.log" 2>&1
trap 'test ! -d "$OUT" || printf "failed_at_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
python3 "$ROOT/scripts/probe_targeted_book_groups.py" --root "$ROOT" --data "$DATA" --output "$OUT"
