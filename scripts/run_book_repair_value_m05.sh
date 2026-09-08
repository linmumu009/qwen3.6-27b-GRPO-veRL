#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT="$ROOT/runs/book-repair-value-20260908-01"
exec 9>"$ROOT/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}/scripts:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
exec >"$OUT.launch.log" 2>&1
trap 'test ! -d "$OUT" || printf "failed_at_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
python3 "$ROOT/scripts/probe_book_repair_value.py" prepare --root "$ROOT" --out "$OUT"
for model in step120 cpt; do
 python3 "$ROOT/scripts/probe_book_repair_value.py" run --root "$ROOT" --out "$OUT" --model "$model"
done
python3 "$ROOT/scripts/probe_book_repair_value.py" grade --root "$ROOT" --out "$OUT" --api-config "$ROOT/private/chat_api_config.user.json"
