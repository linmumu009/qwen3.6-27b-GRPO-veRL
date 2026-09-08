#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT="$ROOT/runs/logistics-strategy-diagnostic-20260908"
exec 9>"$ROOT/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:${ROOT}/runtime:${ROOT}:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
umask 077
exec >"$OUT/evidence.launch.log" 2>&1
trap 'printf "evidence_failed_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
for model in step120 cpt; do
 python3 "$ROOT/scripts/run_logistics_strategy_evidence.py" run --root "$ROOT" --out "$OUT" --model "$model"
done
python3 "$ROOT/scripts/run_logistics_strategy_evidence.py" summarize --root "$ROOT" --out "$OUT"
