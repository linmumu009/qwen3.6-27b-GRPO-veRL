#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=$ROOT/runs/targeted-pilot-nll-20260908
[[ ! -e "$OUT" ]] || exit 2
exec 9>"$ROOT/runs/.logistics-exam-cpt.lock"
flock -n 9 || exit 3
umask 077
mkdir -p "$OUT"
exec >"$OUT/pipeline.log" 2>&1
trap 'printf "failed_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="/vllm:$ROOT/runtime:$ROOT:/verl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset ASCEND_RT_VISIBLE_DEVICES
for label in baseline step5 step10; do
 if [[ "$label" == baseline ]]; then MODEL=$ROOT/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116
 else MODEL=/opt/llin-targeted-book-pilot-20260908/hf_step_${label#step}; fi
 printf '%s\n' "$label" > "$OUT/status.txt"
 python3 "$ROOT/scripts/measure_targeted_pilot_nll.py" --root "$ROOT" --model "$MODEL" --label "$label" --output "$OUT/$label"
done
printf 'complete\n' > "$OUT/status.txt"
