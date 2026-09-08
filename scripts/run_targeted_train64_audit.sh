#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT=/opt/llin-targeted-train64-audit-20260908
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
python3 "$ROOT/scripts/prepare_targeted_train64.py"
for step in 5 10; do
 printf 'probe_step_%s\n' "$step" > "$OUT/status.txt"
 python3 "$ROOT/scripts/probe_targeted_book_groups.py" --root "$ROOT" --data "$ROOT/runs/targeted-book-train64-20260908" \
  --model "/opt/llin-targeted-book-pilot-20260908/hf_step_$step" --model-label "targeted_book_pilot_step_$step" --output "$OUT/book_probe_step_$step"
done
printf 'grading\n' > "$OUT/status.txt"
python3 "$ROOT/scripts/grade_targeted_book_pilot.py" --root "$ROOT" --pilot "$OUT" --training-candidates \
 --output "$ROOT/runs/targeted-book-train64-grading-20260908" --api-config "$ROOT/private/chat_api_config.user.json"
printf 'complete\n' > "$OUT/status.txt"
