#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT="$ROOT/runs/book-capability-sft-aligned-20260909-01"
umask 077
mkdir -p "$OUT"
exec 9>"$ROOT/runs/.logistics-exam-cpt.lock"
flock -n 9 || { echo 'Another logistics job holds the lock' >&2; exit 3; }
test ! -e "$OUT/status.txt"
trap 'echo failed > "$OUT/status.txt"' ERR
CASES="$ROOT/runs/logistics-cpt-diagnostics-20260904/private/public_eval/frozen_cases_source.jsonl"
for arm in cpt sft; do
  if [[ "$arm" == cpt ]]; then
    MODEL="$ROOT/runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116"
  else
    MODEL=/opt/llin-capability-sft-cpt-20260909-01/hf_export
  fi
  echo "evaluating_$arm" > "$OUT/status.txt"
  MODEL_PATH="$MODEL" MODEL_LABEL="capability-aligned-$arm" CASES_PATH="$CASES" \
    REPEATS=3 MAX_MODEL_LEN=4096 MAX_OUTPUT_TOKENS=96 \
    PRIVATE_OUTPUT="$OUT/$arm.jsonl" SAFE_OUTPUT="$OUT/$arm.safe.json" \
    bash "$ROOT/scripts/run_logistics_mcq_on_m05.sh" > "$OUT/$arm.log" 2>&1
  PYTHONPATH="$ROOT:${PYTHONPATH:-}" python3 "$ROOT/scripts/summarize_mcq_repeats.py" \
    --repeat "$OUT/$arm.jsonl" --repeat "$OUT/$arm.repeat2.jsonl" --repeat "$OUT/$arm.repeat3.jsonl" \
    --cases "$CASES" --model-label "capability-aligned-$arm" \
    --repeat-safe "$OUT/$arm.safe.json" --repeat-safe "$OUT/$arm.safe.repeat2.json" --repeat-safe "$OUT/$arm.safe.repeat3.json" \
    --private-output "$OUT/$arm.majority.private.jsonl" --safe-output "$OUT/$arm.majority.safe.json" \
    > "$OUT/$arm.majority.log" 2>&1
done
echo domain_recheck_complete_regressions_pending > "$OUT/status.txt"
