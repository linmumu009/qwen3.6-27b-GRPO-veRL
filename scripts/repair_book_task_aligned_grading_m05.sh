#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT="$ROOT/runs/book-task-aligned-value-20260908-01"
umask 077
export PYTHONPATH="$ROOT/scripts:$ROOT:${PYTHONPATH:-}"
exec 9>"$ROOT/runs/.book-task-aligned-value-grading.lock"
flock -n 9 || exit 3
exec >"$OUT.grading_repair.log" 2>&1
trap 'printf "grading_repair_failed_at_line_%s\n" "$LINENO" > "$OUT/status.txt"' ERR
python3 "$ROOT/scripts/grade_book_task_aligned_value.py" grade --out "$OUT" --api-config "$ROOT/private/chat_api_config.user.json" --repair
python3 "$ROOT/scripts/grade_book_task_aligned_value.py" summarize --out "$OUT" --repair
