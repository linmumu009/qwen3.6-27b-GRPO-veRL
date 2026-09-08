#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/workspace/llin-verl-grpo
OUT="$ROOT/runs/book-task-aligned-groups-20260908-03"
umask 077
exec 9>"$ROOT/runs/.book-task-aligned-generation.lock"
flock -n 9 || exit 3
export PYTHONPATH="$ROOT/scripts:$ROOT:${PYTHONPATH:-}"
python3 "$ROOT/scripts/build_book_task_aligned_groups.py" --root "$ROOT" --api-config "$ROOT/private/chat_api_config.user.json" --output "$OUT" --resume-from "$ROOT/runs/book-task-aligned-groups-20260908-02"
