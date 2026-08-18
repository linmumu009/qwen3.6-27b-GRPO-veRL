# Qwen3.8-27B 70-task GRPO formal-run contract

Date: 2026-08-18

## Decision

The owner-authorized pool contains 70 unique, explicitly reviewed table tasks from
v15/v20/v21. Each task is exposed exactly twice, producing 140 prompt groups.
Every group samples eight fresh on-policy responses and all eight are retained for
GRPO; there is no Fastest-K discard. Two groups form one optimizer update, for 70
updates in total.

The run starts from the native Qwen3.8-27B Hugging Face base model. Machine 5 uses
16 NPUs for Megatron training with TP4/PP2/CP2. Machine 6 uses 16 NPUs for rollout
with TP4/DP4 and at most 16 active sequences per replica. This keeps the proven
higher-throughput TP4 rollout topology while avoiding a third-host weight-sync and
failure domain.

## Context and sampling

The private 70-task aggregate contains 286 historical Qwen3.8 trajectories. The
largest initial prompt was 1,289 tokens; among 234 normally completed trajectories,
the longest response was 43,401 tokens. The formal limits are therefore:

- prompt: 4,096 tokens;
- response: 49,152 tokens;
- total context: 53,248 tokens;
- temperature/top-p/top-k: 1.0/0.95/20;
- reasoning effort: medium;
- PI-Agent wall-clock timeout: 1,800 seconds.

The response cap leaves 5,751 tokens (13.25%) above the observed completed maximum,
while avoiding the wasted KV-cache capacity of the earlier 94,208-token sampling
ceiling. The timeout wraps the complete PI-Agent run, including vLLM admission,
generation and tool execution. The in-flight limit is 48 trajectories, below the
64 physical TP4/DP4 sequence slots, so normal vLLM admission does not create an
unbounded queue outside that timer.

## Data and reward gates

The frozen data contract validates 70 unique identities, 140 schedule rows, exactly
two appearances per identity, and source counts 21/20/29 from machines 0/5/6. The
v15/v20/v21 counts are 12/39/19; difficulty levels 1–5 contain 2/25/17/16/10 tasks.
Both machines independently validated the data hashes and the three runtime sandbox
environments, including read-only SQLite access and required documents.

Training uses `banded-v2-strict-table-v1`. Structured table answers must preserve
category/value binding, cardinality and uniqueness; executable read-only SQL replay
is cached and time-bounded. Although only 20 tasks retained strict mixed variance in
the historical replay, the owner explicitly authorized all 70 for this run. Model
promotion remains disabled and requires a separate held-out decision.

## Checkpoint policy

Only the final step-70 model checkpoint is saved. Retention is one checkpoint and
the payload is `model,extra`; optimizer state is intentionally omitted. Validation
is disabled because all 70 authorized tasks are used for training and no held-out
rows are silently reused.

## Launch

Formal run name: `llin-qwen38-grpo-train70-2x-banded-v2-20260818-01`.
The unattended supervisor was started on machine 5 on 2026-08-18 and performs data,
sandbox, model, idle-NPU, Ray topology and 1×16 HCCL fan-out gates before entering
the training process. Sensitive Parquet files, prompts, gold values, SQL, task IDs,
server paths, checkpoints and raw logs are not committed to Git.
