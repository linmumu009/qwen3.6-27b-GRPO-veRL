# Qwen3.8 canary5 first-batch stall — safe source notes

Snapshot time: 2026-08-23 13:09:45 +08:00.

## Current run evidence

- Run identity: `llin-qwen38-approved43-tiered-v1-canary5-20260823-02`.
- Supervisor start: 09:01:17 +08:00; state remained `training_canary_active`; no exit file existed at the snapshot.
- The first nominal training step emitted one `[LLIN_TRAIN_STAGE]` record at 10:06:21 +08:00:
  - queue wait: 1,618.780555 s;
  - deserialize: 0.045035 s;
  - assemble: 0.038809 s;
  - reward: 0.000039 s;
  - reference log-prob: 111.877229 s;
  - advantage: 0.004915 s;
  - actor update: 0.000000 s;
  - total step: 1,730.945782 s.
- Rollout monitor after the step remained at two produced group-samples, message queue size zero, and pending queue size nineteen. In this runtime one produced group-sample is one eight-trajectory GRPO group.
- At the snapshot, both hosts' 16 NPU entries reported AICore utilization 0%. The trainer host retained roughly 24.8–26.6 GiB HBM per card and the rollout host retained roughly 54.5–56.1 GiB per card, showing resident models without active accelerator compute.
- The task runner stack waited in `ray.wait()` at `fully_async_main.py:231`. The trainer, 16 `WorkerDict` actors, vLLM engines, and rollout workers were in event-loop, futex, or poll waits.
- The two observed `AttributeError` records were the optional Apex `MixedFusedLayerNorm` capability probe. They were not the earlier TensorDict or ref configuration failures and did not terminate the run.

## Causal code evidence

The live patched `FullyAsyncTrainer._fit_update_weights()` returns immediately when `actor/update_skipped_no_strict_mixed == 1.0`. The same function calls `rollouter.reset_staleness()` only after the weight-sync block. Therefore a skipped batch performs neither a weight broadcast nor the independent rollout-window reset.

With `staleness_threshold=0`, the rollouter produces exactly the current two-group allowance and then waits for `reset_staleness()`. Because the first batch had no eligible strict-mixed group, actor update was correctly skipped, policy version correctly stayed unchanged, and the early return prevented the rollouter from receiving a new same-policy allowance. This forms a deterministic liveness deadlock.

The early-return branch was introduced by commit `4e9591e` (`feat: launch Qwen3.8 tiered formal GRPO`). Existing tests asserted that the skip and weight-sync-skip markers were present, but did not execute the liveness sequence `uniform batch -> no optimizer -> same-policy rollout resumes`.

## Historical comparison

- Qwen3.8 engineering smoke with 16 trajectories: generation 816.20 s and actor update 296.09 s.
- Qwen3.8 mixed27 historical step: actor update 206.173 s and full step 1,378.85 s.
- Current first nominal step before the stall: 1,730.946 s. It is slower but within the same tens-of-minutes regime; it does not explain the multi-hour elapsed time.

## Derived timing at the snapshot

- Startup/model/sealed work before the first nominal step: approximately 36.22 min.
- First batch queue/rollout wait: 26.98 min.
- Reference log-prob: 1.86 min.
- Other first-step work: approximately 0.005 min.
- Post-skip deadlock: at least 183.40 min.
- Total elapsed: 248.47 min; the deadlock accounts for approximately 73.8%.

No task prompt, gold value, SQL text, tool return, trajectory identity, credential, or model parameter is included in these notes.

The portable report keeps the native chart contract in its artifact manifest and an exact stage-table fallback in the self-contained HTML. The shared native-chart extraction reader remained in its fallback state for both this report and a previously valid comparison artifact on the current host, so the official semantic fallback was retained; no alternate chart runtime was introduced.
