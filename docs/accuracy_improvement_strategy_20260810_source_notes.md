# Accurate-model strategy source notes

## Reporting job

- Question: how to improve model answer accuracy after the Step 120 dense trial and force-final sentinel.
- Audience: product stakeholders making the next experiment decision.
- Decision: which intervention to run before spending another 20–100 online GRPO steps or expanding context.
- Primary outcome: boss-original numeric correctness on fixed held-out tasks.
- Driver metrics: completion, process score, mixed-correct group rate, task-contract coverage, and final-answer/tool-result consistency.
- Baselines: Step 100, Step 120, and Step 200 on the same val20; the first and second 100-step training phases; force-final Run06 and Run07.

## Required-structure mapping

- Title: `title` block.
- Executive Summary: `executive_summary` block.
- Key findings with visual evidence: checkpoint, group-signal, decline-driver, force-final, oracle, supervision, GRPO, and reward sections.
- Recommended next steps: `recommended_steps` plus `priority_table`.
- Further questions: `further_questions`.
- Caveats and assumptions: `caveats`.

## Chart map

| Section | Question | Family / type | Fields | Supported claim | Palette policy |
| --- | --- | --- | --- | --- | --- |
| Checkpoint comparison | Did reward and correctness move together? | grouped categorical bar | checkpoint, metric, value | Step 120 improved reward/completion/process while exact correctness fell | relaxed multi-category; metric identity is meaningful |
| Group signal | How often could GRPO distinguish correct from wrong? | 100% stacked bar | phase, signal, rate | roughly 78% of groups were all wrong in both 100-step phases | relaxed multi-category with direct legend |
| Decline diagnosis | What explained the Step 100→200 loss? | horizontal ranked bar | driver, share | numeric correctness contributed the largest share of the net decline | single-root preferred; ordering carries rank |

All charts use discrete comparisons rather than line trends because the evidence contains only two or three anchor periods. Bars start at zero. Every chart is backed by bounded aggregate rows and has an adjacent interpretation paragraph in the report.

## Calculation and source checks

- Checkpoint task sets are identical and all use 20 tasks; prompt and ground-truth identities were previously audited.
- Numeric correctness rates were independently recomputed as `3/20`, `2/20`, and `1/20` for Step 100, 120, and 200.
- Group-signal shares reconcile exactly to 400 groups in each phase: `313+75+12=400` and `314+72+14=400`.
- Decline contribution shares sum to 1 within floating-point precision.
- Force-final comparisons are mechanism diagnostics, not population estimates. Guardrail reward drift is not attributed to the policy when the policy did not trigger.
- `2 groups × 8 responses` is a proposed canary. It keeps 16 trajectories per update relative to `4×4`, but its effect on mixed-correct rate is unverified.

## Source authority and conflicts

- Boss-original `reward_judge.py` output controls the reported numeric correctness and reward totals.
- Project online reward is retained as a training metric but does not control the accuracy conclusion when it disagrees with boss-original correctness.
- The force-final report controls intervention audit and exact scoring for Run06/Run07.
- Label-quality audit proves executable SQL and expected-value consistency, not full human semantic correctness. The 271/276 warning count is not treated as an error rate.

## Validation assessment

- Overall: Share with caveats.
- Verified: denominator, same-task comparisons, group arithmetic, reward-driver reconciliation, and force-final terminal/correctness distinction.
- Material caveats: val20 is small; force-final samples are targeted; the proposed oracle ladder, correction SFT, and `2×8` GRPO have not yet been run.
- No causal claim is made that one reward component alone caused Step 200 regression.
