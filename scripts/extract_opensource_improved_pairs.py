"""Read historical team evaluations; export private paired improvements.

Uses recorded scores, without inference, executing benchmark answers, or changing
server files. Conflicting model labels are kept in a separate review collection.
"""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import zipfile

import yaml


BASE_TAG = "qwen36-27b"
TRAINED_TAG = "llin-step120-opensource"


def read_jsonl(path):
    # splitlines also splits U+2028/U+0085 inside valid JSON string values.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]
    indexed = {r["task_id"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate task_id: {path}")
    return indexed


def config_metadata(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    vllm = config.get("vllm", {})
    benchmark = config["benchmark"]
    # Deliberately exclude API keys, environment variables and service URLs.
    return {
        "model": {k: config.get("model", {}).get(k) for k in ("tag", "name", "path")},
        "benchmark": {k: benchmark.get(k) for k in
                      ("name", "data", "type", "shot", "max_tokens", "scorer")},
        "inference": {k: vllm.get(k) for k in ("temperature", "enable_thinking", "mtp")},
        "seed": (config.get("sampling") or {}).get("seed"),
    }


def load_run(directory):
    predictions = read_jsonl(directory / "predictions.jsonl")
    scores = read_jsonl(directory / "scores.jsonl")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if set(predictions) != set(scores) or len(scores) != summary["total"]:
        raise ValueError(f"prediction/score/summary coverage mismatch: {directory}")
    if any(type(s["correct"]) is not bool for s in scores.values()):
        raise ValueError("correctness flags must be booleans")
    if sum(s["correct"] for s in scores.values()) != summary["correct"]:
        raise ValueError(f"recorded summary score mismatch: {directory}")
    metadata = config_metadata(directory / "config_snapshot.yaml")
    if any(r["benchmark"] != summary["benchmark"] for r in predictions.values()):
        raise ValueError("prediction benchmark mismatch")
    return predictions, scores, summary, metadata


def compare_runs(native, trained, dataset, mode, *, pending=False):
    np, ns, nsummary, nc = load_run(native)
    tp, ts, tsummary, tc = load_run(trained)
    if set(np) != set(tp):
        raise ValueError("paired task coverage mismatch")
    if nsummary["benchmark"] != dataset or tsummary["benchmark"] != dataset:
        raise ValueError("paired dataset mismatch")
    max_tokens = int(mode.rsplit("_", 1)[1])
    if nc["benchmark"]["max_tokens"] != max_tokens or tc["benchmark"]["max_tokens"] != max_tokens:
        raise ValueError("output budgets differ from selected mode")
    for task_id in np:
        if ns[task_id]["golden_answer"] != ts[task_id]["golden_answer"]:
            raise ValueError("paired gold answers differ")
    prompt_mismatches = sum(np[k]["prompt"] != tp[k]["prompt"] for k in np)
    tag_conflict = nc["model"]["tag"] != BASE_TAG or tc["model"]["tag"] != TRAINED_TAG
    audit = {
        "dataset": dataset, "mode": mode, "items": len(np),
        "native_correct": nsummary["correct"], "trained_correct": tsummary["correct"],
        "recorded_gains": sum(not ns[k]["correct"] and ts[k]["correct"] for k in np),
        "recorded_losses": sum(ns[k]["correct"] and not ts[k]["correct"] for k in np),
        "prompt_mismatches": prompt_mismatches, "model_tag_conflict": tag_conflict,
        "native_run": str(native), "trained_run": str(trained),
        "native_config": nc, "trained_config": tc,
        "status": "pending_model_identity" if pending else "included",
        "excluded_candidate_reasons": {},
    }
    if prompt_mismatches:
        audit.update(status="excluded_prompt_mismatch", exported_items=0)
        return [], audit
    if tag_conflict and not pending:
        audit.update(status="excluded_model_tag_conflict", exported_items=0)
        return [], audit
    excluded = Counter()
    pairs = []
    for task_id in np:
        if ns[task_id]["correct"] or not ts[task_id]["correct"]:
            continue
        if (np[task_id].get("error") or tp[task_id].get("error") or
                np[task_id]["finish_reason"] == "error" or tp[task_id]["finish_reason"] == "error"):
            excluded["request_error"] += 1
            continue
        if tp[task_id]["finish_reason"] == "length":
            excluded["trained_response_truncated"] += 1
            continue
        pairs.append({
            "pair_id": f"{dataset}-{mode}-{task_id}", "dataset": dataset, "mode": mode,
            "task_id": task_id, "question": np[task_id]["prompt"],
            "question_sha256": hashlib.sha256(np[task_id]["prompt"].encode()).hexdigest(),
            "gold_answer": ns[task_id]["golden_answer"],
            "native": {"prediction": np[task_id], "scoring": ns[task_id]},
            "trained": {"prediction": tp[task_id], "scoring": ts[task_id]},
            "source_status": "模型归属待核验" if pending else "按历史模型标签及逐题评分配对",
        })
    audit["exported_items"] = len(pairs)
    audit["excluded_candidate_reasons"] = dict(excluded)
    if not pairs and not pending:
        audit["status"] = "no_complete_improvement"
    return pairs, audit


def find_runs(root, model, mode):
    # Latest canonical archived run, chosen by name, never by achieved accuracy.
    found = {}
    for path in sorted((root / model / mode).glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if all((path.parent / name).is_file() for name in
               ("predictions.jsonl", "scores.jsonl", "config_snapshot.yaml")):
            found[summary["benchmark"]] = path.parent
    return found


def collect(root):
    mode = "nothink_8192"
    native = find_runs(root, BASE_TAG, mode)
    trained = find_runs(root, TRAINED_TAG, mode)
    cases, review, audits = [], [], []
    for dataset in sorted(set(native) & set(trained)):
        if dataset == "LogistikaBench" or dataset.startswith("SC-bench"):
            continue  # Separate logistics comparison already delivered.
        pairs, audit = compare_runs(native[dataset], trained[dataset], dataset, mode)
        cases.extend(pairs)
        audits.append(audit)
    # Preserve the specifically requested AIME examples without asserting that
    # a directory label overrides their conflicting baseline model snapshot.
    nr = find_runs(root, BASE_TAG, "nothink_30720")
    tr = find_runs(root, TRAINED_TAG, "nothink_30720")
    for dataset in ("AIME24", "AIME25"):
        if dataset in nr and dataset in tr:
            pairs, audit = compare_runs(nr[dataset], tr[dataset], dataset,
                                         "nothink_30720", pending=True)
            review.extend(pairs)
            audits.append(audit)
    sources = {}
    for audit in audits:
        for role in ("native", "trained"):
            directory = Path(audit[role + "_run"])
            sources[str(directory)] = {f: hashlib.sha256((directory / f).read_bytes()).hexdigest()
                for f in ("predictions.jsonl", "scores.jsonl", "summary.json", "config_snapshot.yaml")}
    safe = {
        "date": "2026-10-24", "private_content_included": False,
        "native_model_tag": BASE_TAG, "trained_model_tag": TRAINED_TAG,
        "trained_model": "Qwen3.6-27B GRPO Step120-opensource, not logistics SFT best150",
        "selection": "recorded native wrong and trained correct, identical prompts and golds, no API errors or truncated trained answers",
        "primary_mode": mode, "score_basis": "existing archived per-task scores; no fresh inference or benchmark code execution",
        "selected_items": len(cases), "by_dataset": dict(Counter(c["dataset"] for c in cases)),
        "pending_aime_items": len(review), "pending_by_dataset": dict(Counter(c["dataset"] for c in review)),
        "audits": audits, "source_sha256": sources,
        "limitations": [
            "Historical labels and snapshots do not independently prove which weights were served.",
            "GSM8K's trained model snapshot has a PLACEHOLDER path; its archived model tag is retained.",
            "8K AIME24 has only a scored-correct truncated candidate; 8K AIME25 has no gains.",
            "30K AIME model labels conflict with the baseline model snapshots, so those cases are separate pending review.",
            "8K C-Eval-val, CMMLU and MMLU-Pro prompts differ between models and are excluded.",
            "Selected improvements are not net score gains and do not establish causality from training.",
        ],
    }
    return {"manifest": safe, "cases": cases, "pending_aime": review}


def confirm_aime_identity(payload):
    """Apply only after the human explicitly confirms the AIME serving identity."""
    confirmation = {
        "date": "2026-10-24", "source": "explicit user reply in this conversation",
        "confirmed_model_tag": TRAINED_TAG, "reply": "确认是开源续训 Step120",
        "snapshot_conflict_preserved": True,
    }
    for case in payload["pending_aime"]:
        case["source_status"] = "用户确认开源续训Step120；原配置快照标签不一致"
        payload["cases"].append(case)
    payload["pending_aime"] = []
    safe = payload["manifest"]
    for audit in safe["audits"]:
        if audit["status"] == "pending_model_identity" and audit["dataset"] in ("AIME24", "AIME25"):
            audit["status"] = "included_user_confirmed_identity"
            audit["identity_confirmation"] = confirmation
    safe.update(selected_items=len(payload["cases"]),
                by_dataset=dict(Counter(c["dataset"] for c in payload["cases"])),
                pending_aime_items=0, pending_by_dataset={},
                aime_identity_confirmation=confirmation)
    safe["limitations"] = [note for note in safe["limitations"]
        if not note.startswith("30K AIME model labels")]
    safe["limitations"].append("30K AIME identity was confirmed by the user; conflicting configuration snapshots remain unchanged and documented.")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stdout-zip", action="store_true")
    parser.add_argument("--confirm-aime-step120", action="store_true",
                        help="Use only after explicit human confirmation of the historical AIME model identity")
    args = parser.parse_args()
    payload = collect(args.root)
    if args.confirm_aime_step120:
        payload = confirm_aime_identity(payload)
    if args.stdout_zip:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("paired_cases.private.json", json.dumps(payload, ensure_ascii=False))
            archive.writestr("audit.safe.json", json.dumps(payload["manifest"], ensure_ascii=False, indent=2))
        print(base64.b64encode(buffer.getvalue()).decode())
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
