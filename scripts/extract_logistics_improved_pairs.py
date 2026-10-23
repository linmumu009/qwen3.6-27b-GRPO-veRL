"""Extract private, paired native-wrong/best-correct benchmark records.

Uses saved inference responses only. Raw questions and responses must stay out
of version control; the separate safe manifest contains counts and hashes.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from scripts.evaluate_logistics_knowledge import parse_answers
from scripts.run_vllm_logistics_mcq import load_items


def verified_predictions(path, items):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexed = {row["item_hash"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != {item.item_hash for item in items}:
        raise ValueError("prediction coverage or uniqueness mismatch")
    for item in items:
        row = indexed[item.item_hash]
        for field, value in asdict(item).items():
            if isinstance(value, tuple):
                value = list(value)
            if row.get(field) != value:
                raise ValueError(f"prediction/source metadata mismatch: {field}")
        parsed, valid = parse_answers(row["prediction"], len(item.options))
        if (row.get("error") or row["correct"] is not (valid and parsed == item.expected)
                or row["parse_ok"] is not valid or row["parsed"] != list(parsed)):
            raise ValueError("failed request or inconsistent saved answer/scoring")
    return indexed


def response_record(row, options):
    parsed, valid = parse_answers(row["prediction"], len(options))
    return {
        "prediction": row["prediction"],
        "reasoning": row.get("reasoning") or "",
        "parsed": list(parsed),
        "parse_ok": valid,
        "correct": row["correct"],
        "answer_texts": [options[i] for i in parsed] if valid else [],
    }


def improved_pairs(items, native, best):
    pairs = []
    counts = Counter()
    for item in items:
        n, b = native[item.item_hash], best[item.item_hash]
        if n["correct"] or not b["correct"]:
            continue
        counts[item.dataset] += 1
        prefix = {"LogistikaBench": "LOG", "SC-bench-knowledge": "SC"}[item.dataset]
        pairs.append({
            **asdict(item),
            "pair_id": f"{prefix}-{counts[item.dataset]:04d}",
            "gold_texts": [item.options[i] for i in item.expected],
            "native": response_record(n, item.options),
            "best": response_record(b, item.options),
        })
    return pairs


def extract(source, native_path, best_path, selection_path, *, expected_best_step):
    items = load_items(source)
    if Counter(i.dataset for i in items) != {"LogistikaBench": 1446, "SC-bench-knowledge": 226}:
        raise ValueError("full benchmark coverage mismatch")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection["best"]
    if chosen["step"] != expected_best_step:
        raise ValueError("selected best step differs from the requested final best")
    native = verified_predictions(native_path, items)
    best = verified_predictions(best_path, items)
    if chosen["correct"] != sum(row["correct"] for row in best.values()):
        raise ValueError("best prediction score differs from selection record")
    actual = Counter(i.dataset for i in items if best[i.item_hash]["correct"])
    if actual != {d["dataset"]: d["correct"] for d in chosen["by_dataset"]}:
        raise ValueError("best per-dataset scores differ from selection record")
    pairs = improved_pairs(items, native, best)
    safe = {
        "date": "2026-10-23", "private_content_included": False,
        "filter": "native incorrect AND selected final best correct",
        "native_model": "Qwen3.6-27B original local base (not Step120)",
        "best_additional_step": chosen["step"],
        "best_cumulative_sft_step": 103 + chosen["step"],
        "independent_test": False,
        "evaluation_items": len(items), "selected_items": len(pairs),
        "by_dataset": dict(Counter(p["dataset"] for p in pairs)),
        "native_correct": sum(row["correct"] for row in native.values()),
        "best_correct": sum(row["correct"] for row in best.values()),
        "selected_max_option_count": max(len(p["options"]) for p in pairs),
        "selected_option_rows": dict(Counter({d: sum(len(p["options"]) for p in pairs
            if p["dataset"] == d) for d in actual})),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in [("cases", source), ("native", native_path),
                               ("best", best_path), ("selection", selection_path)]},
        "all_saved_predictions_reparsed": True,
        "all_question_metadata_verified": True,
        "all_options_retained": True,
        "answers_resolved_to_full_option_text": True,
        "same_order_and_pair_ids_in_each_dataset": True,
    }
    return {"manifest": safe, "pairs": pairs}, safe


def export_json_files(payload, output_dir):
    """Write four UTF-8 JSON documents, with one model's actual answers each."""
    documents = {}
    for dataset, label in [("LogistikaBench", "LogistikaBench"),
                           ("SC-bench-knowledge", "SC-bench知识题")]:
        pairs = [p for p in payload["pairs"] if p["dataset"] == dataset]
        for role, role_label in [("native", "原始基座"), ("best", "Best_Model")]:
            questions = []
            for pair in pairs:
                answer = pair[role]
                questions.append({
                    **{key: pair[key] for key in ("pair_id", "item_hash", "source_id",
                        "category", "question_type", "question", "options")},
                    "gold_answer": {"indices": pair["expected"], "texts": pair["gold_texts"]},
                    "model_answer": {
                        "raw_response": answer["prediction"],
                        "indices": answer["parsed"], "texts": answer["answer_texts"],
                        "reasoning": answer["reasoning"],
                        "parse_ok": answer["parse_ok"], "correct": answer["correct"],
                    },
                })
            documents[output_dir / f"{label}_{role_label}.json"] = {
                "dataset": dataset, "model_role": role,
                "model": (payload["manifest"]["native_model"] if role == "native" else
                          f"Final Best Model, additional step {payload['manifest']['best_additional_step']}, "
                          f"cumulative SFT step {payload['manifest']['best_cumulative_sft_step']}"),
                "selection_filter": payload["manifest"]["filter"],
                "independent_test": payload["manifest"]["independent_test"],
                "option_index_base": 0, "question_count": len(questions),
                "questions": questions,
            }
    if any(path.exists() for path in documents):
        raise ValueError("refusing to overwrite existing JSON deliverables")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, document in documents.items():
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return list(documents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "native", "best", "selection", "output", "safe_output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--expected-best-step", type=int, required=True)
    parser.add_argument("--json-output-dir", type=Path)
    args = parser.parse_args()
    payload, safe = extract(args.source, args.native, args.best, args.selection,
                            expected_best_step=args.expected_best_step)
    for path, data in [(args.output, payload), (args.safe_output, safe)]:
        if path.exists():
            raise ValueError(f"refusing overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json_output_dir:
        export_json_files(payload, args.json_output_dir)
    print(json.dumps(safe, ensure_ascii=False))


if __name__ == "__main__":
    main()
