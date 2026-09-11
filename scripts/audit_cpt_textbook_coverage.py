"""Read-only lexical evidence retrieval; candidates are NOT semantic coverage verdicts.

Private cases and excerpts must stay outside Git. Requires scikit-learn.
Uses expected options only to locate evidence, never produces training data.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def read(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blocks = read(args.corpus / "staging.jsonl")
    train = read(args.corpus / "train.jsonl")
    sources = {s["source_id"]: s for s in read(args.corpus / "sources.jsonl")}
    retained = collections.defaultdict(list)
    for row in train:
        for bid in row["block_ids"]:
            retained[bid].append(row["id"])
    rejected = collections.defaultdict(set)
    for row in read(args.corpus / "quarantine.jsonl"):
        for bid in row["block_ids"]:
            rejected[bid].update(row["reasons"])
    stats = {}
    for sid, src in sources.items():
        subset = [b for b in blocks if b["source_id"] == sid]
        counts = collections.Counter()
        for b in subset:
            counts["staging_blocks"] += 1
            counts["staging_words"] += len(b["text"].split())
            if b["block_id"] in retained:
                counts["retained_blocks"] += 1
                counts["retained_words"] += len(b["text"].split())
            else:
                for reason in rejected[b["block_id"]]:
                    counts["excluded_reason:" + reason] += 1
        stats[sid] = {"filename": src["filename"], **counts}
    # Adjacent context makes short headings and multi-line definitions searchable.
    documents = []
    for i, b in enumerate(blocks):
        neighbors = [n["text"] for n in blocks[max(0, i-1):i+2]
                     if n["source_id"] == b["source_id"] and n["page"] == b["page"]]
        documents.append("\n".join(neighbors))
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 sublinear_tf=True, max_features=400000)
    matrix = vectorizer.fit_transform(documents)
    cases = read(args.cases)
    wrong = [c for c in cases if not c["after"]]
    queries = []
    for case in wrong:
        answer = " ".join(str(case["options"][i]) for i in case["expected"])
        # Some benchmark families reuse an unrelated containerisation preamble.
        question = case["question"].split("Which of the following best matches:")[-1]
        queries.append(question + " " + answer)
    scores = vectorizer.transform(queries) @ matrix.T
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "retrieval.private.jsonl").open("w", encoding="utf-8") as out:
        for ci, case in enumerate(wrong):
            row_scores = scores.getrow(ci).toarray().ravel()
            order = np.argsort(-row_scores, kind="stable")
            candidates, seen = [], set()
            for i in order:
                b = blocks[int(i)]
                key = (b["source_id"], b["page"])
                if key in seen or row_scores[i] <= 0:
                    continue
                seen.add(key)
                context = []
                for n in blocks[max(0, int(i)-2):int(i)+3]:
                    if n["source_id"] == b["source_id"] and n["page"] == b["page"]:
                        context.append({"block_id": n["block_id"], "text": n["text"],
                                        "retained": n["block_id"] in retained,
                                        "train_ids": retained.get(n["block_id"], []),
                                        "exclusion_reasons": sorted(rejected[n["block_id"]])})
                candidates.append({"source": sources[b["source_id"]]["filename"],
                                   "page": b["page"], "score": float(row_scores[i]),
                                   "context": context})
                if len(candidates) == 8:
                    break
            out.write(json.dumps({**case, "candidates": candidates}, ensure_ascii=False) + "\n")
    hashes = {name: hashlib.sha256((args.corpus / name).read_bytes()).hexdigest()
              for name in ["train.parquet", "train.jsonl", "staging.jsonl", "quarantine.jsonl"]}
    summary = {"method": "TF-IDF lexical candidates with adjacent blocks; no semantic verdict",
               "limitations": "Staging excludes unextracted visual content; word retention is not knowledge coverage. Reasons overlap. Page numbers are PDF physical pages; ZIP page 0 is not a citation.",
               "cases": len(cases), "still_wrong": len(wrong),
               "wrong_by_dataset": dict(collections.Counter(c["dataset"] for c in wrong)),
               "corpus_sha256": hashes, "sources": stats}
    (args.output / "retrieval_summary.safe.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
