"""Collect source-inclusive book pages and Wikipedia definitions for concept review.

No model call, training launch or corpus replacement. Lexical retrieval is only
evidence discovery; original page structure and exclusions remain auditable.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    out = args.directory
    topics = json.loads((out / "topics.private.json").read_text(encoding="utf-8"))
    cases = read(args.cases)
    assert len(cases) == 300 and {i for t in topics for i in t["case_indices"]} == set(range(len(cases)))
    sources = {r["source_id"]: r for r in read(args.corpus / "sources.jsonl")}
    grouped = defaultdict(list)
    for block in read(args.corpus / "staging.jsonl"):
        # Preserve short definitions, lists, mathematical text and figure labels.
        # Only recurring page furniture is removed, not source-family overlap.
        if block["text"] and "running_header_footer" not in block["flags"]:
            grouped[(block["source_id"], block.get("member") or block["page"])].append(block)
    pages = []
    for (sid, locator), blocks in grouped.items():
        if sum("front_matter" in b["flags"] or "back_matter" in b["flags"] for b in blocks) > len(blocks) / 2:
            continue
        text = "\n\n".join(b["text"] for b in blocks)
        pages.append({"id": hashlib.sha256(f"{sid}:{locator}".encode()).hexdigest()[:20],
                      "source_id": sid, "source_file": sources[sid]["filename"],
                      "locator": locator, "text": text, "block_ids": [b["block_id"] for b in blocks],
                      "additional_pages": sorted({p for b in blocks for p in b.get("additional_pages", [])}),
                      "original_flags": sorted({f for b in blocks for f in b["flags"]}),
                      "status": "candidate_requires_semantic_and_layout_review"})
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=350000)
    matrix = vectorizer.fit_transform(p["text"] for p in pages)
    selected = set()
    topic_rows = []
    for topic in topics:
        score = (vectorizer.transform([topic["query"]]) @ matrix.T).toarray().ravel()
        ranking = np.argsort(-score, kind="stable")
        candidates = [int(i) for i in ranking[:20] if score[i] >= .065]
        evidence = {i: {"page_id": pages[i]["id"], "score": float(score[i]), "via": ["topic_terms"]} for i in candidates}
        for prefix, locators in topic.get("book_anchor_pages", {}).items():
            for i, page in enumerate(pages):
                if page["source_file"].startswith(prefix) and page["locator"] in locators:
                    evidence.setdefault(i, {"page_id": page["id"], "score": float(score[i]), "via": []})["via"].append("editor_selected_source_context")
        for ci in topic["case_indices"]:
            case = cases[ci]
            question = case["question"].split("Which of the following best matches:")[-1]
            # Do not invent evidence for number-only, letter-only or broken stems.
            if len(re.findall(r"[A-Za-z]+", question)) < 3:
                continue
            query = question + " " + " ".join(str(case["options"][i]) for i in case["expected"])
            case_score = (vectorizer.transform([query]) @ matrix.T).toarray().ravel()
            for i in np.argsort(-case_score, kind="stable")[:3]:
                i = int(i)
                if case_score[i] >= max(.12, float(case_score.max()) * .65):
                    entry = evidence.setdefault(i, {"page_id": pages[i]["id"], "score": float(case_score[i]), "via": []})
                    entry["via"].append("case_" + str(ci))
        selected.update(evidence)
        topic_rows.append({**topic, "book_candidates": list(evidence.values())})
    (out / "book_pages.private.jsonl").write_text("".join(json.dumps(pages[i], ensure_ascii=False)+"\n" for i in sorted(selected)), encoding="utf-8")
    (out / "topic_evidence.private.json").write_text(json.dumps(topic_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"case_count": len(cases), "topic_count": len(topics), "sources_searched": len(sources),
               "parsed_page_or_html_units_searched": len(pages), "candidate_units_collected": len(selected),
               "candidate_units_by_source": dict(Counter(pages[i]["source_file"] for i in selected)),
               "warning": "Candidate retrieval is not exhaustive semantic coverage or training approval; formulas and visual dependencies remain explicitly flagged.",
               "original_train_parquet_sha256": hashlib.sha256((args.corpus / "train.parquet").read_bytes()).hexdigest()}
    (out / "collection.safe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
