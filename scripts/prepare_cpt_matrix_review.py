"""Validate private draft structure and export a label-hidden review packet."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    matrix = read(root / "matrix.v2.private.json")
    sources = read(root / "source_only.private.json")
    train = read(root / "train_author.private.json")["candidates"]
    development = read(root / "development_author.private.json")["items"]
    specs = {x["id"]: x for x in matrix["training_pairs"]}
    reserves = {x["id"]: x for x in matrix["reserved_transfer"]}
    texts = {x["family_id"]: x["evidence_text"] for x in sources["families"]}
    assert len(train) == 20 and {x["id"] for x in train} == {f"{k}_{arm}" for k in specs for arm in ("D", "T")}
    assert len(development) == 8 and {x["id"] for x in development} == set(reserves)
    rows = []
    for task in train + development:
        assert len(task["options"]) == len(task["option_reasons"]) == 4
        assert len({x.strip().casefold() for x in task["options"]}) == 4
        indices = task["correct_indices"]
        assert 0 < len(indices) < 4 and indices == sorted(set(indices))
        assert all(type(i) is int and 0 <= i < 4 for i in indices)
        assert task["source_quotes"] and all(q and q in texts[task["family"]] for q in task["source_quotes"]), task["id"]
        if task["id"] not in reserves:
            pair = specs[task["pair_id"]]
            assert task["id"] == f'{task["pair_id"]}_{task["arm"]}'
            assert task["family"] == pair["family"]
            assert task["fact_ids"] == pair[task["arm"]]["atomic_fact_ids"]
            assert task.get("explanation")
        else:
            assert task["family"] == reserves[task["id"]]["family"]
            assert task["measurement_tier"] == reserves[task["id"]]["measurement_tier"]
            assert task.get("scored_requirements") and task.get("requirement_option_map")
        # Do not expose author-selected evidence or any label-bearing explanations.
        rows.append({k: task[k] for k in ("id", "family", "question", "options")})
    result = {"sources": sources, "tasks": rows,
              "input_sha256": {n: sha(root / n) for n in ("train_author.private.json", "development_author.private.json", "source_only.private.json")}}
    destination = root / "blind_review_input.private.json"
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if destination.exists():
        assert destination.read_text(encoding="utf-8") == payload, "Frozen review input differs"
    else:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    print(json.dumps({"drafts": len(rows), "blind_packet_sha256": sha(destination)}))


if __name__ == "__main__":
    main()
