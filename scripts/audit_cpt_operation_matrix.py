"""Audit a private operation specification; export only counts and fingerprints."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def audit(matrix, review):
    errors = []
    pairs = matrix.get("training_pairs", [])
    reserved = matrix.get("reserved_transfer", [])
    ids = [r["id"] for r in pairs]
    expected = {f"{p}{i}" for p in ("U", "T") for i in range(1, 6)}
    if len(ids) != 10 or set(ids) != expected:
        errors.append("training IDs differ from the frozen ten specifications")
    if sorted(Counter(r["family"] for r in pairs).values()) != [5, 5]:
        errors.append("family allocation differs")
    for row in pairs:
        d, t = row.get("D", {}), row.get("T", {})
        facts = d.get("atomic_fact_ids", [])
        if not facts or len(set(facts)) != len(facts) or facts != t.get("atomic_fact_ids"):
            errors.append(f'{row["id"]}: unmatched atomic facts')
        if not d.get("concrete_input_state") or not row.get("exposure_limitation"):
            errors.append(f'{row["id"]}: unbound state or missing exposure limitation')
    reserve_ids = [r["id"] for r in reserved]
    if len(reserve_ids) != 8 or set(reserve_ids) != {f"M{p}{i}" for p in ("U", "T") for i in range(1, 5)}:
        errors.append("reserved IDs differ")
    for row in reserved:
        if not row.get("components") or not set(row["components"]) <= set(ids):
            errors.append(f'{row["id"]}: invalid teaching dependency')
    for section, required, flags in [
        ("teaching", ids, ("pass", "source_supported", "contrast_genuine", "no_overclaim")),
        ("transfer", reserve_ids, ("pass", "source_supported", "allocation_distinct")),
    ]:
        rows = review.get(section, [])
        if len(rows) != len(required) or {r["id"] for r in rows} != set(required):
            errors.append(f"{section}: incomplete review")
        for row in rows:
            if any(row.get(flag) is not True for flag in flags):
                errors.append(f'{row["id"]}: review did not pass')
    if review.get("overall_authoring_release") is not True or review.get("remaining_blockers") != []:
        errors.append("independent reviewer has not released authoring")
    policy = matrix.get("policy", {})
    if policy.get("training_allowed") is not False or policy.get("development_measurement_not_independent_holdout") is not True:
        errors.append("scope boundaries missing")
    return {"schema": 1, "authoring_released": not errors, "errors": errors,
            "training_pair_specifications": len(pairs), "reserved_development_specifications": len(reserved),
            "training_allowed": False, "student_generation_calls": 0,
            "independent_holdout": False, "state_exposures_matched": False}


def main():
    p = argparse.ArgumentParser()
    for name in ("matrix", "review", "source", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    args = p.parse_args()
    raw = {name: getattr(args, name).read_bytes() for name in ("matrix", "review", "source")}
    matrix, review = (json.loads(raw[name]) for name in ("matrix", "review"))
    result = audit(matrix, review)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    for name in ("matrix", "source"):
        if review.get(f"reviewed_{name}_sha256") != hashes[name]:
            result["errors"].append(f"review does not bind current {name}")
            result["authoring_released"] = False
    if hashes["source"] != matrix.get("source_sha256"):
        result["errors"].append("source fingerprint mismatch")
        result["authoring_released"] = False
    result["sha256"] = hashes
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["authoring_released"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
