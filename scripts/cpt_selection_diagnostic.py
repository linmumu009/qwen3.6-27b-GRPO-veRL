"""Fixed output-form diagnostic; prompt construction never consumes labels."""
import copy
import json
from collections import Counter
from cpt_two_family import digest
from evaluate_logistics_knowledge import parse_answers


def messages(original, form, position=None):
    result = copy.deepcopy(original)
    if form == "S":
        assert position is None
        return result
    assert form in ("V", "B") and len(result) == 2
    assert result[1]["content"].count("Select all correct statements.") == 1
    result[1]["content"] = result[1]["content"].replace("Select all correct statements.", "Assess the statements under the stated scope.")
    common = "Judge whether the displayed statements are supported under the question's stated scope. Return exactly one JSON object with only the key \"supported\". Do not include reasoning, markdown, or other keys. "
    if form == "V":
        assert position is None
        result[0]["content"] = common + "Its value must be an array of exactly four JSON booleans, one for each displayed option in order from index 0 to index 3. True means supported; false means not supported. Judge every option."
    else:
        assert type(position) is int and 0 <= position < 4
        result[0]["content"] = common + "Its value must be one JSON boolean: true if the designated option is supported, or false if it is not supported. All options remain visible as context; judge only the designated option."
        result[1]["content"] += f"\n\nDesignated option to judge: [{position}]."
    return result


def strict_supported(text, form):
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError("duplicate key")
            obj[k] = v
        return obj
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        if type(value) is not dict or set(value) != {"supported"}:
            return None
        value = value["supported"]
        if form == "V":
            return value if type(value) is list and len(value) == 4 and all(type(x) is bool for x in value) else None
        if form == "B":
            return value if type(value) is bool else None
        raise ValueError(form)
    except (ValueError, TypeError):
        return None


def score(packet, rows):
    assert len(rows) == len(packet["requests"]) == 144
    outputs = {r["id"]: r for r in rows}
    assert len(outputs) == 144 and set(outputs) == {r["id"] for r in packet["requests"]}
    units = {}
    request_stats = {f: Counter() for f in ("S", "V", "B")}
    for req in packet["requests"]:
        row = outputs[req["id"]]
        assert row["messages_sha256"] == digest(req["messages"])
        form = req["form"]
        stats = request_stats[form]
        stats["calls"] += 1
        stats["output_tokens"] += row["output_tokens"]
        truncated = row["finish_reason"] == "length"
        stats["truncated"] += truncated
        if form == "S":
            indices, valid = parse_answers(row["text"], 4)
            value = [i in indices for i in range(4)] if valid else None
        else:
            value = strict_supported(row["text"], form)
            valid = value is not None
        stats["invalid"] += not valid
        key = (req["task"], req["shift"], req["repeat"], form)
        unit = units.setdefault(key, dict(task=req["task"], shift=req["shift"], repeat=req["repeat"], form=form,
                                       values=[None]*4, valid=True))
        unit["valid"] &= valid and not truncated
        if form == "B":
            unit["values"][req["position"]] = value if valid and not truncated else None
        else:
            unit["values"] = value if valid and not truncated else [None]*4
    conditions = {(g["id"], shift): c for g in packet["groups"] for shift, c in enumerate(g["conditions"])}
    ledger = []
    for key, unit in sorted(units.items()):
        c = conditions[(unit["task"], unit["shift"])]
        expected = [i in c["expected"] for i in range(4)]
        unit["valid"] &= all(type(x) is bool for x in unit["values"])
        unit["correct"] = unit["valid"] and unit["values"] == expected
        unit["omitted_original"] = [c["order"][i] for i in range(4) if expected[i] and unit["values"][i] is False]
        unit["false_positive_original"] = [c["order"][i] for i in range(4) if not expected[i] and unit["values"][i] is True]
        ledger.append(unit)
    summary = {}
    for form in ("S", "V", "B"):
        selected = [r for r in ledger if r["form"] == form]
        main = [r for r in selected if r["repeat"] == 0]
        by_task = []
        for task in ("MT1", "MT2", "MT3", "MT4"):
            primary = [r for r in main if r["task"] == task]
            first = sorted((r for r in selected if r["task"] == task and r["shift"] == 0), key=lambda r:r["repeat"])
            assert len(primary) == 4 and len(first) == 3
            by_task.append(dict(id=task, stable_four=all(r["correct"] for r in primary),
                                first_three_correct=[r["correct"] for r in first],
                                first_three_valid=all(r["valid"] for r in first),
                                first_three_vectors_equal=all(r["valid"] for r in first) and all(r["values"] == first[0]["values"] for r in first),
                                main_correct=sum(r["correct"] for r in primary)))
        summary[form] = dict(request_stats[form], stable_tasks=sum(t["stable_four"] for t in by_task),
                             main_complete_sets_correct=sum(r["correct"] for r in main), main_complete_sets=16,
                             all_complete_sets_correct=sum(r["correct"] for r in selected), all_complete_sets=24,
                             observed_omissions=sum(len(r["omitted_original"]) for r in selected),
                             observed_false_positives=sum(len(r["false_positive_original"]) for r in selected),
                             tasks=by_task)
    return ledger, summary
