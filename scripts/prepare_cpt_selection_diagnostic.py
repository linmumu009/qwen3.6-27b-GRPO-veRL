"""Recompute historical omissions and freeze the independently reviewed protocol."""
import argparse
from pathlib import Path
from cpt_two_family import read, sha, freeze, digest
from evaluate_logistics_knowledge import parse_answers
from cpt_selection_diagnostic import messages


def prepare(root, old):
    reg = read(old / "execution.safe.json")
    assert sha(old / "packet.private.json") == reg["packet_sha256"]
    assert sha(old / "remote_run/p1/predictions.private.json") == read(old / "remote_run/p1/completed.safe.json")["predictions_sha256"]
    prior = read(old / "packet.private.json")
    raw = {r["id"]: r for r in read(old / "remote_run/p1/predictions.private.json")}
    groups, reuse, errors = [], [], []
    for task in ("MT1", "MT2", "MT3", "MT4"):
        g = next(g for g in prior["groups"] if g["id"] == task)
        groups.append(dict(id=task, conditions=[g["conditions"][f"closed:{i}"] for i in range(4)]))
        for shift in range(4):
            c = g["conditions"][f"closed:{shift}"]
            reuse.append(dict(messages=c["messages"], historical_prompt_token_sha256=raw[f"{task}:closed:{shift}"]["prompt_token_sha256"]))
        for key, c in g["conditions"].items():
            r = raw[task + ":" + key]
            assert r["messages_sha256"] == digest(c["messages"])
            parsed, valid = parse_answers(r["text"], 4)
            assert valid and r["finish_reason"] != "length"
            if set(parsed) != set(c["expected"]):
                errors.append(dict(id=r["id"], batch_index=next(i for i,x in enumerate(prior["requests"]) if x["id"]==r["id"])//16,
                                   omitted_display=sorted(set(c["expected"])-set(parsed)),
                                   omitted_original=sorted(c["order"][i] for i in set(c["expected"])-set(parsed)),
                                   false_positive_original=sorted(c["order"][i] for i in set(parsed)-set(c["expected"]))))
    requests = []
    for repeat in range(3):
        for g in groups:
            for shift in (range(4) if repeat == 0 else (0,)):
                original = g["conditions"][shift]["messages"]
                for form, position in [("S",None), ("V",None)] + [("B",i) for i in range(4)]:
                    requests.append(dict(id=f'{g["id"]}:{shift}:{repeat}:{form}:{position}', task=g["id"], shift=shift,
                                         repeat=repeat, form=form, position=position, messages=messages(original,form,position)))
    assert len(requests) == 144
    packet = dict(groups=groups, requests=requests, reuse=reuse, training_allowed=False)
    root.mkdir(exist_ok=True)
    freeze(root / "packet.private.json", packet)
    freeze(root / "historical_reanalysis.safe.json", dict(calls_reused_for_offline_analysis=40, errors=errors,
        all_five_errors_within_batch_zero=all(e["batch_index"]==0 for e in errors),
        limitation="Only observed omission identities/positions/batch membership; not a causal attribution.",
        inputs={"packet":sha(old/"packet.private.json"),"predictions":sha(old/"remote_run/p1/predictions.private.json")}))


def release(root, old):
    review = read(root / "protocol_review.private.json")
    assert review["passed"] is True and review["remaining_blockers"] == []
    paths = {"packet.private.json":root/"packet.private.json",
             "registration":Path("docs/cpt_selection_diagnostic_registration_20260923.md"),
             "builder":Path("scripts/prepare_cpt_selection_diagnostic.py"),
             "protocol":Path("scripts/cpt_selection_diagnostic.py"),
             "runner":Path("scripts/run_cpt_selection_diagnostic.py")}
    for key, path in paths.items():
        assert review["input_sha256"][key] == sha(path), f"reviewed input changed: {key}"
    quality = dict(passed=True, inputs={k:sha(p) for k,p in paths.items()}, review_sha256=sha(root/"protocol_review.private.json"), training_allowed=False)
    freeze(root / "quality.safe.json", quality)
    previous = read(old / "execution.safe.json")
    reg = dict(id="llin-selection-diagnostic-20260923-01", max_calls=144,training_allowed=False, quality_released=True,
               quality_sha256=sha(root/"quality.safe.json"), packet_sha256=sha(root/"packet.private.json"),
               runner_sha256=sha(paths["runner"]),protocol_code_sha256=previous["protocol_code_sha256"],model_path=previous["model_path"],
               temperature=0,seed=1024,max_tokens=96,max_model_len=8192,tp=8,max_num_seqs=32,
               registered_plan_sha256=sha(paths["registration"]), calls_by_form={"S":24,"V":24,"B":96}, historical_token_prompts_checked=16)
    freeze(root / "execution.safe.json",reg)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--old",type=Path,default=Path("CPT_resources/llin-operation-matrix-20260923-01"))
    p.add_argument("--release",action="store_true")
    a=p.parse_args()
    release(a.root,a.old) if a.release else prepare(a.root,a.old)
