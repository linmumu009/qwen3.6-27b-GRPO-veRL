"""Reparse individual judgments and compare complete sets to joint variant zero."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import cpt_composed_probe as common
import run_cpt_atomic_probe as atomic


def parse(text):
    match=re.fullmatch(r'\s*\{\s*"correct"\s*:\s*(true|false)\s*\}\s*',text)
    return match.group(1)=='true' if match else None


def verify(packet,audit,result,joint_result):
    cases=atomic.selected_cases(packet,audit);by_id={c['id']:c for c in cases};base=Path(result)
    registration=json.loads((base/'registration.safe.json').read_text())
    assert registration['case_sha256']==common.sha(packet) and registration['semantic_review_sha256']==common.sha(audit)
    for key,value in dict(max_tokens=16,per_case_condition_total_output_cap=96,repeated_input_compute=6,max_model_len=8192,tp=8,max_num_seqs=16,thinking=False,temperature=0,seed=1024,paired_joint_variant=0).items():assert registration[key]==value
    rows=[json.loads(s) for s in (base/'predictions.private.jsonl').read_text().splitlines()]
    key=lambda r:(r['id'],r['condition'],r['target'])
    mapped={key(r):r for r in rows}
    expected={(c['id'],cond,i) for c in cases for cond in ('closed_book','source_evidence') for i in range(6)}
    assert len(mapped)==len(rows)==len(expected) and set(mapped)==expected
    for r in rows:
        c=by_id[r['id']];p=parse(r['text']);truth=r['target'] in c['expected']
        assert type(r['expected']) is bool and r['expected']==truth and p==r['parsed']
        assert r['correct']==(p is not None and p==truth and r['finish_reason']=='stop')
        assert r['prompt_text_sha256']==hashlib.sha256(atomic.prompt(c,r['condition'],r['target']).encode()).hexdigest()
        assert r['output_tokens']<=16 and r['prompt_tokens']+16<=8192
    old=[json.loads(s) for s in Path(joint_result).read_text(encoding='utf-8').splitlines()]
    old={(r['id'],r['condition']):r for r in old if r['variant']==0 and r['id'] in by_id}
    assert len(old)==len(cases)*2
    counts={cond:Counter() for cond in ('closed_book','source_evidence')};details=[]
    for c in cases:
        for cond in counts:
            batch=[mapped[(c['id'],cond,i)] for i in range(6)]
            joint=old[(c['id'],cond)]
            assert joint['order']==common.variants(c)[0]
            old_parsed=common.parse(joint['text'],6)
            old_set=sorted(joint['order'][i] for i in old_parsed) if old_parsed is not None else None
            joint_correct=old_set==c['expected'] and joint['finish_reason']=='stop'
            assert joint_correct==joint['correct']
            valid=all(r['parsed'] is not None and r['finish_reason']=='stop' for r in batch)
            atom_set=[i for i,r in enumerate(batch) if r['parsed'] is True]
            atom_correct=valid and atom_set==c['expected']
            stats=counts[cond];stats.update(claims=6,claims_correct=sum(r['correct'] for r in batch),
                cases=1,atomic_complete_correct=int(atom_correct),joint_complete_correct=int(joint_correct),
                joint_wrong_atomic_right=int(not joint_correct and atom_correct),joint_right_atomic_wrong=int(joint_correct and not atom_correct),
                invalid=sum(r['parsed'] is None for r in batch),truncated=sum(r['finish_reason']!='stop' for r in batch))
            details.append(dict(id=c['id'],condition=cond,category=c['category'],atomic_complete_correct=atom_correct,joint_complete_correct=joint_correct,claims_correct=sum(r['correct'] for r in batch),atomic_semantic_answers=atom_set,joint_semantic_answers=old_set))
    return dict(status='mechanically_verified',cases=len(cases),requests=len(rows),by_condition=counts,
        case_sha256=common.sha(packet),semantic_review_sha256=common.sha(audit),prediction_sha256=common.sha(base/'predictions.private.jsonl'),joint_prediction_sha256=common.sha(joint_result),
        raw_answers_independently_reparsed=True,text_prompts_reconstructed=True,token_hash_reconstruction='pending archived-tokenizer check',
        training_started=False,official_evaluation_changed=False,
        limitations='Diagnostic decomposition of the same source-based scenarios. Six16-token caps match one96-token output cap, but prompt input is repeated six times and computation differs. Not an independent benchmark or a training gain. Joint comparator uses only its first, preregistered option order.'),details


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['cases','audit','result','joint-result','out','details']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();summary,details=verify(a.cases,a.audit,a.result,a.joint_result)
    a.out.write_text(json.dumps(summary,indent=2)+'\n');a.details.write_text(json.dumps(details,indent=2)+'\n');print(json.dumps(summary,indent=2))
