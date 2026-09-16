"""Independent response reparse and paired semantic answer comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import cpt_composed_probe as probe


def independent_parse(text,count):
    hits=re.findall(r'\{\s*"answers"\s*:\s*\[[^\]]*\]\s*\}',text)
    if len(hits)!=1:return None
    try: values=json.loads(hits[0])['answers']
    except (ValueError,KeyError,TypeError):return None
    if not isinstance(values,list) or not 0<len(values)<count or any(type(v) is not int or not 0<=v<count for v in values):return None
    if len(set(values))!=len(values):return None
    return sorted(values)


def verify(packet,result_dir):
    packet=Path(packet);base=Path(result_dir)
    cases=[json.loads(s) for s in packet.read_text(encoding='utf-8').splitlines()];probe.validate(cases)
    by_id={c['id']:c for c in cases};reg=json.loads((base/'registration.safe.json').read_text())
    assert reg['case_sha256']==probe.sha(packet)
    for key,value in dict(max_tokens=96,temperature=0,seed=1024,thinking=False,tp=8,max_num_seqs=16,max_model_len=8192,option_orders=2,repeats_per_order_condition=1).items():assert reg[key]==value
    rows=[json.loads(s) for s in (base/'predictions.private.jsonl').read_text(encoding='utf-8').splitlines()]
    key=lambda r:(r['id'],r['variant'],r['condition'])
    raw={key(r):r for r in rows};specs=probe.specifications(cases)
    assert len(raw)==len(rows)==len(specs) and set(raw)=={key(r) for r in specs}
    per_case={};totals=Counter();categories={}
    for spec in specs:
        r=raw[key(spec)];c=by_id[r['id']]
        for k in ('order','expected'):assert r[k]==spec[k]
        assert r['prompt_text_sha256']==hashlib.sha256(probe.prompt(c,r['order'],r['condition']).encode()).hexdigest()
        parsed=independent_parse(r['text'],len(c['options']))
        assert parsed==r['parsed']
        correct=parsed==spec['expected'] and r['finish_reason']=='stop'
        assert correct==r['correct'] and r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        canonical=sorted(r['order'][i] for i in parsed) if parsed is not None else None
        per_case.setdefault(r['id'],{})[(r['variant'],r['condition'])]=dict(correct=correct,canonical=canonical)
        totals[r['condition']+'_requests']+=1;totals[r['condition']+'_correct']+=int(correct)
        totals['invalid']+=parsed is None;totals['truncated']+=r['finish_reason']!='stop'
    details=[]
    for cid,scores in per_case.items():
        c=by_id[cid]
        closed=sum(scores[(v,'closed_book')]['correct'] for v in (0,1))
        evidence=sum(scores[(v,'source_evidence')]['correct'] for v in (0,1))
        item=dict(id=cid,category=c['category'],closed_correct_orders=closed,evidence_correct_orders=evidence,
            closed_content_changes_with_order=scores[(0,'closed_book')]['canonical']!=scores[(1,'closed_book')]['canonical'],
            evidence_content_changes_with_order=scores[(0,'source_evidence')]['canonical']!=scores[(1,'source_evidence')]['canonical'],
            closed_both_wrong_evidence_both_right=closed==0 and evidence==2,
            source_ids=[s['id'] for s in c['sources']])
        details.append(item)
        stats=categories.setdefault(c['category'],Counter())
        for target in (stats,totals):
            target['scenarios']+=1;target['closed_both_correct']+=closed==2;target['closed_one_correct']+=closed==1
            target['closed_both_wrong']+=closed==0;target['evidence_both_correct']+=evidence==2
            target['closed_both_wrong_evidence_both_right']+=item['closed_both_wrong_evidence_both_right']
            target['closed_content_changes_with_order']+=item['closed_content_changes_with_order']
    summary=dict(status='mechanically_verified',totals=dict(totals),by_category=categories,
        case_sha256=probe.sha(packet),prediction_sha256=probe.sha(base/'predictions.private.jsonl'),
        code_sha256=reg['code_sha256'],raw_answers_independently_reparsed=True,text_prompts_reconstructed=True,
        token_hash_reconstruction='pending separate archived-tokenizer check',training_started=False,official_evaluation_changed=False,
        limitations='Agent-authored screening scenarios, not independent generalization evaluation. Two option orders are not independent seeds. Cases combine and overlap source rules; gaps cannot be added to prior counts without semantic consolidation. Evidence-assisted scores are not closed-book learning gains.')
    return summary,details


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--details',type=Path,required=True);a=p.parse_args()
    summary,details=verify(a.cases,a.result)
    a.out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    a.details.write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
