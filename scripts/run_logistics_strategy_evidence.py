"""Bounded second-stage evidence test; all question/answer content remains private."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from scripts.run_logistics_strategy_diagnostic import read, write, digest, item_of, majority
from scripts.evaluate_logistics_knowledge import build_messages, parse_answers


def prepare(out):
    decision=json.load(sys.stdin)
    candidates=read(out/'review_candidates.private.json')
    blind=read(out/'review_blind.private.json')
    assert len(candidates)==len(blind['cases'])<=12
    assert len(decision['reviews'])==len(candidates)
    assert {r['item_hash'] for r in decision['reviews']}=={r['item_hash'] for r in candidates}
    target=out/'evidence';target.mkdir(exist_ok=False)
    # Freeze source review BEFORE exposing or comparing the gold answers.
    write(target/'blind_decisions.private.json',decision)
    chosen=[];safe=[]
    mapping={r['item_hash']:r for r in candidates}
    for review in decision['reviews']:
        row=mapping[review['item_hash']]
        label=review.get('source_supported_label')
        match=None
        if label:
            indices=[i for i,text in enumerate(row['options']) if text==label]
            assert len(indices)==1
            match=indices==row['expected']
            if match:chosen.append(row)
        safe.append(dict(item_hash=row['item_hash'],book_evidence_sufficient=review['book_evidence_sufficient'],
            status=review['status'],supplemental_source_supported=bool(label),gold_agreement=match))
    write(target/'cases.private.json',chosen)
    write(target/'evidence.private.json',{'text':decision['supplemental_text'],'source':decision['source']})
    write(target/'review.safe.json',dict(reviewed=len(candidates),book_sufficient=sum(r['book_evidence_sufficient'] for r in safe),
        supplemental_supported=sum(r['supplemental_source_supported'] for r in safe),eligible=len(chosen),rows=safe,
        decisions_sha256=digest(target/'blind_decisions.private.json'),cases_sha256=digest(target/'cases.private.json'),
        source=decision['source'],private_content_included=False,training=False,
        limits=['Single assistant source review, not independent human expert validation.',
                'Old top-three retrieval only; no claim that the full book lacks all facts.',
                'Supplement is a common source-grounded glossary paraphrase, not new training data.',
                'Supplementary candidates have an incongruous inherited question prefix; full original wording is retained.']))
    print(json.dumps(dict(reviewed=len(candidates),book_sufficient=sum(r['book_evidence_sufficient'] for r in safe),eligible=len(chosen))))


def run(root,out,model):
    target=out/'evidence';cases=read(target/'cases.private.json');review=read(target/'review.safe.json')
    assert digest(target/'cases.private.json')==review['cases_sha256']
    assert 0<len(cases)<=12
    evidence=read(target/'evidence.private.json')['text']
    paths={'step120':root/'runs/llin-step120-opensource-20260825-02/hf_export_step120_opensource',
        'cpt':root/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116'}
    from vllm import LLM,SamplingParams
    (out/'status.txt').write_text(f'evidence_loading_{model}\n')
    llm=LLM(model=str(paths[model]),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
        max_model_len=8192,max_num_seqs=64,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
    tok=llm.get_tokenizer();jobs={}
    for condition in ('closed','with_evidence'):
        tasks=[]
        for row in cases:
            messages=build_messages(item_of(row))
            if condition=='with_evidence':
                messages[0]['content']=messages[0]['content'].replace('closed-book logistics question','logistics question using the supplied reference')
                messages[1]['content']+='\n\nReference terminology:\n'+evidence
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False)
            assert len(ids)+96<=8192
            tasks.append((row,ids))
        jobs[condition]=tasks
    with (target/f'{model}.private.jsonl').open('x') as handle:
        for repeat in range(3):
            for condition in (('closed','with_evidence') if repeat%2==0 else ('with_evidence','closed')):
                tasks=jobs[condition]
                outputs=llm.generate([{'prompt_token_ids':ids} for _,ids in tasks],SamplingParams(temperature=0,max_tokens=96,seed=1024))
                assert len(outputs)==len(tasks)
                for (row,ids),output in zip(tasks,outputs):
                    answer=output.outputs[0];parsed,valid=parse_answers(answer.text,len(row['options']))
                    valid=valid and answer.finish_reason=='stop'
                    handle.write(json.dumps(dict(item_hash=row['item_hash'],condition=condition,repeat=repeat,mapped=list(parsed),
                        parse_ok=valid,correct=valid and list(parsed)==row['expected'],text=answer.text,
                        finish_reason=answer.finish_reason,output_tokens=len(answer.token_ids)))+'\n')
                handle.flush()
                print(json.dumps(dict(model=model,condition=condition,repeat=repeat+1,items=len(cases))),flush=True)


def summarize(out):
    target=out/'evidence';cases=read(target/'cases.private.json');summary={}
    for model in ('step120','cpt'):
        rows=[json.loads(s) for s in (target/f'{model}.private.jsonl').read_text().splitlines()]
        assert len(rows)==len(cases)*6
        metrics={c:{r['item_hash']:majority([x for x in rows if x['item_hash']==r['item_hash'] and x['condition']==c]) for r in cases} for c in ('closed','with_evidence')}
        summary[model]={c:dict(items=len(cases),correct=sum(v['correct'] for v in predictions.values()),parse_failures=sum(v['parse_failures'] for v in predictions.values())) for c,predictions in metrics.items()}
        summary[model]['gains']=sum(not metrics['closed'][r['item_hash']]['correct'] and metrics['with_evidence'][r['item_hash']]['correct'] for r in cases)
        summary[model]['losses']=sum(metrics['closed'][r['item_hash']]['correct'] and not metrics['with_evidence'][r['item_hash']]['correct'] for r in cases)
    write(target/'result.safe.json',dict(items=len(cases),requests=len(cases)*12,models=summary,training=False,private_content_included=False,
        limits=['Selected stable errors and source-supported subset; not population prevalence.',
                'Reference assistance does not prove absence of parametric knowledge.']))
    (out/'status.txt').write_text('analysis_inference_complete\n')
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','summarize'));p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=('step120','cpt'));a=p.parse_args();os.umask(0o077)
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='run':run(a.root,a.out,a.model)
    else:summarize(a.out)


if __name__=='__main__':main()
