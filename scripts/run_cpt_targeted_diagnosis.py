"""Reuse frozen diagnostic cases on pure book CPT; no authoring or training."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from scripts.run_step120_qa_diagnostic import evaluation_messages, question_item, majority, summarize
from scripts.evaluate_logistics_knowledge import parse_answers


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_profile(rows):
    groups={}
    for r in rows:
        key=(r['dataset'],r['category'])
        g=groups.setdefault(key,{'dataset':key[0],'category':key[1],'items':0,'wrong':0})
        g['items']+=1; g['wrong']+=int(not r['correct'])
    return sorted([dict(g,error_rate=g['wrong']/g['items']) for g in groups.values()],key=lambda g:-g['wrong'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); os.umask(0o077)
    old=a.root/'runs/step120-qa-diagnostic-20260907'
    baseline=a.root/'runs/logistics-cpt-exposure-curve-diagnostics-20260904/safe/public_eval/cpt_4x.majority.safe.json'
    report=read(baseline); before={r['item_hash']:r for r in report['rows']}
    rows=read(old/'cases.private.json'); probes=read(old/'probes.private.json')
    cohort=read(old/'cohort.safe.json')
    if sha(old/'cases.private.json')!=cohort['cases_sha256']:raise ValueError('Frozen cohort mismatch')
    if len(before)!=1672 or sum(r['correct'] for r in before.values())!=1371:raise ValueError('Wrong baseline')
    if len(rows)!=363 or len({r['item_hash'] for r in rows})!=363:raise ValueError('Wrong old cohort')
    if set(probes)!={r['item_hash'] for r in rows}:raise ValueError('Probe coverage')
    for r in rows:
        r['historical_cohort']=r['cohort']
        r['cohort']='cpt_historical_wrong' if not before[r['item_hash']]['correct'] else 'cpt_historical_correct'
    model=a.root/'runs/logistics-cpt-book-exposure-curve-2x4x-20260904-01/hf_export_step_116'
    a.output.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (a.output/name).write_text(json.dumps(value,indent=2))
    covered={r['item_hash'] for r in rows}
    manifest={'model':str(model),'baseline_sha256':sha(baseline),'cases_sha256':sha(old/'cases.private.json'),
        'probes_sha256':sha(old/'probes.private.json'),'baseline_items':1672,'baseline_wrong':301,
        'old_cohort_items':363,'cpt_wrong_in_old_cohort':sum(not before[k]['correct'] for k in covered),
        'cpt_wrong_outside_old_cohort':sum(not r['correct'] and k not in covered for k,r in before.items()),
        'error_profile':baseline_profile(report['rows']),'training':False,'new_qa_generated':False,
        'benchmark_text_sent_to_api':False,'old_probe_quality_not_upgraded':True,
        'cohort_basis':'Old Step120 errors plus matched controls; not a representative new CPT cohort.'}
    write('manifest.safe.json',manifest)
    (a.output/'status.txt').write_text('loading_cpt_model\n')
    from vllm import LLM,SamplingParams
    llm=LLM(model=str(model),tensor_parallel_size=8,dtype='bfloat16',trust_remote_code=True,
        max_model_len=8192,max_num_seqs=64,gpu_memory_utilization=.8,seed=1024,enforce_eager=True)
    tokenizer=llm.get_tokenizer()
    conditions={k:[] for k in ('original_closed','original_with_book','basic_closed','basic_with_book')}
    for row in rows:
        for condition in conditions:
            basic=condition.startswith('basic')
            if basic and not probes[row['item_hash']]['basic_pass']:continue
            item=question_item(row,probes[row['item_hash']]['candidate'] if basic else None)
            messages=evaluation_messages(item,row['evidence'] if condition.endswith('with_book') else None)
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer.encode(prompt,add_special_tokens=False)
            if len(ids)+96>8192:raise ValueError('Prompt overflow')
            conditions[condition].append((row['item_hash'],item,{'prompt_token_ids':ids}))
    predictions={r['item_hash']:{k:[] for k in conditions} for r in rows}
    for repeat in range(3):
        for condition in (list(conditions) if repeat%2==0 else list(reversed(conditions))):
            tasks=conditions[condition]
            (a.output/'status.txt').write_text(f'{condition}_repeat{repeat+1}\n')
            outputs=llm.generate([t[2] for t in tasks],SamplingParams(temperature=0,max_tokens=96,seed=1024))
            if len(outputs)!=len(tasks):raise ValueError('Output coverage')
            for (key,item,_),o in zip(tasks,outputs):
                answer=o.outputs[0]
                parsed,valid=parse_answers(answer.text,len(item.options))
                predictions[key][condition].append({'answers':list(parsed),'parse_ok':valid and answer.finish_reason=='stop',
                    'text':answer.text,'finish_reason':answer.finish_reason,'tokens':len(answer.token_ids)})
            write('progress.private.json',predictions)
            print(json.dumps({'condition':condition,'repeat':repeat+1,'items':len(tasks)}),flush=True)
    results={r['item_hash']:{} for r in rows}
    for condition,tasks in conditions.items():
        for key,item,_ in tasks:results[key][condition]=majority(predictions[key][condition],item.expected)
    write('results.private.json',results)
    final=summarize(rows,probes,results)
    final.update(model='pure_book_CPT4x_step116',author_reviewer_and_test_model_same=False,
        manifest=manifest,finish_reasons=dict(Counter(p['finish_reason'] for rr in predictions.values() for ps in rr.values() for p in ps)),
        inherited_probes_from='Step120',no_new_training=True)
    final['limitations'].extend(['Inherited probe/evidence quality is not upgraded by changing the test model.',
        'Current closed/open results use diagnostic prompts, not official benchmark scores.',
        'Do not use any case, probe, targeted evidence pack or answer from this run as training data.'])
    write('diagnostic.safe.json',final)
    (a.output/'status.txt').write_text('complete_provisional_diagnostic\n')
    print(json.dumps({'complete':True,'cohorts':final['cohorts']}),flush=True)


if __name__=='__main__':main()
