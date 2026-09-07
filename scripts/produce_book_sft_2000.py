"""Batch-generate, audit once, discard failures, deduplicate and export 2,000 QA."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from audit_book_sft_pilot import grams
from build_book_sft_api_pilot import words


def duplicate(tokens, shingles, seen):
    return any(tokens==old or (shingles and other and len(shingles&other)/len(shingles|other)>=.45) for old,other in seen)


def main():
    p=argparse.ArgumentParser()
    for key in ('source','api-config','benchmark','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); os.umask(0o077)
    a.output.mkdir(parents=True,exist_ok=False)
    benchmark_raw=a.benchmark.read_bytes()
    benchmark_hash=hashlib.sha256(benchmark_raw).hexdigest()
    if benchmark_hash!='b652b2108cb552346df11d005c15ff3137c50a756a7b24eb35302683ec33ed99':
        raise ValueError('Unexpected benchmark')
    exact=set(); index=set()
    for r in map(json.loads,benchmark_raw.decode().splitlines()):
        exact.add(tuple(words(r['question'])))
        for t in [r['question']]+[x for x in r['options'] if isinstance(x,str)]: index.update(grams(t))
    retained=[]; seen=[]; counts=Counter(); usage=Counter(); batch_reports=[]
    protocol={'target':2000,'batch_size':500,'max_candidates':6000,'workers':64,'model':'qwen3.8-max',
        'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'benchmark_sha256':benchmark_hash,
        'policy':'One generation and one audit; discard all concerns, no repairs. Benchmark only in local lexical screen.',
        'existing_pilots_reused':False,'training_started':False}
    (a.output/'protocol.safe.json').write_text(json.dumps(protocol,indent=2))
    for batch in range(1,13):
        dest=a.output/f'batch{batch:02d}'
        with (a.output/f'batch{batch:02d}.log').open('x') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('build_book_sft_revision40.py')),
                '--source',str(a.source),'--api-config',str(a.api_config),'--output',str(dest),
                '--count','500','--workers','64','--batch-id',str(batch),'--seed',str(20261000+batch)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((dest/'summary.safe.json').read_text()); usage.update(summary['usage'])
        before=len(retained)
        for r in map(json.loads,(dest/'candidates.private.jsonl').read_text().splitlines()):
            status=r['status']
            if status=='auto_pass':
                q=r['qa']; tokens=words(q['question']); shingles=grams(q['question'],5)
                if tuple(tokens) in exact or (grams(q['question'])|grams(q['answer']))&index:
                    status='benchmark_overlap_reject'
                elif duplicate(tokens,shingles,seen): status='cross_batch_duplicate_reject'
                elif len(retained)>=2000: status='surplus_not_selected'
                else:
                    seen.append((tokens,shingles)); retained.append(r); status='retained'
                    with (a.output/'accepted.private.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
            counts[status]+=1
        batch_reports.append({'batch':batch,'candidates':500,'new_retained':len(retained)-before,'retained_total':len(retained)})
        progress=dict(protocol,generated=batch*500,retained=len(retained),counts=dict(counts),usage=dict(usage),batches=batch_reports)
        temp=a.output/'progress.safe.tmp'; temp.write_text(json.dumps(progress,indent=2)); temp.replace(a.output/'progress.safe.json')
        print(json.dumps(batch_reports[-1]),flush=True)
        if len(retained)>=2000: break
        if sum(summary['status'].get(k,0) for k in ('request_or_parse_failure','invalid_audit'))>=400:
            break
    if len(retained)<2000:
        (a.output/'status.txt').write_text('target_not_reached; inspect progress before more API spend\n'); return
    # Keep whole chapters out of SFT training; this is not unseen-to-CPT knowledge.
    dev_chapters={4,15,26,37}
    split_counts=Counter()
    for split in ('train','dev'):
        with (a.output/f'{split}.messages.private.jsonl').open('x') as f:
            for r in retained:
                assigned='dev' if r['chapter'] in dev_chapters else 'train'
                if assigned!=split: continue
                record={'id':r['id'],'chapter':r['chapter'],'source_id':r['source_id'],
                    'messages':[{'role':'user','content':r['qa']['question']},{'role':'assistant','content':r['qa']['answer']}],
                    'enable_thinking':False}
                f.write(json.dumps(record)+'\n'); split_counts[split]+=1
    progress.update(splits=dict(split_counts),dev_chapters=sorted(dev_chapters),dataset_ready=True,
        quality_basis='Automated source-grounded filtering, not human expert verification.',
        semantic_decontamination_proven=False)
    (a.output/'dataset.safe.json').write_text(json.dumps(progress,indent=2))
    (a.output/'status.txt').write_text('dataset_ready_training_not_started\n')


if __name__=='__main__': main()
