"""Build and audit source-authored train/dev/retention candidates, without benchmark input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from cpt_composed_probe import arithmetic
from cpt_remediation_domain_tasks import add_domain
from cpt_remediation_retention_tasks import add_retention
from prepare_cpt_expansion_sources import read, validate_evidence
from prepare_cpt_transfer_screen import SOURCE_SHA

POSITIONS={
    'pick_lines':[166], 'batch_tradeoff':[167], 'serial_parallel':[168], 'transport_capacity':[194],
    'pest':[28], 'abc':[44], 'reorder':[109], 'rail_transfer':[57,58], 'postponement':[191], 'contracts':[195],
    'warehouse_roles':[73], 'shipping_controls':[77], 'outsourcing':[171], 'holding_cost':[197]}
HELDOUT={'pest','abc','reorder','rail_transfer','postponement','contracts'}
SHARED={'warehouse_roles','shipping_controls','outsourcing','holding_cost'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate_row(row):
    assert len(row['options'])==len(row['option_truth'])==len(row['option_reasons'])==4
    assert all(type(v) is bool for v in row['option_truth'])
    assert row['correct_indices']==[i for i,v in enumerate(row['option_truth']) if v]
    assert row['correct_indices'] and len(set(row['options']))==4
    assert row['question'].endswith('Select all correct statements.')
    assert len(row['question'].split())<=170 and all(len(x.split())<=40 for x in row['options'])
    assert all(isinstance(v,str) and v.strip() for v in row['option_reasons'])
    for a,b in row['arithmetic_checks']:
        assert arithmetic(a)==arithmetic(b),row['id']+': '+a
    assert row['training_allowed'] is False
    assert row['sources'] and all(s['source_text'] and s['evidence'] for s in row['sources'])
    assert all(s['source_text_sha256']==hashlib.sha256(s['source_text'].encode()).hexdigest() for s in row['sources'])


def build(resources):
    path=resources/'llin-cpt-sft-search-20260912/source_generation_requests.private.jsonl'
    assert sha(path)==SOURCE_SHA
    sources=read(path)
    archive=resources/'llin-knowledge-complete-20260911/core/sources.private.jsonl'
    evidence=validate_evidence(sources,read(archive))
    frozen=resources/'llin-transfer-expansion-sources-20260917-02'
    manifest=json.loads((frozen/'manifest.safe.json').read_text(encoding='utf-8'))
    catalog_path=frozen/'catalog.private.jsonl'
    assert sha(catalog_path)==manifest['files']['catalog.private.jsonl']
    catalog={r['id']:r for r in read(catalog_path)}
    rows=[]
    def add(unit,split,design,question,options,checks,novelty):
        selected=[]
        for i in POSITIONS[unit]:
            source=sources[i];entry=catalog[source['id']]
            assert entry['historical_split']==('dev' if unit in HELDOUT else 'train')
            assert entry['source_text_sha256']==hashlib.sha256(source['source_text'].encode()).hexdigest()
            selected.append(dict({k:source[k] for k in ('id','title','scope','source_text','evidence','topics')},
                source_text_sha256=entry['source_text_sha256'],historical_group=entry['group'],
                historical_split=entry['historical_split'],p1_domain=entry['p1_domain'],p1_replay=entry['p1_replay']))
        truth=[t for _,t,_ in options]
        row=dict(id='remediation-'+unit+'-'+design,unit=unit,split=split,design=design,
            question=question+' Select all correct statements.',options=[o for o,_,_ in options],
            option_truth=truth,correct_indices=[i for i,t in enumerate(truth) if t],option_reasons=[r for _,_,r in options],
            arithmetic_checks=checks,reasoning_requirement=novelty,authoring_scope='Preserve the archived source definitions and all explicit hypothetical premises.',
            sources=selected,retention_stratum='historical_dev_sources' if unit in HELDOUT else 'shared_train_groups' if unit in SHARED else None,
            training_allowed=False,purpose='reviewed_source_candidates',review_status='operator_authored_pending_separate_audit',
            source_snapshot_sha256=SOURCE_SHA)
        validate_row(row);rows.append(row)
    add_domain(add);add_retention(add)
    assert len({r['id'] for r in rows})==len(rows)==44
    assert Counter(r['split'] for r in rows)=={'train':16,'dev':8,'retention':20}
    assert Counter(r['retention_stratum'] for r in rows if r['split']=='retention')=={'historical_dev_sources':12,'shared_train_groups':8}
    train_groups={s['historical_group'] for r in rows if r['split']=='train' for s in r['sources']}
    heldout_groups={s['historical_group'] for r in rows if r['retention_stratum']=='historical_dev_sources' for s in r['sources']}
    assert not train_groups & heldout_groups
    for unit in ('pick_lines','batch_tradeoff','serial_parallel','transport_capacity'):
        assert sum(r['unit']==unit and r['split']=='train' for r in rows)==4
        assert sum(r['unit']==unit and r['split']=='dev' for r in rows)==2
    summary=dict(source_snapshot_sha256=SOURCE_SHA,archive_sha256=sha(archive),evidence_check=evidence,
        cases=44,splits=dict(Counter(r['split'] for r in rows)),
        retention_strata=dict(Counter(r['retention_stratum'] for r in rows if r['split']=='retention')),
        answer_cardinalities={split:dict(Counter(len(r['correct_indices']) for r in rows if r['split']==split)) for split in ('train','dev','retention')},
        arithmetic_equalities=sum(len(r['arithmetic_checks']) for r in rows),
        source_units=len({s['id'] for r in rows for s in r['sources']}),
        train_source_groups=len(train_groups),historical_dev_retention_groups=len(heldout_groups),
        source_group_separation_verified=True,training_allowed=False,
        author='Codex operator, source-authored; not external expert certification',
        limits=['Only four remediation capabilities in one historical training source group.',
            'Training and development share source rules; their task constructions are distinct, not independent source domains.',
            'Eight warehouse/business regression checks use training-partition sources; report separately from the twelve source-separated checks.',
            'Known-source retention can include past P1 exposure; it is not an unseen-knowledge test.',
            'No real Agent tool-use or process-execution capability is measured. That separate task is not a gate for freezing this knowledge-MCQ candidate packet.',
            'No dataset release or training is implied until the downstream audit is complete.'])
    return rows,summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows,summary=build(a.resources);a.out.mkdir(exist_ok=False)
    for name,data in [('cases.private.jsonl',rows)]+[(s+'.private.jsonl',[r for r in rows if r['split']==s]) for s in ('train','dev','retention')]:
        (a.out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8',newline='\n')
    summary['files']={p.name:sha(p) for p in a.out.glob('*.jsonl')}
    summary['authoring_code_sha256']={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('cpt_remediation_domain_tasks.py'),Path(__file__).with_name('cpt_remediation_retention_tasks.py'))}
    (a.out/'manifest.safe.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
