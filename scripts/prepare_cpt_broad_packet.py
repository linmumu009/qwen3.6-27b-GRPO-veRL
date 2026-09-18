"""Build a frozen multisource candidate packet; authoring has no benchmark input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from prepare_cpt_transfer_screen import SOURCE_SHA
from prepare_cpt_expansion_sources import read,validate_evidence,CARDS
from prepare_cpt_remediation_packet import sha,validate_row
from cpt_broad_warehouse_tasks import add_warehouse
from cpt_broad_planning_tasks import add_planning
from cpt_broad_transport_tasks import add_transport
from cpt_broad_retention_tasks import add_retention

UNITS=['warehouse_roles','handling_granularity','shipping_controls','activity_distribution','storage_tradeoffs',
       'metric_denominators','mrp_coordination','network_decisions','outsourcing_governance','holding_cost','road_capacity','rail_track_line']
RETENTION=dict(retain_pest=28,retain_safety=32,retain_abc=44,retain_uld=93,retain_epal=96,retain_reorder=109,
    retain_tkm=115,retain_capital=127,retain_outgoing_rail=139,retain_aircraft_moment=172,retain_postponement=191,
    retain_packages=193,retain_contracts=195,retain_belly=198,retain_payload=201,retain_air_cargo=0)
POSITIONS={name:positions for name,positions,*_ in CARDS if name in UNITS}
POSITIONS.update({name:[i] for name,i in RETENTION.items()})
SCOPES={name:scope for name,_,_,_,scope in CARDS}

def build(resources):
    path=resources/'llin-cpt-sft-search-20260912/source_generation_requests.private.jsonl'
    assert sha(path)==SOURCE_SHA;sources=read(path)
    archive=resources/'llin-knowledge-complete-20260911/core/sources.private.jsonl'
    evidence=validate_evidence(sources,read(archive))
    prior=resources/'llin-transfer-expansion-sources-20260917-02'
    manifest=json.loads((prior/'manifest.safe.json').read_text(encoding='utf-8'))
    assert sha(prior/'catalog.private.jsonl')==manifest['files']['catalog.private.jsonl']
    catalog={r['id']:r for r in read(prior/'catalog.private.jsonl')};rows=[]
    def add(unit,split,design,question,options,checks=()):
        assert unit in POSITIONS and split in ('train','dev','retention')
        assert (unit in RETENTION)==(split=='retention')
        selected=[]
        for i in POSITIONS[unit]:
            source=sources[i];entry=catalog[source['id']]
            assert entry['historical_split']==('dev' if split=='retention' else 'train')
            assert entry['source_text_sha256']==hashlib.sha256(source['source_text'].encode()).hexdigest()
            if split=='retention':assert not entry['p1_domain'] and not entry['p1_replay']
            selected.append(dict({k:source[k] for k in ('id','title','scope','source_text','evidence','topics')},
                source_text_sha256=entry['source_text_sha256'],historical_group=entry['group'],historical_split=entry['historical_split'],p1_domain=entry['p1_domain'],p1_replay=entry['p1_replay']))
        identity='broad-'+unit+'-'+design
        order=list(range(4));random.Random(int(hashlib.sha256(identity.encode()).hexdigest()[:16],16)).shuffle(order)
        displayed=[options[i] for i in order];truth=[t for _,t,_ in displayed]
        row=dict(id=identity,unit=unit,split=split,design=design,question=question+' Select all correct statements.',
            options=[s for s,_,_ in displayed],option_truth=truth,correct_indices=[i for i,t in enumerate(truth) if t],
            option_reasons=[s for _,_,s in displayed],arithmetic_checks=list(checks),author_option_permutation=order,
            reasoning_requirement=('Held-out construction: '+design+'; shared source rules, not an independent source domain.' if split=='dev' else
                'Historical dev-source retention only; never use for training.' if split=='retention' else 'Apply the cited rule and explicit premises: '+design+'.'),
            authoring_scope=SCOPES.get(unit,'Preserve the cited historical definition, vocabulary and explicit scenario assumptions.'),sources=selected,
            retention_stratum='historical_dev_sources' if split=='retention' else None,training_allowed=False,
            purpose='source_authored_candidates',review_status='operator_authored_pending_blind_objections_and_release',source_snapshot_sha256=SOURCE_SHA)
        validate_row(row);rows.append(row)
    for function in (add_warehouse,add_planning,add_transport,add_retention):function(add)
    assert len(rows)==len({r['id'] for r in rows})==88
    assert Counter(r['split'] for r in rows)=={'train':48,'dev':24,'retention':16}
    assert Counter(len(r['correct_indices']) for r in rows if r['split']=='dev')=={1:6,2:6,3:6,4:6}
    for unit in UNITS:
        assert Counter(r['split'] for r in rows if r['unit']==unit)=={'train':4,'dev':2}
        assert Counter(len(r['correct_indices']) for r in rows if r['unit']==unit and r['split']=='train')=={1:1,2:1,3:1,4:1}
    train_sources={s['id']:s for r in rows if r['split']=='train' for s in r['sources']}
    train_groups={s['historical_group'] for s in train_sources.values()}
    heldout_groups={s['historical_group'] for r in rows if r['split']=='retention' for s in r['sources']}
    assert not train_groups & heldout_groups and len(train_groups)==6
    summary=dict(version='broad-v2',cases=88,splits=dict(Counter(r['split'] for r in rows)),capabilities=UNITS,
        train_source_units=len(train_sources),train_source_groups=len(train_groups),new_to_p1_source_units=sum(not(s['p1_domain'] or s['p1_replay']) for s in train_sources.values()),
        retention_source_units=16,retention_source_groups=len(heldout_groups),source_group_separation_verified=True,
        source_snapshot_sha256=SOURCE_SHA,archive_sha256=sha(archive),evidence_check=evidence,
        answer_cardinalities={split:dict(Counter(len(r['correct_indices']) for r in rows if r['split']==split)) for split in ('train','dev','retention')},
        arithmetic_equalities=sum(len(r['arithmetic_checks']) for r in rows),training_allowed=False,
        prospective_measurement=dict(models=['step120_current','p1'],closed_option_orders=4,closed_calls_per_model=352,p1_blind_source_reviews=88,total_new_calls=792,
            closed_seed=20922,review_seed=20922,temperature=0,max_closed_tokens=96,max_review_tokens=1536,thinking=False,tp=8,max_num_seqs=16,max_model_len=8192,chunk=8),
        limitations=['Source units and connected groups are not independent samples.','Development shares training source rules but reserves distinct task constructions.',
            'Retention uses the known historical development source pool, not a fresh holdout.','Source-authored and operator-reviewed, not external expert certification.',
            'Four cyclic orders are correlated and do not exhaust all twenty-four permutations.','No training recipe is released by this build.'])
    return rows,summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows,summary=build(a.resources);a.out.mkdir(exist_ok=False)
    for name,data in [('cases.private.jsonl',rows)]+[(s+'.private.jsonl',[r for r in rows if r['split']==s]) for s in ('train','dev','retention')]:
        (a.out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8',newline='\n')
    summary['files']={p.name:sha(p) for p in a.out.glob('*.jsonl')}
    summary['authoring_code_sha256']={p.name:sha(p) for p in [Path(__file__)]+sorted(Path(__file__).parent.glob('cpt_broad_*_tasks.py'))}
    (a.out/'manifest.safe.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
