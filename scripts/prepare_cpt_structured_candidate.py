"""Verify archived source/gate evidence and export a source-only writer handoff."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p,data):
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    private=Path('CPT_resources')
    packet=private/'llin-stage-b-20260920-01/sources.private.json'
    manifest_path=Path('reports/cpt_stage_b_20260920/manifest.safe.json')
    manifest=read(manifest_path)
    assert sha(packet)==manifest['source_packet_sha256']
    for rel,digest in manifest['archives'].items():
        assert sha(private/rel.replace('\\','/'))==digest
    sources={g['id']:g for g in read(packet)['groups']}
    spec_path=Path('docs/cpt_structured_candidate_spec_20260922.safe.json')
    profiles=read(spec_path)['profiles']
    assert len(profiles)==8 and {p['id'] for p in profiles}==set(sources)
    approved={g['id']:g for g in manifest['groups']}
    export=[]
    for p in profiles:
        source=sources[p['id']]
        digest=hashlib.sha256(source['source_text'].encode()).hexdigest()
        assert digest==p['source_text_sha256']
        # Historical manifests hash a JSON-encoded string, not raw UTF-8 text.
        historical_digest=hashlib.sha256(json.dumps(source['source_text'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        assert historical_digest==approved[p['id']]['source_text_sha256']
        assert p['source_refs']==source['source_refs']
        export.append({k:p[k] for k in ('id','scope','operation','source_refs','source_text_sha256',
            'required_coverage','rejection_checks','minimum_option_count')} | {'source_text':source['source_text']})
    ledger_path=private/'llin-stage-a-20260920-03/ledger.private.json'
    result_path=private/'llin-stage-a-20260920-03/result.private.json'
    reviews=[r['review'] for r in read(ledger_path)]
    assert len(reviews)==269
    coverage=[]
    for expected in read(result_path)['datasets']:
        rows=[r for r in reviews if r['dataset']==expected['dataset']]
        counts=dict(Counter(r['status'] for r in rows))
        confirmed=[r for r in rows if r['status']=='A']
        assert counts==expected['statuses']
        assert len(confirmed)==expected['confirmed_items']
        assert len({r['dedup_group'] for r in confirmed})==expected['confirmed_deduplicated_items']
        coverage.append(dict(dataset=expected['dataset'],source_statuses=counts,
            source_confirmed_items=len(confirmed),source_confirmed_groups=len({r['capability_group'] for r in confirmed}),
            source_confirmed_deduplicated=len({r['dedup_group'] for r in confirmed})))
    first_path=Path('reports/cpt_stage_b_20260920/run04_author_audit.safe.json')
    second_path=Path('reports/cpt_stage_b_small_20260920/result.safe.json')
    first,second=read(first_path),read(second_path)
    assert first['complete_groups']==second['quality_pass_groups']==0
    assert second['automatic_authoring_stopped'] is True
    handoff=dict(schema=1,role='Source-only independent writer handoff; no tasks or model answers supplied',
        authoring_mode='Human-led or independently reviewed authoring; previous automatic authoring loop remains stopped',
        source_text_hash_encoding='Raw UTF-8; historical manifest JSON-string hashes verified separately',
        groups=export,forms=['definition_and_scope','competing_concepts','conditional_application','counterexample'],
        screening_tasks_per_group=4,maximum_screening_tasks_if_commissioned=32,
        tasks_created=0,model_calls=0,training_allowed=False,
        isolation='Do not access official questions, options, labels, historic generated questions, or model predictions. Audit environment handles exclusion checks after drafting.',
        evidence_contract='Every required fact and option judgment needs source spans and explicit applicability; unknown is not automatically false. Preserve scope and exceptions. No target labels in prompts.',
        stop='No generation or evaluation launched by this package; submit drafts for source and isolation review before any scoring.')
    forbidden={'case','cases','scores','prediction','predictions','answer','gold','options','training_messages','item_hash'}
    def check(value):
        if isinstance(value,dict):
            assert not forbidden.intersection(value)
            for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    check(handoff)
    out=private/'llin-structured-candidate-20260922-01';out.mkdir(exist_ok=True)
    dump(out/'writer_handoff.private.json',handoff)
    report=dict(date='2026-09-22',scope='Archived project evidence only; no new source or question adjudication',
        inputs_sha256={str(p):sha(p) for p in (packet,manifest_path,spec_path,ledger_path,result_path,first_path,second_path)},
        source_coverage=coverage,candidate_packet_groups=8,stage_b_complete_groups=0,
        stage_c_gate=dict(minimum_total_qualified=12,minimum_sc_qualified=3,minimum_logistika_qualified=9,
            current_qualified_total=0,current_qualified_sc=0,current_qualified_logistika=0,passed=False),
        sc_source_group_shortfall_minimum=2,
        sc_note='One source-qualified SC group is not B-qualified; at least two additional source groups needed before testing three. All three must still pass B.',
        log_note='Eight packet candidates are not B-qualified; source-rich 39 groups do not imply any training-value qualification.',
        writer_handoff_sha256=sha(out/'writer_handoff.private.json'),source_fingerprints_verified=8,
        forbidden_structural_fields_absent=True,previous_authoring_stop_preserved=True,
        new_tasks=0,new_model_calls=0,new_training_runs=0,training_allowed=False)
    dump(Path('docs/cpt_structured_candidate_gate_20260922.safe.json'),report)
    print(json.dumps(dict(coverage=coverage,candidate_groups=8,qualified_groups=0,training_allowed=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
