"""Validate the bounded pre-screen and export no original text or answers."""
import json
from pathlib import Path
from cpt_logistika_independent import read, sha
from prepare_cpt_remaining_operations import BASE, SOURCE_HASH


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(base=BASE):
    require(sha(base/'source_input.private.json') == SOURCE_HASH, 'source input changed')
    sources = {g['id']: g for g in read(base/'source_input.private.json')['groups']}
    source_review = read(base/'source_review.private.json')
    history = read(base/'history_review.private.json')
    selection = read(base/'selection.private.json')
    require(selection['source_review_sha256'] == sha(base/'source_review.private.json'), 'source review changed')
    require(selection['history_review_sha256'] == sha(base/'history_review.private.json'), 'history review changed')
    require(len(source_review['groups']) == len(history['groups']) == len(sources) == 10, 'pre-screen coverage')
    require({g['id'] for g in source_review['groups']} == {g['id'] for g in history['groups']} == set(sources), 'pre-screen identities')
    require(source_review['historical_context_seen'] is False, 'source review isolation')
    quote_count = 0
    for g in source_review['groups']:
        texts = [s['text'] for s in sources[g['id']]['sources']]
        require(set(g['operations']) == {'scope','competing_rules','application','counterexample'}, 'operation coverage')
        for op in g['operations'].values():
            for quote in op['exact_source_quotes']:
                require(bool(quote) and any(quote in t for t in texts), 'source quote mismatch')
                quote_count += 1
    corpus_path = Path('CPT_resources/llin-logistika-independent-20260922-01/overlap_corpus.private.json')
    corpus = read(corpus_path)['records']
    history_anchors = 0
    for g in history['groups']:
        for op in g['existing']:
            for e in op['evidence']:
                r = corpus[e['record_index']]
                require(r['source'] == e['source'] and r['question'] == e['question'], 'historical anchor mismatch')
                history_anchors += 1
    registration = read(base/'registration.private.json')
    require(registration['selection_sha256'] == sha(base/'selection.private.json'), 'selection changed')
    require(registration['writer_input_sha256'] == sha(base/'writer_input.private.json'), 'writer input changed')
    safe = dict(registration='llin-remaining-operations-20260923-01', date='2026-09-23',
        status='authoring_or_review_pending', source_items=10,
        source_complete_groups=sum(g['source_feasible'] for g in source_review['groups']),
        source_partial_groups=sum(not g['source_feasible'] for g in source_review['groups']),
        exact_quotes_checked=quote_count, historical_anchors_checked=history_anchors,
        selected_ids=selection['selected_ids'], pre_screen_exclusions=selection['decisions'],
        draft_slots=registration['maximum_draft_records'], correction_rounds=0,
        potential_diagnostic_cap=registration['maximum_diagnostic_calls'],
        student_model_calls=0, training_runs=0, diagnostic_released=False,
        fingerprints={p.name:sha(p) for p in [base/'source_input.private.json',base/'source_review.private.json',
            base/'history_review.private.json',base/'selection.private.json',base/'writer_input.private.json',corpus_path]})
    if (base/'comparison.safe.json').exists():
        compare=read(base/'comparison.safe.json')
        safe['content_pass_items']=sum(r['evidence_agreement_pass'] for r in compare['records'])
        safe['content_complete_groups']=[gid for gid in selection['selected_ids'] if all(r['evidence_agreement_pass'] for r in compare['records'] if r['group']==gid)]
        safe['fingerprints'].update({p.name:sha(p) for p in [base/'author.private.json',base/'review.private.json',base/'review_freeze.private.json']})
        safe['content_failures']=[dict(id=r['id'],errors=r['errors']) for r in compare['records'] if not r['evidence_agreement_pass']]
    if (base/'root_quality.private.json').exists():
        root=read(base/'root_quality.private.json')
        require(root['review_sha256']==sha(base/'review.private.json'), 'root review binding')
        require(root['author_sha256']==sha(base/'author.private.json'), 'root author binding')
        require(len(root['groups'])==len(selection['selected_ids']) and {g['id'] for g in root['groups']}==set(selection['selected_ids']), 'root coverage')
        safe['root_quality']=root['groups']
        safe['quality_complete_groups']=[g['id'] for g in root['groups'] if g['pass']]
        require(set(safe['quality_complete_groups'])<=set(safe['content_complete_groups']), 'cannot release failed content')
        safe['status']='quality_complete_pending_execution' if safe['quality_complete_groups'] else 'closed_no_complete_group'
        safe['fingerprints']['root_quality.private.json']=sha(base/'root_quality.private.json')
        # Positive diagnostic release needs its own execution freeze and identity audit.
    run=base/'remote_run'
    if (run/'status.safe.json').exists():
        status=read(run/'status.safe.json')
        require(status['state']=='complete_pending_local_verification' and status['calls']==80, 'remote incomplete')
        execution=read(base/'execution.safe.json')
        safe['fingerprints']['execution.safe.json']=sha(base/'execution.safe.json')
        safe['fingerprints']['diagnostic.private.json']=sha(base/'diagnostic.private.json')
        require(read(run/'execution.safe.json')==execution, 'execution freeze mismatch')
        require(execution['packet_sha256']==sha(base/'diagnostic.private.json'), 'diagnostic changed')
        require(execution['root_quality_sha256']==sha(base/'root_quality.private.json'), 'quality changed')
        require(execution['runner_sha256']==sha(Path('scripts/run_cpt_remaining_diagnostic.py')), 'runner changed')
        for name,h in execution['protocol_code_sha256'].items():
            require(sha(Path('scripts')/name)==h,'local protocol changed')
        from cpt_stage_b import summarize as diagnostic_summary, specs
        packet=read(base/'diagnostic.private.json')
        summaries={}
        for label,expected in [('step120_current',16),('p1',64)]:
            folder=run/label;rows=read(folder/'predictions.private.json');done=read(folder/'completed.safe.json')
            require(done['calls']==len(rows)==expected and done['predictions_sha256']==sha(folder/'predictions.private.json'), 'prediction integrity')
            plan=specs(packet,label)
            batches=[]
            for start in range(0,len(plan),16):
                result=folder/f'batch-{start:03d}.private.json'
                complete=read(folder/f'batch-{start:03d}.completed.safe.json')
                reserved=read(folder/f'batch-{start:03d}.reserved.safe.json')
                require(complete['sha256']==sha(result) and complete['calls']==reserved['calls']==len(plan[start:start+16]), 'batch integrity')
                require(reserved['packet_sha256']==execution['packet_sha256'], 'batch packet changed')
                batches.extend(read(result))
            require(batches==rows,'batch/aggregate mismatch')
            summaries[label]=diagnostic_summary(packet,rows,label)
            require(summaries[label]==read(folder/'summary.safe.json'), 'summary mismatch')
            safe['fingerprints'][label+'_predictions']=sha(folder/'predictions.private.json')
            safe['fingerprints'][label+'_identity']=sha(folder/'identity.safe.json')
            safe['fingerprints'][label+'_prompts']=sha(folder/'prompts.safe.json')
        safe.update(status='closed_validation_complete',diagnostic_released=True,student_model_calls=80,
            diagnostic=summaries,training_value_signal=summaries['p1']['groups'][0]['training_value_screen_pass'],
            promotion=False,formal_reruns=0)
        safe['protocol']=dict(max_tokens=96,seed=1024,temperature=0,thinking=False,option_orders=4,
            prefix_caching=False,model_identity='matches previous metadata and shard size/mtime; not full-weight content hashes')
        # Independently decode the constrained JSON, remap option positions, and
        # recount the main observations without the shared scoring helper.
        independent=[]
        for label in ('step120_current','p1'):
            rows=read(run/label/'predictions.private.json')
            for task in packet['tasks']:
                for condition in (['closed','evidence'] if label=='p1' else ['closed']):
                    main=[r for r in rows if r['id']==task['id'] and r['condition']==condition and r['repeat']==0]
                    require(len(main)==4,'main observation count')
                    correct=0
                    for r in main:
                        obj=json.loads(r['text'])
                        require(set(obj)=={'answers'} and len(obj['answers'])==len(set(obj['answers'])),'independent parse')
                        require(all(type(i) is int and 0<=i<len(r['order']) for i in obj['answers']),'answer index')
                        require(r['finish_reason']=='stop' and r['output_tokens']<=96,'generation protocol')
                        correct+=sorted(r['order'][i] for i in obj['answers'])==task['correct_indices']
                    independent.append(dict(model=label,form=task['form'],condition=condition,correct_orders=correct,total_orders=4))
        for label in summaries:
            for condition in (['closed','evidence'] if label=='p1' else ['closed']):
                count=sum(r['correct_orders']==4 for r in independent if r['model']==label and r['condition']==condition)
                require(count==summaries[label]['groups'][0][condition+'_stable_correct'],'independent score disagreement')
        safe['independent_main_score_check']=independent
    path=Path('docs/cpt_remaining_operations_results_20260923.safe.json')
    path.write_text(json.dumps(safe,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:safe[k] for k in ['status','exact_quotes_checked','historical_anchors_checked','draft_slots']}))


if __name__=='__main__':
    summarize()
