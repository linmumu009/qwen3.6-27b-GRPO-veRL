"""Safe quality profile and source-bound question inventory; no training export."""
import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from scripts.build_book_task_aligned_groups import KINDS, FIELDS, save, digest, words, HELDOUT, source_units, ids_valid
except ModuleNotFoundError:
    from build_book_task_aligned_groups import KINDS, FIELDS, save, digest, words, HELDOUT, source_units, ids_valid


def profile(folder):
    summary=json.loads((folder/'summary.safe.json').read_text())
    manifest=json.loads((folder/'manifest.safe.json').read_text())
    rows=[json.loads(x) for x in (folder/'groups.private.jsonl').read_text().splitlines()]
    sources={r['record_id']:r['text'] for r in map(json.loads,(folder.parent.parent/'runs/logistics-cpt-20260903/private/handbook8e_cpt_4096.jsonl').read_text().splitlines())}
    assert len(rows)==summary['completed_groups'] and 1<=len(rows)<=120
    assert len({r['id'] for r in rows})==len(rows)
    failures=Counter();structural=Counter();audit_meta=Counter();examples=[];structures=[];solver_counts=Counter()
    for r in rows:
        if r['status']=='solver_unresolved':
            v=r.get('solver',{})
            qs=v.get('questions',[])
            if [q.get('kind') for q in qs]!=list(KINDS):solver_counts['kind_order_mismatch']+=1
            solver_counts['question_count_'+str(len(qs))]+=1
            for q in qs:
                if not isinstance(q.get('answer'),str) or not q.get('answer','').strip():solver_counts[q.get('kind','unknown')+':answer_text_invalid']+=1
                if not ids_valid(q.get('source_ids'),source_units(sources[r['source_id']])):solver_counts[q.get('kind','unknown')+':source_ids_invalid']+=1
                solver_counts[q.get('kind','unknown')+':indices_type_'+type(q.get('selected_indices')).__name__]+=1
                if q.get('kind')!='evidence_selection' and q.get('selected_indices'):solver_counts[q.get('kind','unknown')+':unexpected_indices']+=1
                if q.get('kind')=='evidence_selection':
                    ids=q.get('selected_indices',[])
                    if isinstance(ids,list):
                        solver_counts['selection_count_'+str(len(ids))]+=1
                        if any(type(i) is not int or not 0<=i<len(r['solver_mapping']) for i in ids):solver_counts['selection_index_range']+=1
                if len(q.get('source_ids',[]))!=len(set(q.get('source_ids',[]))):solver_counts['duplicate_source_ids']+=1
                for key in ('answerable','unambiguous'):
                    if q.get(key) is not True:solver_counts[q.get('kind','unknown')+':'+key+'_'+str(q.get(key))]+=1
                if 'selected_indices' not in q:solver_counts[q.get('kind','unknown')+':missing_indices']+=1
                if q.get('selected_indices') is None:solver_counts[q.get('kind','unknown')+':null_indices']+=1
                if not 1<=len(q.get('source_ids',[]))<=6:solver_counts[q.get('kind','unknown')+':citation_count']+=1
        if 'audit' in r:
            v=r['audit'];audit_meta['same_concept_'+str(v.get('same_concept'))]+=1
            for q in v.get('questions',[]):
                audit_meta['issues_empty_'+str(q.get('issues')==[])]+=1
                audit_meta['citation_count_'+str(len(q.get('source_ids',[])))]+=1
            if len(examples)<3:examples.append({'id':r['id'],'audit':v})
        for q in r.get('audit',{}).get('questions',[]):
            for k in FIELDS+('solver_equivalent',):
                if q.get(k) is not True:failures[q.get('kind','unknown')+':'+k]+=1
        if r['status']=='structural_reject':
            g=r.get('group',{});qs=g.get('questions',[]) if isinstance(g,dict) else []
            if len(structures)<2:structures.append({'id':r['id'],'group':g})
            if not 1<=len(g.get('source_ids',[]))<=6:structural['group_citation_count']+=1
            if len(qs)!=4:structural['task_count']+=1
            for q in qs:
                if not isinstance(q,dict):structural['non_object']+=1;continue
                if len(words(q.get('answer','')))>120:structural['long_answer']+=1
                if not 1<=len(q.get('source_ids',[]))<=6:structural['question_citation_count']+=1
                if not 1<=len(q.get('rubric',[]))<=3:structural['rubric_count']+=1
                if q.get('kind')!='evidence_selection' and ('options' in q or 'correct_indices' in q):structural['unexpected_option_fields']+=1
                if q.get('kind')=='evidence_selection':
                    if len(q.get('options',[])) not in range(6,11):structural['candidate_count']+=1
                    if any(any(s in x.casefold() for s in ('above','below','none of','all of')) for x in q.get('options',[]) if isinstance(x,str)):structural['position_reference_filter']+=1
    result={'completed_groups':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
      'nonexclusive_audit_failure_fields':dict(failures),'nonexclusive_structural_failure_hints':dict(structural),
      'audit_metadata_counts':dict(audit_meta),
      'solver_unresolved_nonexclusive_hints':dict(solver_counts),
      'training_ready':False,'target_probe_completed':False,'source_only':True,'new_sft_holdout_only':True}
    if summary.get('semantic_screen_completed'):
        path=folder/'candidates.private.jsonl'
        assert digest(path)==summary['candidates_sha256']
        kept=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(kept)==summary['retained_groups']
        assert len({r['id'] for r in kept})==len(kept)
        inventory=[];exact=Counter();concept=Counter()
        for r in kept:
            assert (r['chapter'] in HELDOUT)==(r['split']=='dev')
            assert r['status']=='quality_pass'
            assert [q['kind'] for q in r['group']['questions']]==list(KINDS)
            concept[tuple(words(r['group']['concept']))]+=1
            for q in r['group']['questions']:
                exact[tuple(words(q['question']))]+=1
                inventory.append(dict(id=r['id']+'-'+q['kind'],concept_group=r['id'],split=r['split'],chapter=r['chapter'],source_id=r['source_id'],source_hash=r['source_hash'],concept=r['group']['concept'],**q))
        train={r['source_id'] for r in kept if r['split']=='train'};dev={r['source_id'] for r in kept if r['split']=='dev'}
        assert not train&dev
        target=folder/'question_inventory.private.jsonl'
        if target.exists():raise ValueError('Refuse to overwrite private inventory')
        with target.open('x') as f:
            for r in inventory:f.write(json.dumps(r)+'\n')
        result.update(retained_groups=len(kept),questions=len(inventory),by_kind=dict(Counter(r['kind'] for r in inventory)),
          split_questions=dict(Counter(r['split'] for r in inventory)),source_overlap_count=0,
          exact_duplicate_question_groups=sum(v>1 for v in exact.values()),exact_duplicate_concept_names=sum(v>1 for v in concept.values()),
          answer_word_counts={'minimum':min([len(words(r['answer'])) for r in inventory],default=0),'maximum':max([len(words(r['answer'])) for r in inventory],default=0)},
          option_counts=dict(Counter(len(r['options']) for r in inventory if r['kind']=='evidence_selection')),
          gold_cardinality=dict(Counter(len(r['correct_indices']) for r in inventory if r['kind']=='evidence_selection')),
          inventory_sha256=digest(target),inventory_is_training_messages=False)
    save(folder/'quality.safe.json',result)
    print(json.dumps(result,indent=2))
    if __import__('os').environ.get('BOOK_GROUP_DEBUG')=='1':print(json.dumps({'audit_examples':examples,'structure_examples':structures},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);profile(p.parse_args().data)
