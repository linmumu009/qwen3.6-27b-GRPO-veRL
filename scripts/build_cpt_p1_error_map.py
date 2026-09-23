"""Offline full-population error/evidence map; never relabel, evaluate or train."""
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from audit_cpt_transfer import CASE_SHA, read_rows, verify_predictions, verify_protocols

BASE=Path('CPT_resources/llin-p1-error-map-20260923-01')
CASES=Path('CPT_resources/llin-transfer-audit-20260915-01/frozen_cases.private.jsonl')
LEDGER=Path('CPT_resources/llin-stage-a-20260920-03/ledger.private.json')
ROOT=Path('CPT_resources')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def require(ok,message):
    if not ok:raise ValueError(message)
def norm(s):return ' '.join(s.casefold().split())


def matching_pairs(cases):
    groups=defaultdict(list)
    for c in cases.values():
        marker='Which of the following best matches:'
        if marker not in c['question']:continue
        key=(norm(c['question'].split(marker,1)[1]),tuple(sorted(norm(c['options'][i]) for i in c['expected'])))
        groups[key].append(c)
    return [(a['item_hash'],b['item_hash']) for group in groups.values()
            for a in group if len(a['options'])==269 for b in group if len(b['options'])<269]


def build():
    require(sha(CASES)==CASE_SHA,'frozen cases changed')
    case_rows=read_rows(CASES);cases={c['item_hash']:c for c in case_rows}
    require(len(cases)==len(case_rows)==1672,'population/unique ids')
    inputs=[CASES,LEDGER,BASE/'source_gap_review.private.json',Path('datasets/LogistikaBench.csv')]
    dirs={'p1':ROOT/'llin-transfer-formal-20260917-01/p1','step120':ROOT/'llin-transfer-formal-20260917-01/step120_current',
          'K':ROOT/'llin-mechanism-20260920-01/results/K_formal','R':ROOT/'llin-mechanism-20260920-01/results/R_formal'}
    scores={};raws={};protocols={}
    for label,folder in dirs.items():
        raw=read_rows(folder/'predictions.private.jsonl');raws[label]=raw
        scores[label]=verify_predictions(cases,raw,read(folder/'scores.safe.json'))
        protocols[label]=read(folder/'protocol.safe.json')
        inputs.extend(folder/n for n in ['predictions.private.jsonl','scores.safe.json','protocol.safe.json'])
    verify_protocols(protocols,CASE_SHA,1672)
    errors={k for k,s in scores['p1'].items() if not s['correct']}
    require(len(errors)==287,'P1 error count')
    old_rows=read(LEDGER);old={r['review']['item_hash']:r['review'] for r in old_rows}
    require(len(old)==len(old_rows)==269 and set(old)<=errors,'historical ledger coverage')
    delta=read(BASE/'source_gap_review.private.json')['items'];new={r['item_hash']:r for r in delta}
    require(len(new)==len(delta)==18 and set(new)==errors-set(old),'gap review coverage')
    archives={}
    for r in delta:
        for ref in r['source_refs']:
            path=Path(ref['path'])
            if path not in archives:archives[path]=read_rows(path);inputs.append(path)
            source=archives[path][ref['line_1_based']-1]
            require(source.get('key',source.get('id'))==ref['key_or_id'],'source id')
            require(hashlib.sha256(source['text'].encode()).hexdigest()==ref['text_sha256'],'source text hash')
        for quote in r['exact_quotes']:
            candidates=[s['text'] for rows in archives.values() for s in rows if s.get('key',s.get('id'))==quote['key_or_id']]
            require(any(quote['text'] in t for t in candidates),'exact gap quote')
    reviews={**old,**new}
    # Verify the input artifact independently of the evaluator's CSV loader.
    with Path('datasets/LogistikaBench.csv').open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    frozen_by_id={c['source_id']:c for c in cases.values() if c['dataset']=='LogistikaBench'}
    require(len(original)==len(frozen_by_id)==1446,'raw CSV coverage')
    for r in original:
        c=frozen_by_id[r['question_id']]
        require(r['question'].strip()==c['question'] and [x.strip() for x in json.loads(r['choices'])]==c['options'] and sorted(json.loads(r['answer']))==sorted(c['expected']),'raw-to-frozen drift')
    pairs=matching_pairs(cases)
    links=defaultdict(list)
    for a,b in pairs:
        links[a].append(b);links[b].append(a)
    discordant={k for k in errors if any(scores['p1'][j]['correct'] for j in links[k])}
    large_discordant={a for a,b in pairs if a in errors and scores['p1'][b]['correct']}
    evidence_root=ROOT/'llin-transfer-audit-20260915-01/evidence_result'
    evidence=read_rows(evidence_root/'predictions.private.jsonl')
    evidence_reg=read(evidence_root/'registration.safe.json')
    require('step120' in evidence_reg['model'],'historical evidence model identity')
    inputs.extend([evidence_root/'predictions.private.jsonl',evidence_root/'registration.safe.json'])
    evidence_by_id=defaultdict(list)
    for r in evidence:evidence_by_id[r['item_hash']].append({k:r[k] for k in ['condition','correct','valid','finish_reason']})
    ledger=[]
    for key in sorted(errors):
        c=cases[key];review=reviews[key];s=scores['p1'][key]
        require(review['status'] in 'ABCD','review status')
        expected=set(c['expected']);actual=set(s['answer'] or [])
        missing=expected-actual;extra=actual-expected
        symptom='no_majority' if s['answer'] is None else 'missing_and_extra' if missing and extra else 'missing_only' if missing else 'extra_only'
        cause='knowledge_vs_rule_application_unresolved'
        if review['status']!='A':cause='source_or_label_context_pending'
        elif key in discordant:cause='cross_presentation_disagreement_observed_cause_unresolved'
        elif review.get('evidence_type')=='self_contained_derivation':cause='self_contained_arithmetic_answer_error_cause_unresolved'
        ledger.append(dict(item_hash=key,dataset=c['dataset'],capability_group=review['capability_group'],source_status=review['status'],
            source_review_origin='prior_269' if key in old else 'gap_18',evidence_type=review.get('evidence_type','historical_source_review'),
            cause_status=cause,answer_symptom=symptom,missing_indices=sorted(missing),extra_indices=sorted(extra),
            option_count=len(c['options']),p1=s,model_results={m:scores[m][key] for m in scores},
            paired_case_ids=links[key],paired_p1_correct_ids=[j for j in links[key] if scores['p1'][j]['correct']],
            historical_step120_evidence=evidence_by_id[key],historical_evidence_not_p1=True,
            source_review=review,case=c))
    save(BASE/'ledger.private.json',ledger)
    save(BASE/'paired_cases.private.json',[dict(large=a,small=b,large_correct=scores['p1'][a]['correct'],small_correct=scores['p1'][b]['correct']) for a,b in pairs])
    queues={
        'presentation_control_first':sorted(k for k in large_discordant if reviews[k]['status']=='A'),
        'small_choice_rule_discrimination':sorted(k for k in errors if reviews[k]['status']=='A' and reviews[k]['capability_group'] in ['modal_contract_dimensions','gvc_trajectory']),
        'source_or_adjudication_first':sorted(k for k in errors if reviews[k]['status']!='A'),
        'general_arithmetic_not_cpt':sorted(k for k in errors if reviews[k].get('evidence_type')=='self_contained_derivation'),
    }
    save(BASE/'decision_queues.private.json',dict(queues=queues,model_calls_authorized_by_this_artifact=False,
        limitations='Research prioritization from known errors, not held-out evaluation or a causal diagnosis.'))
    families=[]
    for group in sorted({r['capability_group'] for r in ledger}):
        rows=[r for r in ledger if r['capability_group']==group]
        families.append(dict(group=group,n=len(rows),source_status=dict(Counter(r['source_status'] for r in rows)),
            large_bank=sum(r['option_count']==269 for r in rows),paired_other_correct=sum(bool(r['paired_p1_correct_ids']) for r in rows),
            unstable=sum(r['p1']['repeat_unstable'] for r in rows),historical_step120_evidence=sum(bool(r['historical_step120_evidence']) for r in rows)))
    large=[k for k,c in cases.items() if len(c['options'])==269]
    pair_counts=Counter(f"large_{int(scores['p1'][a]['correct'])}_small_{int(scores['p1'][b]['correct'])}" for a,b in pairs)
    changes={}
    for label in ['step120','K','R']:
        changes[label]=dict(correct=sum(s['correct'] for s in scores[label].values()),
            correct_among_p1_errors=sum(scores[label][k]['correct'] for k in errors),
            loses_p1_correct=sum(scores['p1'][k]['correct'] and not scores[label][k]['correct'] for k in cases))
    safe=dict(id='llin-p1-error-map-20260923-01',status='offline_complete',new_model_calls=0,training_runs=0,formal_scores_changed=False,
        population=1672,p1_correct=1385,p1_errors=len(errors),errors_by_dataset=dict(Counter(r['dataset'] for r in ledger)),
        prior_review_count=269,new_review_count=18,source_status=dict(Counter(r['source_status'] for r in ledger)),
        source_status_by_dataset={ds:dict(Counter(r['source_status'] for r in ledger if r['dataset']==ds)) for ds in sorted({r['dataset'] for r in ledger})},
        self_contained_derivations=sum(r['evidence_type']=='self_contained_derivation' for r in ledger),
        cause_status=dict(Counter(r['cause_status'] for r in ledger)),answer_symptoms=dict(Counter(r['answer_symptom'] for r in ledger)),
        error_item_repeat_unstable=sum(r['p1']['repeat_unstable'] for r in ledger),
        error_item_invalid_calls=sum(r['p1']['invalid'] for r in ledger),error_item_truncated_calls=sum(r['p1']['truncated'] for r in ledger),
        formal_history=changes,natural_pairs=dict(pairs=len(pairs),unique_large=len({a for a,b in pairs}),outcomes=dict(pair_counts),
            unique_large_wrong_small_correct=len(large_discordant),unique_discordant_error_items=len(discordant),
            source_A_large_wrong_small_correct=sum(reviews[k]['status']=='A' for k in large_discordant),
            source_A_discordant_error_items=sum(reviews[k]['status']=='A' for k in discordant),
            method='Exact normalized description suffix plus normalized gold option text; labels and original scoring unchanged.',
            limitation='Not randomized: prefix, option count, distractors, order and question ID vary together. Repeated definitions are not independent knowledge units.'),
        large_bank=dict(total=len(large),errors=sum(k in errors for k in large),source_A_errors=sum(k in errors and reviews[k]['status']=='A' for k in large),
            unique_option_pools=len({tuple(cases[k]['options']) for k in large}),
            unique_prefixes=len({cases[k]['question'].split('Which of the following best matches:')[0].strip() for k in large}),
            raw_csv_matches_all_1446=True),
        families=sorted(families,key=lambda r:(-r['n'],r['group'])),
        input_sha256={str(p).replace('\\','/'):sha(p) for p in dict.fromkeys(inputs)},private_ledger_sha256=sha(BASE/'ledger.private.json'))
    safe['decision_queue_counts']={k:len(v) for k,v in queues.items()}
    safe['decision_queues_sha256']=sha(BASE/'decision_queues.private.json')
    independent_path=BASE/'pair_verification.private.json'
    verification=read(independent_path)
    for entry in verification['inputs'].values():
        require(sha(Path(entry['path']))==entry['sha256'],'independent input changed')
    check=verification['p1_verification']
    require(not check['per_prediction_discrepancies'] and not check['safe_score_discrepancies'],'independent score discrepancy')
    pc=verification['pair_counts']
    require(pc['pairs']==len(pairs) and pc['unique_large_items']==len({a for a,b in pairs}),'independent pairing count')
    for primary,other in [('large_1_small_1','large_correct__small_correct'),('large_0_small_1','large_wrong__small_correct'),
                          ('large_0_small_0','large_wrong__small_wrong'),('large_1_small_0','large_correct__small_wrong')]:
        require(pair_counts[primary]==pc['four_outcomes'][other],'independent pairing outcome')
    safe['independent_verification']=dict(sha256=sha(independent_path),matched=True,
        raw_prediction_count=check['predictions'],raw_invalid_all_population=check['invalid_raw_predictions'],
        raw_csv_question_exact_match=1444,raw_csv_question_trailing_space_only=2,
        raw_csv_options_labels_exact_match=1446)
    save(Path('docs/cpt_p1_error_map_20260923.safe.json'),safe)
    print(json.dumps({k:safe[k] for k in ['p1_errors','source_status','cause_status','natural_pairs']},ensure_ascii=False))


if __name__=='__main__':build()
