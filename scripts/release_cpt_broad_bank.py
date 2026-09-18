"""Release source-qualified candidates, separately from approval of a curriculum."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare_cpt_remediation_packet import sha
from run_cpt_broad_review import read

QUARANTINE={'broad-storage_tradeoffs-idealized_utilization_scope':'The governing formula is supplied in the question; retain as comprehension-only diagnostic, exclude from domain-acquisition training.'}
ADJUDICATIONS={
 'broad-handling_granularity-explicit_exception_costs':'Reviewer output has malformed JSON (false followed by a stray quote) and a contradictory final option reference. Independent arithmetic is P=40+90=130 and C=70+10=80; P handling cost is lower, C total is lower. Original gold retained.',
 'broad-metric_denominators-overlapping_audits':'Reviewer boolean for the 15-item union contradicts its own reasoning. Its suggested correct-count range is also wrong: overlap successes must be 3..5, yielding 11..13 distinct successes, as independently enumerated. Exact accuracy remains unidentified. Original gold retained.',
 'broad-metric_denominators-inverse_rate_target':'Reviewer marks nine new successes true, then correctly explains 43/50=86%<88%. Enumeration requires all ten new successes. Original gold retained.',
 'broad-mrp_coordination-lead_time_release':'Reviewer marks day-8 release false but computes 8+4=12 and includes it in final indices. Increasing lead time to five moves release to day 7. Original gold retained.',
 'broad-network_decisions-cut_capacity':'Reviewer emits five judgments for four options and initially misuses aggregate capacity. Independent enumeration finds no flow without B-to-X and a feasible witness A-X=6, B-X=1, B-Y=3 with the added link. Costs cannot relax the cut. Original gold retained.',
 'broad-outsourcing_governance-explicit_total_cost':'Reviewer marks the lower-total claim true while explaining 82+12+9=103>100 and omits it from final indices. Original gold retained.',
 'broad-holding_cost-inventory_mix_rate':'Reviewer emits five judgments for four options after correcting its own false boolean. Total cost is 100*0.1+300*0.2=70; weighted rate is 70/400=17.5%, not the unweighted 15%. Original gold retained.',
 'broad-road_capacity-gross_vs_goods':'Reviewer emits five judgments for four options after correcting its headroom judgment. Gross headroom is 18-7=11, but the separate goods limit is 10; a ten-tonne load meets both. Source includes occupants in gross weight. Original gold retained.',
 'broad-road_capacity-volume_only_change':'Reviewer marks the original volume violation false but its reasoning and final indices say true: 15>12. After redesign 15<=18 while 9>8 still violates mass. Original gold retained.',
 'broad-rail_track_line-partial_double_tracking':'Reviewer final indices omit the unchanged 18-km line route despite its true judgment and explanation. Running track is 6*2+12=24, eligible siding adds two to 26, and the specified line route remains 18. Original gold retained.',
}

def release(packet,verification):
    rows=read(packet/'cases.private.jsonl');v=json.loads(verification.read_text(encoding='utf-8'))
    assert v['status']=='completed_verified_local' and v['packet_sha256']==sha(packet/'cases.private.jsonl')
    assert v['new_calls']==792 and v['independent_closed_json_parsing_passed'] and v['all_prompt_tokens_rebuilt']
    pre=json.loads((packet/'pre_result_scope_review.safe.json').read_text());assert set(pre['exclude_from_training'])==set(QUARANTINE)
    decisions={r['id']:r for r in v['models']['p1']['review_decisions']};assert len(decisions)==88
    failed={k for k,d in decisions.items() if not d['accepted']}
    assert failed==set(ADJUDICATIONS),'Every objection must have an explicit reviewed adjudication; unexpected failures block release.'
    items=[]
    for row in rows:
        d=decisions[row['id']]
        items.append(dict(id=row['id'],split=row['split'],source_ids=[s['id'] for s in row['sources']],
            source_and_premise_labels_accepted=True,blind_model_accepted=d['accepted'],
            option_review=[dict(option=i,correct=t,reason=row['option_reasons'][i]) for i,t in enumerate(row['option_truth'])],
            adjudication=ADJUDICATIONS.get(row['id'],'Codex source/premise review agrees with the blind review; no label change.'),
            allowed_as_training_candidate=row['split']=='train' and row['id'] not in QUARANTINE,
            exclusion_reason=QUARANTINE.get(row['id'])))
    review=dict(packet_sha256=sha(packet/'cases.private.jsonl'),reviewer='Codex author and post-review operator; separate P1 blind model objections, no independent human expert certification.',
        model_passed=88-len(failed),model_total=88,model_failures_preserved=len(failed),all_gold_labels_unchanged=True,items=items)
    review_path=packet/'operator_review.safe.json';assert not review_path.exists();review_path.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    approved=[r for r in rows if r['split']=='train' and r['id'] not in QUARANTINE];assert len(approved)==47
    train_path=packet/'released_train.private.jsonl';assert not train_path.exists()
    train_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in approved),encoding='utf-8')
    by_id={r['id']:r for r in v['models']['p1']['per_task']}
    anchors=[r['id'] for r in approved if by_id[r['id']]['correct_orders']==4]
    unstable=[r['id'] for r in approved if 0<by_id[r['id']]['correct_orders']<4]
    result=dict(version='broad-source-bank-v1',training_allowed=True,scope='Only the explicit approved candidate IDs in released_train; this is a source-qualified bank, not an approved wholesale curriculum.',
        packet_sha256=sha(packet/'cases.private.jsonl'),train_file='released_train.private.jsonl',train_file_sha256=sha(train_path),
        operator_review_sha256=sha(review_path),verification_sha256=sha(verification),exclusion_audit_sha256=sha(packet/'audit.safe.json'),
        approved_train_ids=[r['id'] for r in approved],evaluation_only_ids=[r['id'] for r in rows if r['split']!='train' or r['id'] in QUARANTINE],
        quarantined_train_ids=QUARANTINE,train_answer_cardinalities=dict(Counter(len(r['correct_indices']) for r in approved)),
        p1_all_four_correct_anchor_ids=anchors,p1_order_unstable_candidate_ids=unstable,
        p1_no_order_correct_candidate_ids=[r['id'] for r in approved if by_id[r['id']]['correct_orders']==0],
        model_passed=review['model_passed'],model_total=88,all_gold_labels_unchanged=True,
        wholesale_training_ready=False,automatic_launch=False,training_started=False,
        reason='Most candidates are already solved under all four orders. The bank is useful for qualified anchors and diagnosed weaknesses; it does not test the long-list format where P4 lost formal score.',
        next_measurement='Source-authored, fixed-semantics short-versus-long candidate lists, balanced answer positions and identical-prompt repetitions; separate format effects from content and runtime variation before selecting a training remedy.')
    assert len(anchors)==40 and len(unstable)==7
    path=packet/'release.safe.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);a=p.parse_args();r=release(a.packet,a.verification)
    print(json.dumps(dict(approved=len(r['approved_train_ids']),anchors=len(r['p1_all_four_correct_anchor_ids']),unstable=len(r['p1_order_unstable_candidate_ids']),wholesale_training_ready=r['wholesale_training_ready']),indent=2))
