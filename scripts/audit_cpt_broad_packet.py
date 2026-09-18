"""Downstream exclusion and finite-reference audit; never an authoring input."""
import argparse
from collections import Counter
from fractions import Fraction
from itertools import permutations,product
import json
from pathlib import Path
from audit_cpt_remediation_packet import audit as prior_audit,records
from prepare_cpt_remediation_packet import sha
from cpt_transfer_draft import normalized_question
from prepare_cpt_broad_packet import build

def audit(resources,packet):
    result=prior_audit(resources,packet)
    rows=records(packet/'cases.private.jsonl');rebuilt,summary=build(resources);assert rows==json.loads(json.dumps(rebuilt))
    additional=[resources/'llin-remediation-packet-20260917-01/cases.private.jsonl',resources/'llin-remediation-packet-20260917-03/cases.private.jsonl']
    extra={normalized_question(r) for p in additional for r in records(p)}
    assert not any(normalized_question(r) in extra for r in rows)
    for p in additional:result['exclusion_inputs'].append(dict(path=str(p.relative_to(resources)).replace('\\','/'),sha256=sha(p)))
    excluded=set()
    def collect(o):
        if isinstance(o,dict):
            if isinstance(o.get('question'),str):excluded.add(normalized_question(o))
            for v in o.values():collect(v)
        elif isinstance(o,list):
            for v in o:collect(v)
    for item in result['exclusion_inputs']:
        assert 'sealed' not in item['path']
        for r in records(resources/item['path']):collect(r)
    result['excluded_normalized_questions']=len(excluded)
    checked=0
    for row in rows:
        for order in permutations(range(4)):
            gold=sorted(i for i,j in enumerate(order) if row['option_truth'][j])
            back=sorted(order[i] for i in gold)
            assert back==row['correct_indices'];checked+=1
    # Independent finite-state references for feasibility, inverse and overlap tasks.
    max_packages=max(n for n in range(11) if 2*n<=6 and 3*n<=10)
    assert max_packages==3
    overlap_successes=[z for z in range(6) if 0<=8-z<=5]
    union_correct=sorted({16-z for z in overlap_successes});assert union_correct==[11,12,13]
    extra_correct=[c for c in range(11) if Fraction(34+c,50)>=Fraction(88,100)];assert extra_correct==[10]
    flows_without=[];flows_with=[]
    for ax,ay,bx,by in product(range(8),repeat=4):
        if ax+ay<=6 and bx+by<=6 and ax+bx==7 and ay+by==3:
            flows_with.append((ax,ay,bx,by))
            if bx==0:flows_without.append((ax,ay,bx,by))
    assert not flows_without and flows_with and (6,0,1,3) in flows_with
    references=dict(integer_package_maximum=max_packages,overlap_distinct_success_counts=union_correct,
        inverse_accuracy_required_correct=extra_correct,restricted_network_feasible_count=len(flows_without),
        relaxed_network_feasible_count=len(flows_with),relaxed_network_witness=[6,0,1,3],
        mrp_component_requirement=5*(2+3),mrp_net_requirement=18-6-4,mrp_release_day=10-3,
        partial_running_track_length=6*2+12,including_public_siding=6*2+12+2,
        mixed_inventory_rate=str(Fraction(100*10+300*20,100*(100+300))),
        nonduplicated_net_saving=12-8,known_peak_lower_bound=90,
        note='Independent arithmetic/finite-state checks support the named premises; they do not certify every semantic option.')
    result.update(version='broad-v2',row_rebuild_matches=True,all24_option_permutations_checked=checked,
        independent_references=references,training_allowed=False,
        semantic_review='All authored option explanations remain visible for operator review; blind model objections pending.',
        limitations='Known-source development, correlated options, and prior benchmarks used for category-level direction. Exact normalized exclusion is not semantic-independence certification. No sealed items read.')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit(a.resources,a.packet);a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:r[k] for k in ('cases','excluded_normalized_questions','arithmetic_equalities_checked','all24_option_permutations_checked','independent_references')},indent=2))
