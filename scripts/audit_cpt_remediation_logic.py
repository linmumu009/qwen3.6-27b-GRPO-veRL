"""Independent finite-state references for nontrivial synthetic scenario claims.

These hand-transcribed references supplement source/option review; they do not
automatically establish that arbitrary natural-language tasks match the models.
"""
from collections import Counter
from itertools import product


def completion_with_break(preemptible):
    # Picks run on separate workers at 0 and last 6 and 10. The next worker
    # is absent on [11,14); at most one consolidation minute runs per minute.
    unavailable=set(range(11,14))
    if preemptible:
        minute=10;remaining=4
        while remaining:
            if minute not in unavailable:remaining-=1
            minute+=1
    else:
        minute=next(start+4 for start in range(10,30)
                    if all(t not in unavailable for t in range(start,start+4)))
    return minute+2


def references():
    merged=Counter(['A','B','C']+['B','C','D'])
    assert dict(merged)=={'A':1,'B':2,'C':2,'D':1}
    ranges=[s+space for s,space in product(range(12,21),range(7,15))]
    caps18=[min(s,18)+min(space,8)+2 for s,space in product(range(12,21),range(7,15))]
    caps20=[min(s,20)+min(space,8)+2 for s,space in product(range(12,21),range(7,15))]
    assert (min(ranges),max(ranges),max(caps18),max(caps20))==(19,34,28,30)
    waves={str(a)+str(b):25*(a+b)-12*(a+b)-(22 if a or b else 0) for a,b in product((0,1),repeat=2)}
    assert waves=={'00':0,'01':-9,'10':-9,'11':4}
    feasible_releases=[r for r in range(21) if max(8,r+5)+3+2<=17]
    assert feasible_releases==list(range(8))
    assert completion_with_break(True)==19 and completion_with_break(False)==20
    flexible=[min(11-x,8)+min(3+x,8) for x in range(3)]
    assert flexible==[11,12,13]
    uncertain={str(cap):min(9-2,cap)+min(3+2,8) for cap in range(4,8)}
    assert uncertain=={'4':9,'5':10,'6':11,'7':12}
    inverse=[(a,b) for a,b in product(range(31),repeat=2) if min(a,10)==8 and min(a,10)+min(b,10)==16]
    assert inverse==[(8,8)]
    sums={0}
    for _ in range(14):sums={a+b for a,b in product(sums,range(1,5))}
    assert sums==set(range(14,57))
    costs={'none':60-55,'I':60-8+2-55,'J':60-10+3-55}
    assert costs=={'none':5,'I':-1,'J':-2}
    return dict(pick_merge=dict(bin_quantities=dict(merged),lines=len(merged),grabs=sum(merged.values())),
        robust_cost_ranges=dict(combinations=len(ranges),original_min=min(ranges),original_max=max(ranges),cap18_worst=max(caps18),cap20_worst=max(caps20)),
        shared_setup_net_savings=waves,latest_release=max(feasible_releases),
        interrupted_completion=dict(pause_resume=19,uninterrupted=20,active_work=22),
        limited_flexible_served=flexible,uncertain_capacity_served=uncertain,
        inverse_capacity=dict(enumerated_demands_per_departure=[0,30],solutions=inverse,
            unbounded_proof='min(d,10)=8<10 implies d=8, so there are no additional solutions above the enumeration range.'),
        fourteen_instruction_grab_counts=dict(minimum=min(sums),maximum=max(sums),possible_totals=len(sums)),
        intervention_net_costs=costs,
        limitations='Explicit finite-state checks for the listed constructions; source interpretation and premise sufficiency remain operator-audited.')


if __name__=='__main__':
    import json
    print(json.dumps(references(),indent=2))
