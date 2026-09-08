"""Reconcile only safe summaries; no private benchmark text or model output."""
import argparse
import json
from pathlib import Path


def analyze(manifest, result):
    assert result['items']==len(result['rows'])==96
    assert result['requests']==1728
    assert len({r['item_hash'] for r in result['rows']})==96
    assert not result['training'] and not result['private_content_included']
    assert sum(r['items'] for r in manifest['historical_cross'])==1672
    out={'items':96,'requests':1728,'models':{},'history':{},'private_content_included':False}
    for label, predicate in [
        ('long_pool',lambda r:r['choice_count']==269),
        ('material_other',lambda r:r['category']=='material_handling' and r['choice_count']!=269),
        ('transport',lambda r:r['category']=='transport'),
        ('warehousing',lambda r:r['category']=='warehousing')]:
        selected=[r for r in manifest['historical_cross'] if predicate(r)]
        n=sum(r['items'] for r in selected)
        wrong=sum(r['items']-r['cpt_correct'] for r in selected)
        out['history'][label]={'items':n,'wrong':wrong,'error_rate':wrong/n}
    for m in ('step120','cpt'):
        conditions={}
        for c in ('original','permuted','deliberate'):
            rows=[r for r in result['table'] if r['model']==m and r['condition']==c]
            assert sum(r['items'] for r in rows)==96
            conditions[c]={k:sum(r[k] for r in rows) for k in ('items','correct','gains','losses','answer_changed','parse_failures','truncated','repeat_unstable')}
            conditions[c].update(result['costs'][m][c])
        clean=[r for r in result['rows'] if not r['position_sensitive_flag'] and r['results'][m]['permuted']['valid_majority_pair']]
        clean_gains=sum(not r['results'][m]['original']['correct'] and r['results'][m]['permuted']['correct'] for r in clean)
        clean_losses=sum(r['results'][m]['original']['correct'] and not r['results'][m]['permuted']['correct'] for r in clean)
        out['models'][m]={'conditions':conditions,'unflagged_valid_permutation':{'items':len(clean),'gains':clean_gains,'losses':clean_losses,'answer_changed':sum(r['results'][m]['permuted']['answer_changed'] for r in clean)}}
    out['stable_wrong_cpt_all_conditions']=result['stable_wrong_cpt_all_conditions']
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--docs',type=Path,required=True);a=p.parse_args()
    def read(name):return json.loads((a.docs/name).read_text(encoding='utf-8'))
    result=analyze(read('logistics_strategy_diagnostic_manifest_20260908.safe.json'),read('logistics_strategy_diagnostic_result_20260908.safe.json'))
    (a.docs/'logistics_strategy_diagnostic_analysis_20260908.safe.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
