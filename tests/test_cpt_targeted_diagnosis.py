from scripts.run_cpt_targeted_diagnosis import baseline_profile
from scripts.select_targeted_sft_records import decision
import json
from pathlib import Path


def test_profile_uses_cpt_errors_not_sft_regressions():
    rows=[{'dataset':'x','category':'a','correct':True},{'dataset':'x','category':'a','correct':False},
          {'dataset':'x','category':'b','correct':True}]
    profile=baseline_profile(rows)
    assert profile[0]['wrong']==1 and profile[0]['error_rate']==.5
    assert profile[1]['wrong']==0
    assert sum(r['items'] for r in profile)==3


def sample():
    return {'id':'x','concept_group':'g','source_hash':'hash','split':'train','kind':'application',
        'quality':{k:True for k in ('source_verified','standalone','answer_correct','conditions_complete',
          'no_benchmark_derivation','decontamination_pass','dedup_pass','group_split_frozen')},
        'target_probe':{'model':'pure_book_CPT4x_step116','closed_scores':[0,1],'open_scores':[2,2],
          'all_natural_stop':True,'judge_evidence_valid':True,'prerequisite_closed_pass':True}}


def test_value_gate_needs_target_and_evidence():
    r=sample();assert decision(r)=='application_opportunity'
    r['target_probe']['model']='Step120';assert decision(r)=='wrong_or_missing_target'
    r=sample();r['quality']['source_verified']=False;assert decision(r)=='quality_not_passed'
    r=sample();r['target_probe']['open_scores']=[1,1];assert decision(r)=='unresolved_review'
    r=sample();r['target_probe']['closed_scores']=[True,1];assert decision(r)=='invalid_scores'


def test_value_gate_separates_known_unstable_and_dev():
    r=sample();r['target_probe']['closed_scores']=[2,2];assert decision(r)=='maintenance_pool'
    r['target_probe']['closed_scores']=[2,1];assert decision(r)=='unstable_review'
    r['split']='dev';assert decision(r)=='development_only'


def test_frozen_results_reconcile():
    path=Path(__file__).parents[1]/'docs/cpt_targeted_diagnosis_results_20260907.safe.json'
    r=json.loads(path.read_text(encoding='utf-8'))
    assert r['finish_reasons']=={'stop':2634}
    assert sum(g['items'] for g in r['cohorts'].values())==363
    assert r['manifest']['cpt_wrong_in_old_cohort']+r['manifest']['cpt_wrong_outside_old_cohort']==301
    for g in r['cohorts'].values():
        c=g['conditions'];t=g['original_transitions_all']
        assert c['original_with_book']['correct']-c['original_closed']['correct']==t['closed_wrong_book_correct']-t['closed_correct_book_wrong']
