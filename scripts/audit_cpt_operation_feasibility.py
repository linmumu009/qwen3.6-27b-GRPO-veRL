"""Recompute source-only review checks and bounded historical-operation audit.

No generation, training, or automatic semantic novelty certification. Original
excerpts and historical records remain in the private audit directory.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'CPT_resources/llin-operation-feasibility-20260922-01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    source_path = BASE / 'source_review_input.private.json'
    review_path = BASE / 'source_review.private.json'
    require(sha(source_path) == 'ec061be95e690aaa81731f6f0e5daab6515c618b3275bdffec61369d9b48ce5c', 'source input changed')
    require(sha(review_path) == '14aac98606f294bfa8b8e66aabb944ab8d705bf68964061a4c5d27e8360f16c4', 'review changed')
    source, review = read(source_path), read(review_path)
    require(review['historical_context_seen'] is False, 'review not source-only')
    require(len(review['inputs_accessed']) == 1 and Path(review['inputs_accessed'][0]).resolve() == source_path.resolve(), 'unexpected declared input')
    groups = {g['id']: g for g in source['groups']}
    require(len(review['groups']) == len(groups) == 4, 'group count')
    require({g['id'] for g in review['groups']} == set(groups), 'group identity')
    quotes = 0
    for group in review['groups']:
        sources = {s['id']: s for s in groups[group['id']]['sources']}
        for s in sources.values():
            require(hashlib.sha256(s['text'].encode()).hexdigest() == s['text_sha256'], 'source hash')
        require(len(group['operations']) == 4 and {o['form'] for o in group['operations']} == {'scope', 'competing_rules', 'application', 'counterexample'}, 'forms')
        for op in group['operations']:
            require(bool(op['source_support']), 'missing source support')
            for support in op['source_support']:
                require(bool(support['quote']) and support['quote'] in sources[support['source_id']]['text'], 'quote not exact')
                quotes += 1

    ledger_path = ROOT / 'CPT_resources/llin-stage-a-20260920-03/ledger.private.json'
    accepted = [r['review'] for r in read(ledger_path) if r['review']['status'] == 'A' and r['review']['dataset'] == 'LogistikaBench']
    buckets = {
        'recent_eight_family_trial': 'rail_fixed_composition rail_modular_composition railcar_scope rail_tractive_scope rail_loading_scope port_forecast_bottomup gvc_trajectory model_method_scope low_water_response modal_contract_dimensions picking_workload'.split(),
        'previous_kr_units': 'rail_principal_line rigid_freight_vehicle iwt_passenger_scope iwt_propulsion_boundary river_tug_scope vehicle_authorised_load'.split(),
        'lookup_first_triage': 'cargo_code_perishable cargo_code_vulnerable cargo_code_live cargo_code_valuable cargo_code_heavy cargo_code_dry_ice rail_dangerous_goods'.split(),
        'previous_composition_probe': ['rail_line_speed'],
        'current_four_source_families': 'invoice_finance_mechanism distribution_center_function port_traffic_control port_landfall_services pipeline_network_scope pipeline_goods_scope'.split(),
        'not_deeply_reviewed_this_round': 'rail_accident_intent collision_derailment_priority iwt_freight_scope iwt_authorised_capacity seagoing_ship_scope general_cargo_ship road_train_configuration offered_transport_capacity'.split(),
    }
    assigned = [g for values in buckets.values() for g in values]
    require(len(assigned) == len(set(assigned)) == 39, 'inventory duplicated')
    require(set(assigned) == {r['capability_group'] for r in accepted}, 'inventory mismatch')
    corpus_path = ROOT / 'CPT_resources/llin-logistika-independent-20260922-01/overlap_corpus.private.json'
    records = read(corpus_path)['records']
    # Human-selected concrete anchors, not a keyword-derived novelty score.
    selectors = {
        'trade_finance_structure': ('approved-invoice financing model', 'application', 'conditional_credit_basis_already_used'),
        'warehouse_functions_flow': ('According to the general Handling Rule', 'application', 'handling_unit_cost_rule_already_used'),
        'port_operations_objectives': ('Select one INCORRECT statement about balancing capacity', 'competing_rules', 'utilization_waiting_conflict_already_used'),
        'pipeline_reporting_boundaries': ('Select correct statements about pipeline transport', 'application', 'both_density_conversion_bases_already_exposed'),
    }
    private_evidence, safe_groups = [], []
    for gid, (needle, form, decision) in selectors.items():
        matches = [r for r in records if needle in r['question']]
        require(bool(matches), 'historical anchor missing')
        if gid == 'pipeline_reporting_boundaries':
            require(any('average density' in r['text'] and 'predominantly' in r['text'] for r in matches), 'density evidence missing')
        private_evidence.append({'id': gid, 'reviewed_form': form, 'records': matches, 'decision': decision})
        safe_groups.append({'id': gid, 'source_feasible': next(g['source_feasible'] for g in review['groups'] if g['id'] == gid), 'historical_anchor_count': len(matches), 'reviewed_form': form, 'decision': decision, 'new_complete_group_certified': False})
    inventory = [{'capability_group': g, 'triage': bucket, 'review_indices_zero_based': [r['review_index_zero_based'] for r in accepted if r['capability_group'] == g]} for bucket, values in buckets.items() for g in values]
    private = {'inventory': inventory, 'historical_evidence': private_evidence}
    (BASE / 'audit.private.json').write_text(json.dumps(private, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    safe = {
        'registration': 'llin-operation-feasibility-20260922-01', 'status': 'bounded_pre_authoring_audit_complete',
        'input_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [source_path, review_path, ledger_path, corpus_path]},
        'accepted_logistika_records': len(accepted), 'capability_groups': len(assigned),
        'inventory_counts': {k: len(v) for k, v in buckets.items()},
        'source_families_reviewed': 4, 'abstract_operations_reviewed': 16, 'exact_support_quotes_checked': quotes,
        'historical_corpus_records': len(records), 'groups': safe_groups,
        'new_complete_groups_certified': 0, 'new_questions': 0, 'student_model_calls': 0, 'training_runs': 0,
        'limits': ['Source-only agent used separate context but shared filesystem; not an external human review.', 'One concrete historical anchor per family blocks certification of the proposed complete group; this is not proof that every possible operation is exhausted.', 'Shared source alone is not a rejection criterion; exposed facts do not prove identical reasoning.', 'Eight inventory groups were not deeply reviewed in this round.'],
        'remaining_leads': ['finance_partial_observation_disambiguation', 'harbour_to_port_converse'],
        'remaining_leads_status': 'source_supported_but_novelty_and_measurement_not_certified',
        'private_audit_sha256': sha(BASE / 'audit.private.json'),
    }
    out = ROOT / 'docs/cpt_operation_feasibility_20260922.safe.json'
    out.write_text(json.dumps(safe, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: safe[k] for k in ['accepted_logistika_records', 'capability_groups', 'exact_support_quotes_checked', 'new_complete_groups_certified']}))


if __name__ == '__main__':
    main()
