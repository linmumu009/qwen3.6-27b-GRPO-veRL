"""Record local, non-blind option review of the twelve structurally eligible tasks."""
import argparse
from pathlib import Path
from cpt_stage_b import read,save,sha

# Zero-based option order; supported/contradicted refer to the option statement,
# not whether it matches the author's key. Unresolved is never accepted as false.
REVIEWS={
 'transport_models-counterexample':(['supported','supported','contradicted','contradicted'],'key_supported_design_pending','Loop 3 and transparent assumptions are supported. The scenario tests correction/application; the requested counterexample operation needs separate review.'),
 'modal_dimensions-competing_rules':(['supported','supported','contradicted','contradicted'],'key_supported_design_pending','Single contract across rail/sea versus two tickets within air is supported; carrier change does not imply mode change. Verify source attribution and independent scenario overlap before release.'),
 'gvc_trajectories-scope':(['unresolved','supported','contradicted','supported'],'hold_semantic','Option 0 moves difficulty of host-country value capture to the company division. The source does not establish equivalence of country and company outcomes.'),
 'gvc_trajectories-counterexample':(['supported','supported','supported','contradicted'],'hold_design','Replication claims match the historical framework, but reduced fragmentation also occurs under reshoring. The request for features distinguishing it from all other trajectories is ambiguous, and no explicit overgeneralized claim is tested.'),
 'low_water_response-scope':(['contradicted','supported','supported','contradicted','supported','contradicted'],'key_supported_design_pending','ALW is computed by location; its maintenance reference is not real-time riverbed lowering. Historical fleet withdrawal, smaller-vessel limits and listed response measures support the key.'),
 'low_water_response-application':(['supported','contradicted','supported','supported','contradicted','supported'],'key_supported_design_pending','Within the textbook drought scenario, reduced loads require more trips for the same volume; actual total capacity can still fall. Check similarity to scope task before any release.'),
 'low_water_response-counterexample':(['supported','contradicted','contradicted','supported','supported','contradicted'],'key_supported_design_pending','The first claim is tied to the observed historical data, not a universal threshold law. Reduced load, incomplete replacement of capacity and listed modal/production responses support the key.'),
 'port_forecast-scope':(['supported','contradicted','supported','contradicted'],'key_supported_design_pending','Macro demand estimation and competitive port allocation have different purposes; contested cargo calls for a mix rather than exclusive bottom-up or replacement of top-down.'),
 'port_forecast-application':(['supported','contradicted','supported','supported'],'hold_design','The answer set is supported, but arbitrary cargo shares and demographic numbers do not participate in a source-backed calculation and violate the registered author design.'),
 'picking_workload-competing_rules':(['unresolved','unresolved','unresolved','unresolved'],'hold_semantic','The source does not prohibit relocating small items to another zone, quantify a travel/extraction trade-off, or specify item-size-dependent searching/extraction proportions. The blind reviewer approved an overstrong invalidity claim.'),
 'picking_workload-application':(['unresolved','unresolved','unresolved','unresolved'],'hold_semantic','An aggregate 55% travel share does not establish lower travel share for pallets. Relative layout priority needs quantities and frequencies; source does not prove chaotic putaway advantage or identify the new small-item line as high volume.'),
 'rail_rolling_stock-application':(['contradicted','supported','contradicted','supported','supported','unresolved'],'hold_semantic','Power source, railcar versus locomotive, passenger-body counting and authorized private wagons are supported. Dedicated-line design is absent for Beta, so the rationale cannot assert the high-speed category from 250 km/h alone; formation does not determine it.')}


def build(audit,out):
    records=read(audit)
    selected={r['id']:r for r in records if r['eligible_for_semantic_review']}
    assert set(selected)==set(REVIEWS)
    rows=[dict(id=id,option_assessment=labels,disposition=verdict,reason=reason,task_sha256=selected[id]['task_sha256'],source_text_sha256=selected[id]['source_text_sha256']) for id,(labels,verdict,reason) in REVIEWS.items()]
    save(out,dict(review_type='local_nonblind_source_audit',reviewer_has_seen_author_keys=True,not_independent_confirmation=True,
        candidates_reviewed=len(rows),semantic_holds=sum(r['disposition']=='hold_semantic' for r in rows),design_holds=sum(r['disposition']=='hold_design' for r in rows),
        key_supported_design_pending=sum(r['disposition']=='key_supported_design_pending' for r in rows),all_items_still_require_release_review=True,
        released_groups=0,new_model_calls=0,training_allowed=False,citation_audit_sha256=sha(audit),records=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.audit,a.out)
