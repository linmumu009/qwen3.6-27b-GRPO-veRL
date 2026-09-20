"""Register an eight-task evidence-first authoring pilot; source-only inputs."""
import argparse
from pathlib import Path
from cpt_stage_b import read,save,sha,digest,FORMS
from cpt_stage_b_citations import source_blocks

DESIGNS={
 'low_water_response':{
  'scope':'Apply the distinction between the design ALD/ALW reference and actual low-water navigability. Use a planning decision, not four paraphrases of definitions.',
  'competing_rules':'Contrast load reduction, alternate route or mode, and deferred demand under explicit operational constraints. Do not infer a universally optimal measure or invent relative cost/productivity.',
  'application':'Combine the historically observed withdrawal sequence and incomplete capacity replacement with a dispatch response. Keep the historical Rhine dry-bulk scope. Do not assert that every small vessel is immune in every drought.',
  'counterexample':'Present the explicit overgeneralized claim that increased small-vessel activity must fully preserve aggregate throughput. Test a concrete historical counterexample plus the distinction between more trips and total cargo capacity.'},
 'port_forecast':{
  'scope':'Distinguish regional demand from the share captured by one competing port. Make the requested quantity explicit for each option; do not conflate the two.',
  'competing_rules':'Compare captive industrial cargo without a port alternative and contested cargo with alternative ports. Test why bottom-up alone versus a mixture of methods fits those conditions.',
  'application':'Combine an individual industry development plan and a changed transport-network node to identify necessary forecast inputs. Do not invent demographic numbers, percentages or unsupported quantitative effects.',
  'counterexample':'Present the claim that reducing a port transport cost guarantees it captures all regional cargo. Use the source multi-criterion route choice and competition framework to give a concrete counterexample, without claiming a numerical share.'}}

PROMPT='''Create ONE new logistics diagnostic task, using only SOURCE and DESIGN below.
First work out each option's evidence relation, then write a concise scenario.
Use exactly {count} options, each <=40 words, question <=150 words. Include exactly
the instruction "Select all correct statements." in the question. Preserve the
named textbook/historical framework. Do not copy its governing rule into the stem.
Every option must be either entailed or explicitly contradicted by the source plus
scenario. Absence of evidence is NOT contradiction. If any option remains
insufficient, return {{"skip_reason":"..."}} instead of forcing an answer.
Avoid arbitrary numerical parameters, trivial unrelated distractors, universal
claims not supported by the source, lettered scenario names, option index references,
or all/none of the above. Test the particular reasoning operation in DESIGN.
At least one entailed and one contradicted option are required. Do not quote long
source passages. Cite only supplied paragraph IDs for EVERY option; explain the
logical connection, including all conditions. ID validity alone does not establish
truth. Return JSON only with question, options, option_evidence, reasoning_structure.
option_evidence is a list with one object per zero-based option:
{{"option_index":0,"relation":"entailed|contradicted","source_ids":["group:p000"],"reason":"..."}}.
This is authoring only, not a student evaluation or training task.
DESIGN: {design}
SOURCE: {source}
'''


def build(source_path,out):
    import json
    assert sha(source_path)=='00b4f343d9b35211dd86d0e19d44247fe5db6a4754b3f6f3e4c13b1db12031ca'
    groups=[g for g in read(source_path)['groups'] if g['id'] in DESIGNS]
    requests=[]
    for g in groups:
        source=dict(id=g['id'],scope=g['scope'],blocks=source_blocks(g))
        for form in FORMS:
            id=g['id']+'-'+form
            prompt=PROMPT.format(count=g['option_count'],design=DESIGNS[g['id']][form],source=json.dumps(source,ensure_ascii=False))
            requests.append(dict(id=id,group=g['id'],form=form,messages=[dict(role='user',content=prompt)]))
    assert len(requests)==8
    out.mkdir(parents=True,exist_ok=False)
    save(out/'author_requests.private.json',dict(mode='author',requests=requests,training_allowed=False))
    save(out/'groups.private.json',dict(groups=groups))
    save(out/'registration.safe.json',dict(version='two-group-quality-pilot-v1',source_packet_sha256=sha(source_path),requests_sha256=sha(out/'author_requests.private.json'),
        groups=list(DESIGNS),designs=DESIGNS,author_calls_max=8,review_remote_calls_max=0,diagnostic_calls_max=144,
        original_stage_b_diagnostic_cap=1200,author_temperature=.4,author_seed=20922,author_max_tokens=2048,
        reviewer='local assistant, different model from Step120 author, blinded to author evidence and keys until judgments frozen',
        review_is_not_human_or_external_independent=True,no_retries=True,training_allowed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.source,a.out)
