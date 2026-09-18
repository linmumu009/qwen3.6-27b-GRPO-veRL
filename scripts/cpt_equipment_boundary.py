"""Evaluation-only close-equipment contrasts under a named historical taxonomy."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
from cpt_list_probe import sha,save,read,digest,answer

SEED=20924
SIZES=(4,16)
REPEATS=2
UNIQUE=128
CALLS=256
SOURCE_HASHES={
 'conveyors':'e4e14d427db751f60c0709be41de7ae38925a79d5d5cdd833bcb71f66518d804',
 'trucks':'4ebbc3dd7763ad84a9c8efa897fe1043386c0a9af01be728fe54dcb8d5109252',
 'storage':'c231838c5eab7773a9dbc9ff34c8a8edfe83ea999352ac248f38aa95720492d2',
 'cranes':'3ad5fd866b48c20c2819173c6758241c47a0ef2d21e7f74f1bc2c4afc2204e4f'}
# Each group has four competing names. Questions and rationales are paraphrases.
TASKS={
 'conveyors':[
  ('Trolley conveyor','chain_wait','An overhead chain moves carriers at fixed spacings. A carrier cannot stop independently while that chain continues moving. Which listed conveyor category fits?','Fixed coupling prevents individual queuing.','16'),
  ('Power-and-free conveyor','chain_wait','An overhead installation must park individual carriers while its chain continues moving. Disconnected carriers queue on a separate unpowered track. Which listed conveyor category fits?','A separate free track permits waiting.','17'),
  ('Cross-belt transfer sorter','sort_discharge','Each parcel carrier discharges sideways using a small powered belt perpendicular to its direction of travel. The support does not tip. Which listed sorter fits?','Transverse powered cells eject the load.','19e'),
  ('Tilt-tray sorter','sort_discharge','Each parcel carrier discharges by inclining its load-supporting tray. It has no transverse powered belt cell. Which listed sorter fits?','Tilting the tray discharges its load.','19d')],
 'trucks':[
  ('Powered pallet jack','walk_stack','The operator walks beside a powered pallet vehicle. Its forks lift only enough to clear the floor for travel; it cannot place loads onto shelves. Which category fits?','Walking and low lift without stacking.','2b'),
  ('Powered walkie stacker','walk_stack','The operator walks beside a powered pallet vehicle. Its forks also place loads onto elevated storage tiers. Which category fits?','Walking with powered stacking capability.','3b'),
  ('Narrow-aisle straddle truck','leg_clearance','A standing-rider narrow-aisle truck supports a load using legs beside it during placement. Storage positions need gaps accommodating those legs. Which category fits?','Legs flank the load during placement.','7'),
  ('Narrow-aisle reach truck','leg_clearance','A standing-rider truck uses scissor extension while remaining outside the rack. Its legs do not enter the storage position; neighboring loads lack leg-clearance gaps. Which category fits?','Reaching avoids leg entry for this placement.','8')],
 'storage':[
  ('Drive-in rack','lane_access','A truck enters a lane between uprights. Pallets remain stationary on supporting rails, and the rear end is closed. Which rack category fits?','One accessible end limits the lane.','4'),
  ('Drive-through rack','lane_access','A truck enters a lane between uprights. Pallets remain stationary on supporting rails, and both ends are accessible. Which rack category fits?','Opposite ends are accessible.','3'),
  ('Flow-through rack','incline_access','Pallets advance downhill without a truck entering their lane. Replenishment occurs at the upper end and removal at the lower end. Which rack category fits?','Gravity connects separate entry and exit ends.','5'),
  ('Push-back rack','incline_access','Pallets move along inclined supports. Replenishment and removal both occur at the lower face; the upper end is closed. Which rack category fits?','Gravity storage uses the same lower face.','6')],
 'cranes':[
  ('Jib crane','support_structure','A hoist travels along a wall-mounted rotating arm. There is no span translating on parallel runway rails. Which crane category fits?','A rotating arm carries the hoist.','1'),
  ('Bridge crane','support_structure','A hoisted load hangs from a traveling span supported overhead along opposite building walls. There are no legs extending to the floor. Which crane category fits?','The span has overhead wall support.','2'),
  ('Gantry crane','load_interface','A hoisted load hangs from a span carried by two legs on ground-level runways, without building-wall support. Which crane category fits?','Ground-reaching legs support the span.','3'),
  ('Stacker crane','load_interface','A storage machine travels on rails. A fork-bearing mast replaces the suspended hoist and inserts unit loads into racks. Which crane category fits?','A mast handles loads instead of a hoist.','4')]
}

def build(source_dir):
    manifest=json.loads((source_dir/'manifest.safe.json').read_text())
    for key,h in SOURCE_HASHES.items():assert sha(source_dir/(key+'.html'))==manifest[key]['sha256']==h
    bank=[];rows=[]
    for family,items in TASKS.items():
        for label,pair,question,reason,section in items:
            i=len(bank);bank.append(dict(index=i,label=label,family=family))
            rows.append(dict(id='equipment-'+str(i),target=i,target_label=label,family=family,pair=family+'/'+pair,
                question=question,reason=reason,source_key=family,section=section,training_allowed=False))
    for row in rows:
        core=[x['index'] for x in bank if x['family']==row['family'] and x['index']!=row['target']]
        rest=[x['index'] for x in bank if x['family']!=row['family']]
        random.Random(SEED+row['target']).shuffle(core);random.Random(SEED-row['target']).shuffle(rest)
        row['distractors']=core+rest
    assert len(bank)==len(rows)==16 and len({r['target_label'].lower() for r in rows})==16
    return dict(version='equipment-boundary-v1',rows=rows,bank=bank,sources=manifest,seed=SEED,sizes=list(SIZES),repeats=REPEATS,
        training_allowed=False,limitations=[
         'Equipment names follow the cited CICMHE taxonomy, not all contemporary vendor terminology.',
         'Newly authored without formal item text; neither unseen pretraining nor semantic independence from benchmarks is established.',
         'Sixteen cases in four related source pages; contrasts are paired descriptions, not sixteen independent domains.',
         'Only first/last positions and 4/16 options; this does not recreate the formal 269-option vocabulary.',
         'Long lists add other equipment families; the short list already contains all four same-family competitors.',
         'Two identical-batch repeats within a model process do not estimate cross-restart or cross-hardware variation.',
         'Exact-label control deliberately gives the target term. All cases remain evaluation-only.',
         'Scenario contrasts are source-grounded alternatives, not controlled single-feature causal interventions.'])

def specs(packet):
    base=[]
    for row in packet['rows']:
        for n in SIZES:
            for pos,index in enumerate((0,n-1)):
                order=row['distractors'][:n-1].copy();order.insert(index,row['target'])
                for mode in ('concept','exact_label'):
                    base.append(dict(id=row['id'],size=n,position=pos,expected=index,order=order,mode=mode))
    random.Random(SEED).shuffle(base)
    return [dict(s,repeat=r) for r in range(REPEATS) for s in base]

def prompt(packet,s):
    row=next(r for r in packet['rows'] if r['id']==s['id'])
    q=row['question'] if s['mode']=='concept' else 'Find the option whose complete text is exactly '+json.dumps(row['target_label'])+'.'
    return ('Use the MHI/CICMHE Material Handling Equipment Taxonomy categories, not overlapping vendor names. Return only a JSON object with key "answers" containing exactly one zero-based option index.\nQuestion:\n'
        +q+'\nOptions:\n'+'\n'.join(f'[{i}] {packet["bank"][j]["label"]}' for i,j in enumerate(s['order'])))

def summarize(packet,raw):
    plan=specs(packet);assert len(plan)==len(raw)==CALLS
    by_id={r['id']:r for r in packet['rows']};totals=defaultdict(Counter);by_prompt=defaultdict(list);details=[]
    for s,r in zip(plan,raw):
        assert all(s[k]==r[k] for k in s)
        assert r['prompt_text_sha256']==hashlib.sha256(prompt(packet,s).encode()).hexdigest()
        assert 0<r['output_tokens']<=96 and r['prompt_tokens']+96<=8192
        pred=answer(r['text'],s['size']) if r['finish_reason']=='stop' else None
        row=by_id[s['id']];d={k:s[k] for k in ('id','size','position','mode','repeat')}
        d.update(family=row['family'],pair=row['pair'],correct=pred==s['expected'],prediction=pred,expected=s['expected'],
            predicted_label=packet['bank'][s['order'][pred]]['label'] if pred is not None else None,invalid=pred is None,truncated=r['finish_reason']!='stop')
        details.append(d);by_prompt[(s['id'],s['size'],s['position'],s['mode'])].append(d)
        for group in (s['mode']+'/'+str(s['size']),s['mode']+'/'+str(s['size'])+'/position'+str(s['position']),s['mode']+'/'+str(s['size'])+'/'+row['family']):
            totals[group].update(calls=1,correct=d['correct'],invalid=d['invalid'],truncated=d['truncated'])
    stability=[]
    for key,ds in by_prompt.items():
        assert len(ds)==REPEATS
        stability.append(dict(zip(('id','size','position','mode'),key),correct_repeats=sum(d['correct'] for d in ds),
            invariant=len({d['prediction'] for d in ds})==1,all_valid=all(not d['invalid'] for d in ds)))
    return dict(calls=CALLS,groups={k:dict(v) for k,v in totals.items()},identical_prompt_groups=UNIQUE,
        unstable_identical_prompts=sum(not d['invariant'] for d in stability),per_prompt=stability,per_call=details,training_allowed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    packet=build(a.sources);a.out.mkdir(exist_ok=False);save(a.out/'packet.private.json',packet)
    save(a.out/'manifest.safe.json',dict(packet_sha256=sha(a.out/'packet.private.json'),source_hashes=SOURCE_HASHES,cases=16,
        unique_prompts_per_model=UNIQUE,calls_per_model=CALLS,total_calls=CALLS*3,sizes=list(SIZES),repeats=REPEATS,
        authoring_script_sha256=sha(Path(__file__)),training_allowed=False,limitations=packet['limitations']))
