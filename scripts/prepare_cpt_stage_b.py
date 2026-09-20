"""Build the source-only Stage B authoring packet from audited primary archives."""
import argparse
import re
from pathlib import Path
from cpt_stage_b import read, save, sha, digest, FORMS


def build(root, out):
    import json
    archive=root/'llin-knowledge-complete-20260911/core/sources.private.jsonl'
    passages=root/'llin-stage-a-20260920-02/primary_passages.private.json'
    mit=root/'llin-stage-a-20260920-03/mit_models.private.json'
    expected={archive:'7da53a1a3246381181c5a28fabfb0b68223aed0ee76ff6242710bf257e5af4c2', passages:'49e70c66c8f9ebb0e6e15945e2f71e27c84ad128c116d54786649e9d0e5f0a40', mit:'fb21b555dff31ec34f06b94540897ed7e13c61d39cd122db5db238260151c04c'}
    for p,h in expected.items():
        if sha(p)!=h: raise ValueError('primary archive changed: '+str(p))
    sources={r['key']:r for r in map(json.loads,archive.read_text(encoding='utf-8').splitlines())}
    groups=[]
    def add(id,scope,operation,n,text,refs):
        groups.append(dict(id=id,scope=scope,operation=operation,option_count=n,source_text=text,source_refs=refs))
    m=read(mit)
    add('transport_models','MIT Transportation Systems, 2004, chapter 10, slides 4, 10, 13','Distinguish mathematical models, qualitative frameworks, system evaluation and transparent assumptions in planning decisions.',4,'\n\n'.join(p['text'] for p in m['pages']),[m['url']])
    web=read(passages)
    def lines(pattern,lo,hi):
        return '\n'.join(t for i,t in re.findall(pattern,web,re.M) if lo<=int(i)<=hi)
    add('modal_dimensions','The Geography of Transport Systems: Intermodalism, Multimodalism and Transmodalism; archived 2026-09-20','Compare mode changes and contract boundaries; distinguish contract integration from within-mode connections and efficiency claims.',4,lines(r'^L(\d+): (.*)$',8,12),['https://transportgeography.org/contents/chapter5/intermodal-transportation-containerization/intermodalism-multimodalism-transmodalism/'])
    add('gvc_trajectories','UNCTAD World Investment Report 2020, key messages, page xiii','Compare reshoring, diversification, regionalization and replication across fragmentation, geography, value concentration and digital manufacturing.',4,'\n'.join(t for i,t in re.findall(r'^L(\d+)@P3: (.*)$',web,re.M) if 204<=int(i)<=234),['https://unctad.org/node/3989'])
    entries=[
      ('low_water_response','Ports and Waterways textbook; low-water navigation context','Compare operational and infrastructure responses to low water, including prerequisites and limits.',6,['23e9a80444aca87fc443']),
      ('rail_loading','Glossary for Transport Statistics, archived edition and railway statistical definitions','Apply loading, unloading, national/international transport and transit boundaries including rail-ferry exceptions.',6,['503554d08d7c3c219cc9','03936680691d4e0b0fbe']),
      ('port_forecast','Ports and Waterways textbook, cargo forecast methods; examples are historical teaching assumptions','Distinguish top-down and bottom-up evidence, forecast assumptions and port competition limitations.',4,['44247ab41f17672e477e']),
      ('picking_workload','MITx supply-chain teaching material, archived order-picking section','Apply unit-size differences to picking workload and automation choices without inventing universal productivity ratios.',4,['e24132919e309f5e6174']),
      ('rail_rolling_stock','Glossary for Transport Statistics, archived railway rolling-stock definitions','Apply propulsion, passenger capacity, coupled-unit and fixed-formation boundaries; compare overlapping classes, not equipment name lookup.',6,['7a99a7c52f9c2bace58b','92ba8cbfe8931dcab751','ac6926e38c57fbfab1db'])]
    for id,scope,op,n,keys in entries:
        rows=[sources['book:'+k] for k in keys]
        add(id,scope,op,n,'\n\n'.join(r['text'] for r in rows),['book:'+k for k in keys])
    assert len(groups)==8 and all(len(g['source_text'])>300 for g in groups)
    packet=dict(schema=1,groups=groups,forms=list(FORMS),training_allowed=False,author_calls_max=64,diagnostic_calls_max=432)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'sources.private.json',packet)
    save(out/'manifest.safe.json',dict(source_packet_sha256=sha(out/'sources.private.json'),archives={str(p.relative_to(root)):h for p,h in expected.items()},groups=[{k:v for k,v in g.items() if k!='source_text'}|{'source_text_sha256':digest(g['source_text'])} for g in groups],author_calls_max=64,diagnostic_calls_max=432,training_allowed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('CPT_resources'));p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root,a.out)
