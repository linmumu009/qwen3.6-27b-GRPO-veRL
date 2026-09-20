"""Versioned citation audit of frozen author outputs; never changes v1 decisions."""
import argparse
import collections
import json
import re
from pathlib import Path
from cpt_stage_b import read,save,sha,digest,object_of,validate_task


def normalized_map(text):
    chars=[];positions=[]
    for match in re.finditer(r'\s+|\S',text):
        char=' ' if match.group().isspace() else match.group()
        if char==' ' and not chars:continue
        chars.append(char);positions.append((match.start(),match.end()))
    if chars and chars[-1]==' ':chars.pop();positions.pop()
    return ''.join(chars),positions


def locate(source,quote):
    normalized,mapping=normalized_map(source)
    parts=[p.strip() for p in re.split(r'\.{3,}|…+',quote) if p.strip()]
    result=[]
    for fragment in parts:
        needle,_=normalized_map(fragment)
        matches=[];start=0
        while needle:
            pos=normalized.find(needle,start)
            if pos<0:break
            lo,hi=mapping[pos][0],mapping[pos+len(needle)-1][1]
            assert normalized_map(source[lo:hi])[0]==needle
            matches.append(dict(start=lo,end=hi,source_span=source[lo:hi]));start=pos+1
        result.append(dict(fragment=fragment,located=bool(matches),short_anchor=len(needle)<20,matches=matches))
    return result


def source_blocks(group):
    """Immutable paragraph IDs: future reviewers cite IDs instead of rewriting quotes."""
    source=group['source_text'];result=[]
    separators=list(re.finditer(r'\n\s*\n',source))
    starts=[0]+[m.end() for m in separators];ends=[m.start() for m in separators]+[len(source)]
    for start,end in zip(starts,ends):
        raw=source[start:end]
        if not raw.strip():continue
        lo=start+len(raw)-len(raw.lstrip());hi=end-(len(raw)-len(raw.rstrip()));text=source[lo:hi]
        result.append(dict(id=group['id']+':p'+str(len(result)).zfill(3),start=lo,end=hi,text=text,text_sha256=digest(text)))
    assert result and ''.join(''.join(b['text'].split()) for b in result)==''.join(source.split())
    return result


def validate_option_evidence(rows,option_count,blocks):
    """Structural check only; valid IDs do not certify logical entailment."""
    if not isinstance(rows,list) or len(rows)!=option_count:raise ValueError('one assessment per option required')
    known={b['id'] for b in blocks}
    seen=[]
    for row in rows:
        if set(row)!={'option_index','relation','source_ids','reason'}:raise ValueError('assessment fields')
        i=row['option_index'];seen.append(i)
        if type(i) is not int or i not in range(option_count):raise ValueError('option index')
        if row['relation'] not in ('entailed','contradicted','insufficient'):raise ValueError('relation')
        ids=row['source_ids']
        if not isinstance(ids,list) or any(not isinstance(v,str) or v not in known for v in ids) or len(ids)!=len(set(ids)):raise ValueError('source IDs')
        if row['relation']!='insufficient' and not ids:raise ValueError('missing supporting evidence')
        if not isinstance(row['reason'],str) or not row['reason'].strip():raise ValueError('missing reason')
    if len(set(seen))!=option_count:raise ValueError('duplicate options')
    return True


def build(packet_dir,run,out):
    packet=read(packet_dir/'sources.private.json');groups={g['id']:g for g in packet['groups']}
    old={r['id']:r for r in read(run/'candidates.private.json')}
    calls=[json.loads(x) for x in (run/'calls.private.jsonl').read_text(encoding='utf-8').splitlines()]
    records=[]
    for call in calls:
        if call['stage']!='author':continue
        row=old[call['id']];g=groups[row['group']];task=object_of(call['text'])
        parts=locate(g['source_text'],task['source_quote'])
        # Run unchanged non-citation checks separately. This replacement exists
        # only in memory; it is not evidence and is never written as a task quote.
        structural=None
        try:validate_task(dict(task,source_quote=g['source_text'][:40]),g)
        except (ValueError,TypeError,KeyError) as e:structural=str(e)
        located=bool(parts) and all(p['located'] for p in parts)
        records.append(dict(id=row['id'],group=row['group'],form=row['form'],original_rejection=row.get('rejection'),
            original_blind_review_pass=row['blind_review_pass'],fragments=parts,all_fragments_located=located,
            noncitation_rejection=structural,eligible_for_semantic_review=located and structural is None,
            semantic_review_pass=False,task_sha256=digest(task),source_text_sha256=digest(g['source_text'])))
    assert len(records)==32
    out.mkdir(parents=True,exist_ok=False)
    save(out/'citation_audit.private.json',records)
    save(out/'source_blocks.private.json',dict(version='source-blocks-v2',groups=[dict(id=g['id'],scope=g['scope'],source_text_sha256=digest(g['source_text']),blocks=source_blocks(g)) for g in groups.values()],contains_tasks=False,training_allowed=False))
    safe=[{k:v for k,v in r.items() if k!='fragments'}|dict(fragments=len(r['fragments']),unlocated=sum(not p['located'] for p in r['fragments']),short_anchors=sum(p['short_anchor'] for p in r['fragments'])) for r in records]
    report=dict(version='citation-audit-v2',original_acceptance_unchanged=True,normalization='whitespace only; explicit ellipsis splitting; exact case and punctuation',
        candidates=32,all_fragments_located=sum(r['all_fragments_located'] for r in records),eligible_for_semantic_review=sum(r['eligible_for_semantic_review'] for r in records),
        eligible_by_group={g:sum(r['eligible_for_semantic_review'] for r in records if r['group']==g) for g in groups},
        structural_rejections=dict(collections.Counter(r['noncitation_rejection'] for r in records if r['noncitation_rejection'])),
        new_model_calls=0,diagnostic_calls=0,training_allowed=False,source_packet_sha256=sha(packet_dir/'sources.private.json'),calls_sha256=sha(run/'calls.private.jsonl'),records=safe)
    save(out/'citation_audit.safe.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('packet-dir','run','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();build(a.packet_dir,a.run,a.out)
