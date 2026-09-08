"""Inspect judge schema failures without emitting question/answer/source text."""
from collections import Counter
import argparse
import json
from pathlib import Path


def main(out):
    cases={r['id']:r for r in json.loads((out/'cases.private.json').read_text())}
    counts=Counter();examples=[]
    for r in map(json.loads,(out/'grading/reviews.private.jsonl').read_text().splitlines()):
        if r['status']!='invalid_grade':continue
        v=r.get('verdict',{});counts['invalid_records']+=1
        for k in ('question_valid','rubric_valid'):
            if type(v.get(k)) is not bool:counts[k+'_'+type(v.get(k)).__name__]+=1
        answers=v.get('answers',[]);counts['answer_count_'+str(len(answers))]+=1
        if len(examples)<3:examples.append({'id':r['id'],'repeat':r['repeat'],'keys':list(v),'question_valid':v.get('question_valid'),'rubric_valid':v.get('rubric_valid'),
          'answer_shapes':[{k:x.get(k) for k in ('id','score','source_ids')} for x in answers if isinstance(x,dict)],'allowed_source_ids':list(cases[r['id']]['reference_units'])})
        for a in answers:
            if not isinstance(a,dict):counts['non_object']+=1;continue
            if type(a.get('score')) is not int or a['score'] not in (0,1,2):counts['invalid_score']+=1
            if a.get('id') not in 'ABCD':counts['invalid_id']+=1
            ids=a.get('source_ids')
            if not isinstance(ids,list):counts['invalid_source_list']+=1
            else:
                if not 1<=len(ids)<=6:counts['invalid_source_count']+=1
                if len(set(ids))!=len(ids):counts['duplicate_source_ids']+=1
                if any(i not in cases[r['id']]['reference_units'] for i in ids):counts['unknown_source_ids']+=1
            if not isinstance(a.get('reason'),str) or not a['reason'].strip():counts['missing_reason']+=1
    value={'counts':dict(counts),'examples_without_content':examples}
    (out/'judge_schema.safe.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
