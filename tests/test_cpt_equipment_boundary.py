import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpt_equipment_boundary import specs,prompt,summarize
from verify_cpt_equipment_boundary import independent_index

class EquipmentBoundaryTests(unittest.TestCase):
    def packet(self):
        bank=[dict(index=i,label='term'+str(i),family=str(i//4)) for i in range(16)]
        rows=[]
        for i in range(16):
            core=[j for j in range(16) if j!=i and j//4==i//4]
            rest=[j for j in range(16) if j//4!=i//4]
            rows.append(dict(id='toy'+str(i),target=i,target_label=bank[i]['label'],family=str(i//4),pair=str(i//2),
                question='Which device fits the conditions?',distractors=core+rest))
        return dict(bank=bank,rows=rows)

    def test_close_competitors_nested_positions_and_repeats(self):
        p=self.packet();plan=specs(p);self.assertEqual(len(plan),256)
        texts=[prompt(p,s) for s in plan];self.assertEqual(texts[:128],texts[128:])
        for s in plan:
            target=int(s['id'][3:]);self.assertEqual(s['order'][s['expected']],target)
            self.assertEqual(len(set(s['order'])),s['size'])
            self.assertIn(s['expected'],(0,s['size']-1))
            if s['size']==4:self.assertEqual({p['bank'][i]['family'] for i in s['order']},{str(target//4)})

    def test_control_hides_labels_only_in_concept_mode(self):
        p=self.packet()
        for s in specs(p):
            q=prompt(p,s).split('Question:\n')[1].split('\nOptions:')[0]
            self.assertEqual('term' in q,s['mode']=='exact_label')

    def test_repeated_error_and_metadata_integrity(self):
        p=self.packet();plan=specs(p);raw=[]
        for s in plan:
            raw.append(dict(s,text=json.dumps({'answers':[s['expected']]}),finish_reason='stop',output_tokens=8,prompt_tokens=100,
                prompt_text_sha256=hashlib.sha256(prompt(p,s).encode()).hexdigest()))
        self.assertEqual(summarize(p,raw)['unstable_identical_prompts'],0)
        raw[128]['text']=json.dumps({'answers':[(plan[128]['expected']+1)%plan[128]['size']]})
        changed=summarize(p,raw);self.assertEqual(changed['unstable_identical_prompts'],1)
        self.assertEqual(sum(d['correct'] for d in changed['per_call']),255)
        raw[128]['expected']+=1
        with self.assertRaises(AssertionError):summarize(p,raw)

    def test_independent_parser_bounds_and_truncation(self):
        self.assertEqual(independent_index(dict(text='{"answers":[15]}',finish_reason='stop'),16),15)
        for text,end in [('{"answers":[16]}','stop'),('{"answers":[true]}','stop'),('{"answers":[0,1]}','stop'),('{"answers":[15]}','length')]:
            self.assertIsNone(independent_index(dict(text=text,finish_reason=end),16))

if __name__=='__main__':unittest.main()
