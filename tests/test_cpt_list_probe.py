import json
import hashlib
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpt_list_probe import specs,prompt,answer,summarize
from verify_cpt_list_probe import independent_index

class ListProbeTests(unittest.TestCase):
    def packet(self):
        return dict(bank=[dict(label='term'+str(i)) for i in range(128)],rows=[dict(id='toy',target=7,target_label='term7',question='Which concept?',distractors=[i for i in range(128) if i!=7])])

    def test_nested_sets_position_and_identical_repetitions(self):
        p=self.packet();plan=specs(p);self.assertEqual(len(plan),72)
        texts=[prompt(p,s) for s in plan]
        self.assertEqual(texts[:24],texts[24:48]);self.assertEqual(texts[:24],texts[48:])
        for s in plan:
            self.assertEqual(s['order'][s['expected']],7)
            self.assertEqual(len(set(s['order'])),s['size'])
            if s['mode']=='concept':self.assertNotIn('exactly "term7"',prompt(p,s))
        for pos in range(4):
            sets=[set(next(s['order'] for s in plan if s['size']==n and s['position']==pos)) for n in (4,32,128)]
            self.assertTrue(sets[0]<sets[1]<sets[2])

    def test_high_indices_and_invalid_outputs(self):
        self.assertEqual(answer('{"answers":[127]}',128),127)
        self.assertEqual(answer('{"answers":[0]}',4),0)
        for text in ('{"answers":[128]}','{"answers":[true]}','{"answers":[1,2]}','{"answers":[-1]}','{"answers":["3"]}','answer: 3'):
            self.assertIsNone(answer(text,128))

    def test_same_prompt_variation_is_distinct_from_position_variation(self):
        p=self.packet();p['rows']=[dict(p['rows'][0],id='toy'+str(i)) for i in range(8)]
        plan=specs(p);raw=[]
        for s in plan:
            raw.append(dict(s,text=json.dumps({'answers':[s['expected']]}),finish_reason='stop',output_tokens=8,prompt_tokens=100,
                prompt_text_sha256=hashlib.sha256(prompt(p,s).encode()).hexdigest()))
        good=summarize(p,raw);self.assertEqual(good['unstable_identical_prompts'],0)
        # One identical-prompt repetition changes answer; other position groups stay valid.
        raw[192]['text']=json.dumps({'answers':[(plan[192]['expected']+1)%plan[192]['size']]})
        changed=summarize(p,raw);self.assertEqual(changed['unstable_identical_prompts'],1)
        self.assertEqual(sum(d['correct'] for d in changed['per_call']),575)
        raw[192]['order']=list(reversed(raw[192]['order']))
        with self.assertRaises(AssertionError):summarize(p,raw)

    def test_independent_parser_rejects_truncation_and_boolean_indices(self):
        self.assertEqual(independent_index(dict(text='{"answers":[127]}',finish_reason='stop'),128),127)
        for text,finish in [('{"answers":[127]}','length'),('{"answers":[true]}','stop'),('{"answers":[128]}','stop')]:
            self.assertIsNone(independent_index(dict(text=text,finish_reason=finish),128))

if __name__=='__main__':unittest.main()
