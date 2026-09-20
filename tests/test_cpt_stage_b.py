import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpt_stage_b import FORMS,specs,messages,summarize,digest,author_prompt,reviewer_prompt,validate_task,device_idle


class ProtocolTest(unittest.TestCase):
    def test_token_encoding_contract(self):
        from cpt_stage_b import encode_prompt
        class Tokenizer:
            def apply_chat_template(self,messages,**kw):
                return 'rendered' if not kw['tokenize'] else {'input_ids':[1,2]}
            def encode(self,text,**kw):return [1,2]
        self.assertEqual(encode_prompt(Tokenizer(),'question'),[1,2])
        for bad in [{'input_ids':[1]},['1'],[True],[-1],[1]*8192]:
            t=Tokenizer();t.encode=lambda *a,**kw:bad
            with self.assertRaises(RuntimeError):encode_prompt(t,'question')

    def test_device_driver_baseline_and_fail_closed(self):
        baseline='\n'.join(['3137 / 65536']*16+['No running processes found in NPU '+str(i) for i in range(8)])
        self.assertTrue(device_idle(baseline)[0])
        for bad in [baseline.replace('3137','59000'),baseline.replace('NPU 7','NPU 6'),baseline.replace('3137 / 65536','missing',1),baseline+'\n| 0 0 | 1234 | worker |']:
            self.assertFalse(device_idle(bad)[0])

    def packet(self,n=8):
        groups=[dict(id=str(i),source_text='A source supports a distinct operational rule.',option_count=6,scope='historical',operation='boundary',source_refs=['primary']) for i in range(n)]
        tasks=[dict(id=g['id']+'-'+f,group=g['id'],form=f,question='Select all correct statements.',options=['a','b','c','d','e','f'],correct_indices=[0,3]) for g in groups for f in FORMS]
        return dict(groups=groups,tasks=tasks)

    def predictions(self,p,label):
        out=[]
        for s in specs(p,label):
            key=[i for i,j in enumerate(s['order']) if j in (0,3)]
            out.append(dict(s,messages_sha256=digest(messages(p,s)),text=json.dumps({'answers':key}),finish_reason='stop',output_tokens=9))
        return out

    def test_budget(self):
        p=self.packet();self.assertEqual(len(specs(p,'step120_current'))+len(specs(p,'p1')),432)
        p=self.packet(1);self.assertEqual(len(specs(p,'step120_current'))+len(specs(p,'p1')),80)

    def test_option_mapping_and_repeats(self):
        p=self.packet(1);plan=specs(p,'p1')
        for t in p['tasks']:
            self.assertEqual(len({tuple(s['order']) for s in plan if s['id']==t['id'] and not s['repeat'] and s['condition']=='closed'}),4)
        for s in plan:
            if s['repeat']:
                original=next(x for x in plan if x['id']==s['id'] and x['variant']==s['variant'] and x['condition']=='closed' and not x['repeat'])
                self.assertEqual(messages(p,s),messages(p,original))
        summary=summarize(p,self.predictions(p,'p1'),'p1')
        self.assertEqual(summary['groups'][0]['closed_stable_correct'],4)
        self.assertFalse(summary['groups'][0]['training_value_screen_pass'])
        self.assertEqual(summary['stability_answer_changes'],0)

    def test_gate_uses_all_four_orders(self):
        p=self.packet(1);raw=self.predictions(p,'p1')
        for r in raw:
            if r['condition']=='closed' and not r['repeat'] and r['variant']==0:r['text']='{"answers":[1]}'
        self.assertTrue(summarize(p,raw,'p1')['groups'][0]['training_value_screen_pass'])

    def test_missing_group_and_corrupt_prompt_rejected(self):
        p=self.packet(1);raw=self.predictions(p,'p1');raw[0]['messages_sha256']='wrong'
        with self.assertRaises(ValueError):summarize(p,raw,'p1')
        p['tasks'].pop()
        with self.assertRaises(ValueError):specs(p,'p1')

    def test_reviewer_never_gets_author_key(self):
        p=self.packet(1);task=p['tasks'][0]|{'correct_indices':[5],'option_reasons':['SENTINEL_KEY']*6}
        prompt=reviewer_prompt(p['groups'][0],task,'scope')
        self.assertNotIn('SENTINEL_KEY',prompt)
        self.assertNotIn('"correct_indices": [5]',prompt)

    def test_reject_bad_options(self):
        t=dict(question='Select all correct statements.',options=[1]*6,correct_indices=[0],option_reasons=['x']*6,source_quote='A source supports a distinct',reasoning_structure='conditions')
        with self.assertRaises(ValueError):validate_task(t,self.packet(1)['groups'][0])

    def test_evidence_has_no_answers(self):
        p=self.packet(1);s=next(s for s in specs(p,'p1') if s['condition']=='evidence')
        user=messages(p,s)[1]['content']
        self.assertIn(p['groups'][0]['source_text'],user)
        self.assertNotIn('correct_indices',user)


if __name__=='__main__':unittest.main()
