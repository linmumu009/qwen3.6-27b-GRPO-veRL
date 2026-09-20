import sys
import tempfile
import json
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cpt_stage_b import FORMS,specs,save,digest
from audit_cpt_stage_b_small import normalize,blind
from prepare_cpt_stage_b_small import DESIGNS,PROMPT


class SmallPilotTest(unittest.TestCase):
    def test_blind_export_hides_author_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'author').mkdir()
            req=[dict(id=str(i),group='g',form='scope',messages=[dict(role='user',content='source')]) for i in range(8)]
            save(p/'author_requests.private.json',dict(requests=req))
            task=dict(question='question',options=['one','two'],option_evidence=[dict(reason='SECRET_AUTHOR_REASON')],correct_indices=[1])
            rows=[dict(id=r['id'],text=json.dumps(task),messages_sha256=digest(r['messages']),finish_reason='stop') for r in req]
            (p/'author/predictions.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            result=blind(p)
            self.assertEqual(len(result),8)
            encoded=json.dumps(result)
            self.assertNotIn('SECRET_AUTHOR_REASON',encoded)
            self.assertNotIn('correct_indices',encoded)
            self.assertNotIn('option_evidence',encoded)
            rows[0]['text']='{"question":"unfinished';rows[0]['finish_reason']='length'
            (p/'author/predictions.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            result=blind(p)
            self.assertFalse(result[0]['complete']);self.assertIsNone(result[0]['question'])
            self.assertTrue(result[1]['complete'])

    def test_all_forms_have_distinct_operations(self):
        self.assertEqual(set(DESIGNS),{'low_water_response','port_forecast'})
        for designs in DESIGNS.values():
            self.assertEqual(set(designs),set(FORMS));self.assertEqual(len(set(designs.values())),4)

    def test_author_requires_evidence_not_absence_as_false(self):
        p=PROMPT.format(count=6,design='scope',source='PRIMARY_SOURCE')
        self.assertIn('Absence of evidence is NOT contradiction',p)
        self.assertIn('PRIMARY_SOURCE',p)
        self.assertIn('Select all correct statements.',p)

    def test_both_groups_diagnostic_budget_is_144(self):
        groups=[dict(id=g) for g in DESIGNS]
        tasks=[dict(id=g['id']+'-'+f,group=g['id'],form=f,options=['a','b','c','d']) for g in groups for f in FORMS]
        p=dict(groups=groups,tasks=tasks)
        self.assertEqual(len(specs(p,'step120_current')),32)
        self.assertEqual(len(specs(p,'p1')),112)

    def test_number_renaming_not_novelty(self):
        self.assertEqual(normalize('Cargo 123 takes 2.5 hours'),normalize('cargo 987 takes 4.3 HOURS'))


if __name__=='__main__':unittest.main()
