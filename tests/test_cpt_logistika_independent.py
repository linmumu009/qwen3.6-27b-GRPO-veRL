import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec=importlib.util.spec_from_file_location('pilot',Path(__file__).parents[1]/'scripts/cpt_logistika_independent.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class IsolationTests(unittest.TestCase):
    def test_frozen_artifact_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'frozen.json'
            pilot.freeze(path,{'decision':False})
            pilot.freeze(path,{'decision':False})
            with self.assertRaises(ValueError):pilot.freeze(path,{'decision':True})
            self.assertEqual(json.loads(path.read_text()),{'decision':False})

    def test_unknown_and_fabricated_evidence_never_pass(self):
        source='A vehicle is classified by its actual propulsion equipment.'
        evidence=[dict(option_index=i,relation=r,source_quotes=[source],reason='Synthetic test')
                  for i,r in enumerate(['entailed','contradicted'])]
        self.assertEqual(pilot.evidence_key(evidence,['one','two'],source),[0])
        evidence[1]['relation']='insufficient'
        with self.assertRaises(ValueError):pilot.evidence_key(evidence,['one','two'],source)
        evidence[1]['relation']='contradicted'
        evidence[1]['source_quotes']=['Invented text absent from source.']
        with self.assertRaises(ValueError):pilot.evidence_key(evidence,['one','two'],source)

    def test_blind_export_excludes_author_keys_reasons_and_skips(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            slots=[dict(id=str(i),group='synthetic',form='scope') for i in range(32)]
            pilot.freeze(base/'writer_input.private.json',dict(groups=[]))
            pilot.freeze(base/'registration.private.json',dict(slots=slots,
                writer_input_sha256=pilot.sha(base/'writer_input.private.json')))
            rows=[dict(id=str(i),question='Synthetic question',options=['one','two'],
                correct_indices=[0],option_evidence=[{'reason':'SECRET_REASON'}],
                reasoning_structure='SECRET_STRUCTURE') for i in range(32)]
            rows[-1]=dict(id='31',skip_reason='SECRET_SKIP')
            pilot.freeze(base/'author.private.json',dict(records=rows,inputs_accessed=['synthetic'],historical_context_seen=False))
            pilot.blind(base)
            text=(base/'reviewer_input.private.json').read_text(encoding='utf-8')
            for secret in ['SECRET_REASON','SECRET_STRUCTURE','SECRET_SKIP','correct_indices']:
                self.assertNotIn(secret,text)
            review=json.loads(text)
            self.assertEqual(len(review['records']),32)
            self.assertFalse(review['records'][-1]['authored'])
            self.assertEqual(set(review['records'][0]),{'id','group','form','question','options','authored'})


if __name__=='__main__':unittest.main()
