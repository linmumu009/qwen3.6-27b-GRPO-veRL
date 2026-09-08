import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_logistics_strategy_evidence import prepare


class EvidenceTests(unittest.TestCase):
    def test_gold_mismatch_is_excluded_after_blind_decision_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cases=[dict(item_hash='a',options=['term','other'],expected=[0]),dict(item_hash='b',options=['term','other'],expected=[1])]
            (root/'review_candidates.private.json').write_text(json.dumps(cases))
            (root/'review_blind.private.json').write_text(json.dumps({'cases':cases}))
            decision={'reviews':[dict(item_hash=k,book_evidence_sufficient=False,status='missing_definition',source_supported_label='term') for k in ('a','b')],
                'supplemental_text':'Synthetic reference.','source':{'title':'unit-test fixture'}}
            with patch('sys.stdin',io.StringIO(json.dumps(decision))):prepare(root)
            self.assertEqual(json.loads((root/'evidence/blind_decisions.private.json').read_text()),decision)
            safe=json.loads((root/'evidence/review.safe.json').read_text())
            self.assertEqual(safe['reviewed'],2)
            self.assertEqual(safe['eligible'],1)
            self.assertFalse(safe['rows'][1]['gold_agreement'])
            self.assertEqual(len(json.loads((root/'evidence/cases.private.json').read_text())),1)


if __name__=='__main__':unittest.main()
