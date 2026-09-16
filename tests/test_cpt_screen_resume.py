import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.run_cpt_transfer_screen import generation_prompt, load_generation_prefix


class ResumeTest(unittest.TestCase):
    def test_preserve_outputs_and_reject_changed_order_or_partial_batch(self):
        rows=[dict(id=str(i),title='title',scope='scope',source_text='source') for i in range(16)]
        raw=[dict(source_id=r['id'],prompt_text_sha256=hashlib.sha256(generation_prompt(r).encode()).hexdigest(),text='raw rejected output',finish_reason='length',output_tokens=3072,prompt_token_sha256='fingerprint',prompt_tokens=100) for r in rows[:8]]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.jsonl'
            def save(records):
                p.write_text(''.join(json.dumps(x)+'\n' for x in records),encoding='utf-8')
                return hashlib.sha256(p.read_bytes()).hexdigest()
            h=save(raw)
            self.assertEqual(load_generation_prefix(p,h,rows),raw)
            with self.assertRaises(ValueError):load_generation_prefix(p,'wrong',rows)
            with self.assertRaises(ValueError):load_generation_prefix(p,h,list(reversed(rows)))
            h=save(raw[:7])
            with self.assertRaises(ValueError):load_generation_prefix(p,h,rows)

if __name__=='__main__':unittest.main()
