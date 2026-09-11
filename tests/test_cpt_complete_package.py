import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from build_cpt_complete_package import checked_row, deduplicate, digest, EOS, TOKENIZER_SHA


class Tokenizer:
    def __init__(self, ids): self.ids = ids
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids':self.ids})()


class PackageTests(unittest.TestCase):
    def row(self):
        return dict(id='a', text='Definition', text_sha256=digest(b'Definition'),
                    content_tokens=2, token_count=3, tokenizer_sha256=TOKENIZER_SHA)

    def test_duplicate_retains_both_sources_and_original_ids(self):
        a = checked_row(self.row(), Tokenizer([1,2]), 'book', ['book:1'])
        r = self.row(); r['id'] = 'b'
        b = checked_row(r, Tokenizer([1,2]), 'book', ['book:2'])
        result = deduplicate([a,b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['source_keys'], ['book:1','book:2'])
        self.assertEqual(result[0]['original_record_ids'], ['a','b'])

    def test_wrong_budget_rejected(self):
        with self.assertRaises(ValueError): checked_row(self.row(), Tokenizer([1]), 'book', ['x'])

    def test_embedded_eos_rejected(self):
        with self.assertRaises(ValueError): checked_row(self.row(), Tokenizer([1,EOS]), 'book', ['x'])

    def test_changed_text_rejected(self):
        r = self.row(); r['text'] = 'Changed'
        with self.assertRaises(ValueError): checked_row(r, Tokenizer([1,2]), 'book', ['x'])

    def test_provenance_required(self):
        with self.assertRaises(ValueError): checked_row(self.row(), Tokenizer([1,2]), 'book', [])


if __name__ == '__main__': unittest.main()
