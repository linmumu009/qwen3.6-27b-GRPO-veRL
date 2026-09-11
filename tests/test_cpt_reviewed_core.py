import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('core', Path(__file__).parents[1] / 'scripts/build_cpt_reviewed_core.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type('Encoding', (), {'ids': [1] * len(text)})()


class ReviewedCoreTests(unittest.TestCase):
    def setUp(self):
        text = 'A trailer. Agricultural trailers are excluded.'
        self.sources = {'book:1': {'text': text, 'sha256': core.digest(text.encode())}}
        self.unit = dict(title='Trailer', body=text, scope='2019 glossary', topics=['vehicles'],
                         kind='source_excerpt', review_status='reviewed_selected_assertions',
                         evidence=[dict(key='book:1', sha256=self.sources['book:1']['sha256'], excerpt=text)])

    def test_short_definition_preserves_heading_scope_exclusion_and_eos(self):
        text, ids = core.validate_unit(self.unit, self.sources, Tokenizer())
        self.assertIn('Trailer\n\n2019 glossary', text)
        self.assertIn('Agricultural trailers are excluded.', text)
        self.assertEqual(ids[-1], core.EOS)
        self.assertEqual(ids.count(core.EOS), 1)

    def test_unreviewed_candidate_rejected(self):
        self.unit['review_status'] = 'candidate'
        with self.assertRaises(ValueError): core.validate_unit(self.unit, self.sources, Tokenizer())

    def test_silent_exclusion_removal_rejected(self):
        self.unit['body'] = 'A trailer.'
        with self.assertRaises(ValueError): core.validate_unit(self.unit, self.sources, Tokenizer())

    def test_changed_source_rejected(self):
        self.sources['book:1']['text'] += ' changed'
        with self.assertRaises(ValueError): core.validate_unit(self.unit, self.sources, Tokenizer())

    def test_oversize_unit_rejected_without_truncation(self):
        self.unit['kind'] = 'editorial_synthesis'
        self.unit['body'] = 'x' * 4096
        with self.assertRaises(ValueError): core.validate_unit(self.unit, self.sources, Tokenizer())


if __name__ == '__main__':
    unittest.main()
