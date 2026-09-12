import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_cpt_sft_search_data import audit_records


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return {'prompt': [1, 2], 'answer': [3, 4]}[text]


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(id=str(i), input_ids=[1, 2, 3, 4, 99], answer_start=2,
                          loss_tokens=3) for i in range(3)]
        self.messages = [dict(id=str(i), messages=[dict(role='user', content='prompt'),
                          dict(role='assistant', content='answer')]) for i in range(3)]

    def test_full_budget(self):
        r = audit_records(self.rows, self.messages, Tokenizer(), 3)
        self.assertEqual((r['sequence_tokens'], r['loss_tokens'], r['batches']), (15, 9, 1))

    def test_corruptions_fail(self):
        for field, value in [('answer_start', 1), ('loss_tokens', 2),
                             ('input_ids', [1, 2, 7, 4, 99]),
                             ('input_ids', [1, 2, 3, 4, 98]),
                             ('input_ids', [1, 2, 3.0, 4, 99])]:
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    audit_records(rows, self.messages, Tokenizer(), 3)

    def test_tail_rejected(self):
        with self.assertRaises(ValueError):
            audit_records(self.rows, self.messages, Tokenizer(), 2)

    def test_duplicate_messages_rejected(self):
        self.messages[1]['id'] = '0'
        with self.assertRaises(ValueError):
            audit_records(self.rows, self.messages, Tokenizer(), 3)


if __name__ == '__main__':
    unittest.main()
