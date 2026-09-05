import pytest

from scripts.probe_logistics_exam_answers import split_answer, encode_case, summarize
from scripts.repair_logistics_exam_knowledge import risk_flags, candidate_valid, audit_valid
from scripts.compare_logistics_answer_probes import compare
from scripts.probe_logistics_packed_answers import answer_spans


def test_split_preserves_exact_historical_boundary():
    for text in ['Question: Q\nCorrect answer: Air', 'Question: Q\nCorrect answers: Air; Rail',
                 'Statement: "X"\nThis statement is false.']:
        prefix, answer = split_answer(text)
        assert prefix + answer == text
        assert answer.startswith(' ')
    with pytest.raises(ValueError):
        split_answer('unrecognized')


def test_answer_token_crossing_prefix_boundary_is_counted():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            boundary = len('Question: Q\nCorrect answer:')
            return {'input_ids': [10,20,30], 'offset_mapping': [(0,8),(8,boundary-1),(boundary-1,len(text))]}
    result = encode_case(Tokenizer(), {'training_document': 'Question: Q\nCorrect answer: Air',
                                       'item_hash':'x', 'arm':'direct'})
    assert result['start'] == 2
    assert result['boundary_crossing'] is True


def test_loss_aggregation_is_token_weighted():
    rows = [{'arm':'direct','answer_tokens':n,'answer_nll_sum':v,'prefix_tokens':1,'prefix_nll_sum':2,
             'answer_top1':1,'greedy_gold_token_prefix_exact':False,'generation_censored':False,
             'boundary_crossing':False} for n,v in [(1,1),(3,9)]]
    assert summarize(rows)['direct']['answer_nll'] == 2.5


def test_packed_answer_spans_exclude_other_questions_and_eos():
    text = 'Question: One\nCorrect answer: Alpha<eos>\n\nStatement: "Two"\nThis statement is false.'
    spans = answer_spans(text,'<eos>\n\n')
    assert [text[a:b] for a,b in spans] == [' Alpha',' false.']


def test_risk_flags_and_false_wrapper_rejected():
    row = {'question':'Which is NOT a warehouse task?', 'question_type':'single_choice',
           'options':['All of the above'], 'expected':[0]}
    assert set(risk_flags(row)) == {'negative_stem','option_dependency'}
    assert not candidate_valid({'status':'candidate','text':'Statement: X\nThis statement is false.'})
    assert not candidate_valid({'status':'withheld','text':'maybe'})
    assert candidate_valid({'status':'candidate','text':'Line-haul transport is not a warehouse operation.'})
    assert not audit_valid({'entailed_by_gold':True})
    assert not audit_valid(dict.fromkeys(['entailed_by_gold','polarity_preserved','scope_preserved',
                                         'all_gold_covered','standalone','no_unsupported_facts'], 'true'))


def test_paired_comparison_rejects_different_source_or_coverage():
    with pytest.raises(ValueError, match='source mismatch'):
        compare({'source_hashes':{'a':'x'}}, {'source_hashes':{'a':'y'}})
    row = {'arm':'direct','item_hash':'x','answer_tokens':2,'prefix_tokens':2,'answer_nll_sum':4,
           'prefix_nll_sum':2,'answer_top1':1,'greedy_gold_token_prefix_exact':False,
           'generation_censored':False,'boundary_crossing':False}
    base = {'source_hashes':{'a':'x'}, 'rows':[row]}
    candidate = {'source_hashes':{'a':'x'}, 'rows':[{**row,'answer_nll_sum':2,'greedy_gold_token_prefix_exact':True}]}
    result = compare(base, candidate)['direct']
    assert result['relative_answer_nll_reduction_percent'] == 50
    assert result['greedy_improved'] == 1
    candidate['rows'][0]['answer_tokens'] = 3
    with pytest.raises(ValueError, match='tokenization mismatch'):
        compare(base,candidate)
