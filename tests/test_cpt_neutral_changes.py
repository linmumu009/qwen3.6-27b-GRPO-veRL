import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from audit_cpt_neutral_changes import equivalent_indices, same_answer_text


def test_duplicate_indices_keep_original_pool_positions():
    assert equivalent_indices(['TRAILER', 'Other', ' Trailer ', 'Semi-trailer']) == [[0, 2]]


def test_equivalence_only_normalizes_case_and_space():
    options = ['ARTICULATED VEHICLE', 'Articulated  Vehicle', 'ULD', 'Unit Load Device']
    assert same_answer_text(options, [0], [1])
    # Abbreviation equivalence requires source review, not a guessed string rule.
    assert not same_answer_text(options, [2], [3])
    assert not same_answer_text(options, None, None)


def test_multiselect_comparison_does_not_erase_extra_selected_option():
    options = ['X', 'x', 'Y']
    assert same_answer_text(options, [0, 2], [2, 1])
    assert not same_answer_text(options, [0], [0, 1])
