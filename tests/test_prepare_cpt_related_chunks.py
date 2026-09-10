from collections import Counter

import pytest

from scripts.prepare_cpt_related_chunks import pack_related, render, vectors, cosine


def test_pack_preserves_every_unit_once_and_joins_related_cross_chapters():
    units = [dict(chapter=1, text='inventory storage warehouse stock holding costs'),
             dict(chapter=2, text='vehicle routing transport delivery road distance'),
             dict(chapter=3, text='inventory stock warehouse'),
             dict(chapter=4, text='vehicle transport road')]
    bins, _ = pack_related(units, len, groups=2, limit=140)
    assert Counter(i for b in bins for i in b) == Counter(range(4))
    assert any(set(b) == {0, 2} for b in bins)
    assert any(set(b) == {1, 3} for b in bins)
    assert all(len(render(units, b)) <= 140 for b in bins)


def test_oversize_intact_unit_is_rejected_without_truncation():
    with pytest.raises(ValueError, match='exceeds'):
        pack_related([dict(chapter=1, text='x'*100)], len, groups=1, limit=50)


def test_no_room_never_drops_source_units():
    units = [dict(chapter=i, text='inventory stock'*3) for i in range(3)]
    with pytest.raises(ValueError, match='Cannot fit'):
        pack_related(units, len, groups=1, limit=90)


def test_cosine_empty_and_identical_documents():
    a, b = vectors(['inventory stock', 'inventory stock'])
    assert cosine(a, b) == pytest.approx(1)
    assert cosine({}, b) == 0
