from __future__ import annotations

import pytest

from caypollard.graphs.context import (
    ADJACENT_TO,
    PART_OF,
    context_graph_report,
    context_triples,
    plate_position,
)


def _records():
    return [
        {"id": "img:a", "filename": "uiuc_832w91o_pic012.jpg", "book_id": "uiuc_832w91o"},
        {"id": "img:b", "filename": "uiuc_832w91o_pic003.jpg", "book_id": "uiuc_832w91o"},
        {"id": "img:c", "filename": "uiuc_832w91o_pic030.jpg", "book_id": "uiuc_832w91o"},
        {"id": "img:d", "filename": "embhab_uk-70_pic205.jpg", "book_id": "embhab_uk-70"},
        # 83% of the real corpus looks like this: no recoverable volume.
        {"id": "img:e", "filename": "IIHIM_-859728949.jpg", "book_id": None},
    ]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("uiuc_832w91o_pic012.jpg", 12),
        ("bsb00027852_559.jpg", 559),
        ("120d13_002r_min_a1.jpg", 1),
        ("embhab_uk-70_pic205.jpg", 205),
        ("noplatenumber.jpg", None),
    ],
)
def test_plate_position_reads_the_trailing_number(filename, expected):
    assert plate_position(filename) == expected


def test_emits_membership_for_every_attributed_image():
    triples = context_triples(_records(), adjacency_window=0)
    memberships = {(h, t) for h, r, t in triples if r == PART_OF}
    assert memberships == {
        ("img:a", "book:uiuc_832w91o"),
        ("img:b", "book:uiuc_832w91o"),
        ("img:c", "book:uiuc_832w91o"),
        ("img:d", "book:embhab_uk-70"),
    }


def test_never_emits_an_edge_for_an_unattributed_image():
    # Linking these to a placeholder node would invent provenance the source
    # does not supply, and would pool 73 099 unrelated images on the real corpus.
    triples = context_triples(_records())
    assert not any("img:e" in (head, tail) for head, _, tail in triples)


def test_adjacency_follows_plate_order_not_input_order():
    triples = context_triples(_records(), adjacency_window=1)
    adjacent = {(h, t) for h, r, t in triples if r == ADJACENT_TO}
    # pic003 -> pic012 -> pic030, regardless of the order records arrived in.
    assert adjacent == {("img:b", "img:a"), ("img:a", "img:c")}


def test_adjacency_window_widens_the_neighbourhood():
    triples = context_triples(_records(), adjacency_window=2)
    adjacent = {(h, t) for h, r, t in triples if r == ADJACENT_TO}
    assert ("img:b", "img:c") in adjacent


def test_adjacency_never_crosses_a_book_boundary():
    triples = context_triples(_records(), adjacency_window=5)
    for head, relation, tail in triples:
        if relation == ADJACENT_TO:
            assert not (head == "img:d" or tail == "img:d")


def test_output_is_deterministic_and_deduplicated():
    records = _records()
    first = context_triples(records)
    assert first == context_triples(list(reversed(records)))
    assert len(first) == len(set(first))


def test_rejects_a_negative_window():
    with pytest.raises(ValueError):
        context_triples(_records(), adjacency_window=-1)


def test_report_states_that_no_target_label_entered_the_graph():
    records = _records()
    report = context_graph_report(context_triples(records), records=records)
    assert report["carries_target_labels"] is False
    assert report["n_images_in_graph"] == 4
    assert report["n_books"] == 2
    assert report["n_images_total"] == 5
    assert report["graph_coverage"] == pytest.approx(0.8)
    assert set(report["relations"]) == {PART_OF, ADJACENT_TO}
