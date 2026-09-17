from __future__ import annotations

import pytest

from caypollard.datasets.iconclass_provenance import (
    annotate_book_ids,
    book_coverage_report,
    book_identifier,
)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Emblem books: collection, shelfmark, then a numbered plate.
        ("uiuc_832w91o_pic012.jpg", "uiuc_832w91o"),
        ("uiuc_emblems0012_pic329.jpg", "uiuc_emblems0012"),
        ("embhab_hab_li77441_pic142.jpg", "embhab_hab_li77441"),
        ("embhab_uk-70_pic205.jpg", "embhab_uk-70"),
        ("embstma_609_pic523.jpg", "embstma_609"),
        # Digitisation identifiers and manuscript shelfmarks with folio numbers.
        ("bsb00027852_559.jpg", "bsb00027852"),
        ("120d13_002r_min_a1.jpg", "120d13"),
        ("128c1_dl1_403v_min.jpg", "128c1_dl1"),
    ],
)
def test_recovers_volumes_from_structured_filenames(filename, expected):
    assert book_identifier(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "IIHIM_-859728949.jpg",  # 72% of the corpus; a flat per-item id
        "IIHIM_1956438510.jpg",
        "ursicula_01915.jpg",
        "folger_ill_fac063106.jpg",
        "biblia_sacra_20051004027.jpg",
    ],
)
def test_refuses_to_read_flat_collection_ids_as_books(filename):
    # Treating these as books would pool tens of thousands of unrelated images
    # into one group and destroy the very splits grouping is meant to protect.
    assert book_identifier(filename) is None


def test_returns_none_rather_than_guessing_on_unstructured_names():
    assert book_identifier("mysteryimage.jpg") is None
    assert book_identifier("single_token.jpg") is None


def test_annotate_adds_book_ids_without_mutating_input():
    records = [{"id": "a", "filename": "uiuc_832w91o_pic012.jpg"}, {"id": "b"}]
    annotated = annotate_book_ids(records)
    assert annotated[0]["book_id"] == "uiuc_832w91o"
    assert annotated[1]["book_id"] is None
    assert "book_id" not in records[0]


def test_coverage_report_separates_emblems_and_unattributed_images():
    records = annotate_book_ids(
        {"filename": name}
        for name in (
            "uiuc_832w91o_pic012.jpg",
            "uiuc_832w91o_pic013.jpg",
            "embhab_uk-70_pic205.jpg",
            "bsb00027852_559.jpg",
            "IIHIM_-859728949.jpg",
        )
    )
    report = book_coverage_report(records)
    assert report["n_images"] == 5
    assert report["n_images_with_book"] == 4
    assert report["n_images_without_book"] == 1
    assert report["n_books"] == 3
    assert report["n_multi_image_books"] == 1
    assert report["largest_book_size"] == 2
    assert report["n_emblem_images"] == 3
    assert report["book_coverage"] == pytest.approx(0.8)
