from __future__ import annotations

import pytest

from caypollard.datasets.emblematica import (
    EmblemRecord,
    book_id_from_pictura,
    emblem_book_triples,
    iconclass_coverage,
    parse_emblem_xml,
)

# Glasgow writes unprefixed elements with skos:notation.
GLASGOW = """<emblem xmlns="http://diglib.hab.de/rules/schema/emblem" xml:id="FPAb046">
  <motto><transcription xml:lang="la">Horrent commota moveri.</transcription></motto>
  <pictura xlink:href="http://example.org/FPAb046.jpg">
    <figDesc>A bear breathing smoke.</figDesc>
    <iconclass rdf:about="http://www.iconclass.org/rkd/25F23(BEAR)">
      <skos:notation>25F23(BEAR)</skos:notation>
    </iconclass>
    <iconclass rdf:about="http://www.iconclass.org/rkd/31A22221">
      <skos:notation>31A22221</skos:notation>
    </iconclass>
  </pictura>
</emblem>"""

# Wolfenbüttel writes emblem:-prefixed elements with tei:p and no iconclass.
HAB = """<emblem:emblem xmlns:emblem="http://diglib.hab.de/rules/schema/emblem">
  <emblem:motto><emblem:transcription xml:lang="la">
    <tei:p xml:lang="la">Charae custodia prolis.</tei:p>
  </emblem:transcription></emblem:motto>
  <emblem:subscriptio><emblem:transcription xml:lang="la"/></emblem:subscriptio>
</emblem:emblem>"""


def test_parses_the_unprefixed_glasgow_dialect():
    record = parse_emblem_xml(GLASGOW, emblem_id="E028541", collection="Glasgow")
    assert record.iconclass == ("25F23(BEAR)", "31A22221")
    assert record.motto == "Horrent commota moveri."
    assert record.pictura_description == "A bear breathing smoke."


def test_parses_the_prefixed_wolfenbuettel_dialect():
    # Contributing libraries do not share a house style; one parser must read both.
    record = parse_emblem_xml(HAB, emblem_id="E000000", collection="HAB")
    assert record.motto == "Charae custodia prolis."
    assert record.iconclass == ()
    assert record.subscriptio is None  # present but empty


def test_text_keys_survive_parsing():
    # 25F23(BEAR) must not become 25F23 — protocol v0.5 depends on the key.
    record = parse_emblem_xml(GLASGOW, emblem_id="E1")
    assert "25F23(BEAR)" in record.iconclass


def test_notations_are_deduplicated_in_order():
    doubled = GLASGOW.replace("</pictura>", """
      <iconclass><skos:notation>25F23(BEAR)</skos:notation></iconclass></pictura>""")
    assert parse_emblem_xml(doubled, emblem_id="E1").iconclass == ("25F23(BEAR)", "31A22221")


def test_an_emblem_without_any_annotation_yields_empty_fields():
    record = parse_emblem_xml("<emblem/>", emblem_id="E2")
    assert record.iconclass == ()
    assert record.motto is None
    assert record.pictura_description is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://emblemimages.library.illinois.edu/379477343/_access/pictura/E0.jpg", "379477343"),
        ("http://emblemimages.library.illinois.edu/FPAb/_access/pictura/E1.jpg", "FPAb"),
        ("http://elsewhere.example.org/x.jpg", None),
        (None, None),
    ],
)
def test_book_id_is_recovered_from_the_pictura_url(url, expected):
    # Reading the book from the URL avoids one detail request per emblem.
    assert book_id_from_pictura(url) == expected


def test_manifest_record_matches_the_iconclass_pipeline_shape():
    record = EmblemRecord("E1", book_id="FPAb", collection="Glasgow", iconclass=("25F23(BEAR)",))
    row = record.as_manifest_record()
    assert row["id"] == "emblematica:E1"
    assert row["iconclass"] == ["25F23(BEAR)"]
    assert row["book_id"] == "FPAb"
    assert row["split"] is None and row["group_id"] is None


def test_membership_triples_skip_emblems_without_a_book():
    records = [EmblemRecord("E1", book_id="FPAb"), EmblemRecord("E2")]
    assert emblem_book_triples(records) == (("emblematica:E1", "part_of", "book:FPAb"),)


def test_coverage_report_breaks_down_by_collection():
    records = [
        EmblemRecord(
            "E1", book_id="A", collection="Glasgow", iconclass=("25F23(BEAR)",), motto="m"
        ),
        EmblemRecord("E2", book_id="A", collection="Glasgow", iconclass=("31A",)),
        EmblemRecord("E3", book_id="B", collection="HAB", motto="m"),
    ]
    report = iconclass_coverage(records)
    assert report["n_emblems"] == 3
    assert report["n_with_iconclass"] == 2
    assert report["iconclass_coverage"] == pytest.approx(2 / 3)
    assert report["n_assignments"] == 2
    assert report["n_books"] == 2
    assert report["n_with_motto"] == 2
    assert report["by_collection"]["Glasgow"] == {"labelled": 2, "total": 2}
    assert report["by_collection"]["HAB"] == {"labelled": 0, "total": 1}


def test_book_metadata_adapts_a_catalogue_row():
    from caypollard.datasets.emblematica import book_metadata

    volume = book_metadata(
        {
            "bookID": "1006844",
            "Title": "Nutzliche Anweisung zu fruchtbarer Betrachtung",
            "PublicationDate": "anno 1701",
            "PublicationPlace": "Nèurnberg",
            "Authors": ["Schade, Johann Caspar, 1666-1698", "Otto, Andreas, 1658-1723"],
        }
    )
    assert volume.volume_id == "1006844"
    assert volume.year == 1701
    assert volume.place == "neurnberg"
    assert volume.creator_names == ("schade", "otto")
    assert volume.work_title == "Nutzliche Anweisung zu"
    # The catalogue carries neither printer nor authority identifier.
    assert volume.printer is None and volume.creator_gnds == ()


def test_book_metadata_tolerates_a_sparse_row():
    from caypollard.datasets.emblematica import book_metadata

    volume = book_metadata({"bookID": "x"})
    assert volume.volume_id == "x"
    assert volume.year is None and volume.place is None and volume.creator_names == ()


def test_two_books_by_one_author_share_a_person_node():
    from caypollard.datasets.bsb_metadata import metadata_triples
    from caypollard.datasets.emblematica import book_metadata

    rows = [
        {"bookID": "b1", "Authors": ["Alciati, Andrea, 1492-1550"]},
        {"bookID": "b2", "Authors": ["Alciati, Andrea, 1492-1550"]},
    ]
    triples = metadata_triples([book_metadata(r) for r in rows])
    linked = {h for h, r, t in triples if t == "person:name-alciati"}
    assert linked == {"book:b1", "book:b2"}


def test_commented_out_template_is_not_read_as_a_notation():
    # Unindexed records carry a cataloguer's template inside an XML comment.
    # Reading it would give every unindexed emblem the same fake label.
    templated = """<emblem><pictura>
      <!--Please replace with iconclass headings assigned:
        <iconclass rdf:about="http://iconclass.org/sw/[notation]">
          <skos:notation>[notation]</skos:notation>
        </iconclass>
      -->
    </pictura></emblem>"""
    assert parse_emblem_xml(templated, emblem_id="E1").iconclass == ()


def test_real_notations_survive_alongside_a_template_comment():
    mixed = """<emblem><pictura>
      <!-- <skos:notation>[notation]</skos:notation> -->
      <iconclass><skos:notation>31A233</skos:notation></iconclass>
    </pictura></emblem>"""
    assert parse_emblem_xml(mixed, emblem_id="E2").iconclass == ("31A233",)
