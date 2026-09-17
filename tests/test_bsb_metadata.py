from __future__ import annotations

import pytest

from caypollard.datasets.bsb_metadata import (
    VolumeMetadata,
    metadata_report,
    metadata_triples,
    parse_manifest,
    slug,
)


def _entry(label: str, value):
    return {"label": [{"@language": "en", "@value": label}], "value": value}


def _manifest(**fields):
    return {"metadata": [_entry(label, value) for label, value in fields.items()]}


def test_slug_folds_accents_and_strips_cataloguer_brackets():
    assert slug("[Heidelberg]") == "heidelberg"
    assert slug("Auguste vindelicor[um]") == "auguste-vindelicorum"
    assert slug("Nürnberg") == "nurnberg"
    assert slug("   ") is None
    assert slug(None) is None


def test_imprint_splits_on_the_first_colon_only():
    # Printer statements contain further colons; splitting on all of them truncates.
    volume = parse_manifest(
        _manifest(Published="Auguste vindelicor[um] : Erhardi ratdolt : viri solertis"),
        volume_id="bsb1",
    )
    assert volume.place == "auguste-vindelicorum"
    assert volume.printer == "erhardi-ratdolt-viri-solertis"


def test_imprint_without_a_printer_still_yields_a_place():
    assert parse_manifest(_manifest(Published="[Heidelberg]"), volume_id="b").place == "heidelberg"


def test_year_is_recovered_from_a_messy_date_statement():
    volume = parse_manifest(
        _manifest(Date="xiiij. kal[endas] Dece[m]bris. M.cccc.lxxxviij. [1488.11.18.]"),
        volume_id="b",
    )
    assert volume.year == 1488
    assert parse_manifest(_manifest(Date="[nicht nach 1488]"), volume_id="b").year == 1488
    assert parse_manifest(_manifest(Date="undatiert"), volume_id="b").year is None


def test_gnd_identifiers_are_extracted_and_deduplicated():
    volume = parse_manifest(
        _manifest(
            Creator="Abū-Maʿšar -- (GND: <a href='https://d-nb.info/gnd/11914512X'>11914512X</a>)",
            Contributor="Johannes, Hispanus -- (GND: <a href='https://d-nb.info/gnd/100949754'>x</a>)",
        ),
        volume_id="b",
    )
    assert volume.creator_gnds == ("11914512X", "100949754")


def test_html_is_stripped_from_values():
    volume = parse_manifest(_manifest(Title="<span>Totentanz</span>"), volume_id="b")
    assert volume.title == "Totentanz"


def test_authority_identifiers_win_over_names():
    # Two people can share a spelling; an authority id cannot be conflated that way.
    with_gnd = VolumeMetadata("b1", creator_gnds=("118", ), creator_names=("mueller",))
    triples = metadata_triples([with_gnd])
    assert ("book:b1", "created_by", "person:gnd-118") in triples
    assert not any(t[2].startswith("person:name-") for t in triples)


def test_names_are_used_only_when_no_authority_id_exists():
    without = VolumeMetadata("b2", creator_names=("ratdolt",))
    assert ("book:b2", "created_by", "person:name-ratdolt") in metadata_triples([without])


def test_two_volumes_by_one_author_share_a_node():
    a = VolumeMetadata("b1", creator_gnds=("118",))
    b = VolumeMetadata("b2", creator_gnds=("118",))
    triples = metadata_triples([a, b])
    people = {t[0] for t in triples if t[2] == "person:gnd-118"}
    assert people == {"book:b1", "book:b2"}  # the cross-volume edge G1 could not provide


def test_dates_become_decades_and_can_be_ablated():
    volume = VolumeMetadata("b", year=1488)
    assert ("book:b", "published_in", "decade:1480s") in metadata_triples([volume])
    assert not any(t[1] == "published_in" for t in metadata_triples([volume], decade_nodes=False))


def test_triples_are_sorted_deduplicated_and_order_independent():
    volumes = [VolumeMetadata("b1", place="mainz"), VolumeMetadata("b2", place="mainz")]
    first = metadata_triples(volumes)
    assert first == metadata_triples(list(reversed(volumes)))
    assert len(first) == len(set(first))


def test_a_volume_with_no_usable_field_contributes_nothing():
    assert metadata_triples([VolumeMetadata("bare")]) == ()


def test_report_exposes_field_coverage():
    volumes = [
        VolumeMetadata("b1", place="mainz", year=1488, creator_gnds=("118",)),
        VolumeMetadata("b2", place="mainz"),
    ]
    report = metadata_report(volumes)
    assert report["n_volumes"] == 2
    assert report["coverage"]["place"] == pytest.approx(1.0)
    assert report["coverage"]["year"] == pytest.approx(0.5)
    assert report["n_distinct_places"] == 1
    assert report["year_range"] == [1488, 1488]
