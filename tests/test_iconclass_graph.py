from pathlib import Path

from caypollard.graphs.iconclass import (
    build_parent_index,
    child_edges,
    hierarchical_similarity,
    normalize_notation,
    parse_notations,
    semantic_distance,
    to_skos_graph,
)

FIXTURE = Path("data/samples/iconclass_notations_fixture.txt")


def test_parse_documented_iconclass_branch():
    records = parse_notations(FIXTURE)
    assert records["25G41"]["C"][:3] == ["25G41(...)", "25G411", "25G412"]
    assert ("25G4", "25G41") in child_edges(records)


def test_normalize_key_and_name_qualifiers():
    assert normalize_notation("31A24(+1)") == "31A24"
    assert normalize_notation("11H(BASIL THE GREAT)") == "11H"
    assert normalize_notation("25F23(LION)(+46)") == "25F23"


def test_hierarchical_distance_and_similarity():
    parents = build_parent_index(child_edges(parse_notations(FIXTURE)))
    assert semantic_distance("25G411", "25G412", parents) == 2
    assert hierarchical_similarity("25G41", "25G411", parents) == 0.5
    assert hierarchical_similarity("25G411", "25G412", parents) == 1 / 3
    assert hierarchical_similarity("25G41(+1)", "25G41", parents) == 0.5


def test_skos_export_contains_broader_relation():
    graph = to_skos_graph(parse_notations(FIXTURE))
    ttl = graph.serialize(format="turtle")
    assert "skos:broader" in ttl
    assert "25G411" in ttl


def test_multilabel_image_similarity_uses_best_supported_relation():
    from caypollard.graphs.iconclass import image_hierarchical_similarity

    parents = build_parent_index(child_edges(parse_notations(FIXTURE)))
    assert image_hierarchical_similarity(["unknown", "25G411"], ["25G412"], parents) == 1 / 3
    assert image_hierarchical_similarity([], ["25G412"], parents) == 0.0
