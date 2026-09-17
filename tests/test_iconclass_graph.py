from pathlib import Path

import pytest

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


def test_strip_structural_keys_keeps_text_keys():
    from caypollard.graphs.iconclass import strip_structural_keys

    # (+N) is a modifier and folds away; the bracketed text names what is depicted.
    assert strip_structural_keys("25G4(PUMPKIN)(+34)") == "25G4(PUMPKIN)"
    assert strip_structural_keys("31A24(+1)") == "31A24"
    assert strip_structural_keys("86(USQUE RECURRIT)") == "86(USQUE RECURRIT)"


def test_key_augmentation_makes_two_mottoes_siblings_not_synonyms():
    from caypollard.graphs.iconclass import hierarchical_similarity, key_augmented_parents

    parents = {"86": set()}
    labels = ["86(USQUE RECURRIT)", "86(UT CAPIAS CAPIARE PRIUS)"]

    # Default policy: every emblem looks perfectly relevant to every other emblem.
    assert hierarchical_similarity(labels[0], labels[1], parents) == 1.0

    augmented = key_augmented_parents(parents, labels)
    assert augmented[labels[0]] == {"86"}
    assert hierarchical_similarity(labels[0], labels[1], augmented) == pytest.approx(1 / 3)
    # The same motto repeated is still a perfect match.
    assert hierarchical_similarity(labels[0], labels[0], augmented) == 1.0


def test_key_augmentation_only_adds_keys_present_in_the_data():
    from caypollard.graphs.iconclass import key_augmented_parents

    parents = {"86": set()}
    augmented = key_augmented_parents(parents, ["86(ONE)"])
    assert "86(ONE)" in augmented
    assert "86(TWO)" not in augmented


def test_key_augmentation_leaves_existing_vocabulary_nodes_alone():
    from caypollard.graphs.iconclass import key_augmented_parents

    parents = {"25G4": set(), "25G41": {"25G4"}, "25G41(DAISY)": {"25G41"}}
    augmented = key_augmented_parents(parents, ["25G41(DAISY)"])
    assert augmented["25G41(DAISY)"] == {"25G41"}  # not re-attached to 25G4


def test_key_augmentation_ignores_keys_whose_base_is_unknown():
    from caypollard.graphs.iconclass import key_augmented_parents

    assert "99Z(X)" not in key_augmented_parents({"86": set()}, ["99Z(X)"])


def test_resolve_prefers_the_most_specific_available_node():
    from caypollard.graphs.iconclass import key_augmented_parents, resolve_notation

    augmented = key_augmented_parents({"86": set()}, ["86(A)"])
    assert resolve_notation("86(A)", augmented) == "86(A)"
    assert resolve_notation("86(A)(+3)", augmented) == "86(A)"   # plus-key folded
    assert resolve_notation("86(B)", augmented) == "86"          # unobserved key -> base
