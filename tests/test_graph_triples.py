from pathlib import Path

from caypollard.graphs.triples import (
    ProjectionAudit,
    entity_ids,
    mask_target_relations,
    read_triples_tsv,
    relation_ids,
    triples_digest,
)

FIXTURE = Path("data/samples/context_triples_fixture.tsv")


def test_triple_fixture_is_canonical_and_digest_is_stable():
    triples = read_triples_tsv(FIXTURE)
    assert triples == tuple(sorted(triples))
    assert triples_digest(triples) == triples_digest(reversed(triples))
    assert "image:a" in entity_ids(triples)
    assert "has_iconclass" in relation_ids(triples)


def test_mask_target_relations_removes_only_held_out_label_edges():
    triples = read_triples_tsv(FIXTURE)
    kept, removed = mask_target_relations(
        triples,
        target_entities={"image:a", "image:c"},
        relations={"has_iconclass"},
    )
    assert set(removed) == {
        ("image:a", "has_iconclass", "iconclass:25G411"),
        ("image:c", "has_iconclass", "iconclass:25G412"),
    }
    assert ("image:a", "part_of", "book:alpha") in kept
    assert ("image:b", "has_iconclass", "iconclass:25G411") in kept


def test_projection_audit_records_graph_shape_and_mask_count():
    triples = read_triples_tsv(FIXTURE)
    audit = ProjectionAudit.from_triples(triples, projection_id="G-fixture", masked_triples=2)
    assert audit.n_triples == len(triples)
    assert audit.n_entities == len(entity_ids(triples))
    assert audit.n_relations == len(relation_ids(triples))
    assert audit.masked_triples == 2
    assert len(audit.sha256) == 64
