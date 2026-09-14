from pathlib import Path

from caypollard.datasets.iconclass import (
    audit_annotations,
    build_manifest,
    load_testset_annotations,
)

SAMPLE = Path("data/samples/iconclass_testset_excerpt.json")


def test_load_official_excerpt():
    annotations = load_testset_annotations(SAMPLE)
    assert len(annotations) == 4
    assert "25G41" in annotations["IIHIM_-859728949.jpg"]


def test_manifest_is_deterministic_and_explicit_about_missing_groups():
    manifest = build_manifest(load_testset_annotations(SAMPLE))
    assert manifest[0]["id"].startswith("iconclass-ai:")
    assert all(record["group_id"] is None for record in manifest)
    assert manifest == sorted(manifest, key=lambda record: record["filename"])


def test_annotation_audit_counts_assignments():
    audit = audit_annotations(load_testset_annotations(SAMPLE))
    assert audit["n_images"] == 4
    assert audit["n_assignments"] == 13
    assert audit["n_unique_notations"] == 13
