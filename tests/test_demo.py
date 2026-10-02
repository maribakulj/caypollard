"""The demo engine: pure parts that need no corpus, and the fitted-model contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from caypollard.demo import explain, namer
from caypollard.demo.models import Atoms, l2
from caypollard.demo.pool import CHANNELS, Pool


def test_every_step_the_page_orders_is_explained() -> None:
    page = (Path(__file__).resolve().parents[1] / "viewer/demo.js").read_text(encoding="utf-8")
    start = page.index("const ORDER = [")
    order = page[start : page.index("];", start)]
    for key in explain.STEPS:
        assert f'"{key}"' in order, f"{key} is explained but never shown"
    for step in explain.STEPS.values():
        assert step["title"] and step["what"] and step["code"]


def test_defaults_cover_every_documented_parameter() -> None:
    documented = {k for step in explain.STEPS.values() for k in step.get("params", {})}
    assert documented <= set(explain.DEFAULTS)


def test_namer_parse_keeps_the_original_rules() -> None:
    words = namer.vocabulary()
    assert len(words) == 116
    answer = json.dumps(
        {
            "nodes": [
                {"concept": "man", "place": "centre"},
                {"concept": "man", "place": "centre"},  # duplicate pair: dropped
                {"concept": "unicorn", "place": "centre"},  # not in the vocabulary: dropped
                {"concept": "table", "place": "nowhere"},  # not a cell: dropped
                {"concept": "Table", "place": "bas-centre"},  # case is normalised
            ]
        }
    )
    nodes = namer.parse("```json\n" + answer + "\n```", max_nodes=6)
    assert nodes == [{"name": "man", "cell": 4}, {"name": "table", "cell": 7}]
    assert namer.parse("no json here", max_nodes=6) == []
    assert (
        len(
            namer.parse(
                json.dumps({"nodes": [{"concept": w, "place": "centre"} for w in words[:9]]}),
                max_nodes=3,
            )
        )
        == 3
    )


def test_atoms_assign_is_nearest_centre_in_standardised_space(tmp_path: Path) -> None:
    centres = np.asarray([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    np.savez(
        tmp_path / "atoms.npz",
        centre=np.zeros(2, np.float32),
        scale=np.ones(2, np.float32),
        centres=centres,
    )
    atoms = Atoms(tmp_path / "atoms.npz")
    described = np.asarray([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0]], dtype=np.float32)
    assert atoms.assign(described).tolist() == [0, 1, 2]
    assert atoms.distances(described).shape == (3, 3)


class FakePool(Pool):
    """A pool whose tables are given, so mixing and ranking can be checked by hand."""

    def __init__(self, tables: dict[str, tuple[list[str], np.ndarray]]) -> None:
        super().__init__(items_path=Path("/nonexistent"))
        self._given = tables
        self._index = {
            item: i for i, item in enumerate(sorted({x for ids, _ in tables.values() for x in ids}))
        }

    def table(self, key):  # type: ignore[override]
        ids, rows = self._given[key]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return ids, (rows / norms).astype(np.float32)


def test_search_ranks_by_cosine_and_mixes_by_weighted_concatenation() -> None:
    a = (["x", "y", "z"], np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    b = (["x", "y"], np.asarray([[0, 1], [1, 0]], dtype=np.float32))
    pool = FakePool({"pixels": a, "palette": b})
    query = {"pixels": np.asarray([1, 0], np.float32), "palette": np.asarray([1, 0], np.float32)}

    single = pool.search(query, {"pixels": 1.0}, k=3)
    assert [r["id"] for r in single["results"]] == ["x", "z", "y"]
    assert single["pool"] == 3

    mixed = pool.search(query, {"pixels": 1.0, "palette": 1.0}, k=3)
    assert mixed["pool"] == 2, "only pictures present in every channel are ranked"
    assert [r["id"] for r in mixed["results"]] == ["x", "y"]
    assert mixed["results"][0]["score"] == pytest.approx(0.5)  # pixels agree, palette disagrees

    assert pool.search(query, {"pixels": 0.0}, k=3)["results"] == []
    assert [r["id"] for r in pool.search(query, {"pixels": 1.0}, k=3, exclude="x")["results"]] == [
        "z",
        "y",
    ]


def test_channels_name_their_tables_and_regimes() -> None:
    assert set(CHANNELS) == {
        "pixels",
        "croquis",
        "traits",
        "silhouette",
        "formes",
        "relations",
        "composition",
        "repetition",
        "record",
        "palette",
        "signal",
        "nommes",
        "pose",
        "mix",
    }
    assert all(
        c.regime in {"image", "trait", "surface", "semantique", "mixte"} for c in CHANNELS.values()
    )


def test_l2_leaves_the_zero_vector_alone() -> None:
    assert l2(np.zeros(3, np.float32)).tolist() == [0.0, 0.0, 0.0]
    assert np.linalg.norm(l2(np.asarray([3.0, 4.0]))) == pytest.approx(1.0)
