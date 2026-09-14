"""Deterministic knowledge-graph triple handling and leakage controls."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

Triple = tuple[str, str, str]


def normalize_triples(triples: Iterable[Sequence[str]]) -> tuple[Triple, ...]:
    """Return sorted, deduplicated, non-empty string triples."""
    normalized: set[Triple] = set()
    for index, triple in enumerate(triples, start=1):
        if len(triple) != 3:
            raise ValueError(f"triple {index} must contain exactly three values")
        head, relation, tail = (str(value).strip() for value in triple)
        if not head or not relation or not tail:
            raise ValueError(f"triple {index} contains an empty value")
        normalized.add((head, relation, tail))
    if not normalized:
        raise ValueError("at least one triple is required")
    return tuple(sorted(normalized))


def triples_digest(triples: Iterable[Sequence[str]]) -> str:
    """Return a stable SHA-256 over canonical TSV triples."""
    rows = normalize_triples(triples)
    payload = "".join("\t".join(row) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_triples_tsv(path: str | Path, *, has_header: bool = False) -> tuple[Triple, ...]:
    """Read a UTF-8 three-column TSV file into canonical triples."""
    source = Path(path)
    rows: list[Triple] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row or (row and row[0].lstrip().startswith("#")):
                continue
            if has_header and not rows and [cell.strip().lower() for cell in row] == [
                "head",
                "relation",
                "tail",
            ]:
                continue
            if len(row) != 3:
                raise ValueError(f"{source}:{line_number}: expected three TSV columns")
            rows.append((row[0], row[1], row[2]))
    return normalize_triples(rows)


def write_triples_tsv(triples: Iterable[Sequence[str]], path: str | Path) -> str:
    """Write canonical triples and return their digest."""
    rows = normalize_triples(triples)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join("\t".join(row) + "\n" for row in rows)
    target.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def entity_ids(triples: Iterable[Sequence[str]]) -> tuple[str, ...]:
    rows = normalize_triples(triples)
    return tuple(sorted({head for head, _, _ in rows} | {tail for _, _, tail in rows}))


def relation_ids(triples: Iterable[Sequence[str]]) -> tuple[str, ...]:
    rows = normalize_triples(triples)
    return tuple(sorted({relation for _, relation, _ in rows}))


def mask_target_relations(
    triples: Iterable[Sequence[str]],
    *,
    target_entities: Iterable[str],
    relations: Iterable[str] = ("has_iconclass",),
    match_tail: bool = False,
) -> tuple[tuple[Triple, ...], tuple[Triple, ...]]:
    """Remove target-label edges for held-out entities.

    By default only triples whose *head* is an evaluation entity are masked,
    matching the project's image -> target-concept edge policy. ``match_tail``
    can additionally protect projections where the target entity appears on the
    tail side of an equivalent relation.
    """
    rows = normalize_triples(triples)
    targets = {str(value) for value in target_entities}
    blocked_relations = {str(value) for value in relations}
    if not targets:
        raise ValueError("target_entities must not be empty")
    if not blocked_relations:
        raise ValueError("relations must not be empty")

    kept: list[Triple] = []
    removed: list[Triple] = []
    for triple in rows:
        head, relation, tail = triple
        should_mask = relation in blocked_relations and (
            head in targets or (match_tail and tail in targets)
        )
        (removed if should_mask else kept).append(triple)
    if not kept:
        raise ValueError("masking removed every triple from the graph")
    return tuple(kept), tuple(removed)


@dataclass(frozen=True)
class ProjectionAudit:
    """Minimal machine-readable audit for a graph projection."""

    projection_id: str
    n_triples: int
    n_entities: int
    n_relations: int
    sha256: str
    masked_triples: int = 0

    @classmethod
    def from_triples(
        cls,
        triples: Iterable[Sequence[str]],
        *,
        projection_id: str,
        masked_triples: int = 0,
    ) -> ProjectionAudit:
        rows = normalize_triples(triples)
        return cls(
            projection_id=projection_id,
            n_triples=len(rows),
            n_entities=len(entity_ids(rows)),
            n_relations=len(relation_ids(rows)),
            sha256=triples_digest(rows),
            masked_triples=masked_triples,
        )
