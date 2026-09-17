"""Build a context-only ``G1`` graph from recovered book provenance.

``G1`` requires relations that are not the evaluation target. On the Iconclass AI
Test Set the only such relations available are those recovered from filenames by
:mod:`caypollard.datasets.iconclass_provenance`: which volume a plate came from,
and where it sits in that volume.

That is a thin graph, and the thinness is the point. It supports one narrow
question — whether co-membership of a book, and proximity within it, carry
iconographic signal a visual encoder misses — and it cannot support the broader
contextual hypothesis, which needs creators, printers, places, and dates. Those
relations are reachable by joining UIUC and HAB catalogue metadata on the
shelfmarks already present in these filenames, but they are not in this source.

No image-to-Iconclass edge is emitted here. Feeding the evaluation labels into
the graph is what makes the ``G0`` taxonomy projection an oracle rather than
evidence, and this module exists precisely to avoid that.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

PART_OF = "part_of"
"""Image belongs to a digitised volume."""

ADJACENT_TO = "adjacent_to"
"""Two plates sit next to each other in the same volume."""

_SEQUENCE = re.compile(r"(\d+)(?!.*\d)")


def plate_position(filename: str) -> int | None:
    """Return the trailing number of a filename, used as position within a book.

    Plate and page identifiers end in a number (``pic125``, ``_559``), and folio
    identifiers end in a number plus a recto/verso marker (``002r``). The final
    number is therefore a usable ordinal. It is *not* a page count: gaps and
    restarts are expected, so it orders plates without claiming to measure
    distance between them.
    """
    stem = filename.rsplit(".", 1)[0]
    match = _SEQUENCE.search(stem)
    return int(match.group(1)) if match else None


def context_triples(
    records: Iterable[dict[str, Any]],
    *,
    id_key: str = "id",
    filename_key: str = "filename",
    book_key: str = "book_id",
    adjacency_window: int = 1,
) -> tuple[tuple[str, str, str], ...]:
    """Emit ``part_of`` and ``adjacent_to`` triples for records carrying a book.

    Records without a recovered book contribute nothing rather than being linked
    to a placeholder node, so the graph covers a sub-corpus and says so by its
    size. ``adjacency_window`` of 0 disables the ordering relation, leaving pure
    volume membership — the minimal G1 worth testing.
    """
    if adjacency_window < 0:
        raise ValueError("adjacency_window cannot be negative")

    by_book: dict[str, list[tuple[int | None, str]]] = {}
    triples: list[tuple[str, str, str]] = []

    for record in records:
        book = record.get(book_key)
        identifier = record.get(id_key)
        if book is None or identifier is None:
            continue
        head = str(identifier)
        tail = f"book:{book}"
        triples.append((head, PART_OF, tail))
        by_book.setdefault(str(book), []).append(
            (plate_position(str(record.get(filename_key, ""))), head)
        )

    for members in by_book.values():
        # Unpositioned plates sort last and deterministically by identifier, so a
        # book whose numbering cannot be read still yields a stable ordering.
        ordered = sorted(members, key=lambda item: (item[0] is None, item[0] or 0, item[1]))
        for index, (_, head) in enumerate(ordered):
            for offset in range(1, adjacency_window + 1):
                if index + offset < len(ordered):
                    triples.append((head, ADJACENT_TO, ordered[index + offset][1]))

    return tuple(sorted(set(triples)))


def context_graph_report(
    triples: Sequence[Sequence[str]], *, records: Iterable[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Summarise the coverage and shape of a context projection."""
    images = {triple[0] for triple in triples if triple[1] == PART_OF}
    books = {triple[2] for triple in triples if triple[1] == PART_OF}
    relations: dict[str, int] = {}
    for triple in triples:
        relations[triple[1]] = relations.get(triple[1], 0) + 1

    report: dict[str, Any] = {
        "n_triples": len(triples),
        "n_images_in_graph": len(images),
        "n_books": len(books),
        "relations": dict(sorted(relations.items())),
        "carries_target_labels": False,
    }
    if records is not None:
        total = sum(1 for _ in records)
        report["n_images_total"] = total
        report["graph_coverage"] = len(images) / total if total else 0.0
    return report
