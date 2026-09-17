"""Recover book-level provenance from Iconclass AI Test Set filenames.

``data.json`` exposes only ``filename -> notations``, which is why
:func:`caypollard.datasets.iconclass.build_manifest` leaves ``group_id`` null
rather than inventing provenance. The filenames themselves, however, are not
opaque: a substantial minority encode the digitised object an image was cut
from, in the form ``<collection>_<book>_<plate>``.

Recovering that identifier matters twice over. It supplies the only source-borne
grouping available for leakage-controlled splitting — two plates of one emblem
book must not straddle a train/test boundary — and it is the sole non-target
relation in this corpus, so it is also the only material from which a
context-only ``G1`` projection could be built here at all.

Coverage is partial and deliberately conservative. Roughly 85% of the test set
consists of flat per-item identifiers (``IIHIM_-859728949``, ``ursicula_01915``,
``folger_ill_fac063106``) where the leading token names a *collection*, not a
volume. Reading those as books would pool tens of thousands of unrelated images
into a single group and silently destroy the splits it is meant to protect, so
they are reported as ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

EMBLEM_COLLECTIONS = ("uiuc", "embhab", "embstma", "embmne", "embepu")
"""Emblem-book collections whose second token is a per-book shelfmark."""

FLAT_COLLECTIONS = ("IIHIM", "ursicula", "folger", "dpd", "biblia")
"""Collections whose identifiers are per-item, with no recoverable volume."""

_PLATE = re.compile(r"_(?:pic|fac|ill)[0-9].*$", re.IGNORECASE)
_FOLIO = re.compile(r"_\d{1,4}[rv](?:_.*)?$")
_PAGE = re.compile(r"_\d{1,5}$")

# A single-token remainder is only a volume when it reads like a shelfmark or a
# digitisation identifier -- it must contain a digit and no separator, as in
# ``bsb00027852`` or the manuscript shelfmark ``120d13``.
_SHELFMARK = re.compile(r"[a-z]*\d[a-z0-9-]*", re.IGNORECASE)


def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


def book_identifier(filename: str) -> str | None:
    """Return the digitised volume a filename belongs to, or ``None``.

    Returns ``None`` whenever the volume cannot be recovered with confidence,
    which is the honest outcome for most of this corpus.
    """
    stem = _stem(filename)
    collection = stem.split("_", 1)[0]
    if collection in FLAT_COLLECTIONS:
        return None

    for pattern in (_PLATE, _FOLIO, _PAGE):
        trimmed = pattern.sub("", stem)
        if trimmed == stem:
            continue
        if not trimmed or (trimmed == collection and "_" not in trimmed):
            # The remainder collapsed to the bare collection name; only accept it
            # when that name is itself a shelfmark rather than a collection word.
            if _SHELFMARK.fullmatch(collection):
                return collection
            return None
        if collection in EMBLEM_COLLECTIONS or "_" in trimmed:
            return trimmed
        if _SHELFMARK.fullmatch(trimmed):
            return trimmed
        return None
    return None


def annotate_book_ids(
    records: Iterable[dict[str, Any]],
    *,
    filename_key: str = "filename",
    book_key: str = "book_id",
) -> list[dict[str, Any]]:
    """Return copies of ``records`` carrying a recovered ``book_id`` or ``None``."""
    rows = []
    for record in records:
        row = dict(record)
        filename = row.get(filename_key)
        row[book_key] = book_identifier(str(filename)) if filename else None
        rows.append(row)
    return rows


def book_coverage_report(
    records: Iterable[dict[str, Any]], *, book_key: str = "book_id"
) -> dict[str, Any]:
    """Summarise how much of the corpus carries recoverable book provenance."""
    counts: dict[str, int] = {}
    total = 0
    without = 0
    for record in records:
        total += 1
        book = record.get(book_key)
        if book is None:
            without += 1
            continue
        counts[str(book)] = counts.get(str(book), 0) + 1

    multi = sorted((size for size in counts.values() if size > 1), reverse=True)
    with_book = total - without
    emblem = sum(
        size
        for book, size in counts.items()
        if book.split("_", 1)[0] in EMBLEM_COLLECTIONS
    )
    return {
        "n_images": total,
        "n_images_with_book": with_book,
        "book_coverage": with_book / total if total else 0.0,
        "n_images_without_book": without,
        "n_books": len(counts),
        "n_multi_image_books": len(multi),
        "largest_book_size": multi[0] if multi else 0,
        "n_emblem_images": emblem,
    }
