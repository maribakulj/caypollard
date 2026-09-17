"""Parse Emblematica Online emblem records into benchmark-ready material.

The Iconclass AI Test Set is a sample drawn from Arkyves, a subscription
database, so its filenames carry Arkyves-internal collection codes with no public
key back to any catalogue. That is what capped bibliographic enrichment at zero
reachable hard positives: the identifiers could not be resolved, not because the
relations were missing.

Emblematica Online removes the problem rather than working around it. One open
corpus carries, for the same object: the pictura image, the Iconclass notations
that serve as evaluation ground truth, the motto and subscriptio as a genuine
text modality, and a book identifier that resolves to author, place and date. It
is therefore the first source in this project where the graph and the labels
coexist natively.

Coverage is not uniform and is measured rather than assumed: `iconclass_coverage`
reports how many emblems actually carry notations, sampled at ingestion time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .bsb_metadata import VolumeMetadata, slug


# Records come from several contributing libraries and are not consistently
# namespaced: Glasgow writes unprefixed elements under a default namespace with
# `skos:notation`, Wolfenbüttel writes `emblem:`-prefixed elements with `tei:p`.
# Matching is therefore prefix-agnostic rather than assuming one house style.
def _tag(name: str) -> str:
    return rf"<(?:\w+:)?{name}\b[^>]*>(.*?)</(?:\w+:)?{name}>"


_ICONCLASS = re.compile(_tag("iconclass"), re.S)
_NOTATION = re.compile(_tag("notation"), re.S)
_PARAGRAPH = re.compile(
    r"<(?:\w+:)?(?:p|transcription)\b[^>]*>(.*?)</(?:\w+:)?(?:p|transcription)>", re.S
)
_FIGDESC = re.compile(_tag("figDesc"), re.S)
_BLOCK = {
    "motto": re.compile(_tag("motto"), re.S),
    "subscriptio": re.compile(_tag("subscriptio"), re.S),
}
_TAGS = re.compile(r"<[^>]+>")
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_PICTURA_BOOK = re.compile(r"emblemimages\.library\.illinois\.edu/(\w+)/")


def _plain(fragment: str | None) -> str | None:
    if not fragment:
        return None
    text = re.sub(r"\s+", " ", _TAGS.sub(" ", fragment)).strip()
    return text or None


def book_id_from_pictura(url: str | None) -> str | None:
    """Recover the book identifier embedded in a pictura image URL.

    The search endpoint returns a pictura URL but not the book, while the detail
    endpoint returns the book at the cost of one request per emblem. Since the
    book is already present in the URL path, reading it there avoids 33 000
    extra round trips against a library server.
    """
    if not url:
        return None
    match = _PICTURA_BOOK.search(url)
    return match.group(1) if match else None


@dataclass(frozen=True)
class EmblemRecord:
    """One emblem: its labels, its texts, and the volume it belongs to."""

    emblem_id: str
    book_id: str | None = None
    collection: str | None = None
    iconclass: tuple[str, ...] = field(default_factory=tuple)
    motto: str | None = None
    subscriptio: str | None = None
    pictura_description: str | None = None
    pictura_url: str | None = None

    def as_manifest_record(self) -> dict[str, Any]:
        """Shape the record like an Iconclass-AI manifest row.

        Reusing the existing field names lets the audit, splitting, retrieval and
        fusion code run unchanged on this corpus, so the second benchmark is a
        new data source rather than a second pipeline.
        """
        return {
            "id": f"emblematica:{self.emblem_id}",
            "filename": f"{self.emblem_id}.jpg",
            "iconclass": list(self.iconclass),
            "source_dataset": "Emblematica Online",
            "source_url": "https://emblematica.library.illinois.edu/",
            "book_id": self.book_id,
            "collection": self.collection,
            "motto": self.motto,
            "subscriptio": self.subscriptio,
            "pictura_description": self.pictura_description,
            "pictura_url": self.pictura_url,
            "group_id": None,
            "sha256": None,
            "split": None,
        }


def parse_emblem_xml(
    xml: str,
    *,
    emblem_id: str,
    book_id: str | None = None,
    collection: str | None = None,
    pictura_url: str | None = None,
) -> EmblemRecord:
    """Extract notations and transcribed texts from one SPINE emblem record.

    Comments are removed first. Records awaiting indexing carry a commented-out
    template — ``<!--Please replace with iconclass headings assigned: ...
    <skos:notation>[notation]</skos:notation> ... -->`` — and reading it as data
    invents a shared label for every unindexed emblem, which would make thousands
    of them score as mutually relevant. An emblem holding only that template is
    unannotated, and must be reported as such rather than as labelled.
    """
    xml = _COMMENT.sub(" ", xml)
    notations: list[str] = []
    for block in _ICONCLASS.findall(xml):
        found = _NOTATION.findall(block)
        for raw in found or [block]:
            text = _plain(raw)
            if text:
                notations.append(text)

    texts: dict[str, str | None] = {}
    for name, pattern in _BLOCK.items():
        match = pattern.search(xml)
        if not match:
            texts[name] = None
            continue
        paragraphs = [_plain(p) for p in _PARAGRAPH.findall(match.group(1))]
        joined = " ".join(p for p in paragraphs if p)
        texts[name] = joined or None

    description = _FIGDESC.search(xml)
    return EmblemRecord(
        emblem_id=emblem_id,
        book_id=book_id,
        collection=collection,
        iconclass=tuple(dict.fromkeys(notations)),
        motto=texts["motto"],
        subscriptio=texts["subscriptio"],
        pictura_description=_plain(description.group(1)) if description else None,
        pictura_url=pictura_url,
    )


def emblem_book_triples(
    records: Iterable[EmblemRecord],
) -> tuple[tuple[str, str, str], ...]:
    """Emit ``part_of`` edges from emblems to their volume.

    Author, printer, place and date edges come from the book catalogue and are
    added by the bibliographic builder; this function supplies only the
    membership relation the emblem records themselves carry.
    """
    triples = {
        (f"emblematica:{record.emblem_id}", "part_of", f"book:{record.book_id}")
        for record in records
        if record.book_id
    }
    return tuple(sorted(triples))


def iconclass_coverage(records: Sequence[EmblemRecord]) -> dict[str, Any]:
    """Report how much of the corpus is usable as a labelled benchmark."""
    total = len(records)
    labelled = [r for r in records if r.iconclass]
    by_collection: dict[str, list[int]] = {}
    for record in records:
        bucket = by_collection.setdefault(record.collection or "unknown", [0, 0])
        bucket[1] += 1
        if record.iconclass:
            bucket[0] += 1

    assignments = sum(len(r.iconclass) for r in labelled)
    return {
        "n_emblems": total,
        "n_with_iconclass": len(labelled),
        "iconclass_coverage": len(labelled) / total if total else 0.0,
        "n_assignments": assignments,
        "mean_labels_per_labelled_emblem": assignments / len(labelled) if labelled else 0.0,
        "n_with_motto": sum(1 for r in records if r.motto),
        "n_with_subscriptio": sum(1 for r in records if r.subscriptio),
        "n_with_pictura_description": sum(1 for r in records if r.pictura_description),
        "n_books": len({r.book_id for r in records if r.book_id}),
        "by_collection": {
            name: {"labelled": counts[0], "total": counts[1]}
            for name, counts in sorted(by_collection.items())
        },
    }


_YEAR = re.compile(r"\b(1[0-9]{3})\b")


def book_metadata(row: dict[str, Any]) -> VolumeMetadata:
    """Adapt one Emblematica catalogue record to the shared volume shape.

    Reusing :class:`VolumeMetadata` means the bibliographic triple builder written
    for the Munich manifests applies unchanged here, so both corpora produce the
    same relation vocabulary and can be compared without a translation layer.

    The catalogue supplies no printer and no authority identifier, so creators are
    matched on slugged surnames. That is weaker than a GND match and will merge
    distinct people who share a surname; it is accepted because the alternative —
    no creator edge at all — removes the only relation capable of linking two
    emblem books by one author, which is what the hard-positive question needs.
    """
    authors = row.get("Authors") or []
    names = [name for name in (slug(str(a).split(",")[0]) for a in authors) if name]
    date = str(row.get("PublicationDate") or "")
    year = _YEAR.search(date)
    title = row.get("Title")
    return VolumeMetadata(
        volume_id=str(row.get("bookID")),
        title=title,
        # The catalogue has no normalised work title; the opening words of the
        # title are a deliberately crude stand-in, and only match when two
        # editions were catalogued with the same opening.
        work_title=" ".join(str(title).split()[:3]) if title else None,
        place=slug(row.get("PublicationPlace")),
        printer=None,
        year=int(year.group(1)) if year else None,
        language=None,
        creator_gnds=(),
        creator_names=tuple(dict.fromkeys(names)),
    )
