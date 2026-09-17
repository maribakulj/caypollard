"""Parse BSB IIIF manifests into normalised bibliographic entities.

``G1`` links plates only within a volume, which is why it cannot reach the hard
positives the project cares about: those pairs are almost never two plates of one
book. Cross-volume edges need bibliographic identity — the same author, printer,
place, period, or work behind two different volumes — and the Munich manifests
supply exactly that for the ``bsb`` portion of the corpus.

Normalisation is the whole difficulty. Early modern imprints spell places in
Latin, in the ablative, and in brackets when the cataloguer inferred them, so
matching on raw strings both over- and under-merges. Two defences are used here:
GND authority URIs are preferred over names wherever the manifest supplies one,
so ``Erhard Ratdolt`` and ``Erhardi ratdolt Augustensis`` collapse to one node by
identifier rather than by spelling; and everything else is slugged conservatively,
accepting that some genuine matches are missed rather than inventing links that
are not there.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

_GND = re.compile(r"gnd/(\w+)")
_YEAR = re.compile(r"\b(1[0-9]{3})\b")
_TAGS = re.compile(r"<[^>]+>")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


def _text(value: Any, *, language: str = "en") -> str | None:
    """Flatten a IIIF value, which may be a string, a dict, or a language list."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = _TAGS.sub(" ", value).strip()
        return cleaned or None
    if isinstance(value, dict):
        return _text(value.get("@value"), language=language)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("@language") == language:
                return _text(item.get("@value"), language=language)
        return _text(value[0], language=language) if value else None
    return None


def slug(value: str | None, *, max_length: int = 60) -> str | None:
    """Return a conservative ASCII slug, or ``None`` when nothing survives."""
    if not value:
        return None
    # Brackets are removed rather than replaced by a space: in these imprints they
    # mark an expansion *inside* a word, so "vindelicor[um]" is one word, not two,
    # and a conjectural "[Heidelberg]" is still simply Heidelberg.
    stripped = value.replace("[", "").replace("]", "")
    folded = unicodedata.normalize("NFKD", stripped)
    ascii_text = "".join(ch for ch in folded if not unicodedata.combining(ch)).lower()
    slugged = _NON_SLUG.sub("-", ascii_text).strip("-")
    return slugged[:max_length] or None


@dataclass(frozen=True)
class VolumeMetadata:
    """Bibliographic facts recovered for one digitised volume."""

    volume_id: str
    title: str | None = None
    work_title: str | None = None
    place: str | None = None
    printer: str | None = None
    year: int | None = None
    language: str | None = None
    creator_gnds: tuple[str, ...] = field(default_factory=tuple)
    creator_names: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "volume_id": self.volume_id,
            "title": self.title,
            "work_title": self.work_title,
            "place": self.place,
            "printer": self.printer,
            "year": self.year,
            "language": self.language,
            "creator_gnds": list(self.creator_gnds),
            "creator_names": list(self.creator_names),
        }


def _fields(payload: dict[str, Any]) -> dict[str, list[str]]:
    """Collect metadata values with markup intact.

    Tags are left in place here because the GND authority identifier lives inside
    an anchor href; stripping markup at collection time would silently discard the
    only reliable way to tell two same-named people apart. Callers that want
    display text strip it themselves.
    """
    collected: dict[str, list[str]] = {}
    for entry in payload.get("metadata") or []:
        label = _text(entry.get("label"))
        value = _raw_text(entry.get("value"))
        if label and value:
            collected.setdefault(label, []).append(value)
    return collected


def _raw_text(value: Any, *, language: str = "en") -> str | None:
    """Flatten a IIIF value without removing markup."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        return _raw_text(value.get("@value"), language=language)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("@language") == language:
                return _raw_text(item.get("@value"), language=language)
        return _raw_text(value[0], language=language) if value else None
    return None


def _split_imprint(published: str | None) -> tuple[str | None, str | None]:
    """Split a ``place : printer`` imprint statement.

    Only the first colon separates the two: printer statements routinely contain
    further colons ("... eximia industria: [et] ..."), and splitting on all of
    them would truncate the printer.
    """
    if not published:
        return None, None
    if ":" not in published:
        return published.strip() or None, None
    place, printer = published.split(":", 1)
    return place.strip() or None, printer.strip() or None


def parse_manifest(payload: dict[str, Any], *, volume_id: str) -> VolumeMetadata:
    """Extract normalised bibliographic fields from one IIIF manifest."""
    fields = _fields(payload)

    def first(*labels: str) -> str | None:
        for label in labels:
            values = fields.get(label)
            if values:
                cleaned = _TAGS.sub(" ", values[0]).strip()
                return re.sub(r"\s+", " ", cleaned) or None
        return None

    place, printer = _split_imprint(first("Published"))

    raw_creators = fields.get("Creator", []) + fields.get("Contributor", [])
    gnds: list[str] = []
    names: list[str] = []
    for raw in raw_creators:
        gnds.extend(_GND.findall(raw))
        # The name precedes the authority note, which the manifest separates by "--".
        name = slug(raw.split("--")[0].split(",")[0])
        if name:
            names.append(name)

    date_text = first("Date") or ""
    year_match = _YEAR.search(date_text)

    return VolumeMetadata(
        volume_id=volume_id,
        title=first("Title"),
        work_title=first("Work title"),
        place=slug(place),
        printer=slug(printer),
        year=int(year_match.group(1)) if year_match else None,
        language=slug(first("Language")),
        creator_gnds=tuple(dict.fromkeys(gnds)),
        creator_names=tuple(dict.fromkeys(names)),
    )


def metadata_triples(
    volumes: Iterable[VolumeMetadata], *, decade_nodes: bool = True
) -> tuple[tuple[str, str, str], ...]:
    """Emit cross-volume bibliographic triples.

    Dates become decade nodes rather than one entity per year: a decade is a
    historically usable grouping, whereas a thousand singleton year nodes would
    add entities without adding structure. ``decade_nodes=False`` drops dates
    entirely for an ablation that isolates the person and place relations.
    """
    triples: list[tuple[str, str, str]] = []
    for volume in volumes:
        head = f"book:{volume.volume_id}"
        # An authority identifier is preferred over a name; falling back to names
        # for every volume would merge distinct people who share a spelling.
        for gnd in volume.creator_gnds:
            triples.append((head, "created_by", f"person:gnd-{gnd}"))
        if not volume.creator_gnds:
            for name in volume.creator_names:
                triples.append((head, "created_by", f"person:name-{name}"))
        if volume.printer:
            triples.append((head, "printed_by", f"printer:{volume.printer}"))
        if volume.place:
            triples.append((head, "published_at", f"place:{volume.place}"))
        if volume.language:
            triples.append((head, "in_language", f"language:{volume.language}"))
        work = slug(volume.work_title)
        if work:
            triples.append((head, "instance_of", f"work:{work}"))
        if decade_nodes and volume.year is not None:
            triples.append((head, "published_in", f"decade:{volume.year // 10 * 10}s"))
    return tuple(sorted(set(triples)))


def metadata_report(volumes: Sequence[VolumeMetadata]) -> dict[str, Any]:
    """Summarise field coverage, so thin fields are visible before modelling."""
    total = len(volumes)

    def filled(attribute: str) -> int:
        return sum(1 for volume in volumes if getattr(volume, attribute))

    return {
        "n_volumes": total,
        "coverage": {
            name: (filled(name) / total if total else 0.0)
            for name in ("creator_gnds", "creator_names", "printer", "place", "year",
                         "language", "work_title")
        },
        "n_distinct_creators_gnd": len({g for v in volumes for g in v.creator_gnds}),
        "n_distinct_printers": len({v.printer for v in volumes if v.printer}),
        "n_distinct_places": len({v.place for v in volumes if v.place}),
        "n_distinct_works": len({slug(v.work_title) for v in volumes if slug(v.work_title)}),
        "year_range": [
            min((v.year for v in volumes if v.year), default=None),
            max((v.year for v in volumes if v.year), default=None),
        ],
    }
