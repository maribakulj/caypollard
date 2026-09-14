"""Small provenance and deterministic-manifest utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_jsonl(records: Iterable[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: str(record.get("id", "")))
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in ordered
    )


def manifest_digest(records: Iterable[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl(records).encode("utf-8")).hexdigest()


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> str:
    content = canonical_jsonl(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
