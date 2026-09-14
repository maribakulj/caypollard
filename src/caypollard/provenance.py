"""Small provenance and deterministic-manifest utilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list of object records."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> str:
    content = canonical_jsonl(records)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
