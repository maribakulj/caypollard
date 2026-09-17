#!/usr/bin/env python3
"""Encode the emblem corpus's own words into the T modality.

The model matrix declares V, G, T and their combinations; three of those
conditions had no vectors because nothing encoded text. Emblematica transcribes
a motto for 97.7% of its emblems, so that is the field encoded here by default,
with the option to concatenate the pictura description where one exists.

Items whose chosen text field is empty are skipped and counted rather than
encoded as an empty string, which would give every one of them the same vector
and make them mutually nearest neighbours.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from caypollard.text.encoders import DEFAULT_MODELS, HuggingFaceTextEncoder, resolve_text_spec

FIELDS = ("motto", "subscriptio", "pictura_description")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preset", choices=sorted(DEFAULT_MODELS), default="labse")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument(
        "--fields",
        default="motto",
        help="Comma-separated manifest text fields, joined in order when several are given",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    unknown = [field for field in fields if field not in FIELDS]
    if unknown:
        raise SystemExit(f"unknown text field(s): {', '.join(unknown)}; expected {FIELDS}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    records = load_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]

    texts: list[str] = []
    ids: list[str] = []
    empty = 0
    for record in records:
        parts = [str(record.get(field)).strip() for field in fields if record.get(field)]
        if not parts:
            empty += 1
            continue
        ids.append(str(record["id"]))
        texts.append(" ".join(parts))
    if not ids:
        raise SystemExit("no record carries text in the requested fields")

    spec = resolve_text_spec(None if args.model_id else args.preset, args.model_id)
    if args.revision:
        spec = type(spec)(spec.family, spec.model_id, args.revision, spec.query_prefix,
                          spec.max_length)
    encoder = HuggingFaceTextEncoder(spec, device=args.device)

    import numpy as np

    started = time.monotonic()
    chunks = []
    for start in range(0, len(texts), args.batch_size):
        chunks.append(encoder.encode(texts[start : start + args.batch_size]))
        done = min(start + args.batch_size, len(texts))
        if args.progress_every and done % args.progress_every < args.batch_size:
            rate = done / max(time.monotonic() - started, 1e-9)
            print(f"{done}/{len(texts)} — {rate:.0f}/s", file=sys.stderr, flush=True)

    vectors = np.concatenate(chunks, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    with staging.open("wb") as handle:
        np.savez(handle, ids=np.array(ids, dtype=np.str_), vectors=vectors)
    staging.replace(args.output)

    metadata = {
        **encoder.metadata,
        "embedding_file": args.output.name,
        "manifest": str(args.manifest),
        "fields": fields,
        "n_items": len(ids),
        "n_records": len(records),
        "n_skipped_without_text": empty,
        "dimension": int(vectors.shape[1]),
        "l2_normalized": True,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    Path(str(args.output) + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{len(ids)} textes encodés, {empty} sans texte — {args.output}")


if __name__ == "__main__":
    main()
