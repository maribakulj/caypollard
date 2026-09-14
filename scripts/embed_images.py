#!/usr/bin/env python3
"""Extract reproducible visual embeddings for a manifest of local images."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caypollard.provenance import sha256_file
from caypollard.vision.encoders import (
    DEFAULT_MODELS,
    HuggingFaceVisionEncoder,
    VisionModelSpec,
)
from caypollard.vision.store import save_embedding_table


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Line {line_number} is not a JSON object")
                records.append(value)
    return records


def resolve_spec(args: argparse.Namespace) -> VisionModelSpec:
    if args.preset:
        base = DEFAULT_MODELS[args.preset]
        return VisionModelSpec(base.family, base.model_id, args.revision or base.revision)
    if not args.family or not args.model_id:
        raise ValueError("Use --preset or provide both --family and --model-id")
    return VisionModelSpec(args.family, args.model_id, args.revision)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preset", choices=sorted(DEFAULT_MODELS))
    parser.add_argument("--family", choices=["dinov2", "clip", "siglip"])
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the vision dependencies with `uv sync --extra vision`") from exc

    records = load_jsonl(args.manifest)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("manifest contains no records")

    spec = resolve_spec(args)
    encoder = HuggingFaceVisionEncoder(spec, device=args.device)

    ids: list[str] = []
    vectors = []
    missing: list[str] = []
    for start in range(0, len(records), args.batch_size):
        rows = records[start : start + args.batch_size]
        images = []
        batch_ids = []
        for record in rows:
            filename = record.get("filename")
            item_id = record.get("id")
            if not filename or not item_id:
                raise ValueError("manifest rows require id and filename fields")
            image_path = args.image_dir / str(filename)
            if not image_path.is_file():
                missing.append(str(filename))
                continue
            with Image.open(image_path) as image:
                images.append(image.convert("RGB").copy())
            batch_ids.append(str(item_id))
        if images:
            batch_vectors = encoder.encode(images, normalize=True)
            ids.extend(batch_ids)
            vectors.append(batch_vectors)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {preview}")
    if not vectors:
        raise ValueError("no images were encoded")

    import numpy as np

    matrix = np.concatenate(vectors, axis=0)
    metadata = encoder.metadata | {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_file": args.manifest.name,
        "manifest_sha256": sha256_file(args.manifest),
        "image_dir": str(args.image_dir),
        "normalization": "L2",
        "pooling": "CLS token" if spec.family == "dinov2" else "model.get_image_features",
    }
    save_embedding_table(args.output, ids=ids, vectors=matrix, metadata=metadata, normalize=True)
    print(json.dumps(metadata | {"n_items": len(ids), "dimension": matrix.shape[1]}, indent=2))


if __name__ == "__main__":
    main()
