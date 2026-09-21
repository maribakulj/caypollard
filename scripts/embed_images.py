#!/usr/bin/env python3
"""Extract reproducible visual embeddings for a manifest of local images."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from caypollard.embeddings.store import save_embedding_table
from caypollard.provenance import sha256_file
from caypollard.vision.encoders import (
    DEFAULT_MODELS,
    HuggingFaceVisionEncoder,
    VisionModelSpec,
)


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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10000,
        help="Persist partial vectors every N images; 0 disables. A multi-hour run "
             "that only writes at the end loses everything to a single crash.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a checkpoint left by an interrupted run and encode only what is missing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Report throughput to stderr every N images; 0 disables. A multi-hour "
             "extraction that prints nothing is indistinguishable from a stalled one.",
    )
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

    import numpy as np

    checkpoint_path = args.output.with_suffix(args.output.suffix + ".partial")

    ids: list[str] = []
    vectors = []
    missing: list[str] = []
    encoded: set[str] = set()

    if args.resume and checkpoint_path.is_file():
        saved = np.load(checkpoint_path, allow_pickle=False)
        ids = [str(value) for value in saved["ids"]]
        vectors = [saved["vectors"]]
        encoded = set(ids)
        print(f"reprise : {len(ids)} vecteurs déjà calculés", file=sys.stderr, flush=True)

    def write_checkpoint() -> None:
        if not vectors:
            return
        # Write beside the target then rename, so an interruption mid-write cannot
        # leave a truncated checkpoint in place of a good one. The handle is opened
        # explicitly because `np.savez` appends `.npz` to a path that lacks it,
        # which would make the rename target a file that does not exist.
        staging = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        with staging.open("wb") as handle:
            np.savez(
                handle,
                # A plain unicode array, not dtype=object: object arrays can only be
                # read back with pickle enabled, which the loader refuses on purpose.
                ids=np.array(ids, dtype=np.str_),
                vectors=np.concatenate(vectors, axis=0),
            )
        staging.replace(checkpoint_path)

    started = time.monotonic()
    last_report = 0
    last_checkpoint = 0
    last_release = 0
    for start in range(0, len(records), args.batch_size):
        rows = records[start : start + args.batch_size]
        images = []
        batch_ids = []
        for record in rows:
            filename = record.get("filename")
            item_id = record.get("id")
            if not filename or not item_id:
                raise ValueError("manifest rows require id and filename fields")
            if str(item_id) in encoded:
                continue
            # A merged manifest spans several source directories and records
            # where each image actually is; a single-corpus one does not, and
            # falls back to the directory given on the command line.
            recorded = record.get('image_path')
            image_path = Path(recorded) if recorded else args.image_dir / str(filename)
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

        done = start + len(rows)
        # The MPS allocator keeps freed blocks cached, and over thousands of
        # images that cache is what the system runs out of, not the model. A
        # run of this corpus was killed at 2 000 images without it.
        if done - last_release >= 400:
            last_release = done
            encoder.release_cache()

        if args.progress_every and done - last_report >= args.progress_every:
            last_report = done
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (len(records) - done) / rate if rate else float("nan")
            print(
                f"{done}/{len(records)} images  {rate:.1f}/s  "
                f"reste ~{remaining / 60:.0f} min",
                file=sys.stderr,
                flush=True,
            )

        if args.checkpoint_every and done - last_checkpoint >= args.checkpoint_every:
            last_checkpoint = done
            write_checkpoint()

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {preview}")
    if not vectors:
        raise ValueError("no images were encoded")

    matrix = np.concatenate(vectors, axis=0)
    metadata = encoder.metadata | {
        "created_at": datetime.now(UTC).isoformat(),
        "manifest_file": args.manifest.name,
        "manifest_sha256": sha256_file(args.manifest),
        "image_dir": str(args.image_dir),
        "normalization": "L2",
        "pooling": "CLS token" if spec.family == "dinov2" else "model.get_image_features",
    }
    save_embedding_table(args.output, ids=ids, vectors=matrix, metadata=metadata, normalize=True)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(metadata | {"n_items": len(ids), "dimension": matrix.shape[1]}, indent=2))


if __name__ == "__main__":
    main()
