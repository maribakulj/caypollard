#!/usr/bin/env python3
"""Assemble the multi-channel record: shape, grammar, and a palette kept apart.

The channels are separate on purpose, and the reason is measured rather than
stylistic. Colour identifies the scanning session: within one collection, one
format and seventeen books it names the volume 61.3% of the time against a 31.0%
baseline, and quantising to eleven canonical terms with grey-world balancing only
brings that to 39.7%. So the palette belongs in the record -- it answers "what
tones has this object reached us in" -- and must never be matched on across
media. Keeping it in its own channel is what makes that distinction expressible
instead of a footnote.

Channels are concatenated with explicit weights after each is L2-normalised, so a
channel contributes in proportion to its weight and not to its dimensionality. A
weight of zero removes a channel from matching while leaving it in the record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import l2_normalize, load_embedding_table, save_embedding_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel",
        action="append",
        required=True,
        metavar="NAME=PATH:WEIGHT",
        help="A named channel table and its matching weight, repeatable",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    channels: dict[str, tuple[np.ndarray, dict[str, int], float]] = {}
    common: set[str] | None = None
    for spec in args.channel:
        name, _, rest = spec.partition("=")
        path, _, weight = rest.rpartition(":")
        table = load_embedding_table(path)
        row_of = {item: index for index, item in enumerate(table.ids)}
        channels[name] = (l2_normalize(table.vectors), row_of, float(weight))
        common = set(row_of) if common is None else common & set(row_of)

    assert common is not None
    ids = sorted(common)
    if not ids:
        raise SystemExit("the channels share no item")

    parts = []
    used = {}
    for name, (matrix, row_of, weight) in channels.items():
        used[name] = {"weight": weight, "dimensions": int(matrix.shape[1])}
        if weight == 0.0:
            continue
        parts.append(weight * np.stack([matrix[row_of[item]] for item in ids]))
    if not parts:
        raise SystemExit("every channel has weight zero; nothing to match on")

    combined = l2_normalize(np.concatenate(parts, axis=1).astype(np.float32))
    output = Path(args.output)
    save_embedding_table(
        output,
        ids=tuple(ids),
        vectors=combined,
        metadata={
            "family": "symbolic",
            "method": "weighted concatenation of normalised channels",
            "channels": used,
            "note": "a zero-weighted channel stays in the record and leaves the matching",
        },
        normalize=True,
    )
    print(json.dumps({"items": len(ids), "channels": used, "output": str(output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
