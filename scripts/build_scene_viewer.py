#!/usr/bin/env python3
"""Build what the scene-graph page of the viewer reads (viewer/scenes.html).

For every picture of the relations bench and every scene reader whose readings are cached, it writes the reading (figures,
relations, the subject the reader named), the nearest other picture by the reader's graph words, whether that picture has
the same subject, and the rank of the first picture that does. Nothing is computed by the page.

Outputs, under `data/derived/viewer/scenes/`: `scenes.json` and `thumbs/<id>.jpg` (longest side 320 px).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from caypollard.atelier.benches import BENCHES

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/derived/atelier/relations"
OUT = ROOT / "data/derived/viewer/scenes"
READERS = {  # cache name -> label shown
    "opus-scene": "Opus",
    "gemini-flash-scene": "Gemini 3.8 Flash",
    "haiku-scene": "Haiku",
    "qwen25vl7b-lora-scene": "Qwen 7B + LoRA",
    "qwen25vl7b-bf16-scene": "Qwen 7B",
    "head-pe-spatial-scene": "Tête maison (PE-Spatial)",
}


def main() -> None:
    bench = BENCHES["relations"]
    items = bench.items()
    labels = [it["label"] for it in items]
    (OUT / "thumbs").mkdir(parents=True, exist_ok=True)
    for it in items:
        thumb = OUT / "thumbs" / f"{it['id']}.jpg"
        if not thumb.exists():
            im = ImageOps.exif_transpose(Image.open(bench.image(it["id"]))).convert("RGB")
            im.thumbnail((320, 320))
            im.save(thumb, quality=82)
    readers = {}
    for name, label in READERS.items():
        folder, matrix = CACHE / f"scene--{name}", CACHE / f"scene--{name}.npy"
        if not matrix.exists() or not folder.exists():
            continue
        dist = np.load(matrix)
        rows = []
        for i, it in enumerate(items):
            path = folder / f"{it['id']}.json"
            d = json.loads(path.read_text()) if path.exists() else {}
            order = [j for j in np.argsort(dist[i], kind="stable") if j != i]
            first_same = next(r for r, j in enumerate(order, 1) if labels[j] == labels[i])
            rows.append({
                "figures": [f for f in d.get("figures") or [] if isinstance(f, dict)],
                "relations": [r for r in d.get("relations") or [] if isinstance(r, dict)],
                "subject": str(d.get("subject") or ""),
                "seconds": d.get("seconds"),
                "nearest": [items[j]["id"] for j in order[:5]],
                "hit": labels[order[0]] == labels[i],
                "rank": first_same,
            })
        readers[name] = {"label": label, "rows": rows}
    out = {"bench": "relations", "items": [{"id": it["id"], "subject": it["label"]} for it in items], "readers": readers}
    (OUT / "scenes.json").write_text(json.dumps(out, ensure_ascii=False))
    print(len(items), "pictures,", ", ".join(readers))


if __name__ == "__main__":
    main()
