#!/usr/bin/env python3
"""Pick the pictures a scene-graph student is trained on, away from every bench.

From the Iconclass test set: 600 pictures of the sixteen subjects of the ``relations``
bench and 600 pictures of other narrative scenes (Old and New Testament, saints,
mythology and ancient history, relations between persons: notations starting with 71,
73, 11H, 92-95, 33), drawn at random (seed 0). Every picture looked at while building the
``relations`` bench (the 48 candidates per subject, so its 440 pictures too) is left
out. Near-copies of bench pictures are removed later, on the GPU, by DINOv2 similarity
(``scripts/remote_train_scene_lora.py``). Opus then writes each picture's scene graph
(``atelier.py scene entrainement --brique opus-scene``).

Run: ``uv run python scripts/build_scene_train_set.py``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_relations_bench import SOURCE, SUBJECTS, candidates

OUT = Path("data/derived/scene-train")
PER_HALF = 600
OTHER = ("71", "73", "11H", "92", "93", "94", "95", "33")


def main() -> None:
    codes = json.loads((SOURCE / "data.json").read_text())
    seen = {name for names in candidates().values() for name in names}
    same, other = [], []
    for name, notations in sorted(codes.items()):
        if name in seen or not (SOURCE / name).exists():
            continue
        if any(c.startswith(s) for c in notations for s in SUBJECTS):
            same.append(name)
        elif any(c.startswith(o) for c in notations for o in OTHER):
            other.append(name)
    rng = random.Random(0)
    chosen = rng.sample(same, PER_HALF) + rng.sample(other, PER_HALF)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in chosen:
        item = name.removesuffix(".jpg")
        link = OUT / "images" / f"{item}.jpg"
        if not link.exists():
            link.symlink_to((SOURCE / name).resolve())
        rows.append({"id": item, "label": "train", "iconclass": codes[name]})
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} pictures ({len(same)} and {len(other)} available)")


if __name__ == "__main__":
    main()
