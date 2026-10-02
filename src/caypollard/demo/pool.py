"""The frozen pool, one table per representation, and exact cosine search over it.

Search is deliberately the simplest thing that is correct: every vector is L2-normalised,
the score is the dot product, the whole pool is scanned and the k largest scores are kept.
No index, no approximation: with at most twenty-one thousand rows the scan takes
milliseconds, and an approximate index would add a source of error to a demonstration
whose point is to be read.

Mixing representations is the weighted concatenation `scripts/build_visual_record.py`
uses: each channel's vector is normalised, multiplied by its weight, concatenated, and
the result normalised again; the pool is built the same way over the pictures present in
every chosen table. Dropping a weight to zero removes the channel exactly.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from caypollard.embeddings.store import load_embedding_table

from .bridge import ROOT
from .models import l2

SHAPES = ROOT / "data/derived/shapes"
SKETCH = ROOT / "data/derived/sketch"


@dataclass(frozen=True)
class Channel:
    key: str
    name: str
    regime: str  # image | semantique | surface
    table: Path
    dimension: int
    what: str


CHANNELS: dict[str, Channel] = {
    c.key: c
    for c in [
        Channel(
            "pixels",
            "pixels (DINOv2)",
            "image",
            SHAPES / "pixels-mixte.npz",
            768,
            "l'image telle quelle, lue par un encodeur photographique",
        ),
        Channel(
            "croquis",
            "croquis (DINOv2 sur le dessin au trait)",
            "trait",
            SKETCH / "dinov2-sketch.npz",
            768,
            "le même encodeur, sur les contours à deux couleurs au lieu de l'image",
        ),
        Channel(
            "traits",
            "traits (128 signes de trait, pondérés par la longueur)",
            "trait",
            SKETCH / "stroke-signs.npz",
            128,
            "les 40 traits les plus longs du croquis : taille, direction, courbure, place",
        ),
        Channel(
            "silhouette",
            "silhouette (DINOv2 sur le rendu)",
            "image",
            ROOT / "data/derived/museums-v0.2/embeddings/dinov2-silhouette.npz",
            768,
            "le même encodeur, sur une silhouette noir et blanc de l'image",
        ),
        Channel(
            "formes",
            "formes muettes (256 signes)",
            "image",
            SHAPES / "v2-256/shape-signs.npz",
            256,
            "histogramme des signes de forme, pondéré par l'aire",
        ),
        Channel(
            "composition",
            "composition",
            "image",
            SHAPES / "composition-v3.npz",
            87,
            "où est la masse d'encre : grille 8×8, symétries, profils, nuage des centroïdes",  # noqa: RUF001
        ),
        Channel(
            "repetition",
            "répétition",
            "image",
            SHAPES / "repetition-v3.npz",
            1028,
            "combien de fois chaque signe revient",
        ),
        Channel(
            "record",
            "record (composites + composition ×2 + répétition ×½)",  # noqa: RUF001
            "image",
            SHAPES / "record-v3.npz",
            1243,
            "la concaténation pondérée des canaux symboliques",
        ),
        Channel(
            "relations",
            "relations spatiales (64 signes, 9 518 triplets)",
            "image",
            SHAPES / "relations-wide64.npz",
            9518,
            "quel signe est au-dessus, à côté, dans quel autre : la grammaire, comptée",
        ),
        Channel(
            "pose",
            "pose (angles des membres) — 1 903 images seulement",
            "image",
            SHAPES / "pose-partial.npz",
            57,
            "angles des membres par rapport au torse, moyenne et dispersion des figures",
        ),
        Channel(
            "palette",
            "palette (11 couleurs)",
            "surface",
            SHAPES / "palette-mixte.npz",
            11,
            "fractions de onze termes de couleur, après équilibrage des blancs",
        ),
        Channel(
            "signal",
            "signal (16 mesures)",
            "surface",
            SHAPES / "signal-mixte.npz",
            16,
            "histogramme de luminance, contraste, rugosité : le grain de la numérisation",
        ),
        Channel(
            "nommes",
            "nœuds nommés (116 mots)",
            "semantique",
            SHAPES / "nodes-mixte.npz",
            116,
            "les trois premiers objets que le nommeur voit, comptés",
        ),
        Channel(
            "mix",
            "tout mélangé (nommés + pixels + palette + signal + formes v4)",
            "mixte",
            SHAPES / "mix-tout.npz",
            1167,
            "les cinq canaux concaténés à poids égaux",
        ),
    ]
}


class Pool:
    def __init__(self, items_path: Path | None = None) -> None:
        self.items_path = items_path or ROOT / "data/derived/viewer/items.json"
        self._tables: dict[str, tuple[list[str], np.ndarray]] = {}
        self._mixes: dict[str, tuple[list[str], np.ndarray]] = {}
        self._index: dict[str, int] | None = None
        self._lock = threading.Lock()

    # ---- items of the viewer, so results can be shown as thumbnails ----
    def index(self) -> dict[str, int]:
        if self._index is None:
            if not self.items_path.exists():
                raise FileNotFoundError(f"{self.items_path} is missing: run `make viewer-build`")
            data = json.loads(self.items_path.read_text(encoding="utf-8"))
            self._index = {it["id"]: i for i, it in enumerate(data["items"])}
        return self._index

    # ---- tables ----
    def table(self, key: str) -> tuple[list[str], np.ndarray]:
        """Ids and L2-normalised rows of one channel, restricted to pictures the viewer shows."""
        with self._lock:
            if key not in self._tables:
                channel = CHANNELS[key]
                loaded = load_embedding_table(channel.table)
                index = self.index()
                keep = [i for i, item in enumerate(loaded.ids) if item in index]
                rows = loaded.vectors[keep]
                norms = np.linalg.norm(rows, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._tables[key] = (
                    [loaded.ids[i] for i in keep],
                    (rows / norms).astype(np.float32),
                )
            return self._tables[key]

    def frozen_vector(self, key: str, item_id: str) -> np.ndarray | None:
        ids, rows = self.table(key)
        try:
            return rows[ids.index(item_id)]
        except ValueError:
            return None

    def mixed(self, weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
        """The pool concatenated over the channels with a non-zero weight, on their common ids."""
        active = {k: float(w) for k, w in weights.items() if w and k in CHANNELS}
        signature = json.dumps(sorted(active.items()))
        with self._lock:
            cached = self._mixes.get(signature)
        if cached is not None:
            return cached
        tables = {k: self.table(k) for k in active}
        common = sorted(set.intersection(*(set(ids) for ids, _ in tables.values())))
        blocks = []
        for k, (ids, rows) in tables.items():
            position = {item: i for i, item in enumerate(ids)}
            blocks.append(rows[[position[item] for item in common]] * active[k])
        matrix = np.concatenate(blocks, axis=1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        result = (common, (matrix / norms).astype(np.float32))
        with self._lock:
            if len(self._mixes) > 8:
                self._mixes.clear()
            self._mixes[signature] = result
        return result

    # ---- search ----
    def search(
        self,
        vectors: dict[str, np.ndarray],
        weights: dict[str, float],
        k: int = 50,
        exclude: str | None = None,
    ) -> dict:
        """Top-k of the pool for a query described by per-channel vectors and channel weights."""
        active = {key: float(w) for key, w in weights.items() if w and key in vectors}
        if not active:
            return {"results": [], "pool": 0, "channels": {}}
        if len(active) == 1:
            key = next(iter(active))
            ids, rows = self.table(key)
            query = l2(vectors[key])
        else:
            ids, rows = self.mixed(active)
            query = l2(np.concatenate([l2(vectors[key]) * active[key] for key in active]))
        scores = rows @ query
        order = np.argsort(-scores)
        index = self.index()
        results = []
        for i in order:
            item = ids[i]
            if item == exclude:
                continue
            results.append({"i": index[item], "id": item, "score": float(scores[i])})
            if len(results) >= k:
                break
        return {"results": results, "pool": len(ids), "channels": active}
