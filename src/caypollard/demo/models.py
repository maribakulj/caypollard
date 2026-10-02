"""The fitted models a new picture needs, as `scripts/fit_demo_models.py` saved them."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np

from .bridge import ROOT

DEMO = ROOT / "data/derived/demo"


class Atoms:
    """A shape vocabulary: standardisation and cluster centres, assignment by nearest centre."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as z:
            self.centre = z["centre"].astype(np.float32)
            self.scale = z["scale"].astype(np.float32)
            self.centres = z["centres"].astype(np.float32)
        self.size = int(self.centres.shape[0])
        self.name = path.stem

    def assign(self, described: np.ndarray) -> np.ndarray:
        z = (described - self.centre) / self.scale
        d = ((z[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    def distances(self, described: np.ndarray) -> np.ndarray:
        z = (described - self.centre) / self.scale
        return np.sqrt(((z[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=2))


@cache
def atoms(name: str) -> Atoms:
    path = DEMO / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `uv run python scripts/fit_demo_models.py`"
        )
    return Atoms(path)


@cache
def composites() -> np.ndarray:
    path = DEMO / "composites-v3.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `uv run python scripts/fit_demo_models.py`"
        )
    with np.load(path) as z:
        return z["centres"].astype(np.float32)


def l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


@cache
def relation_keys() -> tuple[list[tuple[int, str, int]], dict[tuple[int, str, int], int]]:
    """The (sign, relation, sign) attributes relations-wide64 keeps, and their column."""
    path = DEMO / "relations-64.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: run `uv run python scripts/fit_demo_models.py`"
        )
    with np.load(path) as z:
        keys = [
            (int(a), str(r), int(b))
            for a, r, b in zip(z["sign_a"], z["relation"], z["sign_b"], strict=True)
        ]
    return keys, {k: n for n, k in enumerate(keys)}


@cache
def stroke_atoms() -> Atoms:
    """The stroke vocabulary `scripts/build_sketch_channel.py` fitted, with its centres saved."""
    path = ROOT / "data/derived/sketch/stroke-atoms.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run `make sketch-build`")
    return Atoms(path)
