"""Distances between pictures: by their figures' poses, by whole-picture vectors.

The pose distance is the one fixed for the copies test before any result was seen:
each figure's skeleton normalised to its hips, torso up, unit torso; two figures differ
by the mean distance between their joints (a joint in one only counts as a full miss)
plus half the gap between where they stand in their frames; two pictures by the best
matching of their figures, an unmatched figure costing a full miss. A picture is also
read mirrored (x reversed, left and right swapped) and the closer reading is kept,
since copies are often printed the other way round.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from caypollard.figures import KEYPOINTS, mid, normalised_skeleton, relations, torso_axis

MISS = 1.0  # a joint, or a figure, that has no counterpart
LAYOUT = 0.5  # weight of where a figure stands in its frame, against its pose


def mirrored_keypoints(k: dict, width: float) -> dict:
    """The skeleton seen in a mirror: x reversed, the figure's left and right swapped."""
    out = {}
    for name, p in k.items():
        other = (
            name.replace("left_", "\0").replace("right_", "left_").replace("\0", "right_")
            if name.startswith(("left_", "right_"))
            else name
        )
        out[other] = None if not p else [width - p[0], p[1]]
    return out


def figures_of(description: dict, mirror: bool, joints: set | None = None) -> list[dict]:
    """Each figure as a normalised skeleton plus where it stands in the frame."""
    w, h = description["size"]
    out = []
    for f in description.get("figures", []):
        k = {
            n: p
            for n, p in (f.get("keypoints") or {}).items()
            if n in KEYPOINTS and p and (joints is None or n in joints)
        }
        box = f.get("box") or [0, 0, w, h]
        if mirror:
            k = mirrored_keypoints(k, w)
            box = [w - box[2], box[1], w - box[0], box[3]]
        out.append(
            {
                "skeleton": normalised_skeleton(k),
                "centre": ((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h),
                "height": (box[3] - box[1]) / h,
            }
        )
    return out


def figure_distance(a: dict, b: dict, layout: float = LAYOUT) -> float:
    sa, sb = a["skeleton"], b["skeleton"]
    names = set(sa) | set(sb)
    if not names:
        return MISS
    gaps = [min(math.dist(sa[n], sb[n]), 1.5) if n in sa and n in sb else MISS for n in names]
    pose = sum(gaps) / len(gaps)
    where = math.dist(a["centre"], b["centre"]) + abs(a["height"] - b["height"])
    return pose + layout * where


def scene_distance(fa: list[dict], fb: list[dict], layout: float = LAYOUT) -> float:
    from scipy.optimize import linear_sum_assignment

    if not fa or not fb:
        return MISS * 2
    cost = np.array([[figure_distance(a, b, layout) for b in fb] for a in fa])
    rows, cols = linear_sum_assignment(cost)
    unmatched = len(fa) + len(fb) - 2 * len(rows)
    return float(cost[rows, cols].sum() + MISS * unmatched) / max(len(fa), len(fb))


def pose_matrix(
    descriptions: Sequence[dict], layout: float = LAYOUT, joints: set | None = None
) -> np.ndarray:
    """Pairwise distances between pictures from their figures; the mirror is tried."""
    plain = [figures_of(d, False, joints) for d in descriptions]
    flipped = [figures_of(d, True, joints) for d in descriptions]
    n = len(descriptions)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = min(
                scene_distance(plain[i], plain[j], layout),
                scene_distance(flipped[i], plain[j], layout),
            )
    return out


def vector_matrix(plain: np.ndarray, flipped: np.ndarray | None) -> np.ndarray:
    """Distances from L2-normalised picture vectors; with the mirror, the closer reading."""
    similarity = plain @ plain.T
    if flipped is not None:
        similarity = np.maximum(similarity, flipped @ plain.T)
    return -similarity  # lower is closer


def dense_matrix(
    plain: np.ndarray, flipped: np.ndarray | None, symmetric: bool, block: int = 4
) -> np.ndarray:
    """Distances from patch features (pictures x patches x dims, L2-normalised).

    Each patch of the query finds its most similar patch in the other picture; the
    similarity of two pictures is the mean of these best matches. ``symmetric`` averages
    both directions (two pictures with the same content); otherwise only the query's
    patches are matched (a detail taken from a larger picture still finds it). With the
    mirror, the query is also read flipped and the closer reading is kept.
    """
    import torch

    from .bricks import device

    dev = device()
    gallery = torch.from_numpy(plain).to(dev, torch.float16)
    n = len(plain)
    out = np.empty((n, n), dtype=np.float32)
    readings = [plain] + ([flipped] if flipped is not None else [])
    for start in range(0, n, block):
        best = None
        for reading in readings:
            q = torch.from_numpy(reading[start : start + block]).to(dev, torch.float16)
            # (queries, gallery, query patches, gallery patches)
            sim = torch.einsum("apd,bqd->abpq", q, gallery)
            score = sim.amax(3).float().mean(2)
            if symmetric:
                score = (score + sim.amax(2).float().mean(2)) / 2
            best = score if best is None else torch.maximum(best, score)
        out[start : start + block] = best.cpu().numpy()
    return -out  # lower is closer


def posture(k: dict) -> str:
    """A figure's posture from its points: lying, leaning, kneeling or seated, standing.

    The torso (hips to shoulders) against the vertical: beyond 60 degrees the figure lies,
    beyond 30 it leans. An upright torso whose knees come up to less than half a torso below
    the hips kneels or sits; with lower knees it stands. Without hips and shoulders: unknown.
    """
    axis = torso_axis(k)
    if not axis:
        return "inconnue"
    (hx, hy), (sx, sy) = axis
    torso = math.hypot(sx - hx, sy - hy)
    if torso == 0:
        return "inconnue"
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, (hy - sy) / torso))))
    if tilt > 60:
        return "couchée"
    if tilt > 30:
        return "penchée"
    knees = mid(k.get("left_knee"), k.get("right_knee"))
    if knees and knees[1] - hy < 0.5 * torso:
        return "agenouillée-assise"
    return "debout"


def relation_words(description: dict, relations_too: bool = True) -> dict[str, int]:
    """One picture as a bag of words: how many figures, each figure's posture, and (with
    ``relations_too``) each geometric relation between two figures with both postures.

    The relations are the pilot's (``caypollard.figures.relations``) read on boxes and
    points only, without silhouettes or objects: touches, reaches toward, looks toward,
    turns its back to, above, taller than. Who is who is not known, only the postures.
    """
    figures = [
        {"id": str(i), "box": f["box"], "keypoints": f.get("keypoints") or {}}
        for i, f in enumerate(description.get("figures", []))
        if f.get("box")
    ]
    n = len(figures)
    words: dict[str, int] = {}

    def add(word: str) -> None:
        words[word] = words.get(word, 0) + 1

    add(
        "figures:"
        + ("0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-4" if n < 5 else "5+")
    )
    kinds = {f["id"]: posture(f["keypoints"]) for f in figures}
    for kind in kinds.values():
        add("posture:" + kind)
    if relations_too:
        for r in relations(figures, []):
            add(f"{r['rel']}:{kinds[r['a']]}>{kinds[r['b']]}")
    return words


def scene_words(scene: dict, relations_too: bool = True) -> dict[str, int]:
    """A scene graph read by a vision-language model as a bag of words: how many figures,
    each figure's role and posture, and (with ``relations_too``) each relation with the
    roles at both ends. The subject the model names is not used."""
    figures = {f.get("id"): f for f in scene.get("figures", []) if isinstance(f, dict)}
    n = len(figures)
    words: dict[str, int] = {}

    def add(word: str) -> None:
        words[word] = words.get(word, 0) + 1

    add(
        "figures:"
        + ("0" if n == 0 else "1" if n == 1 else "2" if n == 2 else "3-4" if n < 5 else "5+")
    )
    for f in figures.values():
        add(f"figure:{f.get('role')}:{f.get('posture')}")
    if relations_too:
        for r in scene.get("relations", []):
            a, b = figures.get(r.get("a")), figures.get(r.get("b"))
            if a and b:
                add(f"{a.get('role')}:{r.get('rel')}:{b.get('role')}")
    return words


def scene_phrases(scene: dict) -> list[str]:
    """A scene graph as short phrases, one per figure and one per relation, so that a text
    encoder can bring synonyms together ("holds" and "carries")."""
    figures = {f.get("id"): f for f in scene.get("figures", []) if isinstance(f, dict)}
    out = [f"{f.get('posture')} {f.get('role')}" for f in figures.values()]
    for r in scene.get("relations", []):
        a, b = figures.get(r.get("a")), figures.get(r.get("b"))
        if a and b:
            out.append(f"{a.get('role')} {r.get('rel')} {b.get('role')}")
    return out


def set_matrix(sets: Sequence[np.ndarray]) -> np.ndarray:
    """Distances between sets of unit vectors: each vector takes its best match in the
    other set, mean of both directions (an empty set is far from everything)."""
    n = len(sets)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = sets[i], sets[j]
            if len(a) and len(b):
                m = a @ b.T
                sim[i, j] = sim[j, i] = (m.max(1).mean() + m.max(0).mean()) / 2
    return -sim  # lower is closer


def words_matrix(bags: Sequence[dict[str, int]]) -> np.ndarray:
    """Distances between bags of words: cosine over counts (a picture without words is far
    from everything)."""
    vocabulary = sorted({w for bag in bags for w in bag})
    index = {w: i for i, w in enumerate(vocabulary)}
    counts = np.zeros((len(bags), len(vocabulary)))
    for row, bag in enumerate(bags):
        for w, c in bag.items():
            counts[row, index[w]] = c
    norms = np.linalg.norm(counts, axis=1, keepdims=True)
    unit = np.divide(counts, norms, out=np.zeros_like(counts), where=norms > 0)
    return -(unit @ unit.T)  # lower is closer


def zsum(*matrices: np.ndarray) -> np.ndarray:
    """Distances combined after putting each on the same scale (mean 0, deviation 1)."""
    return sum((m - m.mean()) / m.std() for m in matrices)
