"""A picture reduced to its strokes: contours in two colours, then fewer and straighter lines.

The record's other channels start from tone -- a silhouette is a threshold, a region is a
blob darker or lighter than the page. This starts from *edges* instead, at a scale coarse
enough that the hatching of an engraving and the brushwork of a painting are smoothed away
and only the boundaries of forms remain. Those boundaries are thinned to one-pixel curves,
walked into strokes (continuing through crossings along the straightest branch), simplified
to polylines, and then rationed: a sketch keeps the forty longest strokes, a pictogram the
twelve, the way an aleph keeps of an ox the few lines that still say "ox".

Everything is classical and parametric, so every step can be shown and every knob turned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DEFAULTS: dict[str, float | int] = {
    "side": 320,  # working size, longest side
    "sigma": 2.0,  # smoothing before the gradient: what kills hatching
    "high": 90.0,  # percentile of edge strength kept as a strong edge
    "low": 0.4,  # weak edges survive at this fraction of the strong threshold, if attached
    "turn": 60.0,  # largest bend, in degrees, a stroke may take through a crossing
    "epsilon": 1.5,  # polyline simplification, in pixels
    "sketch": 40,  # strokes kept for the sketch
    "pictogram": 12,  # strokes kept for the pictogram
    "pict_epsilon": 4.0,  # coarser simplification for the pictogram
}


@dataclass
class Sketch:
    size: tuple[int, int]  # width, height of the working frame
    gray: np.ndarray
    edges: np.ndarray  # one-pixel contours, True where ink
    strokes: list[list[tuple[int, int]]] = field(default_factory=list)  # (row, col) polylines
    sketch: list[list[tuple[int, int]]] = field(default_factory=list)
    pictogram: list[list[tuple[int, int]]] = field(default_factory=list)


# ---------- image to edges ----------


def to_gray(image: Image.Image, side: int) -> np.ndarray:
    im = image.convert("L")
    w, h = im.size
    s = side / max(w, h)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.0)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def contours(a: np.ndarray, *, sigma: float, high: float, low: float) -> np.ndarray:
    """Edges of the smoothed image: gradient, non-maximum suppression, hysteresis."""
    gy = ndimage.gaussian_filter(a, sigma, order=(1, 0))
    gx = ndimage.gaussian_filter(a, sigma, order=(0, 1))
    mag = np.hypot(gx, gy)
    q = (np.round(np.arctan2(gy, gx) / (np.pi / 4)) % 4).astype(int)
    P = np.pad(mag, 1)
    n1, n2 = np.empty_like(mag), np.empty_like(mag)
    for code, (a1, a2) in enumerate(
        [((1, 2), (1, 0)), ((2, 2), (0, 0)), ((2, 1), (0, 1)), ((2, 0), (0, 2))]
    ):
        m = q == code
        n1[m] = P[a1[0] : a1[0] + mag.shape[0], a1[1] : a1[1] + mag.shape[1]][m]
        n2[m] = P[a2[0] : a2[0] + mag.shape[0], a2[1] : a2[1] + mag.shape[1]][m]
    nms = (mag >= n1) & (mag >= n2) & (mag > 0)
    nz = mag[nms]
    strong_at = float(np.percentile(nz, high)) if nz.size else 1.0
    strong = nms & (mag >= strong_at)
    weak = nms & (mag >= low * strong_at)
    labels, count = ndimage.label(weak, structure=np.ones((3, 3)))
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def thin(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to one-pixel curves."""
    img = img.copy().astype(np.uint8)

    def step(sub: int) -> bool:
        P = np.pad(img, 1)
        p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
        p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
        p8, p9 = P[1:-1, :-2], P[:-2, :-2]
        nb = [p2, p3, p4, p5, p6, p7, p8, p9]
        B = sum(x.astype(np.int16) for x in nb)
        A = sum(((nb[j] == 0) & (nb[(j + 1) % 8] == 1)).astype(np.int16) for j in range(8))
        c = (
            (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            if sub == 0
            else (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
        )
        m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
        img[m] = 0
        return bool(m.any())

    while step(0) | step(1):
        pass
    return img.astype(bool)


# ---------- edges to strokes ----------


def trace(skeleton: np.ndarray, *, turn: float) -> list[list[tuple[int, int]]]:
    """Walk the skeleton into strokes; at a crossing, continue along the straightest branch."""
    pts = set(zip(*(x.tolist() for x in np.nonzero(skeleton)), strict=True))

    def nbrs(p):
        return [
            (p[0] + dy, p[1] + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and (p[0] + dy, p[1] + dx) in pts
        ]

    deg = {p: len(nbrs(p)) for p in pts}
    used: set[tuple] = set()
    lines: list[list[tuple[int, int]]] = []

    def walk(start, nxt):
        line = [start, nxt]
        used.add((start, nxt))
        used.add((nxt, start))
        cur, prev = nxt, start
        while True:
            cand = [q for q in nbrs(cur) if q != prev and (cur, q) not in used]
            if not cand:
                break
            if len(cand) == 1 and deg[cur] == 2:
                q = cand[0]
            else:
                tail = np.asarray(line[-6:], dtype=float)
                d = tail[-1] - tail[0]
                if np.hypot(*d) < 1e-9:
                    break
                best, best_angle = None, turn
                for q in cand:
                    v = np.asarray(q, dtype=float) - np.asarray(cur, dtype=float)
                    cos = np.dot(d, v) / (np.hypot(*d) * np.hypot(*v))
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
                    if angle < best_angle:
                        best, best_angle = q, angle
                if best is None:
                    break
                q = best
            used.add((cur, q))
            used.add((q, cur))
            line.append(q)
            prev, cur = cur, q
        return line

    for p in sorted(pts):
        if deg[p] != 2:
            for q in nbrs(p):
                if (p, q) not in used:
                    lines.append(walk(p, q))
    for p in sorted(pts):  # closed loops with no end and no crossing
        if deg[p] == 2:
            for q in nbrs(p):
                if (p, q) not in used:
                    lines.append(walk(p, q))
    return lines


def simplify(line: list[tuple[int, int]], epsilon: float) -> list[tuple[int, int]]:
    """Ramer-Douglas-Peucker."""
    if len(line) < 3:
        return line
    a = np.asarray(line, dtype=float)
    s, e = a[0], a[-1]
    v = e - s
    if np.hypot(*v) < 1e-9:
        d = np.linalg.norm(a - s, axis=1)
    else:
        d = np.abs(v[0] * (a - s)[:, 1] - v[1] * (a - s)[:, 0]) / np.hypot(*v)
    i = int(d.argmax())
    if d[i] > epsilon:
        return simplify(line[: i + 1], epsilon)[:-1] + simplify(line[i:], epsilon)
    return [line[0], line[-1]]


def length(line: list[tuple[int, int]]) -> float:
    a = np.asarray(line, dtype=float)
    return float(np.hypot(*(a[1:] - a[:-1]).T).sum()) if len(a) > 1 else 0.0


def longest(lines: list[list[tuple[int, int]]], n: int) -> list[list[tuple[int, int]]]:
    return sorted(lines, key=length, reverse=True)[:n]


def sketch(image: Image.Image, params: dict | None = None) -> Sketch:
    p = {**DEFAULTS, **(params or {})}
    gray = to_gray(image, int(p["side"]))
    edges = contours(gray, sigma=float(p["sigma"]), high=float(p["high"]), low=float(p["low"]))
    raw = trace(thin(edges), turn=float(p["turn"]))
    strokes = [simplify(line, float(p["epsilon"])) for line in raw if len(line) > 1]
    picked = longest(strokes, int(p["sketch"]))
    pict = [
        simplify(line, float(p["pict_epsilon"])) for line in longest(strokes, int(p["pictogram"]))
    ]
    return Sketch(
        size=(gray.shape[1], gray.shape[0]),
        gray=gray,
        edges=edges,
        strokes=strokes,
        sketch=picked,
        pictogram=pict,
    )


# ---------- strokes to images and numbers ----------


def render(
    lines: list[list[tuple[int, int]]], size: tuple[int, int], *, width: int = 2
) -> Image.Image:
    im = Image.new("L", size, 255)
    draw = ImageDraw.Draw(im)
    for line in lines:
        if len(line) > 1:
            draw.line([(x, y) for y, x in line], fill=0, width=width)
    return im


def edges_image(sk: Sketch) -> Image.Image:
    return Image.fromarray(((~sk.edges) * 255).astype(np.uint8))


STROKE_FEATURES = (
    "log longueur / côté",
    "rectitude (corde / longueur)",
    "cos 2θ",
    "sin 2θ",
    "virage moyen par pas",
    "virage total / 2π",
    "fermé",
    "log sommets",
    "élongation de la boîte",
    "x",
    "y",
    "virages de moins de 15°",
    "de 15 à 30°",
    "de 30 à 60°",
    "de 60 à 90°",
    "de plus de 90°",
)


def stroke_descriptor(line: list[tuple[int, int]], size: tuple[int, int]) -> np.ndarray:
    """A support-free description of one stroke: its size, direction, bending and place."""
    a = np.asarray(line, dtype=float)
    side = float(max(size))
    seg = a[1:] - a[:-1]
    seglen = np.hypot(*seg.T)
    total = float(seglen.sum())
    chord = float(np.hypot(*(a[-1] - a[0])))
    theta = math.atan2(float(a[-1][0] - a[0][0]), float(a[-1][1] - a[0][1]))
    angles = np.arctan2(seg[:, 0], seg[:, 1])
    turns = np.diff(angles)
    turns = (turns + np.pi) % (2 * np.pi) - np.pi if turns.size else turns
    absturn = np.abs(turns)
    hist = np.histogram(np.degrees(absturn), bins=[0, 15, 30, 60, 90, 181])[0].astype(float)
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    closed = 1.0 if len(a) > 3 and chord < 0.1 * max(total, 1e-6) else 0.0
    box = a.max(axis=0) - a.min(axis=0)
    return np.asarray(
        [
            math.log1p(total / side * 10),
            chord / max(total, 1e-6),
            math.cos(2 * theta),
            math.sin(2 * theta),
            float(absturn.mean()) if absturn.size else 0.0,
            float(turns.sum()) / (2 * math.pi) if turns.size else 0.0,
            closed,
            math.log1p(len(a)),
            math.log1p(max(box) / max(min(box), 1.0)),
            float(a[:, 1].mean()) / max(size[0], 1),
            float(a[:, 0].mean()) / max(size[1], 1),
            *hist,
        ],
        dtype=np.float32,
    )
