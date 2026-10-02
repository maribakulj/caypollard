"""Figures, their poses, the objects they hold, and the relations between them.

The pilot's representation of a picture is not a bag of regions: it is a short list of
*figures* -- each a named body with a skeleton and a silhouette -- and *objects* -- each a
named outline -- from which relations are computed by geometry alone: whose hand is on
whom, who holds what, who faces whom, whose arm reaches toward whom.

Everything here is pure geometry on plain lists, so a corrected keypoint recomputes the
relations at once and the same code runs in the pipeline, in the viewer's server and in
the tests. Coordinates are image pixels, x to the right and y downward. Left and right
in keypoint names are the figure's own.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

Point = Sequence[float]
Polygon = Sequence[Point]

KEYPOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Limb segments read against the torso axis, after Impett and Süsstrunk.
SEGMENTS = (
    ("bras gauche", "left_shoulder", "left_elbow"),
    ("avant-bras gauche", "left_elbow", "left_wrist"),
    ("bras droit", "right_shoulder", "right_elbow"),
    ("avant-bras droit", "right_elbow", "right_wrist"),
    ("cuisse gauche", "left_hip", "left_knee"),
    ("jambe gauche", "left_knee", "left_ankle"),
    ("cuisse droite", "right_hip", "right_knee"),
    ("jambe droite", "right_knee", "right_ankle"),
)

HANDS = (("main gauche", "left_wrist", "left_elbow"), ("main droite", "right_wrist", "right_elbow"))
FEET = (("pied gauche", "left_ankle"), ("pied droit", "right_ankle"))


def mid(a: Point | None, b: Point | None) -> tuple[float, float] | None:
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return tuple(a or b) if (a or b) else None  # type: ignore[return-value]


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < x0 + (p[1] - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def distance_to_polygon(p: Point, poly: Polygon) -> float:
    """Zero inside, otherwise the distance to the nearest edge."""
    if len(poly) >= 3 and point_in_polygon(p, poly):
        return 0.0
    best = math.inf
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x1 - x0, y1 - y0
        t = (
            0.0
            if dx == dy == 0
            else max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / (dx * dx + dy * dy)))
        )
        best = min(best, math.hypot(p[0] - (x0 + t * dx), p[1] - (y0 + t * dy)))
    return best


def box_polygon(box: Sequence[float]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def outline(item: dict) -> Polygon:
    return item.get("polygon") or box_polygon(item["box"])


def centre(item: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = item["box"]
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def height(figure: dict) -> float:
    return figure["box"][3] - figure["box"][1]


def torso_axis(k: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """(hip midpoint, shoulder midpoint), or None when either end is missing."""
    shoulders = mid(k.get("left_shoulder"), k.get("right_shoulder"))
    hips = mid(k.get("left_hip"), k.get("right_hip"))
    return (hips, shoulders) if shoulders and hips else None


def angle(v: Point) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def wrap(a: float) -> float:
    return (a + 180) % 360 - 180


def pose_angles(k: dict) -> dict[str, float | None]:
    """Each limb's angle against the torso axis, in degrees; None where a point is missing.

    0 means the limb points along the torso from hips to shoulders (an arm raised), 180 the
    reverse (an arm hanging); positive turns toward the picture's right as seen with the
    torso upright. Invariant to the figure's size, position and overall tilt.
    """
    axis = torso_axis(k)
    out: dict[str, float | None] = {}
    for name, a, b in SEGMENTS:
        if not axis or not k.get(a) or not k.get(b):
            out[name] = None
            continue
        torso = angle((axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]))
        limb = angle((k[b][0] - k[a][0], k[b][1] - k[a][1]))
        out[name] = round(wrap(limb - torso), 1)
    return out


def facing(k: dict) -> int:
    """-1 if the head faces the picture's left, +1 its right, 0 when it cannot be told.

    The nose is compared with the ears, else the eyes, else the shoulders: in profile the
    nose leads; seen frontally it sits between them and the answer is 0.
    """
    nose = k.get("nose")
    if not nose:
        return 0
    for pair, tolerance in (
        (("left_ear", "right_ear"), 0.15),
        (("left_eye", "right_eye"), 0.25),
        (("left_shoulder", "right_shoulder"), 0.15),
    ):
        a, b = k.get(pair[0]), k.get(pair[1])
        ref = mid(a, b)
        if not ref:
            continue
        span = abs(a[0] - b[0]) if a and b else 0.0
        offset = nose[0] - ref[0]
        if a and b and abs(offset) <= tolerance * span:
            return 0
        if abs(offset) > 2:
            return 1 if offset > 0 else -1
    return 0


def _merge(found: dict, subject: str, rel: str, target: str, part: str, **evidence) -> None:
    """Add a relation, or name one more hand or foot supporting one already found."""
    key = (rel, target)
    if key in found:
        found[key]["par"] += f" et {part}"
        for name, value in evidence.items():
            found[key][name] = min(found[key][name], value)
    else:
        found[key] = {"a": subject, "rel": rel, "b": target, "par": part, **evidence}


def relations(figures: list[dict], objects: list[dict], *, reach: float = 0.06) -> list[dict]:
    """Relations computed from geometry. ``reach`` is a fraction of the figure's height.

    Each relation names its subject, its verb, its object and the evidence for it, so the
    viewer can say *why* and a correction can be seen to change it.
    """
    out: list[dict] = []
    for a in figures:
        k = a.get("keypoints") or {}
        tol = reach * height(a)
        # One relation per (verb, target), naming every hand or foot that supports it.
        found: dict[tuple[str, str], dict] = {}
        add = functools.partial(_merge, found, a["id"])

        for hand, wrist, elbow in HANDS:
            w = k.get(wrist)
            if not w:
                continue
            for o in objects:
                d = distance_to_polygon(w, outline(o))
                if d <= tol:
                    add("tient", o["id"], hand, distance=round(d))
            for b in figures:
                if b is a:
                    continue
                d = distance_to_polygon(w, outline(b))
                if d <= tol:
                    add("touche", b["id"], hand, distance=round(d))
                e = k.get(elbow)
                if e and d > tol:
                    arm = (w[0] - e[0], w[1] - e[1])
                    to_b = (centre(b)[0] - w[0], centre(b)[1] - w[1])
                    gap = abs(wrap(angle(arm) - angle(to_b)))
                    if math.hypot(*arm) > 0 and gap <= 30:
                        add("tend le bras vers", b["id"], hand, écart=round(gap))
        for foot, ankle in FEET:
            p = k.get(ankle)
            if not p:
                continue
            for o in objects:
                d = distance_to_polygon(p, outline(o))
                if d <= tol:
                    add("pose le pied sur", o["id"], foot, distance=round(d))
        # A foot on another figure is not read: where two silhouettes overlap in depth, a
        # foot falls inside the other outline without standing on it (5 false of 5 in the
        # pilot's first pass).
        out.extend(found.values())
        face = facing(k)
        for b in figures:
            if b is a:
                continue
            side = 1 if centre(b)[0] > centre(a)[0] else -1
            if face == side:
                out.append({"a": a["id"], "rel": "regarde vers", "b": b["id"]})
            elif face == -side:
                out.append({"a": a["id"], "rel": "tourne le dos à", "b": b["id"]})
            if a["box"][3] < b["box"][1] + 0.25 * height(b):
                out.append({"a": a["id"], "rel": "au-dessus de", "b": b["id"]})
            if height(a) > 1.3 * height(b):
                out.append(
                    {
                        "a": a["id"],
                        "rel": "plus grand que",
                        "b": b["id"],
                        "rapport": round(height(a) / height(b), 2),
                    }
                )
    return out


def build_record(description: dict, polygons: dict, corrections: dict | None = None) -> dict:
    """One picture's representation: figures, objects, relations, and the model's sentence.

    ``description`` is the proposal (``scripts/pilot_describe.py``), ``polygons`` the outline
    of each figure and object by id, ``corrections`` what a person changed: per figure id,
    ``keypoints`` (replacing the proposed ones point by point) and ``name``; per object id,
    ``name``; and ``removed``, ids to drop. Relations are always recomputed.
    """
    corrections = corrections or {}
    removed = set(corrections.get("removed", []))
    edits = corrections.get("items", {})
    figures, objects = [], []
    for f in description.get("figures", []):
        if f["id"] in removed:
            continue
        edit = edits.get(f["id"], {})
        k = {name: (f.get("keypoints") or {}).get(name) for name in KEYPOINTS}
        k.update(edit.get("keypoints", {}))
        figure = {
            "id": f["id"],
            "name": edit.get("name", f.get("name", "")),
            "attributes": f.get("attributes", []),
            "box": f["box"],
            "keypoints": k,
            "polygon": polygons.get(f["id"]),
            "corrected": sorted(edit.get("keypoints", {})),
        }
        figure["pose"] = pose_angles(k)
        figure["skeleton"] = normalised_skeleton(k)
        figure["facing"] = facing(k)
        figures.append(figure)
    for o in description.get("objects", []):
        if o["id"] in removed:
            continue
        edit = edits.get(o["id"], {})
        objects.append(
            {
                "id": o["id"],
                "name": edit.get("name", o.get("name", "")),
                "box": o["box"],
                "polygon": polygons.get(o["id"]),
                "held_by_said": o.get("held_by"),
            }
        )
    return {
        "size": description.get("_size"),
        "model": description.get("_model"),
        "phrase": description.get("phrase", ""),
        "figures": figures,
        "objects": objects,
        "relations": relations(figures, objects),
        "corrections": corrections,
    }


def normalised_skeleton(k: dict) -> dict[str, tuple[float, float]]:
    """The skeleton moved to its hip midpoint, scaled to unit torso, torso pointing up.

    This is the figure's abstraction in the Impett sense: what is left once position,
    size and tilt are discarded.
    """
    axis = torso_axis(k)
    if not axis:
        return {}
    (hx, hy), (sx, sy) = axis
    length = math.hypot(sx - hx, sy - hy) or 1.0
    rot = math.radians(-90 - angle((sx - hx, sy - hy)))
    c, s = math.cos(rot), math.sin(rot)
    out = {}
    for name, p in k.items():
        if p:
            x, y = (p[0] - hx) / length, (p[1] - hy) / length
            out[name] = (round(c * x - s * y, 3), round(s * x + c * y, 3))
    return out
