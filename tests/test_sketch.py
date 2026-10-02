"""The stroke chain on drawn shapes whose strokes are known."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from caypollard import sketch as sketching


def drawing(size: int = 200) -> Image.Image:
    """A house: a square with a triangle roof, on a grey ground, plus fine hatching."""
    im = Image.new("L", (size, size), 200)
    draw = ImageDraw.Draw(im)
    draw.rectangle([50, 100, 150, 170], fill=90)
    draw.polygon([(40, 100), (100, 50), (160, 100)], fill=60)
    for x in range(55, 145, 4):  # hatching that a coarse scale must ignore
        draw.line([(x, 105), (x, 165)], fill=110, width=1)
    return im


def test_contours_are_thin_two_colour_and_ignore_hatching() -> None:
    sk = sketching.sketch(drawing(), {"side": 200})
    assert sk.edges.dtype == bool
    assert 0.005 < sk.edges.mean() < 0.08, "a contour drawing is mostly page"
    # the hatching runs every 4 px; at sigma 2 it must not produce ~20 vertical strokes
    verticals = [
        line
        for line in sk.strokes
        if abs(line[0][1] - line[-1][1]) < 3 and sketching.length(line) > 40
    ]
    assert len(verticals) <= 6


def test_strokes_are_rationed_and_sorted_by_length() -> None:
    sk = sketching.sketch(drawing(), {"side": 200, "sketch": 5, "pictogram": 2})
    assert len(sk.sketch) == min(5, len(sk.strokes))
    assert len(sk.pictogram) == min(2, len(sk.strokes))
    lengths = [sketching.length(line) for line in sk.sketch]
    assert lengths == sorted(lengths, reverse=True)


def test_simplify_keeps_ends_and_drops_collinear_points() -> None:
    line = [(0, i) for i in range(20)]  # a straight horizontal stroke
    assert sketching.simplify(line, 1.0) == [(0, 0), (0, 19)]
    bent = [(0, i) for i in range(10)] + [(j, 9) for j in range(1, 10)]
    assert sketching.simplify(bent, 1.0) == [(0, 0), (0, 9), (9, 9)]


def test_stroke_descriptor_reads_direction_and_bending() -> None:
    size = (100, 100)
    horizontal = sketching.stroke_descriptor([(50, 10), (50, 90)], size)
    vertical = sketching.stroke_descriptor([(10, 50), (90, 50)], size)
    assert horizontal.shape == (len(sketching.STROKE_FEATURES),)
    assert horizontal[1] == 1.0 and vertical[1] == 1.0  # straight
    assert horizontal[2] > 0.99 and vertical[2] < -0.99  # cos 2θ separates them
    square = [(10, 10), (10, 90), (90, 90), (90, 10), (10, 10)]
    closed = sketching.stroke_descriptor(square, size)
    assert closed[6] == 1.0
    assert (
        abs(abs(closed[5]) - 0.75) < 0.05
    )  # three right-angle turns, in quarter turns of 2π... 270°/360°


def test_render_draws_ink_on_a_white_page() -> None:
    im = sketching.render([[(5, 5), (5, 50)]], (60, 60), width=1)
    a = np.asarray(im)
    assert a[5, 5] == 0 and a[30, 30] == 255
