"""The workshop's measure, benches and bricks, without loading any model."""

from __future__ import annotations

import numpy as np
import pytest

from caypollard.atelier.benches import _score, top1_detail
from caypollard.atelier.bricks import BRICKS, COCO_KEPT, _grid, brick, convert_vitpose
from caypollard.atelier.measure import (
    dense_matrix,
    mirrored_keypoints,
    pose_matrix,
    posture,
    relation_words,
    scene_phrases,
    scene_words,
    set_matrix,
    vector_matrix,
    words_matrix,
    zsum,
)

STANDING = {
    "left_shoulder": [60, 40],
    "right_shoulder": [40, 40],
    "left_hip": [58, 80],
    "right_hip": [42, 80],
    "left_wrist": [90, 20],  # the figure's left arm raised
    "right_wrist": [35, 70],
}


def picture(keypoints, box=(30, 10, 95, 120), size=(200, 150)):
    return {"size": size, "figures": [{"box": list(box), "keypoints": keypoints}]}


def test_mirror_swaps_sides_and_reverses_x():
    m = mirrored_keypoints({"left_wrist": [90, 20], "nose": [50, 30]}, 200)
    assert m == {"right_wrist": [110, 20], "nose": [150, 30]}


def test_a_mirrored_copy_is_as_close_as_the_original():
    flipped = mirrored_keypoints(STANDING, 200)
    other = dict(STANDING, left_wrist=[70, 110])  # the arm lowered
    pictures = [
        picture(STANDING),
        picture(flipped, box=(105, 10, 170, 120)),
        picture(other),
    ]
    d = pose_matrix(pictures)
    assert d[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert d[0, 2] > d[0, 1]


def test_joints_filter_drops_the_others():
    with_head = dict(STANDING, crown=[50, 0])
    d_all = pose_matrix([picture(with_head), picture(STANDING)])
    d_coco = pose_matrix([picture(with_head), picture(STANDING)], joints=set(COCO_KEPT))
    assert d_all[0, 1] > d_coco[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_vector_matrix_keeps_the_closer_reading():
    plain = np.array([[1.0, 0.0], [0.0, 1.0]])
    flipped = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert vector_matrix(plain, None)[0, 1] == 0.0
    assert vector_matrix(plain, flipped)[0, 1] == -1.0


def test_dense_matrix_finds_a_detail_from_the_query_side():
    e = np.eye(4, dtype=np.float16)
    whole = e[[0, 1, 2, 3]]  # four different squares
    detail = e[[1, 1, 1, 1]]  # one of them, filling the picture
    other = e[[3, 3, 3, 3]] * 0 + np.float16(0.5)  # nothing in common, equally
    pictures = np.stack([detail, whole, other])
    one_way = dense_matrix(pictures, None, symmetric=False)
    both = dense_matrix(pictures, None, symmetric=True)
    assert one_way[0, 1] == pytest.approx(-1.0)  # every patch of the detail is found
    assert both[0, 1] == pytest.approx(-(1.0 + 0.25) / 2)  # most of the whole is not
    assert one_way[1, 0] == pytest.approx(-0.25)


def test_dense_matrix_keeps_the_mirrored_reading():
    e = np.eye(2, dtype=np.float16)
    plain = np.stack([e[[0, 0]], e[[1, 1]]])
    flipped = np.stack([e[[1, 1]], e[[0, 0]]])
    assert dense_matrix(plain, None, True)[0, 1] == pytest.approx(0.0)
    assert dense_matrix(plain, flipped, True)[0, 1] == pytest.approx(-1.0)


def test_posture_reads_the_torso_and_the_knees():
    hips = {"left_hip": [58, 80], "right_hip": [42, 80]}
    up = dict(hips, left_shoulder=[60, 40], right_shoulder=[40, 40])
    assert posture(dict(up, left_knee=[58, 120], right_knee=[42, 120])) == "debout"
    assert posture(dict(up, left_knee=[60, 90], right_knee=[40, 90])) == "agenouillée-assise"
    flat = dict(hips, left_shoulder=[10, 82], right_shoulder=[10, 78])
    assert posture(flat) == "couchée"
    assert posture({}) == "inconnue"


def test_relation_words_count_figures_postures_and_relations():
    standing = dict(STANDING, left_knee=[58, 120], right_knee=[42, 120], nose=[60, 30])
    d = {
        "size": (300, 200),
        "figures": [
            {"box": [30, 10, 95, 130], "keypoints": standing},
            {"box": [100, 10, 165, 130], "keypoints": standing},
        ],
    }
    words = relation_words(d)
    assert words["figures:2"] == 1 and words["posture:debout"] == 2
    assert any(w.startswith("regarde vers:debout>debout") for w in words)
    assert not any(":" in w and ">" in w for w in relation_words(d, relations_too=False))


def test_scene_words_use_roles_not_the_subject():
    scene = {
        "figures": [
            {"id": "f1", "role": "woman", "posture": "standing"},
            {"id": "f2", "role": "old woman", "posture": "bending"},
        ],
        "relations": [
            {"a": "f1", "rel": "embraces", "b": "f2"},
            {"a": "f1", "rel": "x", "b": "f9"},
        ],
        "subject": "the Visitation",
    }
    words = scene_words(scene)
    assert words == {
        "figures:2": 1,
        "figure:woman:standing": 1,
        "figure:old woman:bending": 1,
        "woman:embraces:old woman": 1,
    }
    assert "woman:embraces:old woman" not in scene_words(scene, relations_too=False)


def test_scene_phrases_and_set_matrix():
    scene = {
        "figures": [{"id": "f1", "role": "woman", "posture": "standing"}],
        "relations": [{"a": "f1", "rel": "holds", "b": "f1"}],
    }
    assert scene_phrases(scene) == ["standing woman", "woman holds woman"]
    e = np.eye(2)
    d = set_matrix([e[[0]], e[[0, 1]], np.zeros((0, 2))])
    assert d[0, 1] == pytest.approx(-0.75) and d[0, 2] == 0.0


def test_words_matrix_is_a_cosine_and_empty_is_far():
    d = words_matrix([{"a": 1}, {"a": 2}, {}])
    assert d[0, 1] == pytest.approx(-1.0) and d[0, 2] == 0.0


def test_zsum_puts_matrices_on_one_scale():
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(zsum(a, 10 * a), 2 * (a - a.mean()) / a.std())


def test_score_all_and_cross():
    items = [
        {"id": "a", "label": "x", "group": "emblem"},
        {"id": "b", "label": "x", "group": "museum"},
        {"id": "c", "label": "y", "group": "museum"},
        {"id": "d", "label": "y", "group": "emblem"},
    ]
    dist = np.array(
        [
            [0, 1, 2, 0.5],  # a: nearest overall is d (wrong), nearest museum work is b
            [1, 0, 2, 2],
            [2, 2, 0, 1],
            [0.5, 2, 1, 0],
        ]
    )
    everything = _score(dist, items, None)
    assert everything["top1"] == 0.5 and everything["queries"] == 4
    emblems = _score(dist, items, "emblem")
    assert emblems["top1"] == 1.0 and emblems["queries"] == 2


def test_top1_detail_counts_ties_as_equally_likely():
    items = [{"label": "x"}, {"label": "y"}, {"label": "x"}]
    dist = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=float)
    detail = top1_detail(dist, items)
    assert detail[0] == {"hit": False, "expected": 0.5, "tied": 2}  # manifest order picks y
    assert detail[2]["hit"] and detail[2]["expected"] == 1.0


def test_grid_pools_patch_tokens_to_16_by_16():
    import torch

    tokens = torch.zeros(1, 32 * 32, 4)
    tokens[0, :, 0] = 1.0  # every patch points the same way
    g = _grid(tokens)
    assert g.shape == (1, 256, 4) and np.allclose(g[0, :, 0], 1.0)


def test_vitpose_conversion_names_every_weight():
    import torch

    mm = {
        "backbone.pos_embed": torch.zeros(1),
        "backbone.patch_embed.projection.weight": torch.zeros(1),
        "backbone.patch_embed.projection.bias": torch.zeros(1),
        "backbone.ln1.weight": torch.zeros(1),
        "backbone.ln1.bias": torch.zeros(1),
        "head.deconv_layers.0.weight": torch.zeros(1),
        "head.deconv_layers.3.weight": torch.zeros(1),
        "head.final_layer.weight": torch.zeros(1),
        "head.final_layer.bias": torch.zeros(1),
    }
    for part in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        mm[f"head.deconv_layers.1.{part}"] = torch.zeros(1)
        mm[f"head.deconv_layers.4.{part}"] = torch.zeros(1)
    for part in ("weight", "bias"):
        mm[f"backbone.layers.0.attn.qkv.{part}"] = torch.arange(6.0)
        for name in ("attn.proj", "ln1", "ln2", "ffn.layers.0.0", "ffn.layers.1"):
            mm[f"backbone.layers.0.{name}.{part}"] = torch.zeros(1)
    out = convert_vitpose(mm)
    assert out["backbone.encoder.layer.0.attention.attention.key.weight"].tolist() == [2.0, 3.0]
    assert "head.batchnorm2.running_var" in out and len(out) == 5 + 16 + 14


def test_bricks_are_found_by_kind_and_name():
    assert brick("figures", "opus").kind == "figures"
    assert brick("pose", "opus").kind == "pose"
    assert {k for k, _ in BRICKS} == {"figures", "pose", "regions", "scene", "vector"}
    with pytest.raises(SystemExit):
        brick("pose", "nothing")
