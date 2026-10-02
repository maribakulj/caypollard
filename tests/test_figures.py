from caypollard.figures import (
    distance_to_polygon,
    facing,
    normalised_skeleton,
    pose_angles,
    relations,
)

UPRIGHT = {
    "nose": [50, 10],
    "left_ear": [45, 10],
    "right_ear": [55, 10],
    "left_shoulder": [40, 30],
    "right_shoulder": [60, 30],
    "left_elbow": [40, 50],
    "left_wrist": [40, 70],
    "right_elbow": [80, 30],
    "right_wrist": [100, 30],
    "left_hip": [45, 80],
    "right_hip": [55, 80],
}


def test_distance_is_zero_inside_and_euclidean_outside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert distance_to_polygon((5, 5), square) == 0.0
    assert distance_to_polygon((13, 14), square) == 5.0


def test_pose_angles_read_limbs_against_the_torso():
    angles = pose_angles(UPRIGHT)
    assert abs(angles["avant-bras gauche"]) == 180.0  # hanging
    assert angles["avant-bras droit"] == 90.0  # held out toward the picture's right
    assert angles["cuisse gauche"] is None


def test_pose_angles_ignore_the_figure_tilt():
    lying = {k: [v[1], -v[0]] for k, v in UPRIGHT.items()}  # the whole figure turned 90°
    assert pose_angles(lying) == pose_angles(UPRIGHT)


def test_facing_reads_the_nose_against_ears_eyes_or_shoulders():
    assert facing(UPRIGHT) == 0
    assert facing({**UPRIGHT, "nose": [62, 10]}) == 1
    assert facing({"nose": [30, 10], "left_shoulder": [40, 30], "right_shoulder": [60, 30]}) == -1


def test_normalised_skeleton_puts_hips_at_origin_and_shoulders_up():
    sk = normalised_skeleton(UPRIGHT)
    assert sk["left_hip"][1] == 0.0
    assert sk["left_shoulder"][1] < 0


def test_relations_from_geometry():
    death = {"id": "F1", "box": [0, 0, 100, 100], "keypoints": {**UPRIGHT, "nose": [62, 10]}}
    youth = {"id": "F2", "box": [98, 0, 200, 100], "keypoints": {}}
    trumpet = {"id": "O1", "box": [30, 65, 45, 75]}
    found = {(r["a"], r["rel"], r["b"]) for r in relations([death, youth], [trumpet])}
    assert ("F1", "tient", "O1") in found  # left wrist inside the trumpet's box
    assert ("F1", "touche", "F2") in found  # right wrist at x=100, inside the youth's box
    assert ("F1", "regarde vers", "F2") in found
    assert not any(a == "F2" for a, _, _ in found)  # no keypoints, no claims
