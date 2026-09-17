from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from caypollard.vision.encoders import DEFAULT_MODELS, _projected_image_features


@dataclass
class _Tensor:
    """Stand-in for a torch tensor: the only trait that matters here is `detach`."""

    value: str

    def detach(self) -> _Tensor:
        return self


@dataclass
class _VisionOutput:
    """Stand-in for transformers' BaseModelOutputWithPooling."""

    pooler_output: Any
    last_hidden_state: Any = None


def test_accepts_the_bare_tensor_returned_by_transformers_4():
    tensor = _Tensor("projected")
    assert _projected_image_features(tensor) is tensor


def test_unwraps_the_output_object_returned_by_transformers_5():
    # Transformers 5 replaces pooler_output with the projected features and
    # returns the whole vision output; extracting the tensor is what keeps the
    # CLIP and SigLIP presets working across the major-version boundary.
    projected = _Tensor("projected")
    assert _projected_image_features(_VisionOutput(pooler_output=projected)) is projected


def test_rejects_an_output_object_without_a_usable_tensor():
    with pytest.raises(TypeError, match="pooler_output"):
        _projected_image_features(_VisionOutput(pooler_output=None))


def test_presets_cover_three_distinct_representation_families():
    # The phase-2 exit criterion requires substantially different families under
    # one split and metric definition, so the preset table must not collapse.
    families = {spec.family for spec in DEFAULT_MODELS.values()}
    assert families == {"dinov2", "clip", "siglip"}
    assert len({spec.model_id for spec in DEFAULT_MODELS.values()}) == len(DEFAULT_MODELS)
