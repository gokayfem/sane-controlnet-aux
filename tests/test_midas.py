from typing import Literal
from controlnet_aux import MidasDetector
from PIL import Image
import numpy as np

import pytest

from tests.conftest import validate_image_type


@pytest.fixture(scope="module")
def detector():
    return MidasDetector.from_pretrained()


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("depth_and_normal", [True, False])
def test_output_type(detector, image, output_type: str, depth_and_normal: bool):
    result = detector(
        image,
        output_type=output_type,
        depth_and_normal=depth_and_normal,
    )

    outputs = result if isinstance(result, tuple) else (result,)

    if depth_and_normal:
        assert len(outputs) == 2

    for output in outputs:
        validate_image_type(output, output_type)
