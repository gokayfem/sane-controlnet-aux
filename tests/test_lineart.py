from controlnet_aux import LineartDetector
from PIL import Image
import numpy as np

import pytest

from tests.conftest import validate_image_type


@pytest.fixture()
def detector():
    return LineartDetector.from_pretrained()


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("coarse", [False, True])
def test_output_type(detector, image, output_type: str, coarse: bool):
    output = detector(image, output_type=output_type, coarse=coarse)

    validate_image_type(output, output_type)
