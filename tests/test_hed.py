from controlnet_aux import HEDDetector
from PIL import Image
import numpy as np

import pytest

from tests.conftest import validate_image_type


@pytest.fixture(scope="module")
def detector():
    return HEDDetector.from_pretrained()


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("scribble", [True, False])
def test_output_type(detector, image, output_type: str, safe: bool, scribble: bool):
    output = detector(image, output_type=output_type, safe=safe, scribble=scribble)

    validate_image_type(output, output_type)
