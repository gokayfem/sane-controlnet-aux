from controlnet_aux import ScribbleDetector
from PIL import Image
import numpy as np

import pytest


@pytest.fixture()
def detector():
    return ScribbleDetector.from_pretrained()


@pytest.mark.parametrize("output_type", ["pil", "np"])
def test_output_type(
    detector,
    image,
    output_type: str,
):
    output = detector(image, output_type=output_type)

    if output_type == "pil":
        assert isinstance(output, Image.Image)
    elif output_type == "np":
        assert isinstance(output, np.ndarray)
    else:
        assert False, f"Unknown output type: {output_type}"
