from controlnet_aux import HEDDetector
from controlnet_aux.utils import load_image
from PIL import Image
import numpy as np

import pytest


@pytest.fixture()
def detector():
    return HEDDetector.from_pretrained()


@pytest.fixture()
def image():
    return load_image("tests/data/pose_sample.jpg")


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("scribble", [True, False])
def test_output_type(detector, image, output_type: str, safe: bool, scribble: bool):
    output = detector(image, output_type=output_type, safe=safe, scribble=scribble)

    if output_type == "pil":
        assert isinstance(output, Image.Image)
    elif output_type == "np":
        assert isinstance(output, np.ndarray)
    else:
        assert False, f"Unknown output type: {output_type}"
