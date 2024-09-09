from controlnet_aux import LineartDetector
from controlnet_aux.utils import load_image
from PIL import Image
import numpy as np

import pytest


@pytest.fixture()
def detector():
    return LineartDetector.from_pretrained()


@pytest.fixture()
def image():
    return load_image("tests/data/pose_sample.jpg")


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("coarse", [False, True])
def test_output_type(detector, image, output_type: str, coarse: bool):
    output = detector(image, output_type=output_type, coarse=coarse)

    if output_type == "pil":
        assert isinstance(output, Image.Image)
        output.save(f"{coarse}_lineart.jpg")
    elif output_type == "np":
        assert isinstance(output, np.ndarray)
    else:
        assert False, f"Unknown output type: {output_type}"
