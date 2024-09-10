from typing import Literal
import pytest
from controlnet_aux.utils import load_image

from PIL import Image

import numpy as np


@pytest.fixture()
def image():
    return load_image("tests/data/pose_sample.jpg")


def validate_image_type(output_image: object, output_type: Literal["pil", "np"]):
    if output_type == "pil":
        assert isinstance(output_image, Image.Image)
    elif output_type == "np":
        assert isinstance(output_image, np.ndarray)
    else:
        assert False, f"Unknown output type: {output_type}"
