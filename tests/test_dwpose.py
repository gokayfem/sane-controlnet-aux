from controlnet_aux import DWPoseDetector
from controlnet_aux.utils import load_image

import pytest


@pytest.fixture()
def detector():
    return DWPoseDetector.from_pretrained()

@pytest.fixture()
def image():
    return load_image("tests/data/pose_sample.jpg")

def test_output_type_pil(detector, image):
    from PIL import Image

    output = detector(image, output_type="pil")
    assert isinstance(output, Image.Image)

def test_output_type_np(detector, image):
    import numpy as np

    output = detector(image, output_type="np")
    assert isinstance(output, np.ndarray)
