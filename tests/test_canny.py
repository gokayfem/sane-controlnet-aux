from controlnet_aux import CannyDetector

import pytest


@pytest.fixture()
def detector():
    return CannyDetector()


def test_output_type_pil(detector, image):
    from PIL import Image

    output = detector(image, output_type="pil")
    assert isinstance(output, Image.Image)


def test_output_type_np(detector, image):
    import numpy as np

    output = detector(image, output_type="np")
    assert isinstance(output, np.ndarray)
