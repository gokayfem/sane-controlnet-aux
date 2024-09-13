from controlnet_aux import CannyDetector

import pytest

from tests.conftest import validate_image_type


@pytest.fixture(scope="module")
def detector():
    return CannyDetector()


@pytest.mark.parametrize("output_type", ["pil", "np"])
def test_output_type(detector, image, output_type: str):
    output = detector(image, output_type=output_type)

    validate_image_type(output, output_type)
