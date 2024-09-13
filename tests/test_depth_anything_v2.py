from controlnet_aux import DepthAnythingV2Detector

import pytest

from tests.conftest import validate_image_type


@pytest.fixture(scope="module")
def detector():
    return DepthAnythingV2Detector.from_pretrained()


@pytest.mark.parametrize("output_type", ["pil", "np"])
def test_output_type(
    detector,
    image,
    output_type: str,
):
    output = detector(image, output_type=output_type)
    validate_image_type(output, output_type)
