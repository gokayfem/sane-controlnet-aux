from controlnet_aux import PiDiDetector
from PIL import Image
import numpy as np

import pytest

from tests.conftest import validate_image_type


@pytest.fixture()
def detector():
    return PiDiDetector.from_pretrained()


@pytest.mark.parametrize("output_type", ["pil", "np"])
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("scribble", [True, False])
@pytest.mark.parametrize("apply_filter", [True, False])
def test_output_type(
    detector, image, output_type: str, safe: bool, scribble: bool, apply_filter: bool
):
    output = detector(
        image,
        output_type=output_type,
        safe=safe,
        scribble=scribble,
        apply_filter=apply_filter,
    )

    validate_image_type(output, output_type)
