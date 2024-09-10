import pytest
from controlnet_aux.utils import load_image


@pytest.fixture()
def image():
    return load_image("tests/data/pose_sample.jpg")
