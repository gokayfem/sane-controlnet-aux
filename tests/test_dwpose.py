from controlnet_aux import DWPoseProcessor
from controlnet_aux.utils import load_image


def test_processor():
    image = load_image("tests/test.jpg")
    print(image.shape)
    processor = DWPoseProcessor()
    out = processor(image)
