from controlnet_aux import DWPoseProcessor
from controlnet_aux.utils import load_image


def test_processor():
    image = load_image("tests/test.jpg")
    print(image.shape)
    processor = DWPoseProcessor(device="mps")
    print("Processor created.")
    out = processor(image)
    breakpoint()
