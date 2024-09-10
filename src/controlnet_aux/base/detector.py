from typing import Union
from PIL import Image

import numpy as np


class BaseDetector:
    def validate_input(
        self, input_image: Union[Image.Image, np.ndarray], output_type: str
    ):
        if not isinstance(input_image, (Image.Image, np.ndarray)):
            raise ValueError(
                f"Input image must be a PIL Image or a numpy array. "
                f"Got {type(input_image)} instead."
            )

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        return (input_image, output_type)

    @classmethod
    def from_pretrained(cls):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()
