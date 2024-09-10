from typing import Literal, Union
from PIL import Image

import numpy as np


class BaseDetector:
    def validate_input(self, image: Union[Image.Image, np.ndarray], output_type: str):
        if not isinstance(image, (Image.Image, np.ndarray)):
            raise ValueError(
                f"Input image must be a PIL Image or a numpy array. "
                f"Got {type(image)} instead."
            )

        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        return (image, output_type)

    @classmethod
    def from_pretrained(cls):
        raise NotImplementedError()

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        output_type: Literal["pil", "np"] = "pil",
    ):
        raise NotImplementedError()
