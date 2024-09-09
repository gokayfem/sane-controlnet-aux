from typing import Literal, Union
import cv2
import numpy as np
from PIL import Image
from controlnet_aux.utils import resize_image_with_pad, HWC3


class CannyDetector:
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

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        low_threshold: int = 100,
        high_threshold: int = 200,
        detect_resolution: int = 512,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method="INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        detected_map, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )
        detected_map = cv2.Canny(detected_map, low_threshold, high_threshold)
        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
