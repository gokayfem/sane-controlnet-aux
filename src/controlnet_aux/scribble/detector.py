from typing import Literal, Union
import numpy as np
from controlnet_aux.hed.detector import HEDDetector

from PIL import Image


class ScribbleDetector(HEDDetector):
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        detect_resolution: int = 512,
        safe: bool = False,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method="INTER_CUBIC",
    ):
        return super().__call__(
            image=image,
            detect_resolution=detect_resolution,
            safe=safe,
            scribble=True,
            output_type=output_type,
            upscale_method=upscale_method,
        )
