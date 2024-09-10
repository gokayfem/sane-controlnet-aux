from typing import Literal, Union
import cv2
import numpy as np
import torch
from PIL import Image

from controlnet_aux.base.detector import BaseDetector
from controlnet_aux.utils import HWC3, custom_hf_download, resize_image_with_pad

from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .utils import pred_lines


class MLSDDetector(BaseDetector):
    def __init__(self, model: MobileV2_MLSD_Large):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="lllyasviel/Annotators",
        filename="mlsd_large_512_fp32.pth",
    ):
        subfolder = (
            "annotator/ckpts"
            if pretrained_model_or_path == "lllyasviel/ControlNet"
            else ""
        )
        model_path = custom_hf_download(
            pretrained_model_or_path, filename, subfolder=subfolder
        )
        model = MobileV2_MLSD_Large()
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        score_threshold: float = 0.1,
        distance_threshold: float = 0.1,
        detect_resolution: int = 512,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method="INTER_AREA",
    ):
        image, output_type = self.validate_input(image, output_type)
        detected_map, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )
        img = detected_map
        img_output = np.zeros_like(img)
        try:
            with torch.no_grad():
                lines = pred_lines(
                    img,
                    self.model,
                    [img.shape[0], img.shape[1]],
                    score_threshold,
                    distance_threshold,
                )
                for line in lines:
                    x_start, y_start, x_end, y_end = [int(val) for val in line]
                    cv2.line(
                        img_output,
                        (x_start, y_start),
                        (x_end, y_end),
                        [255, 255, 255],
                        1,
                    )
        except Exception:
            # TODO: find a test case
            pass

        detected_map = remove_pad(HWC3(img_output[:, :, 0]))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
