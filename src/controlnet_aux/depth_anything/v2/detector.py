from typing import Literal, Union
import numpy as np
import torch
from einops import repeat
from PIL import Image
from controlnet_aux.base.detector import BaseDetector
from controlnet_aux.depth_anything.v2.utils import (
    DEPTH_ANYTHING_V2_MODEL_NAME_DICT,
    MODEL_CONFIGS,
)
from controlnet_aux.utils import resize_image_with_pad, custom_hf_download
import cv2
from depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingV2Detector(BaseDetector):
    def __init__(self, model: DepthAnythingV2, filename: str, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.filename = filename

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path: str | None = None,
        filename: str = "depth_anything_v2_vits.pth",
    ):
        if pretrained_model_or_path is None:
            pretrained_model_or_path = DEPTH_ANYTHING_V2_MODEL_NAME_DICT[filename]

        model_path = custom_hf_download(pretrained_model_or_path, filename)
        model = DepthAnythingV2(**MODEL_CONFIGS[filename])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.eval()

        return cls(model, filename)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        detect_resolution: str = 512,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method: str = "INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        image, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )

        depth = self.model.infer_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if "metric" in self.filename:
            depth = 255 - depth

        detected_map = repeat(depth, "h w -> h w 3")
        detected_map = remove_pad(detected_map)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
