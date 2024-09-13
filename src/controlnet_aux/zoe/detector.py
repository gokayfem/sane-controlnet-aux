from typing import Literal, Union
from einops import rearrange
import numpy as np
from controlnet_aux.base.detector import BaseDetector
from controlnet_aux.utils import HWC3, resize_image_with_pad
import torch
from PIL import Image


class ZoeDetector(BaseDetector):
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.to(device)

    @classmethod
    def from_pretrained(cls):
        repo = "isl-org/ZoeDepth"
        model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True, trust_repo=True)
        return cls(model_zoe_n)

    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        detect_resolution: int = 512,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method: str = "INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        image, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )

        image_depth = image
        with torch.no_grad():
            image_depth = torch.from_numpy(image_depth).float().to(self.device)
            image_depth = image_depth / 255.0
            image_depth = rearrange(image_depth, "h w c -> 1 c h w")
            depth = self.model.infer(image_depth)

            depth = depth[0, 0].cpu().numpy()

            vmin = np.percentile(depth, 2)
            vmax = np.percentile(depth, 85)

            depth -= vmin
            depth /= vmax - vmin
            depth = 1.0 - depth
            depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = remove_pad(HWC3(depth_image))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
