from typing import Any, Literal, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from controlnet_aux.base.detector import BaseDetector
from controlnet_aux.pidi.model import PiDiNet, pidinet
from controlnet_aux.utils import (
    HWC3,
    nms,
    resize_image_with_pad,
    safe_step,
    custom_hf_download,
)


class PiDiDetector(BaseDetector):
    def __init__(self, model: PiDiNet):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="lllyasviel/Annotators",
        filename="table5_pidinet.pth",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        model = pidinet()
        state_dict: dict[str, Any] = torch.load(model_path, weights_only=True)[
            "state_dict"
        ]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()

        return cls(model)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        detect_resolution: int = 512,
        safe: bool = False,
        output_type: Literal["pil", "np"] = "pil",
        scribble: bool = False,
        apply_filter: bool = False,
        upscale_method: str = "INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        detected_map, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )

        detected_map = detected_map[:, :, ::-1].copy()
        with torch.no_grad():
            image_pidi = torch.from_numpy(detected_map).float().to(self.device)
            image_pidi = image_pidi / 255.0
            image_pidi = rearrange(image_pidi, "h w c -> 1 c h w")
            edge = self.model(image_pidi)[-1]
            edge = edge.cpu().numpy()

            if apply_filter:
                edge = edge > 0.5

            if safe:
                edge = safe_step(edge)

            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge[0, 0]

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
