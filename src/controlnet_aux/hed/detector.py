from typing import Literal, Union
import cv2
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from controlnet_aux.base.detector import BaseDetector
from controlnet_aux.hed.model import HED
from controlnet_aux.utils import (
    custom_hf_download,
    nms,
    resize_image_with_pad,
    HWC3,
    safe_step,
)


class HEDDetector(BaseDetector):
    def __init__(self, model: HED):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="lllyasviel/Annotators",
        filename="ControlNetHED.pth",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)

        model = HED()
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.float().eval()

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
        scribble: bool = False,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method="INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        input_image, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )

        assert input_image.ndim == 3

        H, W, C = input_image.shape

        with torch.no_grad():
            image_hed = torch.from_numpy(input_image).float().to(self.device)
            image_hed = rearrange(image_hed, "h w c -> 1 c h w")
            edges = self.model(image_hed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))

            if safe:
                edge = safe_step(edge)

            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = edge

        if scribble:
            detected_map = nms(detected_map, 127, 3.0)
            detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
            detected_map[detected_map > 4] = 255
            detected_map[detected_map < 255] = 0

        detected_map = HWC3(remove_pad(detected_map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
