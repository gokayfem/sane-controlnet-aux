from typing import Literal, Union
import torch

from controlnet_aux.lineart.generator import Generator
from controlnet_aux.utils import HWC3, custom_hf_download, resize_image_with_pad
from PIL import Image
import numpy as np
from einops import rearrange


class LineartDetector:
    def __init__(self, model: Generator, coarse_model: Generator):
        self.model = model
        self.model_coarse = coarse_model
        self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="lllyasviel/Annotators",
        filename="sk_model.pth",
        coarse_filename="sk_model2.pth",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        coarse_model_path = custom_hf_download(
            pretrained_model_or_path, coarse_filename
        )

        model = Generator(3, 1, 3)
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        )
        model.eval()

        coarse_model = Generator(3, 1, 3)
        coarse_model.load_state_dict(
            torch.load(
                coarse_model_path, map_location=torch.device("cpu"), weights_only=True
            )
        )
        coarse_model.eval()

        return cls(model, coarse_model)

    def to(self, device):
        self.model.to(device)
        self.model_coarse.to(device)
        self.device = device
        return self

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
        image,
        coarse: bool = False,
        detect_resolution: int = 512,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method: str = "INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        detected_map, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )

        model = self.model_coarse if coarse else self.model

        # TODO: find a test case for this
        assert detected_map.ndim == 3

        with torch.no_grad():
            image = torch.from_numpy(detected_map).float().to(self.device)
            image = image / 255.0
            image = rearrange(image, "h w c -> 1 c h w")
            line = model(image)[0][0]

            line = line.cpu().numpy()
            line = (line * 255.0).clip(0, 255).astype(np.uint8)

        detected_map = HWC3(line)
        detected_map = remove_pad(255 - detected_map)

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
