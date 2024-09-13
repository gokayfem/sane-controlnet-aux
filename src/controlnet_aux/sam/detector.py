from typing import Literal, Union
import numpy as np
from PIL import Image
from controlnet_aux.base.detector import BaseDetector
from controlnet_aux.utils import custom_hf_download, resize_image_with_pad, HWC3
from segment_anything import SamAutomaticMaskGenerator
from controlnet_aux.sam.sam_registry import sam_model_registry


class SamDetector(BaseDetector):
    def __init__(self, mask_generator, device: str = "cuda"):
        self.mask_generator = mask_generator
        self.to(device)

    def to(self, device: str):
        model = self.mask_generator.predictor.model.to(device)
        model.train(False)
        self.mask_generator = SamAutomaticMaskGenerator(model)
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path: str = "dhkim2810/MobileSAM",
        model_type: str = "vit_t",
        filename: str = "mobile_sam.pt",
    ):
        model_path = custom_hf_download(pretrained_model_or_path, filename)
        sam = sam_model_registry[model_type](checkpoint=model_path)
        mask_generator = SamAutomaticMaskGenerator(sam)
        return cls(mask_generator)

    def show_anns(self, anns):
        if len(anns) == 0:
            return

        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        h, w = anns[0]["segmentation"].shape
        final_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")

        for ann in sorted_anns:
            m = ann["segmentation"]
            img = np.empty((m.shape[0], m.shape[1], 3), dtype=np.uint8)

            for i in range(3):
                img[:, :, i] = np.random.randint(255, dtype=np.uint8)

            final_img.paste(
                Image.fromarray(img, mode="RGB"),
                (0, 0),
                Image.fromarray(np.uint8(m * 255)),
            )

        return np.array(final_img, dtype=np.uint8)

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        detect_resolution: int = 512,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method="INTER_CUBIC",
    ):
        image, output_type = self.validate_input(image, output_type)
        image, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )

        masks = self.mask_generator.generate(image)
        map = self.show_anns(masks)

        detected_map = HWC3(remove_pad(map))

        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)

        return detected_map
