from .detector import DWPoseDetector

from typing import Optional
import numpy as np
import torch


class DWPoseProcessor:
    def __init__(
        self,
        bbox_detector="yolox_l.onnx",
        pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
        device: Optional[str] = None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if bbox_detector == "yolox_l.onnx":
            yolo_repo = "yzd-v/DWPose"
        elif "yolox" in bbox_detector:
            yolo_repo = "hr16/yolox-onnx"
        elif "yolo_nas" in bbox_detector:
            yolo_repo = "hr16/yolo-nas-fp16"
        else:
            raise NotImplementedError(f"Download mechanism for {bbox_detector}")

        if pose_estimator == "dw-ll_ucoco_384.onnx":
            pose_repo = "yzd-v/DWPose"
        elif pose_estimator.endswith(".onnx"):
            pose_repo = "hr16/UnJIT-DWPose"
        elif pose_estimator.endswith(".torchscript.pt"):
            pose_repo = "hr16/DWPose-TorchScript-BatchSize5"
        else:
            raise NotImplementedError(f"Download mechanism for {pose_estimator}")

        model = DWPoseDetector.from_pretrained(
            pose_repo,
            yolo_repo,
            det_filename=bbox_detector,
            pose_filename=pose_estimator,
            torchscript_device=device,
        )

        self.model = model

    def __call__(
        self,
        image: np.ndarray,
        detect_body: bool = True,
        detect_hand: bool = False,
        detect_face: bool = False,
        resolution: int = 512,
    ) -> np.ndarray:
        """
        Args:
            image: Image to process. Can be a single image or a batch of images.
                Image should be in the format (H, W, C) or (B, H, W, C).
            detect_body: Whether to predict body keypoints. Defaults to True.
            detect_hand: Whether to predict hand keypoints. Defaults to False.
            detect_face: Whether to predict face keypoints. Defaults to False.
            resolution: Resolution to use for detection. Defaults to 512.

        Raises:
            ValueError: If the image shape is invalid.

        Returns:
            np.ndarray: Processed image with keypoints drawn.
            Shape is (B, H, W, C).
        """

        import numpy as np

        if len(image.shape) == 3:
            image = image[np.newaxis]
        elif len(image.shape) == 4:
            pass
        else:
            raise ValueError(f"Invalid shape {image.shape}")

        batch_size = image.shape[0]
        pose_images = None

        for i, image in enumerate(image):
            pose_image, openpose_dict = self.model(
                image,
                output_type="np",
                detect_resolution=resolution,
                include_hand=detect_hand,
                include_face=detect_face,
                include_body=detect_body,
                image_and_json=True,
            )

            if pose_images is None:
                pose_images = np.zeros(
                    (batch_size, *pose_image.shape), dtype=pose_image.dtype
                )
            pose_images[i] = pose_image

        return pose_images, openpose_dict
