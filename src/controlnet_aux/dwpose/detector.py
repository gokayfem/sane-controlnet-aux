# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.


import json
import torch
import numpy as np
from . import util
from .body import BodyResult, Keypoint
from .types import PoseResult
from controlnet_aux.utils import (
    HWC3,
    resize_image_with_pad,
    custom_hf_download,
)
from PIL import Image

from typing import Literal, Tuple, List, Union, Optional


def draw_poses(
    poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True
):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(
    json_string: str, normalize_coords: bool = False
) -> Tuple[List[PoseResult], int, int]:
    """Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.
        normalize_coords: Whether to normalize coordinates of each keypoint by canvas height/width.
                          `draw_pose` only accepts normalized keypoints. Set this param to True if
                          the input coords are not normalized.

    Returns:
        poses
        canvas_height
        canvas_width
    """
    pose_json = json.loads(json_string)
    height = pose_json["canvas_height"]
    width = pose_json["canvas_width"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def decompress_keypoints(
        numbers: Optional[List[float]],
    ) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    return (
        [
            PoseResult(
                body=BodyResult(
                    keypoints=decompress_keypoints(pose.get("pose_keypoints_2d"))
                ),
                left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
            for pose in pose_json["people"]
        ],
        height,
        width,
    )


def encode_poses_as_dict(
    poses: List[PoseResult], canvas_height: int, canvas_width: int
) -> str:
    """Encode the pose as a dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """

    def compress_keypoints(
        keypoints: Union[List[Keypoint], None],
    ) -> Union[List[float], None]:
        if not keypoints:
            return None

        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        "people": [
            {
                "pose_keypoints_2d": compress_keypoints(pose.body.keypoints),
                "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d": compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        "canvas_height": canvas_height,
        "canvas_width": canvas_width,
    }


class DWPoseDetector:
    """A class for detecting human poses in images using the Dwpose model."""

    def __init__(self, det_model_path=None, pose_model_path=None, device="cpu"):
        from .wholebody import Wholebody

        self.pose_estimation = Wholebody(det_model_path, pose_model_path, device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_or_path="yzd-v/DWPose",
        pretrained_det_model_or_path="yzd-v/DWPose",
        det_filename="yolox_l.onnx",
        pose_filename="dw-ll_ucoco_384.onnx",
        device="cuda",
    ):
        pretrained_det_model_or_path = (
            pretrained_det_model_or_path or pretrained_model_or_path
        )
        det_filename = det_filename or "yolox_l.onnx"
        pose_filename = pose_filename or "dw-ll_ucoco_384.onnx"

        det_model_path = custom_hf_download(pretrained_det_model_or_path, det_filename)
        pose_model_path = custom_hf_download(pretrained_model_or_path, pose_filename)

        return cls(det_model_path, pose_model_path, device=device)

    def detect_poses(self, oriImg) -> List[PoseResult]:
        from .wholebody import Wholebody

        with torch.no_grad():
            keypoints_info = self.pose_estimation(oriImg.copy())
            return Wholebody.format_result(keypoints_info)

    def validate_input(self, input_image, output_type):
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
        detect_resolution: int = 512,
        detect_body: bool = True,
        detect_hand: bool = False,
        detect_face: bool = False,
        output_type: Literal["pil", "np"] = "pil",
        upscale_method="INTER_CUBIC",
        return_pose_dict: bool = False,
        **kwargs,
    ):
        """
        Args:
            image: Image to process.
            detect_resolution: Resolution to use for detection. Defaults to 512.
            detect_body: Whether to predict body keypoints. Defaults to True.
            detect_hand: Whether to predict hand keypoints. Defaults to False.
            detect_face: Whether to predict face keypoints. Defaults to False.
            output_type: Type of the output. Defaults to "pil".
            upscale_method: The interpolation method to use for upscaling the image.
            return_pose_dict: Whether to return the pose dictionary in addition to the image.
            Defaults to False.
        """
        image, output_type = self.validate_input(image, output_type, **kwargs)
        image, remove_pad = resize_image_with_pad(
            image, detect_resolution, upscale_method
        )
        input_height, input_width = image.shape[:2]

        poses = self.detect_poses(image)
        canvas = draw_poses(
            poses,
            input_height,
            input_width,
            draw_body=detect_body,
            draw_face=detect_face,
            draw_hand=detect_hand,
        )

        pose_image = HWC3(remove_pad(canvas))

        if output_type == "pil":
            pose_image = Image.fromarray(pose_image)

        if return_pose_dict:
            return (
                pose_image,
                encode_poses_as_dict(poses, input_height, input_width),
            )

        return pose_image
