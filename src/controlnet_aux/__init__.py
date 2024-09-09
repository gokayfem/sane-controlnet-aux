import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .dwpose import DWPoseDetector  # noqa
from .canny import CannyDetector  # noqa

__all__ = ["DWPoseDetector", "CannyDetector"]
