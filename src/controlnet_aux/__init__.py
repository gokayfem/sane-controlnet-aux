import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .dwpose import DWPoseDetector  # noqa

__all__ = ["DWPoseDetector"]
