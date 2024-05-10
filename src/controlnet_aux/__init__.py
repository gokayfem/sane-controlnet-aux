import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .dwpose import DWPoseProcessor # noqa: E402

__all__ = ["DWPoseProcessor"]
