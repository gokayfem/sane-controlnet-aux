import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .dwpose import DWPoseDetector  # noqa
from .canny import CannyDetector  # noqa
from .lineart import LineartDetector  # noqa
from .hed import HEDDetector  # noqa
from .mlsd import MLSDDetector  # noqa
from .scribble import ScribbleDetector  # noqa

__all__ = [
    "DWPoseDetector",
    "CannyDetector",
    "LineartDetector",
    "HEDDetector",
    "MLSDDetector",
    "ScribbleDetector",
]
