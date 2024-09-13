import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .dwpose import DWPoseDetector  # noqa
from .canny import CannyDetector  # noqa
from .lineart import LineartDetector  # noqa
from .hed import HEDDetector  # noqa
from .mlsd import MLSDDetector  # noqa
from .scribble import ScribbleDetector  # noqa
from .pidi import PiDiDetector  # noqa
from .teed import TEEDDetector  # noqa
from .midas import MidasDetector  # noqa
from .sam import SamDetector  # noqa
from .zoe import ZoeDetector  # noqa

__all__ = [
    "DWPoseDetector",
    "CannyDetector",
    "LineartDetector",
    "HEDDetector",
    "MLSDDetector",
    "ScribbleDetector",
    "PiDiDetector",
    "TEEDDetector",
    "MidasDetector",
    "SamDetector",
    "ZoeDetector",
]
