"""The one seam that varies: how a field boundary is detected in a frame.

field_detector.type in the prototype's CONFIG ("sam_mask_v1") already hints
at this: production will eventually swap the color-threshold placeholder for
a real segmentation model, and other sports need a different detector
entirely. The pipeline only knows this Protocol - it never imports a
concrete detector class.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class Detection(NamedTuple):
    """One accepted field-boundary detection in a single frame."""

    polygon: NDArray[np.int32]  # (N, 2) pixel-coordinate vertices, N >= 3
    coverage_ratio: float  # polygon area / frame area, in (0, 1]


@runtime_checkable
class FieldDetector(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> Detection | None:
        """Return the field-boundary detection for one BGR frame, or None.

        None means "no boundary found in this frame" - a camera cut, a
        close-up with no pitch visible, or a detection too weak/small to
        trust. It is not an error.
        """
        ...
