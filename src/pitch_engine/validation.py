from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def touches_all_edges(polygon: NDArray, frame_shape: tuple[int, ...], margin: int = 2) -> bool:
    # The mask picks the outer boundary of the whole green region, not the
    # drawn pitch outline, whenever the outline forms a closed loop inside
    # the frame - see DECISIONS.md. That spurious detection's bounding box
    # always touches all four edges; a genuine inset boundary never does.
    height, width = frame_shape[0], frame_shape[1]
    x_min, y_min = polygon.min(axis=0)
    x_max, y_max = polygon.max(axis=0)
    return bool(
        x_min <= margin
        and y_min <= margin
        and x_max >= width - 1 - margin
        and y_max >= height - 1 - margin
    )
