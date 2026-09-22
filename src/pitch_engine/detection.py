from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class Detection(NamedTuple):
    polygon: NDArray[np.int32]
    coverage_ratio: float


@runtime_checkable
class FieldDetector(Protocol):
    def detect(self, frame: NDArray[np.uint8]) -> Detection | None: ...
