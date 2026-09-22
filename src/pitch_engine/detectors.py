from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from pitch_engine.config import DetectorConfig
from pitch_engine.detection import Detection, FieldDetector


class ColorThresholdFieldDetector:
    _LOWER_GREEN = np.array([35, 40, 40])
    _UPPER_GREEN = np.array([85, 255, 255])

    def __init__(self, config: DetectorConfig) -> None:
        self._min_area = config.min_area
        self._confidence_threshold = config.confidence_threshold

    def detect(self, frame: NDArray[np.uint8]) -> Detection | None:
        mask = self._extract_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area <= self._min_area:
            return None

        points = largest.reshape(-1, 2)
        if len(points) < 3:
            return None

        frame_area = frame.shape[0] * frame.shape[1]
        coverage_ratio = float(area) / frame_area
        if coverage_ratio < self._confidence_threshold:
            return None

        return Detection(polygon=points.astype(np.int32), coverage_ratio=coverage_ratio)

    def _extract_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self._LOWER_GREEN, self._UPPER_GREEN)


_REGISTRY: dict[str, Callable[[DetectorConfig], FieldDetector]] = {
    "sam_mask_v1": ColorThresholdFieldDetector,
}


def build_detector(config: DetectorConfig) -> FieldDetector:
    return _REGISTRY[config.kind](config)
