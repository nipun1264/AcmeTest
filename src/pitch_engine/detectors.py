"""Concrete detectors and the factory that builds one from config.

ColorThresholdFieldDetector below is a direct port of
FieldBoundaryAnalyzer._extract_mask + ._derive_polygon_from_mask from
synthetic_field_prototype.py. Behavior is unchanged except:
  - it never silently swallows unexpected exceptions (the prototype's
    `except Exception: pass` is gone; Part 3 will decide precisely which
    per-frame failures are recoverable vs fatal)
  - confidence_threshold, which the prototype read into self.threshold but
    never used, now gates acceptance by area-coverage ratio
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from pitch_engine.config import DetectorConfig
from pitch_engine.detection import Detection, FieldDetector


class ColorThresholdFieldDetector:
    """Football-pitch detector: green-color mask -> largest contour -> polygon.

    A different sport or a real segmentation model plugs in here as another
    class implementing the same detect() method; nothing else changes.

    KNOWN LIMITATION (inherited unchanged from the prototype, verified by
    tests/test_detectors.py): synthetic_generator.py fills the whole frame
    green before drawing the white pitch outline on top of it, so the green
    mask covers almost the entire frame for ANY frame that has green in it -
    a real pitch view, a close-up, and a noise frame all look the same to
    RETR_EXTERNAL here. Only a frame with no green at all (a true camera
    cut) is rejected correctly. See DECISIONS.md.
    """

    # HSV green band, unchanged from the prototype.
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
    """Build the detector named in the config."""
    return _REGISTRY[config.kind](config)
