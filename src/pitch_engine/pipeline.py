from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

from pitch_engine.detection import FieldDetector
from pitch_engine.video import FrameSource


@dataclass(frozen=True, slots=True)
class RunSummary:
    frames_processed: int
    frames_skipped: int
    detections_found: int
    mean_frame_overlap: float | None


class FieldPipeline:
    def __init__(self, detector: FieldDetector) -> None:
        self._detector = detector
        self._frame_bounds: Polygon | None = None

    def run(self, frames: FrameSource) -> RunSummary:
        frames_processed = 0
        detections_found = 0
        overlap_sum = 0.0

        for frame in frames:
            frames_processed += 1

            detection = self._detector.detect(frame)
            if detection is None:
                continue

            try:
                polygon = Polygon(detection.polygon)
            except ValueError:
                continue
            if not polygon.is_valid:
                continue

            bounds = self._frame_bounds_for(frame.shape)
            detections_found += 1
            overlap_sum += polygon.intersection(bounds).area

        mean_overlap = overlap_sum / detections_found if detections_found else None
        return RunSummary(
            frames_processed=frames_processed,
            frames_skipped=frames.frames_skipped,
            detections_found=detections_found,
            mean_frame_overlap=mean_overlap,
        )

    def _frame_bounds_for(self, shape: tuple[int, ...]) -> Polygon:
        if self._frame_bounds is None:
            height, width = shape[0], shape[1]
            self._frame_bounds = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        return self._frame_bounds
