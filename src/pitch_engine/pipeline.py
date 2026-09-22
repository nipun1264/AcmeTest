"""Pipeline core: runs a detector over sampled frames and aggregates the result.

Two behavioral fixes versus FieldBoundaryAnalyzer.process_video, both
Part 2 (efficiency) work:

1. Frame sampling: iterating `frames` (a FrameSource) already skips frames
   cheaply instead of decoding every one - see video.py.
2. The prototype rebuilt an identical `outer_boundary` Polygon on every
   single frame:
       outer_boundary = Polygon([(0, 0), (1280, 0), (1280, 720), (0, 720)])
   even though the frame size never changes mid-run. _frame_bounds_for()
   below builds it once and reuses it - the same discipline the assignment
   asks for on "any other expensive calculation performed more than once".

Knows nothing about config files, the CLI, or which detector it was given.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon

from pitch_engine.detection import FieldDetector
from pitch_engine.video import FrameSource


@dataclass(frozen=True, slots=True)
class RunSummary:
    frames_processed: int  # frames actually decoded and inspected
    frames_skipped: int  # frames skipped via a cheap grab() (frame_stride > 1)
    detections_found: int
    mean_frame_overlap: float | None  # mean of (polygon intersect frame bounds) area


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

            polygon = Polygon(detection.polygon)
            if not polygon.is_valid:
                # A malformed contour. Part 3 will classify and count this
                # explicitly instead of just dropping it silently.
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
