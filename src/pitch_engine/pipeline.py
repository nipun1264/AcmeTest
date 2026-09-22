from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import Polygon

from pitch_engine.detection import FieldDetector
from pitch_engine.errors import FatalError
from pitch_engine.validation import touches_all_edges
from pitch_engine.video import FrameSource

logger = logging.getLogger(__name__)

PROGRESS_INTERVAL = 200
MAX_CONSECUTIVE_FRAME_ERRORS = 10


class TooManyFrameFailures(FatalError):
    pass


@dataclass(frozen=True, slots=True)
class RunSummary:
    frames_processed: int
    frames_skipped: int
    frames_with_no_candidate: int
    frames_with_rejected_detection: int
    frames_with_processing_error: int
    valid_detections: int
    mean_frame_overlap: float | None


class FieldPipeline:
    def __init__(self, detector: FieldDetector) -> None:
        self._detector = detector
        self._frame_bounds: Polygon | None = None

    def run(
        self,
        frames: FrameSource,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> RunSummary:
        frames_processed = 0
        no_candidate = 0
        rejected = 0
        processing_errors = 0
        consecutive_errors = 0
        valid_detections = 0
        overlap_sum = 0.0

        for frame in frames:
            frames_processed += 1

            try:
                detection = self._detector.detect(frame)
            except Exception:
                logger.warning("detector raised on frame %d", frames_processed, exc_info=True)
                processing_errors += 1
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_FRAME_ERRORS:
                    raise TooManyFrameFailures(
                        f"{consecutive_errors} consecutive frame failures, "
                        f"stopping at frame {frames_processed}"
                    )
                self._report_progress(frames_processed, frames, valid_detections, on_progress)
                continue
            consecutive_errors = 0

            if detection is None:
                no_candidate += 1
                self._report_progress(frames_processed, frames, valid_detections, on_progress)
                continue

            try:
                polygon = Polygon(detection.polygon)
            except ValueError:
                rejected += 1
                self._report_progress(frames_processed, frames, valid_detections, on_progress)
                continue
            if not polygon.is_valid or touches_all_edges(detection.polygon, frame.shape):
                rejected += 1
                self._report_progress(frames_processed, frames, valid_detections, on_progress)
                continue

            bounds = self._frame_bounds_for(frame.shape)
            valid_detections += 1
            overlap_sum += polygon.intersection(bounds).area

            self._report_progress(frames_processed, frames, valid_detections, on_progress)

        mean_overlap = overlap_sum / valid_detections if valid_detections else None
        summary = RunSummary(
            frames_processed=frames_processed,
            frames_skipped=frames.frames_skipped,
            frames_with_no_candidate=no_candidate,
            frames_with_rejected_detection=rejected,
            frames_with_processing_error=processing_errors,
            valid_detections=valid_detections,
            mean_frame_overlap=mean_overlap,
        )
        logger.info("run complete: %s", summary)
        return summary

    def _frame_bounds_for(self, shape: tuple[int, ...]) -> Polygon:
        if self._frame_bounds is None:
            height, width = shape[0], shape[1]
            self._frame_bounds = Polygon([(0, 0), (width, 0), (width, height), (0, height)])
        return self._frame_bounds

    def _report_progress(
        self,
        frames_processed: int,
        frames: FrameSource,
        valid_detections: int,
        on_progress: Callable[[int, int, int], None] | None,
    ) -> None:
        # Runs at every PROGRESS_INTERVAL-th frame processed, regardless of
        # what that frame's own outcome was - a long stretch of rejected or
        # no-candidate frames must not make an unattended run look stalled.
        if frames_processed % PROGRESS_INTERVAL != 0:
            return
        logger.info(
            "progress: %d frames processed, %d valid detections",
            frames_processed,
            valid_detections,
        )
        if on_progress is not None:
            try:
                on_progress(frames_processed, frames.frames_skipped, valid_detections)
            except Exception:
                # A broken progress callback (e.g. the reporting client)
                # must never take the actual video processing down with it.
                logger.warning("on_progress callback raised", exc_info=True)
