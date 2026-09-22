import numpy as np
import pytest
from shapely.geometry import Polygon

import pitch_engine.pipeline as pipeline_module
from pitch_engine.pipeline import FieldPipeline
from pitch_engine.detection import Detection


class Frames(list):
    """A list that also satisfies what FieldPipeline.run() needs from a real
    FrameSource: a .frames_skipped count (FrameSource itself is exercised
    separately in test_video.py; this just isolates the pipeline's own
    bookkeeping from frame sampling)."""

    frames_skipped = 0


class StubDetector:
    """Returns a fixed sequence of detections/None, one per call, ignoring
    the frame's actual content - lets the pipeline's own bookkeeping be
    tested independently of any real detection logic."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def detect(self, frame):
        result = self._results[self.calls]
        self.calls += 1
        return result


def make_frame(height=100, width=100):
    return np.zeros((height, width, 3), dtype=np.uint8)


def square_detection(x0, y0, size, height=100, width=100):
    pts = np.array([[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]])
    area_ratio = (size * size) / (height * width)
    return Detection(polygon=pts, coverage_ratio=area_ratio)


def test_counts_frames_and_detections():
    detector = StubDetector([
        square_detection(0, 0, 40),
        None,
        square_detection(10, 10, 20),
        None,
        None,
    ])
    frames = Frames(make_frame() for _ in range(5))
    summary = FieldPipeline(detector).run(frames)

    assert summary.frames_processed == 5
    assert summary.detections_found == 2
    assert summary.mean_frame_overlap == pytest.approx((40 * 40 + 20 * 20) / 2)


def test_no_detections_reports_none_not_a_crash():
    detector = StubDetector([None, None, None])
    summary = FieldPipeline(detector).run(Frames(make_frame() for _ in range(3)))
    assert summary.detections_found == 0
    assert summary.mean_frame_overlap is None


def test_frames_skipped_is_read_from_the_frame_source():
    class FramesWithSkipCount(Frames):
        frames_skipped = 7

    detector = StubDetector([None, None])
    frames = FramesWithSkipCount([make_frame(), make_frame()])
    summary = FieldPipeline(detector).run(frames)
    assert summary.frames_skipped == 7


def test_frame_bounds_polygon_is_built_once_not_once_per_frame(monkeypatch):
    # The prototype rebuilt the frame-bounds rectangle on every single frame.
    # This proves the fix: with N detections, Polygon() should be called
    # N + 1 times (once per detected polygon, plus exactly one for the
    # shared frame bounds) - not 2N times.
    call_count = {"n": 0}
    real_polygon = pipeline_module.Polygon

    def counting_polygon(*args, **kwargs):
        call_count["n"] += 1
        return real_polygon(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "Polygon", counting_polygon)

    detector = StubDetector([
        square_detection(0, 0, 10),
        square_detection(5, 5, 10),
        square_detection(20, 20, 10),
    ])
    frames = Frames(make_frame() for _ in range(3))
    FieldPipeline(detector).run(frames)

    assert call_count["n"] == 3 + 1


def test_invalid_polygon_is_skipped_without_crashing():
    # Fewer than 3 distinct usable points -> not a valid polygon.
    degenerate = Detection(polygon=np.array([[0, 0], [0, 0]]), coverage_ratio=0.1)
    detector = StubDetector([degenerate])
    summary = FieldPipeline(detector).run(Frames([make_frame()]))
    assert summary.detections_found == 0
    assert summary.mean_frame_overlap is None
