import numpy as np
import pytest
from shapely.geometry import Polygon

import pitch_engine.pipeline as pipeline_module
from pitch_engine.pipeline import FieldPipeline, TooManyFrameFailures
from pitch_engine.detection import Detection


class Frames(list):
    frames_skipped = 0


class StubDetector:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def detect(self, frame):
        result = self._results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
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
    assert summary.valid_detections == 2
    assert summary.frames_with_no_candidate == 3
    assert summary.mean_frame_overlap == pytest.approx((40 * 40 + 20 * 20) / 2)


def test_no_detections_reports_none_not_a_crash():
    detector = StubDetector([None, None, None])
    summary = FieldPipeline(detector).run(Frames(make_frame() for _ in range(3)))
    assert summary.valid_detections == 0
    assert summary.frames_with_no_candidate == 3
    assert summary.mean_frame_overlap is None


def test_frames_skipped_is_read_from_the_frame_source():
    class FramesWithSkipCount(Frames):
        frames_skipped = 7

    detector = StubDetector([None, None])
    frames = FramesWithSkipCount([make_frame(), make_frame()])
    summary = FieldPipeline(detector).run(frames)
    assert summary.frames_skipped == 7


def test_frame_bounds_polygon_is_built_once_not_once_per_frame(monkeypatch):
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
    degenerate = Detection(polygon=np.array([[0, 0], [0, 0]]), coverage_ratio=0.1)
    detector = StubDetector([degenerate])
    summary = FieldPipeline(detector).run(Frames([make_frame()]))
    assert summary.valid_detections == 0
    assert summary.frames_with_rejected_detection == 1


def test_a_detection_touching_all_frame_edges_is_rejected():
    height, width = 100, 100
    full_frame = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    detector = StubDetector([Detection(polygon=full_frame, coverage_ratio=0.999)])
    summary = FieldPipeline(detector).run(Frames([make_frame(height, width)]))
    assert summary.valid_detections == 0
    assert summary.frames_with_rejected_detection == 1


def test_processing_error_is_counted_and_does_not_stop_the_run():
    detector = StubDetector([RuntimeError("boom"), None, square_detection(0, 0, 10)])
    frames = Frames(make_frame() for _ in range(3))
    summary = FieldPipeline(detector).run(frames)
    assert summary.frames_with_processing_error == 1
    assert summary.frames_with_no_candidate == 1
    assert summary.valid_detections == 1


def test_too_many_consecutive_processing_errors_is_fatal():
    n = pipeline_module.MAX_CONSECUTIVE_FRAME_ERRORS
    detector = StubDetector([RuntimeError("boom")] * n)
    frames = Frames(make_frame() for _ in range(n))
    with pytest.raises(TooManyFrameFailures):
        FieldPipeline(detector).run(frames)


def test_a_recovered_frame_resets_the_consecutive_error_count():
    n = pipeline_module.MAX_CONSECUTIVE_FRAME_ERRORS
    results = [RuntimeError("boom")] * (n - 1) + [None] + [RuntimeError("boom")] * (n - 1)
    detector = StubDetector(results)
    frames = Frames(make_frame() for _ in range(len(results)))
    summary = FieldPipeline(detector).run(frames)
    assert summary.frames_with_processing_error == 2 * (n - 1)


def test_on_progress_callback_is_invoked_at_the_progress_interval():
    n = pipeline_module.PROGRESS_INTERVAL * 2
    detector = StubDetector([None] * n)
    frames = Frames(make_frame() for _ in range(n))
    calls = []

    FieldPipeline(detector).run(frames, on_progress=lambda *args: calls.append(args))

    assert calls == [
        (pipeline_module.PROGRESS_INTERVAL, 0, 0),
        (pipeline_module.PROGRESS_INTERVAL * 2, 0, 0),
    ]


def test_a_broken_on_progress_callback_does_not_crash_the_run():
    n = pipeline_module.PROGRESS_INTERVAL
    detector = StubDetector([None] * n)
    frames = Frames(make_frame() for _ in range(n))

    def bad_callback(*args):
        raise RuntimeError("reporting is down")

    summary = FieldPipeline(detector).run(frames, on_progress=bad_callback)

    assert summary.frames_processed == n
