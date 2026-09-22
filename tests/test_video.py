import numpy as np
import pytest

from pitch_engine.video import FrameSource, VideoOpenError


class FakeCapture:
    """A CaptureLike stand-in: no real video file or OpenCV I/O involved."""

    def __init__(self, num_frames, opened=True):
        self._num_frames = num_frames
        self._opened = opened
        self._index = 0
        self.read_calls = 0
        self.grab_calls = 0
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        self.read_calls += 1
        if self._index >= self._num_frames:
            return False, None
        frame = np.full((2, 2, 3), self._index, dtype=np.uint8)
        self._index += 1
        return True, frame

    def grab(self):
        self.grab_calls += 1
        if self._index >= self._num_frames:
            return False
        self._index += 1
        return True

    def release(self):
        self.released = True


def test_stride_one_reads_every_frame_like_the_prototype():
    capture = FakeCapture(num_frames=10)
    source = FrameSource(capture, frame_stride=1)
    frames = list(source)
    assert len(frames) == 10
    assert capture.read_calls == 11  # 10 successful + 1 that signals the end
    assert capture.grab_calls == 0
    assert source.frames_skipped == 0


def test_stride_skips_frames_with_the_cheap_grab_call():
    capture = FakeCapture(num_frames=10)
    source = FrameSource(capture, frame_stride=3)
    frames = list(source)
    # Indices 0, 3, 6, 9 are read (4 frames); the rest are grabbed (6 frames).
    assert len(frames) == 4
    assert source.frames_skipped == 6
    # 6 successful skips + 1 final grab() past the end that signals the
    # stream is over (mirrors read_calls == 11 in the stride=1 test above).
    assert capture.grab_calls == 7


def test_release_is_always_called():
    capture = FakeCapture(num_frames=3)
    list(FrameSource(capture, frame_stride=1))
    assert capture.released is True


def test_unopened_capture_raises_immediately():
    capture = FakeCapture(num_frames=5, opened=False)
    with pytest.raises(VideoOpenError):
        FrameSource(capture, frame_stride=1)
