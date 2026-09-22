"""Frame access with sampling (Part 2: processing efficiency).

synthetic_field_prototype.py called cap.read() for every single frame:

    while True:
        ret, frame = cap.read()
        ...

read() always fully decodes the frame. FrameSource below instead calls the
much cheaper grab() (which advances the stream without decoding) for frames
we're not going to inspect, and only decodes with read() for the ones we
keep. With frame_stride=1 this behaves exactly like the prototype - every
frame is read(). With frame_stride=N>1, processing cost scales with
frames actually inspected (roughly total_frames / N), not the full file.
"""

from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np
from numpy.typing import NDArray


class CaptureLike(Protocol):
    """The subset of cv2.VideoCapture's interface FrameSource depends on.

    Kept as a narrow Protocol (rather than importing cv2.VideoCapture
    directly) so tests can supply a fake capture with no real video file or
    OpenCV video I/O involved.
    """

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, NDArray[np.uint8]]: ...
    def grab(self) -> bool: ...
    def release(self) -> None: ...


class VideoOpenError(Exception):
    """The video source could not be opened for reading."""


class FrameSource:
    def __init__(self, capture: CaptureLike, frame_stride: int) -> None:
        if not capture.isOpened():
            raise VideoOpenError("Could not open the video source")
        self._capture = capture
        self._frame_stride = frame_stride
        self.frames_skipped = 0

    def __iter__(self) -> Iterator[NDArray[np.uint8]]:
        index = 0
        try:
            while True:
                if index % self._frame_stride == 0:
                    ok, frame = self._capture.read()
                    if not ok:
                        return
                    yield frame
                else:
                    if not self._capture.grab():
                        return
                    self.frames_skipped += 1
                index += 1
        finally:
            self._capture.release()


def open_video(path: str) -> CaptureLike:
    import cv2

    return cv2.VideoCapture(path)
