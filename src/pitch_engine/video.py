from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np
from numpy.typing import NDArray

from pitch_engine.errors import FatalError


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, NDArray[np.uint8]]: ...
    def grab(self) -> bool: ...
    def release(self) -> None: ...


class VideoOpenError(FatalError):
    pass


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
