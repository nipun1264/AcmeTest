import cv2
import numpy as np
import pytest

from pitch_engine.config import DetectorConfig
from pitch_engine.detection import FieldDetector
from pitch_engine.detectors import ColorThresholdFieldDetector, build_detector

DEFAULT = DetectorConfig(
    kind="sam_mask_v1",
    sport="football",
    min_area=1000,
    confidence_threshold=0.05,
)


def pitch_frame(width=1280, height=720, shift=0):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    pts = np.array(
        [
            [100 + shift, 100],
            [1180 - shift, 100],
            [1230 - shift, 620],
            [50 + shift, 620],
        ],
        np.int32,
    )
    cv2.polylines(frame, [pts], True, (255, 255, 255), 5)
    return frame


def blank_frame(width=1280, height=720):
    return np.zeros((height, width, 3), dtype=np.uint8)


def green_only_frame(width=1280, height=720):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    return frame


def noise_frame(width=1280, height=720):
    frame = green_only_frame(width, height)
    pts = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], np.int32)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
    return frame


def isolated_green_patch(patch_size, width=400, height=300):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[10 : 10 + patch_size, 10 : 10 + patch_size] = (34, 139, 34)
    return frame


def test_built_detector_satisfies_the_protocol():
    assert isinstance(build_detector(DEFAULT), FieldDetector)


def test_detects_something_in_a_normal_pitch_frame():
    detection = ColorThresholdFieldDetector(DEFAULT).detect(pitch_frame())
    assert detection is not None
    assert len(detection.polygon) >= 3
    assert 0.0 < detection.coverage_ratio <= 1.0


def test_a_camera_cut_with_no_green_is_correctly_rejected():
    assert ColorThresholdFieldDetector(DEFAULT).detect(blank_frame()) is None


def test_close_up_is_detected_as_if_it_were_the_full_pitch():
    detection = ColorThresholdFieldDetector(DEFAULT).detect(green_only_frame())
    assert detection is not None
    assert detection.coverage_ratio > 0.9


def test_noise_frame_is_detected_as_if_it_were_the_full_pitch():
    detection = ColorThresholdFieldDetector(DEFAULT).detect(noise_frame())
    assert detection is not None
    assert detection.coverage_ratio > 0.9


def test_min_area_rejects_a_small_isolated_green_patch():
    config = DetectorConfig(kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.0)
    small_patch = isolated_green_patch(patch_size=20)
    assert ColorThresholdFieldDetector(config).detect(small_patch) is None


def test_min_area_accepts_a_large_enough_isolated_green_patch():
    config = DetectorConfig(kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.0)
    big_patch = isolated_green_patch(patch_size=50)
    detection = ColorThresholdFieldDetector(config).detect(big_patch)
    assert detection is not None


def test_confidence_threshold_rejects_low_frame_coverage():
    config = DetectorConfig(
        kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.5
    )
    patch = isolated_green_patch(patch_size=50, width=4000, height=4000)
    assert ColorThresholdFieldDetector(config).detect(patch) is None


def test_confidence_threshold_accepts_high_frame_coverage():
    config = DetectorConfig(
        kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.5
    )
    detection = ColorThresholdFieldDetector(config).detect(pitch_frame())
    assert detection is not None


def test_the_pitch_outline_shifts_frame_to_frame_and_a_detection_still_occurs():
    for shift in (0, 20, 40, 58):
        assert ColorThresholdFieldDetector(DEFAULT).detect(pitch_frame(shift=shift)) is not None
