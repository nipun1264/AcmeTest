"""
IMPORTANT - a real limitation found by running these tests, not by inspection:

ColorThresholdFieldDetector (ported unchanged from
FieldBoundaryAnalyzer._extract_mask/._derive_polygon_from_mask) thresholds
on GREEN, then takes the largest EXTERNAL contour of that mask. But
synthetic_generator.py fills the *entire* canvas green before drawing the
white pitch outline on top of it. So for every frame that has any green at
all, the green mask covers almost the whole frame, and its external contour
is just the frame's own border - not the white quadrilateral. The white
outline's shape and position never actually reach the returned polygon.

Net effect: a real pitch frame, a close-up with no pitch, and a noise frame
are all "detected" with near-total frame coverage. Only a fully black frame
(a real camera cut) is correctly rejected, because it has no green at all.

This is inherited prototype behavior, not something introduced while
restructuring it (Part 1/2 is architecture, not detection accuracy), so it's
left unchanged here and captured in these tests as documentation of the
current state. It's flagged for DECISIONS.md and is a natural candidate for
Part 3 ("real video feeds are noisy... handle boundaries that are missing,
obscured, or invalid") - e.g. using RETR_CCOMP to find the hole the white
outline punches in the green mask, instead of thresholding on green at all.
"""

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
    """A green frame with a pitch outline, matching synthetic_generator.py's
    "normal frame" branch (the white polyline drawn on a green canvas)."""
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
    """A fully black frame, matching the generator's "camera cut" branch."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def green_only_frame(width=1280, height=720):
    """A green frame with no pitch outline, matching the "close-up" branch."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (34, 139, 34)
    return frame


def noise_frame(width=1280, height=720):
    """A green frame with only the generator's tiny noise square (30x20 px),
    matching the "detection noise" branch. Drawn on the same full-green
    canvas the generator itself fills first."""
    frame = green_only_frame(width, height)
    pts = np.array([[10, 10], [40, 10], [40, 30], [10, 30]], np.int32)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
    return frame


def isolated_green_patch(patch_size, width=400, height=300):
    """A small green square on an otherwise BLACK background - unlike every
    frame the real generator produces, but the only way to isolate min_area
    / confidence_threshold as unit tests, since the generator never yields a
    frame where the green area is small relative to the frame."""
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
    # The one case this detector handles correctly: no green anywhere means
    # no contour at all, regardless of the full-frame-mask limitation above.
    assert ColorThresholdFieldDetector(DEFAULT).detect(blank_frame()) is None


def test_known_limitation_close_up_is_detected_as_if_it_were_the_full_pitch():
    # Documents current behavior, not desired behavior - see module docstring.
    detection = ColorThresholdFieldDetector(DEFAULT).detect(green_only_frame())
    assert detection is not None
    assert detection.coverage_ratio > 0.9


def test_known_limitation_noise_frame_is_detected_as_if_it_were_the_full_pitch():
    # Documents current behavior, not desired behavior - see module docstring.
    detection = ColorThresholdFieldDetector(DEFAULT).detect(noise_frame())
    assert detection is not None
    assert detection.coverage_ratio > 0.9


def test_min_area_rejects_a_small_isolated_green_patch():
    config = DetectorConfig(kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.0)
    small_patch = isolated_green_patch(patch_size=20)  # area 400, below min_area
    assert ColorThresholdFieldDetector(config).detect(small_patch) is None


def test_min_area_accepts_a_large_enough_isolated_green_patch():
    config = DetectorConfig(kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.0)
    big_patch = isolated_green_patch(patch_size=50)  # area 2500, above min_area
    detection = ColorThresholdFieldDetector(config).detect(big_patch)
    assert detection is not None


def test_confidence_threshold_rejects_low_frame_coverage():
    # A patch that clears min_area but covers only a sliver of a large frame.
    config = DetectorConfig(
        kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.5
    )
    patch = isolated_green_patch(patch_size=50, width=4000, height=4000)  # ratio ~0.00016
    assert ColorThresholdFieldDetector(config).detect(patch) is None


def test_confidence_threshold_accepts_high_frame_coverage():
    config = DetectorConfig(
        kind="sam_mask_v1", sport="football", min_area=1000, confidence_threshold=0.5
    )
    detection = ColorThresholdFieldDetector(config).detect(pitch_frame())
    assert detection is not None  # a real pitch frame covers nearly the whole frame


def test_the_pitch_outline_shifts_frame_to_frame_and_a_detection_still_occurs():
    # Mirrors the generator's per-frame camera-pan shift.
    for shift in (0, 20, 40, 58):
        assert ColorThresholdFieldDetector(DEFAULT).detect(pitch_frame(shift=shift)) is not None
