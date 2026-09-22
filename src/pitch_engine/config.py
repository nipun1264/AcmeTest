"""Validated configuration, replacing synthetic_field_prototype.py's raw CONFIG dict.

Design rules:
- Every field is required: a missing value is an error, never a silent default.
- Unknown keys are rejected (a typo in the original dict would previously be
  ignored silently; here it fails at load time instead).
- Strict types: "1000" is not accepted where an int is expected.
- Everything is validated once, at load time, before any frame is touched.

Mapping from the prototype's CONFIG dict:
  video_path                    -> video.path
  target_fps                    -> video.target_fps  (still informational only;
                                    see DECISIONS.md - nothing cross-checks it
                                    against the file's real frame rate yet)
  field_detector.type           -> detector.kind
  field_detector.sport          -> detector.sport
  field_detector.min_area       -> detector.min_area
  confidence_threshold          -> detector.confidence_threshold (moved under
                                    detector: it's a detection-quality knob,
                                    not a video-level one; the prototype read
                                    it into self.threshold but never used it -
                                    it's now wired into the detector as an
                                    area-coverage gate, see detectors.py)
  crop_search                   -> crop_search (kept, but still unused by this
                                    pipeline - the README says the downstream
                                    crop step isn't built yet; kept so a
                                    config matching the original shape still
                                    validates)
  debug_mode                    -> debug_mode
  (new)                         -> sampling.frame_stride (Part 2: how many
                                    frames to skip between inspected ones)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ConfigError(Exception):
    """The configuration could not be read, parsed, or validated."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class VideoConfig(_StrictModel):
    path: str = Field(min_length=1)
    target_fps: int = Field(gt=0, le=240)


class DetectorConfig(_StrictModel):
    # To add a detector for another sport/model: add its name here, write the
    # class, register it in detectors.py. Nothing else in the pipeline changes.
    kind: Literal["sam_mask_v1"]
    sport: str = Field(min_length=1)
    min_area: float = Field(gt=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class SamplingConfig(_StrictModel):
    # 1 = inspect every frame, matching the prototype's behavior exactly.
    frame_stride: int = Field(ge=1, le=1000)


class CropSearchConfig(_StrictModel):
    aspect_ratio: str = Field(pattern=r"^\d+:\d+$")
    padding_px: int = Field(ge=0)


class AppConfig(_StrictModel):
    video: VideoConfig
    detector: DetectorConfig
    sampling: SamplingConfig
    crop_search: CropSearchConfig
    debug_mode: bool


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid configuration in {path} ({exc.error_count()} problem(s)):"]
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {where}: {err['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> AppConfig:
    """Read, parse and validate a JSON config file, or raise ConfigError."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} must contain a JSON object at the top level, "
            f"got {type(data).__name__}"
        )

    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(path, exc)) from exc
