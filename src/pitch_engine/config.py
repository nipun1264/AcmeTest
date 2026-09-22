from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pitch_engine.errors import FatalError


class ConfigError(FatalError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class VideoConfig(_StrictModel):
    path: str = Field(min_length=1)
    target_fps: int = Field(gt=0, le=240)


class DetectorConfig(_StrictModel):
    kind: Literal["sam_mask_v1"]
    sport: str = Field(min_length=1)
    min_area: float = Field(gt=0)
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class SamplingConfig(_StrictModel):
    frame_stride: int = Field(ge=1, le=1000)


class CropSearchConfig(_StrictModel):
    aspect_ratio: str = Field(pattern=r"^\d+:\d+$")
    padding_px: int = Field(ge=0)


class ReportingConfig(_StrictModel):
    base_url: str = Field(min_length=1)
    enabled: bool = True


class AppConfig(_StrictModel):
    video: VideoConfig
    detector: DetectorConfig
    sampling: SamplingConfig
    crop_search: CropSearchConfig
    reporting: ReportingConfig
    debug_mode: bool


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"Invalid configuration in {path} ({exc.error_count()} problem(s)):"]
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {where}: {err['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> AppConfig:
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
