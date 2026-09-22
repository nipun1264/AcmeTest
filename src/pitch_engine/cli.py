from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pitch_engine.config import ConfigError, load_config
from pitch_engine.detectors import build_detector
from pitch_engine.errors import FatalError
from pitch_engine.pipeline import FieldPipeline
from pitch_engine.video import FrameSource, VideoOpenError, open_video

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_VIDEO_NOT_FOUND = 3
EXIT_PIPELINE_FAILED = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitch-engine")
    parser.add_argument("--config", required=True, help="Path to the JSON config file")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    video_path = Path(config.video.path)
    if not video_path.exists():
        print(
            f"error: video file not found: {video_path}\n"
            "  Generate it first, e.g.: python synthetic_generator.py",
            file=sys.stderr,
        )
        return EXIT_VIDEO_NOT_FOUND

    try:
        capture = open_video(str(video_path))
        frames = FrameSource(capture, config.sampling.frame_stride)
    except VideoOpenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VIDEO_NOT_FOUND

    pipeline = FieldPipeline(build_detector(config.detector))
    try:
        summary = pipeline.run(frames)
    except FatalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_PIPELINE_FAILED

    print(summary)
    return EXIT_OK
