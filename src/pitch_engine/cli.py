from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path

from pitch_engine.config import AppConfig, ConfigError, load_config
from pitch_engine.detectors import build_detector
from pitch_engine.errors import FatalError
from pitch_engine.pipeline import FieldPipeline, RunSummary
from pitch_engine.reporting import JobEvent, Reporter, ReportingError
from pitch_engine.video import FrameSource, VideoOpenError, open_video

logger = logging.getLogger(__name__)

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
    parser.add_argument(
        "--job-id",
        default=None,
        help="Identifier reported to the platform for this run (default: a generated UUID)",
    )
    return parser


def _build_reporter(config: AppConfig, job_id: str) -> Reporter | None:
    if not config.reporting.enabled:
        return None
    # MOCK_API_URL lets the container-compose deployment point at the
    # mock_api service by its Docker network name without editing the
    # checked-in config file; the config value remains the default/local one.
    base_url = os.environ.get("MOCK_API_URL", config.reporting.base_url)
    return Reporter(base_url, job_id)


def _report_event(reporter: Reporter | None, event: JobEvent) -> None:
    # A failure to reach the platform is never allowed to change the
    # pipeline's own outcome or exit code - it's only ever logged here.
    if reporter is None:
        return
    try:
        reporter.report_event(event)
    except ReportingError:
        logger.warning("failed to report job outcome to the platform", exc_info=True)


def _report_progress(reporter: Reporter | None, frames_processed: int, frames_skipped: int, valid_detections: int) -> None:
    if reporter is None:
        return
    try:
        reporter.report_progress(frames_processed, frames_skipped, valid_detections)
    except ReportingError:
        logger.warning("failed to report progress to the platform", exc_info=True)


def _completion_event(job_id: str, summary: RunSummary) -> JobEvent:
    return JobEvent(
        job_id=job_id,
        status="completed",
        exit_code=EXIT_OK,
        message="run completed",
        frames_processed=summary.frames_processed,
        valid_detections=summary.valid_detections,
        mean_frame_overlap=summary.mean_frame_overlap,
    )


def _failure_event(job_id: str, exit_code: int, message: str) -> JobEvent:
    return JobEvent(job_id=job_id, status="failed", exit_code=exit_code, message=message)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    job_id = args.job_id or uuid.uuid4().hex

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        # No validated config yet, so there's no reporting.base_url to
        # report through either - this failure can only go to stderr.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    reporter = _build_reporter(config, job_id)

    video_path = Path(config.video.path)
    if not video_path.exists():
        message = (
            f"video file not found: {video_path}\n"
            "  Generate it first, e.g.: python synthetic_generator.py"
        )
        print(f"error: {message}", file=sys.stderr)
        _report_event(reporter, _failure_event(job_id, EXIT_VIDEO_NOT_FOUND, message))
        return EXIT_VIDEO_NOT_FOUND

    try:
        capture = open_video(str(video_path))
        frames = FrameSource(capture, config.sampling.frame_stride)
    except VideoOpenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _report_event(reporter, _failure_event(job_id, EXIT_VIDEO_NOT_FOUND, str(exc)))
        return EXIT_VIDEO_NOT_FOUND

    pipeline = FieldPipeline(build_detector(config.detector))

    def on_progress(frames_processed: int, frames_skipped: int, valid_detections: int) -> None:
        _report_progress(reporter, frames_processed, frames_skipped, valid_detections)

    try:
        summary = pipeline.run(frames, on_progress=on_progress)
    except FatalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        _report_event(reporter, _failure_event(job_id, EXIT_PIPELINE_FAILED, str(exc)))
        return EXIT_PIPELINE_FAILED

    _report_event(reporter, _completion_event(job_id, summary))
    print(summary)
    return EXIT_OK
