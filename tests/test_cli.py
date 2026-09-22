import json

import pitch_engine.cli as cli_module
from pitch_engine.cli import main
from pitch_engine.reporting import ReportingError

BASE_CONFIG = {
    "video": {"path": "PLACEHOLDER", "target_fps": 30},
    "detector": {
        "kind": "sam_mask_v1",
        "sport": "football",
        "min_area": 1000,
        "confidence_threshold": 0.05,
    },
    "sampling": {"frame_stride": 1},
    "crop_search": {"aspect_ratio": "16:9", "padding_px": 20},
    "reporting": {"base_url": "http://mock_api:5000", "enabled": True},
    "debug_mode": True,
}


def write_config(tmp_path, video_path, **overrides):
    data = json.loads(json.dumps(BASE_CONFIG))
    data["video"]["path"] = str(video_path)
    data.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


class FakeCapture:
    """An always-empty capture: isOpened() is True, but the very first
    read() signals end-of-stream, so the pipeline runs (and completes)
    without needing a real video file or cv2 at all."""

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def grab(self):
        return False

    def release(self):
        pass


def test_missing_config_file_exits_with_the_config_error_code_and_reports_nothing(tmp_path, monkeypatch):
    reported = []
    monkeypatch.setattr(cli_module.Reporter, "report_event", lambda self, event: reported.append(event))

    exit_code = main(["--config", str(tmp_path / "does_not_exist.json")])

    assert exit_code == cli_module.EXIT_CONFIG_ERROR
    assert reported == []


def test_missing_video_file_is_reported_as_a_failed_event(tmp_path, monkeypatch):
    reported = []
    monkeypatch.setattr(cli_module.Reporter, "report_event", lambda self, event: reported.append(event))
    config_path = write_config(tmp_path, tmp_path / "does_not_exist.mp4")

    exit_code = main(["--config", str(config_path)])

    assert exit_code == cli_module.EXIT_VIDEO_NOT_FOUND
    assert len(reported) == 1
    assert reported[0].status == "failed"
    assert reported[0].exit_code == cli_module.EXIT_VIDEO_NOT_FOUND


def test_a_successful_run_is_reported_as_completed(tmp_path, monkeypatch):
    reported = []
    monkeypatch.setattr(cli_module.Reporter, "report_event", lambda self, event: reported.append(event))
    monkeypatch.setattr(cli_module, "open_video", lambda path: FakeCapture())
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    config_path = write_config(tmp_path, video_path)

    exit_code = main(["--config", str(config_path)])

    assert exit_code == cli_module.EXIT_OK
    assert len(reported) == 1
    assert reported[0].status == "completed"
    assert reported[0].exit_code == cli_module.EXIT_OK


def test_a_reporting_failure_never_changes_the_pipeline_exit_code(tmp_path, monkeypatch):
    # This is the Part 4 requirement: mock_api being unreachable must not
    # turn a successful (or failed) pipeline run into a different outcome.
    def always_unreachable(self, event):
        raise ReportingError("mock_api is unreachable")

    monkeypatch.setattr(cli_module.Reporter, "report_event", always_unreachable)
    monkeypatch.setattr(cli_module, "open_video", lambda path: FakeCapture())
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    config_path = write_config(tmp_path, video_path)

    exit_code = main(["--config", str(config_path)])

    assert exit_code == cli_module.EXIT_OK


def test_reporting_disabled_in_config_means_the_reporter_is_never_used(tmp_path, monkeypatch):
    def fail_if_constructed(self, *args, **kwargs):
        raise AssertionError("Reporter must not be built when reporting.enabled is False")

    monkeypatch.setattr(cli_module.Reporter, "__init__", fail_if_constructed)
    monkeypatch.setattr(cli_module, "open_video", lambda path: FakeCapture())
    video_path = tmp_path / "video.mp4"
    video_path.touch()
    config_path = write_config(
        tmp_path, video_path, reporting={"base_url": "http://mock_api:5000", "enabled": False}
    )

    exit_code = main(["--config", str(config_path)])

    assert exit_code == cli_module.EXIT_OK
