import requests
import pytest
from pydantic import ValidationError

from pitch_engine.reporting import JobEvent, Reporter, ReportingError


class FakeResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_report_progress_posts_to_the_progress_endpoint(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse(200)

    monkeypatch.setattr("pitch_engine.reporting.requests.post", fake_post)

    Reporter("http://mock_api:5000", job_id="job-1").report_progress(
        frames_processed=10, frames_skipped=2, valid_detections=8
    )

    assert captured["url"] == "http://mock_api:5000/api/v1/jobs/progress"
    assert captured["json"] == {
        "job_id": "job-1",
        "status": "running",
        "frames_processed": 10,
        "frames_skipped": 2,
        "valid_detections": 8,
    }


def test_report_event_posts_to_the_events_endpoint(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "pitch_engine.reporting.requests.post",
        lambda url, json, timeout: (captured.update(url=url, json=json), FakeResponse(200))[1],
    )

    event = JobEvent(job_id="job-1", status="completed", exit_code=0, message="done")
    Reporter("http://mock_api:5000/", job_id="job-1").report_event(event)

    # A trailing slash on base_url must not produce a doubled slash in the path.
    assert captured["url"] == "http://mock_api:5000/api/v1/jobs/events"
    assert captured["json"]["status"] == "completed"


def test_a_network_error_raises_reporting_error_not_the_raw_requests_exception(monkeypatch):
    def raise_connection_error(url, json, timeout):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("pitch_engine.reporting.requests.post", raise_connection_error)

    with pytest.raises(ReportingError):
        Reporter("http://mock_api:5000", job_id="job-1").report_progress(1, 0, 1)


def test_a_non_2xx_response_raises_reporting_error(monkeypatch):
    monkeypatch.setattr(
        "pitch_engine.reporting.requests.post", lambda url, json, timeout: FakeResponse(500)
    )

    with pytest.raises(ReportingError):
        Reporter("http://mock_api:5000", job_id="job-1").report_progress(1, 0, 1)


def test_job_event_rejects_an_unknown_status():
    with pytest.raises(ValidationError):
        JobEvent(job_id="job-1", status="finished", exit_code=0, message="oops")
