from __future__ import annotations

import logging
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ReportingError(Exception):
    """The platform reporting service could not be reached, or rejected a
    report. Deliberately not a PitchEngineError: callers must be able to
    tell "the pipeline failed" apart from "we couldn't tell anyone about
    it", and must not let the second one masquerade as the first."""


class ProgressReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    job_id: str
    status: Literal["running"] = "running"
    frames_processed: int
    frames_skipped: int
    valid_detections: int


class JobEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    job_id: str
    status: Literal["completed", "failed"]
    exit_code: int
    message: str
    frames_processed: int | None = None
    valid_detections: int | None = None
    mean_frame_overlap: float | None = None


class Reporter:
    """Talks to the platform's job-reporting API. Every method raises
    ReportingError on any network problem or non-2xx response - it never
    raises the underlying requests exception directly, so callers only
    need to know about one failure type."""

    def __init__(self, base_url: str, job_id: str, timeout: float = 5.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._job_id = job_id
        self._timeout = timeout

    @property
    def job_id(self) -> str:
        return self._job_id

    def report_progress(self, frames_processed: int, frames_skipped: int, valid_detections: int) -> None:
        report = ProgressReport(
            job_id=self._job_id,
            frames_processed=frames_processed,
            frames_skipped=frames_skipped,
            valid_detections=valid_detections,
        )
        self._post("/api/v1/jobs/progress", report)

    def report_event(self, event: JobEvent) -> None:
        self._post("/api/v1/jobs/events", event)

    def _post(self, path: str, model: BaseModel) -> None:
        url = f"{self._base_url}{path}"
        try:
            response = requests.post(url, json=model.model_dump(mode="json"), timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ReportingError(f"failed to reach {url}: {exc}") from exc
