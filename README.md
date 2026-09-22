# Automated Pitch Boundary & Crop Engine

A configurable pipeline that ingests a match video feed, detects the playing-field boundary per sampled frame, and
reports its progress and outcome to a platform reporting service. Built from the TrackBox take-home prototype
(`synthetic_field_prototype.py`, kept in the repo unchanged as a reference/baseline) as a reusable library plus a
thin CLI, with validated configuration, bounded failure handling, and observability suited to running unattended.

See [`DECISIONS.md`](DECISIONS.md) for the design rationale, trade-offs, and a real detection limitation found by
running this against generated video (not by reading the code).

## Layout

```
src/pitch_engine/
  config.py      - validated, fail-fast configuration (Pydantic)
  detection.py   - the FieldDetector protocol / seam the pipeline depends on
  detectors.py   - ColorThresholdFieldDetector, ported unchanged from the prototype
  video.py       - FrameSource: stride-based frame sampling (cheap grab() vs. read())
  validation.py  - touches_all_edges(): rejects the spurious full-frame detection (see DECISIONS.md)
  errors.py      - PitchEngineError / FatalError hierarchy
  pipeline.py    - FieldPipeline: runs a detector over a FrameSource, aggregates a RunSummary
  reporting.py   - Reporter + validated ProgressReport/JobEvent models, talks to mock_api over HTTP
  cli.py         - entry point: wires config -> video -> pipeline -> reporting, maps failures to exit codes
tests/           - one test module per src module above
configs/default.json - sample configuration
mock_api/        - the starter's reporting service (unmodified)
Dockerfile, docker-entrypoint.sh, docker-compose.yml - containerizes the pipeline alongside mock_api
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate      # on Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Running the tests

```bash
python -m pytest
```

## Running locally

Generate the synthetic feed once, then run the CLI against a config:

```bash
python -c "from synthetic_generator import generate_synthetic_video; generate_synthetic_video()"
pitch-engine --config configs/default.json
```

(`pitch-engine` is the console script `pyproject.toml` registers; `python -m pitch_engine --config ...` works the
same way.) By default this also tries to report progress/outcome to `configs/default.json`'s
`reporting.base_url` (`http://localhost:5000`) — run `python mock_api/app.py` first if you want to see that
traffic locally, or set `"reporting": {"enabled": false, ...}` in your config to skip it. A reporting failure never
changes the pipeline's own exit code (see `DECISIONS.md`, section 3).

## Running with Docker

```bash
docker compose up --build
```

This brings up `mock_api` and a `runner` service together. The runner's entrypoint generates the synthetic feed on
first run (if it isn't already present) and then executes the CLI, reporting to `mock_api` over the compose
network (`MOCK_API_URL=http://mock_api:5000`, overriding the config file's default for exactly this deployment).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Completed |
| 2 | Invalid or unreadable configuration |
| 3 | Video file not found or could not be opened |
| 4 | Pipeline failed (e.g. too many consecutive frame-processing errors) |
