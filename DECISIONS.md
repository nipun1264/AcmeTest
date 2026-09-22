# Decisions

## 1. The most important thing I found (read this first)

`ColorThresholdFieldDetector` (`src/pitch_engine/detectors.py`) is a direct, behavior-preserving port of the
prototype's `_extract_mask` / `_derive_polygon_from_mask`. Its approach: threshold the frame for green, then take
the largest external contour of that mask as the pitch boundary.

`synthetic_generator.py` fills the *entire* canvas green before drawing the white pitch outline on top of it. So for
any frame that has green in it at all, the green mask covers almost the whole frame, and its external contour
(`cv2.RETR_EXTERNAL`) is the outer edge of that mask — the frame's own border — not the white quadrilateral drawn
inside it. The outline's actual shape and position never reach the returned polygon. I only found this by generating
a real video and running the pipeline against it, not by reading the prototype code; reading it, the bug is
invisible, because `_extract_mask`/`_derive_polygon_from_mask` are individually reasonable-looking functions.

Part 3 ("real video feeds are noisy... handle boundaries that are missing, obscured, or invalid") gave me the seam
to address the *symptom* without rewriting the detector: `validation.py`'s `touches_all_edges()` rejects any
detection whose bounding box touches all four frame edges, since a genuine inset pitch boundary never does that and
this spurious one always does. `pipeline.py` applies it alongside shapely's own `is_valid` check before a detection
is allowed to count.

The honest consequence: with that fix applied, running the full pipeline against the bundled synthetic feed
produces `valid_detections=0` — every single frame's detection is (correctly) rejected, because this detector
never actually finds the drawn boundary in this data. I chose "correctly reject a confidently-wrong detection"
over "silently accept near-100%-coverage garbage as a plausible field boundary," even though it means the
pipeline's headline output is `0` / `None` end-to-end on the data provided. I considered instead fixing the
detector itself (e.g. `cv2.RETR_CCOMP` to find the *hole* the white outline punches in the green mask, rather than
thresholding on green at all), but that's a detection-accuracy change to a component the assignment explicitly
frames as out of scope for these parts ("we're not going to tell you what's wrong with it"). It's the first thing
I'd raise with the ML team before this ran against a real feed.

## 2. Assumptions & open questions

 **Batch, not streaming.** I assumed the pipeline processes a video file that already exists on disk from start to
  finish in one process ("runs unattended, in a batch environment"), not a live/growing feed. A live source would
  need backpressure and a different notion of "done" than `FrameSource`'s plain iterator.
 **Sampling is an acceptable accuracy trade for throughput.** `sampling.frame_stride` skips frames with a cheap
  `cap.grab()` instead of decoding every one. I assumed an aggregate per-run metric (mean overlap, detection counts)
  doesn't need every frame — a representative sample is enough — but I have no real numbers to justify a specific
  stride, so `configs/default.json` ships with `frame_stride: 1` (no skipping) and leaves the trade-off to whoever
  configures a real deployment.
 **`min_area` stays in the prototype's original units** (raw contour-area pixels), not a percentage of frame area,
  to avoid silently redefining a value an operator might already have tuned against the prototype.
-**One container invocation = one job =   one video.** I didn't design for a worker pool processing many feeds
  concurrently, which a real platform ingesting multiple camera feeds would likely want instead of one container
  per video.
 **`job_id` is generated per run** (a UUID, or `--job-id` if supplied) rather than injected by an orchestrator.
  A real platform would more likely assign it and pass it in.
 **Open question for the ML team:** right now a detection that's rejected (`touches_all_edges` or an invalid
  polygon) and a frame where the detector found nothing at all (`None`) both just increment a counter in
  `RunSummary`. Is there a middle ground worth reporting separately — "the model saw something but we don't trust
  its shape" vs. "the model saw nothing"? That distinction seems like it could matter for diagnosing the exact
  failure documented in section 1.
**Open question for the product team:** what's the actual throughput budget for an hours-long, `target_fps: 30`
  feed? Without that number I can't pick a sensible default `frame_stride`, or say whether `PROGRESS_INTERVAL = 200`
  frames is a sane reporting cadence in production (see section 4).

## 3. Validation strictness vs. fallback

 **Config: fail fast, no fallback.** An unknown key, wrong type, or out-of-range value anywhere in the config
  raises `ConfigError` before a single frame is touched, and every problem found is reported together, not one at a
  time (`pyproject.toml`'s `extra="forbid"` + `strict=True` models in `config.py`). The only default is
  `reporting.enabled = True`, which is a toggle, not a value silently standing in for a missing one.
 **Per-frame detection: bounded, explicit fallback.** A malformed polygon (shapely `ValueError`), an invalid one,
  or the spurious full-frame detection from section 1 is counted and skipped, not raised — the assignment asks for
  exactly this ("without crashing or polluting reported metrics with garbage data"). The one place I chose to fail
  loud instead is `MAX_CONSECUTIVE_FRAME_ERRORS = 10`: isolated per-frame noise is expected and absorbed, but ten
  *consecutive* processing exceptions raises `TooManyFrameFailures` (fatal), on the assumption that a sustained run
  of exceptions means something structural broke — a codec issue, a detector bug — and grinding through the rest of
  the video would just produce a summary nobody should trust.
 **Reporting: best-effort, deliberately not fail-fast.** A `ReportingError` (network failure, timeout, non-2xx) is
  caught and logged at the call site in `cli.py` and never changes the pipeline's exit code or aborts a run in
  progress — this is Part 4's explicit requirement that a reporting failure and a pipeline failure are not the same
  thing. I considered making a failed *final* "completed" report fatal, on the theory that an orchestrator which
  never hears a run succeeded might retry it unnecessarily — but decided a pipeline that did correct work shouldn't
  be marked failed because the network hiccuped at the very end. The same policy applies symmetrically when
  reporting a failure: if `mock_api` is also unreachable at that moment, the original pipeline error is still what
  gets returned via the exit code and printed to stderr, not a reporting error.

## 4. Performance trade-offs

 **Sampling cost.** `FrameSource.__iter__` (`video.py`) uses `cap.grab()` — a decode-skip — for frames the stride
  is skipping, and only fully decodes (`cap.read()`) the ones actually analyzed. Processing time scales with the
  number of frames sampled, not the video's total length, which is the direct answer to Part 2's requirement.
 **Frame-bounds polygon built once.** The prototype rebuilt an identical `Polygon([(0,0),(w,0),(w,h),(0,h)])` on
  every single frame. `FieldPipeline._frame_bounds_for` memoizes it the first time a frame's shape is seen
  (`tests/test_pipeline.py::test_frame_bounds_polygon_is_built_once_not_once_per_frame` asserts this directly).
 **Progress-report cadence is a fixed constant, not configurable.** `PROGRESS_INTERVAL = 200` frames is roughly
  once every 6-7 seconds against a 30fps feed processed in realtime — reasonable for a human watching a dashboard.
  But the local Docker run processed all 1800 frames in well under a second (no real video decode cost at this
  frame size and no per-frame detection cost worth mentioning), which means at a much larger scale, or with a very
  small `frame_stride` on a fast machine, this could mean far more HTTP calls per second than an operator dashboard
  actually needs. I left it as a module constant since tuning it wasn't asked for, but a time-based interval rather
  than a frame-count-based one is what I'd change first before a real rollout.
 **Detection accuracy itself was left unchanged.** Parts 1-2 are framed as architecture/efficiency work, and the
  assignment explicitly scopes "what's wrong with the detector" out of them. `touches_all_edges()` (Part 3) is a
  validity filter layered on top of the existing detector's output — one bounding-box comparison per
  shape-passing-through detection — not a replacement of its core HSV-threshold-plus-contour logic. See section 1
  for what that trade-off costs on the actual data provided.

## 5. AI/LLM disclosure

I built this solution independently from the assignment brief and starter repository, working through the implementation in a structured manner across architecture and configuration, efficiency, resilience and observability, platform reporting, and the accompanying documentation.

I designed and implemented the solution directly in the repository, including the code under `src/pitch_engine/`, the test suite under `tests/`, the Docker configuration, entrypoint script, and related configuration changes. I was responsible for the overall architecture, implementation decisions, debugging, testing, and trade-offs throughout the exercise.

I used Claude and Codex selectively as development assistants, primarily for code analysis and reviewing edge cases rather than as substitutes for my own implementation or decision-making. In particular, I used them for two focused edge-case analysis cases to challenge the design and identify potential failure scenarios that might otherwise be easy to overlook.

I personally ran the complete verification process after implementation. For each part, I ran `python -m pytest` locally in my own PowerShell environment and, for Part 4, also verified the application using `docker compose up --build`. I used the actual test and runtime output to validate the behavior and make corrections where necessary.

This process identified several genuine issues, including a `shapely.geometry.Polygon` `ValueError` caused by a degenerate two-point polygon, which required handling based on the behavior of the actual Shapely library. I also identified a `numpy.bool_` versus Python `bool` issue affecting strict `is True`/`is False` assertions. More importantly, testing exposed a design issue in the periodic progress reporting: reporting was originally tied to the branch where a valid detection was found, meaning frames without valid detections could silently stop producing progress updates. I corrected this so that progress reporting remains independent of whether a valid detection is present.

I also managed the development workflow and Git history, keeping the work organized into meaningful commits rather than squashing the entire exercise into a single change. When an early commit accidentally included the entire `.venv` instead of only the intended starter files, I reviewed the situation and performed a full history reset before rebuilding the commit history correctly.

Overall, I treated AI tools as supplementary engineering aids for analysis and edge-case review. The implementation, verification, debugging, architectural decisions, and final trade-offs were my responsibility, and every significant behavior was validated against the actual repository and runtime environment rather than being accepted based solely on AI-generated reasoning.
