# Phase 6 — Deferred Items

Out-of-scope discoveries logged during execution, not fixed in the plan that found them.

## 06-01 — `test_api_app.py` session-registry assertion is coupled to "no weights on disk"

**Found during:** 06-01 Task 2 (after `pixi run fetch-models --only dinov2-small`).

**Issue:** `tests/test_api_app.py::test_lifespan_migrates_store_and_builds_empty_session_registry`
asserts `app.state.sessions == {}` with the comment "empty because Phase 3 ships no ONNX weights."
The lifespan (`api/lifespan.py::build_session_registry`) loads a session for **every weight present
on disk**. Phase 6 is exactly when the first ONNX weight (`dinov2_small.onnx`) lands, so once a
developer runs `fetch-models`, the registry is no longer empty and this test fails **locally**.

**Why not fixed here:** CI keeps `models/` gitignored, so the weight is absent and the test passes
in CI (verified: full suite is 283 passed / 3 skipped / 90.66% coverage with the model hidden).
The fix belongs to the API-wiring plan **06-02**, which registers `dino_dense` and reads
`app.state.sessions["dinov2-small"]` — that plan owns `api/` and should update this test to assert
"a session exists when its weight is present" rather than "the registry is always empty." Touching
`test_api_app.py` from 06-01 (whose `files_modified` is the inferencer, not the API) would be scope
creep into 06-02.

**Action for 06-02:** update `test_lifespan_*` to reflect that a present weight yields a loaded
session; keep an absent-weight case for the CI-empty path.
