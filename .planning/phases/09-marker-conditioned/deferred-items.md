# Deferred items — Phase 09 (marker-conditioned)

## FastSAM inference fails under the CoreML execution provider on this macOS host (environment, pre-existing)

**Found during:** 09-02 human-verify readiness (server-side `POST /search` with `exploration=marker-conditioned`).

**Symptom:** A marker search through the running server persists as `outcome=error` with
`[ONNXRuntimeError] : 1 : FAIL ... CoreMLExecutionProvider ... Error in building plan`. The API
handles it correctly (API-08: a typed, persisted error run, no crash), but the marker overlay cannot
be produced through the browser on this host.

**Root cause (not introduced by this plan):** ONNX sessions default to *all* available providers —
`api/lifespan.py:67` uses `ort.get_available_providers()`, and `search/proposals.py:default_backend`
passes `providers=None`. On this host that puts `CoreMLExecutionProvider` first, and CoreML fails to
build a plan for the FastSAM graph. Any FastSAM-backed path (including Method 5 `propose-retrieve`)
is affected, so this is pre-existing infrastructure behavior, not caused by the marker UI/docs/assets
work in this plan.

**Verified workaround:** the `CPUExecutionProvider` runs FastSAM correctly (10 proposals on a marker
scene). The committed marker sample gallery is rendered with the CPU provider for exactly this reason
(`cli.py render-samples` → `default_backend(providers=["CPUExecutionProvider"])`), which is also what
makes the panels reproducible.

**Why deferred (not auto-fixed here):** the fix point is ONNX provider selection in `lifespan.py` /
`proposals.py` — shared inference infrastructure outside this plan's declared files (UI, docs, demo
assets). Changing global provider selection affects every learned method's latency and the
determinism guarantees the inference layer exists to provide (Rule 4 / scope boundary). It should be
a deliberate infra change, e.g. an opt-in `providers` override (env var) threaded through
`build_session_registry` and `default_backend`, with a documented CPU-first fallback on macOS.

**For the orchestrator browser human-verify:** run on a host where the CoreML EP builds the FastSAM
plan, or start the server with a CPU-only provider selection, then follow the marker flow in
`09-02-PLAN.md`'s checkpoint (pick "Marker-conditioned", draw a box around one arrow on a marker demo
image, Search; expect marker boxes, a direction arrow per marker, and the pointed-at object boxed as
the chosen proposal). The UI, routing, overlay, and scoring are all in place and unit-tested; only the
FastSAM CoreML execution is environment-blocked on this specific host.
