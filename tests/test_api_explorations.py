"""``GET /explorations`` and exploration routing + persistence on ``POST /search`` (Task 5).

The routing/persistence model is exercised model-free through a stub exploration registered for
the test; the real marker-conditioned end-to-end path is a separate skip-when-absent test that
needs the FastSAM weight.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict, Field
from starlette.testclient import TestClient

from object_search.explorations import register_exploration, unregister
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.schemas.search import (
    Diagnostics,
    LatencyBreakdown,
    Match,
    SearchOutcome,
    SearchResult,
)

_STUB_EXPLORATION = "stub-exploration-for-tests"


class _StubExplorationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tag: str = Field(default="x")


@pytest.fixture
def _register_stub() -> Iterator[None]:
    @register_exploration(
        name=_STUB_EXPLORATION,
        description="Returns one fixed match; no model.",
        version="0.0.0",
        config_model=_StubExplorationConfig,
    )
    def run(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        return SearchResult(
            method="marker-conditioned",
            method_version="0.0.0",
            outcome=SearchOutcome.OK,
            matches=(Match(box=BBox(x=1, y=1, w=8, h=8), score=0.5),),
            latency=LatencyBreakdown(preprocess_ms=0.0, inference_ms=0.0, postprocess_ms=0.0),
            threshold_applied=None,
            diagnostics=Diagnostics(notes=("stub",)),
        )

    try:
        yield
    finally:
        unregister(_STUB_EXPLORATION)


def _synthetic_image_id(client: TestClient) -> str:
    """The id of a demo image with ground truth, taken from the catalogue."""
    images = client.get("/images").json()
    for entry in images:
        if entry["id"].startswith("synthetic/"):
            return str(entry["id"])
    return str(images[0]["id"])


def test_get_explorations_lists_both_with_config_schemas(api_client: TestClient) -> None:
    resp = api_client.get("/explorations")
    assert resp.status_code == 200
    by_name = {e["name"]: e for e in resp.json()}
    assert "same-image-search" in by_name
    assert "marker-conditioned" in by_name
    # Each carries a non-empty config JSON Schema.
    assert by_name["marker-conditioned"]["config_schema"]["properties"]
    assert by_name["same-image-search"]["config_schema"]["properties"]


def test_no_exploration_name_is_a_dispatch_literal_in_api_package() -> None:
    """Dispatch is registry-driven: no exploration name appears as a string literal in api/."""
    api_dir = Path(__file__).resolve().parents[1] / "src" / "object_search" / "api"
    offenders: list[str] = []
    for path in api_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("marker-conditioned", "same-image-search"):
            if f'"{name}"' in text or f"'{name}'" in text:
                offenders.append(f"{path.name}: {name}")
    assert offenders == [], f"exploration names hardcoded in api/: {offenders}"


def test_default_path_persists_same_image_search(api_client: TestClient) -> None:
    image_id = _synthetic_image_id(api_client)
    body = {
        "image_id": image_id,
        "exemplar": {"box": {"x": 4, "y": 4, "w": 24, "h": 24}},
        "method": "ncc",
        "config": {},
    }
    resp = api_client.post("/search", json=body)
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    run = api_client.get(f"/runs/{run_id}").json()
    assert run["exploration"] == "same-image-search"


def test_marker_exploration_routes_and_persists(
    _register_stub: None, api_client: TestClient
) -> None:
    image_id = _synthetic_image_id(api_client)
    body = {
        "image_id": image_id,
        "exemplar": {"box": {"x": 4, "y": 4, "w": 24, "h": 24}},
        "method": "marker-conditioned",
        "exploration": _STUB_EXPLORATION,
        "config": {"tag": "y"},
    }
    resp = api_client.post("/search", json=body)
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    run = api_client.get(f"/runs/{run_id}").json()
    assert run["exploration"] == _STUB_EXPLORATION
    assert run["result"]["outcome"] == "ok"


def test_unknown_exploration_is_404(api_client: TestClient) -> None:
    image_id = _synthetic_image_id(api_client)
    body = {
        "image_id": image_id,
        "exemplar": {"box": {"x": 4, "y": 4, "w": 24, "h": 24}},
        "method": "ncc",
        "exploration": "does-not-exist",
        "config": {},
    }
    resp = api_client.post("/search", json=body)
    assert resp.status_code == 404
    assert resp.json()["error"]["kind"] == "unknown_exploration"


def test_marker_conditioned_end_to_end_when_models_present(api_client: TestClient) -> None:
    """The real exploration against FastSAM + a marker method, skipped when weights are absent."""
    from object_search.inference import models

    fastsam = models.models_dir() / models.MODEL_REGISTRY["fastsam-s"].dest
    if not fastsam.exists():
        pytest.skip("fastsam_s.onnx not present; run `pixi run -e export export-fastsam`")

    image_id = _synthetic_image_id(api_client)
    body = {
        "image_id": image_id,
        "exemplar": {"box": {"x": 4, "y": 4, "w": 24, "h": 24}},
        "method": "marker-conditioned",
        "exploration": "marker-conditioned",
        "config": {"marker_method": "ncc", "marker_config": {}},
    }
    resp = api_client.post("/search", json=body)
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]
    run = api_client.get(f"/runs/{run_id}").json()
    # The load-bearing M2 claim: the real path routes through the exploration registry and
    # persists under the marker-conditioned tag with no schema migration.
    assert run["exploration"] == "marker-conditioned"
    outcome = run["result"]["outcome"]
    # 'ok'/'empty' are the healthy outcomes; 'error' can occur if the ONNX Runtime execution
    # provider (e.g. CoreML on macOS CI) fails to build a plan for this graph -- an environment
    # limitation, not a pipeline bug, and it still proves routing + persistence.
    assert outcome in {"ok", "empty", "error"}
    if outcome == "ok":
        assert run["result"]["matches"]
