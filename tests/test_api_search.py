"""Task 3: ``/search`` persists provenance + latency + candidates; errors are typed (API-08).

A throwaway method that always raises is registered to exercise the error path without
depending on a real method failing. It is registered under a name that appears in no
production module (so the API-01 grep test is unaffected) **and only for the duration of this
module's tests** -- an autouse fixture registers it in setup and unregisters it in teardown, so
it never leaks into the global registry that another test file (e.g. the sample renderer, which
runs every registered method) enumerates.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel, ConfigDict
from starlette.testclient import TestClient

from object_search.schemas.geometry import ExemplarBox
from object_search.schemas.search import SearchResult
from object_search.search import register_method, unregister

_RAISER = "test-raiser"


class _RaiserConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@pytest.fixture(autouse=True)
def _register_raiser() -> Iterator[None]:
    """Register the always-raising stub for this module's tests, then remove it again.

    Scoped and cleaned up on purpose: a module-level registration would permanently pollute
    the global registry, and a later test that renders a sample per registered method would
    then try to run a method whose whole job is to raise.
    """

    @register_method(
        name=_RAISER,
        description="Always raises; exercises the API-08 structured-error path.",
        version="0.0.0",
        config_model=_RaiserConfig,
    )
    def _always_raise(
        image: npt.NDArray[np.uint8],
        exemplar: ExemplarBox,
        config: BaseModel,
    ) -> SearchResult:
        raise RuntimeError("boom: the method exploded")

    try:
        yield
    finally:
        unregister(_RAISER)


# A concrete exemplar taken from chipset-01's ground truth (box 0, 24x24 near the top edge).
_CHIPSET_IMAGE = "chipset/chipset-01.png"
_CHIPSET_EXEMPLAR = {"box": {"x": 293, "y": 12, "w": 24, "h": 24}, "label": "chip"}


def test_search_persists_provenance_latency_and_candidates(api_client: TestClient) -> None:
    response = api_client.post(
        "/search",
        json={
            "image_id": _CHIPSET_IMAGE,
            "exemplar": _CHIPSET_EXEMPLAR,
            "method": "ncc",
            "config": {},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    run_id = payload["run_id"]
    assert run_id >= 1
    result = payload["result"]
    assert result["outcome"] in {"ok", "empty"}
    assert result["method"] == "ncc"

    # Re-read the persisted run over HTTP and assert the evidence the evaluation design needs.
    got = api_client.get(f"/runs/{run_id}")
    assert got.status_code == 200, got.text
    run = got.json()

    prov = run["provenance"]
    assert prov["git_sha"]
    assert prov["config_hash"]
    assert prov["cv2_version"]
    # config_hash on the run row matches the provenance hash (same validated config).
    assert run["config_hash"] == prov["config_hash"]

    latency = run["result"]["latency"]
    assert {"preprocess_ms", "inference_ms", "postprocess_ms"} <= latency.keys()
    assert all(latency[k] >= 0.0 for k in ("preprocess_ms", "inference_ms", "postprocess_ms"))

    # Sub-threshold candidates are logged with their raw scores (EVAL-08).
    candidates = run["result"]["candidates"]
    assert candidates, "expected sub-threshold candidates to be persisted"
    assert all("score" in candidate for candidate in candidates)


def test_a_raising_method_yields_a_typed_error_and_persisted_error_run(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/search",
        json={
            "image_id": _CHIPSET_IMAGE,
            "exemplar": _CHIPSET_EXEMPLAR,
            "method": _RAISER,
            "config": {},
        },
    )
    # Not a 500, not a stack trace: a typed result with outcome='error'.
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload["result"]
    assert result["outcome"] == "error"
    assert result["error"] is not None
    assert result["error"]["kind"] == "RuntimeError"
    assert "boom" in result["error"]["message"]

    # The error run is persisted as evidence (EVAL-12), retrievable with outcome='error'.
    run = api_client.get(f"/runs/{payload['run_id']}").json()
    assert run["result"]["outcome"] == "error"
    assert run["result"]["error"]["kind"] == "RuntimeError"


def test_unknown_method_is_404_with_known_names(api_client: TestClient) -> None:
    response = api_client.post(
        "/search",
        json={
            "image_id": _CHIPSET_IMAGE,
            "exemplar": _CHIPSET_EXEMPLAR,
            "method": "does-not-exist",
            "config": {},
        },
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["kind"] == "unknown_method"
    # The message lists the known methods so a typo is diagnosable.
    assert "ncc" in error["message"]


def test_invalid_config_is_422_not_500(api_client: TestClient) -> None:
    response = api_client.post(
        "/search",
        json={
            "image_id": _CHIPSET_IMAGE,
            "exemplar": _CHIPSET_EXEMPLAR,
            "method": "ncc",
            # nms_iou has an upper bound of 1.0; 5.0 must be rejected as a config error.
            "config": {"nms_iou": 5.0},
        },
    )
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["kind"] == "invalid_config"
    assert error["detail"], "expected per-field validation errors in the 422 body"


def test_run_not_found_is_404(api_client: TestClient) -> None:
    response = api_client.get("/runs/999999")
    assert response.status_code == 404
    assert response.json()["error"]["kind"] == "run_not_found"
