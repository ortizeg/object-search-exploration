"""Task 2: ``/methods`` and ``/images`` over the real HTTP stack.

The load-bearing test here is :func:`test_api_package_hardcodes_no_method_name` -- it greps
the entire ``api/`` package for every registered method name and asserts none appears,
enforcing API-01 mechanically rather than by reviewer vigilance.
"""

from __future__ import annotations

import re
from pathlib import Path

from starlette.testclient import TestClient

import object_search.api as api_package
from object_search.search import list_methods


def test_get_methods_returns_ncc_with_a_complete_config_schema(api_client: TestClient) -> None:
    response = api_client.get("/methods")
    assert response.status_code == 200
    methods = {entry["name"]: entry for entry in response.json()}

    assert "ncc" in methods
    ncc = methods["ncc"]
    assert ncc["description"]
    assert ncc["version"]

    schema = ncc["config_schema"]
    assert schema, "config_schema must not be empty"
    properties = schema.get("properties", {})
    # A representative sample of NCCConfig's fields must be present in the rendered schema --
    # this is the exact object UI-07 turns into a form.
    for field in ("scales", "angles_deg", "threshold", "calibration", "peaks"):
        assert field in properties, f"expected {field!r} in the ncc config schema"


def test_api_package_hardcodes_no_method_name() -> None:
    """API-01: no registered method name appears anywhere in the api/ package source."""
    names = [spec.name for spec in list_methods()]
    assert names, "expected at least one registered method to make this test meaningful"

    api_dir = Path(api_package.__file__).parent
    source_files = sorted(api_dir.glob("*.py"))
    assert source_files

    offenders: list[str] = []
    for source in source_files:
        text = source.read_text()
        for name in names:
            if re.search(rf"\b{re.escape(name)}\b", text):
                offenders.append(f"{source.name} contains method name {name!r}")
    assert not offenders, "api/ package must name no method (API-01): " + "; ".join(offenders)


def test_get_images_reports_ground_truth_true_for_chipset(api_client: TestClient) -> None:
    response = api_client.get("/images")
    assert response.status_code == 200
    images = response.json()
    assert images

    by_id = {img["id"]: img for img in images}
    # Every entry carries the flag and real dimensions.
    for img in images:
        assert "has_ground_truth" in img
        assert img["width"] >= 1 and img["height"] >= 1

    chipset = [img for img in images if img["id"].startswith("chipset/")]
    assert chipset, "expected chipset demo images in the catalogue"
    # Chipset ships .gt.json sidecars, so ground truth is objectively available.
    assert all(img["has_ground_truth"] for img in chipset)

    # Basketball frames ship no sidecar -> not objectively scorable.
    basketball = [img for img in images if img["id"].startswith("basketball/")]
    if basketball:
        assert all(not img["has_ground_truth"] for img in basketball)

    # Spot-check a concrete id resolves and its dimensions match the sidecar (320x240).
    assert "chipset/chipset-01.png" in by_id
    assert by_id["chipset/chipset-01.png"]["width"] == 320
    assert by_id["chipset/chipset-01.png"]["height"] == 240
