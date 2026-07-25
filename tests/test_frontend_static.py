"""Plan 04-01: the static mount that serves the canvas frontend, and its guarantees.

These tests assert the Python side of Phase 4 -- the parts a browser-free CI can prove:
``/app`` serves the shell, ``/`` redirects to it, the ES modules and the selfcheck harness
are served, the raw-image route feeds the canvas while still refusing a path-traversal id,
and two design invariants that are cheap to regress: the exploration-mode selector sits above
the method selector (the Milestone 2 seam), and no registered method name leaks into
``frontend/`` (UI-07). The visual/DPR transform proof is browser-driven and lives in
``frontend/dev/selfcheck.html``; it is verified separately.
"""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from object_search.api.static import frontend_dir
from object_search.search import list_methods


def test_root_redirects_to_app(api_client: TestClient) -> None:
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308)
    assert response.headers["location"].startswith("/app")


def test_app_serves_index_html(api_client: TestClient) -> None:
    response = api_client.get("/app/")
    assert response.status_code == 200
    body = response.text
    assert "<html" in body.lower()
    assert 'id="stage"' in body  # the canvas centre stage
    assert 'src="./js/main.js"' in body  # the ES-module entry point


def test_exploration_selector_is_above_method_selector(api_client: TestClient) -> None:
    # The Milestone 2 seam (UI-09): exploration mode must appear before the method select in
    # source order, so Milestone 2 adds an option rather than forking the app.
    body = api_client.get("/app/").text
    assert 'id="exploration"' in body
    assert 'id="method"' in body
    assert body.index('id="exploration"') < body.index('id="method"')


def test_frontend_modules_are_served(api_client: TestClient) -> None:
    for path in ("/app/js/viewport.js", "/app/js/form.js", "/app/js/main.js", "/app/js/api.js"):
        response = api_client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"].startswith(("text/javascript", "application/"))


def test_selfcheck_harness_is_served(api_client: TestClient) -> None:
    assert api_client.get("/app/dev/selfcheck.html").status_code == 200
    assert api_client.get("/app/dev/selfcheck.js").status_code == 200


def test_draw_handler_is_gated_behind_method_selection(api_client: TestClient) -> None:
    # UI-01: assert the served source carries the gate. drawingEnabled() requires a method,
    # and pointerdown returns early when it is false -- drawing is impossible before then.
    main_js = api_client.get("/app/js/main.js").text
    assert "state.method !== null" in main_js
    assert "if (!drawingEnabled())" in main_js


def test_raw_image_route_serves_a_demo_image(api_client: TestClient) -> None:
    images = api_client.get("/images").json()
    assert images, "expected at least one demo image in assets/demo"
    image_id = images[0]["id"]
    response = api_client.get("/image", params={"image_id": image_id})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert len(response.content) > 0


def test_raw_image_route_rejects_path_traversal(api_client: TestClient) -> None:
    response = api_client.get("/image", params={"image_id": "../../etc/passwd"})
    assert response.status_code == 404


def test_no_method_name_appears_in_frontend() -> None:
    # UI-07 / Phase 4 decision 3: the form is entirely schema-driven, so no registered method
    # name may appear anywhere in frontend/. This is the grep, made a hard test.
    names = [spec.name for spec in list_methods()]
    assert names, "expected registered methods to check against"
    offenders: list[str] = []
    for source in sorted(frontend_dir().rglob("*")):
        if source.suffix not in {".js", ".html", ".css"}:
            continue
        text = source.read_text(encoding="utf-8")
        for name in names:
            if name in text:
                offenders.append(f"{source.relative_to(frontend_dir())}: {name!r}")
    assert not offenders, f"method name(s) leaked into frontend/: {offenders}"


def test_frontend_dir_points_at_the_repo_frontend() -> None:
    directory = frontend_dir()
    assert directory.name == "frontend"
    assert (directory / "index.html").is_file()
    assert isinstance(directory, Path)
