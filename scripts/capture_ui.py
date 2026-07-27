"""Capture the walkthrough screenshots for the docs interface guide.

This drives the *real* canvas UI with Playwright — no mocked screens. It loads the running
app, selects a method, drives the exact pointer-event drag the frontend listens for (a valid
box on ``pointerup`` auto-triggers ``POST /search``), waits for the result overlay to render,
and writes crisp 2x PNGs into ``docs/assets/ui/``.

Prerequisites (maintainer, one-off — the PNGs are committed, so nobody browsing the docs runs
this):

    pixi run serve                     # in one terminal: app on http://127.0.0.1:8000
    pixi run -e capture playwright-install   # once: fetch the Chromium binary
    pixi run -e capture capture-ui     # in another terminal: capture the shots

The demo query is chosen to be model-free and deterministic: method ``ncc`` (no ONNX weights)
on the ground-truthed ``synthetic/lattice-plain.png`` (twelve identical crosses), with the
exemplar box placed on the first ground-truth instance so a result overlay (all twelve matches
plus the similarity heatmap) is guaranteed.

Coordinate note: the canvas fits the image with a letterbox (see frontend/js/viewport.js
``fitContain``). devicePixelRatio cancels out at the CSS level, so image->client is a plain
fit-contain in CSS pixels: ``client = image * min(rectW/natW, rectH/natH) + letterbox + rect``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from playwright.sync_api import Page, sync_playwright

# --- The demo query, pinned for reproducibility ---------------------------------------------
BASE_URL = "http://127.0.0.1:8000"
# The iconic scene: a 4x3 lattice of twelve identical crosses that fills the frame — the same
# scene the README leads with. Dense and full-bleed, so the "find every instance" overlay reads
# clearly (unlike the sparse chipset board).
IMAGE_ID = "synthetic/lattice-plain.png"
IMAGE_W, IMAGE_H = 960, 640  # natural size of lattice-plain (from GET /images)
METHOD = "ncc"  # zero-model baseline — no weights, deterministic, always available
# Exemplar box in IMAGE pixels: the first ground-truth cross in lattice-plain.gt.json, padded a
# few px so the drawn box comfortably encloses one cross.
EXEMPLAR = {"x": 120, "y": 103, "w": 63, "h": 63}

# A 3:2 canvas fits the 960x640 lattice with little letterbox at this window size.
VIEWPORT = {"width": 1500, "height": 860}

# The rail/panel are designed to scroll inside a fixed layout height (both carry
# `overflow-y: auto`), but `.layout` sets only `min-height: calc(100vh - 64px)`, so a tall
# config form (ncc's is ~1644px) stretches the grid row and pushes the width-fitted canvas far
# below the fold. Pinning the layout to the viewport height engages the intended scroll and
# lets the canvas fill the window — the same view a user gets on a tall-enough monitor.
LAYOUT_HEIGHT_CAP = ".layout { height: calc(100vh - 64px); }"

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "ui"


def _image_to_client(ix: float, iy: float, rect: dict[str, float]) -> tuple[float, float]:
    """Image px -> viewport client px, replicating the canvas fit-contain in CSS space."""
    zoom = min(rect["width"] / IMAGE_W, rect["height"] / IMAGE_H)
    pan_x = (rect["width"] - IMAGE_W * zoom) / 2
    pan_y = (rect["height"] - IMAGE_H * zoom) / 2
    return rect["x"] + pan_x + ix * zoom, rect["y"] + pan_y + iy * zoom


def _draw_box(page: Page) -> None:
    """Drive the real left-drag the canvas listens for; a valid box auto-runs the search."""
    stage = page.locator("#stage")
    rect = stage.bounding_box()
    if rect is None:
        raise RuntimeError("canvas #stage has no bounding box — is the image loaded?")
    x0, y0 = _image_to_client(EXEMPLAR["x"], EXEMPLAR["y"], rect)
    x1, y1 = _image_to_client(EXEMPLAR["x"] + EXEMPLAR["w"], EXEMPLAR["y"] + EXEMPLAR["h"], rect)
    logger.info("Dragging exemplar box from ({:.0f},{:.0f}) to ({:.0f},{:.0f})", x0, y0, x1, y1)
    page.mouse.move(x0, y0)
    page.mouse.down()
    # Several intermediate moves so the live rubber-band (pointermove) fires like a real drag.
    page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=6)
    page.mouse.move(x1, y1, steps=6)
    page.mouse.up()


def _shot(page: Page, name: str) -> None:
    path = OUT_DIR / name
    page.screenshot(path=str(path))
    logger.info("wrote {}", path.relative_to(OUT_DIR.parent.parent))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        logger.info("opening {}", BASE_URL)
        page.goto(BASE_URL, wait_until="networkidle")
        # Pin the layout height so the rail/panel scroll (as designed) instead of stretching the
        # canvas off-screen when a tall config form is present.
        page.add_style_tag(content=LAYOUT_HEIGHT_CAP)

        # Bootstrap populates the method list and loads the first image; wait for "Ready".
        page.wait_for_function(
            "document.getElementById('status')?.textContent?.startsWith('Ready')",
            timeout=30_000,
        )

        # 1. Overview: load the demo scene but choose NO method yet, so the UI-01 gate is visible
        #    ("pick a method first") over a meaningful image rather than the default photo.
        with page.expect_response(lambda r: IMAGE_ID.split("/")[-1] in r.url, timeout=30_000):
            page.select_option("#image", IMAGE_ID)
        page.wait_for_timeout(500)  # let the image paint
        _shot(page, "01-overview.png")

        # 2. Pick a method — the config form is rebuilt from the method's schema and drawing is
        #    enabled (aria-disabled flips to "false" once an image is loaded AND a method chosen).
        page.select_option("#method", METHOD)
        page.wait_for_selector('#stage[aria-disabled="false"]', timeout=30_000)
        page.wait_for_timeout(300)
        _shot(page, "02-method-selected.png")

        # 3. Draw the exemplar box — this auto-runs the search — then wait for the overlay.
        _draw_box(page)
        page.wait_for_function(
            "document.getElementById('status')?.textContent?.includes('match(es)')",
            timeout=60_000,
        )
        page.wait_for_timeout(600)  # let the heatmap + result boxes finish compositing
        _shot(page, "03-results-overlay.png")

        # 3b. A tighter crop of just the canvas stage (image + matches + overlay toggles).
        page.locator(".stage-wrap").screenshot(path=str(OUT_DIR / "03b-results-canvas.png"))
        logger.info("wrote docs/assets/ui/03b-results-canvas.png")

        browser.close()
    logger.info("done — {} screenshots in {}", len(list(OUT_DIR.glob("*.png"))), OUT_DIR)


if __name__ == "__main__":
    main()
