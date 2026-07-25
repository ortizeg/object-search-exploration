"""The sample-run renderer: a fixed query set, run through every registered method (DOC-02).

Why this exists
---------------
A committed gallery of "here is what each method does on a known image" is the fastest way
for a practitioner to see a method's behaviour without running anything. It is driven by a
**fixed, committed** manifest of ``(image_id -> ExemplarBox)`` pairs so the query set is
stable and the whole gallery regenerates from one CLI command (``pixi run samples``).

Adding a method costs zero edits here
-------------------------------------
:func:`render_samples` iterates the **method registry** (``list_methods``), not a hardcoded
list of names. So when Phase 5/6/7 registers a new method, it automatically gains a full set
of sample runs -- one directory, one ``index.md``, one PNG per manifest image -- with **no
change to this file**. That is the payoff of the one-import-per-method registry (INFRA-10).

Determinism (a success criterion)
---------------------------------
Rendering twice must produce byte-identical output. Two things make that true:

1. **The search is deterministic.** Every method's default config is reproducible (the one
   stochastic step, the GMM calibrator, is off by default and seeded when on).
2. **The writers embed no timestamps.** Overlays and panels are drawn with OpenCV and
   encoded with ``cv2.imwrite`` / ``cv2.imencode`` (no embedded time, unlike a default
   matplotlib ``savefig``). The ``index.md`` deliberately carries only reproducible columns
   -- instance count, outcome, threshold -- and **not** wall-clock latency, which cannot be
   byte-stable. Latency is reported live on the CLI instead. A test renders twice into two
   temp dirs and asserts every emitted file is byte-for-byte equal.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_search.explorations import list_explorations
from object_search.explorations.marker_conditioned import MarkerConditionedConfig
from object_search.explorations.marker_conditioned import run as _marker_run
from object_search.provenance import repo_root
from object_search.schemas import BBox, ExemplarBox, SearchResult
from object_search.search import get_method, list_methods
from object_search.search.common import viz
from object_search.search.proposals import ProposalBackend
from object_search.synthetic.generator import (
    DEMO_SPECS,
    MARKER_DEMO_SPECS,
    synthesize,
    synthesize_markers,
)

# The committed default output root. The gallery lives in the repo at docs/samples/<method>/.
_DEFAULT_OUT = repo_root() / "docs" / "samples"

# Composed panels are downscaled to this width before writing. A full three-tile panel at
# native resolution is ~2900px wide and the noisy clutter scene PNG-compresses to >3 MB,
# which is both over the repo's large-file gate and needless for a skimmable doc image.
# INTER_AREA downscaling is deterministic, so byte-identical regeneration is preserved.
_MAX_PANEL_WIDTH = 1440

# The FIXED query set: image_id -> the exemplar box drawn on that image. Each box was chosen
# to land squarely on one real instance of its synthetic scene (a textured region, never a
# flat patch). image_ids match DEMO_SPECS keys, so the scene is synthesized deterministically
# from the same spec. Sorted-key iteration everywhere below keeps the gallery order stable.
SAMPLE_MANIFEST: dict[str, ExemplarBox] = {
    "cluttered-distractors": ExemplarBox(box=BBox(x=37, y=274, w=57, h=57)),
    "lattice-plain": ExemplarBox(box=BBox(x=562, y=292, w=57, h=57)),
    "lattice-touching": ExemplarBox(box=BBox(x=124, y=351, w=73, h=73)),
    "scatter-scaled": ExemplarBox(box=BBox(x=689, y=333, w=59, h=74)),
}


def _load_scene(image_id: str) -> npt.NDArray[np.uint8]:
    """Return the BGR scene for a manifest ``image_id``.

    Every manifest entry is a synthetic ``DEMO_SPECS`` scene, synthesized deterministically
    from its seed -- so the gallery does not depend on any committed image bytes.

    Raises:
        KeyError: If ``image_id`` names no known synthetic spec, so a typo in the manifest is
            a loud failure rather than a silently skipped sample.
    """
    if image_id not in DEMO_SPECS:
        known = ", ".join(sorted(DEMO_SPECS))
        raise KeyError(f"unknown sample image_id {image_id!r}; known synthetic specs: {known}")
    return synthesize(DEMO_SPECS[image_id]).image


def _decode_heatmap(result: SearchResult) -> npt.NDArray[np.uint8] | None:
    """Decode the diagnostics heatmap PNG back to a BGR image, or None when absent."""
    payload = result.diagnostics.similarity_heatmap
    if payload is None:
        return None
    raw = np.frombuffer(base64.b64decode(payload.png_b64), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    return np.asarray(decoded, dtype=np.uint8) if decoded is not None else None


def _render_one(
    scene: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    result: SearchResult,
    method_name: str,
) -> npt.NDArray[np.uint8]:
    """Compose the sample panel: the query, the matches overlay, and the similarity heatmap."""
    query_tile = viz.draw_matches(scene, [], exemplar=exemplar)
    matches_tile = viz.draw_matches(scene, result.matches)
    tiles: list[tuple[str, npt.NDArray[np.uint8]]] = [
        ("query (exemplar)", query_tile),
        (f"{method_name}: {len(result.matches)} match(es)", matches_tile),
    ]
    heatmap = _decode_heatmap(result)
    if heatmap is not None:
        tiles.append(("similarity heatmap", heatmap))
    panel = viz.compose_panel(tiles)
    return _downscale_to_width(panel, _MAX_PANEL_WIDTH)


def _downscale_to_width(image: npt.NDArray[np.uint8], max_width: int) -> npt.NDArray[np.uint8]:
    """Deterministically shrink ``image`` to at most ``max_width`` px wide (never enlarge)."""
    width = image.shape[1]
    if width <= max_width:
        return image
    scale = max_width / width
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def _write_index(out_dir: Path, method_name: str, rows: list[tuple[str, SearchResult]]) -> Path:
    """Write a deterministic, skimmable ``index.md`` table -- reproducible columns only.

    Latency is intentionally omitted: it is wall-clock and cannot be byte-stable, and this
    file is part of the byte-identical-regeneration guarantee. Latency is on the CLI instead.
    """
    lines = [
        f"# `{method_name}` sample runs",
        "",
        "Regenerated by `pixi run samples`. Reproducible from the committed manifest; do not",
        "hand-edit. Latency is reported live on the CLI, not here (it is not byte-stable).",
        "",
        "| image | outcome | instances found | threshold |",
        "| --- | --- | --- | --- |",
    ]
    for image_id, result in rows:
        threshold = "n/a" if result.threshold_applied is None else f"{result.threshold_applied:.4f}"
        lines.append(
            f"| [{image_id}]({image_id}.png) | {result.outcome.value} | "
            f"{len(result.matches)} | {threshold} |"
        )
    lines.append("")
    index_path = out_dir / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    return index_path


def render_samples(
    method_names: list[str] | None = None,
    out_root: Path = _DEFAULT_OUT,
) -> list[Path]:
    """Render every manifest image through each requested method; return the files written.

    Args:
        method_names: Restrict to these registered methods; ``None`` renders **all** of them.
            The default path iterates the registry, which is what makes a newly registered
            method gain samples for free.
        out_root: Root directory; each method writes to ``out_root/<method>/``.

    Returns:
        Every path written (PNG panels and ``index.md`` files), sorted, so a caller can report
        or assert on them deterministically.
    """
    specs = (
        [get_method(name) for name in method_names]
        if method_names is not None
        else list(list_methods())
    )

    written: list[Path] = []
    for spec in sorted(specs, key=lambda s: s.name):
        out_dir = out_root / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[tuple[str, SearchResult]] = []
        for image_id in sorted(SAMPLE_MANIFEST):
            scene = _load_scene(image_id)
            exemplar = SAMPLE_MANIFEST[image_id]
            result = spec.fn(scene, exemplar, spec.config_model())
            panel = _render_one(scene, exemplar, result, spec.name)
            png_path = out_dir / f"{image_id}.png"
            if not cv2.imwrite(str(png_path), panel):
                raise OSError(f"failed to write sample panel to {png_path}")
            written.append(png_path)
            rows.append((image_id, result))
            logger.debug(
                "rendered sample {}/{}: {} match(es)", spec.name, image_id, len(result.matches)
            )
        written.append(_write_index(out_dir, spec.name, rows))

    return sorted(written)


# -- the marker exploration gallery (Milestone 2) -----------------------------------------
#
# The exploration analogue of the method loop above. Where ``render_samples`` iterates the
# *method* registry, ``render_marker_samples`` iterates the *exploration* registry and renders
# the marker-conditioned exploration over its own committed demo images -- the same fixed-query,
# byte-identical-regeneration contract, one directory below ``docs/samples/<exploration>/``.
#
# A model-free marker finder keeps the marker step reproducible and dependency-light; only the
# proposal stage needs a model, and it is injected as ``backend`` so a test can drive the whole
# renderer with a deterministic stub and no ONNX weight at all. The committed gallery is rendered
# by ``render-samples`` with a real (CPU, so reproducible) FastSAM backend.

# The marker-finding method used for the committed marker gallery: model-free and deterministic.
# Named here (not in the api/frontend layers, which stay method-name-free) purely as render config.
_MARKER_FINDER = "ncc"

# Overlay colours (BGR), mirroring frontend/js/overlay.js so the committed panels and the live UI
# read the same: marker boxes + arrows in gold, the chosen proposal in blue, the connector orange.
_MARKER_BOX_BGR = (102, 209, 255)
_CHOSEN_PROPOSAL_BGR = (255, 160, 90)
_MARKER_LINK_BGR = (66, 140, 255)
# Fixed arrow length in scene pixels; LINE_8 (no anti-aliasing) keeps every draw byte-stable.
_MARKER_ARROW_LEN = 26.0


def _draw_marker_result(
    scene: npt.NDArray[np.uint8], result: SearchResult, exploration_name: str
) -> npt.NDArray[np.uint8]:
    """Draw the marker layer on the scene: marker boxes, per-marker pointing arrows and their
    reference points, the chosen proposal per marker, and a connector between the two.

    Everything is drawn with ``cv2.LINE_8`` (no anti-aliasing) so the panel is byte-identical on
    re-render, exactly like the Milestone 1 gallery.
    """
    canvas = viz.draw_matches(scene, [])  # a fresh BGR copy with nothing drawn on it yet
    diagnostics = result.diagnostics
    markers = diagnostics.markers or ()
    references = diagnostics.marker_reference_points or ()
    directions = diagnostics.marker_directions or ()

    for box in markers:
        cv2.rectangle(canvas, (box.x, box.y), (box.x2 - 1, box.y2 - 1), _MARKER_BOX_BGR, 2)
    for match in result.matches:
        b = match.box
        cv2.rectangle(canvas, (b.x, b.y), (b.x2 - 1, b.y2 - 1), _CHOSEN_PROPOSAL_BGR, 2)

    for i, reference in enumerate(references):
        origin = (round(reference.x), round(reference.y))
        if i < len(result.matches):
            chosen = result.matches[i].box
            cv2.line(
                canvas,
                origin,
                (round(chosen.cx), round(chosen.cy)),
                _MARKER_LINK_BGR,
                1,
                cv2.LINE_8,
            )
        direction = directions[i] if i < len(directions) else None
        if direction is not None:
            tip = (
                round(reference.x + direction[0] * _MARKER_ARROW_LEN),
                round(reference.y + direction[1] * _MARKER_ARROW_LEN),
            )
            cv2.arrowedLine(
                canvas, origin, tip, _MARKER_BOX_BGR, 2, line_type=cv2.LINE_8, tipLength=0.35
            )
        cv2.circle(canvas, origin, 3, _MARKER_BOX_BGR, -1, lineType=cv2.LINE_8)

    _ = exploration_name  # reserved for a future per-exploration caption; kept explicit.
    return canvas


def _render_marker_one(
    scene: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    result: SearchResult,
    exploration_name: str,
) -> npt.NDArray[np.uint8]:
    """Compose the marker sample panel: the drawn marker exemplar beside the resolved result."""
    query_tile = viz.draw_matches(scene, [], exemplar=exemplar)
    result_tile = _draw_marker_result(scene, result, exploration_name)
    tiles: list[tuple[str, npt.NDArray[np.uint8]]] = [
        ("query (marker exemplar)", query_tile),
        (f"{exploration_name}: {len(result.matches)} marker(s) resolved", result_tile),
    ]
    panel = viz.compose_panel(tiles)
    return _downscale_to_width(panel, _MAX_PANEL_WIDTH)


def render_marker_samples(
    backend: ProposalBackend | None = None,
    out_root: Path = _DEFAULT_OUT,
) -> list[Path]:
    """Render the marker-conditioned exploration over every marker demo image (Milestone 2).

    Args:
        backend: The proposal backend. ``None`` builds the default FastSAM backend (needs the
            weight); a test injects a deterministic stub so the whole renderer runs model-free.
        out_root: Root directory; the exploration writes to ``out_root/<exploration>/``.

    Returns:
        Every path written (PNG panels and ``index.md`` files), sorted, so a caller can assert on
        them deterministically. Empty if no marker-conditioned exploration is registered.

    The exemplar for each demo image is that image's first marker's exact ground-truth box -- the
    "drawn" marker crop -- derived from the same seed the image is generated from, so the query set
    is fixed without a second committed copy.
    """
    written: list[Path] = []
    config = MarkerConditionedConfig(marker_method=_MARKER_FINDER)
    for spec in list_explorations():
        # Registry-driven: render the exploration whose config is the marker one, by identity of
        # its config model rather than by a hardcoded exploration name.
        if spec.config_model is not MarkerConditionedConfig:
            continue
        out_dir = out_root / spec.name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[tuple[str, SearchResult]] = []
        for image_id in sorted(MARKER_DEMO_SPECS):
            marker_image = synthesize_markers(MARKER_DEMO_SPECS[image_id])
            exemplar = ExemplarBox(box=marker_image.markers[0].box)
            result = _marker_run(marker_image.image, exemplar, config, backend=backend)
            panel = _render_marker_one(marker_image.image, exemplar, result, spec.name)
            png_path = out_dir / f"{image_id}.png"
            if not cv2.imwrite(str(png_path), panel):
                raise OSError(f"failed to write marker sample panel to {png_path}")
            written.append(png_path)
            rows.append((image_id, result))
            logger.debug(
                "rendered marker sample {}/{}: {} marker(s)",
                spec.name,
                image_id,
                len(result.matches),
            )
        written.append(_write_index(out_dir, spec.name, rows))

    return sorted(written)
