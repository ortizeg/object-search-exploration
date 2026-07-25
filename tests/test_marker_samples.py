"""Tests for the marker exploration sample gallery (Milestone 2).

The load-bearing assertions mirror the Milestone 1 gallery's: the marker-conditioned exploration
is rendered by the registry-iterating renderer, its output lands under
``docs/samples/<exploration>/``, and -- the success criterion -- it regenerates byte-for-byte.

Everything here is model-free: a deterministic stub proposal backend stands in for FastSAM, so
the whole renderer runs with no ONNX weight, exactly like the other learned-method paths are kept
skippable. The committed gallery itself is rendered by ``pixi run samples`` with a real backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from object_search.explorations.marker_conditioned import MarkerConditionedConfig
from object_search.inference import Proposal
from object_search.samples import render_marker_samples
from object_search.schemas.geometry import BBox
from object_search.synthetic.generator import (
    MARKER_DEMO_SPECS,
    save_marker_image,
    synthesize_markers,
)


class _StubBackend:
    """A proposal backend returning a fixed, image-independent set -- fully deterministic."""

    def propose(self, image: npt.NDArray[np.uint8], config: BaseModel) -> list[Proposal]:
        _ = (image, config)
        return [
            Proposal(box=BBox(x=40, y=40, w=30, h=30), mask=None, objectness=0.9),
            Proposal(box=BBox(x=120, y=90, w=26, h=26), mask=None, objectness=0.7),
            Proposal(box=BBox(x=200, y=150, w=34, h=34), mask=None, objectness=0.6),
        ]


def _marker_exploration_name() -> str:
    """The marker exploration's registered name, found via its config model (never hardcoded)."""
    from object_search.explorations import list_explorations

    spec = next(s for s in list_explorations() if s.config_model is MarkerConditionedConfig)
    return spec.name


def test_render_writes_a_dir_for_the_marker_exploration(tmp_path: Path) -> None:
    written = render_marker_samples(backend=_StubBackend(), out_root=tmp_path)
    assert written, "expected the marker exploration to produce sample artifacts"

    produced_dirs = {p.parent.name for p in written}
    assert produced_dirs == {_marker_exploration_name()}

    out_dir = tmp_path / _marker_exploration_name()
    pngs = sorted(out_dir.glob("*.png"))
    assert len(pngs) == len(MARKER_DEMO_SPECS)
    assert (out_dir / "index.md").is_file()


def test_marker_index_lists_every_demo_image(tmp_path: Path) -> None:
    render_marker_samples(backend=_StubBackend(), out_root=tmp_path)
    index_text = (tmp_path / _marker_exploration_name() / "index.md").read_text(encoding="utf-8")
    for image_id in MARKER_DEMO_SPECS:
        assert f"{image_id}.png" in index_text


def test_two_marker_renders_are_byte_identical(tmp_path: Path) -> None:
    """Success criterion: the marker gallery regenerates byte-for-byte, PNGs and index alike."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    files_a = render_marker_samples(backend=_StubBackend(), out_root=first)
    files_b = render_marker_samples(backend=_StubBackend(), out_root=second)

    rel_a = sorted(p.relative_to(first) for p in files_a)
    rel_b = sorted(p.relative_to(second) for p in files_b)
    assert rel_a == rel_b, "the two marker renders produced different file sets"

    for rel in rel_a:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), f"{rel} differs"


def test_marker_assets_write_image_and_sidecar(tmp_path: Path) -> None:
    """`pixi run markers` writes each PNG next to a `.markers.json` GT sidecar."""
    spec = next(iter(sorted(MARKER_DEMO_SPECS)))
    image = synthesize_markers(MARKER_DEMO_SPECS[spec])
    png_path = save_marker_image(image, tmp_path / f"{spec}.png")

    assert png_path.is_file()
    sidecar = png_path.with_suffix(".markers.json")
    assert sidecar.is_file()
    assert '"markers"' in sidecar.read_text(encoding="utf-8")
