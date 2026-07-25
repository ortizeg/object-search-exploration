"""Tests for the sample-run renderer.

The load-bearing assertions: the manifest is a fixed query set, the renderer is driven by the
registry (so later methods get samples for free), and the gallery regenerates byte-identically
(a Phase 2 success criterion).
"""

from __future__ import annotations

from pathlib import Path

from object_search.samples import SAMPLE_MANIFEST, render_samples
from object_search.schemas import ExemplarBox
from object_search.search import list_methods


def test_manifest_is_a_fixed_set_of_at_least_four_queries() -> None:
    assert len(SAMPLE_MANIFEST) >= 4
    assert all(isinstance(box, ExemplarBox) for box in SAMPLE_MANIFEST.values())


def test_render_writes_a_dir_per_registered_method(tmp_path: Path) -> None:
    written = render_samples(out_root=tmp_path)

    # Driven by the registry, not a hardcoded method list: every registered method appears.
    registered = {spec.name for spec in list_methods()}
    produced = {p.parent.name for p in written}
    assert produced == registered
    assert "ncc" in produced


def test_render_writes_pngs_and_an_index_for_ncc(tmp_path: Path) -> None:
    render_samples(["ncc"], out_root=tmp_path)

    ncc_dir = tmp_path / "ncc"
    pngs = sorted(ncc_dir.glob("*.png"))
    assert len(pngs) == len(SAMPLE_MANIFEST)

    index = ncc_dir / "index.md"
    assert index.is_file()
    index_text = index.read_text(encoding="utf-8")
    # A per-image row for every manifest entry.
    for image_id in SAMPLE_MANIFEST:
        assert f"{image_id}.png" in index_text


def test_two_renders_are_byte_identical(tmp_path: Path) -> None:
    """Success criterion 3: the gallery regenerates byte-for-byte, PNGs and index alike."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    files_a = render_samples(["ncc"], out_root=first)
    files_b = render_samples(["ncc"], out_root=second)

    rel_a = sorted(p.relative_to(first) for p in files_a)
    rel_b = sorted(p.relative_to(second) for p in files_b)
    assert rel_a == rel_b, "the two renders produced different file sets"

    for rel in rel_a:
        assert (first / rel).read_bytes() == (second / rel).read_bytes(), f"{rel} differs"


def test_lattice_sample_finds_multiple_instances(tmp_path: Path) -> None:
    """The committed lattice sample should visibly find the repeated instances, not one box."""
    render_samples(["ncc"], out_root=tmp_path)
    # index.md records the instance count; the plain lattice must report more than one.
    index_text = (tmp_path / "ncc" / "index.md").read_text(encoding="utf-8")
    lattice_row = next(
        line for line in index_text.splitlines() if line.startswith("| [lattice-plain]")
    )
    # Column order: | image | outcome | instances found | threshold |
    found = int(lattice_row.split("|")[3].strip())
    assert found > 1
