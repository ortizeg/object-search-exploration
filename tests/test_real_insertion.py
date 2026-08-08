"""Tests for the real-object insertion benchmark set.

No test here touches the network or the gitignored FastSAM weight: the FastSAM boundary
(`extract_cutout`) is exercised through a stub ``ProposalBackend`` with fabricated
:class:`~object_search.inference.Proposal` objects (mirroring how
`decode_fastsam`/`test_proposals.py` are CI-tested with synthetic tensors), and the placement/
compositing logic is exercised with fabricated solid-colour :class:`Cutout` objects, never real
photos -- the same "stub the model boundary" discipline used elsewhere in this repo for model-free
coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from pydantic import BaseModel

from object_search.inference import FastSAMConfig, Proposal
from object_search.schemas.geometry import BBox
from object_search.synthetic import real_insertion as real_insertion_mod
from object_search.synthetic.real_insertion import (
    REAL_BACKGROUND_MANIFEST,
    REAL_BUSY_BACKGROUND_MANIFEST,
    REAL_INSERTION_SPECS,
    REAL_OBJECT_MANIFEST,
    Cutout,
    PhotoProvenance,
    RealInsertionImageSpec,
    _alpha_bbox,
    _select_object_proposal,
    _warp_cutout,
    extract_cutout,
    generate_real_insertion_image,
    write_real_insertion,
)

# -- the committed manifest + REAL_INSERTION_SPECS (no fabrication, the real module data) ---


_ALLOWED_LICENSE_PREFIXES = ("CC0", "CC-BY", "Public Domain")


def test_manifest_has_at_least_ten_objects_and_ten_backgrounds() -> None:
    assert len(REAL_OBJECT_MANIFEST) >= 10
    assert len(REAL_BACKGROUND_MANIFEST) >= 10


def test_manifest_has_at_least_two_busy_backgrounds() -> None:
    assert len(REAL_BUSY_BACKGROUND_MANIFEST) >= 2


def test_manifest_categories_are_unique_across_all_manifests() -> None:
    # Categories are used as dict keys after write_real_insertion merges the clean and busy
    # background manifests into one lookup, so uniqueness must hold globally, not just per file.
    object_categories = [entry.category for entry in REAL_OBJECT_MANIFEST]
    background_categories = [
        entry.category for entry in (*REAL_BACKGROUND_MANIFEST, *REAL_BUSY_BACKGROUND_MANIFEST)
    ]
    assert len(object_categories) == len(set(object_categories))
    assert len(background_categories) == len(set(background_categories))


def test_every_manifest_entry_has_an_allowed_permissive_licence() -> None:
    for entry in (*REAL_OBJECT_MANIFEST, *REAL_BACKGROUND_MANIFEST, *REAL_BUSY_BACKGROUND_MANIFEST):
        assert entry.license.startswith(_ALLOWED_LICENSE_PREFIXES), (
            f"{entry.category}: unexpected licence {entry.license!r}"
        )


def test_real_insertion_specs_has_three_regimes_per_object() -> None:
    assert len(REAL_INSERTION_SPECS) == len(REAL_OBJECT_MANIFEST) * 3
    assert sorted({spec.regime for spec in REAL_INSERTION_SPECS}) == [
        "cluttered",
        "plain",
        "varied",
    ]


def test_plain_regime_is_fixed_scale_and_upright_with_no_distractors() -> None:
    spec = next(s for s in REAL_INSERTION_SPECS if s.regime == "plain")
    assert spec.scale_min == spec.scale_max == 1.0
    assert spec.rotation_deg == 0.0
    assert spec.n_distractors == 0
    assert spec.distractor is None


def test_varied_regime_jitters_scale_and_rotation() -> None:
    spec = next(s for s in REAL_INSERTION_SPECS if s.regime == "varied")
    assert spec.scale_max > spec.scale_min
    assert spec.rotation_deg > 0.0


def test_cluttered_regime_has_a_distractor_from_a_different_category() -> None:
    spec = next(s for s in REAL_INSERTION_SPECS if s.regime == "cluttered")
    assert spec.n_distractors > 0
    assert spec.distractor is not None
    assert spec.distractor != spec.target


def test_cluttered_regime_draws_backgrounds_from_the_busy_pool_only() -> None:
    clean_categories = {entry.category for entry in REAL_BACKGROUND_MANIFEST}
    busy_categories = {entry.category for entry in REAL_BUSY_BACKGROUND_MANIFEST}
    for spec in REAL_INSERTION_SPECS:
        if spec.regime == "cluttered":
            assert spec.background in busy_categories
        else:
            assert spec.background in clean_categories


class _StubBackend:
    """A :class:`~object_search.search.proposals.ProposalBackend` returning fixed proposals."""

    def __init__(self, proposals: list[Proposal]) -> None:
        self._proposals = proposals

    def propose(self, image: npt.NDArray[np.uint8], config: BaseModel) -> list[Proposal]:
        return self._proposals


def _solid_cutout(category: str, *, size: int, bgr: tuple[int, int, int]) -> Cutout:
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[:, :, 0] = bgr[0]
    rgba[:, :, 1] = bgr[1]
    rgba[:, :, 2] = bgr[2]
    rgba[:, :, 3] = 255
    return Cutout(rgba=rgba, category=category)


# -- _select_object_proposal / extract_cutout (the FastSAM boundary, stubbed) -------------------


def test_select_object_proposal_prefers_large_and_centred() -> None:
    small_edge = Proposal(box=BBox(x=0, y=0, w=10, h=10), mask=None, objectness=0.1)
    large_centre = Proposal(box=BBox(x=40, y=40, w=20, h=20), mask=None, objectness=0.1)
    chosen = _select_object_proposal([small_edge, large_centre], image_w=100, image_h=100)
    assert chosen is large_centre


def test_select_object_proposal_empty_list_returns_none() -> None:
    assert _select_object_proposal([], image_w=100, image_h=100) is None


def test_extract_cutout_selects_and_crops_the_best_proposal() -> None:
    image = np.full((100, 100, 3), 5, dtype=np.uint8)
    image[30:70, 20:80] = (0, 0, 255)  # red "object" region

    small_mask = np.zeros((100, 100), dtype=np.float32)
    small_mask[0:10, 0:10] = 1.0  # small, corner -> should lose
    big_mask = np.zeros((100, 100), dtype=np.float32)
    big_mask[30:70, 20:80] = 1.0  # large, centred -> should win

    proposals = [
        Proposal(box=BBox(x=0, y=0, w=10, h=10), mask=small_mask, objectness=0.9),
        Proposal(box=BBox(x=20, y=30, w=60, h=40), mask=big_mask, objectness=0.5),
    ]
    cutout = extract_cutout(image, "widget", backend=_StubBackend(proposals))

    assert cutout is not None
    # One pixel is shaved off each side by the erosion pass that trims the real-photo edge fringe.
    assert cutout.rgba.shape == (38, 58, 4)
    assert np.all(cutout.rgba[:, :, 3] == 255)
    assert np.all(cutout.rgba[:, :, 2] == 255)  # BGR index 2 == red channel


def test_extract_cutout_returns_none_when_backend_finds_nothing() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    assert extract_cutout(image, "widget", backend=_StubBackend([])) is None


def test_extract_cutout_returns_none_when_proposal_has_no_mask() -> None:
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    proposals = [Proposal(box=BBox(x=0, y=0, w=10, h=10), mask=None, objectness=0.9)]
    assert extract_cutout(image, "widget", backend=_StubBackend(proposals)) is None


def test_mask_solidity_full_box_is_one() -> None:
    alpha = np.full((10, 10), 255, dtype=np.uint8)
    assert real_insertion_mod._mask_solidity(alpha) == pytest.approx(1.0)


def test_mask_solidity_empty_is_zero() -> None:
    alpha = np.zeros((10, 10), dtype=np.uint8)
    assert real_insertion_mod._mask_solidity(alpha) == 0.0


def test_mask_solidity_checkerboard_spanning_the_full_box_is_half() -> None:
    # A solid half-block's own tight bbox IS that half-block (solidity 1.0, correctly) -- a
    # checkerboard that still touches every edge is what actually exercises "half of its own box".
    rows, cols = np.indices((10, 10))
    alpha = np.where((rows + cols) % 2 == 0, np.uint8(255), np.uint8(0))
    assert real_insertion_mod._mask_solidity(alpha) == pytest.approx(0.5)


class _ConfThresAwareBackend:
    """A stub backend returning different proposals depending on ``config.conf_thres``.

    Models the real recovery path: FastSAM's default confidence surfaces one (bad) proposal,
    a lower confidence surfaces others -- without needing a real ONNX session.
    """

    def __init__(self, by_conf_thres: dict[float, list[Proposal]], default: list[Proposal]) -> None:
        self._by_conf_thres = by_conf_thres
        self._default = default

    def propose(self, image: npt.NDArray[np.uint8], config: BaseModel) -> list[Proposal]:
        assert isinstance(config, FastSAMConfig)
        return self._by_conf_thres.get(config.conf_thres, self._default)


def _fragmented_mask(size: int) -> npt.NDArray[np.float32]:
    """Isolated corner pixels -- too small to survive the erosion pass at all (returns None)."""
    mask = np.zeros((size, size), dtype=np.float32)
    mask[0, 0] = 1.0
    mask[size - 1, size - 1] = 1.0
    mask[0, size - 1] = 1.0
    return mask


def _scattered_blocks_mask(size: int) -> npt.NDArray[np.float32]:
    """Five isolated 3x3 blocks -- each survives erosion as a single pixel, so the resulting mask
    is non-empty but has very low solidity relative to its own (large) bounding box."""
    mask = np.zeros((size, size), dtype=np.float32)
    for cy, cx in ((2, 2), (2, size - 3), (size - 3, 2), (size - 3, size - 3), (size // 2,) * 2):
        mask[cy - 1 : cy + 2, cx - 1 : cx + 2] = 1.0
    return mask


def _near_full_frame_mask(size: int, margin: int) -> npt.NDArray[np.float32]:
    """Solid except for a thin border -- high solidity AND coverage above the ceiling."""
    mask = np.zeros((size, size), dtype=np.float32)
    mask[margin:-margin, margin:-margin] = 1.0
    return mask


def _small_centred_mask(size: int, half: int) -> npt.NDArray[np.float32]:
    """A modest solid square centred in the frame -- comfortably under the coverage ceiling."""
    mask = np.zeros((size, size), dtype=np.float32)
    centre = size // 2
    mask[centre - half : centre + half, centre - half : centre + half] = 1.0
    return mask


def test_extract_cutout_retries_at_a_different_confidence_when_the_default_pick_is_fragmented() -> (
    None
):
    image = np.full((60, 60, 3), 30, dtype=np.uint8)
    fragmented = [
        Proposal(box=BBox(x=0, y=0, w=60, h=60), mask=_scattered_blocks_mask(60), objectness=0.9)
    ]
    clean = [
        Proposal(box=BBox(x=0, y=0, w=60, h=60), mask=_small_centred_mask(60, 15), objectness=0.3)
    ]
    recovery_conf_thres = real_insertion_mod._CONF_THRES_LADDER[1]
    backend = _ConfThresAwareBackend({recovery_conf_thres: clean}, fragmented)

    cutout = extract_cutout(image, "widget", backend=backend)

    assert cutout is not None
    assert cutout.rgba.shape[:2] != (60, 60)  # the small centred mask, not the scattered one


def test_extract_cutout_retries_when_the_default_pick_is_near_full_frame() -> None:
    image = np.full((60, 60, 3), 30, dtype=np.uint8)
    near_full_frame = [
        Proposal(box=BBox(x=0, y=0, w=60, h=60), mask=_near_full_frame_mask(60, 1), objectness=0.9)
    ]
    tight = [
        Proposal(box=BBox(x=20, y=20, w=20, h=20), mask=_small_centred_mask(60, 10), objectness=0.3)
    ]
    recovery_conf_thres = real_insertion_mod._CONF_THRES_LADDER[1]
    backend = _ConfThresAwareBackend({recovery_conf_thres: tight}, near_full_frame)

    cutout = extract_cutout(image, "widget", backend=backend)

    assert cutout is not None
    assert cutout.rgba.shape[0] < 40 and cutout.rgba.shape[1] < 40


def test_extract_cutout_gives_up_when_every_ladder_rung_is_fragmented() -> None:
    image = np.full((60, 60, 3), 30, dtype=np.uint8)
    fragmented = [
        Proposal(box=BBox(x=0, y=0, w=60, h=60), mask=_fragmented_mask(60), objectness=0.9)
    ]
    backend = _ConfThresAwareBackend({}, default=fragmented)
    assert extract_cutout(image, "widget", backend=backend) is None


def test_extract_cutout_gives_up_when_every_ladder_rung_is_near_full_frame() -> None:
    image = np.full((60, 60, 3), 30, dtype=np.uint8)
    near_full_frame = [
        Proposal(box=BBox(x=0, y=0, w=60, h=60), mask=_near_full_frame_mask(60, 1), objectness=0.9)
    ]
    backend = _ConfThresAwareBackend({}, default=near_full_frame)
    assert extract_cutout(image, "widget", backend=backend) is None


# -- _select_merge_partners / _keep_largest_component (the compound-object case) ------------


def test_box_overlap_area_disjoint_is_zero() -> None:
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=20, y=20, w=10, h=10)
    assert real_insertion_mod._box_overlap_area(a, b) == 0.0


def test_box_overlap_area_partial_overlap() -> None:
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=5, w=10, h=10)
    assert real_insertion_mod._box_overlap_area(a, b) == 25.0  # 5x5 overlap square


def test_select_merge_partners_includes_a_touching_disjoint_confident_candidate() -> None:
    # Mirrors the measured screwdriver case: handle + shaft, touching, both high objectness.
    primary = Proposal(box=BBox(x=100, y=0, w=100, h=100), mask=None, objectness=0.88)
    shaft = Proposal(
        box=BBox(x=0, y=20, w=100, h=20), mask=np.zeros((1, 1), dtype=np.float32), objectness=0.88
    )
    partners = real_insertion_mod._select_merge_partners(primary, [primary, shaft], 300, 300)
    assert partners == [shaft]


def test_select_merge_partners_excludes_a_heavily_overlapping_candidate() -> None:
    # Mirrors the measured tennis-ball regression: a near-duplicate, mostly-overlapping box that
    # extends slightly past the primary must NOT be treated as a separate object part.
    primary = Proposal(box=BBox(x=0, y=0, w=100, h=100), mask=None, objectness=0.74)
    near_duplicate = Proposal(
        box=BBox(x=10, y=10, w=100, h=100), mask=np.zeros((1, 1), dtype=np.float32), objectness=0.68
    )
    partners = real_insertion_mod._select_merge_partners(
        primary, [primary, near_duplicate], 300, 300
    )
    assert partners == []


def test_select_merge_partners_excludes_low_objectness_even_if_touching() -> None:
    primary = Proposal(box=BBox(x=100, y=0, w=100, h=100), mask=None, objectness=0.88)
    weak = Proposal(
        box=BBox(x=0, y=20, w=100, h=20), mask=np.zeros((1, 1), dtype=np.float32), objectness=0.2
    )
    partners = real_insertion_mod._select_merge_partners(primary, [primary, weak], 300, 300)
    assert partners == []


def test_select_merge_partners_excludes_a_far_away_candidate() -> None:
    primary = Proposal(box=BBox(x=0, y=0, w=20, h=20), mask=None, objectness=0.88)
    far = Proposal(
        box=BBox(x=250, y=250, w=20, h=20), mask=np.zeros((1, 1), dtype=np.float32), objectness=0.9
    )
    partners = real_insertion_mod._select_merge_partners(primary, [primary, far], 300, 300)
    assert partners == []


def test_keep_largest_component_drops_small_specks() -> None:
    alpha = np.zeros((50, 50), dtype=np.uint8)
    alpha[5:40, 5:40] = 255  # the main blob
    alpha[45, 45] = 255  # an isolated 1px speck
    cleaned = real_insertion_mod._keep_largest_component(alpha)
    assert cleaned[45, 45] == 0
    assert np.array_equal(cleaned[5:40, 5:40], alpha[5:40, 5:40])


def test_keep_largest_component_is_a_no_op_with_one_component() -> None:
    alpha = np.zeros((20, 20), dtype=np.uint8)
    alpha[5:15, 5:15] = 255
    assert np.array_equal(real_insertion_mod._keep_largest_component(alpha), alpha)


def test_extract_cutout_merges_a_touching_disjoint_high_confidence_part() -> None:
    image = np.full((100, 100, 3), 40, dtype=np.uint8)
    handle_mask = np.zeros((100, 100), dtype=np.float32)
    handle_mask[20:80, 50:90] = 1.0  # right part
    shaft_mask = np.zeros((100, 100), dtype=np.float32)
    shaft_mask[40:60, 10:50] = 1.0  # left part, touching the handle at x=50

    proposals = [
        Proposal(box=BBox(x=50, y=20, w=40, h=60), mask=handle_mask, objectness=0.88),
        Proposal(box=BBox(x=10, y=40, w=40, h=20), mask=shaft_mask, objectness=0.88),
    ]
    cutout = extract_cutout(image, "widget", backend=_StubBackend(proposals))

    assert cutout is not None
    # The merged footprint spans both parts (x from ~10 to ~90), not just the handle (50 to 90).
    assert cutout.rgba.shape[1] > 60


# -- pure compositing helpers ---------------------------------------------------------------


def test_alpha_bbox_offsets_into_canvas_coordinates() -> None:
    alpha = np.zeros((10, 10), dtype=np.uint8)
    alpha[2:6, 3:9] = 255
    assert _alpha_bbox(alpha, x=100, y=50) == BBox(x=103, y=52, w=6, h=4)


def test_alpha_bbox_returns_none_when_mask_is_empty() -> None:
    alpha = np.zeros((10, 10), dtype=np.uint8)
    assert _alpha_bbox(alpha, x=0, y=0) is None


def test_warp_cutout_rotation_grows_the_footprint() -> None:
    rgba = np.zeros((60, 20, 4), dtype=np.uint8)
    rgba[:, :, :3] = 200
    rgba[:, :, 3] = 255
    cutout = Cutout(rgba=rgba, category="rod")

    _, alpha0 = _warp_cutout(cutout, scale=1.0, angle_deg=0.0)
    assert alpha0.shape == (60, 20)

    _, alpha45 = _warp_cutout(cutout, scale=1.0, angle_deg=45.0)
    assert alpha45.shape[0] > 60 or alpha45.shape[1] > 20  # rotated AABB is strictly larger


# -- generate_real_insertion_image (placement / exact-GT invariants) ------------------------


def test_pairwise_boxes_are_non_overlapping() -> None:
    cutout = _solid_cutout("obj", size=25, bgr=(10, 20, 30))
    background = np.full((500, 500, 3), 90, dtype=np.uint8)
    spec = RealInsertionImageSpec(
        image_id="ov", background="bg", target="obj", n_instances=8, seed=3
    )
    result = generate_real_insertion_image(spec, background, {"obj": cutout})
    for i in range(len(result.boxes)):
        for j in range(i + 1, len(result.boxes)):
            assert result.boxes[i].iou(result.boxes[j]) == 0.0


def test_achieved_count_matches_slice_metadata_and_never_exceeds_requested() -> None:
    cutout = _solid_cutout("obj", size=25, bgr=(10, 20, 30))
    background = np.full((300, 300, 3), 60, dtype=np.uint8)
    spec = RealInsertionImageSpec(
        image_id="cnt", background="bg", target="obj", n_instances=6, seed=11
    )
    result = generate_real_insertion_image(spec, background, {"obj": cutout})
    assert len(result.boxes) == result.slice_metadata.true_instance_count
    assert len(result.boxes) <= spec.n_instances


def test_generate_real_insertion_image_is_deterministic() -> None:
    cutout = _solid_cutout("obj", size=25, bgr=(10, 20, 30))
    background = np.full((300, 300, 3), 60, dtype=np.uint8)
    spec = RealInsertionImageSpec(
        image_id="det", background="bg", target="obj", n_instances=5, seed=42
    )
    first = generate_real_insertion_image(spec, background, {"obj": cutout})
    second = generate_real_insertion_image(spec, background, {"obj": cutout})
    assert np.array_equal(first.image, second.image)
    assert first.boxes == second.boxes


def test_distractors_are_pasted_but_excluded_from_ground_truth() -> None:
    target = _solid_cutout("target", size=30, bgr=(0, 0, 255))  # red
    distractor = _solid_cutout("distractor", size=30, bgr=(255, 0, 0))  # blue
    background = np.full((400, 400, 3), 128, dtype=np.uint8)
    spec = RealInsertionImageSpec(
        image_id="dis",
        background="bg",
        target="target",
        n_instances=3,
        seed=7,
        scale_min=1.0,
        scale_max=1.0,
        rotation_deg=0.0,
        n_distractors=3,
        distractor="distractor",
    )
    result = generate_real_insertion_image(
        spec, background, {"target": target, "distractor": distractor}
    )

    assert len(result.boxes) == result.slice_metadata.true_instance_count
    assert len(result.boxes) <= spec.n_instances
    for box in result.boxes:
        region = result.image[box.y : box.y2, box.x : box.x2].reshape(-1, 3)
        mean = region.mean(axis=0)
        assert mean[2] > mean[0]  # red target, not the blue distractor

    blue_mask = np.all(result.image == np.array([255, 0, 0]), axis=-1)
    assert blue_mask.any(), "distractor should be visibly pasted somewhere on the canvas"


# -- _download / fetch_real_photos (network boundary, stubbed) ------------------------------


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


def test_download_returns_bytes_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        real_insertion_mod, "urlopen", lambda request, timeout=30: _FakeResponse(b"hello")
    )
    assert real_insertion_mod._download("https://example.invalid/x.jpg") == b"hello"


def test_download_returns_none_on_url_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(request: object, timeout: int = 30) -> None:
        raise URLError("boom")

    monkeypatch.setattr(real_insertion_mod, "urlopen", _raise)
    assert real_insertion_mod._download("https://example.invalid/x.jpg") is None


def test_fetch_real_photos_writes_files_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    object_entry = PhotoProvenance(
        category="widget",
        title="W",
        file_url="https://example.invalid/widget.jpg",
        page_url="https://example.invalid/wiki/File:widget.jpg",
        author="A",
        license="CC0-1.0",
    )
    background_entry = PhotoProvenance(
        category="table",
        title="T",
        file_url="https://example.invalid/table.png",
        page_url="https://example.invalid/wiki/File:table.png",
        author="B",
        license="CC-BY-4.0",
    )
    monkeypatch.setattr(real_insertion_mod, "REAL_OBJECT_MANIFEST", (object_entry,))
    monkeypatch.setattr(real_insertion_mod, "REAL_BACKGROUND_MANIFEST", (background_entry,))
    monkeypatch.setattr(real_insertion_mod, "REAL_BUSY_BACKGROUND_MANIFEST", ())
    monkeypatch.setattr(real_insertion_mod, "_download", lambda url: b"fake-bytes")

    raw_dir = tmp_path / "raw"
    written = real_insertion_mod.fetch_real_photos(raw_dir)

    assert len(written) == 2
    assert (raw_dir / "objects" / "widget.jpg").read_bytes() == b"fake-bytes"
    assert (raw_dir / "backgrounds" / "table.png").read_bytes() == b"fake-bytes"
    provenance = json.loads((raw_dir / "provenance.json").read_text())
    assert {entry["category"] for entry in provenance} == {"widget", "table"}

    def _fail_if_called(url: str) -> bytes | None:
        raise AssertionError("should not re-download an existing file")

    monkeypatch.setattr(real_insertion_mod, "_download", _fail_if_called)
    written_again = real_insertion_mod.fetch_real_photos(raw_dir)
    assert len(written_again) == 2


def test_fetch_real_photos_skips_a_failed_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    object_entry = PhotoProvenance(
        category="widget",
        title="W",
        file_url="https://example.invalid/widget.jpg",
        page_url="https://example.invalid/wiki/File:widget.jpg",
        author="A",
        license="CC0-1.0",
    )
    monkeypatch.setattr(real_insertion_mod, "REAL_OBJECT_MANIFEST", (object_entry,))
    monkeypatch.setattr(real_insertion_mod, "REAL_BACKGROUND_MANIFEST", ())
    monkeypatch.setattr(real_insertion_mod, "REAL_BUSY_BACKGROUND_MANIFEST", ())
    monkeypatch.setattr(real_insertion_mod, "_download", lambda url: None)

    assert real_insertion_mod.fetch_real_photos(tmp_path / "raw") == []


# -- _raw_photo_path / _load_background / _load_or_build_cutout -----------------------------


def test_raw_photo_path_missing_dir_returns_none(tmp_path: Path) -> None:
    assert real_insertion_mod._raw_photo_path(tmp_path / "nope", "widget") is None


def test_raw_photo_path_finds_first_match(tmp_path: Path) -> None:
    (tmp_path / "widget.jpg").write_bytes(b"x")
    assert real_insertion_mod._raw_photo_path(tmp_path, "widget") == tmp_path / "widget.jpg"


def test_load_background_returns_none_when_missing(tmp_path: Path) -> None:
    entry = PhotoProvenance(
        category="bg", title="t", file_url="u", page_url="p", author="a", license="CC0-1.0"
    )
    assert real_insertion_mod._load_background(entry, tmp_path) is None


def test_load_background_downscales_photos_above_the_working_edge(tmp_path: Path) -> None:
    (tmp_path / "backgrounds").mkdir()
    big = np.full((2000, 3000, 3), 40, dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "backgrounds" / "bg.jpg"), big)
    entry = PhotoProvenance(
        category="bg", title="t", file_url="u", page_url="p", author="a", license="CC0-1.0"
    )
    image = real_insertion_mod._load_background(entry, tmp_path)
    assert image is not None
    assert max(image.shape[:2]) == real_insertion_mod._WORKING_LONG_EDGE


def test_load_or_build_cutout_returns_none_when_raw_photo_missing(tmp_path: Path) -> None:
    entry = PhotoProvenance(
        category="widget", title="t", file_url="u", page_url="p", author="a", license="CC0-1.0"
    )
    result = real_insertion_mod._load_or_build_cutout(
        entry, tmp_path / "raw", tmp_path / "cutouts", force=False, backend=_StubBackend([])
    )
    assert result is None


def test_load_or_build_cutout_rebuilds_when_the_cache_is_corrupt(tmp_path: Path) -> None:
    cutouts_dir = tmp_path / "cutouts"
    cutouts_dir.mkdir()
    raw_dir = tmp_path / "raw"
    (raw_dir / "objects").mkdir(parents=True)
    cv2.imwrite(str(raw_dir / "objects" / "widget.jpg"), np.full((80, 80, 3), 100, dtype=np.uint8))
    # A 3-channel (no alpha) file at the cache path is "corrupt" for this purpose.
    cv2.imwrite(str(cutouts_dir / "widget.png"), np.full((10, 10, 3), 50, dtype=np.uint8))

    entry = PhotoProvenance(
        category="widget", title="t", file_url="u", page_url="p", author="a", license="CC0-1.0"
    )
    # A modest centred mask, not the whole 80x80 frame -- a whole-frame mask is now rejected as
    # "background included" (see the coverage-ceiling tests above).
    mask = _small_centred_mask(80, 20)
    backend = _StubBackend([Proposal(box=BBox(x=0, y=0, w=80, h=80), mask=mask, objectness=0.9)])

    cutout = real_insertion_mod._load_or_build_cutout(
        entry, raw_dir, cutouts_dir, force=False, backend=backend
    )
    assert cutout is not None
    assert cutout.rgba.shape[2] == 4


# -- write_real_insertion (orchestration, with manifests/specs monkeypatched) ---------------


def test_write_real_insertion_skips_a_spec_with_no_target_cutout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    (raw_dir / "backgrounds").mkdir(parents=True)
    (raw_dir / "objects").mkdir(parents=True)
    cv2.imwrite(
        str(raw_dir / "backgrounds" / "table.jpg"), np.full((100, 100, 3), 10, dtype=np.uint8)
    )
    # No object photo on disk for "widget" -> the cutout build fails -> the spec is skipped.

    background_entry = PhotoProvenance(
        category="table", title="t", file_url="u", page_url="p", author="a", license="CC0-1.0"
    )
    spec = RealInsertionImageSpec(
        image_id="real-widget", background="table", target="widget", n_instances=2, seed=1
    )
    monkeypatch.setattr(real_insertion_mod, "REAL_OBJECT_MANIFEST", ())
    monkeypatch.setattr(real_insertion_mod, "REAL_BACKGROUND_MANIFEST", (background_entry,))
    monkeypatch.setattr(real_insertion_mod, "REAL_BUSY_BACKGROUND_MANIFEST", ())
    monkeypatch.setattr(real_insertion_mod, "REAL_INSERTION_SPECS", (spec,))

    written = write_real_insertion(
        tmp_path / "out", raw_dir, tmp_path / "cutouts", backend=_StubBackend([])
    )
    assert written == []


def test_write_real_insertion_emits_sidecars_and_caches_cutouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cutouts_dir = tmp_path / "cutouts"
    (raw_dir / "objects").mkdir(parents=True)
    (raw_dir / "backgrounds").mkdir(parents=True)

    rng = np.random.default_rng(0)
    object_photo = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
    background_photo = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(raw_dir / "objects" / "widget.jpg"), object_photo)
    cv2.imwrite(str(raw_dir / "backgrounds" / "table.jpg"), background_photo)

    object_entry = real_insertion_mod.PhotoProvenance(
        category="widget",
        title="Widget photo",
        file_url="https://example.invalid/widget.jpg",
        page_url="https://example.invalid/wiki/File:widget.jpg",
        author="Test Author",
        license="CC0-1.0",
    )
    background_entry = real_insertion_mod.PhotoProvenance(
        category="table",
        title="Table photo",
        file_url="https://example.invalid/table.jpg",
        page_url="https://example.invalid/wiki/File:table.jpg",
        author="Test Author",
        license="CC0-1.0",
    )
    spec = RealInsertionImageSpec(
        image_id="real-widget", background="table", target="widget", n_instances=3, seed=5
    )

    monkeypatch.setattr(real_insertion_mod, "REAL_OBJECT_MANIFEST", (object_entry,))
    monkeypatch.setattr(real_insertion_mod, "REAL_BACKGROUND_MANIFEST", (background_entry,))
    monkeypatch.setattr(real_insertion_mod, "REAL_BUSY_BACKGROUND_MANIFEST", ())
    monkeypatch.setattr(real_insertion_mod, "REAL_INSERTION_SPECS", (spec,))

    # A modest centred mask, not the whole 80x80 frame -- a whole-frame mask is now rejected as
    # "background included" (see the coverage-ceiling tests above).
    object_mask = _small_centred_mask(80, 20)
    proposals = [Proposal(box=BBox(x=0, y=0, w=80, h=80), mask=object_mask, objectness=0.9)]
    backend = _StubBackend(proposals)

    written = write_real_insertion(out_dir, raw_dir, cutouts_dir, backend=backend)

    assert len(written) == 1
    image_path = out_dir / "real-widget.jpg"
    sidecar = out_dir / "real-widget.gt.json"
    assert image_path.is_file()
    assert sidecar.is_file()
    text = sidecar.read_text(encoding="utf-8")
    assert '"achieved_n"' in text
    assert '"requested_n"' in text
    assert '"exemplar_index"' in text
    assert (cutouts_dir / "widget.png").is_file()  # cutout cache written


def test_write_real_insertion_reuses_cached_cutout_without_backend(tmp_path: Path) -> None:
    """A second call with no ``backend`` reuses the cache -- proves FastSAM is not re-invoked."""
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    cutouts_dir = tmp_path / "cutouts"
    (raw_dir / "objects").mkdir(parents=True)
    (raw_dir / "backgrounds").mkdir(parents=True)

    rng = np.random.default_rng(1)
    cv2.imwrite(
        str(raw_dir / "objects" / "widget.jpg"),
        rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8),
    )
    cv2.imwrite(
        str(raw_dir / "backgrounds" / "table.jpg"),
        rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8),
    )
    cutouts_dir.mkdir(parents=True)
    cv2.imwrite(str(cutouts_dir / "widget.png"), np.full((20, 20, 4), 255, dtype=np.uint8))

    object_entry = real_insertion_mod.PhotoProvenance(
        category="widget",
        title="Widget",
        file_url="https://example.invalid/widget.jpg",
        page_url="https://example.invalid/wiki/File:widget.jpg",
        author="Test Author",
        license="CC0-1.0",
    )
    background_entry = real_insertion_mod.PhotoProvenance(
        category="table",
        title="Table",
        file_url="https://example.invalid/table.jpg",
        page_url="https://example.invalid/wiki/File:table.jpg",
        author="Test Author",
        license="CC0-1.0",
    )
    spec = RealInsertionImageSpec(
        image_id="real-widget", background="table", target="widget", n_instances=2, seed=2
    )

    original_objects = real_insertion_mod.REAL_OBJECT_MANIFEST
    original_backgrounds = real_insertion_mod.REAL_BACKGROUND_MANIFEST
    original_busy_backgrounds = real_insertion_mod.REAL_BUSY_BACKGROUND_MANIFEST
    original_specs = real_insertion_mod.REAL_INSERTION_SPECS
    real_insertion_mod.REAL_OBJECT_MANIFEST = (object_entry,)
    real_insertion_mod.REAL_BACKGROUND_MANIFEST = (background_entry,)
    real_insertion_mod.REAL_BUSY_BACKGROUND_MANIFEST = ()
    real_insertion_mod.REAL_INSERTION_SPECS = (spec,)
    try:
        # backend=None: if the cache were NOT reused, this would try to construct the real
        # FastSAM backend (default_backend()) and raise for the missing gitignored weight.
        written = write_real_insertion(out_dir, raw_dir, cutouts_dir, backend=None)
    finally:
        real_insertion_mod.REAL_OBJECT_MANIFEST = original_objects
        real_insertion_mod.REAL_BACKGROUND_MANIFEST = original_backgrounds
        real_insertion_mod.REAL_BUSY_BACKGROUND_MANIFEST = original_busy_backgrounds
        real_insertion_mod.REAL_INSERTION_SPECS = original_specs

    assert len(written) == 1
