"""Tests for Method 2 (`sparse-geo`) -- opt-in MIRROR (reflected-pose) handling.

Door symbols in a floor plan are routinely drawn as genuine mirror images (same swing arc,
opposite hinge hand), so half the doors in a plan are reflections of the exemplar. Three things
must hold, and each is easy to get wrong:

1. **The shipped default does not change.** ``allow_mirror`` defaults to ``False``, and with it
   off a reflected fit (``det < 0``) is still rejected as degenerate exactly as before.
2. **Relaxing the mirror gate does not weaken the scale gate.** They are independent rejections;
   a reflected model whose scale is implausible is still rejected when ``allow_mirror`` is on.
3. **The relaxation is end-to-end, not just the last gate.** ``_is_degenerate`` is the LAST place
   a mirrored instance can be lost. Upstream, ``_vote_single_4dof`` and the pairwise voting fit
   compute only the orientation-preserving branch, so a mirrored instance's correspondences
   predict a *wrong* pose and never accumulate into a peak -- they never reach the degeneracy
   gate at all. The end-to-end test is what proves the voting side is wired, because relaxing
   ``_is_degenerate`` alone would leave it passing vacuously.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from object_search.eval.tuning import _TUNING_GRIDS
from object_search.schemas import BBox, ExemplarBox
from object_search.search.sparse_geo import (
    SparseGeoConfig,
    _cast_votes,
    _Correspondence,
    _is_degenerate,
    _model_from_complex,
    search,
)

# --------------------------------------------------------------- the degeneracy gate itself


def _reflected(scale: float) -> object:
    """A reflected similarity (``det < 0``) at the requested scale."""
    return _model_from_complex(complex(scale, 0.0), complex(0.0, 0.0), reflect=True)


def test_mirror_is_still_rejected_under_the_shipped_default() -> None:
    # The contract does not change silently: allow_mirror defaults to False.
    assert SparseGeoConfig().allow_mirror is False
    degenerate, reason = _is_degenerate(_reflected(1.0), min_scale=0.2, max_scale=5.0)
    assert degenerate and "mirror" in reason


def test_mirror_is_accepted_when_allow_mirror_is_set() -> None:
    degenerate, reason = _is_degenerate(
        _reflected(1.0), min_scale=0.2, max_scale=5.0, allow_mirror=True
    )
    assert degenerate is False
    assert reason == ""


def test_relaxing_the_mirror_gate_does_not_weaken_the_scale_gate() -> None:
    # A reflected model OUTSIDE the scale bound is still rejected -- the two gates are
    # independent, and the mirror relaxation must not smuggle an implausible fit through.
    for scale in (0.05, 10.0):
        degenerate, reason = _is_degenerate(
            _reflected(scale), min_scale=0.2, max_scale=5.0, allow_mirror=True
        )
        assert degenerate, f"scale {scale} must still be rejected"
        assert "scale" in reason


# ------------------------------------------------------------- the voting side (upstream)


def _framed_corr(
    index: int,
    crop_xy: tuple[float, float],
    scene_xy: tuple[float, float],
) -> _Correspondence:
    return _Correspondence(
        index=index,
        crop_xy=crop_xy,
        scene_xy=scene_xy,
        crop_scale=1.0,
        crop_angle=0.0,
        scene_scale=1.0,
        scene_angle=0.0,
        distance=0.0,
        rank=0,
    )


def _cast(mode: str, allow_mirror: bool) -> object:
    corrs = tuple(
        _framed_corr(i, crop_xy=(float(i), float(2 * i)), scene_xy=(float(i) + 30.0, float(2 * i)))
        for i in range(4)
    )
    return _cast_votes(
        mode,  # type: ignore[arg-type]
        corrs,
        (50.0, 50.0),
        has_frame=True,
        pairwise_cap=100,
        rng=np.random.default_rng(0),
        allow_mirror=allow_mirror,
    )


@pytest.mark.parametrize("mode", ["single-4dof", "translation-2dof", "pairwise-4dof"])
def test_allow_mirror_off_casts_only_orientation_preserving_votes(mode: str) -> None:
    cast = _cast(mode, allow_mirror=False)
    assert all(not vote.reflect for vote in cast.votes)  # type: ignore[attr-defined]


@pytest.mark.parametrize("mode", ["single-4dof", "pairwise-4dof"])
def test_allow_mirror_on_adds_a_reflected_pose_hypothesis_per_correspondence(mode: str) -> None:
    off = _cast(mode, allow_mirror=False)
    on = _cast(mode, allow_mirror=True)
    reflected = [vote for vote in on.votes if vote.reflect]  # type: ignore[attr-defined]
    assert reflected, "the reflected branch must actually produce votes"
    # The proper branch is untouched: every off-vote still appears, and the reflected votes are
    # ADDITIONAL hypotheses, not a replacement.
    assert len(on.votes) == 2 * len(off.votes)  # type: ignore[attr-defined]


def test_translation_2dof_gains_no_reflected_votes() -> None:
    # translation-2dof pins rotation AND chirality at the identity; there is no reflected
    # translation-only pose, so the flag is a genuine no-op for that mode rather than
    # silently doubling the vote count with duplicates.
    off = _cast("translation-2dof", allow_mirror=False)
    on = _cast("translation-2dof", allow_mirror=True)
    assert len(on.votes) == len(off.votes)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- end-to-end


def _mirrored_scene() -> npt.NDArray[np.uint8]:
    """A SIFT-rich tile, one identical copy, and one horizontally MIRRORED copy.

    The identical copy proves the pipeline is healthy on this scene; the mirrored copy is the
    instance the mirror handling exists to recover.
    """
    rng = np.random.default_rng(11)
    small = rng.integers(0, 256, size=(10, 10), dtype=np.uint8)
    tile = cv2.resize(small, (64, 64), interpolation=cv2.INTER_CUBIC)
    scene = np.full((320, 560), 128, dtype=np.uint8)
    scene[20:84, 20:84] = tile  # the exemplar itself
    scene[20:84, 200:264] = tile  # an identical (orientation-preserving) copy
    scene[200:264, 380:444] = tile[:, ::-1]  # the MIRRORED copy
    return np.ascontiguousarray(np.stack([scene] * 3, axis=-1))


_MIRROR_BOX = BBox(x=380, y=200, w=64, h=64)


def _found_mirror(matches: object, iou: float = 0.3) -> bool:
    return any(m.box.iou(_MIRROR_BOX) >= iou for m in matches)  # type: ignore[attr-defined]


def test_mirrored_instance_is_absent_with_the_flag_off() -> None:
    scene = _mirrored_scene()
    exemplar = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))
    result = search(scene, exemplar, SparseGeoConfig(min_inliers=4))
    # The healthy control: the orientation-preserving copy IS found, so the scene is not the
    # reason the mirrored one is missing.
    assert any(m.box.iou(BBox(x=200, y=20, w=64, h=64)) >= 0.3 for m in result.matches)
    assert not _found_mirror(result.matches), "the default must not accept a reflected fit"


def test_mirrored_instance_is_recovered_with_the_flag_on() -> None:
    scene = _mirrored_scene()
    exemplar = ExemplarBox(box=BBox(x=20, y=20, w=64, h=64))
    result = search(scene, exemplar, SparseGeoConfig(min_inliers=4, allow_mirror=True))
    assert _found_mirror(result.matches), (
        "with allow_mirror the reflected instance must be recovered end to end -- if this fails, "
        "the loss is upstream of _is_degenerate (descriptors or voting), which is the finding"
    )


# ------------------------------------------------------------------------ config contracts


def test_every_sparse_geo_tuning_grid_entry_builds_a_valid_config() -> None:
    # Construction only -- never a search run -- so the SuperPoint grid entries stay CI-safe
    # with no ONNX weights on disk.
    for overrides in _TUNING_GRIDS["sparse-geo"]:
        assert isinstance(SparseGeoConfig(**overrides), SparseGeoConfig)


def test_superpoint_still_resolves_to_translation_2dof_and_rejects_single_4dof() -> None:
    assert SparseGeoConfig(backend="superpoint").voting_mode == "translation-2dof"
    with pytest.raises(ValueError, match="single-4dof"):
        SparseGeoConfig(backend="superpoint", voting_mode="single-4dof")
