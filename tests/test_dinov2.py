"""Tests for :class:`DINOv2Inferencer`.

Two tiers, deliberately:

* **Model-free arithmetic** -- the ``_derive_layout`` register maths and the frozen input-spec
  contract. These feed synthetic token counts, need no weight, and so **run in CI** to gate the
  single riskiest piece of logic (the CLS + register slice).
* **Real-model behaviour** -- the non-transposed-grid proof, the load-time layout probe, and the
  ``dense_tokens`` shape/scale contract. These need the gitignored ``dinov2_small.onnx`` and are
  **skipped when the weight is absent**, exactly as the phase context requires (CI cannot fetch
  the weight).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from object_search.inference import DINOv2Inferencer, ONNXContractError, models
from object_search.inference.dinov2 import (
    DINOV2_EMBED_DIM,
    DINOV2_INPUT_SPEC,
    DINOV2_PATCH,
)

# Pin the CPU provider so the real-model runs are identical run to run (the dev machine also
# exposes CoreML).
_CPU = ["CPUExecutionProvider"]

_MODEL_PATH: Path = models.models_dir() / models.MODEL_REGISTRY["dinov2-small"].dest
_HAVE_MODEL: bool = _MODEL_PATH.is_file()
_needs_model = pytest.mark.skipif(
    not _HAVE_MODEL,
    reason=f"dinov2-small weight absent at {_MODEL_PATH} (gitignored; run pixi run fetch-models)",
)


# --------------------------------------------------------------- model-free: the input contract


def test_input_spec_is_snap14_bicubic_rgb_no_crop() -> None:
    """The frozen spec is the preprocessing contract; pin its verified fields."""
    spec = DINOV2_INPUT_SPEC
    assert spec.input_name == "pixel_values"
    assert spec.resize == "snap-to-multiple"
    assert spec.size_multiple == DINOV2_PATCH == 14
    assert spec.interpolation == "bicubic"
    assert spec.color_order == "RGB"
    assert spec.scale == pytest.approx(1.0 / 255.0)
    assert spec.mean == (0.485, 0.456, 0.406)
    assert spec.std == (0.229, 0.224, 0.225)
    # NCHW and there is no centre-crop policy in the spec at all.
    assert spec.layout == "NCHW"


def test_docstring_states_the_verified_contract() -> None:
    """CLAUDE.md requires the exact pre/post numbers in the inferencer docstring."""
    import object_search.inference.dinov2 as module

    doc = module.__doc__ or ""
    for needle in ("pixel_values", "1/255", "0.485", "0.229", "bicubic", "NO centre-crop"):
        assert needle in doc, f"docstring missing {needle!r}"


# ------------------------------------------------------ model-free: the register-slice arithmetic


def test_derive_layout_square_no_registers() -> None:
    # 224x224 -> 16x16 grid, 257 tokens = 256 patches + 1 CLS, no registers.
    gh, gw, n_register = DINOv2Inferencer._derive_layout(257, 224, 224)
    assert (gh, gw, n_register) == (16, 16, 0)


def test_derive_layout_non_square_no_registers() -> None:
    # 644x896 -> 46x64 grid, 2945 tokens, no registers (a verified MODELS.md resolution).
    gh, gw, n_register = DINOv2Inferencer._derive_layout(46 * 64 + 1, 644, 896)
    assert (gh, gw, n_register) == (46, 64, 0)


def test_derive_layout_derives_register_count_not_hardcoded_one() -> None:
    """A with-registers variant must be *derived*, so the slice becomes [1 + n_register:]."""
    # 224x224 grid is 256 patches; 256 + 1 CLS + 4 registers = 261 tokens.
    gh, gw, n_register = DINOv2Inferencer._derive_layout(261, 224, 224)
    assert (gh, gw) == (16, 16)
    assert n_register == 4  # NOT 0, NOT hardcoded -- derived from the count


def test_derive_layout_raises_on_negative_register_count() -> None:
    """Fewer tokens than 1 CLS + the patch grid is a contract violation, caught at load."""
    with pytest.raises(ONNXContractError, match="token-count mismatch"):
        DINOv2Inferencer._derive_layout(200, 224, 224)  # 200 < 256 + 1


# --------------------------------------------------------------------- real-model: construction


@_needs_model
def test_construction_probes_and_pins_layout() -> None:
    inf = DINOv2Inferencer(_MODEL_PATH, providers=_CPU)
    # dinov2-small has no register tokens and a 384-wide embedding.
    assert inf.n_register == 0
    assert inf.embed_dim == DINOV2_EMBED_DIM == 384


# ------------------------------------------------- real-model: the grid is NOT transposed (proof)


@_needs_model
def test_grid_is_not_transposed_on_a_non_square_off_centre_fixture() -> None:
    """The highest-risk test: a transposed similarity map is a plausible-looking bug.

    The fixture is a **448x896** canvas (14*32 high, 14*64 wide) -- deliberately non-square -- with
    a distinctive bright block placed off-centre at roughly image (row 42, col 700), i.e. patch
    ``(gy=3, gx=50)``. The scene is a multiple of 14 on both sides, so no resize happens and the
    token grid maps 1:1 to patches.

    Identify the object token by **content** -- cosine similarity to an orange prototype extracted
    from a *separate* centred fixture -- not by its position, so the check is not circular. Cosine
    (not raw deviation) is essential: dinov2-small ships **no register tokens** and so exhibits the
    well-known high-norm ViT *artifact* tokens in low-information background regions; those
    outliers dominate an L2-deviation peak but are washed out by cosine, which normalises the norm
    away. Then assert the peak lands at the expected patch. A square image cannot catch a
    transpose; this fixture is *structurally incapable* of passing under one, because:

    * ``gh (32) != gw (64)`` -- a transposed reshape would be ``(64, 32, D)``, a different shape;
    * the expected peak column ``~50`` exceeds ``gh - 1 = 31``, so a transposed ``(64, 32)`` grid
      could never place the peak there.
    """
    height, width = 448, 896  # 14*32 x 14*64
    gy_expected, gx_expected = 3, 50  # row 42 // 14, col 700 // 14
    inf = DINOv2Inferencer(_MODEL_PATH, providers=_CPU)
    orange = (0, 128, 255)  # BGR: a saturated colour, a strong (norm-independent) feature signal

    # Content prototype: a *separate* 224x224 (square) scene with a centred orange block. The
    # centre stays the centre under any transpose, so extracting the prototype is orientation-safe.
    ref: npt.NDArray[np.uint8] = np.full((224, 224, 3), 40, dtype=np.uint8)
    ref[70:154, 70:154] = orange  # patches ~[5:11] on the 16x16 grid
    ref_grid, _, _ = inf.dense_tokens(ref)
    prototype = ref_grid[6:10, 6:10].reshape(-1, DINOV2_EMBED_DIM).mean(axis=0)  # interior only
    prototype = prototype / np.linalg.norm(prototype)

    # Main non-square scene: one saturated 3x3-patch block centred on patch (3, 50).
    scene: npt.NDArray[np.uint8] = np.full((height, width, 3), 40, dtype=np.uint8)
    r0, c0 = (gy_expected - 1) * DINOV2_PATCH, (gx_expected - 1) * DINOV2_PATCH  # 28, 686
    r1, c1 = (gy_expected + 2) * DINOV2_PATCH, (gx_expected + 2) * DINOV2_PATCH  # 70, 728
    scene[r0:r1, c0:c1] = orange

    grid, scale_x, scale_y = inf.dense_tokens(scene)

    # Structural guards: a transpose would change these, so they must hold before we trust the peak.
    assert grid.shape == (32, 64, DINOV2_EMBED_DIM), "grid must be (gh, gw, D), height-first"
    assert grid.shape[0] != grid.shape[1], "fixture must be non-square to detect a transpose"
    # Exact multiples of 14 -> no resize -> unit scale.
    assert scale_x == pytest.approx(1.0)
    assert scale_y == pytest.approx(1.0)

    # Content-based peak: the token most cosine-similar to the orange prototype is the block.
    flat = grid.reshape(-1, DINOV2_EMBED_DIM)
    flat_norm = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    similarity = (flat_norm @ prototype).reshape(grid.shape[:2])
    peak_gy, peak_gx = np.unravel_index(int(np.argmax(similarity)), similarity.shape)

    assert peak_gx > 31, "peak column exceeds gh-1=31, impossible under a (64,32) transpose"
    assert abs(int(peak_gy) - gy_expected) <= 1, f"row {peak_gy} not near expected {gy_expected}"
    assert abs(int(peak_gx) - gx_expected) <= 1, f"col {peak_gx} not near expected {gx_expected}"
    assert int(peak_gy) != int(peak_gx), "gy!=gx so the fixture cannot pass symmetrically"


@_needs_model
def test_dense_tokens_returns_scale_factors_for_a_non_multiple_scene() -> None:
    """A non-multiple-of-14 scene snaps, and the returned scale factors invert that snap."""
    inf = DINOv2Inferencer(_MODEL_PATH, providers=_CPU)
    # 225x320 snaps to 224x322: round(320/14)*14 = 23*14 = 322, round(225/14)*14 = 16*14 = 224.
    # (Both use banker's rounding via round(); 320/14 = 22.857 -> 23, 225/14 = 16.07 -> 16.)
    scene = np.full((225, 320, 3), 60, dtype=np.uint8)
    grid, scale_x, scale_y = inf.dense_tokens(scene)
    assert grid.shape == (224 // 14, 322 // 14, DINOV2_EMBED_DIM)  # (16, 23, 384)
    assert scale_x == pytest.approx(322 / 320)
    assert scale_y == pytest.approx(224 / 225)
