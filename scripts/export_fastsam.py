"""Scripted, reproducible export of ``FastSAM-s.pt`` -> ONNX (Phase 7, Method 5).

Run this ONLY in the ``export`` pixi environment, which carries ``ultralytics`` and ``torch``::

    pixi run -e export export-fastsam

Why a standalone script and not just ``fetch-models``
-----------------------------------------------------
``fetch-models`` (``object_search.inference.models._export``) also performs this export as part
of the registry machinery. This script is the *named, self-contained* export artifact the phase
brief asks for: it does the Ultralytics call with the exact verified parameters and then
**verifies the exported graph** before it is trusted, so a silently-wrong export (wrong opset,
non-dynamic axes, a channel-count regression) fails loudly here rather than at first inference.

AGPL-3.0 -- the load-bearing caveat
-----------------------------------
``ultralytics`` is AGPL-3.0 and lives ONLY in the ``export`` environment; the runtime package
never imports it. But the produced ``.onnx`` **still embeds the AGPL licence string** -- the
export-time isolation protects the runtime dependency graph, NOT the weights. Private local use
triggers nothing; publishing this repo or network-exposing the API fires AGPL §13. See
``docs/library-reviews/fastsam.md`` and ``assets/demo/LICENSES.md``.

The verified export contract (runtime-verified in ``.planning/research/MODELS.md``)
-----------------------------------------------------------------------------------
``FastSAM-s.pt`` (ultralytics/assets v8.4.0, 22.7 MB) exported at ``imgsz=1024``,
``dynamic=True``, ``simplify=False``, ``opset=17`` yields a 45.0 MiB graph with:

* input ``images`` f32 NCHW ``[batch, 3, height, width]`` (channel dim static 3);
* ``output0`` f32 ``[batch, 37, anchors]`` -- 4 box + 1 conf + 32 mask coeffs, channels-first;
* ``output1`` f32 ``[batch, 32, mask_h, mask_w]`` -- mask prototypes at stride 4.

At the 1024x1024 operating point ``anchors == 21504`` and ``mask_h == mask_w == 256``:
``output0 [1, 37, 21504]``, ``output1 [1, 32, 256, 256]`` (confirmed by the Ultralytics export
log, which reports ``output shape(s) ((1, 37, 21504), (1, 32, 256, 256))``).

What is static in the graph vs. what is symbolic (verified empirically at export)
--------------------------------------------------------------------------------
Under ``dynamic=True``, ``output0``'s dims export as ``['batch', <symbolic>, 'anchors']`` -- the
**channel dim (37) is symbolic**, not static, because it is produced by a ``Concat`` the exporter
marks dynamic. ``output1`` exports as ``['batch', 32, 'mask_height', 'mask_width']`` -- its
**channel dim (32) IS static**. So this verifier asserts only what the graph actually pins: two
outputs, ranks 3 and 4, and ``output1``'s 32 prototypes. The concrete ``37`` / ``21504`` / ``256``
values are pinned instead by the model-free decoding tests, which feed exactly
``[1, 37, 21504]`` / ``[1, 32, 256, 256]`` tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Make the repo's `src/` importable when this script is run directly (not as a module).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from object_search.inference import models  # noqa: E402  (after sys.path bootstrap)

# The verified static channel counts of the two outputs (see the module docstring / MODELS.md).
_EXPECTED_OUTPUT0_CHANNELS = 37  # 4 box + 1 conf + 32 mask coefficients
_EXPECTED_OUTPUT1_CHANNELS = 32  # mask prototypes


def _verify_graph(onnx_path: Path) -> None:
    """Load the exported graph with ``onnx`` and assert its output contract, or raise.

    Asserts only what the graph actually pins under ``dynamic=True`` (see the module docstring):
    two outputs, ``output0`` rank 3, ``output1`` rank 4 with a **static** 32-channel dim. The
    ``output0`` channel dim (37) exports as symbolic and so is NOT asserted here -- it, the anchor
    count, and the mask resolution are pinned by the model-free decoding tests which feed exactly
    ``[1, 37, 21504]`` / ``[1, 32, 256, 256]`` tensors.
    """
    import onnx  # export env only; not a runtime dependency

    model = onnx.load(str(onnx_path))
    outputs = list(model.graph.output)
    if len(outputs) != 2:
        raise SystemExit(
            f"FastSAM export verification failed: expected 2 outputs, graph has {len(outputs)} "
            f"({[o.name for o in outputs]})"
        )

    def _dims(idx: int) -> list[int | str]:
        dims = outputs[idx].type.tensor_type.shape.dim
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]

    out0_dims, out1_dims = _dims(0), _dims(1)
    logger.info(f"output0 dims: {out0_dims} (channel dim {_EXPECTED_OUTPUT0_CHANNELS} is symbolic)")
    logger.info(f"output1 dims: {out1_dims}")

    problems: list[str] = []
    if len(out0_dims) != 3:
        problems.append(f"output0 must be rank-3 [batch, 37, anchors]; got {out0_dims}")
    if len(out1_dims) != 4 or out1_dims[1] != _EXPECTED_OUTPUT1_CHANNELS:
        problems.append(
            f"output1 must be rank-4 with a static {_EXPECTED_OUTPUT1_CHANNELS}-channel dim; "
            f"got {out1_dims}"
        )
    if problems:
        raise SystemExit("FastSAM export verification failed:\n  - " + "\n  - ".join(problems))

    logger.info(
        f"FastSAM export verified: output0 [batch, {_EXPECTED_OUTPUT0_CHANNELS}, anchors] "
        f"(37 symbolic), output1 [batch, {_EXPECTED_OUTPUT1_CHANNELS}, mask_h, mask_w] "
        f"(anchors=21504, mask=256x256 at the 1024 operating point)"
    )


def main() -> None:
    """Export FastSAM-s to ONNX via the registry, then verify the graph output contract."""
    spec = models.MODEL_REGISTRY["fastsam-s"]
    logger.info(f"Exporting {spec.key} (AGPL-3.0) -> {spec.dest} at imgsz=1024, opset=17")
    dest = models.fetch(spec, force=True)

    if not dest.is_file():
        raise SystemExit(
            f"FastSAM export did not produce {dest}. This script must run in the `export` pixi "
            f"env (torch + ultralytics): pixi run -e export export-fastsam"
        )

    logger.info(f"Exported to {dest} (sha256={models.provenance.file_sha256(dest)})")
    _verify_graph(dest)


if __name__ == "__main__":
    main()
