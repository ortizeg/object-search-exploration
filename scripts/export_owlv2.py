"""Scripted, reproducible export of OWLv2's image-guided vision graph -> ONNX (Method 4).

Run this ONLY in the ``export`` pixi environment, which carries ``transformers`` and ``torch``::

    pixi run -e export export-owlv2

Why a standalone script and not just ``fetch-models``
-----------------------------------------------------
``fetch-models`` (``object_search.inference.models._export_owlv2``) also performs this export as
part of the registry machinery. This script is the *named, self-contained* export artifact the
phase brief asks for: it runs the export with the exact wrapper and then **verifies the exported
graph** before it is trusted, so a silently-wrong export (wrong output names, a rank/last-dim
regression, a renamed head) fails loudly here rather than at first inference.

Apache-2.0 -- the load-bearing difference from FastSAM
------------------------------------------------------
Unlike FastSAM (AGPL-3.0), OWLv2 and its exporter (``transformers`` + ``torch``) are **Apache-2.0**,
so the produced ``.onnx`` carries no copyleft or §13 obligation. The exporter still lives only in
the ``export`` env because it needs ``torch``, which the runtime package deliberately never imports
("ONNX Runtime for every learned model"). See ``docs/library-reviews/owlv2.md``.

The export contract (to be RUNTIME-VERIFIED here on first run, then pinned)
--------------------------------------------------------------------------
``google/owlv2-base-patch16-ensemble`` wrapped to a single ``pixel_values`` input at ``960x960``
and exported at ``opset=17`` yields a graph with:

* input ``pixel_values`` f32 NCHW ``[batch, 3, 960, 960]`` (spatial dims static 960);
* ``class_embeds`` f32 ``[batch, num_patches, 512]`` -- projected per-patch class embeddings;
* ``pred_boxes`` f32 ``[batch, num_patches, 4]`` -- per-patch ``(cx, cy, w, h)`` normalized to
  ``[0, 1]``. At the 960/16 grid, ``num_patches == 3600``.

The concrete ``3600`` value is pinned by the model-free method tests (they feed exactly
``[1, 3600, 512]`` / ``[1, 3600, 4]`` tensors); under ``dynamic_axes`` the patch dim exports as
symbolic, so this verifier asserts only what the graph pins: two outputs with the documented names,
ranks 3 and 3, and static last dims ``512`` and ``4``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Make the repo's `src/` importable when this script is run directly (not as a module).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from object_search.inference import models  # noqa: E402  (after sys.path bootstrap)

# The verified static last dims of the two outputs (see the module docstring / library review).
_EXPECTED_CLASS_EMBED_DIM = 512  # OWLv2-base projection dim
_EXPECTED_BOX_DIM = 4  # (cx, cy, w, h)


def _verify_graph(onnx_path: Path) -> None:
    """Load the exported graph with ``onnx`` and assert its output contract, or raise.

    Asserts only what the graph pins under ``dynamic_axes``: two outputs named ``class_embeds`` and
    ``pred_boxes``, both rank 3, with static last dims 512 and 4. The patch dim exports as symbolic
    and so is NOT asserted here -- it is pinned by the model-free method tests.
    """
    import onnx  # export env only; not a runtime dependency

    model = onnx.load(str(onnx_path))
    outputs = list(model.graph.output)
    by_name = {o.name: o for o in outputs}
    if set(by_name) != {"class_embeds", "pred_boxes"}:
        raise SystemExit(
            "OWLv2 export verification failed: expected outputs {'class_embeds', 'pred_boxes'}, "
            f"graph has {[o.name for o in outputs]}"
        )

    def _dims(name: str) -> list[int | str]:
        dims = by_name[name].type.tensor_type.shape.dim
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]

    class_dims, box_dims = _dims("class_embeds"), _dims("pred_boxes")
    logger.info(f"class_embeds dims: {class_dims}")
    logger.info(f"pred_boxes dims: {box_dims}")

    problems: list[str] = []
    if len(class_dims) != 3 or class_dims[2] != _EXPECTED_CLASS_EMBED_DIM:
        problems.append(
            f"class_embeds must be rank-3 [batch, num_patches, {_EXPECTED_CLASS_EMBED_DIM}]; "
            f"got {class_dims}"
        )
    if len(box_dims) != 3 or box_dims[2] != _EXPECTED_BOX_DIM:
        problems.append(
            f"pred_boxes must be rank-3 [batch, num_patches, {_EXPECTED_BOX_DIM}]; got {box_dims}"
        )
    if problems:
        raise SystemExit("OWLv2 export verification failed:\n  - " + "\n  - ".join(problems))

    logger.info(
        f"OWLv2 export verified: class_embeds [batch, num_patches, {_EXPECTED_CLASS_EMBED_DIM}], "
        f"pred_boxes [batch, num_patches, {_EXPECTED_BOX_DIM}] (num_patches=3600 at the 960 grid)"
    )


def main() -> None:
    """Export the OWLv2 vision graph via the registry, then verify the graph output contract."""
    spec = models.MODEL_REGISTRY["owlv2-base-patch16"]
    logger.info(f"Exporting {spec.key} (Apache-2.0) -> {spec.dest} at imgsz=960, opset=17")
    dest = models.fetch(spec, force=True)

    if not dest.is_file():
        raise SystemExit(
            f"OWLv2 export did not produce {dest}. This script must run in the `export` pixi env "
            f"(torch + transformers): pixi run -e export export-owlv2"
        )

    logger.info(f"Exported to {dest} (sha256={models.provenance.file_sha256(dest)})")
    _verify_graph(dest)
    logger.info(
        "Next: pin this sha256 into MODEL_REGISTRY['owlv2-base-patch16'].sha256 so a "
        "byte-different re-export refuses to install (EVAL-09)."
    )


if __name__ == "__main__":
    main()
