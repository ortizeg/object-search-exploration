"""Scripted, reproducible export of OWLv2's image-guided vision graph -> ONNX (Method 4).

Run this ONLY in the ``export`` pixi environment, which carries ``transformers`` and ``torch``::

    pixi run -e export export-owlv2

    # ... or export a locally fine-tuned checkpoint to a SEPARATE artifact, leaving the shipped
    # pretrained owlv2_base_patch16.onnx on disk untouched (quick task 260801-8zy):
    pixi run -e export python scripts/export_owlv2.py \
        --checkpoint models/finetune/owlv2-floorplans-headonly \
        --out owlv2_base_patch16_floorplans_ft.onnx

With no flags the behaviour is byte-identical to what it was before those flags existed: the same
registry key, the same destination, the same ``_verify_graph`` assertions.

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
  ``[0, 1]``;
* ``logit_shift`` / ``logit_scale`` f32 ``[batch, num_patches, 1]`` -- the model's own learned,
  query-independent per-patch score-calibration terms (``Owlv2ClassPredictionHead``'s
  ``logit_shift``/``logit_scale`` Linear(1) layers over the pre-projection vision features).
  At the 960/16 grid, ``num_patches == 3600``.

The concrete ``3600`` value is pinned by the model-free method tests (they feed exactly
``[1, 3600, 512]`` / ``[1, 3600, 4]`` / ``[1, 3600, 1]`` tensors); under ``dynamic_axes`` the patch
dim exports as symbolic, so this verifier asserts only what the graph pins: four outputs with the
documented names, all rank 3, and static last dims ``512``, ``4``, ``1``, ``1``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

# Make the repo's `src/` importable when this script is run directly (not as a module).
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from object_search.inference import models  # noqa: E402  (after sys.path bootstrap)

# The verified static last dims of the four outputs (see the module docstring / library review).
_EXPECTED_CLASS_EMBED_DIM = 512  # OWLv2-base projection dim
_EXPECTED_BOX_DIM = 4  # (cx, cy, w, h)
_EXPECTED_CALIBRATION_DIM = 1  # logit_shift / logit_scale: one scalar per patch


def _verify_graph(onnx_path: Path) -> None:
    """Load the exported graph with ``onnx`` and assert its output contract, or raise.

    Asserts only what the graph pins under ``dynamic_axes``: four outputs named ``class_embeds``,
    ``pred_boxes``, ``logit_shift``, ``logit_scale``, all rank 3, with static last dims 512, 4, 1, 1.
    The patch dim exports as symbolic and so is NOT asserted here -- it is pinned by the model-free
    method tests.
    """
    import onnx  # export env only; not a runtime dependency

    model = onnx.load(str(onnx_path))
    outputs = list(model.graph.output)
    by_name = {o.name: o for o in outputs}
    expected_names = {"class_embeds", "pred_boxes", "logit_shift", "logit_scale"}
    if set(by_name) != expected_names:
        raise SystemExit(
            f"OWLv2 export verification failed: expected outputs {expected_names}, "
            f"graph has {[o.name for o in outputs]}"
        )

    def _dims(name: str) -> list[int | str]:
        dims = by_name[name].type.tensor_type.shape.dim
        return [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]

    class_dims, box_dims = _dims("class_embeds"), _dims("pred_boxes")
    shift_dims, scale_dims = _dims("logit_shift"), _dims("logit_scale")
    logger.info(f"class_embeds dims: {class_dims}")
    logger.info(f"pred_boxes dims: {box_dims}")
    logger.info(f"logit_shift dims: {shift_dims}")
    logger.info(f"logit_scale dims: {scale_dims}")

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
    for name, dims in (("logit_shift", shift_dims), ("logit_scale", scale_dims)):
        if len(dims) != 3 or dims[2] != _EXPECTED_CALIBRATION_DIM:
            problems.append(
                f"{name} must be rank-3 [batch, num_patches, {_EXPECTED_CALIBRATION_DIM}]; "
                f"got {dims}"
            )
    if problems:
        raise SystemExit("OWLv2 export verification failed:\n  - " + "\n  - ".join(problems))

    logger.info(
        f"OWLv2 export verified: class_embeds [batch, num_patches, {_EXPECTED_CLASS_EMBED_DIM}], "
        f"pred_boxes [batch, num_patches, {_EXPECTED_BOX_DIM}], logit_shift/logit_scale "
        f"[batch, num_patches, {_EXPECTED_CALIBRATION_DIM}] (num_patches=3600 at the 960 grid)"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the two optional flags. With NEITHER, behaviour is identical to before they existed.

    ``--checkpoint`` swaps only the *weights* the wrapper is built from (a local fine-tuned
    HuggingFace checkpoint dir instead of the hub id); ``--out`` swaps only the destination
    filename inside ``models/``. The wrapper, opset, dynamic axes, and ``_verify_graph`` assertions
    are the same in every case -- a fine-tuned graph must satisfy the SAME output contract, or the
    method that consumes it would be silently reading a different graph.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Export from this local fine-tuned HuggingFace checkpoint dir (as written by "
            "`pixi run -e export finetune-owlv2`) instead of the pinned hub id. Default: the hub."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Destination filename inside models/. Default: the registry entry's dest "
            "(owlv2_base_patch16.onnx for the pretrained export)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Export the OWLv2 vision graph, then verify the graph output contract.

    With no flags this is the shipped pretrained export, unchanged: the same registry key, the same
    dest, the same verification. With ``--checkpoint`` it exports fine-tuned weights through the
    identical wrapper, to a separate file, leaving the pretrained artifact on disk untouched.
    """
    args = _parse_args(argv)
    spec = models.MODEL_REGISTRY["owlv2-base-patch16"]
    if args.out is not None:
        spec = spec.model_copy(update={"dest": args.out})

    if args.checkpoint is None:
        logger.info(f"Exporting {spec.key} (Apache-2.0) -> {spec.dest} at imgsz=960, opset=17")
        dest = models.fetch(spec, force=True)
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise SystemExit(
                f"No HuggingFace checkpoint at {checkpoint} (expected a config.json). Produce one "
                f"with: pixi run -e export finetune-owlv2 --out {args.checkpoint}"
            )
        logger.info(
            f"Exporting FINE-TUNED OWLv2 from {checkpoint} -> {spec.dest} at imgsz=960, opset=17"
        )
        # Call the exporter directly (not through `fetch`) so the checkpoint is explicit here
        # rather than smuggled through an env var: this script's flags ARE the contract.
        dest = models._export_owlv2(spec, models.models_dir() / spec.dest, checkpoint=checkpoint)

    if not dest.is_file():
        raise SystemExit(
            f"OWLv2 export did not produce {dest}. This script must run in the `export` pixi env "
            f"(torch + transformers): pixi run -e export export-owlv2"
        )

    logger.info(f"Exported to {dest} (sha256={models.provenance.file_sha256(dest)})")
    # The SAME verification runs on a fine-tuned graph: the output contract must not drift.
    _verify_graph(dest)
    if args.checkpoint is None:
        logger.info(
            "Next: pin this sha256 into MODEL_REGISTRY['owlv2-base-patch16'].sha256 so a "
            "byte-different re-export refuses to install (EVAL-09)."
        )
    else:
        logger.info(
            "Fine-tuned artifact: its sha256 is NOT pinned in the registry (it is a hash of one "
            "local run, not a reproducible source); record it in the report instead."
        )


if __name__ == "__main__":
    main()
