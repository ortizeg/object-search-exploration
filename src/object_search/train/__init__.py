"""Torch-free training glue (EVAL-23/EVAL-24) -- the testable half of the fine-tuning path.

The actual fine-tuning loop needs ``torch`` and ``transformers``, which live **only** in the
``export`` pixi env (the runtime package deliberately never imports torch -- "ONNX Runtime for
every learned model"). So the fine-tuning work is split in two:

* ``scripts/finetune_owlv2.py`` -- all torch code: the model, the Hungarian matcher, the losses,
  the optimizer, the epoch loop. Run with ``pixi run -e export finetune-owlv2``. Not linted or
  type-checked (``pixi run lint`` / ``typecheck`` cover ``src/`` and ``tests/`` only), exactly as
  ``scripts/export_owlv2.py`` already is.
* **this package** -- the torch-free glue that decides what the model is *trained on*: the frozen
  config schema, the class-index mapping, the COCO -> OWLv2 target conversion (whose normalization
  denominator is the single most likely correctness bug in the whole task), and the deterministic
  batch order. It is plain numpy + pydantic + loguru, so ``pixi run test`` gates it in the default
  env with **no weights, no GPU, and no torch**.

Nothing here imports torch, and nothing here is imported by the runtime search path.
"""

from object_search.train.owlv2_targets import (
    FLOORPLAN_CLASSES,
    OWLV2_NUM_PATCHES,
    FinetuneConfig,
    ImageTargets,
    coco_to_owlv2_targets,
    deterministic_batches,
)

__all__ = [
    "FLOORPLAN_CLASSES",
    "OWLV2_NUM_PATCHES",
    "FinetuneConfig",
    "ImageTargets",
    "coco_to_owlv2_targets",
    "deterministic_batches",
]
