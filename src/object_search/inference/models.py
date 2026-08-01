"""The model registry and ``fetch-models`` framework (INFRA-11).

Every learned model this project runs is one entry in :data:`MODEL_REGISTRY`, and every entry
was verified at runtime in ``.planning/research/MODELS.md`` -- downloaded, loaded under
onnxruntime, and probed for its graph I/O -- so these are confirmed facts, not guesses.

Weights never enter git. They are gitignored and arrive only through ``fetch-models``, which
downloads the pre-exported artifacts and scripts the FastSAM export. Two licences constrain
how this repo may later be shared, and both are recorded on the spec and in ``LICENSES.md``:

* **FastSAM is AGPL-3.0**, and the exported ``.onnx`` embeds that licence string. Private
  local use triggers nothing; publishing this repo or network-exposing the API fires AGPL §13.
* **SuperPoint weights are MagicLeap non-commercial research-only**, and the derivatives
  clause covers the ONNX file -- so it must never be redistributed. Gitignoring the weights
  (which INFRA-11 mandates anyway) satisfies that at zero cost.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from object_search import provenance

ModelSource = Literal["hf-hub", "github-release", "export"]


class ModelSpec(BaseModel):
    """One learned model: where it comes from, its licence, and where it lands in ``models/``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    source: ModelSource
    repo_id: str
    revision: str | None
    filename: str
    sha256: str | None
    license: str
    license_note: str
    source_note: str
    dest: str
    added_in_phase: int


# MODEL_REGISTRY -- populated now (INFRA-11 is only satisfied by a non-empty registry). Each
# entry records the phase that first needs it in `added_in_phase` and in the comment beside it.
MODEL_REGISTRY: Mapping[str, ModelSpec] = {
    # Phase 6 (Method 3 dino-dense) and Phase 7 (Method 5 region embeddings).
    "dinov2-small": ModelSpec(
        key="dinov2-small",
        source="hf-hub",
        repo_id="onnx-community/dinov2-small-ONNX",
        revision="08c606e3123472a388efa59181b677d428f69bbd",
        filename="onnx/model.onnx",
        # Pinned at the revision above; sha256 recorded from the first verified fetch so it is
        # now a hard integrity gate (EVAL-09 model hash). Deterministic because the revision is
        # pinned -- a byte-different download refuses to install.
        sha256="6266c3cd72db6953cecdcbfeab9422a9f783d96f1a4e296ba70ffbac43b54a18",
        license="Apache-2.0",
        license_note=(
            "Apache-2.0, inherited from facebook/dinov2-small; the onnx-community derivative "
            "declares no licence field of its own, so the inheritance is recorded here."
        ),
        source_note=(
            "Pre-exported fp32 graph, revision pinned. Fallback: optimum-cli export onnx "
            "--model facebook/dinov2-small --task feature-extraction."
        ),
        dest="dinov2_small.onnx",
        added_in_phase=6,
    ),
    # Phase 5 (Method 2 sparse-geo, learned SuperPoint backend).
    "superpoint": ModelSpec(
        key="superpoint",
        source="github-release",
        repo_id="fabio-sim/LightGlue-ONNX",
        revision="v1.0.0",
        filename="superpoint.onnx",
        # Pinned from the first verified fetch of the immutable v1.0.0 release asset; now a hard
        # integrity gate (EVAL-09). A byte-different download refuses to install.
        sha256="234d12c9f523292efb34e0ca513b011050b0c052700da9c01787b9356a1138d2",
        license="Apache-2.0 code / MagicLeap weights non-commercial research-only",
        license_note=(
            "The MagicLeap weights are NON-COMMERCIAL research-only and the DERIVATIVES clause "
            "covers this ONNX file -- never redistribute it. Acceptable because weights are "
            "gitignored (INFRA-11)."
        ),
        source_note=(
            "Frozen v1.0.0 release asset. The repo's main branch NO LONGER exports SuperPoint "
            "standalone (its CLI now emits only the fused extractor+matcher pipeline), so this "
            "frozen asset is the correct source and IDEA.md §14's reference is stale."
        ),
        dest="superpoint.onnx",
        added_in_phase=5,
    ),
    # Phase 7 (Method 5 propose-retrieve, FastSAM proposals).
    "fastsam-s": ModelSpec(
        key="fastsam-s",
        source="export",
        repo_id="ultralytics/FastSAM-s.pt",
        revision="v8.4.0",
        filename="fastsam_s.onnx",
        sha256=None,
        license="AGPL-3.0",
        license_note=(
            "AGPL-3.0, and the exported .onnx itself embeds that licence string. The "
            "export-time-only dependency protects the runtime dependency graph but NOT the "
            "weights: private local use triggers nothing, but publishing this repo or "
            "network-exposing the FastAPI app fires AGPL §13. Revisit before either."
        ),
        source_note=(
            "Scripted export via Ultralytics (FastSAM-s.pt -> FastSAM-s.onnx), which needs "
            "torch. Run in the `export` pixi env: `pixi run -e export fetch-models --only "
            "fastsam-s`."
        ),
        dest="fastsam_s.onnx",
        added_in_phase=7,
    ),
    # Method 4 (owlv2-oneshot): OWLv2 image-conditioned one-shot detection. Apache-2.0 (Google) --
    # the permissive detector adopted after T-Rex2 / Rex-Omni were rejected as non-commercial.
    "owlv2-base-patch16": ModelSpec(
        key="owlv2-base-patch16",
        source="export",
        repo_id="google/owlv2-base-patch16-ensemble",
        revision="main",
        filename="owlv2_base_patch16.onnx",
        # Pinned from the first verified in-env export (EVAL-09): opset 17, legacy exporter,
        # transformers export env. scripts/export_owlv2.py asserted the graph I/O (class_embeds
        # [b, num_patches, 512], pred_boxes [b, num_patches, 4], 3600 patches at 960). A
        # byte-different re-export refuses to install. NOTE: the .onnx is machine-reproducible but
        # torch/transformers version drift can shift bytes; re-pin if the export env is upgraded.
        sha256="2271d85b1467cbedb07bd5b63cf1b0d9d06dc4574e0cd6e2a450ad431a050728",
        license="Apache-2.0",
        license_note=(
            "Apache-2.0 (Google), the permissive tier -- NO AGPL/§13 and NO non-commercial "
            "clause, unlike T-Rex2 / Rex-Omni (IDEA License 1.0, research-only). Adopting it does "
            "not constrain how this repo may be shared."
        ),
        source_note=(
            "Custom-head export (image-guided vision graph -> class_embeds + pred_boxes) via "
            "transformers + torch, both Apache-2.0. Run in the `export` pixi env: `pixi run -e "
            "export export-owlv2`. Image size 960, patch 16 (60x60 = 3600 patches)."
        ),
        dest="owlv2_base_patch16.onnx",
        added_in_phase=8,
    ),
    # The SAME graph as above, exported from a LOCAL fine-tuned checkpoint instead of the hub id
    # (quick task 260801-8zy: does fine-tuning on floor-plan training data fix owlv2-oneshot's
    # floor-plan precision?). A SEPARATE artifact under a separate `dest`: the shipped
    # `owlv2-base-patch16` entry above -- repo_id, revision, dest, and its pinned sha256 -- is
    # deliberately untouched, and `owlv2-oneshot` still resolves that file unless the explicit
    # `OS_OWLV2_MODEL` opt-in says otherwise.
    "owlv2-base-patch16-floorplans-ft": ModelSpec(
        key="owlv2-base-patch16-floorplans-ft",
        source="export",
        repo_id="google/owlv2-base-patch16-ensemble",
        revision=None,  # the weights come from a LOCAL checkpoint dir, not a hub revision
        filename="owlv2_base_patch16_floorplans_ft.onnx",
        # sha256 is None BY DESIGN, not by omission. This artifact is produced from a local
        # fine-tuning run, so a pinned hash would gate on one machine's run rather than on a
        # reproducible source -- it would be a hash of a result, not an integrity check. The
        # sha256 of the run that produced the reported numbers is recorded in the report instead.
        sha256=None,
        license="Apache-2.0",
        license_note=(
            "Apache-2.0, inherited from the google/owlv2-base-patch16-ensemble base weights it is "
            "fine-tuned from -- no AGPL/§13 and no non-commercial clause. Fine-tuning on the "
            "Roboflow floor-plans-500 export does not change the weight licence."
        ),
        source_note=(
            "NOT fetched by `pixi run fetch-models` on a clean box: it is produced locally by two "
            "commands in the `export` pixi env -- (1) `pixi run -e export finetune-owlv2 --out "
            "models/finetune/owlv2-floorplans-headonly`, then (2) `python scripts/export_owlv2.py "
            "--checkpoint models/finetune/owlv2-floorplans-headonly --out "
            "owlv2_base_patch16_floorplans_ft.onnx`. When the checkpoint dir is absent the "
            "exporter logs how to make it and returns the (absent) dest, so fetch_all stays green."
        ),
        dest="owlv2_base_patch16_floorplans_ft.onnx",
        added_in_phase=8,
    ),
}


def models_dir() -> Path:
    """The gitignored ``models/`` directory at the repo root, created if missing."""
    directory = provenance.repo_root() / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _dest_path(spec: ModelSpec) -> Path:
    return models_dir() / spec.dest


def _github_release_url(spec: ModelSpec) -> str:
    return f"https://github.com/{spec.repo_id}/releases/download/{spec.revision}/{spec.filename}"


def fetch(spec: ModelSpec, *, force: bool = False) -> Path:
    """Download or export one model into ``models/``, verifying its sha256 when declared.

    Never leaves a partial file: downloads go to a ``.part`` path and are renamed on success.
    ``source="export"`` requires the ``export`` pixi env; when Ultralytics is unavailable it
    logs an actionable message and returns the (absent) destination rather than failing.
    """
    dest = _dest_path(spec)
    if dest.is_file() and not force:
        if spec.sha256 is None or provenance.file_sha256(dest) == spec.sha256:
            logger.info(f"{spec.key}: already present at {dest}, skipping")
            return dest
        logger.warning(f"{spec.key}: sha256 mismatch at {dest}, re-fetching")

    if spec.source == "export":
        return _export(spec, dest)

    part = dest.with_suffix(dest.suffix + ".part")
    if spec.source == "hf-hub":
        from huggingface_hub import hf_hub_download  # local import: keep hf out of hot paths

        logger.info(f"{spec.key}: downloading {spec.repo_id}@{spec.revision}/{spec.filename}")
        downloaded = hf_hub_download(
            repo_id=spec.repo_id, filename=spec.filename, revision=spec.revision
        )
        part.write_bytes(Path(downloaded).read_bytes())
    else:  # github-release
        url = _github_release_url(spec)
        logger.info(f"{spec.key}: downloading {url}")
        with urllib.request.urlopen(url) as response:  # noqa: S310  (fixed https GitHub URL)
            part.write_bytes(response.read())

    if spec.sha256 is not None:
        actual = provenance.file_sha256(part)
        if actual != spec.sha256:
            part.unlink(missing_ok=True)
            raise ValueError(
                f"{spec.key}: sha256 mismatch (expected {spec.sha256}, got {actual}); "
                f"refusing to install a model that does not match its pinned hash"
            )
    part.replace(dest)
    logger.info(f"{spec.key}: installed at {dest} (sha256={provenance.file_sha256(dest)})")
    return dest


def _export(spec: ModelSpec, dest: Path) -> Path:
    """Dispatch a ``source="export"`` model to its per-model exporter (export pixi env only)."""
    if spec.key == "fastsam-s":
        return _export_fastsam(spec, dest)
    if spec.key == "owlv2-base-patch16":
        return _export_owlv2(spec, dest)
    if spec.key == "owlv2-base-patch16-floorplans-ft":
        checkpoint = owlv2_finetune_checkpoint()
        if not (checkpoint / "config.json").is_file():
            logger.warning(
                "{}: no fine-tuned checkpoint at {}. This artifact is produced locally, not "
                "downloaded: run `pixi run -e export finetune-owlv2 --out {}` first (or point "
                "{} at an existing checkpoint dir). Returning the absent destination.",
                spec.key,
                checkpoint,
                checkpoint,
                _OWLV2_FT_CHECKPOINT_ENV,
            )
            return dest
        return _export_owlv2(spec, dest, checkpoint=checkpoint)
    raise ValueError(f"no exporter registered for {spec.key!r}")


# pragma-excluded: the export bodies run torch + ultralytics, which live ONLY in the `export`
# pixi env and are absent from the runtime/CI env by design, so they cannot execute under
# coverage. The import-guarded fallback (deps absent -> log + return) is still asserted in
# test_models.py; excluding the function stops export-env-only code reading as an untested gap.
def _export_fastsam(spec: ModelSpec, dest: Path) -> Path:  # pragma: no cover
    """Run the scripted FastSAM export, or explain how to when torch is unavailable."""
    try:
        from ultralytics import FastSAM  # export env only; AGPL-3.0, never in the runtime env
    except ImportError:
        logger.warning(
            f"{spec.key}: export needs the `export` pixi env (torch + ultralytics is not in "
            f"the runtime env). Run: pixi run -e export fetch-models --only {spec.key}"
        )
        return dest

    logger.info(f"{spec.key}: exporting FastSAM-s to ONNX (AGPL-3.0)")
    model = FastSAM("FastSAM-s.pt")
    exported = model.export(format="onnx", imgsz=1024, dynamic=True, simplify=False, opset=17)
    Path(exported).replace(dest)
    logger.info(f"{spec.key}: exported to {dest}")
    return dest


# The OWLv2 image-guided vision graph: given one image, emit per-patch class embeddings and boxes.
# 960/16 = 60 patches per side. Torch/transformers (Apache-2.0) are export-env only; the runtime
# package never imports them. See scripts/export_owlv2.py and docs/library-reviews/owlv2.md.
_OWLV2_IMAGE_SIZE = 960
_OWLV2_OPSET = 17

# Where a fine-tuned OWLv2 checkpoint is read from when exporting the *-floorplans-ft artifact.
# An env override in the same spirit as OS_ONNX_PROVIDERS / OS_OWLV2_MODEL: absent changes nothing.
_OWLV2_FT_CHECKPOINT_ENV = "OS_OWLV2_FT_CHECKPOINT"
_OWLV2_FT_CHECKPOINT_DEFAULT = "finetune/owlv2-floorplans-headonly"


def owlv2_finetune_checkpoint() -> Path:
    """The HuggingFace checkpoint dir the fine-tuned OWLv2 export reads from.

    ``$OS_OWLV2_FT_CHECKPOINT`` when set (absolute, or relative to ``models/``); otherwise
    ``models/finetune/owlv2-floorplans-headonly`` -- the default ``finetune-owlv2`` writes to.
    Pure path arithmetic, so CI gates it with no torch and no checkpoint on disk.
    """
    override = os.environ.get(_OWLV2_FT_CHECKPOINT_ENV, "").strip()
    if not override:
        return models_dir() / _OWLV2_FT_CHECKPOINT_DEFAULT
    candidate = Path(override).expanduser()
    return candidate if candidate.is_absolute() else models_dir() / candidate


def _export_owlv2(  # pragma: no cover (export env only)
    spec: ModelSpec,
    dest: Path,
    *,
    checkpoint: Path | None = None,
) -> Path:
    """Export OWLv2's image-guided vision graph (class_embeds + pred_boxes), or explain how to.

    Wraps ``Owlv2ForObjectDetection`` so the ONNX graph takes a single ``pixel_values`` input and
    returns the two tensors the method needs: projected per-patch ``class_embeds`` and normalized
    per-patch ``pred_boxes``. The query-embedding selection and cosine scoring stay in NumPy in the
    method module -- the graph is deliberately just the shared image encoder.

    ``checkpoint`` selects the *weights only*: ``from_pretrained(checkpoint)`` when given, else
    ``from_pretrained(spec.repo_id)``. The wrapper, opset, dynamic axes, and ``.part`` rename are
    identical either way, so a fine-tuned graph is byte-for-byte the same SHAPE as the shipped one
    and ``owlv2_oneshot`` consumes it with zero method changes -- which is the whole reason
    text-conditioned fine-tuning can improve an image-guided method (the two paths share
    ``vision_model`` / ``class_predictor`` / ``box_predictor``).
    """
    try:
        import torch  # export env only (Apache-2.0); never imported by the runtime package
        from transformers import Owlv2ForObjectDetection
    except ImportError:
        logger.warning(
            f"{spec.key}: export needs the `export` pixi env (torch + transformers is not in "
            f"the runtime env). Run: pixi run -e export fetch-models --only {spec.key}"
        )
        return dest

    weights = str(checkpoint) if checkpoint is not None else spec.repo_id
    logger.info(
        f"{spec.key}: exporting OWLv2 image-guided vision graph to ONNX (Apache-2.0) "
        f"from weights {weights}"
    )
    model = Owlv2ForObjectDetection.from_pretrained(weights)
    model.eval()

    class _VisionGraph(torch.nn.Module):  # type: ignore[misc]  # torch is untyped (Any) in this env
        """Single-input wrapper: pixel_values -> (class_embeds, pred_boxes), per patch."""

        def __init__(self, owlv2: Owlv2ForObjectDetection) -> None:
            super().__init__()
            self.owlv2 = owlv2

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Mirrors transformers' Owlv2ForObjectDetection.image_guided_detection encode path.
            feature_map = self.owlv2.image_embedder(pixel_values=pixel_values)[0]
            batch, grid_h, grid_w, hidden = feature_map.shape
            image_feats = feature_map.reshape(batch, grid_h * grid_w, hidden)
            pred_boxes = self.owlv2.box_predictor(image_feats, feature_map)
            _, class_embeds = self.owlv2.class_predictor(image_feats)
            return class_embeds, pred_boxes

    example = torch.zeros((1, 3, _OWLV2_IMAGE_SIZE, _OWLV2_IMAGE_SIZE), dtype=torch.float32)
    part = dest.with_suffix(dest.suffix + ".part")
    torch.onnx.export(
        _VisionGraph(model),
        (example,),
        str(part),
        input_names=["pixel_values"],
        output_names=["class_embeds", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "class_embeds": {0: "batch", 1: "num_patches"},
            "pred_boxes": {0: "batch", 1: "num_patches"},
        },
        opset_version=_OWLV2_OPSET,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter: honours dynamic_axes (the dynamo path uses dynamic_shapes)
    )
    part.replace(dest)
    logger.info(f"{spec.key}: exported to {dest}")
    return dest


def fetch_all(*, force: bool = False) -> Mapping[str, Path]:
    """Fetch every registered model. Returns ``{key: destination path}``."""
    return {key: fetch(spec, force=force) for key, spec in MODEL_REGISTRY.items()}


def verify_all() -> Mapping[str, bool]:
    """For each model, whether its file exists on disk (and matches its sha256 if pinned)."""
    results: dict[str, bool] = {}
    for key, spec in MODEL_REGISTRY.items():
        dest = _dest_path(spec)
        if not dest.is_file():
            results[key] = False
        elif spec.sha256 is None:
            results[key] = True
        else:
            results[key] = provenance.file_sha256(dest) == spec.sha256
    return results
