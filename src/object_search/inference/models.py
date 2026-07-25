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
        sha256=None,
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
