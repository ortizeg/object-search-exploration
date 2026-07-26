"""Committed split manifests for the research datasets (EVAL-22, D-02/D-03/D-04).

A research dataset's raw images are gitignored and fetched, but **which image is in which split**
is a committed, diffable fact -- so the manifests live under ``conf/datasets/<dataset>.split.json``
(with the Hydra benchmark config), *outside* the gitignored ``datasets/`` tree, and CI can read
them with no fetch (Task 1, option-a).

The manifest is a frozen Pydantic model so a typo (an unknown ``val_strategy``, a non-tuple split)
fails loudly at load. ``val_strategy`` records *how* the val split was obtained, which is not
cosmetic: ``"native"`` (FSCD-147 ships one), ``"seeded-carve"`` (RPINE / FSCD-LVIS carve one from
train, seeded from config -- D-03), and ``"test-only"`` (CARPK / PUCPR+ have no tuning split at
all -- D-04) are three different protocols, and reporting them as if they were the same would
misstate whether test was ever contaminated by tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from object_search.provenance import repo_root

ValStrategy = Literal["native", "seeded-carve", "test-only"]

# Committed manifests live here, beside the Hydra benchmark config and outside the gitignored
# datasets/ tree, so no .gitignore negation is needed (Task 1, option-a).
_MANIFEST_DIR = Path("conf") / "datasets"


class ResearchSplitManifest(BaseModel):
    """Which image ids are in each split of one research dataset, and how val was obtained.

    Attributes:
        dataset: The dataset key, e.g. ``"carpk"``. Matches the ``datasets/<dataset>/`` tree and
            the ``DATASET_REGISTRY`` entry.
        seed: The config seed that any stochastic split step (val-carving, exemplar sampling) is
            derived from via ``np.random.default_rng(seed)`` -- never ``cv2.setRNGSeed`` (D-11).
            Recorded even for ``test-only`` datasets so the field is uniform and byte-stable.
        val_strategy: How the val split was produced -- ``native`` / ``seeded-carve`` /
            ``test-only``. The protocol the report must state honestly (D-02/03/04).
        train: Train image ids (empty for a test-only dataset).
        val: Val image ids (empty for a test-only dataset).
        test: Test image ids -- the frozen evaluation surface.
        provenance_ref: Path (repo-relative) to the provenance manifest that records the SHA-256 /
            source / licence of the raw bytes these ids name -- ``datasets/provenance.json`` once
            fetched. Ties a committed split to the exact bytes it was defined over (EVAL-09).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str = Field(min_length=1)
    seed: int
    val_strategy: ValStrategy
    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()
    provenance_ref: str = Field(min_length=1)

    def ids_for(self, split: Literal["train", "val", "test"]) -> tuple[str, ...]:
        """The image ids in ``split``. A single accessor so callers cannot mistype a field name."""
        return {"train": self.train, "val": self.val, "test": self.test}[split]


def _manifest_path(dataset: str, root: Path | None = None) -> Path:
    base = root if root is not None else repo_root()
    return base / _MANIFEST_DIR / f"{dataset}.split.json"


def load_split_manifest(dataset: str, root: Path | None = None) -> ResearchSplitManifest:
    """Load ``conf/datasets/<dataset>.split.json`` into a validated :class:`ResearchSplitManifest`.

    Args:
        dataset: The dataset key, e.g. ``"carpk"``.
        root: Optional base dir to resolve the manifest under, for tests writing into ``tmp_path``.

    Returns:
        The frozen, validated manifest.

    Raises:
        FileNotFoundError: If no committed manifest exists for ``dataset``.
    """
    path = _manifest_path(dataset, root)
    if not path.is_file():
        raise FileNotFoundError(f"no committed split manifest for {dataset!r} at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ResearchSplitManifest.model_validate(payload)


def research_image_ids(
    dataset: str, split: Literal["train", "val", "test"], root: Path | None = None
) -> tuple[str, ...]:
    """The image ids for ``dataset``'s ``split``, read from its committed manifest.

    The research analogue of :func:`object_search.eval.labels.chipset_image_ids`: it feeds the same
    ``_run_one`` / ``_aggregate`` benchmark path, so a research sweep is the chipset sweep over a
    different id source rather than a second runner.
    """
    return load_split_manifest(dataset, root).ids_for(split)
