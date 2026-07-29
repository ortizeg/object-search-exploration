"""Committed split manifests for the research datasets (EVAL-22, D-02/D-03/D-04).

A research dataset's raw images are gitignored and fetched, but **which image is in which split**
is a committed, diffable fact -- so the manifests live under ``dataset_splits/<dataset>.split.json``
(outside the gitignored ``datasets/`` tree, so CI reads them with no fetch). They are kept out of
``conf/`` on purpose: Hydra treats every ``conf/`` subdirectory as a config group, so a
``conf/datasets/`` dir would collide with the ``datasets`` sweep key and break ``bench-research``.

The manifest is a frozen Pydantic model so a typo (an unknown ``val_strategy``, a non-tuple split)
fails loudly at load. ``val_strategy`` records *how* the val split was obtained, which is not
cosmetic: ``"native"`` (FSCD-147 ships one), ``"seeded-carve"`` (RPINE / FSCD-LVIS carve one from
train, seeded from config -- D-03), and ``"test-only"`` (CARPK / PUCPR+ have no tuning split at
all -- D-04) are three different protocols, and reporting them as if they were the same would
misstate whether test was ever contaminated by tuning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from object_search.provenance import repo_root

ValStrategy = Literal["native", "seeded-carve", "test-only"]

# Committed manifests live here, outside BOTH the gitignored datasets/ tree AND the Hydra config
# dir: conf/datasets/ would register as a Hydra config group and collide with the `datasets` sweep
# key on `pixi run bench-research`. No .gitignore negation is needed (dataset_splits/ is unignored).
_MANIFEST_DIR = Path("dataset_splits")

# The config seed and val fraction the committed seeded-carve manifests were built from, so the
# build is reproducible and the manifests are byte-stable (D-03/D-11). A different seed here moves
# only the train<->val partition; the test split is never derived from the seed.
RESEARCH_VAL_SEED = 0
RESEARCH_VAL_FRACTION = 0.2

# How each dataset's val split is obtained (D-02/D-03/D-04). This is the one place the protocol per
# dataset is declared; the report must state it honestly.
_VAL_STRATEGY: Mapping[str, ValStrategy] = {
    "fscd147": "native",
    "fscd_lvis": "seeded-carve",
    "rpine": "seeded-carve",
    "carpk": "test-only",
    "pucpr_plus": "test-only",
    # Floor-plans ship a native valid split, so val is native (no carve, no contamination). Train is
    # intentionally not converted (exemplar methods do no training), so the manifest's train is
    # empty and only val (tuning) + test (frozen eval) carry ids.
    "floorplans-door": "native",
    "floorplans-window": "native",
}


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
    """Load ``dataset_splits/<dataset>.split.json`` into a validated :class:`ResearchSplitManifest`.

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


class NativeSplits(BaseModel):
    """A dataset's native id lists, before any val is carved -- the input to a manifest build.

    Frozen so the source ids a manifest is derived from cannot be mutated behind the builder's
    back. ``val`` is empty for datasets whose native release ships no val (RPINE, FSCD-LVIS unseen);
    a val slice is then carved from ``train`` deterministically (D-03).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    train: tuple[str, ...] = ()
    val: tuple[str, ...] = ()
    test: tuple[str, ...] = ()


def carve_val(
    train_ids: Sequence[str], *, seed: int, val_fraction: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically carve a val slice from ``train_ids`` (D-03/D-11).

    The single shared val-carve helper, reused by RPINE and FSCD-LVIS (Rule of Three: two concrete
    uses justify one offering -- not a moment sooner). The **test split is never passed in and never
    touched**: carving moves only the train<->val boundary.

    Determinism is the whole point and is guaranteed two ways: ``train_ids`` is sorted to a
    canonical order first (so input order cannot change the result), then a NumPy permutation is
    drawn from ``np.random.default_rng(seed)`` -- **never** ``cv2.setRNGSeed``, which controls
    nothing here (D-11). The same seed therefore yields the byte-identical
    ``(train_remainder, val)`` tuple, and a different seed yields a different val membership.

    Args:
        train_ids: The native train ids to split. Order-independent (sorted internally).
        seed: The config seed the permutation is drawn from.
        val_fraction: Fraction of train carved into val, in ``[0, 1]``; the count is rounded.

    Returns:
        ``(train_remainder, val)`` -- both sorted, disjoint, together equal to ``set(train_ids)``.

    Raises:
        ValueError: If ``val_fraction`` is outside ``[0, 1]``.
    """
    if not (0.0 <= val_fraction <= 1.0):
        raise ValueError(f"val_fraction={val_fraction} must be in [0, 1]")
    ordered = sorted(set(train_ids))
    if not ordered:
        return (), ()
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(ordered))
    n_val = round(len(ordered) * val_fraction)
    val_positions = {int(i) for i in permutation[:n_val]}
    val = tuple(ordered[i] for i in sorted(val_positions))
    remainder = tuple(ordered[i] for i in range(len(ordered)) if i not in val_positions)
    return remainder, val


def build_manifest(
    dataset: str,
    native: NativeSplits,
    *,
    seed: int = RESEARCH_VAL_SEED,
    val_fraction: float = RESEARCH_VAL_FRACTION,
    provenance_ref: str = "datasets/provenance.json",
) -> ResearchSplitManifest:
    """Build one dataset's :class:`ResearchSplitManifest` from its native splits (D-02/03/04).

    Applies the dataset's declared ``_VAL_STRATEGY``:

    * ``native`` -- use the native train/val/test as-is (FSCD-147; de-duplication must already have
      been applied to ``native``, so leaked/duplicate ids never reach here).
    * ``seeded-carve`` -- carve val from train via :func:`carve_val` (RPINE, FSCD-LVIS unseen); test
      is taken straight from ``native.test``, independent of ``seed``.
    * ``test-only`` -- train and val are forced empty (CARPK, PUCPR+; no tuning at all).

    Every split is sorted so the manifest is byte-stable and diffable.
    """
    strategy = _VAL_STRATEGY.get(dataset, "seeded-carve")
    if strategy == "test-only":
        train: tuple[str, ...] = ()
        val: tuple[str, ...] = ()
    elif strategy == "native":
        train = tuple(sorted(native.train))
        val = tuple(sorted(native.val))
    else:  # seeded-carve
        remainder, carved = carve_val(native.train, seed=seed, val_fraction=val_fraction)
        train = tuple(sorted(remainder))
        val = tuple(sorted(carved))
    return ResearchSplitManifest(
        dataset=dataset,
        seed=seed,
        val_strategy=strategy,
        train=train,
        val=val,
        test=tuple(sorted(native.test)),
        provenance_ref=provenance_ref,
    )


def write_split_manifest(manifest: ResearchSplitManifest, *, root: Path | None = None) -> Path:
    """Write ``manifest`` to ``dataset_splits/<dataset>.split.json`` with sorted keys (D-11).

    Sorted-key JSON with a trailing newline mirrors the provenance canonical-JSON discipline, so the
    committed manifest is stable across regenerations and diffs cleanly.
    """
    path = _manifest_path(manifest.dataset, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote split manifest for {} to {}", manifest.dataset, path)
    return path


def build_all_manifests(
    native_splits: Mapping[str, NativeSplits],
    *,
    seed: int = RESEARCH_VAL_SEED,
    val_fraction: float = RESEARCH_VAL_FRACTION,
    provenance_ref: str = "datasets/provenance.json",
    root: Path | None = None,
    write: bool = True,
) -> dict[str, ResearchSplitManifest]:
    """Build (and optionally write) a manifest for every dataset in ``native_splits``.

    Two same-seed calls produce byte-identical seeded-carve manifests; the test split membership in
    every manifest is independent of ``seed`` (only the train<->val partition moves) -- the
    seed-stability guarantee EVAL-22 requires. Pass ``write=False`` to build in memory (e.g. to
    compare two seeds without touching disk).
    """
    manifests: dict[str, ResearchSplitManifest] = {}
    for dataset, native in native_splits.items():
        manifest = build_manifest(
            dataset,
            native,
            seed=seed,
            val_fraction=val_fraction,
            provenance_ref=provenance_ref,
        )
        if write:
            write_split_manifest(manifest, root=root)
        manifests[dataset] = manifest
    return manifests
