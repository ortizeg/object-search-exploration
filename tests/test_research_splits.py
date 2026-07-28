"""Split manifests: seed-independent test, seed-reproducible carved val (EVAL-22 criterion 3).

The load-bearing guarantee: with the same config seed the carved val is **byte-identical**, while
the **test split is independent of the seed** (only the train<->val partition moves). Proven here on
the committed fixtures, offline, plus a round-trip of every committed manifest through the frozen
schema.
"""

from __future__ import annotations

import json

from object_search.eval.converters.fscd147 import dedup_fscd147, load_native_splits
from object_search.eval.splits import (
    NativeSplits,
    build_all_manifests,
    load_split_manifest,
)
from object_search.provenance import file_sha256, repo_root

_RESEARCH = repo_root() / "tests" / "fixtures" / "research"
_CONF = repo_root() / "dataset_splits"


def _read_split_json(dataset: str) -> dict[str, list[str]]:
    return json.loads((_RESEARCH / dataset / "split.json").read_text(encoding="utf-8"))


def _fixture_native_splits() -> dict[str, NativeSplits]:
    """Native id triples from the committed fixtures -- fscd147 de-duplicated first (D-07)."""
    fscd147_raw = load_native_splits(_RESEARCH / "fscd147")
    hashes = {
        path.stem: file_sha256(path) for path in (_RESEARCH / "fscd147" / "images").glob("*.png")
    }
    fscd147 = dedup_fscd147(fscd147_raw, hashes).splits

    lvis = _read_split_json("fscd_lvis")
    rpine = _read_split_json("rpine")
    return {
        "fscd147": NativeSplits(
            train=tuple(fscd147.train), val=tuple(fscd147.val), test=tuple(fscd147.test)
        ),
        "fscd_lvis": NativeSplits(train=tuple(lvis["train"]), test=tuple(lvis["test"])),
        "rpine": NativeSplits(train=tuple(rpine["train"]), test=tuple(rpine["test"])),
        "carpk": NativeSplits(test=("carpk-fixture-01", "carpk-fixture-02", "carpk-fixture-03")),
        "pucpr_plus": NativeSplits(),
    }


# --------------------------------------------------------------------------- seed stability


def test_carved_val_is_byte_identical_across_two_same_seed_builds() -> None:
    native = _fixture_native_splits()
    first = build_all_manifests(native, seed=7, write=False)
    second = build_all_manifests(native, seed=7, write=False)
    for dataset in ("fscd_lvis", "rpine"):
        assert first[dataset].val == second[dataset].val
        assert first[dataset].train == second[dataset].train


def test_test_split_is_independent_of_the_seed() -> None:
    native = _fixture_native_splits()
    seed_a = build_all_manifests(native, seed=7, write=False)
    seed_b = build_all_manifests(native, seed=8, write=False)
    # Test membership is identical across two DIFFERENT seeds for every dataset -- the seed reaches
    # only the train<->val partition, never test (D-04). (That the carved val itself DOES move with
    # the seed is proven robustly by the carve_val unit test in test_research_converters.py.)
    for dataset in native:
        assert seed_a[dataset].test == seed_b[dataset].test


def test_test_only_datasets_have_empty_train_and_val() -> None:
    native = _fixture_native_splits()
    built = build_all_manifests(native, write=False)
    for dataset in ("carpk", "pucpr_plus"):
        assert built[dataset].val_strategy == "test-only"
        assert built[dataset].train == ()
        assert built[dataset].val == ()


# --------------------------------------------------------------------------- committed manifests


def test_committed_seeded_carve_manifests_match_a_fresh_build() -> None:
    # The committed fscd_lvis/rpine manifests are exactly what build_all_manifests produces at the
    # repo's RESEARCH_VAL_SEED -- regenerating is a no-op diff (byte-stable, D-11).
    native = _fixture_native_splits()
    built = build_all_manifests(native, write=False)
    for dataset in ("fscd_lvis", "rpine"):
        committed = load_split_manifest(dataset)
        assert committed == built[dataset]


def test_committed_carpk_and_pucpr_are_test_only() -> None:
    for dataset in ("carpk", "pucpr_plus"):
        manifest = load_split_manifest(dataset)
        assert manifest.val_strategy == "test-only"
        assert manifest.train == () and manifest.val == ()


def test_committed_fscd147_manifest_is_native() -> None:
    manifest = load_split_manifest("fscd147")
    assert manifest.val_strategy == "native"
    # Dedup ran before manifest construction: the leaked/duplicate ids are absent (D-07).
    all_ids = set(manifest.train) | set(manifest.val) | set(manifest.test)
    assert "fscd147-fixture-leak" not in all_ids
    assert "fscd147-fixture-dup-b" not in all_ids


def test_every_committed_manifest_round_trips_with_sorted_keys() -> None:
    for manifest_path in sorted(_CONF.glob("*.split.json")):
        dataset = manifest_path.name[: -len(".split.json")]
        model = load_split_manifest(dataset)
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert model.model_dump(mode="json") == on_disk
