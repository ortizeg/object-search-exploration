"""FSCD-147 de-duplication: the 11-style train<->test leaks + pixel-identical dups (D-07, T-11-07).

Runs entirely on the committed fscd147 fixture, which plants one byte-identical duplicate pair
(``dup-a`` / ``dup-b``, both in val) and one leaked id (``leak``, in both train and test). The
de-duplication must drop both **before** any split manifest is built, so a leaked or duplicated id
can never reach the scorer.
"""

from __future__ import annotations

from pathlib import Path

from object_search.eval.converters.fscd147 import (
    Fscd147Splits,
    dedup_fscd147,
    load_native_splits,
)
from object_search.provenance import file_sha256, repo_root

_FIXTURE_ROOT = repo_root() / "tests" / "fixtures" / "research" / "fscd147"
_IMAGES = _FIXTURE_ROOT / "images"

_PLANTED_DUPLICATE = "fscd147-fixture-dup-b"  # dropped; dup-a is the canonical copy
_CANONICAL_DUPLICATE = "fscd147-fixture-dup-a"
_PLANTED_LEAK = "fscd147-fixture-leak"  # in both train and test


def _image_hashes() -> dict[str, str]:
    return {path.stem: file_sha256(path) for path in _IMAGES.glob("*.png")}


def _dedup() -> tuple[Fscd147Splits, list[str], list[str]]:
    native = load_native_splits(_FIXTURE_ROOT)
    result = dedup_fscd147(native, _image_hashes())
    return result.splits, list(result.dropped_leaks), list(result.dropped_duplicates)


def test_dup_pair_is_byte_identical_in_fixture() -> None:
    # The fixture actually plants a pixel-identical pair (guards against a regenerated fixture that
    # silently makes them differ, which would make the dedup assertion vacuous).
    hashes = _image_hashes()
    assert hashes[_CANONICAL_DUPLICATE] == hashes[_PLANTED_DUPLICATE]


def test_dedup_drops_the_planted_duplicate_keeping_canonical() -> None:
    splits, _leaks, duplicates = _dedup()
    all_ids = set(splits.train) | set(splits.val) | set(splits.test)
    assert _PLANTED_DUPLICATE not in all_ids
    assert _CANONICAL_DUPLICATE in all_ids  # the lexicographically-first copy is kept
    assert duplicates == [_PLANTED_DUPLICATE]


def test_dedup_drops_the_leaked_id_from_every_split() -> None:
    splits, leaks, _duplicates = _dedup()
    all_ids = set(splits.train) | set(splits.val) | set(splits.test)
    assert _PLANTED_LEAK not in all_ids
    assert _PLANTED_LEAK not in splits.test  # explicitly: absent from the held-out test split
    assert leaks == [_PLANTED_LEAK]


def test_no_id_appears_in_more_than_one_split_after_dedup() -> None:
    splits, _leaks, _duplicates = _dedup()
    seen: dict[str, int] = {}
    for ids in (splits.train, splits.val, splits.test):
        for image_id in ids:
            seen[image_id] = seen.get(image_id, 0) + 1
    assert all(count == 1 for count in seen.values()), seen


def test_removed_count_equals_leaks_plus_duplicate_copies() -> None:
    native = load_native_splits(_FIXTURE_ROOT)
    result = dedup_fscd147(native, _image_hashes())
    # 1 leak + (2 duplicate copies - 1 canonical) == 2
    assert result.removed_count == len(result.dropped_leaks) + len(result.dropped_duplicates) == 2


def test_documented_leak_ids_are_dropped_when_supplied() -> None:
    # The structural check catches the 11 real train<->test leaks; an explicitly documented id is
    # also honoured (belt and suspenders). Here a train-only id is force-dropped via the parameter.
    native = load_native_splits(_FIXTURE_ROOT)
    forced = "fscd147-fixture-train-1"
    result = dedup_fscd147(native, _image_hashes(), documented_leak_ids=frozenset({forced}))
    assert forced not in set(result.splits.train)
    assert forced in result.dropped_leaks


def test_dedup_before_manifest_is_the_source_order(tmp_path: Path) -> None:
    # Guard the ordering contract: dedup consumes the RAW native splits (not a manifest), so it can
    # only run before manifest construction. load_native_splits returns the raw triple with the leak
    # still present in two splits, proving dedup sees the contamination it must remove.
    native = load_native_splits(_FIXTURE_ROOT)
    assert _PLANTED_LEAK in native.train and _PLANTED_LEAK in native.test
