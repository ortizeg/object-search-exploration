"""Tests for the EVAL-09 provenance helpers.

The config-hash tests are the substantive ones. Research measured three independent ways a
naive hash drifts for what a practitioner believes is the same config -- key order, float
repr, and int/float coercion -- and each of those has a test here, because a drifting hash
splits one method into two rows on the scoreboard and halves every ``n``.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from object_search import provenance


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    threshold: float
    max_keypoints: int
    mode: str


class _ReorderedConfig(BaseModel):
    """Same fields, declared in a different order -- must hash identically."""

    model_config = ConfigDict(frozen=True)

    mode: str
    max_keypoints: int
    threshold: float


def test_config_hash_is_stable_across_field_declaration_order():
    a = _Config(threshold=0.7, max_keypoints=4096, mode="stretch")
    b = _ReorderedConfig(mode="stretch", max_keypoints=4096, threshold=0.7)
    assert provenance.config_hash(a) == provenance.config_hash(b)


def test_canonical_config_json_sorts_keys_and_strips_whitespace():
    payload = provenance.canonical_config_json(
        _Config(threshold=0.7, max_keypoints=4096, mode="stretch")
    )
    assert payload == '{"max_keypoints":4096,"mode":"stretch","threshold":0.7}'


def test_config_hash_hashes_the_validated_model_not_the_raw_input():
    """Pydantic coerces int 1 to float 1.0; the hash must follow the model, not the JSON."""
    typed_int = _Config(threshold=1, max_keypoints=10, mode="stretch")  # type: ignore[arg-type]
    typed_float = _Config(threshold=1.0, max_keypoints=10, mode="stretch")
    assert provenance.config_hash(typed_int) == provenance.config_hash(typed_float)


def test_config_hash_changes_when_a_value_changes():
    a = _Config(threshold=0.7, max_keypoints=4096, mode="stretch")
    b = _Config(threshold=0.71, max_keypoints=4096, mode="stretch")
    assert provenance.config_hash(a) != provenance.config_hash(b)


def test_config_hash_is_a_sha256_hex_digest():
    digest = provenance.config_hash(_Config(threshold=0.7, max_keypoints=1, mode="stretch"))
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_canonical_config_json_refuses_nan_rather_than_emitting_invalid_json():
    class _NaNConfig(BaseModel):
        value: float

    with pytest.raises(ValueError, match="Out of range float"):
        provenance.canonical_config_json(_NaNConfig(value=float("nan")))


# ------------------------------------------------------------------------------ file hash


def test_file_sha256_matches_a_known_digest(tmp_path: Path):
    target = tmp_path / "weights.onnx"
    target.write_bytes(b"abc")
    # sha256("abc"), the standard published test vector.
    assert provenance.file_sha256(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_file_sha256_handles_a_file_larger_than_one_chunk(tmp_path: Path):
    target = tmp_path / "big.bin"
    target.write_bytes(b"x" * (3 * 1024 * 1024 + 7))
    assert len(provenance.file_sha256(target)) == 64


def test_file_sha256_raises_on_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        provenance.file_sha256(tmp_path / "nope.onnx")


# ------------------------------------------------------------------------------ git / env


def test_current_git_sha_returns_a_sha_in_this_checkout():
    sha = provenance.current_git_sha()
    assert sha == "unknown" or (len(sha) == 40 and set(sha) <= set("0123456789abcdef"))


def test_current_git_sha_degrades_to_unknown_when_git_is_missing(monkeypatch):
    """Provenance is metadata: a missing git must not abort the run it describes."""
    monkeypatch.setattr(provenance.shutil, "which", lambda _name: None)
    assert provenance.current_git_sha() == "unknown"


def test_current_git_sha_degrades_to_unknown_when_git_fails(monkeypatch):
    class _Failed:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(provenance.subprocess, "run", lambda *a, **k: _Failed())
    assert provenance.current_git_sha() == "unknown"


def test_current_git_sha_degrades_to_unknown_when_git_cannot_be_executed(monkeypatch):
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("no exec")

    monkeypatch.setattr(provenance.subprocess, "run", _boom)
    assert provenance.current_git_sha() == "unknown"


def test_pixi_lock_sha256_hashes_the_committed_lockfile():
    assert len(provenance.pixi_lock_sha256()) == 64


def test_pixi_lock_sha256_degrades_to_unknown_when_absent(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(provenance, "_REPO_ROOT", tmp_path)
    assert provenance.pixi_lock_sha256() == "unknown"


def test_repo_root_is_this_checkout():
    assert (provenance.repo_root() / "pixi.toml").is_file()


def test_environment_identity_reports_the_libraries_that_change_results():
    identity = provenance.environment_identity()
    assert set(identity) == {
        "python_version",
        "numpy_version",
        "cv2_version",
        "onnxruntime_version",
        "ort_providers",
        "pixi_lock_sha256",
    }
    assert identity["python_version"].startswith("3.12")
    assert "CPUExecutionProvider" in identity["ort_providers"]
