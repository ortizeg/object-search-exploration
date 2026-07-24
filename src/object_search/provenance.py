"""Helpers that make a run identifiable after the fact (EVAL-09).

The point of provenance is a narrow one: when two ratings disagree, or a method's score
moves without a code change, the record must be enough to tell *why*. That needs four
things, and this module produces all four.

1. **The code** -- :func:`current_git_sha`.
2. **The config** -- :func:`config_hash`, over the *validated* model, with sorted keys.
   Research verified three ways this goes wrong if done naively: ``json.dumps`` of a dict
   yields different digests for different key orders; ``0.1 + 0.2`` serialises as
   ``0.30000000000000004`` while a literal ``0.3`` serialises as ``0.3``; and Pydantic
   coerces ``1`` to ``1.0`` for a float field, so hashing the raw request body gives a
   different hash from hashing the model the method actually ran with. Hence: dump the
   model, ``mode="json"``, ``sort_keys=True``, and hash that exact string. The string
   itself is available from :func:`canonical_config_json` so a hash mismatch is debuggable
   rather than mysterious.
3. **The weights** -- :func:`file_sha256`, so a silently re-exported model is detectable.
4. **The environment** -- :func:`environment_identity`. This is the one people leave out
   and it is measurably load-bearing: OpenCV 4.10.0 and 5.0.0 produce *different*
   ``estimateAffinePartial2D`` results for identical input and opposite constants for the
   flat-template NCC case, and on macOS the CoreML execution provider is available by
   default so an unpinned ``providers`` argument changes the numbers. No git SHA captures
   any of that. ``pixi_lock_sha256`` is the cheapest high-coverage field: one value that
   changes whenever any dependency does.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

_SHA256_CHUNK_BYTES = 1024 * 1024

# Walking up from src/object_search/provenance.py: object_search -> src -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Absolute path of the repository checkout this package was imported from."""
    return _REPO_ROOT


def current_git_sha() -> str:
    """Full SHA of ``HEAD``, or ``"unknown"`` when it cannot be determined.

    Returns ``"unknown"`` rather than raising. Provenance is metadata: a missing git SHA
    should degrade the record, not abort the run that produced it -- the package must stay
    usable from an installed wheel or a source tarball with no ``.git`` directory.
    """
    git = shutil.which("git")
    if git is None:
        logger.warning("git executable not found; recording git_sha='unknown'")
        return "unknown"
    try:
        # S603 is suppressed below deliberately: fixed argument list, absolute executable
        # resolved by shutil.which, no shell, no caller-supplied input anywhere in the
        # call. There is nothing here to inject into.
        completed = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning(f"could not run git rev-parse: {exc}; recording git_sha='unknown'")
        return "unknown"
    if completed.returncode != 0:
        logger.warning(
            f"git rev-parse HEAD failed ({completed.returncode}): "
            f"{completed.stderr.strip()}; recording git_sha='unknown'"
        )
        return "unknown"
    return completed.stdout.strip() or "unknown"


def canonical_config_json(config: BaseModel) -> str:
    """Serialise a *validated* config model to the exact string that gets hashed.

    ``sort_keys=True`` makes the digest independent of field declaration order;
    ``separators`` removes insignificant whitespace; ``allow_nan=False`` makes ``NaN`` and
    ``Infinity`` -- which are not valid JSON -- fail loudly instead of producing a payload
    no other JSON parser can read back.

    Args:
        config: An instance of a method's ``config_model``, already validated.

    Returns:
        Canonical JSON. Store it alongside the hash (API-03 requires both) so a hash
        mismatch can be diffed instead of guessed at.
    """
    return json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(config: BaseModel) -> str:
    """SHA-256 of :func:`canonical_config_json`, hex-encoded.

    Hash the validated model, never the raw request body: Pydantic coerces ``1`` to ``1.0``
    for a float field, so the two differ as JSON while being the same config.
    """
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    """SHA-256 of a file's bytes, hex-encoded, read in 1 MiB chunks.

    Chunked because model weights are tens to hundreds of MiB and reading one into memory
    to hash it is pure waste.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_SHA256_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def pixi_lock_sha256() -> str:
    """SHA-256 of ``pixi.lock``, or ``"unknown"`` when it is not on disk.

    One field that changes whenever *any* dependency does. Cheaper and more complete than
    enumerating package versions, and the right thing to group statistics by when deciding
    whether two runs are comparable at all.
    """
    lock = _REPO_ROOT / "pixi.lock"
    if not lock.is_file():
        logger.warning(f"pixi.lock not found at {lock}; recording pixi_lock_sha256='unknown'")
        return "unknown"
    return file_sha256(lock)


def environment_identity() -> dict[str, str]:
    """Library versions that measurably change numerical results.

    Imports are deliberately local: reading a version string is not a good enough reason to
    pay ``onnxruntime``'s import cost in every module that happens to touch provenance.

    Returns:
        Mapping with keys ``python_version``, ``numpy_version``, ``cv2_version``,
        ``onnxruntime_version``, ``ort_providers`` (comma-joined) and ``pixi_lock_sha256``.
    """
    import platform

    import cv2
    import numpy as np
    import onnxruntime as ort

    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cv2_version": cv2.__version__,
        "onnxruntime_version": ort.__version__,
        "ort_providers": ",".join(ort.get_available_providers()),
        "pixi_lock_sha256": pixi_lock_sha256(),
    }
