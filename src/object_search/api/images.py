"""The demo-image catalogue, image resolution, ground-truth lookup, and upload storage.

Two routes need image bytes -- ``/images`` to list them and ``/search`` to run against one --
so the shared logic lives here rather than being duplicated across two route modules.

Image identity is a **relative path**, never an absolute one: ``"chipset/chipset-01.png"``
under ``assets/demo``, or ``"uploads/<name>"`` under the runtime uploads directory. Every
lookup is confined to its base directory with an explicit containment check, so a crafted
``image_id`` like ``"../../etc/passwd"`` resolves outside the base and is rejected rather than
served.

Ground truth is a sidecar ``<stem>.gt.json`` next to a demo image. Its presence is what
``has_ground_truth`` reports, and its contents seed the run's :class:`SliceMetadata` so a
searched run carries what-kind-of-image context for per-slice analysis (EVAL-10). Uploads and
basketball frames have no sidecar, so their slice metadata is honestly empty (all ``None``) --
never a fabricated zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from object_search.api.schemas import ImageInfo
from object_search.provenance import repo_root
from object_search.schemas.records import SliceMetadata

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_DEMO_SUBDIRS = ("synthetic", "chipset", "basketball", "markers", "textured")
_UPLOADS_PREFIX = "uploads/"


class ImageNotFoundError(FileNotFoundError):
    """The requested ``image_id`` does not resolve to a file inside its base directory."""


class InvalidImageError(ValueError):
    """An uploaded payload is not a decodable image."""


def demo_root() -> Path:
    """The ``assets/demo`` directory shipped with the repo."""
    return repo_root() / "assets" / "demo"


def _gt_sidecar(image_path: Path) -> Path:
    """The ``<stem>.gt.json`` path next to a demo image (whether or not it exists)."""
    return image_path.parent / f"{image_path.stem}.gt.json"


def _read_dimensions(image_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for an image on disk.

    Raises:
        ImageNotFoundError: If the file cannot be read as an image.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageNotFoundError(f"could not read image at {image_path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def list_demo_images() -> list[ImageInfo]:
    """Every demo image under ``assets/demo``, with dimensions and ground-truth presence.

    Sorted by id so the catalogue is deterministic. A subdirectory that is absent (a partial
    checkout) is skipped rather than raising -- the catalogue reflects what is on disk.
    """
    root = demo_root()
    infos: list[ImageInfo] = []
    for subdir in _DEMO_SUBDIRS:
        directory = root / subdir
        if not directory.is_dir():
            continue
        for image_path in sorted(directory.iterdir()):
            if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            width, height = _read_dimensions(image_path)
            image_id = f"{subdir}/{image_path.name}"
            infos.append(
                ImageInfo(
                    id=image_id,
                    width=width,
                    height=height,
                    has_ground_truth=_gt_sidecar(image_path).is_file(),
                )
            )
    infos.sort(key=lambda info: info.id)
    return infos


def resolve_image_path(image_id: str, uploads_dir: Path) -> Path:
    """Map an ``image_id`` to a file path, confined to its base directory.

    ``uploads/<name>`` resolves under ``uploads_dir``; anything else resolves under
    ``assets/demo``. The resolved path must stay inside its base -- a traversal attempt is an
    :class:`ImageNotFoundError`, not a served file.

    Raises:
        ImageNotFoundError: If the id escapes its base directory or names no existing file.
    """
    if image_id.startswith(_UPLOADS_PREFIX):
        base = uploads_dir
        relative = image_id[len(_UPLOADS_PREFIX) :]
    else:
        base = demo_root()
        relative = image_id

    base_resolved = base.resolve()
    candidate = (base_resolved / relative).resolve()
    if not candidate.is_relative_to(base_resolved):
        raise ImageNotFoundError(f"image_id {image_id!r} escapes its base directory")
    if not candidate.is_file():
        raise ImageNotFoundError(f"no image for image_id {image_id!r}")
    return candidate


def load_image_bgr(image_id: str, uploads_dir: Path) -> npt.NDArray[np.uint8]:
    """Load a resolved image as a BGR ``uint8`` array (the scene every method takes).

    Raises:
        ImageNotFoundError: If the id does not resolve or the file cannot be decoded.
    """
    path = resolve_image_path(image_id, uploads_dir)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageNotFoundError(f"could not decode image at {path}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def slice_metadata_for(image_id: str) -> SliceMetadata:
    """Build the run's :class:`SliceMetadata` from a demo image's ground-truth sidecar.

    Synthetic images carry a full ``slice_metadata`` block (exact scale/rotation/clutter);
    chipset images carry a box list from which the true instance count is taken. An image
    with no sidecar -- an upload or a basketball frame -- yields an empty
    :class:`SliceMetadata` (every field ``None`` = "unknown", never a fabricated ``0``).
    """
    if image_id.startswith(_UPLOADS_PREFIX):
        return SliceMetadata()
    sidecar = _gt_sidecar(demo_root() / image_id)
    if not sidecar.is_file():
        return SliceMetadata()

    data = json.loads(sidecar.read_text())
    if isinstance(data.get("slice_metadata"), dict):
        return SliceMetadata.model_validate(data["slice_metadata"])

    boxes = data.get("boxes")
    true_count = data.get("achieved_n")
    if true_count is None and isinstance(boxes, list):
        true_count = len(boxes)
    return SliceMetadata(true_instance_count=true_count)


def save_upload(uploads_dir: Path, filename: str, payload: bytes) -> ImageInfo:
    """Persist an uploaded image under ``uploads_dir`` and return its catalogue entry.

    The filename is reduced to its bare name so an upload can never write outside
    ``uploads_dir``. The payload is decoded before it is written, so a non-image is rejected
    up front rather than stored as a file no search can read.

    Raises:
        InvalidImageError: If the payload does not decode as an image.
    """
    name = Path(filename).name or "upload"
    if Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
        name = f"{name}.png"

    array = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR) if array.size else None
    if image is None:
        raise InvalidImageError("uploaded payload is not a decodable image")

    uploads_dir.mkdir(parents=True, exist_ok=True)
    dest = uploads_dir / name
    dest.write_bytes(payload)
    height, width = image.shape[:2]
    logger.info("stored upload {} ({}x{})", dest, width, height)
    return ImageInfo(
        id=f"{_UPLOADS_PREFIX}{name}",
        width=int(width),
        height=int(height),
        has_ground_truth=False,
    )
