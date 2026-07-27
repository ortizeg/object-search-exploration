"""Tests for the image catalogue / resolution / upload helpers and the ``POST /images`` route.

All model-free: pure filesystem plus cv2 encode/decode. Covers the paths the API depends on but
that no search-level test exercises -- the unreadable-image guard, the missing-subdir skip, the
path-traversal rejection, the decode-failure guards, the ground-truth sidecar branches, and the
upload store (valid, extensionless, and non-image-rejected).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from object_search.api import images as images_mod
from object_search.api.images import (
    ImageNotFoundError,
    InvalidImageError,
    list_demo_images,
    load_image_bgr,
    resolve_image_path,
    save_upload,
    slice_metadata_for,
)
from object_search.schemas.records import SliceMetadata


def _png_bytes(width: int = 8, height: int = 6, color: tuple[int, int, int] = (0, 0, 255)) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((height, width, 3), color, dtype=np.uint8))
    assert ok
    return bytes(buf.tobytes())


def _write_png(path: Path, width: int = 8, height: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height))


@pytest.fixture
def demo_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``repo_root`` at a temp dir so ``demo_root()`` is a fixture-built ``assets/demo``."""
    monkeypatch.setattr(images_mod, "repo_root", lambda: tmp_path)
    return tmp_path


# ------------------------------------------------------------------- list_demo_images


def test_list_demo_images_reports_dimensions_and_ground_truth(demo_repo: Path) -> None:
    """Images are listed sorted by id, with dimensions and per-image ground-truth presence."""
    demo = demo_repo / "assets" / "demo"
    _write_png(demo / "synthetic" / "a.png", width=10, height=7)
    (demo / "synthetic" / "a.gt.json").write_text("{}")  # a sidecar -> has_ground_truth True
    _write_png(demo / "chipset" / "b.png", width=4, height=5)  # no sidecar -> False
    (demo / "chipset" / "notes.txt").write_text("ignored")  # non-image suffix, skipped
    # `basketball` / `markers` / `textured` subdirs are absent -> the missing-subdir skip runs.

    infos = list_demo_images()

    assert [info.id for info in infos] == ["chipset/b.png", "synthetic/a.png"]  # sorted by id
    by_id = {info.id: info for info in infos}
    assert (by_id["synthetic/a.png"].width, by_id["synthetic/a.png"].height) == (10, 7)
    assert by_id["synthetic/a.png"].has_ground_truth is True
    assert by_id["chipset/b.png"].has_ground_truth is False


def test_list_demo_images_raises_on_an_unreadable_image(demo_repo: Path) -> None:
    """A file with an image suffix that cv2 cannot decode is a hard error, not a silent skip."""
    demo = demo_repo / "assets" / "demo"
    corrupt = demo / "synthetic" / "broken.png"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not a real png")

    with pytest.raises(ImageNotFoundError, match="could not read image"):
        list_demo_images()


# ------------------------------------------------------------------- resolve_image_path


def test_resolve_uploads_id_maps_under_uploads_dir(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    _write_png(uploads / "shot.png")
    resolved = resolve_image_path("uploads/shot.png", uploads)
    assert resolved == (uploads / "shot.png").resolve()


def test_resolve_rejects_path_traversal(demo_repo: Path, tmp_path: Path) -> None:
    """A crafted id that escapes the base directory is rejected, never served."""
    with pytest.raises(ImageNotFoundError, match="escapes its base"):
        resolve_image_path("../../etc/passwd", tmp_path / "uploads")


def test_resolve_missing_file_raises(demo_repo: Path, tmp_path: Path) -> None:
    (demo_repo / "assets" / "demo").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ImageNotFoundError, match="no image for image_id"):
        resolve_image_path("synthetic/absent.png", tmp_path / "uploads")


# ------------------------------------------------------------------- load_image_bgr


def test_load_image_bgr_returns_bgr_array(demo_repo: Path, tmp_path: Path) -> None:
    _write_png(demo_repo / "assets" / "demo" / "synthetic" / "img.png", width=12, height=9)
    array = load_image_bgr("synthetic/img.png", tmp_path / "uploads")
    assert array.shape == (9, 12, 3)
    assert array.dtype == np.uint8


def test_load_image_bgr_raises_when_a_resolved_file_will_not_decode(
    demo_repo: Path, tmp_path: Path
) -> None:
    """The file resolves (it exists) but its bytes are not a decodable image."""
    corrupt = demo_repo / "assets" / "demo" / "synthetic" / "corrupt.png"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"garbage bytes")
    with pytest.raises(ImageNotFoundError, match="could not decode"):
        load_image_bgr("synthetic/corrupt.png", tmp_path / "uploads")


# ------------------------------------------------------------------- slice_metadata_for


def test_slice_metadata_for_an_upload_is_empty() -> None:
    """An upload has no sidecar, so every field is None -- never a fabricated zero."""
    meta = slice_metadata_for("uploads/whatever.png")
    assert meta == SliceMetadata()
    assert meta.true_instance_count is None


def test_slice_metadata_without_a_sidecar_is_empty(demo_repo: Path) -> None:
    (demo_repo / "assets" / "demo" / "chipset").mkdir(parents=True, exist_ok=True)
    meta = slice_metadata_for("chipset/no-sidecar.png")
    assert meta == SliceMetadata()


def test_slice_metadata_reads_a_full_slice_block(demo_repo: Path) -> None:
    demo = demo_repo / "assets" / "demo"
    (demo / "synthetic").mkdir(parents=True, exist_ok=True)
    (demo / "synthetic" / "s.gt.json").write_text('{"slice_metadata": {"true_instance_count": 3}}')
    meta = slice_metadata_for("synthetic/s.png")
    assert meta.true_instance_count == 3


def test_slice_metadata_counts_boxes_when_no_explicit_count(demo_repo: Path) -> None:
    """A chipset sidecar with a box list and no achieved_n derives the count from len(boxes)."""
    demo = demo_repo / "assets" / "demo"
    (demo / "chipset").mkdir(parents=True, exist_ok=True)
    (demo / "chipset" / "c.gt.json").write_text('{"boxes": [{"x": 0}, {"x": 1}, {"x": 2}]}')
    meta = slice_metadata_for("chipset/c.png")
    assert meta.true_instance_count == 3


# ------------------------------------------------------------------- save_upload


def test_save_upload_stores_a_decodable_image(tmp_path: Path) -> None:
    uploads = tmp_path / "uploads"
    info = save_upload(uploads, "photo.png", _png_bytes(width=20, height=15))
    assert info.id == "uploads/photo.png"
    assert (info.width, info.height) == (20, 15)
    assert info.has_ground_truth is False
    assert (uploads / "photo.png").is_file()


def test_save_upload_gives_an_extensionless_name_a_png_suffix(tmp_path: Path) -> None:
    info = save_upload(tmp_path / "uploads", "noext", _png_bytes())
    assert info.id == "uploads/noext.png"


def test_save_upload_rejects_a_non_image_payload(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError, match="not a decodable image"):
        save_upload(tmp_path / "uploads", "bad.png", b"definitely not an image")


# ------------------------------------------------------------------- POST /images route


def test_post_images_accepts_a_valid_upload(api_client: TestClient) -> None:
    response = api_client.post(
        "/images", files={"file": ("upload.png", _png_bytes(width=16, height=11), "image/png")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("uploads/")
    assert body["width"] == 16 and body["height"] == 11
    assert body["has_ground_truth"] is False


def test_post_images_rejects_a_non_image_with_422(api_client: TestClient) -> None:
    response = api_client.post("/images", files={"file": ("bad.png", b"not an image", "image/png")})
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid_image"
