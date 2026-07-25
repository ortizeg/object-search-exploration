"""``GET /images`` and ``POST /images`` -- the demo catalogue and ad-hoc uploads.

``GET /images`` reports ``has_ground_truth`` per image so the eval layer and the UI can tell
objectively-scorable images (synthetic + chipset, which ship ``.gt.json`` sidecars) apart
from ones that can only be human-rated (basketball frames, uploads). ``POST /images`` accepts
a multipart upload, decodes it to confirm it is an image, stores it under the runtime uploads
directory, and returns its catalogue entry with ``has_ground_truth=false``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile

from object_search.api.errors import APIError
from object_search.api.images import InvalidImageError, list_demo_images, save_upload
from object_search.api.schemas import ImageInfo

router = APIRouter(tags=["images"])


@router.get("/images", response_model=list[ImageInfo])
def get_images() -> list[ImageInfo]:
    """List demo images with dimensions and whether ground truth exists for each."""
    return list_demo_images()


@router.post("/images", response_model=ImageInfo)
async def post_image(request: Request, file: UploadFile) -> ImageInfo:
    """Accept an uploaded image; store it and return its catalogue entry (ground-truth-less).

    Raises:
        APIError: 422 ``invalid_image`` if the payload does not decode as an image.
    """
    payload = await file.read()
    try:
        return save_upload(request.app.state.uploads_dir, file.filename or "upload", payload)
    except InvalidImageError as exc:
        raise APIError(422, "invalid_image", str(exc)) from exc
