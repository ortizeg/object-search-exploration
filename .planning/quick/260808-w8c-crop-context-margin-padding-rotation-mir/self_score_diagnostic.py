"""Local, CPU-only self_score comparison across all five fine-tuned OWLv2 checkpoints.

Extends 260808-dla's diagnostic (.planning/quick/260808-dla-.../self_score_diagnostic.py) with the
`contrastive-crop-v2` checkpoint, using the SAME deterministic exemplar-selection rule (the first
door-class ground-truth box, by annotation list order, in the first file_name-sorted training image
that has one) so the five numbers are directly comparable to each other. Runs entirely locally (CPU
ONNX Runtime, no GPU).

Run:
    pixi run python .planning/quick/260808-w8c-crop-context-margin-padding-rotation-mir/\
self_score_diagnostic.py
"""

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.search import owlv2_oneshot as m

REPO = Path(__file__).resolve().parents[3]
COCO = REPO / "datasets" / "_incoming" / "floorplans" / "train" / "_annotations.coco.json"
IMAGE_DIR = REPO / "datasets" / "_incoming" / "floorplans" / "train"

CHECKPOINTS = {
    "baseline (pretrained)": "owlv2_base_patch16.onnx",
    "headonly (classification)": "owlv2_base_patch16_floorplans_ft.onnx",
    "contrastive": "owlv2_base_patch16_floorplans_ft_contrastive.onnx",
    "contrastive-crop": "owlv2_base_patch16_floorplans_ft_contrastive_crop.onnx",
    "contrastive-crop-v2": "owlv2_base_patch16_floorplans_ft_contrastive_crop_v2.onnx",
}


def _pick_exemplar() -> tuple[Path, ExemplarBox]:
    """First door-class GT box, by annotation order, in the first sorted image with one."""
    coco = json.loads(COCO.read_text())
    door_cat_id = next(c["id"] for c in coco["categories"] if c["name"].lower() == "door")
    images_by_id = {img["id"]: img for img in coco["images"]}
    images_sorted = sorted(images_by_id.values(), key=lambda img: img["file_name"])
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for img in images_sorted:
        for ann in anns_by_image.get(img["id"], []):
            if ann["category_id"] == door_cat_id:
                x, y, w, h = (round(v) for v in ann["bbox"])
                box = BBox(x=x, y=y, w=max(w, 1), h=max(h, 1))
                return IMAGE_DIR / img["file_name"], ExemplarBox(box=box, label="door")
    raise RuntimeError("no door-class ground-truth box found in the training COCO export")


def _self_score(image: np.ndarray, exemplar: ExemplarBox) -> dict[str, float]:
    """Replicates owlv2_oneshot.search()'s steps 1-6 self_score computation exactly."""
    inferencer = m._get_inferencer()
    if inferencer is None:
        raise RuntimeError("OWLv2 weight not found -- check OS_OWLV2_MODEL / models/")
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

    crop_box = exemplar.box.clipped_to(orig_w, orig_h)
    crop = np.ascontiguousarray(
        image[crop_box.y : crop_box.y2, crop_box.x : crop_box.x2], dtype=np.uint8
    )
    query = inferencer.embed_image(crop)
    query_embedding = m.select_query_embedding(query.class_embeds, query.boxes_cxcywh, 0.8)

    target = inferencer.embed_image(image)
    target_norm = m._l2_normalize(target.class_embeds, axis=1)
    scores_all = np.asarray(target_norm @ query_embedding, dtype=np.float32)

    pixel_boxes = m.boxes_to_pixels(target.boxes_cxcywh, orig_w, orig_h)
    max_box_area = 0.1 * float(orig_w * orig_h)
    boxes: list[BBox] = []
    kept_scores: list[float] = []
    for i, pixel_box in enumerate(pixel_boxes):
        if pixel_box is not None and pixel_box.area <= max_box_area:
            boxes.append(pixel_box)
            kept_scores.append(float(scores_all[i]))
    scores = np.asarray(kept_scores, dtype=np.float32)

    self_overlap = [float(scores[i]) for i, box in enumerate(boxes) if box.iou(exemplar.box) >= 0.3]
    self_score = max(self_overlap) if self_overlap else float(scores.max())
    threshold = self_score * 0.94
    n_above = int((scores > threshold).sum())
    return {
        "self_score": self_score,
        "threshold": threshold,
        "n_patches": int(scores.size),
        "n_above_threshold": n_above,
        "frac_retained": n_above / scores.size,
    }


def main() -> None:
    image_path, exemplar = _pick_exemplar()
    logger.info("exemplar image: {}", image_path.name)
    logger.info(
        "exemplar box: x={} y={} w={} h={}",
        exemplar.box.x,
        exemplar.box.y,
        exemplar.box.w,
        exemplar.box.h,
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read {image_path}")

    for name, onnx_name in CHECKPOINTS.items():
        os.environ["OS_OWLV2_MODEL"] = onnx_name
        m.reset_inferencer_cache()
        result = _self_score(image, exemplar)
        logger.info(
            "{:28s} self_score={:+.4f}  threshold={:+.4f}  retained={}/{} ({:.1f}%)",
            name,
            result["self_score"],
            result["threshold"],
            result["n_above_threshold"],
            result["n_patches"],
            result["frac_retained"] * 100,
        )


if __name__ == "__main__":
    main()
