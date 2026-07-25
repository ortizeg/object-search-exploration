"""Paired comparison: one exemplar box through all four methods in a single call (EVAL-05).

The whole point of a paired comparison is that the confound is removed. Running each method on a
*different* box would mean a ranking reflects which boxes happened to be easy, not which method is
better; running the **same** box through every method makes the results directly comparable. This
is the objective sibling of the human paired-rating UI: where ground truth exists, each method's
result is scored against it and the pairwise winner is recorded automatically, feeding the same
``paired_comparisons`` table (created back in Phase 3) that Bradley-Terry reads.

The comparison scalar is **F1** against ground truth. F1 is ``None`` exactly when a method
returned nothing on an image that has instances -- an abstention, which for "which did better"
ranks below any method that found something, so it maps to ``0.0`` for the comparison. A method
that *raised* is worse still (it is not measurable at all, EVAL-12) and maps below abstention, so a
crash loses to an honest empty result. Equal scores within a tolerance are a **tie**, stored as the
distinct ``'tie'`` outcome so the modelling choice (half a win each) can be revisited later.

When an image has no ground truth, the methods are still run and returned, but **no** comparison is
recorded -- an objective winner cannot be manufactured without truth (EVAL-02). The human paired-
rating path is what fills that gap, and it writes to the same table.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt
from loguru import logger
from pydantic import BaseModel, ConfigDict

from object_search.eval.labels import load_ground_truth
from object_search.eval.metrics import match_predictions, precision_recall_f1
from object_search.schemas.geometry import BBox, ExemplarBox
from object_search.search import get_method, list_methods
from object_search.store.schema import DEFAULT_EXPLORATION

# A crash is not measurable (EVAL-12), so it scores below an honest abstention (0.0). A working
# method scores its F1 in [0, 1]. This ordering makes "working > abstained > crashed".
_ERROR_SCORE = -1.0
# Scores within this tolerance are called a tie -- two CV overlays a human would call "same".
_TIE_TOL = 1e-9


class MethodScore(BaseModel):
    """One method's outcome and comparison score on the shared exemplar box."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str
    outcome: str
    precision: float | None
    recall: float | None
    f1: float | None
    comparison_score: float
    n_matches: int


class PairwiseOutcome(BaseModel):
    """The recorded winner of one ordered method pair (``method_a`` < ``method_b``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_a: str
    method_b: str
    winner: str  # 'a' | 'b' | 'tie'


class PairedResult(BaseModel):
    """Everything one paired comparison produced: per-method scores and pairwise winners."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    gt_available: bool
    scores: tuple[MethodScore, ...]
    comparisons: tuple[PairwiseOutcome, ...]


def _score_method(
    method: str,
    image: npt.NDArray[np.uint8],
    exemplar: ExemplarBox,
    config: BaseModel,
    gt_boxes: Sequence[BBox] | None,
    iou_threshold: float,
) -> MethodScore:
    """Run one method on the shared box and score it against ground truth (if any)."""
    spec = get_method(method)
    try:
        result = spec.fn(image, exemplar, config)
    except Exception as exc:
        # A crash is a distinct, worst outcome (EVAL-12); it must not abort the other methods.
        logger.warning("paired: {} raised on {}: {}", method, exemplar.box.xyxy, exc)
        return MethodScore(
            method=method,
            outcome="error",
            precision=None,
            recall=None,
            f1=None,
            comparison_score=_ERROR_SCORE,
            n_matches=0,
        )

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    if gt_boxes is not None:
        pred_boxes = [match.box for match in result.matches]
        tp, fp, fn = match_predictions(pred_boxes, gt_boxes, iou_threshold)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    # F1 None means "returned nothing on an image with instances" -> 0.0 for the comparison.
    comparison_score = f1 if f1 is not None else 0.0
    return MethodScore(
        method=method,
        outcome=result.outcome.value,
        precision=precision,
        recall=recall,
        f1=f1,
        comparison_score=comparison_score,
        n_matches=len(result.matches),
    )


def _winner(score_a: float, score_b: float) -> str:
    """Return ``'a'``, ``'b'`` or ``'tie'`` for two comparison scores within the tie tolerance."""
    if abs(score_a - score_b) <= _TIE_TOL:
        return "tie"
    return "a" if score_a > score_b else "b"


def _record_comparison(
    conn: sqlite3.Connection,
    image_id: str,
    exemplar: ExemplarBox,
    outcome: PairwiseOutcome,
) -> None:
    """Insert one pairwise outcome into ``paired_comparisons`` (winner kept, ties distinct)."""
    box = exemplar.box
    conn.execute(
        "INSERT INTO paired_comparisons "
        "(exploration, image_id, exemplar_x, exemplar_y, exemplar_w, exemplar_h, "
        " method_a, method_b, winner, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            DEFAULT_EXPLORATION,
            image_id,
            box.x,
            box.y,
            box.w,
            box.h,
            outcome.method_a,
            outcome.method_b,
            outcome.winner,
            datetime.now(UTC).isoformat(),
        ),
    )


def run_paired(
    conn: sqlite3.Connection,
    image: npt.NDArray[np.uint8],
    image_id: str,
    exemplar: ExemplarBox,
    configs: Mapping[str, BaseModel] | None = None,
    methods: Sequence[str] | None = None,
    iou_threshold: float = 0.5,
) -> PairedResult:
    """Run the same exemplar box through several methods and record pairwise winners (EVAL-05).

    Args:
        conn: Open, migrated store connection -- the pairwise outcomes are written to
            ``paired_comparisons`` inside one transaction.
        image: The BGR scene, shared by every method so the comparison is not confounded.
        image_id: Scene identifier, used to look up ground truth and stored with each comparison.
        exemplar: The single box every method searches from.
        configs: Optional per-method config instances; a method absent here uses its default
            config (``config_model()``).
        methods: Registry keys to compare; defaults to every registered method. Restrict this to
            the model-free methods to compare without ONNX weights.
        iou_threshold: IoU at which a predicted box counts as a true positive.

    Returns:
        A :class:`PairedResult` with each method's score and every pairwise winner. When the image
        has no ground truth, ``gt_available`` is ``False`` and no comparisons are recorded.
    """
    names = sorted(methods) if methods is not None else sorted(s.name for s in list_methods())
    configs = configs or {}
    gt = load_ground_truth(image_id)
    gt_boxes = gt.boxes if gt is not None else None

    scores = tuple(
        _score_method(
            name,
            image,
            exemplar,
            configs.get(name, get_method(name).config_model()),
            gt_boxes,
            iou_threshold,
        )
        for name in names
    )

    comparisons: list[PairwiseOutcome] = []
    if gt_boxes is not None:
        by_name = {s.method: s for s in scores}
        # Every unordered pair, ordered alphabetically so (a, b) is stable and the winner label is
        # unambiguous relative to that ordering.
        for idx, name_a in enumerate(names):
            for name_b in names[idx + 1 :]:
                outcome = PairwiseOutcome(
                    method_a=name_a,
                    method_b=name_b,
                    winner=_winner(
                        by_name[name_a].comparison_score, by_name[name_b].comparison_score
                    ),
                )
                comparisons.append(outcome)
                _record_comparison(conn, image_id, exemplar, outcome)
        conn.commit()
    else:
        logger.info("paired: no ground truth for {}; ran methods but recorded no winners", image_id)

    return PairedResult(
        image_id=image_id,
        gt_available=gt_boxes is not None,
        scores=scores,
        comparisons=tuple(comparisons),
    )
