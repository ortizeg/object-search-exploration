"""Tests for the peak-extraction offering.

The load-bearing test is ``test_local_max_separates_touching_instances_that_nms_merges`` --
it is Phase 2 success criterion 2, the concrete demonstration that the default ``local-max``
strategy recovers two nearby instances where the ``nms`` baseline collapses them into one.
"""

import numpy as np
import numpy.typing as npt

from object_search.search.common.peaks import extract_peaks

# A crop size comfortably larger than the inter-bump distance, so the two template-sized boxes
# overlap far past the IoU threshold (forcing the nms baseline to merge the whole above-floor
# blob) while the crop-derived local-max footprint still resolves the two centres. The
# suppression fraction sets that footprint at a quarter of the crop -- small enough to separate
# instances a fifth of a crop apart, which is exactly the regime plain NMS destroys.
TEMPLATE = 48
BUMP_SIGMA = 3.0
FLOOR = 0.4
FRAC = 0.25


def _gaussian(
    shape: tuple[int, int], cx: float, cy: float, sigma: float
) -> npt.NDArray[np.float32]:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
    return g.astype(np.float32)


def _two_touching_bumps(gap: int = 10) -> npt.NDArray[np.float32]:
    """Two unit-height Gaussian bumps ``gap`` px apart on the same row.

    ``np.maximum`` (not sum) keeps each peak at exactly 1.0 so the separation argument is
    about geometry, not about one bump inflating the other.
    """
    h, w = 64, 64
    cx = w // 2
    cy = h // 2
    left = _gaussian((h, w), cx - gap / 2, cy, BUMP_SIGMA)
    right = _gaussian((h, w), cx + gap / 2, cy, BUMP_SIGMA)
    return np.maximum(left, right)


def test_local_max_separates_touching_instances_that_nms_merges():
    # Phase 2 success criterion 2. Same map, two strategies, opposite outcomes.
    response = _two_touching_bumps(gap=10)

    nms_peaks = extract_peaks(
        response,
        strategy="nms",
        template_w=TEMPLATE,
        template_h=TEMPLATE,
        floor=FLOOR,
        suppression_radius_frac=FRAC,
    )
    local_max_peaks = extract_peaks(
        response,
        strategy="local-max",
        template_w=TEMPLATE,
        template_h=TEMPLATE,
        floor=FLOOR,
        suppression_radius_frac=FRAC,
    )

    assert len(nms_peaks) == 1, f"nms should merge the touching bumps, got {len(nms_peaks)}"
    assert len(local_max_peaks) == 2, (
        f"local-max should separate the two bumps, got {len(local_max_peaks)}"
    )


def test_single_bump_returns_exactly_one_peak_under_all_strategies():
    h, w = 64, 64
    response = _gaussian((h, w), w // 2, h // 2, BUMP_SIGMA)
    for strategy in ("nms", "local-max", "watershed"):
        peaks = extract_peaks(
            response,
            strategy=strategy,
            template_w=TEMPLATE,
            template_h=TEMPLATE,
            floor=FLOOR,
            suppression_radius_frac=FRAC,
        )
        assert len(peaks) == 1, f"{strategy} found {len(peaks)} peaks on a single bump"
        # And it lands on the true centre.
        assert abs(peaks[0].x - w // 2) <= 1
        assert abs(peaks[0].y - h // 2) <= 1


def test_all_below_floor_returns_empty_under_all_strategies():
    response = np.full((48, 48), 0.05, dtype=np.float32)
    for strategy in ("nms", "local-max", "watershed"):
        peaks = extract_peaks(
            response,
            strategy=strategy,
            template_w=TEMPLATE,
            template_h=TEMPLATE,
            floor=FLOOR,
            suppression_radius_frac=FRAC,
        )
        assert peaks == []


def test_symmetric_two_bump_map_is_order_deterministic():
    response = _two_touching_bumps(gap=10)
    first = extract_peaks(
        response,
        strategy="local-max",
        template_w=TEMPLATE,
        template_h=TEMPLATE,
        floor=FLOOR,
        suppression_radius_frac=FRAC,
    )
    second = extract_peaks(
        response,
        strategy="local-max",
        template_w=TEMPLATE,
        template_h=TEMPLATE,
        floor=FLOOR,
        suppression_radius_frac=FRAC,
    )
    assert first == second  # same peaks, same order


def test_peaks_are_sorted_by_descending_score():
    # Two bumps of clearly different height -> the taller must come first.
    h, w = 64, 64
    tall = _gaussian((h, w), 20, 32, BUMP_SIGMA)
    short = 0.6 * _gaussian((h, w), 44, 32, BUMP_SIGMA)
    response = np.maximum(tall, short)
    peaks = extract_peaks(
        response,
        strategy="local-max",
        template_w=TEMPLATE,
        template_h=TEMPLATE,
        floor=FLOOR,
        suppression_radius_frac=FRAC,
    )
    assert len(peaks) == 2
    assert peaks[0].score >= peaks[1].score
    assert peaks[0].x == 20  # the tall bump


def test_nan_in_response_is_never_returned_as_a_peak():
    response = _gaussian((48, 48), 24, 24, BUMP_SIGMA)
    response[0, 0] = np.nan  # a NaN that argmax could otherwise surface as the maximum
    peaks = extract_peaks(
        response,
        strategy="local-max",
        template_w=TEMPLATE,
        template_h=TEMPLATE,
        floor=FLOOR,
        suppression_radius_frac=FRAC,
    )
    assert all(not (p.x == 0 and p.y == 0) for p in peaks)
    assert len(peaks) == 1
