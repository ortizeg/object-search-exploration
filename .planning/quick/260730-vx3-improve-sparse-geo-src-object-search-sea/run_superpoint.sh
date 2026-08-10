#!/usr/bin/env bash
# Throwaway driver for quick task 260730-vx3: hypothesis 2 (SuperPoint + pairwise-4dof).
#
# Measured against the ORIGINAL baseline, over the SAME min_inliers x nms_iou grid shape the
# SIFT pairwise-4dof control used, so the SIFT/SuperPoint comparison is apples-to-apples.
#
# Logs land in this directory (NOT /tmp) so progress survives a session interruption.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

DIR=".planning/quick/260730-vx3-improve-sparse-geo-src-object-search-sea"
PIXI="$HOME/.pixi/bin/pixi"

SP_GRID='[
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 5,  "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 5,  "nms_iou": 0.5},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 8,  "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 8,  "nms_iou": 0.5},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 12, "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 12, "nms_iou": 0.5},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 16, "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 16, "nms_iou": 0.5},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 20, "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "pairwise-4dof", "min_inliers": 20, "nms_iou": 0.5}
]'

# Supplementary arm: SuperPoint at its OWN default voting mode. Not the hypothesis under test,
# but it separates "SuperPoint keypoints are the problem" from "pairwise-4dof is the problem".
SP_T2_GRID='[
 {"backend": "superpoint", "voting_mode": "translation-2dof", "min_inliers": 5,  "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "translation-2dof", "min_inliers": 5,  "nms_iou": 0.5},
 {"backend": "superpoint", "voting_mode": "translation-2dof", "min_inliers": 8,  "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "translation-2dof", "min_inliers": 8,  "nms_iou": 0.5},
 {"backend": "superpoint", "voting_mode": "translation-2dof", "min_inliers": 12, "nms_iou": 0.3},
 {"backend": "superpoint", "voting_mode": "translation-2dof", "min_inliers": 12, "nms_iou": 0.5}
]'

run() {  # run <label> <dataset> <grid>
  local label="$1" dataset="$2" grid="$3"
  echo "=== START $label / $dataset  $(date -u +%H:%M:%S) ==="
  "$PIXI" run python "$DIR/measure.py" \
    --dataset "$dataset" --label "${label}" --grid "$grid" \
    --out "$DIR/measurements/${label}.json" \
    > "$DIR/measurements/${label}.log" 2>&1
  echo "=== DONE  $label rc=$?  $(date -u +%H:%M:%S) ==="
}

run sp-pw-door    floorplans-door   "$SP_GRID"
run sp-pw-window  floorplans-window "$SP_GRID"
run sp-t2-door    floorplans-door   "$SP_T2_GRID"
run sp-t2-window  floorplans-window "$SP_T2_GRID"
echo "ALL SUPERPOINT RUNS FINISHED $(date -u +%H:%M:%S)"
