#!/usr/bin/env bash
# Throwaway driver for quick task 260730-vx3: the pairwise-4dof measurement block.
#
# Four conditions are needed to separate "pairwise voting itself helps" from "mirror under
# pairwise helps":
#   pw-mirror-door   (already measured)
#   pw-mirror-window
#   pw-only-door     <- the control
#   pw-only-window   <- the control
#
# Logs land in this directory (NOT /tmp) so progress survives a session interruption.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1

DIR=".planning/quick/260730-vx3-improve-sparse-geo-src-object-search-sea"
PIXI="$HOME/.pixi/bin/pixi"

MIRROR_GRID='[
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 5,  "nms_iou": 0.3},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 5,  "nms_iou": 0.5},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 8,  "nms_iou": 0.3},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 8,  "nms_iou": 0.5},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 12, "nms_iou": 0.3},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 12, "nms_iou": 0.5},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 16, "nms_iou": 0.3},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 16, "nms_iou": 0.5},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 20, "nms_iou": 0.3},
 {"allow_mirror": true, "voting_mode": "pairwise-4dof", "min_inliers": 20, "nms_iou": 0.5}
]'

ONLY_GRID='[
 {"voting_mode": "pairwise-4dof", "min_inliers": 5,  "nms_iou": 0.3},
 {"voting_mode": "pairwise-4dof", "min_inliers": 5,  "nms_iou": 0.5},
 {"voting_mode": "pairwise-4dof", "min_inliers": 8,  "nms_iou": 0.3},
 {"voting_mode": "pairwise-4dof", "min_inliers": 8,  "nms_iou": 0.5},
 {"voting_mode": "pairwise-4dof", "min_inliers": 12, "nms_iou": 0.3},
 {"voting_mode": "pairwise-4dof", "min_inliers": 12, "nms_iou": 0.5},
 {"voting_mode": "pairwise-4dof", "min_inliers": 16, "nms_iou": 0.3},
 {"voting_mode": "pairwise-4dof", "min_inliers": 16, "nms_iou": 0.5},
 {"voting_mode": "pairwise-4dof", "min_inliers": 20, "nms_iou": 0.3},
 {"voting_mode": "pairwise-4dof", "min_inliers": 20, "nms_iou": 0.5}
]'

run() {  # run <label> <dataset> <grid>
  local label="$1" dataset="$2" grid="$3"
  echo "=== START $label / $dataset  $(date -u +%H:%M:%S) ==="
  "$PIXI" run python "$DIR/measure.py" \
    --dataset "$dataset" --label "${label%%-*}" --grid "$grid" \
    --out "$DIR/measurements/${label}.json" \
    > "$DIR/measurements/${label}.log" 2>&1
  echo "=== DONE  $label rc=$?  $(date -u +%H:%M:%S) ==="
}

run pw-only-door     floorplans-door   "$ONLY_GRID"
run pw-mirror-window floorplans-window "$MIRROR_GRID"
run pw-only-window   floorplans-window "$ONLY_GRID"
echo "ALL PAIRWISE RUNS FINISHED $(date -u +%H:%M:%S)"
