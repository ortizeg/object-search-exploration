#!/usr/bin/env bash
# Run the full 6-method research-dataset sweep on a CUDA GPU box (e.g. a vast.ai T4) and render
# the HTML report. This is the reproducible recipe behind the committed docs/reports/research
# report's GPU-latency numbers: CPU-only, the transformer methods (owlv2/dino-dense/propose-
# retrieve) are ~seconds/image; on a T4 they are ~10-40x faster, so the full real datasets become
# feasible AND the report's latency column becomes a real, deployment-relevant figure.
#
# Prereqs on the box: a CUDA 12.x image, git, curl, and an HF_TOKEN exported (the datasets pull
# straight from HuggingFace here -- no local transfer). Usage:
#   HF_TOKEN=hf_xxx BRANCH=worktree/proud-ford bash scripts/gpu_bench.sh
#
# It leaves two artifacts for pull-back: docs/reports/research-report.html and
# docs/benchmark/research-results.json.
set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN (the datasets + owlv2 weights are gated/large)}"
export HF_HUB_DISABLE_XET=1
# Opt this run into the CUDA execution provider (the default is CPU, for bit-identical repro).
export OS_ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
command -v "$PIXI" >/dev/null 2>&1 || { curl -fsSL https://pixi.sh/install.sh | bash; PIXI="$HOME/.pixi/bin/pixi"; }

echo "== 1/6  install envs (default + export) =="
"$PIXI" install
"$PIXI" install -e export
# Swap the pinned CPU onnxruntime for onnxruntime-gpu INSIDE the pixi env (ephemeral box only; the
# committed pin stays CPU for the macOS wheel-tag reason). onnxruntime-gpu >=1.19 bundles CUDA 12.
"$PIXI" run pip uninstall -y onnxruntime >/dev/null 2>&1 || true
"$PIXI" run pip install "onnxruntime-gpu>=1.19,<2"
"$PIXI" run python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())"

echo "== 2/6  fetch / export models (dinov2, superpoint, fastsam, owlv2) =="
"$PIXI" run fetch-models || true                       # dinov2 (HF) + superpoint (GH release)
"$PIXI" run -e export export-fastsam                    # FastSAM (Ultralytics/torch, AGPL)
"$PIXI" run -e export export-owlv2                      # OWLv2 (transformers/torch, Apache-2.0)

echo "== 3/6  fetch + convert datasets (RPINE, FSCD-147, FSCD-LVIS) from HuggingFace =="
"$PIXI" run fetch-datasets

echo "== 4/6  build real split manifests + materialize carved val/ dirs =="
"$PIXI" run python - <<'PYEOF'
import shutil
from pathlib import Path
from object_search.eval.splits import NativeSplits, build_all_manifests

DATASETS = Path("datasets")
def ids(d: Path) -> tuple[str, ...]:
    return tuple(sorted(p.name[:-len(".gt.json")] for p in d.glob("*.gt.json"))) if d.is_dir() else ()

# Native id lists from the freshly converted real tree. FSCD-147 ships a native val/ (normalize
# emits val + test); RPINE / FSCD-LVIS have only train + test, so build_all_manifests carves a
# seeded val from train (D-03).
native = {
    "rpine":     NativeSplits(train=ids(DATASETS/"rpine"/"train"),   test=ids(DATASETS/"rpine"/"test")),
    "fscd147":   NativeSplits(train=ids(DATASETS/"fscd147"/"train"), val=ids(DATASETS/"fscd147"/"val"),
                              test=ids(DATASETS/"fscd147"/"test")),
    "fscd_lvis": NativeSplits(train=ids(DATASETS/"fscd_lvis"/"train"), test=ids(DATASETS/"fscd_lvis"/"test")),
    "carpk":     NativeSplits(), "pucpr_plus": NativeSplits(),
}
manifests = build_all_manifests(native, write=True)

# Materialize datasets/<ds>/val/ for the carve datasets: the carved val ids live physically under
# train/, but the benchmark reads datasets/<ds>/<split>/. Copy each carved-val sidecar + scene.
for ds in ("rpine", "fscd_lvis"):
    val_ids = manifests[ds].val
    src, dst = DATASETS/ds/"train", DATASETS/ds/"val"
    dst.mkdir(parents=True, exist_ok=True)
    for image_id in val_ids:
        for ext in (".gt.json", ".png"):
            s = src/f"{image_id}{ext}"
            if s.is_file():
                shutil.copyfile(s, dst/f"{image_id}{ext}")
    print(f"{ds}: materialized {len(val_ids)} val scenes into {dst}")
for ds, m in manifests.items():
    print(f"{ds}: train={len(m.train)} val={len(m.val)} test={len(m.test)} ({m.val_strategy})")
PYEOF

echo "== 5/6  run the 6-method sweep on GPU (method x dataset x {1,3} x {val,test}) =="
"$PIXI" run bench-research \
  research_root=datasets \
  'datasets=[rpine,fscd147,fscd_lvis]' \
  'methods=[ncc,mosse,sparse-geo,dino-dense,propose-retrieve,owlv2-oneshot]' \
  'splits=[val,test]' \
  'exemplar_counts=[1,3]'

echo "== 6/6  render the HTML report =="
"$PIXI" run report-research
echo "DONE. Artifacts:"
ls -la docs/reports/research-report.html docs/benchmark/research-results.json
