#!/usr/bin/env bash
# Run the full 6-method research-dataset sweep on a CUDA GPU box (e.g. a vast.ai T4), tune each
# method's acceptance threshold to the floor-plan domain, and render the HTML report. This is the
# reproducible recipe behind the committed docs/reports/research report's GPU-latency numbers:
# CPU-only, the transformer methods (owlv2/dino-dense/propose-retrieve) are ~seconds/image; on a T4
# they are ~10-40x faster, so the full real datasets become feasible AND the report's latency column
# becomes a real, deployment-relevant figure.
#
# Prereqs on the box: a CUDA 12.x image, git, curl, and an HF_TOKEN exported (rpine/fscd* pull
# straight from HuggingFace -- no local transfer).
#
# Floor plans (floorplans-door / floorplans-window) is a MANUAL dataset -- there is no ungated URL,
# so its Roboflow COCO export must be copied to the box BEFORE running this script, e.g.:
#   scp -r ~/Downloads/floorPlans \
#       <user>@<box>:/path/to/object-search-exploration/datasets/_incoming/floorplans
# fetch-datasets (step 3) then converts it; if the drop is absent, the floor-plan cells are skipped
# and the rest of the sweep still runs.
#
# Usage:
#   HF_TOKEN=hf_xxx BRANCH=worktree/proud-ford bash scripts/gpu_bench.sh
#
# Artifacts for pull-back: docs/reports/research-report.html, docs/benchmark/research-results.json,
# and (when floor plans were present) docs/benchmark/floorplans-{door,window}-tuning-results.json.
set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN (the datasets + owlv2 weights are gated/large)}"
export HF_HUB_DISABLE_XET=1
# Opt this run into the CUDA execution provider (the default is CPU, for bit-identical repro).
export OS_ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
command -v "$PIXI" >/dev/null 2>&1 || { curl -fsSL https://pixi.sh/install.sh | bash; PIXI="$HOME/.pixi/bin/pixi"; }

echo "== 1/7  install envs (default + export) =="
"$PIXI" install
"$PIXI" install -e export
# Swap the pinned CPU onnxruntime for onnxruntime-gpu INSIDE the pixi env (ephemeral box only; the
# committed pin stays CPU for the macOS wheel-tag reason). TWO gotchas make the naive
# `pixi run pip uninstall/install` a silent no-op that leaves the learned methods on CPU:
#   1. onnxruntime is a CONDA package here, so `pip uninstall onnxruntime` cannot remove it, and
#   2. the pixi env ships NO pip, so `pixi run pip ...` fails outright.
# So: bootstrap pip into the env, force-install the GPU build OVER the conda CPU one (--no-deps keeps
# the pinned numpy/etc.), and put the CUDA image's libs on the load path so the CUDA provider loads.
PYBIN="$("$PIXI" run -q which python)"
"$PYBIN" -m ensurepip --upgrade
"$PYBIN" -m pip install --force-reinstall --no-deps "onnxruntime-gpu>=1.19,<2"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OS_ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
# Run inference via the env python DIRECTLY below (not `pixi run`), so a later `pixi run` cannot
# re-sync the conda CPU onnxruntime back over this pip install. Assert CUDA is actually available.
"$PYBIN" -c "import onnxruntime as o; ps=o.get_available_providers(); print('ORT providers:', ps); assert 'CUDAExecutionProvider' in ps, 'CUDAExecutionProvider missing -- check CUDA libs / driver'"

echo "== 2/7  fetch / export models (dinov2, superpoint, fastsam, owlv2) =="
"$PIXI" run fetch-models || true                       # dinov2 (HF) + superpoint (GH release)
"$PIXI" run -e export export-fastsam                    # FastSAM (Ultralytics/torch, AGPL)
"$PIXI" run -e export export-owlv2                      # OWLv2 (transformers/torch, Apache-2.0)

echo "== 3/7  fetch + convert datasets (RPINE, FSCD-147, FSCD-LVIS) from HuggingFace =="
"$PIXI" run fetch-datasets

echo "== 4/7  build real split manifests + materialize carved val/ dirs =="
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

# Floor plans is manual (step 3 converted it iff the COCO tree was scp'd to _incoming). Its split
# manifests are COMMITTED and native (valid ships a val split), so no carve/materialize is needed --
# the converter writes datasets/floorplans-<class>/{val,test}/ directly. Include the two class keys
# in the sweep + tuning only when present, so a run without the drop still completes.
FP_DATASETS=""
if [ -d datasets/floorplans-door/test ] && [ -d datasets/floorplans-window/test ]; then
  FP_DATASETS=",floorplans-door,floorplans-window"
  echo "floor plans present -> included in the sweep + threshold tuning"
else
  echo "floor plans NOT present (no datasets/_incoming/floorplans drop) -> skipping floor plans"
fi

echo "== 5/7  6-method sweep on GPU -- ONE PROCESS PER METHOD (method x dataset x {1,3} x {val,test}) =="
# Per-method is not cosmetic: the learned methods' ONNX sessions (dino + fastsam + owlv2) do not all
# fit on a 24GB card at once, and onnxruntime re-allocates its CUDA arena per distinct input
# resolution (real datasets vary in image size), so sharing ONE process OOMs dino-dense /
# propose-retrieve on high-res sets like the floor plans. Running each method in its own process
# keeps a single model resident; the per-method result files are then merged into the standard
# research-results.json the report reads. Uses the env python directly (see step 1) so `pixi run`
# cannot re-sync the conda CPU onnxruntime over the GPU build.
ALL_DATASETS="rpine,fscd147,fscd_lvis${FP_DATASETS}"
METHODS="ncc mosse sparse-geo dino-dense propose-retrieve owlv2-oneshot"
mkdir -p docs/benchmark/permethod
for m in $METHODS; do
  echo "--- sweep $m ---"
  "$PYBIN" -m object_search.eval.benchmark --research research_root=datasets \
    "datasets=[${ALL_DATASETS}]" "methods=[$m]" 'splits=[val,test]' 'exemplar_counts=[1,3]' \
    "research_out=docs/benchmark/permethod/rr-$m.json" || echo "SWEEP_FAIL $m"
done
"$PYBIN" - <<'PYEOF'
import json, glob, pathlib
merged = None
for f in sorted(glob.glob("docs/benchmark/permethod/rr-*.json")):
    d = json.load(open(f))
    if merged is None:
        merged = {k: v for k, v in d.items() if k != "cells"}; merged["cells"] = []
    merged["cells"].extend(d.get("cells", []))
if merged:
    pathlib.Path("docs/benchmark/research-results.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print("merged", len(merged["cells"]), "sweep cells")
PYEOF

echo "== 6/7  domain threshold tuning on floor plans (per dataset x method, val->freeze->test) =="
if [ -n "$FP_DATASETS" ]; then
  for ds in floorplans-door floorplans-window; do
    for m in $METHODS; do
      echo "--- tune $ds / $m ---"
      "$PYBIN" - "$ds" "$m" <<'PYEOF' || echo "TUNE_FAIL"
import sys
from object_search.eval.tuning import run_domain_tuning
ds, m = sys.argv[1], sys.argv[2]
run_domain_tuning(ds, "datasets", methods=(m,), exemplar_count=1,
                  out=f"docs/benchmark/permethod/tune-{ds}-{m}.json")
PYEOF
    done
  done
  "$PYBIN" - <<'PYEOF'
import json, glob, pathlib
for ds in ("floorplans-door", "floorplans-window"):
    merged = None
    for f in sorted(glob.glob(f"docs/benchmark/permethod/tune-{ds}-*.json")):
        d = json.load(open(f))
        if merged is None:
            merged = {k: v for k, v in d.items() if k != "methods"}; merged["methods"] = []
        merged["methods"].extend(d.get("methods", []))
    if merged:
        pathlib.Path(f"docs/benchmark/{ds}-tuning-results.json").write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(ds, "->", len(merged["methods"]), "methods")
PYEOF
else
  echo "  skipped (floor plans not present)"
fi

echo "== 7/7  render the HTML report =="
"$PYBIN" scripts/build_research_report.py
echo "DONE. Artifacts:"
ls -la docs/reports/research-report.html docs/benchmark/research-results.json
[ -n "$FP_DATASETS" ] && ls -la \
  docs/benchmark/floorplans-door-tuning-results.json \
  docs/benchmark/floorplans-window-tuning-results.json || true
