#!/usr/bin/env bash
# Fine-tune OWLv2 on the floor-plans train split on a CUDA GPU box (e.g. a vast.ai RTX 3090/4090),
# export both arms to ONNX, and measure baseline-vs-fine-tuned precision/recall on the frozen
# 28-plan floor-plan test splits. This is the reproducible recipe behind the numbers in
# docs/reports/owlv2-floorplans-finetune.md (quick task 260801-8zy).
#
# Two arms, both seeded, both with a val-selected checkpoint:
#   A (primary) headonly -- box_head + class_head only, vision tower frozen;
#   B (stretch) full     -- plus the whole vision tower (the text tower stays frozen in EVERY arm,
#                           because it is not part of the exported graph -- see finetune_owlv2.py).
#
# Prereqs on the box:
#   * a CUDA 12.x image, git, curl;
#   * the Roboflow floor-plans COCO export at datasets/_incoming/floorplans/{train,valid,test}.
#     Floor plans is a MANUAL dataset -- there is NO ungated URL, so it must be copied to the box
#     BEFORE running this script, exactly as scripts/gpu_bench.sh documents:
#       scp -r ~/Downloads/floorPlans <user>@<box>:/path/to/object-search-exploration/datasets/_incoming/floorplans
#     Unlike gpu_bench.sh (which degrades to "skip the floor-plan cells"), this script has nothing
#     to do without them and exits with an actionable message.
#   * HF_TOKEN is OPTIONAL here. Unlike gpu_bench.sh -- which pulls the gated rpine/fscd* datasets
#     and therefore hard-requires it -- everything this script downloads is public: OWLv2's base
#     weights (google/owlv2-base-patch16-ensemble, Apache-2.0) and the DINOv2/SuperPoint artifacts
#     fetch-models already handles. The floor plans arrive by scp, not from the Hub. So a missing
#     HF_TOKEN is a warning, never a failure.
#
# Usage:
#   BRANCH=worktree/radiant-lark bash scripts/gpu_finetune.sh
#
# Artifacts for pull-back (printed again at the end):
#   docs/benchmark/owlv2-finetune/*.json            -- 3 arms x 2 classes tuning reports
#   models/finetune/*/train_log.json                -- the per-epoch train/val curves
#   models/owlv2_base_patch16_floorplans_ft.onnx    -- arm A (the registered artifact)
#   models/owlv2_base_patch16_floorplans_ft_full.onnx -- arm B (unregistered comparison)
set -euo pipefail

export HF_HUB_DISABLE_XET=1
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Everything this script downloads is public (OWLv2 is"
  echo "         Apache-2.0 and ungated; the floor plans arrive by scp), so this is fine -- but if"
  echo "         a Hub download 401s, export HF_TOKEN and re-run."
fi

PIXI="${PIXI:-$HOME/.pixi/bin/pixi}"
command -v "$PIXI" >/dev/null 2>&1 || { curl -fsSL https://pixi.sh/install.sh | bash; PIXI="$HOME/.pixi/bin/pixi"; }

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
ARM_A_DIR="models/finetune/owlv2-floorplans-headonly"
ARM_B_DIR="models/finetune/owlv2-floorplans-full"
ARM_A_ONNX="owlv2_base_patch16_floorplans_ft.onnx"
ARM_B_ONNX="owlv2_base_patch16_floorplans_ft_full.onnx"

echo "== 1/8  install envs (default + export) and assert BOTH CUDA paths =="
"$PIXI" install
"$PIXI" install -e export

# --- 1a. onnxruntime-gpu for the DEFAULT env (the eval path). THREE gotchas, not two:
# onnxruntime is a CONDA package here so `pip uninstall onnxruntime` cannot remove it, and the pixi
# env ships NO pip so `pixi run pip ...` is a silent no-op. Bootstrap pip, force-install over it.
# The third gotcha, found the hard way on a real box: `onnxruntime-gpu>=1.19,<2` resolves to
# whatever is newest (1.28.0 as of this writing), which needs CUDA 13 / cuDNN 9 -- but a
# `pytorch/pytorch:*-cuda12.1-*` box only has CUDA 12.1. The CUDAExecutionProvider then fails to
# LOAD at session-creation time (not at the `get_available_providers()` check below, which only
# reports what's compiled in, not what actually initializes), and onnxruntime silently falls back
# to CPU -- eval that should take minutes takes hours with no error anywhere. Pin to 1.23.2, the
# newest version confirmed to actually load CUDAExecutionProvider on CUDA 12.1/cuDNN 9.
PYBIN="$("$PIXI" run -q which python)"
"$PYBIN" -m ensurepip --upgrade
"$PYBIN" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.23.2"
# Invoke $PYBIN DIRECTLY from here on -- a later `pixi run` can re-sync the conda CPU build over it.
# This only checks what onnxruntime-gpu is COMPILED with, not what actually loads at runtime --
# `get_available_providers()` lists CUDAExecutionProvider even when its .so later fails to dlopen.
# The real load-bearing check is below (after 1b), once the export env's torch -- and therefore
# its bundled cuDNN/cuBLAS libs, which the LD_LIBRARY_PATH setup after 1b depends on -- is known
# to actually be the CUDA build, not before.
"$PYBIN" -c "import onnxruntime as o; ps=o.get_available_providers(); print('ORT providers (compiled in):', ps); assert 'CUDAExecutionProvider' in ps, 'CUDAExecutionProvider missing -- check CUDA libs / driver'"

# --- 1b. NEW gotcha, and the one that silently costs the most: the `export` feature's conda
# `pytorch` can resolve to a CPU-ONLY build on a CUDA box. Training would then run for hours on the
# CPU without a single error. Assert, and force the CUDA wheel in with the same bootstrap-pip /
# --force-reinstall pattern if the assertion fails.
EXPORT_PYBIN="$("$PIXI" run -e export -q which python)"
if ! "$EXPORT_PYBIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "  export env's torch is CPU-only -- force-installing the CUDA wheel"
  "$EXPORT_PYBIN" -m ensurepip --upgrade
  "$EXPORT_PYBIN" -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 torch torchvision
fi
"$EXPORT_PYBIN" -c "import torch; assert torch.cuda.is_available(), 'torch still cannot see the GPU -- refusing to train on CPU'; print('torch CUDA:', torch.cuda.get_device_name(0))"

# --- 1c. cuDNN 9 / cuBLAS libs for onnxruntime's CUDAExecutionProvider: NOT reliably on the
# system CUDA path even on a "cudnn9-devel" image (found by testing, not assumed) -- but always
# present alongside the export env's now-confirmed-CUDA torch install (its wheels bundle them).
# Must run AFTER 1b, not before: if 1b just force-installed the CUDA wheel, that's when these
# paths first exist.
_EXPORT_SITE_PKGS="$(dirname "$(dirname "$EXPORT_PYBIN")")/lib/python3.12/site-packages"
_CUDNN_LIB="$(dirname "$(find "$_EXPORT_SITE_PKGS/nvidia/cudnn/lib" -name 'libcudnn.so.9' -print -quit 2>/dev/null)" 2>/dev/null || true)"
_CUBLAS_LIB="$(dirname "$(find "$_EXPORT_SITE_PKGS/nvidia/cublas/lib" -name 'libcublasLt.so*' -print -quit 2>/dev/null)" 2>/dev/null || true)"
export LD_LIBRARY_PATH="${_CUDNN_LIB}:${_CUBLAS_LIB}:/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export OS_ONNX_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"

# --- 1d. The dataset. This script has nothing to do without it.
for split in train valid test; do
  [ -d "datasets/_incoming/floorplans/$split" ] || {
    echo "FATAL: datasets/_incoming/floorplans/$split is missing."
    echo "Floor plans is a MANUAL dataset. scp the Roboflow COCO export to the box first:"
    echo "  scp -r ~/Downloads/floorPlans <user>@<box>:\$PWD/datasets/_incoming/floorplans"
    exit 1
  }
done
echo "  floor-plan COCO tree present"

echo "== 2/8  fetch models + export the BASELINE pretrained owlv2 graph =="
"$PIXI" run fetch-models || true
"$PIXI" run -e export export-owlv2
test -f models/owlv2_base_patch16.onnx
# The real CUDAExecutionProvider check: does it actually LOAD, not just get reported as compiled
# in. Fail loudly here, before any GPU time is spent training, rather than silently eval-ing on
# CPU for hours later in step 7.
"$PYBIN" -c "
import onnxruntime as o
sess = o.InferenceSession('models/owlv2_base_patch16.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
actual = sess.get_providers()
print('ORT providers (actually loaded):', actual)
assert 'CUDAExecutionProvider' in actual, (
    'CUDAExecutionProvider is compiled in but FAILED TO LOAD (check LD_LIBRARY_PATH for '
    'libcudnn.so.9 / libcublasLt.so.* -- see step 1c) -- would otherwise silently fall back to '
    'CPU for the eval sweep in step 7'
)
"

echo "== 3/8  convert floor plans -> datasets/floorplans-{door,window}/{val,test} =="
"$PIXI" run fetch-datasets
test -d datasets/floorplans-door/test && test -d datasets/floorplans-window/test

echo "== 4/8  arm A (primary): heads only, vision tower frozen =="
"$PIXI" run -e export finetune-owlv2 \
  --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
  --out "$ARM_A_DIR"

echo "== 5/8  arm B (stretch): the whole exported vision path unfrozen =="
"$PIXI" run -e export finetune-owlv2 --unfreeze-all \
  --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" \
  --out "$ARM_B_DIR"

echo "== 6/8  export each arm to its OWN onnx artifact (the shipped baseline stays untouched) =="
"$PIXI" run -e export python scripts/export_owlv2.py --checkpoint "$ARM_A_DIR" --out "$ARM_A_ONNX"
"$PIXI" run -e export python scripts/export_owlv2.py --checkpoint "$ARM_B_DIR" --out "$ARM_B_ONNX"
ls -la "models/$ARM_A_ONNX" "models/$ARM_B_ONNX"
# Re-assert the ORT GPU build: step 6 ran `pixi run`, which can re-sync the conda CPU onnxruntime.
"$PYBIN" -c "import onnxruntime as o; assert 'CUDAExecutionProvider' in o.get_available_providers()" \
  || "$PYBIN" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.23.2"

echo "== 7/8  evaluate 3 arms x 2 classes -- ONE PROCESS PER RUN =="
# Per-run processes mirror gpu_bench.sh: onnxruntime re-allocates its CUDA arena per distinct input
# resolution and the plans vary in size, so a shared process is where the OOMs live. Everything
# except OS_OWLV2_MODEL is identical across the three arms, so the delta is attributable to the
# weights alone. A single failed cell prints TUNE_FAIL and the GPU session continues.
mkdir -p docs/benchmark/owlv2-finetune
for ds in floorplans-door floorplans-window; do
  for arm in baseline headonly full; do
    case "$arm" in
      baseline) MODEL="owlv2_base_patch16.onnx" ;;
      headonly) MODEL="$ARM_A_ONNX" ;;
      full)     MODEL="$ARM_B_ONNX" ;;
    esac
    echo "--- tune $ds / $arm ($MODEL) ---"
    OS_OWLV2_MODEL="$MODEL" "$PYBIN" - "$ds" "$arm" <<'PYEOF' || echo "TUNE_FAIL $arm $ds"
import sys
from object_search.eval.tuning import run_domain_tuning
ds, arm = sys.argv[1], sys.argv[2]
run_domain_tuning(ds, "datasets", methods=("owlv2-oneshot",), exemplar_count=1,
                  out=f"docs/benchmark/owlv2-finetune/{ds}-{arm}.json")
PYEOF
  done
done

echo "== 8/8  sha256 provenance + the pull-back list =="
"$PYBIN" - <<'PYEOF'
from pathlib import Path
from object_search.provenance import file_sha256
for name in ("owlv2_base_patch16.onnx",
             "owlv2_base_patch16_floorplans_ft.onnx",
             "owlv2_base_patch16_floorplans_ft_full.onnx"):
    path = Path("models") / name
    print(f"{name}: {file_sha256(path) if path.is_file() else 'MISSING'}")
PYEOF

echo "DONE. Pull these back:"
ls -la docs/benchmark/owlv2-finetune/*.json || true
ls -la models/finetune/*/train_log.json || true
ls -la "models/$ARM_A_ONNX" "models/$ARM_B_ONNX" || true
echo "Then DESTROY the instance: vastai destroy instance <id>  (and confirm with: vastai show instances)"
