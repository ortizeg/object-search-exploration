#!/usr/bin/env python
"""Build a rich, readable floor-plan evaluation report: dataset statistics + qualitative TP/FP/FN
overlays per method on easy vs hard plans, plus the tuned leaderboard and per-slice tables.

Two output tiers, split by the dataset licence (floor-plan pixels must not be redistributed):

* **Committable (no plan pixels)** -- aggregate statistic charts under ``docs/eval/img/floorplans/``
  (histograms of symbol size, crowding, resolution; per-split instance counts). These are derived
  numbers, safe to commit.
* **Local only (gitignored)** -- the TP/FP/FN overlay PNGs (which embed the floor-plan images) and
  the assembled ``docs/benchmark/floorplans-report.html``, for the user to read but never committed.

Run after the tuned results are present locally at ``docs/benchmark/floorplans-{door,window}-tuning-
results.json`` and the converted data at ``datasets/floorplans-{door,window}/{val,test}/``:

    pixi run python scripts/build_floorplans_report.py
"""

from __future__ import annotations

import base64
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import numpy.typing as npt

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from object_search.eval.labels import GroundTruth, load_research_ground_truth
from object_search.eval.metrics import precision_recall_f1
from object_search.provenance import repo_root
from object_search.schemas.geometry import BBox
from object_search.search import get_method

# Bucket cuts -- kept in lockstep with object_search.eval.benchmark (area fractions of the plan).
_SIZE_SMALL_MAX = 0.004
_SIZE_MEDIUM_MAX = 0.016
_METHODS = ("ncc", "mosse", "sparse-geo", "propose-retrieve", "dino-dense", "owlv2-oneshot")
_DINO_SIDE = 1120
_ROOT = repo_root()
_DATASETS = _ROOT / "datasets"
_IMG_DIR = _ROOT / "docs" / "eval" / "img" / "floorplans"  # committable charts
_OUT_DIR = _ROOT / "docs" / "benchmark"  # gitignored html + overlays
_OVERLAY_DIR = _OUT_DIR / "floorplans-overlays"


def _size_bucket(box: BBox, plan_area: int) -> str:
    frac = box.area / plan_area
    return "small" if frac < _SIZE_SMALL_MAX else "medium" if frac < _SIZE_MEDIUM_MAX else "large"


def _load_split(dataset: str, split: str) -> list[GroundTruth]:
    d = _DATASETS / dataset / split
    gts = [load_research_ground_truth(p) for p in sorted(d.glob("*.gt.json"))]
    return [g for g in gts if g is not None]


# --------------------------------------------------------------------------- dataset statistics


def _dataset_stats() -> dict[str, Any]:
    """Per class+split: image/instance counts, and the size/crowding/resolution distributions."""
    stats: dict[str, Any] = {}
    for dataset in ("floorplans-door", "floorplans-window"):
        cls = dataset.split("-")[1]
        for split in ("val", "test"):
            gts = _load_split(dataset, split)
            fracs: list[float] = []
            counts: list[int] = []
            longsides: list[int] = []
            sizes: Counter[str] = Counter()
            for g in gts:
                counts.append(g.achieved_count)
                if g.width and g.height:
                    area = g.width * g.height
                    longsides.append(max(g.width, g.height))
                    for b in g.boxes:
                        fracs.append(b.area / area)
                        sizes[_size_bucket(b, area)] += 1
            stats[f"{cls}/{split}"] = {
                "images": len(gts),
                "instances": sum(counts),
                "per_plan": counts,
                "fracs": fracs,
                "longsides": longsides,
                "size_buckets": dict(sizes),
            }
    return stats


def _chart_size_distribution(stats: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cls in zip(axes, ("door", "window"), strict=True):
        fr = np.array(stats[f"{cls}/test"]["fracs"] + stats[f"{cls}/val"]["fracs"]) * 100
        ax.hist(fr, bins=40, color="#4C78A8", edgecolor="white")
        ax.axvline(_SIZE_SMALL_MAX * 100, color="#E45756", ls="--", label="small|medium (0.4%)")
        ax.axvline(_SIZE_MEDIUM_MAX * 100, color="#F58518", ls="--", label="medium|large (1.6%)")
        ax.set_title(f"{cls}: symbol area as % of plan (val+test)")
        ax.set_xlabel("box area / plan area (%)")
        ax.set_ylabel("# instances")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = _IMG_DIR / "size-distribution.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def _chart_counts(stats: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # instances by size bucket, per class (test)
    order = ["small", "medium", "large"]
    x = np.arange(len(order))
    for i, cls in enumerate(("door", "window")):
        sb = stats[f"{cls}/test"]["size_buckets"]
        axes[0].bar(x + i * 0.4, [sb.get(k, 0) for k in order], 0.4, label=cls)
    axes[0].set_xticks(x + 0.2)
    axes[0].set_xticklabels(order)
    axes[0].set_title("test instances by symbol-size bucket")
    axes[0].set_ylabel("# instances")
    axes[0].legend()
    # crowding: instances-per-plan histogram (test)
    for cls in ("door", "window"):
        axes[1].hist(stats[f"{cls}/test"]["per_plan"], bins=range(0, 40, 3), alpha=0.6, label=cls)
    axes[1].set_title("test: instances per plan (crowding)")
    axes[1].set_xlabel("# instances in a plan")
    axes[1].set_ylabel("# plans")
    axes[1].legend()
    fig.tight_layout()
    p = _IMG_DIR / "counts.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------- overlays


def _classify(pred: list[BBox], gt: list[BBox], iou: float = 0.5) -> tuple[list[int], list[int]]:
    """Greedy match; return (indices of TP preds, indices of matched GT). FP = the rest of preds."""
    matched_gt: list[int] = []
    used = [False] * len(gt)
    tp_pred: list[int] = []
    for pi, pb in enumerate(pred):
        best, best_iou = -1, iou
        for gi, gb in enumerate(gt):
            if used[gi]:
                continue
            v = pb.iou(gb)
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            used[best] = True
            matched_gt.append(best)
            tp_pred.append(pi)
    return tp_pred, [i for i, u in enumerate(used) if u]


def _draw_overlay(
    scene: npt.NDArray[np.uint8], gt: list[BBox], pred: list[BBox], exemplar: BBox
) -> npt.NDArray[np.uint8]:
    """TP=green (matched GT), FN=yellow (missed GT), FP=red (unmatched pred), exemplar=azure blue.

    Colours are OpenCV **BGR** tuples, so ``(255, 200, 0)`` is azure blue (RGB ``0,200,255``), not
    orange. The exemplar is drawn last and thicker (3px) so it stays visible on top of the green TP
    box it usually also earns -- it is itself a ground-truth instance, not a separate detection.
    """
    img = scene.copy()
    tp_pred, matched_gt = _classify(pred, gt)
    matched_set, tp_set = set(matched_gt), set(tp_pred)
    for i, b in enumerate(gt):  # GT: green if matched (TP), yellow if missed (FN)
        color = (0, 200, 0) if i in matched_set else (0, 220, 220)
        cv2.rectangle(img, (b.x, b.y), (b.x2, b.y2), color, 2)
    for i, b in enumerate(pred):  # FP predictions only (red); TP preds already shown as green GT
        if i not in tp_set:
            cv2.rectangle(img, (b.x, b.y), (b.x2, b.y2), (0, 0, 235), 2)
    cv2.rectangle(img, (exemplar.x, exemplar.y), (exemplar.x2, exemplar.y2), (255, 200, 0), 3)
    return img


def _tuned_config(dataset: str, method: str) -> Any:
    """Build the method's tuned config from the committed tuning-results file (+ dino letterbox)."""
    data = json.loads((_OUT_DIR / f"{dataset}-tuning-results.json").read_text())
    overrides: dict[str, Any] = {}
    for e in data["methods"]:
        if e["method"] == method:
            overrides = dict(e.get("tuned_overrides") or {})
            break
    if method == "dino-dense":
        overrides["fixed_input_side"] = _DINO_SIDE
    spec = get_method(method)
    # tuple-ify list overrides (scales) for the frozen config
    overrides = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items()}
    return spec.config_model(**overrides)


def _run_method(dataset: str, image_id: str, method: str) -> tuple[list[BBox], float] | None:
    """Run one method on one plan with its tuned config; return (pred_boxes, f1) or None on error."""
    split = "test"
    gt = load_research_ground_truth(_DATASETS / dataset / split / f"{image_id}.gt.json")
    if gt is None:
        return None
    scene = cv2.imread(str(_DATASETS / dataset / split / f"{image_id}.png"), cv2.IMREAD_COLOR)
    try:
        spec = get_method(method)
        result = spec.fn(
            np.asarray(scene, np.uint8), gt.exemplar_at(1), _tuned_config(dataset, method)
        )
    except Exception as exc:  # missing weights / OOM etc. -> skip this cell
        logger.warning("{}/{} on {} failed: {}", method, dataset, image_id, exc)
        return None
    pred = [m.box for m in result.matches]
    tp_pred, matched = _classify(pred, list(gt.boxes))
    tp, fp, fn = len(tp_pred), len(pred) - len(tp_pred), len(gt.boxes) - len(matched)
    _, _, f1 = precision_recall_f1(tp, fp, fn)
    return pred, (f1 or 0.0)


def _pick_easy_hard(dataset: str, ranker: str = "ncc") -> tuple[str, str]:
    """Rank test plans (>=3 instances) by the ranker method's F1; return (easy_id, hard_id)."""
    ids = sorted(
        p.name[: -len(".gt.json")] for p in (_DATASETS / dataset / "test").glob("*.gt.json")
    )
    scored = []
    for iid in ids:
        gt = load_research_ground_truth(_DATASETS / dataset / "test" / f"{iid}.gt.json")
        if gt is None or gt.achieved_count < 3:
            continue
        r = _run_method(dataset, iid, ranker)
        if r is not None:
            scored.append((iid, r[1]))
    scored.sort(key=lambda t: t[1])
    return scored[-1][0], scored[0][0]  # highest F1 (easy), lowest (hard)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    _IMG_DIR.mkdir(parents=True, exist_ok=True)
    _OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("computing dataset statistics + charts")
    stats = _dataset_stats()
    size_chart = _chart_size_distribution(stats)
    counts_chart = _chart_counts(stats)

    logger.info("selecting easy/hard plans + rendering per-method overlays")
    overlays: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("floorplans-door", "floorplans-window"):
        easy, hard = _pick_easy_hard(dataset)
        overlays[dataset] = []
        for label, iid in (("easy", easy), ("hard", hard)):
            gt = load_research_ground_truth(_DATASETS / dataset / "test" / f"{iid}.gt.json")
            assert gt is not None  # noqa: S101
            scene = cv2.imread(str(_DATASETS / dataset / "test" / f"{iid}.png"), cv2.IMREAD_COLOR)
            panels: list[dict[str, Any]] = []
            for method in _METHODS:
                r = _run_method(dataset, iid, method)
                if r is None:
                    panels.append({"method": method, "img": None, "f1": None})
                    continue
                pred, f1 = r
                img = _draw_overlay(
                    np.asarray(scene, np.uint8), list(gt.boxes), pred, gt.exemplar.box
                )
                op = _OVERLAY_DIR / f"{dataset}-{label}-{method}.png"
                cv2.imwrite(str(op), img)
                panels.append({"method": method, "img": op, "f1": f1, "n_pred": len(pred)})
            overlays[dataset].append(
                {"label": label, "image_id": iid, "n_gt": gt.achieved_count, "panels": panels}
            )

    _write_html(stats, size_chart, counts_chart, overlays)
    logger.info("wrote {}", _OUT_DIR / "floorplans-report.html")


def _crowding_bucket(count: int) -> str:
    """Instances-per-plan bucket -- identical cuts to the eval's ``by_crowding`` slice."""
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 15:
        return "6-15"
    return "16+"


def _resolution_bucket(long_side: int) -> str:
    """Plan long-side bucket -- identical cuts to the eval's ``by_plan_resolution`` slice."""
    if long_side <= 800:
        return "<=800"
    if long_side <= 1600:
        return "<=1600"
    return ">1600"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render one HTML table from headers + already-formatted cells."""
    head = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _leaderboard(dataset: str, fmt: Any) -> str:
    """Tuned-vs-default leaderboard for one dataset, read from its tuning-results file."""
    path = _OUT_DIR / f"{dataset}-tuning-results.json"
    if not path.is_file():
        return "<p style='color:#999'>no tuning-results file for this dataset</p>"
    data = json.loads(path.read_text())
    rows = []
    for e in data.get("methods", []):
        t, d = e["tuned_test"], e["default_test"]
        rows.append(
            (
                t.get("f1"),
                [
                    f"<code>{e['method']}</code>",
                    f"<b>{fmt(t.get('f1'))}</b>",
                    fmt(d.get("f1")),
                    fmt(e.get("delta_f1")),
                    fmt(t.get("precision")),
                    fmt(t.get("recall")),
                    fmt(t.get("ap50")),
                    fmt(t.get("mae"), 1),
                    f"{t.get('n_scored')}/{t.get('n_images')}",
                    f"<code>{e.get('tuned_overrides')}</code>",
                ],
            )
        )
    rows.sort(key=lambda r: (r[0] is None, -(r[0] or 0)))
    return _table(
        [
            "method",
            "tuned F1",
            "default F1",
            "Δ",
            "P",
            "R",
            "AP50",
            "MAE",
            "coverage",
            "tuned config",
        ],
        [r[1] for r in rows],
    )


def _slice_table(dataset: str, slice_key: str, metric: str, fmt: Any) -> str:
    """Per-method slice table (e.g. recall by symbol size) from the per-method sweep dumps."""
    data: dict[str, dict[str, Any]] = {}
    buckets: list[str] = []
    for fp in sorted((_OUT_DIR / "permethod").glob("rr-*.json")):
        for c in json.loads(fp.read_text()).get("cells", []):
            if (
                c.get("dataset") == dataset
                and c.get("split") == "test"
                and c.get("exemplar_count") == 1
            ):
                sl = (c.get("slices") or {}).get(slice_key) or {}
                if not sl:
                    continue
                data[c["method"]] = {k: v.get(metric) for k, v in sl.items()}
                for b in sl:
                    if b not in buckets:
                        buckets.append(b)
    if not data:
        return "<p style='color:#999'>no slice data (per-method sweep dumps absent)</p>"
    preferred = ["small", "medium", "large", "1", "2-5", "6-15", "16+", "<=800", "<=1600", ">1600"]
    order = [b for b in preferred if b in buckets] + [b for b in buckets if b not in preferred]
    rows = [[f"<code>{m}</code>"] + [fmt(data[m].get(b), 2) for b in order] for m in sorted(data)]
    return _table(["method", *order], rows)


def _write_html(
    stats: dict[str, Any], size_chart: Path, counts_chart: Path, overlays: dict[str, list[Any]]
) -> None:
    def f(x: Any, n: int = 3) -> str:
        return "n/a" if not isinstance(x, (int, float)) else f"{x:.{n}f}"

    h = [
        "<html><head><meta charset='utf-8'><title>Floor-plan evaluation report</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222}"
        "table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px}"
        "img{max-width:100%;border:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}"
        ".panel{font-size:13px}h2{border-bottom:2px solid #eee;padding-bottom:4px;margin-top:2rem}"
        ".legend span{padding:2px 6px;border-radius:3px;color:#fff;margin-right:6px;font-size:12px}</style></head><body>",
    ]
    h.append("<h1>Floor-plan exemplar-search — evaluation report</h1>")
    h.append(
        "<p>Roboflow floor-plans-500, per-class (door/window). Tuned on val (56 plans), reported "
        "on test (28). Overlays use the <b>tuned</b> config per method.</p>"
    )
    h.append(
        "<p class='legend'><span style='background:#0c8'>TP (matched GT)</span>"
        "<span style='background:#dd0;color:#222'>FN (missed GT)</span>"
        "<span style='background:#e00'>FP (spurious pred)</span>"
        "<span style='background:#00c8ff;color:#222'>exemplar (the query box, drawn thicker)</span>"
        "</p><p style='font-size:13px;color:#555'>The exemplar is the one box handed to the method as "
        "the query; it is also a ground-truth instance, so it is normally re-found and its GREEN TP "
        "box sits underneath the thicker blue outline.</p>"
    )

    h.append("<h2>Dataset statistics</h2>")
    h.append(
        "<table><tr><th>class / split</th><th># plans</th><th># instances</th>"
        "<th>small</th><th>medium</th><th>large</th></tr>"
    )
    for k in ("door/val", "door/test", "window/val", "window/test"):
        s = stats[k]
        tot = max(1, sum(s["size_buckets"].values()))
        sb = s["size_buckets"]
        h.append(
            f"<tr><td>{k}</td><td>{s['images']}</td><td>{s['instances']}</td>"
            + "".join(
                f"<td>{sb.get(b, 0)} ({100 * sb.get(b, 0) / tot:.0f}%)</td>"
                for b in ("small", "medium", "large")
            )
            + "</tr>"
        )
    h.append("</table>")
    h.append(
        "<p style='font-size:13px;color:#555'>Size buckets are box-area ÷ plan-area: "
        "<b>small &lt;0.4%</b>, <b>medium &lt;1.6%</b>, <b>large ≥1.6%</b> — the same cuts the "
        "per-slice evaluation uses. Both classes come from the same plans, so only the kept boxes "
        "differ between door and window.</p>"
    )

    # instances-per-plan (crowding) distribution
    h.append("<h3>Crowding — instances per plan</h3>")
    crowd_order = ["1", "2-5", "6-15", "16+"]
    rows = []
    for k in ("door/val", "door/test", "window/val", "window/test"):
        per = stats[k]["per_plan"]
        c = Counter(_crowding_bucket(n) for n in per)
        med = sorted(per)[len(per) // 2] if per else 0
        rows.append(
            [k]
            + [f"{c.get(b, 0)}" for b in crowd_order]
            + [f"{min(per) if per else 0} / {med} / {max(per) if per else 0}"]
        )
    h.append(_table(["class / split"] + [f"{b} inst" for b in crowd_order] + ["min/med/max"], rows))

    # plan-resolution distribution
    h.append("<h3>Plan resolution — long side</h3>")
    res_order = ["<=800", "<=1600", ">1600"]
    rows = []
    for k in ("door/val", "door/test", "window/val", "window/test"):
        ls = stats[k]["longsides"]
        c = Counter(_resolution_bucket(v) for v in ls)
        rows.append(
            [k]
            + [f"{c.get(b, 0)}" for b in res_order]
            + [f"{min(ls) if ls else 0}–{max(ls) if ls else 0} px"]
        )
    h.append(_table(["class / split", *res_order, "range"], rows))

    h.append(f"<img src='data:image/png;base64,{_b64(size_chart)}'>")
    h.append(f"<img src='data:image/png;base64,{_b64(counts_chart)}'>")

    # ---- results: leaderboard + per-slice breakdowns, per dataset
    for dataset in ("floorplans-door", "floorplans-window"):
        h.append(f"<h2>{dataset} — results</h2>")
        h.append("<h3>Tuned-vs-default leaderboard (test, 1 exemplar)</h3>")
        h.append(_leaderboard(dataset, f))
        h.append("<h3>Recall by symbol size</h3>")
        h.append(_slice_table(dataset, "by_symbol_size", "recall", f))
        h.append("<h3>F1 by crowding (instances per plan)</h3>")
        h.append(_slice_table(dataset, "by_crowding", "f1", f))
        h.append("<h3>F1 by plan resolution</h3>")
        h.append(_slice_table(dataset, "by_plan_resolution", "f1", f))

    for dataset in ("floorplans-door", "floorplans-window"):
        h.append(f"<h2>{dataset} — qualitative overlays</h2>")
        for case in overlays[dataset]:
            h.append(
                f"<h3>{case['label'].upper()} plan <code>{case['image_id']}</code> "
                f"({case['n_gt']} ground-truth instances)</h3><div class='grid'>"
            )
            for p in case["panels"]:
                cap = f"<b>{p['method']}</b> — F1={f(p['f1'], 2)}" + (
                    f", {p['n_pred']} preds" if p.get("n_pred") is not None else " (unavailable)"
                )
                body = (
                    f"<img src='data:image/png;base64,{_b64(p['img'])}'>"
                    if p["img"]
                    else "<div style='color:#999'>model weights unavailable</div>"
                )
                h.append(f"<div class='panel'>{cap}<br>{body}</div>")
            h.append("</div>")
    h.append("</body></html>")
    (_OUT_DIR / "floorplans-report.html").write_text("\n".join(h), encoding="utf-8")


if __name__ == "__main__":
    main()
