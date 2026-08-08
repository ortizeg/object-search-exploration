"""Committed, deterministic benchmark charts and tables (EVAL-06).

This module turns ``docs/benchmark/results.json`` (written by
:mod:`object_search.eval.benchmark`) into the committed figures under ``docs/benchmark/`` and
the ``results.md`` tables that the README links. It is invoked by ``pixi run bench-charts``.

Four charts, one story each
---------------------------
* ``metrics_by_method.png`` -- pooled precision / recall / F1 / AP per method: the headline
  "which method actually works" bar chart.
* ``crossover_by_scale.png`` -- recall for ``ncc`` vs ``sparse-geo`` split by scale bucket:
  the NCC-vs-sparse-geo crossover the literature predicts (a locked decision, EVAL-19), made
  *visible* rather than averaged away. On the near-identical, low-texture chipset ``sparse-geo``
  cannot raise 20 SIFT keypoints and abstains, exactly the regime where NCC is strongest.
* ``latency_by_canvas.png`` -- per-method p50 latency across the chipset canvas ramp
  (320x240 -> 6000x4000): the EVAL-19 scaling story a single pooled latency would hide.
* ``thumbs_wilson.png`` -- thumbs-up rate per method with **Wilson score intervals** (EVAL-14)
  where human ratings exist. Rating is a manual activity, so when the sample is empty this
  renders an honest "no human ratings recorded" panel rather than a fabricated bar.

``results.md``: the full detail, not just charts
--------------------------------------------------
Alongside the four PNGs, :func:`write_results_markdown` emits every table a reviewer needs: the
pooled leaderboard, **per-regime** (EASY/TEXTURED/VARIED/CLUTTERED) scoreboards for every method
(the primary result -- pooling averages a method's best and worst regimes together), recall by
**ground-truth box size** (small/medium/large, reusing the floor-plan research path's bucketing),
recall by scale bucket, latency by canvas size, and a computed **Insight** section
(:func:`_insight_section`) that reads facts (per-regime winners, size sensitivity) straight off the
tables above it, so it cannot drift out of sync with the numbers on re-render.

Determinism (the load-bearing part)
-----------------------------------
The charts are committed to git, so re-rendering them must be **byte-identical** or every
regeneration would churn the repo. Two things are pinned:

* **Headless Agg backend**, selected before ``pyplot`` is imported, so rendering never depends
  on a display or a GUI toolkit.
* **The PNG ``Software`` metadata chunk is suppressed** (``metadata={"Software": None}``).
  Matplotlib stamps its own version into that chunk by default; dropping it removes the only
  render-environment-dependent bytes in the PNG, so two renders on any matplotlib produce the
  same file. ``tests/test_charts.py`` asserts two renders into separate directories are equal.

Matplotlib does **not** embed a wall-clock timestamp in PNGs (unlike PDF/SVG/PS), so there is
no date to strip; the ``Software`` chunk is the whole of the non-determinism for this backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

# Select the headless backend *before* importing pyplot -- the import binds the backend, so a
# later `use()` would be ignored and the choice must not depend on a $DISPLAY being present.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from loguru import logger
from matplotlib.figure import Figure

from object_search.provenance import repo_root
from object_search.store.wilson import wilson_interval

# Suppressing the `Software` chunk is what makes a committed PNG regenerate byte-identically
# regardless of the matplotlib version that renders it (EVAL-06 determinism).
_PNG_METADATA: dict[str, str | None] = {"Software": None}

# Fixed method order so every chart's bars/lines are in the same, reproducible sequence rather
# than JSON/dict insertion order (a real reproducibility threat, PITFALLS §6).
_METHOD_ORDER: tuple[str, ...] = (
    "ncc",
    "sparse-geo",
    "dino-dense",
    "owlv2-oneshot",
    "propose-retrieve",
    "mosse",
)

# The four pooled rates drawn in the headline chart, with human-readable labels.
_RATE_LABELS: tuple[tuple[str, str], ...] = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("mean_ap", "AP"),
)

# The four synthetic regimes, keyed by image_id prefix (mirrors scripts/build_report.py's
# regime_of(), duplicated rather than shared per the repo's per-report-script convention --
# results.md and benchmark-report.html are independent artifacts that happen to group the same
# way). real-objects/scatter-scaled/cluttered-distractors ids match none of these prefixes and
# are correctly excluded: results.md's per-regime section is the synthetic (chipset/textured)
# story, same scope as benchmark-report.html; the real-objects side has its own dedicated
# real-objects-report.html / real-objects-findings.md.
_REGIMES: tuple[str, ...] = ("EASY", "TEXTURED", "VARIED", "CLUTTERED")


def _regime_of(image_id: str) -> str | None:
    """Classify a chipset/textured image id into one of :data:`_REGIMES`, or ``None``."""
    if image_id.startswith("chipset-"):
        return "EASY"
    if image_id.startswith("textured-plain"):
        return "TEXTURED"
    if image_id.startswith("textured-varied"):
        return "VARIED"
    if image_id.startswith("textured-cluttered"):
        return "CLUTTERED"
    return None


_DEFAULT_RESULTS = "docs/benchmark/results.json"
_DEFAULT_OUT_DIR = "docs/benchmark"


def load_results(path: str | Path = _DEFAULT_RESULTS) -> dict[str, Any]:
    """Read ``results.json``, resolving a relative path against the repo root.

    Args:
        path: Path to the benchmark results JSON.

    Returns:
        The parsed results mapping.

    Raises:
        FileNotFoundError: If the results file does not exist -- the charts have nothing to
            render, which is a loud error rather than an empty figure.
    """
    results_path = Path(path)
    if not results_path.is_absolute():
        results_path = repo_root() / results_path
    if not results_path.is_file():
        raise FileNotFoundError(
            f"no benchmark results at {results_path}; run `pixi run bench` first"
        )
    data: dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    return data


def _present_methods(results: dict[str, Any]) -> list[str]:
    """Methods actually present in the results, in the fixed order (unknown ones appended)."""
    methods = results.get("methods", {})
    ordered = [m for m in _METHOD_ORDER if m in methods]
    extra = [m for m in methods if m not in _METHOD_ORDER]
    return ordered + sorted(extra)


def _save(fig: Figure, out: Path) -> Path:
    """Save a figure deterministically (suppressed Software chunk) and close it.

    Closing matters under the Agg backend: figures accumulate otherwise, and in a long render
    that is a memory leak with no visible symptom until the process is large.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=100, metadata=_PNG_METADATA)
    plt.close(fig)
    logger.info("charts: wrote {}", out)
    return out


def _rate(overall: dict[str, Any], key: str) -> float:
    """A pooled rate as a float for plotting; ``None`` (abstention) renders as a zero-height bar.

    The bar height is only a visual; the honest ``None``-vs-``0`` distinction is preserved in
    the ``results.md`` tables, which print ``n/a`` for an abstention rather than ``0.00``.
    """
    value = overall.get(key)
    return float(value) if value is not None else 0.0


def render_metrics_chart(results: dict[str, Any], out_dir: Path) -> Path:
    """Grouped bar chart of pooled precision / recall / F1 / AP per method."""
    methods = _present_methods(results)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    n_rates = len(_RATE_LABELS)
    group_width = 0.8
    bar_width = group_width / max(len(methods), 1)
    for m_idx, method in enumerate(methods):
        overall = results["methods"][method]["overall"]
        heights = [_rate(overall, key) for key, _ in _RATE_LABELS]
        positions = [
            r_idx - group_width / 2 + bar_width * (m_idx + 0.5) for r_idx in range(n_rates)
        ]
        ax.bar(positions, heights, width=bar_width, label=method)
    ax.set_xticks(range(n_rates))
    ax.set_xticklabels([label for _, label in _RATE_LABELS])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Pooled detection metrics by method")
    ax.legend(title="method", loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "metrics_by_method.png")


def render_crossover_chart(results: dict[str, Any], out_dir: Path) -> Path:
    """Recall for ``ncc`` vs ``sparse-geo`` by scale bucket -- the crossover, made visible."""
    pair = [m for m in ("ncc", "sparse-geo") if m in results.get("methods", {})]
    buckets = ("fixed", "varied")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    bar_width = 0.35
    for m_idx, method in enumerate(pair):
        slices = results["methods"][method]["slices"]["by_scale_bucket"]
        heights = [_rate(slices.get(bucket, {}), "recall") for bucket in buckets]
        positions = [b_idx + bar_width * (m_idx - 0.5) for b_idx in range(len(buckets))]
        ax.bar(positions, heights, width=bar_width, label=method)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels([f"{b}\nscale" for b in buckets])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("recall")
    ax.set_title("NCC vs sparse-geo crossover (recall by scale bucket)")
    ax.legend(title="method", loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "crossover_by_scale.png")


def _canvas_area(canvas: str) -> int:
    """Pixel area of a ``"WxH"`` canvas label, for ordering the latency x-axis by size."""
    width, _, height = canvas.partition("x")
    return int(width) * int(height)


def render_latency_chart(results: dict[str, Any], out_dir: Path) -> Path:
    """Per-method p50 latency across the canvas-size ramp (the EVAL-19 scaling story)."""
    methods = _present_methods(results)
    all_canvases: set[str] = set()
    for method in methods:
        all_canvases.update(results["methods"][method]["slices"]["by_canvas_size"])
    canvases = sorted(all_canvases, key=_canvas_area)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for method in methods:
        by_canvas = results["methods"][method]["slices"]["by_canvas_size"]
        xs: list[int] = []
        ys: list[float] = []
        for x_idx, canvas in enumerate(canvases):
            p50 = by_canvas.get(canvas, {}).get("latency_ms", {}).get("p50")
            if p50 is not None:
                xs.append(x_idx)
                ys.append(float(p50))
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_xticks(range(len(canvases)))
    ax.set_xticklabels(canvases, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("p50 latency (ms, log scale)")
    ax.set_xlabel("canvas size")
    ax.set_title("Latency by canvas size")
    ax.legend(title="method", loc="upper left", framealpha=0.9)
    ax.grid(axis="both", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "latency_by_canvas.png")


def render_thumbs_chart(ratings: dict[str, tuple[int, int]] | None, out_dir: Path) -> Path:
    """Thumbs-up rate with Wilson intervals (EVAL-14), or an honest empty-state panel.

    Args:
        ratings: ``method -> (n_up, n_rated)``. ``None`` or an all-zero-``n`` mapping means no
            human ratings have been recorded yet, which is the expected state early on -- rating
            is manual. In that case the chart says so plainly rather than drawing a bar from no
            data, the exact false-certainty the Wilson interval exists to prevent.
        out_dir: Output directory.
    """
    populated = {method: (up, n) for method, (up, n) in (ratings or {}).items() if n > 0}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    if not populated:
        ax.text(
            0.5,
            0.5,
            "no human ratings recorded yet (n = 0)\nrate runs in the UI to populate this chart",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title("Thumbs-up rate with Wilson 95% intervals")
        fig.tight_layout()
        return _save(fig, out_dir / "thumbs_wilson.png")

    methods = [m for m in _METHOD_ORDER if m in populated] + sorted(
        m for m in populated if m not in _METHOD_ORDER
    )
    centres: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    labels: list[str] = []
    for method in methods:
        up, n = populated[method]
        interval = wilson_interval(up, n)
        lower, upper = interval if interval is not None else (0.0, 1.0)
        rate = up / n
        centres.append(rate)
        lowers.append(rate - lower)
        uppers.append(upper - rate)
        labels.append(f"{method}\n(n={n})")
    ax.bar(range(len(methods)), centres, width=0.6, color="tab:green", alpha=0.8)
    ax.errorbar(
        range(len(methods)),
        centres,
        yerr=[lowers, uppers],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("thumbs-up rate")
    ax.set_title("Thumbs-up rate with Wilson 95% intervals")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_dir / "thumbs_wilson.png")


def _fmt(value: float | int | None, digits: int = 3) -> str:
    """Format a metric for a Markdown table: ``n/a`` for ``None`` (abstention), never ``0``."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _rows_by_regime(results: dict[str, Any], method: str) -> dict[str, list[dict[str, Any]]]:
    """Group ``method``'s ``per_image`` rows (as loaded from JSON) by :func:`_regime_of`."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results["methods"][method]["per_image"]:
        regime = _regime_of(row["image_id"])
        if regime is not None:
            grouped.setdefault(regime, []).append(row)
    return grouped


def _pooled_rate(rows: list[dict[str, Any]], metric: str) -> float | None:
    """Micro-averaged precision/recall/F1, or macro-averaged AP, over a list of ``per_image`` rows.

    Mirrors :func:`object_search.eval.benchmark._aggregate`'s pooling convention (sum tp/fp/fn then
    divide; mean of per-image AP) but reads the JSON-dict row shape ``write_results_markdown``
    already has in hand, rather than re-deriving it from the Pydantic ``ImageResult`` records.
    """
    scored = [r for r in rows if r.get("tp") is not None]
    if metric == "ap":
        aps = [r["ap"] for r in scored if r.get("ap") is not None]
        return (sum(aps) / len(aps)) if aps else None
    if not scored:
        return None
    tp = sum(r["tp"] for r in scored)
    fp = sum(r["fp"] for r in scored)
    fn = sum(r["fn"] for r in scored)
    if metric == "precision":
        return tp / (tp + fp) if (tp + fp) else None
    if metric == "recall":
        return tp / (tp + fn) if (tp + fn) else None
    if metric == "f1":
        denom = 2 * tp + fp + fn
        return (2 * tp) / denom if denom else None
    raise ValueError(f"unknown metric {metric!r}")


def _insight_section(
    methods: list[str],
    regime_best: dict[str, tuple[str, float]],
    size_gap: dict[str, float],
) -> list[str]:
    """Compute a short "what the tables above show" section -- facts, not fixed prose.

    Every sentence is derived from ``regime_best``/``size_gap`` (themselves computed while
    rendering the tables above), so this section cannot go stale relative to the numbers next to
    it: re-running ``pixi run bench-charts`` after any method/config change updates both together.
    """
    lines = ["## Insight", ""]
    if not regime_best:
        lines.append("_No scored regime rows -- run `pixi run bench` (not `bench-ci`) first._")
        lines.append("")
        return lines

    winners = {regime: method for regime, (method, _f1) in regime_best.items()}
    distinct_winners = sorted(set(winners.values()))
    lines.append(
        "**Per-regime winner (F1):** "
        + "; ".join(
            f"{regime} = `{method}` ({f1:.2f})" for regime, (method, f1) in regime_best.items()
        )
        + "."
    )
    if len(distinct_winners) == 1:
        lines.append(
            f"`{distinct_winners[0]}` wins every regime here -- unusual; check whether the other "
            "methods errored or abstained rather than genuinely lost (see the pooled table's "
            "abstentions/errors columns)."
        )
    else:
        lines.append(
            f"No single method wins every regime ({len(distinct_winners)} different winners across "
            f"{len(regime_best)} regimes) -- this is the reason all {len(methods)} methods are "
            "swept rather than picking one, and why the per-regime table above is the result to "
            "read, not the pooled summary."
        )
    lines.append("")

    if size_gap:
        most_size_sensitive = max(size_gap, key=lambda m: size_gap[m])
        least_size_sensitive = min(size_gap, key=lambda m: size_gap[m])
        lines.append(
            f"**Size sensitivity (large-bucket recall minus small-bucket recall):** "
            f"most size-sensitive is `{most_size_sensitive}` "
            f"({size_gap[most_size_sensitive]:+.2f}, finds large instances much more reliably "
            "than small ones); "
            f"least is `{least_size_sensitive}` "
            f"({size_gap[least_size_sensitive]:+.2f}"
            + (
                ", actually finds SMALL instances more reliably"
                if size_gap[least_size_sensitive] < 0
                else ""
            )
            + ")."
        )
        lines.append("")
    return lines


def write_results_markdown(results: dict[str, Any], out_dir: Path) -> Path:
    """Emit ``results.md``: pooled, per-regime, per-box-size, scale-bucket, and latency tables."""
    methods = _present_methods(results)
    coverage = results.get("coverage", {})
    lines: list[str] = []
    lines.append("# Benchmark results")
    lines.append("")
    lines.append(
        "Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the "
        "benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is "
        "undefined) -- never read it as zero."
    )
    lines.append("")
    lines.append(
        f"- IoU threshold: **{results.get('iou_threshold')}** | "
        f"AP convention: {results.get('ap_convention')}"
    )
    lines.append(
        f"- Coverage: {coverage.get('images_labelled')} labelled / "
        f"{coverage.get('images_requested')} requested"
        + (
            f"; unlabelled: {', '.join(coverage['images_unlabelled'])}"
            if coverage.get("images_unlabelled")
            else ""
        )
    )
    lines.append(f"- Git SHA at run: `{results.get('git_sha')}`")
    lines.append("")

    lines.append("## Pooled metrics by method")
    lines.append("")
    lines.append("| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for method in methods:
        o = results["methods"][method]["overall"]
        lines.append(
            f"| `{method}` | {_fmt(o['precision'])} | {_fmt(o['recall'])} | {_fmt(o['f1'])} | "
            f"{_fmt(o['mean_ap'])} | {o['n_abstentions']} | {o['n_errors']} | "
            f"{_fmt(o['latency_ms']['p50'], 1)} |"
        )
    lines.append("")

    lines.append(
        "## Results by regime -- the real story ([EVAL-DESIGN.md](../EVAL-DESIGN.md) rationale)"
    )
    lines.append("")
    lines.append(
        "Pooling across regimes averages a method's best and worst cases together (the table "
        "above is a **summary, not a verdict**); per-regime is the primary result. "
        "`EASY` = chipset (identical, fixed-scale, low-texture -- the NCC-favourable baseline), "
        "`TEXTURED` = textured-plain (fixed pose, real keypoints), "
        "`VARIED` = textured-varied (scale 0.6-1.6x, rotation +/-35 deg), "
        "`CLUTTERED` = textured-cluttered (mild variation + noisy background + distractors). "
        "Scope note: this is the **synthetic** side only (chipset/textured); see "
        "[real-objects-findings.md](../reports/real-objects-findings.md) for the real-photo "
        "comparison."
    )
    lines.append("")
    regime_best: dict[str, tuple[str, float]] = {}
    for regime in _REGIMES:
        lines.append(f"### {regime}")
        lines.append("")
        lines.append("| method | precision | recall | F1 | AP | n img |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        best_method, best_f1 = None, -1.0
        for method in methods:
            rows = _rows_by_regime(results, method).get(regime, [])
            p = _pooled_rate(rows, "precision")
            r = _pooled_rate(rows, "recall")
            f1 = _pooled_rate(rows, "f1")
            ap = _pooled_rate(rows, "ap")
            lines.append(
                f"| `{method}` | {_fmt(p)} | {_fmt(r)} | {_fmt(f1)} | {_fmt(ap)} | {len(rows)} |"
            )
            if f1 is not None and f1 > best_f1:
                best_method, best_f1 = method, f1
        lines.append("")
        if best_method is not None:
            regime_best[regime] = (best_method, best_f1)

    lines.append("## Recall by scale bucket (the NCC-vs-sparse-geo crossover)")
    lines.append("")
    lines.append("| method | fixed-scale recall | varied-scale recall |")
    lines.append("| --- | --- | --- |")
    for method in methods:
        buckets = results["methods"][method]["slices"]["by_scale_bucket"]
        fixed = buckets.get("fixed", {}).get("recall")
        varied = buckets.get("varied", {}).get("recall")
        lines.append(f"| `{method}` | {_fmt(fixed)} | {_fmt(varied)} |")
    lines.append("")

    lines.append(
        "## Recall by ground-truth box size (small/medium/large, as a fraction of the image)"
    )
    lines.append("")
    lines.append(
        "Pooled over every swept image (chipset + textured + real-objects + the configured "
        'synthetic scenes), not split by regime -- "does this method find small instances?" '
        "Cuts: small < 0.4% of image area, medium < 1.6%, else large (same cuts the floor-plan "
        "research path uses; validated as a non-degenerate three-way split on this set too -- "
        "see the module docstring)."
    )
    lines.append("")
    size_gap: dict[str, float] = {}
    lines.append("| method | small recall (n) | medium recall (n) | large recall (n) |")
    lines.append("| --- | --- | --- | --- |")
    for method in methods:
        by_size = results["methods"][method]["slices"]["by_symbol_size"]
        cells = []
        small_r, large_r = None, None
        for bucket in ("small", "medium", "large"):
            b = by_size.get(bucket, {})
            recall, n_gt = b.get("recall"), b.get("n_gt", 0)
            cells.append(f"{_fmt(recall)} ({n_gt})")
            if bucket == "small":
                small_r = recall
            if bucket == "large":
                large_r = recall
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
        if small_r is not None and large_r is not None:
            size_gap[method] = large_r - small_r
    lines.append("")

    lines.append("## p50 latency by canvas size (ms)")
    lines.append("")
    all_canvases: set[str] = set()
    for method in methods:
        all_canvases.update(results["methods"][method]["slices"]["by_canvas_size"])
    canvases = sorted(all_canvases, key=_canvas_area)
    header = "| method | " + " | ".join(canvases) + " |"
    lines.append(header)
    lines.append("| --- |" + " --- |" * len(canvases))
    for method in methods:
        by_canvas = results["methods"][method]["slices"]["by_canvas_size"]
        cells = [
            _fmt(by_canvas.get(canvas, {}).get("latency_ms", {}).get("p50"), 1)
            for canvas in canvases
        ]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.extend(_insight_section(methods, regime_best, size_gap))

    out = out_dir / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exactly one trailing newline so the committed file matches the end-of-file-fixer hook and
    # a re-render never churns the repo (the whole point of committing the tables).
    out.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    logger.info("charts: wrote {}", out)
    return out


def render_all(
    results_path: str | Path = _DEFAULT_RESULTS,
    out_dir: str | Path = _DEFAULT_OUT_DIR,
    ratings: dict[str, tuple[int, int]] | None = None,
) -> list[Path]:
    """Render every chart and the tables from ``results.json``; return the written paths.

    Args:
        results_path: Path to the benchmark results JSON.
        out_dir: Directory the charts and ``results.md`` are written into.
        ratings: Optional ``method -> (n_up, n_rated)`` for the thumbs chart. ``None`` renders
            the honest empty-state panel.

    Returns:
        The written file paths, in render order.
    """
    results = load_results(results_path)
    out = Path(out_dir)
    if not out.is_absolute():
        out = repo_root() / out
    return [
        render_metrics_chart(results, out),
        render_crossover_chart(results, out),
        render_latency_chart(results, out),
        render_thumbs_chart(ratings, out),
        write_results_markdown(results, out),
    ]


def main() -> None:
    """``pixi run bench-charts`` entry point: render the committed charts and tables."""
    paths = render_all()
    logger.info("charts: rendered {} artifact(s)", len(paths))


if __name__ == "__main__":
    main()
