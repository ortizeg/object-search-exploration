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


def write_results_markdown(results: dict[str, Any], out_dir: Path) -> Path:
    """Emit ``results.md``: the pooled table, the scale-bucket crossover table, and latency."""
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
