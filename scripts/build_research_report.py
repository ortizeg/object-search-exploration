"""Build the committed research-dataset report table (docs/reports/research-report.html).

Mirrors :mod:`scripts.build_report` but for the **research** sweep: it reads the gitignored
``docs/benchmark/research-results.json`` (run ``pixi run bench-research`` first, or regenerate via
``run_research_sweep`` once the licence-gated archives are fetched) and renders one row per
``method x dataset x {1,3 exemplars} x {val,test}`` cell, carrying the literature's own columns --
Precision, Recall, F1, COCO AP@[.5:.95:.05], AP50, AP75, MAE, RMSE, NAE.

Two conventions are carried through from the numeric layer, unchanged:

* **Abstention renders as ``n/a``, never ``0``.** A pooled cell where a method returned nothing has
  ``precision = None``; showing ``0`` would libel an honest abstention as a wrong answer (EVAL-17).
* **The 3-exemplar numbers are "k-shot late fusion".** They come from running the single-exemplar
  method once per exemplar and unioning + NMS-deduping the detections (the eval-layer runner
  ``run_multi_exemplar``); the caption names this so the reader knows how the 3-exemplar column was
  produced and that ``SearchFn`` and the four method files were never touched.

The numbers depend on fetched data + weights, so the JSON is regenerable and gitignored; **this
rendered page is the committed deliverable** (mirror of EVAL-06). Run with ``pixi run report-research``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "docs" / "benchmark" / "research-results.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "reports" / "research-report.html"

# The literature metric columns, in report order. The tuple is the single source for both the
# header row and the per-cell lookups, so the two cannot drift.
_RATE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("P", "precision"),
    ("R", "recall"),
    ("F1", "f1"),
    ("AP", "ap"),
    ("AP50", "ap50"),
    ("AP75", "ap75"),
)
_COUNT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("MAE", "mae"),
    ("RMSE", "rmse"),
    ("NAE", "nae"),
)

_MISSING_RESULTS_MESSAGE = (
    "No research-results.json found at {path}.\n"
    "The research numbers depend on the licence-gated dataset archives (and, for the learned "
    "methods, fetched ONNX weights), so this file is gitignored and regenerable, not committed.\n"
    "To produce it: fetch the data (`pixi run fetch-datasets --list` prints where each archive "
    "drops), then run the research sweep via object_search.eval.benchmark.run_research_sweep over "
    "the fetched tree. The offline fixtures reproduce the same table shape without any download."
)


def _pct(value: float | None) -> str:
    """Render a rate in ``[0,1]`` as a percentage, or ``n/a`` for an abstention (never ``0``)."""
    return "<span class='na'>n/a</span>" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None) -> str:
    """Render a count error (MAE/RMSE/NAE) to two decimals, or ``n/a`` when not assessed."""
    return "<span class='na'>n/a</span>" if value is None else f"{value:.2f}"


def _cell_sort_key(cell: dict[str, Any]) -> tuple[str, str, int, str]:
    """Stable row order: by method, then dataset, then exemplar count, then split."""
    return (
        str(cell.get("method", "")),
        str(cell.get("dataset", "")),
        int(cell.get("exemplar_count", 0)),
        str(cell.get("split", "")),
    )


def _row_html(cell: dict[str, Any]) -> str:
    """One table row for a single sweep cell, with all nine literature columns."""
    overall = cell.get("overall", {})
    cells = [
        f"<td class='mn'>{cell.get('method', '?')}</td>",
        f"<td>{cell.get('dataset', '?')}</td>",
        f"<td>{cell.get('exemplar_count', '?')}</td>",
        f"<td>{cell.get('split', '?')}</td>",
    ]
    cells += [f"<td>{_pct(overall.get(key))}</td>" for _label, key in _RATE_COLUMNS]
    cells += [f"<td>{_num(overall.get(key))}</td>" for _label, key in _COUNT_COLUMNS]
    return f"<tr>{''.join(cells)}</tr>"


def _table_html(results: dict[str, Any]) -> str:
    """The full results table: a header row of the nine literature columns, then one row per cell."""
    headers = ["method", "dataset", "exemplars", "split"]
    headers += [label for label, _key in _RATE_COLUMNS]
    headers += [label for label, _key in _COUNT_COLUMNS]
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    cells = sorted(results.get("cells", []), key=_cell_sort_key)
    body = "".join(_row_html(cell) for cell in cells)
    return f"<table class='metrics'><thead>{head}</thead><tbody>{body}</tbody></table>"


_STYLE = """
:root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaed;--muted:#9aa1ad;--line:#2a2f3a;--accent:#7aa2ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 26px 80px}
h1{font-size:28px;margin:0 0 6px}
.lede{color:var(--muted);max-width:860px;margin:0 0 8px;font-size:15px}
.meta{color:var(--muted);font-size:13px;margin-bottom:24px}.meta code{color:var(--accent)}
table.metrics{width:100%;border-collapse:collapse;margin-top:10px}
.metrics th{text-align:right;font-size:11.5px;color:var(--muted);font-weight:600;padding:8px 9px;border-bottom:1px solid var(--line)}
.metrics th:first-child,.metrics th:nth-child(2),.metrics th:nth-child(4){text-align:left}
.metrics td{text-align:right;padding:7px 9px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;font-size:13px}
.metrics td:first-child,.metrics td:nth-child(2),.metrics td:nth-child(4){text-align:left}
.mn{font-weight:600}.na{color:#c99a3a;font-style:italic}
.callout{background:#141821;border:1px solid var(--line);border-radius:10px;padding:14px 18px;color:var(--muted);font-size:13.5px;margin-top:18px}.callout b{color:var(--ink)}
.callout a{color:var(--accent);text-decoration:none}
"""


def render(results: dict[str, Any]) -> str:
    """Render the full self-contained HTML page from a parsed research-results mapping."""
    git_sha = str(results.get("git_sha") or "unknown")[:8]
    seed = results.get("seed", "?")
    iou = results.get("iou_threshold", "?")
    n_cells = len(results.get("cells", []))
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Object Search — research-dataset report</title><style>{_STYLE}</style></head>
<body><div class='wrap'>
<h1>Object Search — research-dataset evaluation</h1>
<div class='callout' style='border-color:#c99a3a;margin:0 0 18px'><b>Offline fixture smoke-run.</b>
The real research-dataset images are <b>licence-gated and not fetched</b> in this repo, so the
numbers below are the committed <b>offline fixture</b> run (tiny synthetic stand-ins under
<code>tests/fixtures/research/</code>) — they exercise the harness, metrics, and this table end-to-end
without any download, but are <b>not</b> real-dataset results. The real report regenerates via
<code>pixi run report-research</code> once a human accepts each licence and fetches the archives
(<code>pixi run fetch-datasets --list</code>). No real-dataset numbers are claimed. See
<a href='../eval/research-datasets.md'>research-datasets.md</a> for the protocol.</div>
<p class='lede'>Every method on every external research dataset, at <b>1 exemplar</b> (the product's
one-box operating point) and <b>3 exemplars</b> (the published-benchmark convention), tuned on
<b>val</b> and reported on <b>test</b>. The 3-exemplar numbers are produced by
<b>k-shot late fusion</b> &mdash; the single-exemplar method is run once per exemplar and the
detections are unioned then NMS-deduped in the eval layer, so no method file changed. Abstentions
render as <span class='na'>n/a</span>, never zero.</p>
<p class='meta'>{n_cells} cells · IoU <code>{iou}</code> · seed <code>{seed}</code> ·
AP = COCO AP@[.5:.95:.05], AP50/AP75 single-IoU · git <code>{git_sha}</code></p>
{_table_html(results)}
<div class='callout'><b>How these were produced.</b> P/R/F1 and COCO AP/AP50/AP75 are the
detection columns; MAE/RMSE/NAE are the counting columns — the exact metrics the few-shot
counting/detection literature reports. CARPK/PUCPR+ appear on <b>test only</b> (a cross-domain
generalization probe, never tuned on). Dataset descriptions, source links, splits, and the full
protocol are in <a href='../eval/research-datasets.md'>research-datasets.md</a>.</div>
</div></body></html>"""


def build_research_report(results_path: Path, out_path: Path) -> str:
    """Read ``results_path``, render the report, write it to ``out_path``, and return the HTML.

    Args:
        results_path: The research-results JSON (from ``run_research_sweep``).
        out_path: Where the rendered HTML committed report is written.

    Returns:
        The rendered HTML string (also written to ``out_path``).

    Raises:
        FileNotFoundError: If ``results_path`` is absent, with an actionable message -- the report
            is regenerable and the numbers depend on fetched data.
    """
    if not results_path.is_file():
        raise FileNotFoundError(_MISSING_RESULTS_MESSAGE.format(path=results_path))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    html = render(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return html


def main() -> None:
    """Entry point for ``pixi run report-research``: render from the default paths.

    Exits with an actionable message (not a traceback) when the research-results file is absent, so
    a user who has not fetched the data gets told what to do rather than a stack trace.
    """
    try:
        build_research_report(DEFAULT_RESULTS, DEFAULT_OUT)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
    sys.stdout.write(f"wrote {DEFAULT_OUT}\n")


if __name__ == "__main__":
    main()
