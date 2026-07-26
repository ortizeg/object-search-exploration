"""Build the self-contained benchmark report (docs/reports/benchmark-report.html).

Reads the committed docs/benchmark/results.json (run `pixi run bench` first), groups results by
dataset regime, computes 95% confidence intervals by bootstrapping over IMAGES (instances within
one image are not independent), renders the four numbered method descriptions with links to their
docs and primary references, a dataset section, per-regime scoreboards, and side-by-side overlays
of every method in every regime. Charts are inline SVG, overlays are base64 JPEG -- no network, no
JS. Run with `pixi run report`.
"""

from __future__ import annotations

import base64
import json
import statistics as st
from pathlib import Path

import cv2
import numpy as np

from object_search.eval.labels import load_ground_truth, scene_path
from object_search.search import registry
from object_search.search.common import viz

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((REPO_ROOT / "docs/benchmark/results.json").read_text())
OUT = REPO_ROOT / "docs/reports/benchmark-report.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

ORDER = ["ncc", "sparse-geo", "dino-dense", "propose-retrieve", "mosse"]
COLOR = {
    "ncc": "#4C72B0",
    "sparse-geo": "#DD8452",
    "dino-dense": "#55A868",
    "propose-retrieve": "#C44E52",
    "mosse": "#8172B3",
}
CIRCLED = "①②③④⑤"
LABEL = {m: f"{CIRCLED[i]} {m}" for i, m in enumerate(ORDER)}

REGIMES = [
    ("EASY", "identical, axis-aligned, fixed-scale chips on white — the NCC-favourable baseline"),
    ("TEXTURED", "richly-textured emblems (≥20 SIFT keypoints), fixed scale"),
    ("VARIED", "same emblem, scale 0.6–1.6×, rotation ±35°, brightness"),
    ("CLUTTERED", "textured emblem, noisy background + distractors"),
]

METHODS = [
    (
        "1",
        "NCC",
        "ncc",
        "Template matching. Correlate the exemplar crop over the whole scene "
        "(<code>cv2.matchTemplate</code>, normalized cross-correlation) across an image pyramid; peaks "
        "are detections. Zero weights, milliseconds. Best on near-identical, fixed-scale repeats.",
        "OpenCV template matching",
        "https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html",
    ),
    (
        "2",
        "sparse-geo",
        "sparse-geo",
        "SIFT / SuperPoint keypoints matched many-to-many into the scene (Lowe's ratio test "
        "<b>disabled</b>, since it suppresses repeats), then generalized Hough voting in pose space "
        "and per-peak RANSAC recover one geometric model per instance. Needs texture; rotation- and "
        "scale-invariant.",
        "Lowe, IJCV 2004",
        "https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf",
    ),
    (
        "3",
        "dino-dense",
        "dino-dense",
        "DINOv2 dense patch features for scene and crop; score each scene location by its "
        "best-matching-part cosine to the crop tokens (max-token), threshold at a contrast-"
        "calibrated cut, connected components with exemplar-relative area bounds. Handles "
        "pose/lighting variation; coarse (stride-14) on tiny objects.",
        "DINOv2, 2023",
        "https://arxiv.org/abs/2304.07193",
    ),
    (
        "4",
        "propose-retrieve",
        "propose-retrieve",
        "FastSAM proposes class-agnostic regions; each is embedded with the same DINOv2 and matched "
        "by cosine nearest-neighbour to the exemplar embedding, then NMS. Boxes hug object "
        "boundaries; can over-segment dense grids.",
        "FastSAM, 2023",
        "https://arxiv.org/abs/2306.12156",
    ),
    (
        "5",
        "MOSSE",
        "mosse",
        "The FFT cousin of NCC. A small bank of MOSSE/ASEF correlation filters is synthesized from "
        "the warped exemplar (the rotation bank folded into the closed-form solve) and matched by "
        "<code>FFT</code> cross-correlation — a handful of transforms instead of NCC's spatial pass "
        "per rotation. Near-NCC F1 on near-identical repeats at ~6× lower latency; weaker in clutter.",
        "MOSSE, Bolme et al. CVPR 2010",
        "https://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf",
    ),
]

OVERLAY_SCENES = {
    "EASY": "chipset-04",
    "TEXTURED": "textured-plain-05",
    "VARIED": "textured-varied-05",
    "CLUTTERED": "textured-cluttered-05",
}


def regime_of(image_id: str) -> str | None:
    if image_id.startswith("chipset-"):
        return "EASY"
    if image_id.startswith("textured-plain"):
        return "TEXTURED"
    if image_id.startswith("textured-varied"):
        return "VARIED"
    if image_id.startswith("textured-cluttered"):
        return "CLUTTERED"
    return None


def rows_by_regime(method):
    out = {}
    for r in RESULTS["methods"][method]["per_image"]:
        reg = regime_of(r["image_id"])
        if reg is not None:
            out.setdefault(reg, []).append(r)
    return out


def pooled(rows, metric):
    if metric == "precision":
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        return tp / (tp + fp) if (tp + fp) else None
    if metric == "recall":
        tp = sum(r["tp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        return tp / (tp + fn) if (tp + fn) else None
    if metric == "f1":
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        denom = 2 * tp + fp + fn
        return (2 * tp) / denom if denom else None
    aps = [r["ap"] for r in rows if r.get("ap") is not None]
    return st.mean(aps) if aps else None


def all_regime_rows(method):
    """Every scored row across the four labelled regimes (the union of the shown strata)."""
    rr = rows_by_regime(method)
    return [r for reg in ("EASY", "TEXTURED", "VARIED", "CLUTTERED") for r in rr.get(reg, [])]


def bootstrap_ci(rows, metric, n=2000):
    if not rows:
        return None
    rng = np.random.default_rng(12345)
    idx = np.arange(len(rows))
    vals = []
    for _ in range(n):
        sample = [rows[i] for i in rng.choice(idx, size=len(rows), replace=True)]
        v = pooled(sample, metric)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def b64(img, max_w=360):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    _ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 84])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_overlays():
    overlays = {}
    for reg, iid in OVERLAY_SCENES.items():
        gt = load_ground_truth(iid)
        sp = scene_path(iid)
        if gt is None or sp is None:
            continue
        scene = cv2.imread(str(sp))
        exemplar = gt.exemplar
        ref = viz.draw_matches(scene.copy(), (), exemplar=exemplar)
        per = {}
        for m in ORDER:
            spec = registry.get_method(m)
            try:
                res = spec.fn(scene, exemplar, spec.config_model())
                rendered = viz.draw_matches(scene.copy(), res.matches, exemplar=exemplar)
                outcome = res.outcome.value if hasattr(res.outcome, "value") else str(res.outcome)
                per[m] = {"img": b64(rendered), "outcome": outcome, "n": len(res.matches)}
            except Exception as exc:  # report path: a method error must never abort the report
                per[m] = {"img": None, "outcome": "error", "n": 0, "note": str(exc)[:120]}
        overlays[reg] = {
            "ref": b64(ref),
            "n_gt": len(gt.boxes),
            "size": f"{scene.shape[1]}x{scene.shape[0]}",
            "methods": per,
            "iid": iid,
        }
    return overlays


def pct(x):
    return "<span class='na'>n/a</span>" if x is None else f"{x * 100:.0f}%"


def ci_span(c):
    return f" <span class='ci'>[{c[0] * 100:.0f}–{c[1] * 100:.0f}]</span>" if c else ""


def regime_table(reg):
    head = (
        "<tr><th>method</th><th>precision (95% CI)</th><th>recall (95% CI)</th>"
        "<th>AP</th><th>n img</th></tr>"
    )
    rows_html = []
    for m in ORDER:
        rr = rows_by_regime(m).get(reg, [])
        p, r, ap = pooled(rr, "precision"), pooled(rr, "recall"), pooled(rr, "ap")
        pci, rci = bootstrap_ci(rr, "precision"), bootstrap_ci(rr, "recall")
        rows_html.append(
            f"<tr><td class='mn'><span class='dot' style='background:{COLOR[m]}'></span>"
            f"{LABEL[m]}</td><td>{pct(p)}{ci_span(pci)}</td><td>{pct(r)}{ci_span(rci)}</td>"
            f"<td>{pct(ap)}</td><td>{len(rr)}</td></tr>"
        )
    return f"<table class='metrics'><tbody>{head}{''.join(rows_html)}</tbody></table>"


def overall_table():
    """Pooled precision/recall/F1/AP over all four regimes — a summary, not a verdict."""
    head = (
        "<tr><th>method</th><th>precision (95% CI)</th><th>recall (95% CI)</th>"
        "<th>F1</th><th>AP</th><th>n img</th></tr>"
    )
    rows_html = []
    for m in ORDER:
        rr = all_regime_rows(m)
        p, r = pooled(rr, "precision"), pooled(rr, "recall")
        f1, ap = pooled(rr, "f1"), pooled(rr, "ap")
        pci, rci = bootstrap_ci(rr, "precision"), bootstrap_ci(rr, "recall")
        rows_html.append(
            f"<tr><td class='mn'><span class='dot' style='background:{COLOR[m]}'></span>"
            f"{LABEL[m]}</td><td>{pct(p)}{ci_span(pci)}</td><td>{pct(r)}{ci_span(rci)}</td>"
            f"<td>{pct(f1)}</td><td>{pct(ap)}</td><td>{len(rr)}</td></tr>"
        )
    return f"<table class='metrics'><tbody>{head}{''.join(rows_html)}</tbody></table>"


def bar_row(reg, metric):
    x0, bw, rh = 118, 210, 26
    svg = []
    for i, m in enumerate(ORDER):
        rr = rows_by_regime(m).get(reg, [])
        v = pooled(rr, metric)
        ci = bootstrap_ci(rr, metric)
        y = 6 + i * rh
        w = int(bw * (v or 0))
        svg.append(
            f"<text x='{x0 - 6}' y='{y + 13}' text-anchor='end' class='bl'>{LABEL[m]}</text>"
        )
        svg.append(f"<rect x='{x0}' y='{y + 3}' width='{bw}' height='14' class='tr'/>")
        svg.append(f"<rect x='{x0}' y='{y + 3}' width='{w}' height='14' rx='2' fill='{COLOR[m]}'/>")
        if ci:
            lo, hi = int(bw * ci[0]), int(bw * ci[1])
            svg.append(
                f"<line x1='{x0 + lo}' y1='{y + 10}' x2='{x0 + hi}' y2='{y + 10}' class='whisk'/>"
            )
        svg.append(f"<text x='{x0 + bw + 8}' y='{y + 14}' class='bv'>{pct(v)}</text>")
    h = 6 + rh * len(ORDER)
    return f"<svg viewBox='0 0 {x0 + bw + 56} {h}' class='chart'>{''.join(svg)}</svg>"


def methods_section():
    cards = []
    for num, name, slug, desc, ref_label, ref_url in METHODS:
        color = COLOR[slug]
        cards.append(
            f"<div class='mcard'><div class='mhead'>"
            f"<span class='mnum' style='background:{color}'>{num}</span><b>{name}</b></div>"
            f"<p>{desc}</p><div class='mlinks'>"
            f"<a href='../methods/{slug}.md'>method doc &rarr;</a>"
            f"<a href='{ref_url}'>{ref_label} &rarr;</a></div></div>"
        )
    return f"<div class='mgrid'>{''.join(cards)}</div>"


def datasets_section():
    cards = []
    for reg, desc in REGIMES:
        color = {
            "EASY": "#4C72B0",
            "TEXTURED": "#DD8452",
            "VARIED": "#55A868",
            "CLUTTERED": "#C44E52",
        }[reg]
        n = len(rows_by_regime("ncc").get(reg, []))
        cards.append(
            f"<div class='dcard' style='border-left-color:{color}'>"
            f"<b>{reg}</b> <span class='muted'>· {n} images</span><p>{desc}</p></div>"
        )
    return (
        "<p class='sub'>All sets are synthetic with exact ground truth by construction (no "
        "hand-labeling, no licensing); every instance is pasted at a known, non-overlapping "
        "rectangle. Full descriptions, provenance, and regeneration commands are in "
        "<a href='../DATASETS.md'>DATASETS.md</a>.</p>"
        f"<div class='dgrid'>{''.join(cards)}</div>"
    )


def overlay_section(overlays):
    blocks = []
    for reg, _iid in OVERLAY_SCENES.items():
        ov = overlays.get(reg)
        if not ov:
            continue
        tiles = [
            f"<figure class='tile ref'><img src='data:image/jpeg;base64,{ov['ref']}'/>"
            f"<figcaption><b>query</b><br><span class='muted'>{ov['n_gt']} instances · "
            f"{ov['size']}</span></figcaption></figure>"
        ]
        for m in ORDER:
            d = ov["methods"][m]
            img = (
                f"<img src='data:image/jpeg;base64,{d['img']}'/>"
                if d.get("img")
                else "<div class='noimg'>error</div>"
            )
            cap = (
                f"<b style='color:{COLOR[m]}'>{LABEL[m]}</b><br>"
                f"<span class='muted'>{d['outcome']}, {d['n']} boxes</span>"
            )
            tiles.append(f"<figure class='tile'>{img}<figcaption>{cap}</figcaption></figure>")
        desc = next(d for r, d in REGIMES if r == reg)
        blocks.append(
            f"<h3 class='scene'>{reg} — {desc} <span class='muted'>({ov['iid']})</span></h3>"
            f"<div class='grid5'>{''.join(tiles)}</div>"
        )
    return "".join(blocks)


STYLE = """
:root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaed;--muted:#9aa1ad;--line:#2a2f3a;--accent:#7aa2ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:40px 26px 80px}
h1{font-size:30px;margin:0 0 4px}
.lede{color:var(--muted);max-width:820px;margin:0 0 8px;font-size:16px}
.meta{color:var(--muted);font-size:13px;margin-bottom:28px}.meta code{color:var(--accent)}
h2{font-size:20px;margin:44px 0 6px;padding-bottom:8px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:14px 0 2px}
.sub{color:var(--muted);font-size:12.5px;margin:2px 0 12px}
.sub a,.mlinks a,.callout a{color:var(--accent);text-decoration:none}
.sub a:hover,.mlinks a:hover{text-decoration:underline}
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.mcard{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.mhead{font-size:15px;margin-bottom:4px}
.mnum{display:inline-flex;width:22px;height:22px;border-radius:50%;color:#fff;align-items:center;justify-content:center;font-size:13px;font-weight:700;margin-right:9px;vertical-align:middle}
.mcard p{margin:0 0 8px;color:var(--muted);font-size:13px}.mcard code{color:var(--ink);font-size:12px}
.mlinks{display:flex;gap:16px;font-size:12.5px}
.dgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:8px}
.dcard{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line);border-radius:9px;padding:11px 14px}
.dcard p{margin:4px 0 0;color:var(--muted);font-size:12.5px}
.regime{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:14px}
.regime.easy{border-left:3px solid #4C72B0}.regime.textured{border-left:3px solid #DD8452}
.regime.varied{border-left:3px solid #55A868}.regime.cluttered{border-left:3px solid #C44E52}
.rtitle{font-size:16px;font-weight:600;margin:0 0 2px}
table.metrics{width:100%;border-collapse:collapse;margin-top:8px}
.metrics th{text-align:right;font-size:11.5px;color:var(--muted);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}
.metrics th:first-child{text-align:left}
.metrics td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;font-size:13px}
.metrics td:first-child{text-align:left}.metrics tr:last-child td{border-bottom:none}
.mn{font-weight:600}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
.ci{color:var(--muted);font-size:11px}.na{color:#c99a3a;font-style:italic}
.chart{width:100%;height:auto;margin-top:6px}.tr{fill:#20242e}
.bl{fill:var(--ink);font-size:11px}.bv{fill:var(--ink);font-size:11px;font-weight:600}
.whisk{stroke:#e8eaed;stroke-width:1.5;opacity:.7}
.scene{margin:22px 0 8px;font-size:14px}
.grid5{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}
.tile{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.tile.ref{outline:2px solid var(--accent)}.tile img{width:100%;display:block;background:#000}
.tile figcaption{padding:6px 8px;font-size:11px;line-height:1.3}.muted{color:var(--muted)}
.noimg{padding:30px 8px;text-align:center;color:#c99a3a;font-size:11px}
.callout{background:#141821;border:1px solid var(--line);border-radius:10px;padding:14px 18px;color:var(--muted);font-size:13.5px;margin-top:14px}.callout b{color:var(--ink)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.mgrid,.two,.dgrid{grid-template-columns:1fr}.grid5{grid-template-columns:repeat(2,1fr)}}
"""


def build():
    overlays = build_overlays()
    regime_blocks = "".join(
        f"<div class='regime {reg.lower()}'><div class='rtitle'>{reg}</div>"
        f"<div class='sub'>{desc}</div>"
        f"<div class='two'><div><div class='sub'>precision</div>{bar_row(reg, 'precision')}</div>"
        f"<div><div class='sub'>recall</div>{bar_row(reg, 'recall')}</div></div>"
        f"{regime_table(reg)}</div>"
        for reg, desc in REGIMES
    )
    doc = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Object Search — benchmark report</title><style>{STYLE}</style></head><body><div class='wrap'>

<h1>Object Search — four-method benchmark</h1>
<p class='lede'>One hand-drawn box, four interchangeable methods, scored across <b>stratified
datasets</b> — the original easy chips plus textured sets that give the keypoint and deep-feature
methods a fair fight. Confidence intervals are <b>bootstrap over images</b> (instances in one image
are not independent). Abstentions render as <span class='na'>n/a</span>, never zero.</p>
<p class='meta'>IoU 0.5 · AP all-point interpolation · CPU (deterministic) ·
git <code>{RESULTS["git_sha"][:8]}</code> · bootstrap 2000× over images</p>

<h2>The four methods</h2>
<p class='sub'>Labelled 1–4 as implemented. The source-research numbering is 1, 2, 3, 5 — research
Methods 4 (exemplar-conditioned detectors) and 6 (personalized segmentation) were deferred. Each
card links its method doc (algorithm, pseudocode, config reference) and a primary reference.</p>
{methods_section()}

<h2>The datasets</h2>
{datasets_section()}

<h2>Overall <span class='muted' style='font-size:13px'>— pooled across all four regimes</span></h2>
<p class='sub'>Precision, recall, F1, and AP pooled over every image in the four regimes, each rate
with a 95% bootstrap CI. <b>Read this as a summary, not a verdict:</b> pooling averages a method's
best and worst regimes together, so a method that wins decisively in one and abstains in another
lands in a misleading middle. The per-regime tables below are the real result.</p>
{overall_table()}

<h2>Results by regime <span class='muted' style='font-size:13px'>— the real story</span></h2>
<p class='sub'>Per-regime is primary. Precision and recall each carry a 95% bootstrap CI.</p>
{regime_blocks}

<div class='callout'><b>The crossover, made concrete.</b> On <b>EASY</b> chips ① NCC is
near-perfect and ② sparse-geo abstains (too few keypoints). On <b>TEXTURED</b>, ② sparse-geo comes
alive — the emblems carry real keypoints — and the deep-feature methods hold up. On <b>VARIED</b>
(scale + rotation), ① NCC's recall collapses while ② sparse-geo, ③ dino-dense, and ④
propose-retrieve stay competitive. No single method wins everywhere; that is the entire reason four
exist.</div>

<h2>Overlays — judge them yourself</h2>
<p class='sub'>Same query box, all four methods, one representative image per regime. The
<span style='color:var(--accent)'>blue-outlined</span> tile is the exemplar.</p>
{overlay_section(overlays)}

<div class='callout' style='margin-top:26px'><b>Method, honestly.</b> All sets are synthetic with
exact ground truth. Instances within an image are correlated, so CIs are bootstrapped over
<b>images</b>, not instances. Configs are method defaults, not tuned to these images. The full
rigorous protocol (5 regimes × ~40 images, mAP@[.5:.95], paired Bradley-Terry) is specified in
<a href='../EVAL-DESIGN.md'>EVAL-DESIGN.md</a>; this is the practical-size cut with honest
per-regime intervals.</div>

<p class='meta' style='margin-top:30px'>Regenerate: <code>pixi run bench</code> (numbers) then
<code>pixi run report</code> (this page). Datasets: <code>pixi run chipset</code> /
<code>pixi run textured</code>.</p>
</div></body></html>"""
    OUT.write_text(doc)
    print(f"wrote {OUT} ({len(doc) // 1024} KB)")


if __name__ == "__main__":
    build()
