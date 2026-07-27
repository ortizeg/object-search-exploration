# Object Search Exploration

Draw a box around one object in an image; find every other instance of that same object **in
that same image**. Five independent search methods sit behind one interface, selectable
*before* the box is drawn, so the same query can be run through different algorithms and
compared side by side. A rating layer records how well each method did on each query, and a
statistics layer turns those ratings — plus objective metrics on ground-truthed images — into
a per-method scoreboard.

This is an **exploration harness, not a product**. Its value is that each method is readable,
editable, and measurable by an ML practitioner, and that adding a sixth method or a whole new
exploration is a small, obvious diff.

!!! info "Status — Milestone 2 complete"
    All five search methods (including the OWLv2 one-shot detector), the canvas UI, the FastAPI
    backend, the SQLite rating/stats layer, and the evaluation harness are implemented and
    tested. Milestone 2 adds a second *exploration* (`marker-conditioned`): draw a marker, find
    every instance, and resolve the object each one points at. Where a method is weak, the
    [benchmark numbers](benchmark/results.md) say so — see [Limitations](LIMITATIONS.md).

## New here? Start with the walkthrough

The **[Interface walkthrough](walkthrough.md)** is the fastest way in: it shows the actual UI —
picking a method, drawing an exemplar box, and reading the results and diagnostics overlay —
with screenshots of each step.

## Quickstart

Pixi is the only supported environment manager. Do not use `pip`, `venv`, `uv`, or `conda`
directly — a single reproducible lockfile is the whole point.

```bash
pixi install            # solve + install from the committed pixi.lock
pixi run fetch-models   # download ONNX weights into models/ (dino-dense, propose-retrieve, owlv2-oneshot)
pixi run serve          # FastAPI + static canvas UI on http://localhost:8000
```

Open the UI, pick a method, draw a box around one instance, and the matches for that method are
drawn back on the image with a diagnostics overlay. `ncc` and classical `sparse-geo` need no
weights; `dino-dense`, `propose-retrieve`, and `owlv2-oneshot` need `fetch-models` first.

To read these docs locally as a site:

```bash
pixi run docs           # themed, searchable site with live-reload at http://127.0.0.1:8001
```

## The five methods

| # | Method | Idea | Needs weights |
| --- | --- | --- | --- |
| 1 | [`ncc`](methods/ncc.md) | Zero-model baseline: `cv2.matchTemplate` with `TM_CCOEFF_NORMED` over the full scene, then peak extraction and NMS. Pyramid scale search, optional rotation bank. | No |
| 2 | [`sparse-geo`](methods/sparse-geo.md) | Keypoints on the crop matched into the scene, then **many** geometric models recovered rather than one (Hough voting / sequential RANSAC). Classical and SuperPoint backends. | No (classical) / yes (SuperPoint) |
| 3 | [`dino-dense`](methods/dino-dense.md) | Dense deep-feature similarity: DINOv2 patch tokens for scene and exemplar, cosine-similarity map, calibrate, peak-pick. | Yes (DINOv2) |
| 4 | [`owlv2-oneshot`](methods/owlv2-oneshot.md) | Image-conditioned one-shot detection: the exemplar crop is encoded as a query image and read out through OWLv2's own trained detection boxes. | Yes (OWLv2) |
| 5 | [`propose-retrieve`](methods/propose-retrieve.md) | Propose → embed → retrieve: FastSAM class-agnostic proposals, DINOv2 region embeddings, nearest-neighbour rank against the exemplar. | Yes (FastSAM + DINOv2) |

## Where to go next

- **[Interface walkthrough](walkthrough.md)** — the UI, step by step, with screenshots.
- **[Sample runs](samples/ncc/index.md)** — pre-rendered results for every method, reviewable
  without running anything.
- **[Benchmark results](benchmark/results.md)** — the per-method scoreboard on the demo set.
- **[Evaluation design](EVAL-DESIGN.md)** and **[Limitations](LIMITATIONS.md)** — how the
  numbers are computed and where each method breaks.
- **[Robustness backlog](ROBUSTNESS-BACKLOG.md)** — known weaknesses and deferred work.
