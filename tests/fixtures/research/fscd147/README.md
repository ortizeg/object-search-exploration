# FSCD-147 offline test fixture (synthetic — NOT redistributed FSC-147/FSCD-147 data)

Tiny hand-made stand-ins laid out in FSCD-147's *native* on-disk shape so the converter, the
de-duplication, and the split-manifest builder run with **no network and no licence gate**.

```
images/<image_id>.png     # tiny 160x100 RGB scenes
annotations.json          # image_id -> { box_examples_coordinates (3 exemplar polygons),
                          #               points (dots, ignored — D-06), boxes (per-object xyxy) }
split.json                # { "train": [...], "val": [...], "test": [...] }  (FSC-147 native triple)
```

- `boxes` are per-object `[x1, y1, x2, y2]` in FSC-147's **corner** convention (x2/y2 exclusive as
  drawn here). The converter translates each to the repo's half-open
  `BBox(x=x1, y=y1, w=x2-x1, h=y2-y1)` at the boundary (see
  `object_search.eval.converters.fscd147`). Only **val/test** images get sidecars (their boxes are
  human); train boxes are pseudo-labels and are skipped for scoring (D-06).
- `box_examples_coordinates` holds the **3 native exemplar boxes** per image as 4-corner polygons.
  The converter maps each to the index of its matching object box, giving `exemplar_indices`
  (length 3, the first is `exemplar_index`).

## Planted contamination (exercises the de-duplication — D-07)

- **Pixel-identical duplicate pair:** `fscd147-fixture-dup-a` and `fscd147-fixture-dup-b` are
  **byte-identical** PNGs under different ids (both in `val`). `dedup_fscd147` keeps the canonical
  (lexicographically first) copy `dup-a` and drops `dup-b`.
- **Train↔test leak:** `fscd147-fixture-leak` appears in **both** `train` and `test`. `dedup_fscd147`
  drops it from every split (its defining property — an id in more than one split — is exactly the
  11 documented FSC-147 train↔test leaks, arXiv:2409.15953).

`tests/test_research_dedup.py` asserts both planted ids are absent after de-duplication and that the
leaked id is gone from the test split.

**Provenance / licence:** original synthetic content authored for offline testing. It is **not**
derived from, and does not redistribute, any part of FSC-147/FSCD-147. The real data is
licence-gated (VinAI / Counting-DETR terms) and must be fetched by a human — see the phase SUMMARY.
