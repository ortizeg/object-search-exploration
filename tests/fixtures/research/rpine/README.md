# RPINE offline test fixture (synthetic — NOT redistributed RPINE data)

Tiny hand-made stand-ins in RPINE's *native* on-disk shape so the converter and the seeded
val-carve run with **no network and no licence gate**.

```
images/<image_id>.png        # tiny 160x100 RGB scenes
annotations/<image_id>.txt   # one `x1 y1 x2 y2` box per line (x2/y2 exclusive) — ALL repeats
split.json                   # { "train": [...], "test": [...] }  (RPINE has NO official val)
```

- Every line is one annotated repeat; RPINE has no class column (it annotates *repetition*, not
  categories). The converter translates each box to the repo's half-open
  `BBox(x=x1, y=y1, w=x2-x1, h=y2-y1)` at the boundary
  (`object_search.eval.converters.rpine`).
- RPINE ships no native exemplar box, so the converter **samples the exemplar indices from the
  ground-truth boxes, seeded** (`np.random.default_rng` — D-11), byte-stably.
- RPINE has **no official val split**: a seeded val slice is carved from `train` by
  `object_search.eval.splits.carve_val`; the `test` list is never touched (D-03/D-04).

**Provenance / licence:** original synthetic content authored for offline testing; not derived from
and does not redistribute any part of the real RPINE dataset. Real data is fetched by a human — see
the phase SUMMARY.
