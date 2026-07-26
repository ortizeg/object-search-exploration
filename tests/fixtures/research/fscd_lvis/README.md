# FSCD-LVIS (unseen) offline test fixture (synthetic — NOT redistributed FSCD-LVIS data)

Tiny hand-made stand-ins in FSCD-LVIS's *native* COCO-style shape so the converter and the seeded
val-carve run with **no network and no licence gate**.

```
images/<image_id>.png     # tiny 160x100 RGB scenes
annotations.json          # image_id -> { exemplar_category, box_examples_coordinates (3 polys),
                          #               annotations: [ {box: [x1,y1,x2,y2], category}, ... ] }
split.json                # { "train": [...], "test": [...] }  (unseen protocol — NO official val)
```

- FSCD-LVIS scenes are **multi-class** (the distractor-rejection stress). Each fixture image carries
  four boxes of the `exemplar_category` (the instances a correct search must find) **and** two boxes
  of a different category (the distractors). The converter emits **only** the exemplar-category
  boxes as ground truth; the distractor boxes are intentionally excluded, so a method that returns
  them is scored as a false positive (`object_search.eval.converters.fscd_lvis`).
- The three `box_examples_coordinates` polygons are the native exemplar boxes → `exemplar_indices`
  (length 3, first is `exemplar_index`).
- The **unseen** protocol has **no official val**: a seeded val slice is carved from `train` by
  `object_search.eval.splits.carve_val`; `test` is never touched (D-03/D-04).

**Provenance / licence:** original synthetic content authored for offline testing; not derived from
and does not redistribute any part of the real FSCD-LVIS dataset. Real data is fetched by a
human — see the phase SUMMARY.
