# CARPK offline test fixture (synthetic — NOT redistributed CARPK data)

These three tiny images and their annotations are **hand-made synthetic stand-ins** authored
for this repository. They are laid out in CARPK's *native* on-disk format so the converter and
the whole research-dataset tracer can be exercised with **no network and no licence gate**:

```
Images/<image_id>.png          # tiny 160x100 RGB scenes with identical bright "car" blocks
Annotations/<image_id>.txt     # one box per line: `x1 y1 x2 y2 class`  (corner-inclusive)
ImageSets/test.txt             # the CARPK "test" split id list
```

- `x1 y1 x2 y2` are pixel corner coordinates in CARPK's native convention. The converter
  translates them to the repo's half-open `BBox(x=x1, y=y1, w=x2-x1, h=y2-y1)` at the boundary
  (see `object_search.eval.converters.carpk`).
- `class` is always `1` (CARPK is single-class: "car").

**Provenance / licence:** This is original synthetic content created for offline testing. It is
**not** derived from, and does not redistribute, any part of the real CARPK dataset. The real
CARPK data is licence-gated (terms-of-use, non-commercial research) and must be fetched by a
human via `pixi run fetch-datasets` into `datasets/_incoming/carpk/` — see the phase SUMMARY.
