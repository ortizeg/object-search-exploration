# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 90 labelled / 90 requested
- Git SHA at run: `0d3e93903e54f3648a756b82b02e4b513de61fde`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.799 | 0.662 | 0.724 | 0.673 | 0 | 0 | 1813.6 |
| `sparse-geo` | 0.867 | 0.767 | 0.814 | 0.747 | 6 | 0 | 97.4 |
| `dino-dense` | 0.491 | 0.670 | 0.567 | 0.469 | 0 | 0 | 931.8 |
| `owlv2-oneshot` | 0.212 | 0.810 | 0.337 | 0.464 | 0 | 0 | 3981.9 |
| `propose-retrieve` | 0.882 | 0.918 | 0.899 | 0.608 | 0 | 0 | 403.0 |
| `mosse` | 0.759 | 0.715 | 0.736 | 0.706 | 1 | 0 | 745.6 |

## Results by regime -- the real story ([EVAL-DESIGN.md](../EVAL-DESIGN.md) rationale)

Pooling across regimes averages a method's best and worst cases together (the table above is a **summary, not a verdict**); per-regime is the primary result. `EASY` = chipset (identical, fixed-scale, low-texture -- the NCC-favourable baseline), `TEXTURED` = textured-plain (fixed pose, real keypoints), `VARIED` = textured-varied (scale 0.6-1.6x, rotation +/-35 deg), `CLUTTERED` = textured-cluttered (mild variation + noisy background + distractors). Scope note: this is the **synthetic** side only (chipset/textured); see [real-objects-findings.md](../reports/real-objects-findings.md) for the real-photo comparison.

### EASY

| method | precision | recall | F1 | AP | n img |
| --- | --- | --- | --- | --- | --- |
| `ncc` | 1.000 | 1.000 | 1.000 | 1.000 | 10 |
| `sparse-geo` | 1.000 | 0.294 | 0.455 | 0.400 | 10 |
| `dino-dense` | 0.122 | 0.294 | 0.172 | 0.234 | 10 |
| `owlv2-oneshot` | 0.145 | 0.765 | 0.244 | 0.287 | 10 |
| `propose-retrieve` | 0.883 | 0.976 | 0.927 | 0.636 | 10 |
| `mosse` | 0.899 | 0.941 | 0.920 | 0.856 | 10 |

### TEXTURED

| method | precision | recall | F1 | AP | n img |
| --- | --- | --- | --- | --- | --- |
| `ncc` | 1.000 | 1.000 | 1.000 | 1.000 | 16 |
| `sparse-geo` | 1.000 | 1.000 | 1.000 | 1.000 | 16 |
| `dino-dense` | 0.730 | 0.793 | 0.760 | 0.536 | 16 |
| `owlv2-oneshot` | 0.804 | 0.878 | 0.840 | 0.554 | 16 |
| `propose-retrieve` | 0.921 | 1.000 | 0.959 | 0.606 | 16 |
| `mosse` | 1.000 | 1.000 | 1.000 | 1.000 | 16 |

### VARIED

| method | precision | recall | F1 | AP | n img |
| --- | --- | --- | --- | --- | --- |
| `ncc` | 0.629 | 0.361 | 0.459 | 0.398 | 16 |
| `sparse-geo` | 0.765 | 0.755 | 0.760 | 0.711 | 16 |
| `dino-dense` | 0.557 | 0.755 | 0.641 | 0.540 | 16 |
| `owlv2-oneshot` | 0.876 | 0.865 | 0.870 | 0.573 | 16 |
| `propose-retrieve` | 0.942 | 0.935 | 0.939 | 0.600 | 16 |
| `mosse` | 0.559 | 0.458 | 0.504 | 0.517 | 16 |

### CLUTTERED

| method | precision | recall | F1 | AP | n img |
| --- | --- | --- | --- | --- | --- |
| `ncc` | 0.879 | 0.681 | 0.768 | 0.820 | 16 |
| `sparse-geo` | 0.858 | 0.869 | 0.863 | 0.836 | 16 |
| `dino-dense` | 0.615 | 0.787 | 0.690 | 0.548 | 16 |
| `owlv2-oneshot` | 0.735 | 0.938 | 0.824 | 0.489 | 16 |
| `propose-retrieve` | 0.738 | 0.931 | 0.823 | 0.565 | 16 |
| `mosse` | 0.903 | 0.756 | 0.823 | 0.829 | 16 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.922 | 0.365 |
| `sparse-geo` | 0.811 | 0.716 |
| `dino-dense` | 0.664 | 0.677 |
| `owlv2-oneshot` | 0.838 | 0.778 |
| `propose-retrieve` | 0.975 | 0.851 |
| `mosse` | 0.941 | 0.455 |

## Recall by ground-truth box size (small/medium/large, as a fraction of the image)

Pooled over every swept image (chipset + textured + real-objects + the configured synthetic scenes), not split by regime -- "does this method find small instances?" Cuts: small < 0.4% of image area, medium < 1.6%, else large (same cuts the floor-plan research path uses; validated as a non-degenerate three-way split on this set too -- see the module docstring).

| method | small recall (n) | medium recall (n) | large recall (n) |
| --- | --- | --- | --- |
| `ncc` | 0.932 (88) | 0.648 (273) | 0.613 (403) |
| `sparse-geo` | 0.318 (88) | 0.755 (273) | 0.873 (403) |
| `dino-dense` | 0.273 (88) | 0.689 (273) | 0.744 (403) |
| `owlv2-oneshot` | 0.750 (88) | 0.799 (273) | 0.831 (403) |
| `propose-retrieve` | 0.943 (88) | 0.883 (273) | 0.935 (403) |
| `mosse` | 0.886 (88) | 0.700 (273) | 0.687 (403) |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x660 | 1024x683 | 768x1024 | 1024x768 | 777x1024 | 1024x815 | 1024x925 | 1024x1024 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 211.5 | 442.8 | 832.0 | 1499.3 | 1645.9 | 1588.2 | 1610.5 | 1874.2 | 2290.5 | 2155.2 | 2000.3 | 2960.2 | 3077.8 | 4594.5 | 8038.9 | 13700.0 | 21874.0 | 35080.4 | 67040.3 |
| `sparse-geo` | 10.2 | 13.9 | 77.1 | 80.5 | 42.2 | 115.7 | 130.0 | 127.8 | 106.6 | 323.8 | 161.8 | 130.8 | 144.0 | 116.1 | 231.4 | 335.6 | 533.4 | 1051.0 | 1967.5 |
| `dino-dense` | 165.6 | 130.9 | 260.4 | 469.2 | 718.7 | 833.2 | 889.4 | 1037.6 | 1007.0 | 1028.6 | 1149.0 | 1387.3 | 1647.3 | 4063.5 | 4691.5 | 4111.9 | 4079.6 | 4143.7 | 3505.3 |
| `owlv2-oneshot` | 4892.7 | 3966.7 | 4012.4 | 4008.7 | 4248.1 | 3967.3 | 3950.5 | 3955.3 | 3977.5 | 3956.4 | 3963.7 | 4104.4 | 3968.4 | 4061.4 | 4060.1 | 4046.0 | 3983.8 | 3997.0 | 4132.0 |
| `propose-retrieve` | 436.5 | 285.4 | 359.7 | 370.5 | 374.0 | 746.7 | 800.2 | 1088.9 | 417.5 | 890.2 | 840.0 | 721.6 | 393.9 | 241.0 | 350.8 | 553.4 | 384.1 | 681.9 | 326.3 |
| `mosse` | 69.9 | 614.3 | 463.4 | 679.4 | 428.5 | 1237.6 | 1274.0 | 1186.4 | 794.6 | 1078.6 | 1370.6 | 680.1 | 1314.5 | 1512.9 | 13952.6 | 2435.9 | 3871.0 | 21843.9 | 29029.2 |

## Insight

**Per-regime winner (F1):** EASY = `ncc` (1.00); TEXTURED = `ncc` (1.00); VARIED = `propose-retrieve` (0.94); CLUTTERED = `sparse-geo` (0.86).
No single method wins every regime (3 different winners across 4 regimes) -- this is the reason all 6 methods are swept rather than picking one, and why the per-regime table above is the result to read, not the pooled summary.

**Size sensitivity (large-bucket recall minus small-bucket recall):** most size-sensitive is `sparse-geo` (+0.56, finds large instances much more reliably than small ones); least is `ncc` (-0.32, actually finds SMALL instances more reliably).
