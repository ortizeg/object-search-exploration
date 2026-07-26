# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `c450791e0842c330d593913baf0666c4872631c6`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.855 | 0.730 | 0.788 | 0.779 | 0 | 0 | 1663.5 |
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 106.5 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 532.7 |
| `propose-retrieve` | 0.865 | 0.955 | 0.908 | 0.599 | 0 | 0 | 384.0 |
| `owlv2-oneshot` | 0.509 | 0.878 | 0.645 | 0.499 | 0 | 0 | 4926.0 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.959 | 0.412 |
| `sparse-geo` | 0.773 | 0.765 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.971 | 0.934 |
| `owlv2-oneshot` | 0.855 | 0.909 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 246.7 | 454.0 | 932.9 | 1635.9 | 1812.9 | 2635.0 | 4766.3 | 8401.6 | 14292.0 | 22293.6 | 36946.2 | 70968.6 |
| `sparse-geo` | 14.9 | 15.4 | 92.9 | 85.2 | 46.5 | 119.5 | 124.5 | 252.9 | 367.5 | 597.5 | 1578.5 | 2292.8 |
| `dino-dense` | 174.9 | 142.7 | 286.5 | 525.6 | 754.8 | 1088.0 | 4679.1 | 5452.5 | 5019.5 | 6518.4 | 5754.2 | 4716.5 |
| `propose-retrieve` | 505.9 | 328.3 | 373.1 | 382.4 | 388.5 | 393.4 | 259.0 | 370.4 | 579.5 | 393.6 | 697.3 | 338.2 |
| `owlv2-oneshot` | 4935.2 | 4136.4 | 5027.3 | 5023.7 | 5272.6 | 4884.2 | 4222.0 | 4391.0 | 4412.8 | 4481.7 | 4466.5 | 4946.2 |
