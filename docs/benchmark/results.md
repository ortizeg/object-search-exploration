# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `b58df496b2440b3cc965e899974fda359a0dc401`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.972 | 0.531 | 0.687 | 0.364 | 1 | 0 | 104.6 |
| `sparse-geo` | 0.729 | 0.711 | 0.720 | 0.629 | 11 | 0 | 82.7 |
| `dino-dense` | 0.127 | 0.026 | 0.043 | 0.058 | 0 | 0 | 492.6 |
| `propose-retrieve` | 0.716 | 0.985 | 0.829 | 0.598 | 0 | 0 | 371.2 |
| `owlv2-oneshot` | 0.509 | 0.878 | 0.645 | 0.499 | 0 | 0 | 4209.7 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.811 | 0.140 |
| `sparse-geo` | 0.714 | 0.708 |
| `dino-dense` | 0.041 | 0.004 |
| `propose-retrieve` | 0.994 | 0.971 |
| `owlv2-oneshot` | 0.855 | 0.909 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 21.8 | 41.7 | 65.9 | 101.1 | 139.0 | 165.7 | 358.6 | 620.5 | 975.2 | 1582.3 | 2761.4 | 5481.6 |
| `sparse-geo` | 6.6 | 12.8 | 73.5 | 76.3 | 38.3 | 108.5 | 109.5 | 218.7 | 293.1 | 456.7 | 928.8 | 1353.8 |
| `dino-dense` | 190.5 | 189.4 | 272.8 | 490.8 | 728.1 | 1024.5 | 4188.2 | 4257.2 | 4191.7 | 4163.5 | 4216.3 | 3478.9 |
| `propose-retrieve` | 263.2 | 277.6 | 364.4 | 368.0 | 387.8 | 383.2 | 246.6 | 358.0 | 561.7 | 379.5 | 683.7 | 323.1 |
| `owlv2-oneshot` | 4933.9 | 4021.9 | 4208.1 | 4218.2 | 4624.2 | 4225.2 | 4016.2 | 4037.2 | 4028.7 | 4019.1 | 4076.8 | 4137.1 |
