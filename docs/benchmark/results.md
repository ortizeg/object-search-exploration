# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `a052a651704cbc98092f960b35a96010e5ce792a`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.972 | 0.531 | 0.687 | 0.364 | 1 | 0 | 103.4 |
| `sparse-geo` | 0.729 | 0.711 | 0.720 | 0.629 | 11 | 0 | 80.7 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 518.3 |
| `propose-retrieve` | 0.716 | 0.985 | 0.829 | 0.598 | 0 | 0 | 392.1 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.811 | 0.140 |
| `sparse-geo` | 0.714 | 0.708 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.994 | 0.971 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 21.0 | 39.4 | 65.5 | 102.9 | 133.5 | 165.7 | 353.5 | 618.6 | 986.2 | 1574.2 | 2717.2 | 5591.2 |
| `sparse-geo` | 8.1 | 13.1 | 71.7 | 78.5 | 39.8 | 103.1 | 116.0 | 228.2 | 305.6 | 471.2 | 1177.6 | 1615.8 |
| `dino-dense` | 140.6 | 136.3 | 282.6 | 511.0 | 769.1 | 1074.9 | 4223.8 | 4583.6 | 4737.8 | 4361.3 | 4355.4 | 3760.5 |
| `propose-retrieve` | 396.6 | 298.2 | 380.2 | 395.0 | 392.6 | 405.4 | 255.2 | 372.9 | 590.8 | 410.1 | 702.2 | 337.8 |
