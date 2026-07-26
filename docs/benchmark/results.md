# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `48e0b13fcd04d8b5845af99e64c26bdbcfb894ff`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.972 | 0.531 | 0.687 | 0.364 | 1 | 0 | 121.5 |
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 117.3 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 556.6 |
| `propose-retrieve` | 0.716 | 0.985 | 0.829 | 0.598 | 0 | 0 | 412.1 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.811 | 0.140 |
| `sparse-geo` | 0.773 | 0.765 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.994 | 0.971 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 36.8 | 46.2 | 75.6 | 119.5 | 157.6 | 188.4 | 410.7 | 710.1 | 1125.3 | 1767.8 | 3044.1 | 6561.1 |
| `sparse-geo` | 13.4 | 16.5 | 95.5 | 108.3 | 44.1 | 133.8 | 132.8 | 281.0 | 381.8 | 590.2 | 1341.8 | 2240.5 |
| `dino-dense` | 165.9 | 135.3 | 292.4 | 534.3 | 718.3 | 1118.1 | 4409.1 | 5377.1 | 5061.2 | 5002.5 | 5115.8 | 4146.4 |
| `propose-retrieve` | 378.4 | 275.0 | 407.6 | 434.7 | 407.1 | 421.0 | 274.3 | 371.3 | 568.6 | 376.2 | 683.2 | 322.6 |
