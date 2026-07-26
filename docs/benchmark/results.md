# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `f2d69766621774306fa64c9530edad7ce0324287`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.972 | 0.531 | 0.687 | 0.364 | 1 | 0 | 94.0 |
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 90.7 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 471.8 |
| `propose-retrieve` | 0.865 | 0.955 | 0.908 | 0.599 | 0 | 0 | 375.0 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.811 | 0.140 |
| `sparse-geo` | 0.773 | 0.765 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.971 | 0.934 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 20.5 | 37.4 | 59.9 | 91.2 | 125.7 | 150.0 | 329.4 | 581.6 | 905.7 | 1476.1 | 2566.3 | 5126.0 |
| `sparse-geo` | 7.2 | 13.4 | 76.2 | 81.1 | 42.3 | 115.3 | 108.9 | 220.1 | 321.1 | 470.9 | 937.9 | 1405.6 |
| `dino-dense` | 124.7 | 136.3 | 260.7 | 468.5 | 770.4 | 983.4 | 4089.5 | 4108.6 | 4064.4 | 4066.2 | 4076.8 | 3352.8 |
| `propose-retrieve` | 322.2 | 281.7 | 361.8 | 382.2 | 383.1 | 386.7 | 240.9 | 349.9 | 557.9 | 377.1 | 665.4 | 322.3 |
