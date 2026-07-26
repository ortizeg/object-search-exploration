# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `f52d6f5f5afb901c47f429e800d954cd7a3f3ae9`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.855 | 0.730 | 0.788 | 0.779 | 0 | 0 | 1529.6 |
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 90.9 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 474.7 |
| `propose-retrieve` | 0.865 | 0.955 | 0.908 | 0.599 | 0 | 0 | 370.9 |
| `owlv2-oneshot` | 0.509 | 0.878 | 0.645 | 0.499 | 0 | 0 | 3958.7 |

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
| `ncc` | 206.1 | 417.9 | 833.0 | 1510.8 | 1624.6 | 2450.3 | 4561.4 | 7830.0 | 13119.2 | 20769.5 | 33925.1 | 65321.6 |
| `sparse-geo` | 6.9 | 13.6 | 86.6 | 82.1 | 43.0 | 115.4 | 113.2 | 226.5 | 325.1 | 473.1 | 1060.5 | 1560.5 |
| `dino-dense` | 131.0 | 131.6 | 259.4 | 467.4 | 708.6 | 988.8 | 4074.5 | 4316.9 | 4058.1 | 4087.5 | 4487.8 | 3479.8 |
| `propose-retrieve` | 327.9 | 276.6 | 361.2 | 370.3 | 379.7 | 381.4 | 237.7 | 344.8 | 560.5 | 375.2 | 657.4 | 317.2 |
| `owlv2-oneshot` | 4694.0 | 4087.7 | 3953.9 | 3951.4 | 3947.4 | 3969.1 | 4038.3 | 4092.7 | 4151.6 | 4165.9 | 4091.5 | 4176.4 |
