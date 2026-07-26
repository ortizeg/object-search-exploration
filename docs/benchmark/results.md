# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `271f14741cf943c14a0d489e7f160c945c2d23c4`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.855 | 0.730 | 0.788 | 0.779 | 0 | 0 | 1515.7 |
| `sparse-geo` | 0.729 | 0.711 | 0.720 | 0.629 | 11 | 0 | 87.0 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 488.0 |
| `propose-retrieve` | 0.716 | 0.985 | 0.829 | 0.598 | 0 | 0 | 372.9 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.959 | 0.412 |
| `sparse-geo` | 0.714 | 0.708 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.994 | 0.971 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 260.0 | 545.8 | 822.9 | 1486.9 | 1616.4 | 2392.0 | 5786.3 | 10029.6 | 14574.9 | 21903.1 | 34859.6 | 65496.9 |
| `sparse-geo` | 8.5 | 12.5 | 79.5 | 80.4 | 40.8 | 109.1 | 110.4 | 212.3 | 290.0 | 441.1 | 928.9 | 1319.4 |
| `dino-dense` | 153.6 | 137.5 | 267.1 | 486.0 | 727.3 | 1015.1 | 4127.1 | 4168.4 | 4093.0 | 4078.3 | 4137.3 | 3395.0 |
| `propose-retrieve` | 323.8 | 283.6 | 364.7 | 369.7 | 381.1 | 385.9 | 243.3 | 347.9 | 561.6 | 377.4 | 671.4 | 332.7 |
