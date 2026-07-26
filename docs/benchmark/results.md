# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `48e0b13fcd04d8b5845af99e64c26bdbcfb894ff`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.972 | 0.531 | 0.687 | 0.364 | 1 | 0 | 105.3 |
| `sparse-geo` | 0.729 | 0.711 | 0.720 | 0.629 | 11 | 0 | 90.1 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 985.9 |
| `propose-retrieve` | 0.865 | 0.955 | 0.908 | 0.599 | 0 | 0 | 766.1 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.811 | 0.140 |
| `sparse-geo` | 0.714 | 0.708 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.971 | 0.934 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 69.0 | 39.7 | 65.3 | 101.9 | 151.4 | 164.7 | 382.4 | 649.8 | 1006.8 | 1666.5 | 2863.8 | 5599.5 |
| `sparse-geo` | 13.6 | 13.8 | 77.3 | 82.8 | 40.4 | 110.5 | 111.9 | 219.9 | 294.2 | 452.6 | 997.7 | 1576.3 |
| `dino-dense` | 174.6 | 145.7 | 614.9 | 973.0 | 1419.7 | 1890.5 | 4122.4 | 4908.0 | 6348.8 | 6558.2 | 6643.4 | 6011.9 |
| `propose-retrieve` | 669.8 | 652.9 | 732.4 | 752.1 | 794.5 | 814.4 | 541.0 | 725.1 | 1306.7 | 822.2 | 1474.3 | 716.9 |
