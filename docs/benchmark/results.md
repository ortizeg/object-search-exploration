# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `c450791e0842c330d593913baf0666c4872631c6`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.855 | 0.730 | 0.788 | 0.779 | 0 | 0 | 1451.2 |
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 86.8 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 467.5 |
| `owlv2-oneshot` | 0.509 | 0.878 | 0.645 | 0.499 | 0 | 0 | 4026.1 |
| `propose-retrieve` | 0.865 | 0.955 | 0.908 | 0.599 | 0 | 0 | 361.6 |
| `mosse` | 0.722 | 0.665 | 0.692 | 0.734 | 1 | 0 | 209.3 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.959 | 0.412 |
| `sparse-geo` | 0.773 | 0.765 |
| `dino-dense` | 0.661 | 0.749 |
| `owlv2-oneshot` | 0.855 | 0.909 |
| `propose-retrieve` | 0.971 | 0.934 |
| `mosse` | 0.879 | 0.366 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 265.6 | 504.4 | 819.6 | 1413.7 | 1561.9 | 2266.0 | 4678.5 | 8003.5 | 13636.5 | 21172.6 | 35920.7 | 66333.8 |
| `sparse-geo` | 13.9 | 14.9 | 69.0 | 77.9 | 43.9 | 109.9 | 107.8 | 214.4 | 310.5 | 452.2 | 923.6 | 1524.4 |
| `dino-dense` | 159.4 | 134.7 | 258.4 | 465.1 | 692.4 | 976.0 | 7772.3 | 8161.0 | 10010.3 | 4258.4 | 4114.8 | 3711.2 |
| `owlv2-oneshot` | 5133.6 | 3979.3 | 4014.1 | 4003.5 | 4094.7 | 4007.1 | 4057.9 | 4026.8 | 4045.1 | 4055.8 | 4085.4 | 4147.8 |
| `propose-retrieve` | 392.5 | 275.9 | 357.9 | 361.2 | 369.7 | 372.5 | 227.3 | 332.1 | 545.3 | 360.8 | 659.5 | 321.0 |
| `mosse` | 57.0 | 594.2 | 139.0 | 198.6 | 261.7 | 330.9 | 2043.8 | 9147.3 | 2029.3 | 3314.4 | 38710.2 | 29454.7 |
