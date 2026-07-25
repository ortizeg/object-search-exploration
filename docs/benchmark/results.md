# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 12 labelled / 12 requested
- Git SHA at run: `15b4278241e1d227917d3bda2260fed4ef09b3e2`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.913 | 0.922 | 0.918 | 0.484 | 0 | 0 | 238.0 |
| `sparse-geo` | 0.833 | 0.097 | 0.174 | 0.083 | 11 | 0 | 76.4 |
| `dino-dense` | 0.276 | 0.078 | 0.121 | 0.190 | 0 | 0 | 2259.2 |
| `propose-retrieve` | 0.748 | 0.951 | 0.838 | 0.635 | 0 | 0 | 291.5 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.989 | 0.300 |
| `sparse-geo` | 0.108 | 0.000 |
| `dino-dense` | 0.086 | 0.000 |
| `propose-retrieve` | 0.978 | 0.700 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 49.9 | 40.5 | 92.4 | 118.9 | 132.5 | 341.0 | 610.4 | 893.7 | 1449.0 | 2603.0 | 5670.9 |
| `sparse-geo` | 11.0 | 11.6 | 26.5 | 35.3 | 45.1 | 107.6 | 202.7 | 275.2 | 445.9 | 1122.6 | 2289.9 |
| `dino-dense` | 208.5 | 135.8 | 458.1 | 722.1 | 960.5 | 4157.7 | 5052.0 | 4200.1 | 4025.7 | 4088.5 | 3557.9 |
| `propose-retrieve` | 270.0 | 274.1 | 222.5 | 355.1 | 269.0 | 228.4 | 336.7 | 531.5 | 355.2 | 639.7 | 308.9 |
