# Benchmark results

Generated from `results.json` by `pixi run bench-charts`. Do not hand-edit; re-run the benchmark and regenerate. `n/a` is an **abstention** (nothing returned, so the rate is undefined) -- never read it as zero.

- IoU threshold: **0.5** | AP convention: all-point interpolation (COCO-style), from the EVAL-08 candidate log
- Coverage: 60 labelled / 60 requested
- Git SHA at run: `40821b26aa687d68b7c03a01b97d27a44180a7cf`

## Pooled metrics by method

| method | precision | recall | F1 | mean AP | abstentions | errors | p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 0.972 | 0.531 | 0.687 | 0.364 | 1 | 0 | 97.1 |
| `sparse-geo` | 0.884 | 0.770 | 0.823 | 0.751 | 6 | 0 | 92.7 |
| `dino-dense` | 0.474 | 0.698 | 0.565 | 0.475 | 0 | 0 | 481.3 |
| `propose-retrieve` | 0.865 | 0.955 | 0.908 | 0.599 | 0 | 0 | 373.6 |
| `owlv2-oneshot` | 0.509 | 0.878 | 0.645 | 0.499 | 0 | 0 | 4376.1 |

## Recall by scale bucket (the NCC-vs-sparse-geo crossover)

| method | fixed-scale recall | varied-scale recall |
| --- | --- | --- |
| `ncc` | 0.811 | 0.140 |
| `sparse-geo` | 0.773 | 0.765 |
| `dino-dense` | 0.661 | 0.749 |
| `propose-retrieve` | 0.971 | 0.934 |
| `owlv2-oneshot` | 0.855 | 0.909 |

## p50 latency by canvas size (ms)

| method | 320x240 | 512x384 | 640x480 | 800x600 | 960x640 | 1024x768 | 1600x1200 | 2048x1536 | 2560x1920 | 3200x2400 | 4096x3072 | 6000x4000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ncc` | 29.4 | 37.2 | 61.5 | 94.3 | 133.3 | 154.0 | 339.8 | 604.8 | 953.6 | 1507.1 | 2754.0 | 5425.2 |
| `sparse-geo` | 13.6 | 12.4 | 79.4 | 78.8 | 43.3 | 113.3 | 110.9 | 223.4 | 317.1 | 484.8 | 1094.3 | 1528.0 |
| `dino-dense` | 158.0 | 138.3 | 265.0 | 478.4 | 717.7 | 989.4 | 4113.9 | 4444.0 | 4102.9 | 4094.6 | 4115.8 | 3538.6 |
| `propose-retrieve` | 359.4 | 307.3 | 367.7 | 373.4 | 403.5 | 406.6 | 244.2 | 349.8 | 564.4 | 383.1 | 692.8 | 328.1 |
| `owlv2-oneshot` | 5405.6 | 4309.6 | 4475.7 | 4598.4 | 4245.0 | 4419.5 | 4252.3 | 4377.3 | 4290.5 | 4252.5 | 4268.1 | 4495.9 |
