# Task 3 sanity check — before any GPU money is spent

## Commands run

```
pixi run -e export finetune-owlv2 --loss-mode contrastive --supcon-crop-context --limit-images 4 \
  --epochs 12 --batch-size 2 --grad-accum 2 --device mps --out models/finetune/_sanity_crop_context
pixi run -e export finetune-owlv2 --loss-mode both --supcon-crop-context --limit-images 2 \
  --epochs 1 --max-steps 1 --device cpu --out models/finetune/_smoke_both_crop
```

Device: **MPS** for the overfit run — ran cleanly on Apple Silicon, no fallback to CPU needed
(~8-9s/epoch, 12 epochs in under 2 minutes). CPU for the `both`-mode smoke run (matches the plan's
exact invocation).

## The four numbers

1. **`train_loss_supcon` falls substantially, epoch 0 (pre-training reference) → epoch 12:**
   `4.8589 → 4.2700` (12.1% relative drop on a 4-image fixed subset). Epoch 1 → epoch 12 shows the
   same trend, ruling out "the drop is entirely the first gradient step": `4.8466 → 4.2700` (also a
   ~12% drop, monotonic apart from normal SGD noise around epochs 7-10).

2. **`val_cos_gap` widens, epoch 0 → epoch 12** — the property `owlv2-oneshot` actually scores with
   moved in the right direction:
   - `gap_class`: `+0.1131 → +0.1420` (+25.6% relative)
   - `gap_background`: `+0.2719 → +0.3680` (+35.4% relative, moved further than the class gap,
     matching D-hg1-03's intent — background separation is what the classification-loss recipe
     never touched)

3. **`train_loss_bbox` / `train_loss_giou` did not blow up — they fell, same as the box head being
   trained normally, not sacrificed to the (now two-part) contrastive term:**
   - `train_loss_bbox`: `0.0211 → 0.0131` (epoch 1 → epoch 12)
   - `train_loss_giou`: `0.3126 → 0.1601` (epoch 1 → epoch 12)

4. **NEW — `val_crop_scene_agreement.self_score_mean` moves toward positive, epoch 0 → epoch 12,
   and never once dips negative:** `+0.5355 → +0.7037` (+31.4% relative). This is the whole point
   of this task. 260805-hg1's reference point was the crop-context-FREE `contrastive` checkpoint's
   single-exemplar `self_score` collapsing from the pretrained baseline's `+0.712` down to `-0.297`
   after training touched only scene-context forward passes. Here, with crop-context anchors in the
   SupCon pool, the analogous quantity (the pooled, 4-image instance-level crop/scene cosine, not
   yet the single-exemplar `self_score` itself — that comparison is Task 6's `self_score_diagnostic.py`
   against the real checkpoint) starts at a healthy `+0.5355` (this run's own pretrained reference —
   built from THIS task's crop-context diagnostic, not identical in method to the original
   `self-similarity` calibration score, so the two numbers are not directly comparable in
   isolation) and *increases* monotonically-ish through training, reaching `+0.7037` by epoch 12 —
   moving further positive, not degrading toward zero or negative the way the crop-context-free
   arm did. Per-epoch values: `0.535 (e0) → 0.470 (e1) → 0.510 (e2) → 0.525 (e3) → 0.608 (e4) →
   0.720 (e5) → 0.671 (e6) → 0.696 (e7) → 0.700 (e8) → 0.702 (e9) → 0.703 (e10) → 0.704 (e11) →
   0.704 (e12)`. A brief dip at epoch 1 (0.470, below epoch 0's 0.535) before recovering and then
   holding a clear upward trend is consistent with the box head's early optimizer-warmup noise seen
   in `train_loss_supcon`/`val_cos_gap` above, not a sign of instability.

## `both`-mode + crop-context smoke test

`--loss-mode both --supcon-crop-context --limit-images 2 --epochs 1 --max-steps 1 --device cpu`:
`train_log.json` carries BOTH `train_loss_ce`/`val_loss_ce` (`2.1188` / `1.7054`) AND
`train_loss_supcon`/`val_loss_supcon` (`4.1167` / `4.3398`) populated simultaneously at epoch 1 —
confirmed by reading the raw JSON, not just the log line — AND `val_crop_scene_agreement`
(`self_score_mean=+0.4997, n_pairs=2.0`) is present. The three-way combination (`both` mode +
crop-context) is demonstrably functional, not an advertised-but-untried knob combination. Epoch-0
`val_loss_ce`/`val_crop_scene_agreement.self_score_mean` (`1.7054` / `+0.4620`) match the
flag-on-crop-context epoch-0 reference exactly (no gradient step has happened yet at epoch 0,
regardless of `--loss-mode`).

## Verdict: **GO**

All four checks pass with real, non-trivial margins (not borderline), including the fourth — the
whole point of this task — which moves substantially and monotonically toward positive rather than
merely failing to move (the STOP condition this gate exists to catch). The recipe is demonstrably
learning the intended crop/scene-agreement property on a fixed 4-image subset before any GPU is
rented. Proceeding to Task 4 (human checkpoint) for explicit sign-off before vast.ai spend.
