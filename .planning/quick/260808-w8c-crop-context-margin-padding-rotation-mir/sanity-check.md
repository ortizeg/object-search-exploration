# Task 4 sanity check — before any GPU money is spent

Per `margin-verdict.md`: no margin beat 0.0 on both classes, so this sanity check (and the eventual
`contrastive-crop-v2` GPU arm) runs with `--supcon-crop-augment` only, `--supcon-crop-margin-frac 0.0`.

## Commands run

```
pixi run -e export finetune-owlv2 --loss-mode contrastive --supcon-crop-context --supcon-crop-augment \
  --limit-images 4 --epochs 12 --batch-size 2 --grad-accum 2 --device mps --out models/finetune/_sanity_crop_augment
pixi run -e export finetune-owlv2 --loss-mode both --supcon-crop-context --supcon-crop-augment \
  --limit-images 2 --epochs 1 --max-steps 1 --device cpu --out models/finetune/_smoke_both_crop_augment
```

Device: **MPS** for the overfit run (no fallback to CPU needed, ran cleanly on Apple Silicon,
~13s/epoch, 12 epochs in under 3 minutes). CPU for the `both`-mode smoke run (matches the plan's
exact invocation).

## The four numbers

1. **`train_loss_supcon` falls substantially, epoch 1 → epoch 12:**
   `5.1285 → 4.4479` (13.3% relative drop on a 4-image fixed subset).

2. **`val_cos_gap` widens, epoch 0 → epoch 12** — the property `owlv2-oneshot` actually scores with
   moved in the right direction:
   - `gap_class`: `+0.1104 → +0.1197` (+8.4% relative)
   - `gap_background`: `+0.2566 → +0.3162` (+23.2% relative)

3. **`train_loss_bbox` / `train_loss_giou` did not blow up — they fell, same as the box head being
   trained normally, not sacrificed to the contrastive term:**
   - `train_loss_bbox`: `0.0211 → 0.0127` (epoch 1 → epoch 12)
   - `train_loss_giou`: `0.3126 → 0.1564` (epoch 1 → epoch 12)

4. **`val_crop_scene_agreement.self_score_mean` moves toward positive, epoch 0 → epoch 12, and
   never dips negative:** `+0.5252 → +0.7567` (+44.1% relative). This is the same property
   260808-dla's own sanity check measured (which went `+0.5355 → +0.7037`, +31.4%) — the
   rotation/mirror augmentation on top of the already-working crop-context mechanism does not
   disturb it; if anything this run's final self-score is slightly higher, consistent with the
   augmentation teaching a more orientation-robust (and thus more stable) crop embedding.

## `both`-mode + crop-augment smoke test

`--loss-mode both --supcon-crop-context --supcon-crop-augment --limit-images 2 --epochs 1
--max-steps 1 --device cpu`: `train_log.json` carries BOTH `train_loss_ce` (`2.1188`) AND
`train_loss_supcon` (`4.5008`) populated simultaneously at epoch 1 — confirmed by reading the raw
JSON — AND `val_crop_scene_agreement` (`self_score_mean=+0.5219, n_pairs=2.0`) is present. The
three-way combination (`both` mode + crop-context + crop-augment) is demonstrably functional.

## Verdict: **GO**

All four checks pass with real, non-trivial margins, including the fourth (the property this whole
recipe exists to protect), which not only holds but slightly improves over 260808-dla's own
crop-context-only sanity numbers. The recipe is demonstrably still learning the intended property
with augmentation added, on a fixed 4-image subset, before any GPU is rented. Proceeding to Task 5
(human checkpoint) for explicit sign-off before vast.ai spend.
