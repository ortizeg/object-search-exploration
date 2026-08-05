# Task 3 sanity check — before any GPU money is spent

## Commands run

```
pixi run -e export finetune-owlv2 --loss-mode contrastive --limit-images 4 \
  --epochs 12 --batch-size 2 --grad-accum 2 --device mps --out models/finetune/_sanity_contrastive
pixi run -e export finetune-owlv2 --loss-mode both --limit-images 2 --epochs 1 --max-steps 1 \
  --device cpu --out models/finetune/_smoke_both
```

Device: **MPS** for the overfit run (no fallback to CPU needed — ran cleanly, ~4.5s/epoch on Apple
Silicon). CPU for the `both`-mode smoke run (matches the plan's exact invocation).

## The three numbers

1. **`train_loss_supcon` falls substantially, epoch 0 (pre-training reference) → epoch 12:**
   `4.5393 → 4.0086` (12% relative drop on a 4-image fixed subset). Epoch 1 → epoch 12 shows the
   same trend more clearly since epoch 1 is the first post-step measurement: `4.5537 → 4.0086`
   (~12% drop, monotonic-ish with normal SGD noise).

2. **`val_cos_gap` widens, epoch 0 → epoch 12** — the property `owlv2-oneshot` actually scores with
   moved in the right direction:
   - `gap_class`: `+0.1207 → +0.1775` (+47% relative)
   - `gap_background`: `+0.2943 → +0.4742` (+61% relative, moved further than the class gap — matches
     D-hg1-03's intent, since background separation is the property the classification-loss recipe
     (260801-8zy) never touched)

3. **`train_loss_bbox` / `train_loss_giou` did not blow up — they fell, same as the box head being
   trained normally, not sacrificed to the contrastive term:**
   - `train_loss_bbox`: `0.0212 → 0.0130`
   - `train_loss_giou`: `0.3118 → 0.1584`

## `both`-mode smoke test

`--loss-mode both --limit-images 2 --epochs 1 --max-steps 1 --device cpu`: `train_log.json` carries
BOTH `train_loss_ce`/`val_loss_ce` AND `train_loss_supcon`/`val_loss_supcon` populated simultaneously
at every epoch — confirmed by reading the raw JSON, not just the log line. The third mode is
demonstrably functional, not an advertised-but-untried knob. Epoch-0 `loss_ce`/`loss_bbox`/`loss_giou`
values (`2.1188318729400635` / `0.01978086121380329` / `0.28750768303871155`) match the preflight
focal fixture's epoch-1 values exactly, confirming the frozen pre-training reference point is
identical regardless of `--loss-mode` (as it must be — no gradient step has happened yet).

## Verdict: **GO**

All three checks pass with real, non-trivial margins (not borderline). The recipe is demonstrably
learning the intended property on a fixed 4-image subset before any GPU is rented. Proceeding to
Task 4 (human checkpoint) for explicit sign-off before vast.ai spend.
