# Robustness backlog

Deferred robustness work, one section per search method (DOC-05). Each entry is a deliberate
"not now" — a known way to make a method stronger that was scoped out to keep the method
readable and the phase small. Each method's section is mirrored verbatim from that method's
module docstring `ROBUSTNESS BACKLOG` block, so the two cannot drift.

This file grows as methods land; it completes in Phase 8 when every registered method has a
section here.

## `ncc` (Method 1 — normalized cross-correlation)

None of the following is built in Phase 2; all are captured here and in the `ncc.py` docstring.

- **FFT-based correlation for large templates.** The spatial `matchTemplate` is O(H·W·h·w); a
  single full-scene FFT cross-correlation is O(H·W·log(H·W)) and wins decisively once the
  template is large.
- **Log-polar / Fourier-Mellin registration** for joint rotation+scale invariance in one
  correlation, replacing the brute-force rotated-template × pyramid bank.
- **Discriminative correlation filters (MOSSE/KCF)** trained on the single exemplar crop, so
  the filter learns to suppress background instead of correlating raw pixels.
