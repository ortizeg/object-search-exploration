"""The evaluation harness: ground truth, metrics, benchmark, paired comparison, Bradley-Terry.

This package turns "which method is better" into a per-slice number. Nothing here stores a
derived metric: precision, recall, F1 and AP are computed from ground-truth-scored predictions
(this package) or from the store's NULL-propagating views (:mod:`object_search.store.stats`),
never persisted as columns (EVAL-07).

The five layers, each in one readable file, are imported from their submodules directly (there
is no re-export here, matching the codebase convention of explicit ``from ...eval.x import y``):

* :mod:`labels` -- one loader for every ground-truth sidecar (synthetic, chipset, hand-labelled).
* :mod:`metrics` -- greedy-IoU precision/recall/F1 and all-point-interpolation AP, with the
  abstention convention (precision is ``None`` when nothing was returned, never ``0``).
* :mod:`benchmark` -- the Hydra sweep over method x image x config, the one place Hydra is used.
* :mod:`paired` -- one exemplar box through all four methods in a single call (EVAL-05).
* :mod:`bradley_terry` -- paired-comparison strengths with the complete-separation guard.
"""

from __future__ import annotations
