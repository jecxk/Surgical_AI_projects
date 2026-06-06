"""Boundary-aware error analysis for surgical phase predictions.

Splits per-frame errors into:
  - boundary: within `boundary_window` seconds of a ground-truth phase transition
  - interior: everywhere else

Reports accuracy/macro-F1 separately on each subset. Annotation noise is
concentrated at boundaries, so a fair view of model quality requires this split.
This is what TeCNO/Trans-SVNet do not report and what reviewers ask about.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def boundary_mask(labels: np.ndarray, window: int = 5) -> np.ndarray:
    """Return boolean mask of length T marking frames within `window` of a
    phase transition (in either direction). FPS-agnostic; window is in frames.

    A transition is a frame t where labels[t] != labels[t-1].
    """
    labels = np.asarray(labels)
    T = len(labels)
    is_transition = np.zeros(T, dtype=bool)
    diff = labels[1:] != labels[:-1]
    is_transition[1:] = diff
    mask = np.zeros(T, dtype=bool)
    idx = np.where(is_transition)[0]
    for t in idx:
        lo = max(0, t - window); hi = min(T, t + window + 1)
        mask[lo:hi] = True
    return mask


def split_metrics(y_true: np.ndarray, y_pred: np.ndarray, window: int = 5):
    """Compute (boundary, interior, all) accuracy + macro-F1 dicts."""
    valid = y_true >= 0
    yt = y_true[valid]; yp = y_pred[valid]
    bmask = boundary_mask(yt, window=window)
    out = {}
    for tag, m in (("boundary", bmask), ("interior", ~bmask), ("all", np.ones_like(bmask, dtype=bool))):
        if m.sum() == 0:
            out[tag] = {"n": 0, "acc": float("nan"), "macroF1": float("nan")}
            continue
        out[tag] = {
            "n": int(m.sum()),
            "acc": float(accuracy_score(yt[m], yp[m])),
            "macroF1": float(f1_score(yt[m], yp[m], average="macro",
                                       labels=list(range(7)), zero_division=0)),
        }
    return out
