"""Inference-time post-processing for surgical phase sequences.

These transforms require no retraining. They exploit two domain facts about
laparoscopic cholecystectomy (Cholec80):

1. Phases occur in a fixed, non-decreasing order (0..6). Per-frame classifiers
   ignore this and flip between visually similar phases (e.g. CalotTriangle vs
   GallbladderDissection). Monotonic decoding enforces the ordering.

2. The training set is heavily imbalanced (~11:1 between the most and least
   frequent phase), so rare closing phases are under-predicted. Logit adjustment
   shifts decision boundaries using the known class priors at inference time.
"""
from __future__ import annotations

import numpy as np

# Cholec80 train-set (videos 1-40) phase frequencies, fraction of frames.
TRAIN_PRIORS = np.array([0.044, 0.427, 0.085, 0.279, 0.043, 0.084, 0.038])


def logit_adjust(logits: np.ndarray, priors: np.ndarray = TRAIN_PRIORS, tau: float = 1.0) -> np.ndarray:
    """Subtract tau*log(prior) from logits to counteract class imbalance.

    A larger tau more aggressively boosts rare classes. tau=0 is a no-op.
    """
    priors = np.clip(priors, 1e-6, None)
    return logits - tau * np.log(priors)[None, :]


def monotonic_decode(probs: np.ndarray, allow_skip: bool = True) -> np.ndarray:
    """Viterbi decode a per-frame probability matrix [T,7] under the constraint
    that the phase index is non-decreasing over time.

    With allow_skip=True a phase may be skipped (e.g. 2 -> 4); with False only
    same-or-next transitions are allowed. Returns int phase per frame [T].
    """
    T, C = probs.shape
    log_p = np.log(np.clip(probs, 1e-9, None))
    dp = np.full((T, C), -np.inf)
    back = np.zeros((T, C), dtype=int)
    dp[0, 0] = log_p[0, 0]          # surgery must start at phase 0
    if allow_skip:
        for c in range(C):
            dp[0, c] = log_p[0, c] if c == 0 else -np.inf
    for t in range(1, T):
        for c in range(C):
            lo = 0 if allow_skip else max(0, c - 1)
            best_prev, best_val = c, -np.inf
            for pc in range(lo, c + 1):
                if dp[t - 1, pc] > best_val:
                    best_val, best_prev = dp[t - 1, pc], pc
            dp[t, c] = best_val + log_p[t, c]
            back[t, c] = best_prev
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def postprocess_video(
    logits: np.ndarray,
    tau: float = 1.0,
    monotonic: bool = True,
    allow_skip: bool = True,
) -> np.ndarray:
    """Full pipeline for one video's logits [T,7] -> phase predictions [T]."""
    adj = logit_adjust(logits, tau=tau)
    probs = softmax(adj)
    if monotonic:
        return monotonic_decode(probs, allow_skip=allow_skip)
    return probs.argmax(axis=1)
