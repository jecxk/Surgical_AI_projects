"""Causal streaming decoders + calibration for real-time surgical phase recognition.

This module complements `postprocess.py` (which provides the *offline* monotonic
decoder used as an upper bound). The goal here is to deliver decoders that:

  1. Run frame-by-frame with O(1) state per frame (no lookahead).
  2. Use a *learned* transition prior estimated from the training split, instead
     of the uniform monotonic assumption used by the offline decoder.
  3. Operate on calibrated probabilities (temperature-scaled logits) so that the
     dynamic-programming step weighs likelihood against transition prior in a
     well-defined log-probability space.

Three decoders are provided:
  - CausalArgmax     : baseline, argmax per frame (no temporal smoothing).
  - CausalMonotonic  : Viterbi-style streaming under monotonicity prior.
  - CausalHMM        : Viterbi-style streaming under learned transition matrix.

All decoders expose an `update(logits_t)` method that takes the current frame's
logits and returns the predicted phase for that frame. Internally they keep a
log-probability vector over the seven phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


NUM_PHASES = 7
NEG_INF = -1e9


def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    """Divide logits by temperature T (T>1 softens, T<1 sharpens)."""
    return logits / max(T, 1e-6)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def estimate_transition_matrix(label_sequences, num_phases: int = NUM_PHASES,
                               smoothing: float = 1.0) -> np.ndarray:
    """Estimate transition matrix A[i,j] = P(phase_{t+1}=j | phase_t=i) from a
    list of integer label sequences (each of shape [T]). Laplace smoothing.
    Returns a [num_phases, num_phases] row-stochastic matrix.
    """
    A = np.full((num_phases, num_phases), smoothing, dtype=np.float64)
    for seq in label_sequences:
        seq = np.asarray(seq)
        for i, j in zip(seq[:-1], seq[1:]):
            if 0 <= i < num_phases and 0 <= j < num_phases:
                A[i, j] += 1.0
    A /= A.sum(axis=1, keepdims=True)
    return A


def monotonic_transition_matrix(num_phases: int = NUM_PHASES,
                                stay: float = 0.95) -> np.ndarray:
    """Hand-crafted transition matrix that only allows non-decreasing transitions.
    Stays with prob `stay`; advances to any later phase uniformly. Used as a
    sanity baseline against the learned matrix.
    """
    A = np.zeros((num_phases, num_phases), dtype=np.float64)
    for i in range(num_phases):
        A[i, i] = stay
        remaining = 1.0 - stay
        later = num_phases - 1 - i
        if later > 0:
            A[i, i + 1:] = remaining / later
        else:
            A[i, i] = 1.0
    return A


def power_transition_matrix(A: np.ndarray, k: int) -> np.ndarray:
    """Raise a row-stochastic transition matrix to the k-th power.

    If A models the per-second transition probabilities, then A^k models the
    transition over k seconds (Chapman-Kolmogorov for a homogeneous Markov
    chain). This is needed when the decoder consumes sub-sampled frames spaced
    k seconds apart: a per-second matrix with P(stay)~0.99 would otherwise make
    the decoder far too sticky over a 10 s gap. Result is re-normalised to stay
    row-stochastic against floating-point drift.
    """
    k = max(1, int(round(k)))
    M = np.linalg.matrix_power(A, k)
    M = np.clip(M, 0.0, None)
    M /= M.sum(axis=1, keepdims=True)
    return M


@dataclass
class CausalDecoder:
    """Base class. Streaming Viterbi over phases with optional log-transition."""
    num_phases: int = NUM_PHASES
    log_A: Optional[np.ndarray] = None          # [C,C] log transition, None -> uniform
    log_pi: Optional[np.ndarray] = None         # [C] log initial; default delta at 0
    temperature: float = 1.0
    ema_alpha: float = 1.0                       # 1.0 = off; <1 smooths logits causally
    _log_alpha: np.ndarray = field(init=False, default=None)
    _ema: np.ndarray = field(init=False, default=None)
    _t: int = field(init=False, default=0)

    def reset(self):
        self._t = 0
        self._log_alpha = None
        self._ema = None

    def update(self, logits_t: np.ndarray) -> int:
        """Take logits at time t, return argmax phase under streaming Viterbi.

        We maintain max-marginal log-probabilities log_alpha[c] = max over paths
        ending in c at time t. At each step:
            log_alpha_new[c] = log_emit_t[c] + max_{c'} (log_alpha[c'] + log_A[c',c])
        Argmax phase = argmax_c log_alpha_new[c].

        If ema_alpha < 1, the raw logits are first passed through a causal
        exponential moving average (no lookahead) to suppress single-frame
        emission spikes before they reach the Viterbi step. This keeps the
        decoder streaming while damping the brief phase regressions that a very
        confident but wrong frame would otherwise cause.
        """
        if self.ema_alpha < 1.0:
            if self._ema is None:
                self._ema = logits_t.astype(np.float64).copy()
            else:
                self._ema = self.ema_alpha * logits_t + (1.0 - self.ema_alpha) * self._ema
            logits_t = self._ema
        log_emit = log_softmax(temperature_scale(logits_t, self.temperature))
        if self._log_alpha is None:
            # initial
            if self.log_pi is None:
                init = np.full(self.num_phases, NEG_INF)
                init[0] = 0.0
            else:
                init = self.log_pi
            self._log_alpha = init + log_emit
        else:
            if self.log_A is None:
                # uniform transition -> equivalent to per-frame argmax
                trans = self._log_alpha.max()
                self._log_alpha = log_emit + trans
            else:
                # vectorized Viterbi step: for each c, max over c' of alpha+log_A[:,c]
                cand = self._log_alpha[:, None] + self.log_A
                self._log_alpha = log_emit + cand.max(axis=0)
        self._t += 1
        return int(np.argmax(self._log_alpha))


def make_causal_argmax(temperature: float = 1.0) -> CausalDecoder:
    return CausalDecoder(log_A=None, temperature=temperature)


def make_causal_monotonic(stay: float = 0.95, temperature: float = 1.0) -> CausalDecoder:
    A = monotonic_transition_matrix(stay=stay)
    return CausalDecoder(log_A=np.log(A + 1e-12), temperature=temperature)


def make_causal_hmm(transition_matrix: np.ndarray, temperature: float = 1.0,
                    ema_alpha: float = 1.0) -> CausalDecoder:
    return CausalDecoder(log_A=np.log(transition_matrix + 1e-12), temperature=temperature,
                         ema_alpha=ema_alpha)


def make_causal_hmm_monotonic(transition_matrix: np.ndarray, temperature: float = 1.0) -> CausalDecoder:
    """Causal HMM decoder with a HARD non-decreasing constraint.

    Takes the learned transition matrix but masks out every backward transition
    (j < i), then renormalises each row over the allowed forward/stay targets.
    The streaming Viterbi can then only stay in the current phase or advance —
    never regress — which matches the fixed procedural order of cholecystectomy
    while remaining causal (no lookahead). This sits between the soft learned-HMM
    decoder (which may briefly regress when an emission is very confident) and the
    offline monotonic decoder (which is non-causal).
    """
    A = np.array(transition_matrix, dtype=np.float64).copy()
    C = A.shape[0]
    # zero out backward transitions
    for i in range(C):
        A[i, :i] = 0.0
    # renormalise; if a row became all-zero (shouldn't happen) keep self-loop
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A = A / row_sums
    for i in range(C):
        if A[i].sum() == 0:
            A[i, i] = 1.0
    # Build log-transition with a TRUE -inf on backward moves so no emission,
    # however confident, can ever drive the Viterbi path backwards.
    log_A = np.full((C, C), NEG_INF, dtype=np.float64)
    for i in range(C):
        for j in range(i, C):
            log_A[i, j] = np.log(A[i, j]) if A[i, j] > 0 else NEG_INF
    return CausalDecoder(log_A=log_A, temperature=temperature)


def decode_video_causal(decoder: CausalDecoder, logits: np.ndarray) -> np.ndarray:
    """Run a causal decoder over a single video's logits [T, C] -> preds [T]."""
    decoder.reset()
    T = logits.shape[0]
    preds = np.empty(T, dtype=np.int64)
    for t in range(T):
        preds[t] = decoder.update(logits[t])
    return preds


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray,
                    grid: Optional[np.ndarray] = None) -> float:
    """Grid search temperature minimizing NLL on validation set.

    Args:
        val_logits: [N, C] logits aggregated over validation frames.
        val_labels: [N] integer labels in [0, C-1]; -1 to ignore.
        grid: optional temperature grid; defaults to log-spaced 0.5..5.0.

    Returns:
        Best temperature (scalar).
    """
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 2.0, 16), np.linspace(2.1, 5.0, 15)])
    mask = val_labels >= 0
    L = val_logits[mask]; Y = val_labels[mask].astype(int)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        log_p = log_softmax(L / max(T, 1e-6))
        nll = -log_p[np.arange(len(Y)), Y].mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15) -> float:
    """Standard ECE: bin by max-prob confidence and compare to accuracy."""
    mask = labels >= 0
    p, y = probs[mask], labels[mask]
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(conf)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.sum() / N) * abs(acc_bin - conf_bin)
    return float(ece)


def maximum_calibration_error(probs: np.ndarray, labels: np.ndarray,
                              n_bins: int = 15) -> float:
    mask = labels >= 0
    p, y = probs[mask], labels[mask]
    conf = p.max(axis=1); pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        mce = max(mce, abs(correct[m].mean() - conf[m].mean()))
    return float(mce)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    mask = labels >= 0
    p, y = probs[mask], labels[mask].astype(int)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.log(p[np.arange(len(y)), y]).mean())
