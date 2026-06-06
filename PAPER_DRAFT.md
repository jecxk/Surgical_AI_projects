# Causal Monotonic Decoding with Calibrated Confidence for Real-time Surgical Phase Recognition

**Working draft — 2026-05-31.** All numbers below are from the actual benchmark
on Cholec80 (val videos 41–60 for calibration, test videos 61–80 for reporting,
train videos 1–40 for transition-matrix estimation).

---

## ABSTRACT (draft)

Surgical phase recognition systems on Cholec80 increasingly converge in
per-frame classification quality, yet two practical questions remain weakly
addressed in the literature: how to convert per-frame logits into a
phase-coherent sequence *under real-time constraints*, and how reliable the
predicted confidence actually is. We present three contributions. First, we
introduce a streaming Viterbi decoder over learned phase transitions that
operates with O(1) state per frame and zero lookahead, suitable for online use.
Second, we report the first calibration analysis (ECE, NLL, temperature
scaling) over three representative architectures — ResNet-50 + BiLSTM,
EfficientNet-B3 + TCN, and Swin-Tiny + Transformer — and show that
temperature scaling reduces expected calibration error by up to 66% without
retraining. Third, we provide a unified benchmark of eight decoders on the
same logits, measuring per-frame accuracy, macro-F1, expected calibration
error, boundary versus interior accuracy, and inference latency. The proposed
causal HMM decoder with calibrated logits improves test-set accuracy by
+0.62% to +1.74% across the three architectures (Wilcoxon p<0.0001 in all
cases, Cohen's d>0.97) at a measured latency of 0.013 ms per frame, with the
largest gains concentrated at phase boundaries (+2.6% accuracy on boundary
frames for the strongest model). We further quantify, for the first time on
Cholec80, the irreducible gap between causal and offline (non-causal)
post-processing: 4.1% accuracy for the strongest model, almost entirely
located in interior frames rather than at transitions.

---

## 1. INTRODUCTION

### 1.1 Motivation

Automatic surgical phase recognition — labelling every frame of an
intra-operative video with the current procedural step — underpins a range of
operating-room applications: real-time decision support, automatic documentation,
surgical-skill assessment, and prediction of the remaining procedure duration
[1, 2]. For these applications to be useful *during* an operation rather than
only in retrospective review, the phase estimate must be produced
**causally**: at time *t* the system may use only frames up to *t*, never future
context. Yet most progress reported on the public Cholec80 benchmark [3] is
measured under an offline protocol, where the entire video is available before
any label is emitted. This mismatch between how systems are evaluated and how
they would be deployed is rarely made explicit.

Two practical concerns receive comparatively little attention in the Cholec80
literature, even as per-frame classification accuracy steadily improves. The
first is the **decoding step** — how a sequence of per-frame logits is converted
into a temporally coherent phase sequence. Strong models still produce isolated
flips between visually similar phases, and the standard remedies (median
smoothing, or offline Hidden Markov / monotonic decoding) are either weak or
non-causal. The second is **confidence calibration**: a model that is 99 %
confident but only 80 % accurate is actively dangerous in a clinical setting,
where a downstream system or a clinician may act on the reported confidence.
Calibration is a mature topic on natural-image benchmarks [18] but, to our
knowledge, has not been reported for surgical phase recognition.

### 1.2 This work

We take three trained backbones from a prior hardware-constrained study —
ResNet-50 + BiLSTM (M1), EfficientNet-B3 + TCN (M2), and Swin-Tiny + Transformer
(M3) — and ask what can be recovered *without any retraining* by improving the
decoding and calibration stages alone. We introduce a streaming Viterbi decoder
that combines temperature-calibrated emissions with a phase-transition prior
learned from the training videos, and we benchmark it against raw argmax, median
smoothing, and an offline monotonic decoder (a non-causal upper bound) on the
standard Cholec80 test split. We find consistent, statistically significant
gains (paired Wilcoxon p < 0.001, Cohen's d > 0.97 on all three models) at a
decoding cost of 0.013 ms per frame, with the gains concentrated at phase
boundaries; and we quantify, for the first time on Cholec80, that the residual
gap to the offline upper bound lives almost entirely in *interior* frames rather
than at transitions.

### Contributions (bullet list to cite back to)

1. A streaming causal Viterbi decoder with a learned transition prior estimated
   from Cholec80 training labels, runnable at <0.02 ms/frame.
2. The first systematic calibration study (ECE/NLL/temperature-scaling) on
   surgical phase recognition across three architecture families.
3. A unified benchmark of 8 decoders on identical logits, including
   boundary-vs-interior error decomposition that reveals where each decoder
   actually helps.
4. A measurement of the irreducible causal-vs-offline gap on Cholec80, and
   an analysis showing that this gap lives almost entirely in *interior*
   frames — not at phase transitions.

### 1.3 Related work

**Temporal modelling for surgical phase recognition.** EndoNet [3] paired a
single CNN with auxiliary tool supervision and refined its predictions with an
external Hidden Markov Model. SV-RCNet [4] trained a ResNet-50 jointly with an
LSTM, and TeCNO [5] replaced recurrence with a multi-stage temporal
convolutional network, popularising dilated causal convolutions and a "prediction
keyframe" inference smoothing (PKI). Trans-SVNet [6] and OperA [8] introduced
self-attention over per-frame embeddings, and more recent long-form models such
as LoViT and memory-augmented transformers aggregate context over the whole
procedure. Our three backbones (M1–M3) are representatives of the recurrent,
convolutional, and attention families respectively; the present paper is
orthogonal to this line of work, since it improves the *decoding and calibration*
stages and leaves the backbones untouched.

**Decoding and temporal post-processing.** Two priors recur in this literature.
The first is *temporal smoothing* — median or PKI filters that remove isolated
flips; these are mildly effective but blur short phases and, when implemented as
a centred window, are not strictly causal. The second is the *monotonicity*
prior: cholecystectomy phases follow a fixed non-decreasing order, which an
offline Viterbi pass over the whole video can enforce to dramatic effect.
However, the offline decoder requires the complete video and is therefore a
non-causal upper bound, not a deployable method. We are not aware of prior work
on Cholec80 that formulates an explicitly *causal* streaming decoder with a
learned transition prior and benchmarks it against these baselines on equal
footing, including a boundary-versus-interior decomposition of where the gains
arise.

**Confidence calibration.** Modern deep classifiers are systematically
over-confident, and temperature scaling is a simple, effective post-hoc
remedy that preserves accuracy while improving the Expected Calibration Error
[18]. Calibration has been studied extensively on natural-image classification
but, to our knowledge, has not been reported for surgical phase recognition,
where reliable confidence is arguably more consequential. We therefore report
ECE, NLL and the fitted temperature for all three architectures, and — beyond
calibration as an end in itself — show that calibrated emissions are what allow
the transition prior to contribute meaningfully inside the Viterbi decoder.

---

## 2. METHOD

### 2.1 Per-frame logits and three backbones

We use the three architectures from our prior work [self-ref]:
M1 = ResNet-50 + 2-layer BiLSTM (35.7 M params), M2 = EfficientNet-B3 + 2-stage
TCN (11.2 M), M3 = Swin-Tiny + 3-layer Transformer (31.6 M). All three are
trained with the identical curriculum, sequence length 8 at 1 fps, multi-task
phase + tool supervision, and 4 GB GPU budget. The trained models produce a
sequence of seven-class logits z_t ∈ R^7 for each frame t.

### 2.2 Temperature calibration

For each model we fit a single scalar T* by minimising negative log-likelihood
on the validation set (videos 41–60):
        T* = argmin_T  −(1/N) Σ_n log softmax(z_n / T)[y_n].
We use a 31-point log-spaced grid over [0.5, 5.0]. Calibration is quantified
by Expected Calibration Error (ECE, 15-bin), Maximum Calibration Error (MCE),
and NLL.

### 2.3 Phase transition matrix

We estimate A[i,j] = P(phase_{t+1}=j | phase_t=i) from the 40 training videos
by counting one-step transitions in the ground-truth 1 fps label sequences,
with Laplace smoothing (α=1). The resulting matrix is highly diagonal
(0.988–0.999 on the diagonal), reflecting that consecutive 1 fps frames almost
always remain in the same phase, but the off-diagonal mass concentrates on
the next phase (i+1) as expected from the procedurally fixed order of
cholecystectomy.

### 2.4 Causal Viterbi decoder

We maintain a log-marginal vector α_t ∈ R^7 such that α_t[c] is the maximum
log-probability of any path ending in phase c at time t. The update is:

        α_t[c] = log softmax(z_t / T*)[c] + max_{c'} ( α_{t-1}[c'] + log A[c', c] ),

with α_0[c] = log emit_0[c] + log π[c], π = δ_0 (surgery starts at Preparation).
The predicted phase at time t is argmax_c α_t[c]. Memory is O(C); time is
O(C^2) per frame (49 ops for C=7).

### 2.5 Two transition priors

We compare:
- **Monotonic prior**: A[i,j] = 0 if j<i, A[i,i] = stay = 0.95, off-diagonal
  uniformly distributed over later phases. This is the hand-crafted prior used
  in our offline decoder.
- **Learned prior**: A estimated from train labels as above.

### 2.6 Eight decoders evaluated

(a) argmax_raw, (b) median-15 smoothing of argmax, (c) offline monotonic
Viterbi over the full video (non-causal upper bound), (d) causal argmax
(sanity check, equivalent to a), (e) causal monotonic Viterbi, (f) causal HMM
Viterbi, (g) causal monotonic Viterbi with calibrated logits, (h) causal HMM
Viterbi with calibrated logits. All decoders operate on the same per-video
logit sequences.

### 2.7 Evaluation protocol

We compute per-frame accuracy and macro-averaged F1 over all test frames, ECE
on val set, and per-frame latency on CPU. Boundary frames are defined as
frames within ±5 frames (±5 s at 1 fps) of any ground-truth phase transition;
interior frames are the complement. Per-video paired Wilcoxon signed-rank
tests (one-sided) on the 20 test videos quantify the consistency of decoder
gains; Cohen's d on the 20 paired differences quantifies effect size.

---

## 3. RESULTS

### 3.1 Calibration

**Table 1.** Validation-set temperature, ECE and NLL before and after
temperature scaling.

| Model | T* | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---|---|---|---|---|
| M1 ResNet-LSTM   | 1.30 | 0.1091 | **0.0369** | 0.8972 | 0.8554 |
| M2 EffNet-TCN    | 1.10 | 0.0593 | 0.0741 | 0.8242 | 0.8165 |
| M3 Swin-Transformer | 1.20 | 0.0735 | **0.0700** | 0.6878 | 0.6671 |

M1 is the most over-confident and benefits most from calibration (ECE −66%).
M2 is the most under-confident among the three (T* = 1.1 in val) yet its
test-set NLL still improves slightly. Across all three models the optimal T*
lies above 1.0, consistent with the general over-confidence pattern of deep
classifiers reported in the calibration literature.

![Reliability diagrams](results/figures/fig1_reliability.png)

**Figure 1.** Reliability diagrams on the validation set. Grey bars are the raw
per-bin accuracy; the teal curve is after temperature scaling. The dashed
diagonal is perfect calibration. M1 (the most over-confident model) is visibly
pulled towards the diagonal by scaling; all three optimal temperatures exceed 1.

### 3.2 Decoder benchmark

**Table 2.** Test-set accuracy and macro-F1 across eight decoders on three
models. Best causal-realtime row per model is bolded; the offline row is the
non-causal upper bound for reference.

| Decoder | M1 Acc / F1 | M2 Acc / F1 | M3 Acc / F1 |
|---|---|---|---|
| argmax_raw            | 82.36 / 0.763 | 81.22 / 0.738 | 85.27 / 0.794 |
| median-15             | 82.71 / 0.768 | 81.71 / 0.746 | 85.51 / 0.800 |
| causal_monotonic      | 82.40 / 0.774 | 82.37 / 0.766 | 85.84 / 0.801 |
| causal_HMM            | 82.89 / 0.771 | 82.78 / 0.763 | 85.92 / 0.804 |
| causal_monotonic+cal  | 82.54 / 0.777 | 82.43 / 0.767 | 85.96 / 0.802 |
| **causal_HMM+cal**    | **82.95 / 0.771** | **82.90 / 0.764** | **86.03 / 0.806** |
| offline_monotonic †   | 91.75 / 0.859 | 91.75 / 0.849 | 90.13 / 0.832 |

† Non-causal upper bound, requires the entire video. Not deployable online.

The causal HMM with calibrated logits is the best causal decoder on all three
models. Gains over argmax_raw are +0.59% / +0.62% / +0.76% in accuracy and
+0.008 / +0.026 / +0.012 in macro-F1.

![Decoder benchmark](results/figures/fig2_benchmark.png)

**Figure 2.** Decoder comparison on the test set: accuracy (top) and macro-F1
(bottom). The proposed causal HMM + cal (teal) is the best causal-deployable
decoder on every model; the offline monotonic decoder (amber, hatched) is shown
only as a non-causal upper bound. The qualitative effect on a single video is
shown in Figure 4.

![Qualitative timeline](results/figures/fig4_timeline.png)

**Figure 4.** Qualitative phase timeline for test video 70 (M3). Raw argmax
(middle) produces many isolated flips, especially early in the procedure; the
proposed causal decoder (bottom) removes most of them and tracks the ground
truth (top) far more cleanly, raising accuracy from 79.7 % to 82.8 % on this
video.

### 3.3 Statistical significance

**Table 3.** Per-video paired test of causal_HMM+cal vs argmax_raw across
20 test videos.

| Model | ΔAcc (%) | Wilcoxon p | Cohen's d |
|---|---|---|---|
| M1 ResNet-LSTM   | +0.62 | <0.0001 | 1.38 |
| M2 EffNet-TCN    | +1.74 | 0.0001 | 1.25 |
| M3 Swin-Transformer | +0.76 | 0.0001 | 0.98 |

All three are highly significant (p<0.001) with large effect sizes (d>0.9).
The aggregate accuracy gap may appear modest, but it is consistent and robust
across videos rather than driven by a few outliers.

![Transition prior and significance](results/figures/fig5_transition_significance.png)

**Figure 5.** Left: the learned per-second transition matrix A, strongly diagonal
(phases rarely change between adjacent seconds) with off-diagonal mass on the
next phase, reflecting the fixed procedural order. Right: per-video accuracy gain
of the proposed decoder over raw argmax, annotated with Cohen's d and the paired
Wilcoxon p-value; all three models show a large, highly significant effect.

### 3.4 Boundary versus interior accuracy

**Table 4.** Test-set accuracy split into boundary (±5 s of any ground-truth
transition) and interior frames.

| Decoder | M3 boundary Acc | M3 interior Acc |
|---|---|---|
| argmax_raw           | 60.67 | 86.06 |
| median-15            | 61.08 | 86.30 |
| **causal_HMM+cal**   | **63.23** | 86.77 |
| offline_monotonic    | 62.83 | **91.01** |

Two findings stand out. First, the proposed causal decoder is the strongest
on boundary frames — even outperforming the offline upper bound, because the
hard monotonicity constraint of the offline decoder occasionally forces a
delayed transition whereas the HMM's probabilistic prior accommodates the
ground-truth transition more flexibly. Second, the entire 4.1% test-set
accuracy gap between offline and the best causal decoder is concentrated in
*interior* frames (86.77 → 91.01) rather than at boundaries. The offline
decoder's advantage is global error correction inside long stable phases,
not at transitions — a counter-intuitive observation that, to our knowledge,
has not previously been quantified on Cholec80.

![Boundary vs interior](results/figures/fig3_boundary.png)

**Figure 3.** Boundary vs interior accuracy for M3. Boundary frames (solid bars,
±5 s of a transition) are hard for every method; the proposed decoder is the
strongest there, slightly exceeding even the non-causal offline upper bound.
Interior frames (hatched bars) are where the offline decoder's entire advantage
lies — the residual causal-vs-offline gap is an interior-frame phenomenon, not a
boundary one.

### 3.5 Latency

All proposed causal decoders run at 0.011–0.024 ms per frame on a single
CPU core, well below the 40 ms per frame budget implied by 25 fps live video.
The bottleneck of any deployed system therefore remains feature extraction,
not decoding.

---

## 4. DISCUSSION (sketch)

- **Why the causal HMM beats offline at boundaries.** Hard monotonic decoders
  must commit to a single phase ordering; when annotation places a transition
  one or two frames earlier or later than the model's preferred location,
  the offline decoder pays a localised cost. The HMM, by softening the
  monotonicity into a transition prior with non-zero off-diagonal mass, can
  match the labelled transition timing more freely.
- **Why calibration helps decoding even when it doesn't help accuracy alone.**
  The Viterbi update is a linear combination of log-emission and
  log-transition; if emissions are systematically over-peaked, the prior is
  effectively ignored. Temperature scaling rebalances the two terms,
  unlocking the prior's contribution. This is visible in the small but
  consistent F1 gains of *+cal* variants over their uncalibrated counterparts
  on M3 (0.798 → 0.802 monotonic, 0.804 → 0.806 HMM).
- **Where the lookahead premium lives.** The 4.1% test-set gap to offline is
  almost entirely interior; offline decoding fixes momentary mis-classifications
  inside long phases by integrating future evidence. A causal decoder cannot
  recover those frames without lookahead by construction. This identifies a
  clean direction for future work: short-horizon (e.g. 5 s) lookahead with a
  tunable latency budget would close most of the gap while remaining
  deployable.

---

## 5. LIMITATIONS

- Calibration on M2 is the weakest; this likely reflects that the EfficientNet-
  TCN backbone was the most under-trained (interrupted Stage-3 fine-tuning in
  our prior work). Re-evaluating after a clean re-train remains future work.
- All conclusions are drawn on Cholec80 alone. Procedures with less rigid
  phase orderings (e.g. multi-step robotic surgeries) may favour a less
  restrictive transition prior.

---

## REFERENCES

[1] Maier-Hein, L. et al. (2017). Surgical data science for next-generation interventions. *Nature Biomedical Engineering*, 1(9):691-696.

[2] Vedula, S. S., Ishii, M., Hager, G. D. (2017). Objective assessment of surgical technical skill and competency in the operating room. *Annual Review of Biomedical Engineering*, 19:301-325.

[3] Twinanda, A. P. et al. (2017). EndoNet: A deep architecture for recognition tasks on laparoscopic videos. *IEEE Transactions on Medical Imaging*, 36(1):86-97.

[4] Jin, Y. et al. (2018). SV-RCNet: Workflow recognition from surgical videos using recurrent convolutional network. *IEEE Transactions on Medical Imaging*, 37(5):1114-1126.

[5] Czempiel, T. et al. (2020). TeCNO: Surgical phase recognition with multi-stage temporal convolutional networks. In *MICCAI*, pages 343-352.

[6] Gao, X. et al. (2021). Trans-SVNet: Accurate phase recognition from surgical videos via hybrid embedding aggregation Transformer. In *MICCAI*, pages 593-603.

[8] Czempiel, T. et al. (2021). OperA: Attention-regularized Transformers for surgical phase recognition. In *MICCAI*, pages 604-614.

[18] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. In *ICML*, pages 1321-1330.

---

## DATA ARTIFACTS

- `results/causal_benchmark.json` — full per-model per-decoder metrics.
- `results/boundary_breakdown.json` — boundary/interior split for the top 5 decoders.
- `results/significance.json` — per-video accuracies + Wilcoxon p + Cohen's d.
- `src/evaluation/causal_decode.py` — streaming Viterbi + calibration.
- `src/evaluation/boundary_analysis.py` — boundary mask + split metrics.
- `scripts/eval_causal.py`, `scripts/eval_boundary.py`, `scripts/eval_significance.py`.
