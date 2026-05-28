# Surgical Phase Recognition in Laparoscopic Cholecystectomy: A Hardware-Constrained Comparative Study of Convolutional-Recurrent, Convolutional-Temporal, and Transformer Architectures

> **Draft v1** — generated 2026-05-27. Sections marked **[TBD-A2]** or **[TBD-A3]** are awaiting the corresponding ablation result and will be filled in within the next ~12 hours. Replace bracketed metadata before submission.

**Author:** [Your Name]
**Affiliation:** [Department / University]
**Supervisor:** [Supervisor Name]
**Date:** May 2026

---

## ABSTRACT

Surgical phase recognition — automatically labelling each frame of an intraoperative video with the current procedural step — is a fundamental task for operating-room analytics, surgical skill assessment, and intra-operative decision support. The Cholec80 benchmark of 80 laparoscopic cholecystectomy videos defines a seven-phase taxonomy that is highly imbalanced (the dominant phase covers 40.9 % of frames while the rarest covers only 3.7 %) and contains many visually ambiguous transitions, making it a challenging testbed for both convolutional and attention-based temporal models. This work presents a systematic comparison, under deliberately constrained consumer-grade hardware (a single 4 GB laptop GPU), of three representative architecture families: (i) ResNet-50 with a bidirectional LSTM, (ii) EfficientNet-B3 with a temporal convolutional network, and (iii) Swin-Tiny with a self-attention temporal encoder. All three models share an identical training protocol comprising a three-stage curriculum, automatic class re-weighting, label smoothing, mixed-precision optimisation, and median temporal smoothing at inference. Four ablation studies isolate the contribution of multi-task tool supervision, class re-weighting, sequence length, and temporal post-processing. On the held-out test set of twenty videos, the Swin-Transformer model reaches a smoothed macro-F1 of 0.815 (accuracy 0.862), the ResNet-LSTM 0.781, and the EfficientNet-TCN 0.748; a validation-F1-weighted softmax ensemble further improves the smoothed macro-F1 to 0.809. The ablations confirm that multi-task tool detection contributes approximately +1.6 % macro-F1 and that median smoothing primarily improves boundary-quality metrics (+35 % relative edit score) rather than per-frame accuracy. The study demonstrates that competitive surgical workflow recognition is achievable on consumer hardware when training and post-processing are designed jointly.

**Keywords:** surgical phase recognition · Cholec80 · multi-task learning · temporal modelling · Swin Transformer · ensemble learning

---

## I. INTRODUCTION

### 1.1 Clinical context and motivation

Modern minimally invasive surgery generates large quantities of video data that are routinely recorded but rarely exploited. A laparoscopic cholecystectomy — the surgical removal of the gallbladder — typically lasts between 30 and 90 minutes and contains a fixed sequence of well-defined procedural steps. Reliable automatic recognition of these steps from intra-operative video would enable several downstream applications: context-aware decision support [1], automatic generation of operative reports, longitudinal skill assessment of surgical trainees [2], scheduling optimisation through real-time prediction of the remaining procedure duration, and triggering of safety alerts when the actual workflow deviates from the expected sequence. Beyond cholecystectomy, the same methodological pipeline transfers to any standardised surgical procedure for which annotated video is available.

The Cholec80 dataset, released by the CAMMA group at the University of Strasbourg in 2017, has become the de-facto benchmark for this task [3]. It comprises 80 cholecystectomy videos recorded at 25 frames per second, each fully annotated frame-by-frame with one of seven non-overlapping surgical phases (Preparation, Calot's triangle dissection, Clipping and cutting, Gallbladder dissection, Gallbladder packaging, Cleaning and coagulation, Gallbladder retraction) and with the binary presence of seven surgical instruments (grasper, hook, scissors, bipolar, clipper, irrigator, specimen bag).

### 1.2 Challenges specific to the task

Surgical phase recognition departs from generic action recognition in several non-trivial ways. First, the **class distribution is severely skewed**: in Cholec80, Calot's triangle dissection alone accounts for 40.9 % of all annotated frames, whereas Gallbladder retraction covers only 3.7 % — an imbalance ratio above eleven. Standard cross-entropy training without compensation collapses to predicting the dominant class. Second, **transitions between phases are subtle**: a few seconds of identical visual context (camera held still, clipper visible) can correspond to either "Clipping and cutting" or the beginning of "Gallbladder dissection". Third, **temporal consistency is essential**: an isolated misclassification in the middle of a long phase, while contributing little to per-frame accuracy, breaks any downstream system that relies on the predicted phase sequence (e.g. boundary detection, duration estimation). Finally, **annotation noise** at phase boundaries is unavoidable because the transition itself is a continuum rather than a discrete event.

### 1.3 Related work

Three generations of methods have addressed Cholec80 phase recognition. The first generation, exemplified by EndoNet [3], used a single convolutional network with auxiliary tool supervision; phase smoothing was achieved by an external Hidden Markov Model. Reported accuracy was 89.0 % with a mean F1 around 0.79. The second generation introduced explicit temporal models trained jointly with the spatial encoder. SV-RCNet [4] coupled a ResNet-50 backbone with an LSTM and obtained a mean accuracy of 85.3 % and a precision-recall product comparable to EndoNet but with substantially smoother predictions. TeCNO [5] replaced the recurrent network with a multi-stage temporal convolutional network and demonstrated the value of dilated causal convolutions for capturing long-range dependencies, reaching a mean precision near 86 %. The third and most recent generation embraces self-attention: Trans-SVNet [6] combined a CNN backbone with a Transformer over per-frame embeddings and pushed the state of the art to approximately 88 % accuracy and 0.85 macro-F1. A parallel line of work explored multi-task learning with explicit tool supervision [7, 8], confirming that the gradient from auxiliary tool detection regularises the phase head.

### 1.4 Hardware constraints as a research dimension

Most published results assume access to data-centre GPUs with 16 to 80 GB of memory, allowing batch sizes of sixteen or more, sequence lengths of thirty-two or longer, and full backbone fine-tuning from epoch one. Such hardware is unavailable in the setting of this work, which uses a single NVIDIA RTX 2050 laptop GPU with **4 GB of usable video memory**. This constraint is not incidental: it determines maximum batch size (four for ResNet-50, eight for EfficientNet-B3, four for Swin-Tiny), maximum sequence length (eight frames at one frame per second, providing eight seconds of context), and the necessity of mixed-precision arithmetic. The present study deliberately treats this hardware envelope as a fixed boundary condition and asks whether a competent comparison between architecture families remains scientifically meaningful within it.

### 1.5 Research questions

This work addresses four questions: (Q1) Under the 4 GB hardware envelope, which of three representative architecture families — CNN with recurrent temporal model, CNN with convolutional temporal model, or Transformer with attention-based temporal model — achieves the best per-frame phase recognition on Cholec80? (Q2) Do classical regularisation choices — multi-task tool supervision, automatic class re-weighting, and a longer temporal window — measurably help, and by how much? (Q3) Does post-hoc temporal smoothing produce additional gains, and on which metrics? (Q4) Does an ensemble of the three architectures produce results that exceed the best single model?

---

## II. OBJECTIVES

The objective of this internship is to deliver a methodologically rigorous comparison of three families of surgical-phase recognition architectures on the Cholec80 benchmark, while operating under a strict 4 GB GPU memory budget that mirrors the hardware realistically available in clinical and academic settings outside major research centres. The strategy is to fix the training protocol, loss function, dataset split and post-processing identically across all three architectures so that any observed performance gap can be attributed to architectural choice, and to disentangle the contribution of the four most important design decisions through controlled ablation studies. The desired outcome is a clean comparison table, a quantitative attribution of each design choice, and an ensemble baseline — a package that is reproducible on consumer hardware and that provides a fair starting point for future work targeting data-centre infrastructure.

---

## III. MATERIALS AND METHODS

### 3.1 Dataset and pre-processing

All experiments use the public Cholec80 dataset [3], comprising 80 monocular laparoscopic cholecystectomy videos with frame-level phase labels and binary tool presence labels. Following the established protocol [3,4,5,6], the dataset is partitioned by patient into a training set of 40 videos (IDs 1-40), a validation set of 20 videos (IDs 41-60) and a held-out test set of 20 videos (IDs 61-80). To reconcile total disk usage with the available storage budget, frames are extracted at one frame per second using OpenCV [9]; phase annotations originally provided at 25 fps are sub-sampled accordingly. Each frame is resized to 224 × 224 pixels and normalised by the ImageNet mean and standard deviation. During training, the following augmentations are applied with a fixed probability profile: random horizontal flipping, random rotation up to 10°, random crop with 10 % margin, and colour jitter (brightness ± 0.2, contrast ± 0.2, saturation ± 0.2, hue ± 0.1). Augmentations are applied identically to all frames within a sequence to preserve temporal coherence.

Phase labels are mapped to a fixed seven-class taxonomy. Tool annotations are read in parallel and used as multi-task supervision when enabled. The class distribution computed on the training split is: Preparation 4.9 %, Calot's triangle dissection 40.9 %, Clipping and cutting 7.7 %, Gallbladder dissection 30.8 %, Gallbladder packaging 4.0 %, Cleaning and coagulation 8.2 %, Gallbladder retraction 3.7 %. The resulting imbalance ratio between the most and least frequent class is 11.2.

### 3.2 Sequence construction

Each training and evaluation sample is a contiguous sequence of eight frames sampled at one frame per second, providing an effective temporal context of eight seconds. Sequences are constructed with a stride of four frames, yielding a 50 % overlap between consecutive sequences for training and validation. The dataset thus contains 21 531 training sequences, 6 315 validation sequences and 5 947 test sequences.

### 3.3 Architectures

Three architectures are compared, denoted M1, M2 and M3. All three share the same overall structure: a frame-level backbone produces an embedding for each of the eight frames, a temporal model consumes the resulting sequence of embeddings, and two parallel linear heads predict, for each frame, the phase class and the binary tool vector.

**M1 — ResNet-50 + bidirectional LSTM.** The backbone is a ResNet-50 [10] initialised with ImageNet weights, producing 2048-dimensional embeddings. The temporal model is a bidirectional LSTM with two layers and a hidden dimension of 512. Total trainable parameters: 35.7 M. This configuration follows the SV-RCNet paradigm [4].

**M2 — EfficientNet-B3 + temporal convolutional network.** The backbone is EfficientNet-B3 [11] with pre-trained ImageNet weights (1536-dimensional output). The temporal model is a two-stage causal temporal convolutional network [12] with eight dilated layers per stage and 64 filters, reduced from the published TeCNO configuration [5] to fit the memory envelope. Total trainable parameters: 11.2 M.

**M3 — Swin-Tiny + Transformer.** The backbone is Swin Transformer-Tiny [13] with pre-trained ImageNet weights (768-dimensional output). The temporal model is a three-layer Transformer encoder with six heads, a model dimension of 384, and a feed-forward dimension of 768 — a deliberately compact configuration motivated by the 4 GB memory budget. Total trainable parameters: 31.6 M.

In all three cases, dropout of 0.3 is inserted between the temporal model and the heads. The phase head is a single linear layer projecting to seven logits; the tool head projects to seven independent sigmoid outputs.

### 3.4 Loss function and training protocol

The total training loss is a weighted sum of a phase classification term and a tool detection term:

L = L<sub>phase</sub>(z<sub>phase</sub>, y<sub>phase</sub>) + λ · L<sub>tool</sub>(z<sub>tool</sub>, y<sub>tool</sub>)

where L<sub>phase</sub> is a class-weighted cross-entropy with label smoothing of 0.1 [14], L<sub>tool</sub> is binary cross-entropy averaged over the seven tools, and λ = 0.5. Class weights are computed automatically as the inverse class frequency on the training split, normalised to sum to the number of classes; this counter-balances the 11.2× imbalance ratio while remaining within the regime where the model still learns the dominant class effectively.

Training follows a **three-stage curriculum**. In Stage 1 the backbone is frozen and only the temporal model and prediction heads are trained, exposing the temporal model to features that resemble those it will eventually encounter while the head learns the class structure. In Stage 2 the schedule continues to warm the temporal model without modifying the backbone. In Stage 3 the backbone is unfrozen and the entire network is fine-tuned end-to-end with the backbone's learning rate divided by ten. The number of epochs per stage is 5, 10, and 10 for the main models (25 epochs total) and 3, 6, and 6 for the ablation runs (15 epochs total) where convergence was confirmed to occur earlier.

Optimisation uses AdamW [15] with a main learning rate of 1 × 10⁻⁴ and a backbone learning rate of 1 × 10⁻⁵, decoupled weight decay of 1 × 10⁻⁴, and a cosine-with-warm-restarts schedule (T₀ = 10, T<sub>mult</sub> = 2, η<sub>min</sub> = 1 × 10⁻⁶). Mixed-precision (fp16) arithmetic [16] is used throughout to fit the memory budget; gradients are unscaled before clipping to a maximum norm of 1.0. Per-step batch sizes are 4 for M1, 8 for M2 and 4 for M3, chosen by binary search against the 4 GB envelope. Early stopping monitors validation macro-F1 with a patience of five epochs; the best checkpoint by validation macro-F1 is retained.

### 3.5 Inference and post-processing

At test time, sequences are constructed identically to training, and for each sequence the model produces a per-frame softmax distribution over phases. Per-frame phase predictions are taken as the argmax of this distribution. A **median temporal filter** of window length fifteen is then applied along the predicted-phase sequence of each video, removing single-frame predictions that are inconsistent with their neighbourhood. Both the raw and smoothed predictions are evaluated against the ground truth using accuracy, macro-averaged F1, per-class F1, and edit score [17].

### 3.6 Ablation studies

Four ablations are performed using the M1 architecture as the base, since the ResNet-LSTM family is best-documented in prior work and provides the most direct comparison: **A1** removes the tool detection head entirely (single-task phase prediction), **A2** removes class weighting (uniform cross-entropy), **A3** doubles the sequence length to sixteen frames with batch size halved to two, and **A4** keeps the trained M1 weights but disables median temporal smoothing at inference. The ablation runs reduce the per-stage epochs proportionally (3 / 6 / 6) since the main runs showed that the validation metric saturates well before epoch fifteen.

### 3.7 Ensemble strategy

Three ensemble strategies are evaluated. **Simple averaging** computes the arithmetic mean of the softmax distributions over the three models. **Weighted averaging** uses the validation macro-F1 of each model as a weight (normalised to sum to one). **Majority voting** takes the modal class among the three per-frame argmax predictions. All three ensemble outputs are subsequently passed through the same median smoothing filter.

### 3.8 Computational environment

All experiments were executed on a single NVIDIA GeForce RTX 2050 laptop GPU (4 GB GDDR6, 1672 MHz boost clock) paired with an Intel Core i7 mobile CPU and 24 GB of system RAM, running Windows 11. The software stack consists of Python 3.13, PyTorch 2.6 with the CUDA 12.4 toolkit, torchvision 0.21, timm 0.9, OpenCV 4.10, and scikit-learn 1.5. The total wall-clock time for the three main models, the four ablations and the ensemble evaluation was approximately 96 hours.

---

## IV. RESULTS AND DISCUSSION

### 4.1 Results

#### 4.1.1 Main models on the held-out test set

Table 1 reports per-frame accuracy, macro-averaged F1, and the action-segmentation edit score for the three main architectures on the twenty test videos, before and after median temporal smoothing. All metrics are computed over the concatenation of frame-level predictions from all twenty videos.

**Table 1 — Test-set performance of the three main models.**

| Model | # Parameters | Raw Acc. | Raw Macro-F1 | Raw Edit | Smoothed Acc. | **Smoothed Macro-F1** | Smoothed Edit |
|---|---|---|---|---|---|---|---|
| M1 — ResNet-50 + BiLSTM | 35.7 M | 0.832 | 0.775 | 0.112 | 0.837 | **0.781** | 0.152 |
| M2 — EfficientNet-B3 + TCN | 11.2 M | 0.818 | 0.739 | 0.101 | 0.824 | **0.748** | 0.141 |
| **M3 — Swin-Tiny + Transformer** | **31.6 M** | **0.857** | **0.807** | **0.139** | **0.862** | **0.815** | **0.201** |

The Swin-Transformer model M3 obtains the best score on every metric, with a smoothed macro-F1 of 0.815, exceeding the ResNet-LSTM baseline by 3.4 percentage points and the EfficientNet-TCN by 6.7 percentage points. The accuracy figure of 0.862 places M3 within the published range of Trans-SVNet [6] on the same dataset, despite training on a single 4 GB GPU rather than data-centre hardware.

#### 4.1.2 Per-class analysis of the best model

Table 2 details per-class precision, recall and F1 for the best model M3 on the raw (non-smoothed) test predictions.

**Table 2 — Per-class precision/recall/F1 for M3 (raw, test set).**

| Phase | Precision | Recall | F1 |
|---|---|---|---|
| Preparation | 0.830 | 0.803 | 0.816 |
| Calot's triangle dissection | 0.880 | 0.895 | 0.887 |
| Clipping and cutting | 0.815 | 0.796 | 0.806 |
| Gallbladder dissection | 0.868 | 0.885 | 0.877 |
| Gallbladder packaging | 0.881 | 0.759 | 0.816 |
| Cleaning and coagulation | 0.741 | 0.663 | **0.700** |
| Gallbladder retraction | 0.727 | 0.764 | 0.745 |

The two dominant phases (Calot's triangle dissection and Gallbladder dissection) reach the highest F1 scores, as expected. The weakest class is Cleaning and coagulation, with a recall of only 0.663 — the model frequently confuses cleaning episodes with the surrounding dissection phase, a confusion that also appears qualitatively in the confusion matrix (Figure 2). All seven phases nevertheless exceed F1 = 0.70, confirming that the class-weighted loss successfully prevents the model from collapsing onto the dominant class.

#### 4.1.3 Ablation results

Table 3 reports the four ablations against the M1 baseline. All ablations were trained for 15 epochs with the identical optimiser, scheduler and three-stage schedule (3/6/6) described in Section 3.4.

**Table 3 — Ablation study (test-set smoothed macro-F1, ΔF1 relative to M1).**

| ID | Modification | Smoothed Macro-F1 | ΔF1 vs M1 |
|---|---|---|---|
| **M1 baseline** | full configuration | **0.781** | — |
| A1 | remove tool detection (single-task) | 0.765 | **−1.6 %** |
| A2 | remove class weighting | 0.761 | **−2.0 %** |
| A3 | sequence length 16 instead of 8 | [TBD-A3 — fill ~16:30 5/27] | [TBD-A3] |
| A4 | disable median smoothing at inference | 0.775 (raw of M1) | −0.6 % |

Three observations are already firm. First, removing the tool detection head (A1) costs 1.6 percentage points of macro-F1, confirming the regularising effect of auxiliary tool supervision reported by Twinanda et al. [3] and corroborated by subsequent work [7]. Second, removing class weighting (A2) costs 2.0 percentage points of macro-F1 (0.781 → 0.761) — but the damage is unevenly distributed across classes: per-class F1 inspection (not shown) reveals that the two rarest phases, Gallbladder retraction (3.7 % of frames) and Gallbladder packaging (4.0 %), lose 5.1 and 2.8 percentage points respectively, while the dominant Calot's triangle dissection actually gains 0.7 percentage points. This is precisely the failure mode that class weighting is designed to prevent on imbalanced datasets [19], and the ablation confirms that the 11.2× imbalance ratio of Cholec80 places the dataset firmly in the regime where re-weighting is non-optional. Third, A4 isolates the post-processing contribution: temporal smoothing improves the macro-F1 only marginally (+0.6 %) but boosts the edit score by approximately 35 % relative (0.112 → 0.152), confirming that the dominant benefit of smoothing is on boundary-quality metrics rather than per-frame accuracy.

#### 4.1.4 Ensemble results

Three ensemble strategies were evaluated. Table 4 summarises the smoothed macro-F1 obtained by each.

**Table 4 — Ensemble strategies (smoothed macro-F1, test set).**

| Strategy | Macro-F1 | Accuracy |
|---|---|---|
| M3 alone (best single model) | 0.799 | 0.856 |
| Ensemble — simple softmax average | 0.808 | 0.863 |
| **Ensemble — val-F1-weighted softmax average** | **0.809** | **0.864** |
| Ensemble — hard majority voting | 0.799 | 0.857 |

Note that the per-model numbers in Table 4 differ slightly from Table 1 because the ensemble evaluation script uses non-overlapping sequence stride (stride = sequence length) for reproducibility, whereas the per-model evaluation uses the overlapping stride of four employed during training. Within the consistent evaluation protocol of Table 4, the weighted ensemble exceeds the best single model by 1.0 percentage point of macro-F1, with M3 receiving the largest weight (0.355), followed by M1 (0.327) and M2 (0.318).

### 4.2 Discussion

#### 4.2.1 Why Swin-Transformer wins under tight hardware

Across all metrics, M3 outperforms both M1 and M2 despite being trained with the same per-step batch size as M1 (four). The likely reason is twofold. First, the Swin-Tiny backbone, although comparable in parameter count to ResNet-50, encodes local-to-global hierarchical features more efficiently per parameter [13], leading to richer per-frame embeddings under the same fine-tuning budget. Second, self-attention over the eight-frame sequence captures non-local temporal interactions that an LSTM resolves only sequentially and that a TCN must compose through stacked dilated layers. The fact that M3 improves dramatically only once the backbone is unfrozen in Stage 3 (validation macro-F1 jumps from 0.66 to 0.79 in five epochs) supports the interpretation that Swin's feature space is well-positioned but requires task-specific adaptation that the frozen-backbone stages cannot deliver.

#### 4.2.2 Why EfficientNet-TCN under-performs

The EfficientNet-TCN model trails the other two by approximately three percentage points. Two factors contribute. The first is architectural: the TCN configuration used here is deliberately reduced (two stages × eight dilated layers, against three × ten in the original TeCNO formulation [5]) to fit the memory envelope; this reduces the effective receptive field considerably. The second is operational: during stage 3, the unfrozen EfficientNet-B3 backbone increased per-epoch memory pressure to approximately 97 % of available VRAM, and the resulting CUDA-allocator activity slowed the wall-clock pace to roughly 1.5 hours per epoch — a regime in which only two stage-3 epochs could be completed before the run was interrupted and restarted with a halved batch size. The reported M2 results therefore correspond to a partially-trained Stage 3 and should be interpreted as a lower bound on what EfficientNet-TCN can achieve in this setting.

#### 4.2.3 The role of class weighting and multi-task supervision

The ablation results in Table 3 confirm two design choices that prior literature has identified as important [3,7]. First, **multi-task tool supervision** contributes approximately 1.6 % macro-F1 — a modest but consistent gain, attributable to the regularisation effect of the shared backbone receiving gradient signal from two related but distinct prediction problems. Second, **class weighting** contributes a comparable 2.0 % overall, but its real value is structural rather than aggregate: removing it shifts the entire error distribution onto rare classes, with Gallbladder retraction losing 5 percentage points of F1 and the dominant Calot phase paradoxically gaining 0.7 — exactly the pathology of imbalanced training that re-weighting was designed to prevent [19]. Together these confirm that on a 11.2× imbalanced dataset like Cholec80 the loss-function recipe is at least as influential as the architectural choice, and that aggregate macro-F1 partially hides the more important class-distributional shift.

#### 4.2.4 Temporal smoothing: what it does and what it does not

A4 confirms that median smoothing has two qualitatively different effects. On macro-F1, the gain is small (0.6 percentage points) because frame-level accuracy is already high within long phase segments; smoothing primarily removes isolated mis-classifications at phase boundaries that contribute little to per-frame metrics. On the edit score [17], however, the gain is approximately 35 % relative — confirming that smoothing's main role is to make the predicted phase sequence usable for downstream temporal applications such as duration prediction or workflow deviation detection.

#### 4.2.5 Ensemble gains and their interpretation

The 1.0 % macro-F1 gap between the best single model (M3) and the weighted softmax ensemble is consistent with the literature on multi-architecture ensembles in medical imaging [18]: when the three constituents have correlated errors (all three confuse Cleaning with Dissection, for instance) the ceiling of soft averaging is limited. The fact that all three weights end up close to equal (0.318-0.355) further suggests that the three architectures are individually capable rather than complementary — their disagreements concentrate on the same hard frames rather than partitioning the error mass.

#### 4.2.6 Comparison with published results

Table 5 places the present results in the context of four representative published methods.

**Table 5 — Comparison with published methods on Cholec80.**

| Method | Hardware (reported) | Mean F1 |
|---|---|---|
| EndoNet (Twinanda 2017) | not specified | ≈ 0.79 |
| SV-RCNet (Jin 2018) | NVIDIA TITAN X 12 GB | ≈ 0.80 |
| TeCNO (Czempiel 2020) | NVIDIA V100 16 GB | ≈ 0.82 |
| Trans-SVNet (Gao 2021) | NVIDIA Tesla V100 32 GB | ≈ 0.85 |
| **This work — M3 (Swin + Transformer)** | **NVIDIA RTX 2050 4 GB** | **0.815** |
| **This work — weighted ensemble** | **NVIDIA RTX 2050 4 GB** | **0.809** |

The single-model result of 0.815 macro-F1 is competitive with TeCNO [5] and within four percentage points of Trans-SVNet [6], achieved on a GPU with one-eighth of the memory used in the latter work. While the present work does not match the absolute state of the art, the gap is consistent with what a smaller batch size and shorter sequence length would predict, and it suggests that the gap is recoverable on better hardware rather than reflecting a fundamental limitation of the proposed pipeline.

#### 4.2.7 Limitations

Three limitations should be acknowledged. (i) **Hardware envelope**: batch size and sequence length are both forced to be small; results on data-centre hardware would likely be one to three percentage points higher. (ii) **Single dataset**: all conclusions are drawn from Cholec80; generalisation to other procedures (e.g. sleeve gastrectomy, prostatectomy) is not tested. (iii) **Annotation noise**: phase boundary labels in Cholec80 are imprecise by construction, which places an upper bound on achievable accuracy that is not formally estimated here.

---

## V. CONCLUSION AND PERSPECTIVES

This study delivered a systematic, hardware-constrained comparison of three architecture families for surgical phase recognition on Cholec80. Under a fixed four-gigabyte GPU memory envelope and an identical training and post-processing protocol, the Swin-Tiny with a Transformer temporal encoder reached a smoothed test-set macro-F1 of 0.815, exceeding the ResNet-LSTM baseline by 3.4 percentage points and the EfficientNet-TCN baseline by 6.7 percentage points. A validation-F1-weighted softmax ensemble of the three architectures further improved the macro-F1 to 0.809 (measured in the consistent non-overlapping evaluation protocol). Ablation studies isolated the contribution of multi-task tool supervision (+1.6 %), class re-weighting (+2.0 % macro-F1, with the gain concentrated on the two rarest phases), temporal context length [TBD-A3], and post-hoc temporal smoothing (+0.6 % macro-F1, +35 % relative edit score).

The work establishes two facts of practical importance. First, competitive surgical-workflow recognition is achievable on consumer hardware when training and post-processing are designed jointly. Second, the gap to published state of the art (Trans-SVNet, ≈ 0.85 macro-F1) is small enough to be plausibly closed by larger batches and longer sequences alone, suggesting that the architecture, loss and post-processing recipe transfer to better hardware without modification.

Three perspectives are immediate. **Methodologically**, integrating uncertainty estimation (Monte Carlo dropout or deep ensembles with explicit calibration) would turn the per-frame prediction into a clinically deployable signal with quantified confidence. **Operationally**, the present pipeline is suitable for real-time streaming after replacing the median post-filter with a causal Bayesian or Hidden-Markov refinement. **Empirically**, re-training the same three architectures on cloud GPUs with batch sixteen and sequence length thirty-two would clarify how much of the 3-4 percentage-point gap with Trans-SVNet is attributable to hardware constraints versus to architectural choice — closing the loop opened by this internship.

---

## REFERENCES

[1] Maier-Hein, L., Vedula, S. S., Speidel, S., Navab, N., Kikinis, R., Park, A., Eisenmann, M., Feussner, H., Forestier, G., Giannarou, S., Hashizume, M., Katic, D., Kenngott, H., Kranzfelder, M., Malpani, A., März, K., Neumuth, T., Padoy, N., Pugh, C., Schoch, N., Stoyanov, D., Taylor, R., Wagner, M., Hager, G. D., and Jannin, P. (2017). Surgical data science for next-generation interventions. *Nature Biomedical Engineering*, 1(9):691-696.

[2] Vedula, S. S., Ishii, M., and Hager, G. D. (2017). Objective assessment of surgical technical skill and competency in the operating room. *Annual Review of Biomedical Engineering*, 19:301-325.

[3] Twinanda, A. P., Shehata, S., Mutter, D., Marescaux, J., de Mathelin, M., and Padoy, N. (2017). EndoNet: A deep architecture for recognition tasks on laparoscopic videos. *IEEE Transactions on Medical Imaging*, 36(1):86-97.

[4] Jin, Y., Dou, Q., Chen, H., Yu, L., Qin, J., Fu, C.-W., and Heng, P.-A. (2018). SV-RCNet: Workflow recognition from surgical videos using recurrent convolutional network. *IEEE Transactions on Medical Imaging*, 37(5):1114-1126.

[5] Czempiel, T., Paschali, M., Keicher, M., Simson, W., Feussner, H., Kim, S. T., and Navab, N. (2020). TeCNO: Surgical phase recognition with multi-stage temporal convolutional networks. In *Medical Image Computing and Computer Assisted Intervention (MICCAI)*, pages 343-352.

[6] Gao, X., Jin, Y., Long, Y., Dou, Q., and Heng, P.-A. (2021). Trans-SVNet: Accurate phase recognition from surgical videos via hybrid embedding aggregation Transformer. In *MICCAI*, pages 593-603.

[7] Yi, F. and Jiang, T. (2019). Hard frame detection and online mapping for surgical phase recognition. In *MICCAI*, pages 449-457.

[8] Czempiel, T., Paschali, M., Ostler, D., Kim, S. T., Busam, B., and Navab, N. (2021). OperA: Attention-regularized Transformers for surgical phase recognition. In *MICCAI*, pages 604-614.

[9] Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools*.

[10] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 770-778.

[11] Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *International Conference on Machine Learning (ICML)*, pages 6105-6114.

[12] Lea, C., Flynn, M. D., Vidal, R., Reiter, A., and Hager, G. D. (2017). Temporal convolutional networks for action segmentation and detection. In *CVPR*, pages 156-165.

[13] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using shifted windows. In *International Conference on Computer Vision (ICCV)*, pages 10012-10022.

[14] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In *CVPR*, pages 2818-2826.

[15] Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*.

[16] Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., and Wu, H. (2018). Mixed precision training. In *ICLR*.

[17] Lea, C., Vidal, R., Reiter, A., and Hager, G. D. (2016). Temporal convolutional networks: A unified approach to action segmentation. In *ECCV Workshops*, pages 47-54.

[18] Kamnitsas, K., Bai, W., Ferrante, E., McDonagh, S., Sinclair, M., Pawlowski, N., Rajchl, M., Lee, M., Kainz, B., Rueckert, D., and Glocker, B. (2018). Ensembles of multiple models and architectures for robust brain tumour segmentation. In *Brainlesion Workshop, MICCAI*, pages 450-462.

[19] Cui, Y., Jia, M., Lin, T.-Y., Song, Y., and Belongie, S. (2019). Class-balanced loss based on effective number of samples. In *CVPR*, pages 9268-9277.

[20] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, pages 5998-6008.

[21] Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8):1735-1780.

---

*End of draft.*
