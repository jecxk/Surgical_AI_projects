<div align="center" style="text-align:center;">
<p style="font-size:15pt; font-weight:bold; margin-top:40px;">UNIVERSITY OF SCIENCE AND TECHNOLOGY OF HANOI</p>
<p style="font-size:13pt; font-weight:bold;">DEPARTMENT OF INFORMATION AND COMMUNICATION TECHNOLOGY</p>
<p style="font-size:11pt; color:#555; margin-top:8px;">VIETNAM – FRANCE UNIVERSITY</p>
<br><br><br><br>
<p style="font-size:18pt; font-weight:bold; margin-top:48px;">BACHELOR'S THESIS REPORT</p>
<br>
<p style="font-size:15pt; font-weight:bold; max-width:560px; margin:0 auto;">A Controlled Comparison of Convolutional, Recurrent, and Transformer Architectures for Surgical Phase Recognition on Cholec80</p>
<br><br><br><br>
<table style="margin:32px auto; border:none;">
<tr><td style="border:none; text-align:right; font-weight:bold; padding:4px 12px;">Submitted by:</td><td style="border:none; text-align:left; padding:4px 12px;">Nguyen Trong Du</td></tr>
<tr><td style="border:none; text-align:right; font-weight:bold; padding:4px 12px;">Supervisor:</td><td style="border:none; text-align:left; padding:4px 12px;">Dr. Vu Trong Sinh</td></tr>
</table>
<br><br><br><br>
<p style="font-size:12pt; margin-top:48px;">June 2026</p>
</div>

<div style="page-break-after: always;"></div>

## Acknowledgements

First and foremost, I would like to express my sincere gratitude to my supervisor, Dr. Vu Trong Sinh, for his invaluable guidance, continuous support, and deep knowledge of medical imaging. His practical insights and constructive feedback shaped the direction of this work and substantially improved its quality.

I am grateful to the CAMMA research group at the University of Strasbourg for releasing the Cholec80 dataset, without which this study would not have been possible, and to the open-source community behind PyTorch, timm, and scikit-learn, whose tools underpin every experiment reported here.

Finally, I want to thank my family for their unconditional love and unwavering belief in me throughout the long nights of training and debugging. I dedicate this milestone to them.

<div style="page-break-after: always;"></div>

## Abstract

Surgical phase recognition — labelling each frame of an intra-operative video with the procedural step taking place — is a foundational task for operating-room analytics, surgical-skill assessment, and intra-operative decision support. The Cholec80 benchmark of eighty laparoscopic cholecystectomy videos defines a seven-phase taxonomy that is severely imbalanced (the dominant phase covers 42.7 % of frames while the rarest covers only 3.8 %, an 11:1 ratio) and contains many visually ambiguous transitions, making it a demanding test of both convolutional and attention-based temporal models.

This thesis makes two connected contributions. First, it presents a **controlled architecture comparison**: three representative model families — ResNet-50 with a bidirectional LSTM, EfficientNet-B3 with a temporal convolutional network, and Swin-Tiny with a self-attention temporal encoder — are trained under a single identical pipeline (three-stage curriculum, automatic class re-weighting, multi-task tool supervision, label smoothing, and mixed-precision optimisation) so that any performance gap reflects architecture alone. On the held-out test set of twenty videos, the Swin-Transformer model reaches a smoothed macro-F1 of **0.815** (accuracy 0.862), exceeding the ResNet-LSTM baseline by 3.4 points and the EfficientNet-TCN by 6.7 points; a validation-F1-weighted ensemble adds a further small gain. Four controlled ablations attribute roughly +1.6 % macro-F1 to multi-task tool supervision and +2.0 % to class re-weighting — together comparable in magnitude to the gap between the best and worst architecture.

Second, the thesis addresses the **decoding and calibration stages** that prior Cholec80 work largely overlooks. A streaming causal Viterbi decoder with a learned phase-transition prior and temperature-scaled emissions is introduced; without any retraining it improves test-set accuracy by +0.6 to +1.7 percentage points across the three models (paired Wilcoxon *p* < 0.001, Cohen's *d* > 0.97) at a decoding cost of 0.013 ms per frame, with the largest gains concentrated at phase boundaries. The residual gap to a non-causal offline upper bound is shown to lie almost entirely in interior frames rather than at transitions — a finding, to our knowledge, not previously quantified on Cholec80.

The work demonstrates that competitive surgical-workflow recognition is achievable on consumer-grade hardware when training, decoding, and post-processing are designed jointly, and that the loss-function recipe is at least as influential as the architecture on a heavily imbalanced dataset.

**Keywords:** surgical phase recognition · Cholec80 · architecture comparison · multi-task learning · causal decoding · confidence calibration · ablation study

<div style="page-break-after: always;"></div>

## Contents

1. **Introduction**
   1.1 Background and Motivation
   1.2 Problem Statement
   1.3 Objectives
   1.4 Thesis Structure
2. **Literature Review**
   2.1 Surgical Phase Recognition with Deep Learning
   2.2 Evolution of Temporal Architectures
   2.3 Handling Class Imbalance in Surgical Datasets
   2.4 Decoding and Temporal Post-processing
   2.5 Confidence Calibration in Medical AI
   2.6 Positioning This Work
3. **Methodology**
   3.1 Dataset: Cholec80
   3.2 Model Architectures
   3.3 Training Pipeline
   3.4 Evaluation Metrics
   3.5 Causal Decoding and Calibration
   3.6 Ablation and Ensemble Design
4. **Experiments and Results**
   4.1 Individual Model Comparison
   4.2 Ablation Study
   4.3 Ensemble Results
   4.4 Causal Decoding and Calibration
   4.5 Boundary-versus-Interior Analysis
5. **Conclusion and Future Work**
   5.1 Summary of Contributions
   5.2 Key Findings
   5.3 Limitations
   5.4 Future Work

<div style="page-break-after: always;"></div>

## List of Abbreviations

| | |
|---|---|
| **AdamW** | Adam with decoupled Weight decay |
| **AUC** | Area Under the ROC Curve |
| **BiLSTM** | Bidirectional Long Short-Term Memory |
| **CNN** | Convolutional Neural Network |
| **ECE** | Expected Calibration Error |
| **fps** | Frames Per Second |
| **HMM** | Hidden Markov Model |
| **LSTM** | Long Short-Term Memory |
| **MICCAI** | Medical Image Computing and Computer Assisted Intervention |
| **NLL** | Negative Log-Likelihood |
| **TCN** | Temporal Convolutional Network |
| **ViT** | Vision Transformer |

<div style="page-break-after: always;"></div>

# Chapter 1 — Introduction

## 1.1 Background and Motivation

Modern minimally invasive surgery generates large quantities of video that are routinely recorded but rarely analysed. A laparoscopic cholecystectomy — the surgical removal of the gallbladder — typically lasts between thirty and ninety minutes and follows a fixed sequence of well-defined procedural steps. Reliable automatic recognition of these steps from intra-operative video would unlock several downstream applications: context-aware decision support, automatic generation of operative reports, longitudinal skill assessment of surgical trainees, real-time prediction of the remaining procedure duration for scheduling, and safety alerts when the observed workflow deviates from the expected one [1, 2]. The same methodological pipeline transfers to any standardised procedure for which annotated video is available.

The Cholec80 dataset, released by the CAMMA group at the University of Strasbourg in 2017, has become the de-facto benchmark for this task [3]. It comprises eighty cholecystectomy videos recorded at 25 frames per second, each annotated frame-by-frame with one of seven non-overlapping surgical phases and with the binary presence of seven surgical instruments. The seven phases — Preparation, Calot's triangle dissection, Clipping and cutting, Gallbladder dissection, Gallbladder packaging, Cleaning and coagulation, and Gallbladder retraction — occur in a fixed, non-decreasing order, a structural property that this thesis later exploits at the decoding stage.

Deep learning has become the dominant approach to this problem. Early systems used a single convolutional network to label each frame independently; later work added an explicit temporal model — recurrent, convolutional, or attention-based — trained jointly with the spatial encoder. The practical obstacle, however, is rarely architectural capability. It is the data: Cholec80 is dominated by a small number of long phases, so naïve training collapses toward the majority class and reports a deceptively high frame-level accuracy while almost entirely ignoring the short, clinically important closing phases.

## 1.2 Problem Statement

Surgical phase recognition on Cholec80 is shaped by three difficulties that this thesis addresses directly.

**Class imbalance.** Calot's triangle dissection alone accounts for 42.7 % of annotated frames, whereas Gallbladder retraction covers only 3.8 % — an imbalance ratio above eleven. Cross-entropy training without compensation collapses onto the dominant phases, and frame-level accuracy hides this failure because the dominant class is itself easy.

**Confounded architecture comparisons.** Three generations of methods — CNN + LSTM, CNN + TCN, and CNN + Transformer — have each been reported as state of the art at their time, but published comparisons mix different data splits, loss functions, batch sizes, sequence lengths, and post-processing choices. It is therefore genuinely unclear how much of each reported improvement comes from the new architecture and how much from the surrounding training recipe.

**Overlooked decoding.** Most progress is measured under an offline protocol that assumes the entire video is available before any label is emitted. Real-time intra-operative use, however, requires *causal* inference — at time *t* the system may use only frames up to *t*. The step that converts per-frame predictions into a temporally coherent phase sequence, and the question of whether the reported confidences are trustworthy, both receive far less attention than the choice of backbone.

## 1.3 Objectives

To address these challenges, this thesis pursues the following objectives:

1. **Deliver a controlled architecture comparison.** Train three representative families (ResNet-50 + BiLSTM, EfficientNet-B3 + TCN, Swin-Tiny + Transformer) under a single identical pipeline so that any performance gap reflects architecture alone.
2. **Quantify the training recipe.** Through four controlled ablations, isolate the contribution of multi-task tool supervision, class re-weighting, sequence length, and temporal smoothing.
3. **Evaluate ensembling.** Test whether combining the three architectures with soft voting yields a stronger model than the best individual one.
4. **Design a causal decoder.** Introduce a streaming Viterbi decoder with a learned transition prior that runs in real time without retraining, and benchmark it against raw argmax, median smoothing, and a non-causal offline upper bound.
5. **Analyse calibration.** Report Expected Calibration Error and the effect of temperature scaling across all three architectures, and decompose model error into boundary and interior frames.

## 1.4 Thesis Structure

The remainder of this report is organised as follows. **Chapter 2** reviews related work on deep learning for surgical phase recognition, temporal architectures, class imbalance, decoding, and calibration. **Chapter 3** details the dataset, the three architectures, the shared training pipeline, the evaluation metrics, and the proposed causal decoder. **Chapter 4** presents the comparative analysis, the ablation study, the ensemble results, the causal-decoding benchmark, and the boundary-versus-interior error analysis. **Chapter 5** summarises the contributions, distils the key findings, states the limitations, and proposes future directions.

<div style="page-break-after: always;"></div>

# Chapter 2 — Literature Review

## 2.1 Surgical Phase Recognition with Deep Learning

Automatic recognition of surgical workflow from video was formalised on Cholec80 by Twinanda et al. with **EndoNet** [3], which used a single convolutional network with auxiliary tool supervision and refined its frame-level predictions with an external Hidden Markov Model. EndoNet established two ideas that persist throughout the field: that multi-task tool supervision regularises the phase head, and that some form of temporal smoothing is necessary because per-frame classifiers produce locally inconsistent predictions. Reported accuracy was close to 89 % with a mean F1 around 0.79.

Because a cholecystectomy lasts tens of minutes, the central modelling question is how to capture temporal context. The literature has answered it in three successive ways, each represented by one of the architectures compared in this thesis.

## 2.2 Evolution of Temporal Architectures

**Recurrent models.** SV-RCNet [4] coupled a ResNet-50 backbone with an LSTM trained jointly end-to-end, obtaining smoother predictions than EndoNet and a mean accuracy around 85 %. The recurrent state carries information forward through time, but it must be unrolled sequentially and struggles to retain very long-range dependencies — a limitation that matters when phases last several minutes.

**Convolutional temporal models.** TeCNO [5] replaced recurrence with a multi-stage temporal convolutional network operating on per-frame features. Dilated causal convolutions give a TCN a wide receptive field at low computational cost, and the multi-stage refinement progressively cleans up predictions. TeCNO reached a mean precision near 86 % and demonstrated that convolutional temporal modelling can match or exceed recurrent approaches.

**Attention-based models.** Trans-SVNet [6] introduced a Transformer over per-frame embeddings, allowing every frame to attend to every other frame regardless of temporal distance, and pushed the state of the art to roughly 88 % accuracy and 0.85 macro-F1. The hierarchical **Swin Transformer** [9] makes such attention tractable for images by restricting it to shifted local windows, recovering the multi-scale inductive bias of CNNs at linear complexity; it is the backbone used for the Transformer family in this work.

A recurring theme is that training methodology can dominate architectural choice. ConvNeXt [16] showed that a modernised ResNet, trained with Transformer-era recipes, matches Swin performance — implying that a fair comparison must hold the training pipeline fixed, which is precisely the design principle of this thesis.

## 2.3 Handling Class Imbalance in Surgical Datasets

Cholec80's 11:1 imbalance places it firmly in the regime where compensation is non-optional. Buda et al. [11] found that class-balanced mini-batches, achieved through oversampling or weighted sampling, consistently outperform loss reweighting and undersampling. Inverse-frequency class weighting in the loss is a complementary lever, and label smoothing [14] replaces one-hot targets with a soft distribution that prevents the model from assigning probability 1.0 to any class — improving calibration and, on structurally ambiguous datasets, macro-F1. This thesis combines automatic inverse-frequency class weighting, label smoothing, and multi-task supervision, and the ablation study in Chapter 4 quantifies the contribution of each.

## 2.4 Decoding and Temporal Post-processing

Two priors recur in the decoding literature. The first is *temporal smoothing* — median or prediction-keyframe filters that remove isolated flips; these are mildly effective but, when implemented as a centred window, are not strictly causal. The second is the *monotonicity* prior: cholecystectomy phases follow a fixed non-decreasing order, which an offline Viterbi pass over the whole video can enforce to dramatic effect. The offline decoder, however, requires the complete video and is therefore a non-causal upper bound, not a deployable method. The present work formulates an explicitly **causal** streaming decoder with a *learned* transition prior and benchmarks it against these baselines on equal footing, including a boundary-versus-interior decomposition of where the gains arise — an analysis not previously reported on Cholec80.

## 2.5 Confidence Calibration in Medical AI

A model that is 99 % confident but only 80 % accurate is dangerous in a clinical setting, where a downstream system or a clinician may act on the reported confidence. Modern deep classifiers are systematically over-confident, and temperature scaling is a simple, effective post-hoc remedy that preserves accuracy while improving the Expected Calibration Error [18]. Calibration is a mature topic on natural-image benchmarks but, to our knowledge, has not been reported for surgical phase recognition. This thesis therefore reports ECE, NLL, and the fitted temperature for all three architectures, and shows that calibrated emissions are what allow the transition prior to contribute meaningfully inside the Viterbi decoder.

## 2.6 Positioning This Work

Published Cholec80 systems reach mean F1 scores between roughly 0.79 (EndoNet) and 0.85 (Trans-SVNet), typically using data-centre GPUs, large batches, and long sequences. This work makes a different trade-off: three architectures, a fixed training set, a single 4 GB consumer GPU, and an identical pipeline — yet reaches 0.815 smoothed macro-F1 with the Transformer model, competitive with TCN-based work. The contribution is not a new state-of-the-art number; it is (i) a clean attribution of performance to architecture versus training recipe, and (ii) a real-time decoding-and-calibration layer that improves any of the trained models without retraining. The ablation study shows that class weighting and multi-task supervision together account for nearly four points of macro-F1, comparable to the architectural gap itself — evidence that, on heavily imbalanced surgical data, training-recipe quality is at least as important as the choice of backbone.

<div style="page-break-after: always;"></div>

# Chapter 3 — Methodology

This chapter details the dataset, the three model architectures, the shared training pipeline, the evaluation protocol, and the proposed causal decoder. Design decisions are explained where they are non-obvious, and hyperparameters are stated explicitly so that the experiments can be reproduced.

## 3.1 Dataset: Cholec80

All experiments use the public Cholec80 dataset [3], comprising eighty monocular laparoscopic cholecystectomy videos with frame-level phase labels and binary tool-presence labels. Following the established protocol [3, 4, 5, 6], the dataset is partitioned by patient into a training set of 40 videos (IDs 1–40), a validation set of 20 videos (IDs 41–60), and a held-out test set of 20 videos (IDs 61–80). Frames are extracted at one frame per second using OpenCV; the phase annotations originally provided at 25 fps are sub-sampled accordingly. Each frame is resized to 224 × 224 pixels and normalised by the ImageNet mean and standard deviation.

During training, augmentations are applied with a fixed probability profile — random horizontal flipping, random rotation up to 10°, random crop with a 10 % margin, and colour jitter — identically to all frames within a sequence so that temporal coherence is preserved. Each training and evaluation sample is a contiguous sequence of eight frames at one frame per second, providing eight seconds of temporal context.

**Table 3.1 — Phase distribution of the Cholec80 training split (1 fps).**

| Phase | Name | Frames | Fraction |
|---|---|---:|---:|
| 0 | Preparation | 3,758 | 4.4 % |
| 1 | Calot triangle dissection | 36,886 | 42.7 % |
| 2 | Clipping and cutting | 7,329 | 8.5 % |
| 3 | Gallbladder dissection | 24,119 | 27.9 % |
| 4 | Gallbladder packaging | 3,716 | 4.3 % |
| 5 | Cleaning and coagulation | 7,222 | 8.4 % |
| 6 | Gallbladder retraction | 3,314 | 3.8 % |

The dominant phase (Calot triangle dissection) outnumbers the rarest (Gallbladder retraction) by 11.1 to 1, the imbalance that the training pipeline is built to counter. Figure 3.1 shows the distribution across all three splits.

![Class distribution](results/figures/thesis/fig_class_dist.png)

**Figure 3.1** — Phase distribution across the training, validation, and test splits. The severe imbalance between the dominant phases (Calot, Dissection) and the short closing phases (Packaging, Retraction) is clearly visible and is preserved across splits.

## 3.2 Model Architectures

Three architectures are compared, denoted M1, M2, and M3. All three share the same overall structure, illustrated in Figure 3.2: a frame-level backbone produces an embedding for each of the eight frames, a temporal model consumes the resulting sequence of embeddings, and two parallel linear heads predict, for each frame, the phase class and the binary tool vector.

![Transfer learning](results/figures/thesis/fig_transfer.png)

**Figure 3.2** — Transfer learning and multi-task fine-tuning. ImageNet-pretrained backbone weights are loaded, the backbone is fine-tuned through a three-stage curriculum, an eight-frame temporal encoder integrates context, and parallel phase and tool heads produce the final predictions.

**M1 — ResNet-50 + bidirectional LSTM.** The backbone is a ResNet-50 [7] initialised with ImageNet weights, producing 2048-dimensional embeddings. The temporal model is a two-layer bidirectional LSTM with a hidden dimension of 512. Total trainable parameters: 35.7 M. This configuration follows the SV-RCNet paradigm [4].

**M2 — EfficientNet-B3 + temporal convolutional network.** The backbone is EfficientNet-B3 [8] (1536-dimensional output). The temporal model is a two-stage causal temporal convolutional network [5] with eight dilated layers per stage and 64 filters, reduced from the published TeCNO configuration to fit the memory budget. Total trainable parameters: 11.2 M.

**M3 — Swin-Tiny + Transformer.** The backbone is Swin Transformer-Tiny [9] (768-dimensional output). The temporal model is a three-layer Transformer encoder with six heads, a model dimension of 384, and a feed-forward dimension of 768 — a deliberately compact configuration motivated by the 4 GB memory budget. Total trainable parameters: 31.6 M.

In all three cases, a dropout of 0.3 is inserted between the temporal model and the heads. The phase head is a single linear layer projecting to seven logits; the tool head projects to seven independent sigmoid outputs.

**Table 3.2 — Summary of the three evaluated architectures.**

| Model | Backbone | Temporal model | Params (M) | Family |
|---|---|---|---:|---|
| M1 | ResNet-50 | Bidirectional LSTM | 35.7 | CNN + RNN |
| M2 | EfficientNet-B3 | Multi-stage TCN | 11.2 | CNN + TCN |
| M3 | Swin-Tiny | Transformer encoder | 31.6 | ViT + Attention |

## 3.3 Training Pipeline

The training pipeline is built around the 11:1 class imbalance. Its components address the problem from four angles — sampling-level weighting, target-level smoothing, multi-task regularisation, and curriculum scheduling — and their individual contributions are quantified in the ablation study (Section 4.2). Figure 3.3 provides an overview.

![Training pipeline](results/figures/thesis/fig_pipeline.png)

**Figure 3.3** — Overview of the training pipeline. An eight-frame sequence is augmented, passed through the pretrained backbone and the temporal model, and classified by the multi-task head. Optimisation uses AdamW with a cosine warm-restart schedule, class-weighted cross-entropy with label smoothing, and binary cross-entropy on tool presence.

**Loss function.** The total loss is a weighted sum of a phase term and a tool term:

> L = L_phase(z_phase, y_phase) + λ · L_tool(z_tool, y_tool)

where L_phase is class-weighted cross-entropy with label smoothing 0.1, L_tool is binary cross-entropy averaged over the seven tools, and λ = 0.5. Class weights are computed automatically as the inverse class frequency on the training split, normalised to sum to the number of classes; this counter-balances the 11.1× imbalance while keeping the dominant class learnable.

**Three-stage curriculum.** In Stage 1 (5 epochs) the backbone is frozen and only the temporal model and heads are trained. In Stage 2 (10 epochs) the schedule continues to warm the temporal model. In Stage 3 (10 epochs) the backbone is unfrozen and the whole network is fine-tuned end-to-end with the backbone learning rate divided by ten. Ablation runs use a proportionally shorter 3 / 6 / 6 schedule.

**Optimisation.** AdamW [10] is used with a main learning rate of 1×10⁻⁴ and a backbone learning rate of 1×10⁻⁵, decoupled weight decay of 1×10⁻⁴, and a cosine-with-warm-restarts schedule. Mixed-precision (fp16) arithmetic fits the 4 GB memory budget; gradients are unscaled before clipping to a maximum norm of 1.0. Per-step batch sizes are 4 (M1), 8 (M2), and 4 (M3), chosen by binary search against the memory envelope. Early stopping monitors validation macro-F1 with a patience of five epochs.

**Computational environment.** All experiments ran on a single NVIDIA GeForce RTX 2050 laptop GPU (4 GB GDDR6) with an Intel Core i7 mobile CPU and 24 GB of system RAM, under Windows 11, using Python 3.13, PyTorch 2.6 with CUDA 12.4, torchvision, timm, OpenCV, and scikit-learn.

## 3.4 Evaluation Metrics

Frame-level accuracy alone is misleading on Cholec80: a classifier that predicts the two dominant phases for every frame scores deceptively well while ignoring the short closing phases. The following metrics are therefore reported alongside accuracy.

- **Macro-F1.** The unweighted mean of per-class F1 scores. This is the primary model-selection criterion because it gives equal weight to all seven phases regardless of their frequency.
- **Per-class precision / recall / F1.** Reported in confusion-matrix form to expose where each model fails.
- **Edit score.** A segment-level action-segmentation metric that measures the quality of the predicted phase sequence rather than per-frame correctness.
- **Expected Calibration Error (ECE) and Negative Log-Likelihood (NLL).** Reported for the calibration analysis (Section 4.4).

At inference, per-frame phase predictions are taken as the argmax of the softmax distribution, and a median temporal filter of window length fifteen is applied along each video's predicted-phase sequence to remove isolated single-frame errors.

## 3.5 Causal Decoding and Calibration

Beyond the median filter, this thesis introduces a decoding layer that converts per-frame logits into a temporally coherent phase sequence **causally** — using only past and present frames — so that the system could run during surgery.

**Temperature calibration.** For each model a single scalar temperature T\* is fitted on the validation set by minimising negative log-likelihood over a log-spaced grid. Dividing the logits by T\* before the softmax rebalances over-confident emissions, reducing ECE without changing the argmax decision. The fitted temperatures are 1.3 (M1), 1.1 (M2), and 1.2 (M3) — all above 1.0, consistent with the general over-confidence of deep classifiers.

**Learned transition prior.** A phase-transition matrix A, where A[i, j] = P(phase_{t+1} = j | phase_t = i), is estimated from the 40 training videos by counting one-step transitions at 1 fps with Laplace smoothing. The resulting matrix is strongly diagonal — the per-second probability of staying in the current phase ranges from 0.988 to 0.999 — with off-diagonal mass concentrated on the next phase, reflecting the fixed procedural order of cholecystectomy.

**Streaming Viterbi decoder.** A log-marginal vector α_t over the seven phases is maintained such that α_t[c] is the maximum log-probability of any path ending in phase c at time t. At each step,

> α_t[c] = log softmax(z_t / T\*)[c] + max_{c'} ( α_{t−1}[c'] + log A[c', c] ),

and the predicted phase is the argmax of α_t. The update has O(C) memory and O(C²) time per frame (49 operations for seven phases), so it runs far faster than real time. Crucially, the calibrated emissions and the transition prior are combined in a single log-probability space; if the emissions are over-peaked, the prior is effectively ignored, which is why calibration is a prerequisite for the prior to help.

The proposed decoder is benchmarked against four baselines: raw argmax, median-15 smoothing, and an **offline monotonic Viterbi** pass over the entire video, which enforces the non-decreasing phase order using future frames and therefore serves as a non-causal upper bound rather than a deployable method.

## 3.6 Ablation and Ensemble Design

Four ablations, each modifying one design choice relative to the M1 baseline, isolate the contribution of individual pipeline components: **A1** removes the tool-detection head (single-task phase prediction); **A2** removes class weighting (uniform cross-entropy); **A3** doubles the sequence length to sixteen frames with the batch size halved; and **A4** disables the median filter at inference.

Three ensemble strategies combine the three trained models: **simple averaging** of the softmax distributions, **validation-F1-weighted averaging**, and **majority voting** over the per-frame argmax predictions. All ensemble outputs pass through the same median filter before evaluation.

<div style="page-break-after: always;"></div>

# Chapter 4 — Experiments and Results

All experiments were conducted on the single 4 GB GPU described in Section 3.3, with the random seed fixed at 42. Results are reported on the held-out test set of twenty videos (IDs 61–80), which no model saw during training or hyperparameter selection.

## 4.1 Individual Model Comparison

Table 4.1 reports per-frame accuracy and macro-averaged F1 for the three architectures, before and after median smoothing.

**Table 4.1 — Test-set performance of the three main models. Best values in bold.**

| Model | Raw Acc. | Raw Macro-F1 | Smoothed Acc. | Smoothed Macro-F1 |
|---|---:|---:|---:|---:|
| M1 — ResNet-50 + BiLSTM | 0.832 | 0.775 | 0.837 | 0.781 |
| M2 — EfficientNet-B3 + TCN | 0.818 | 0.739 | 0.824 | 0.748 |
| **M3 — Swin-Tiny + Transformer** | **0.857** | **0.807** | **0.862** | **0.815** |

The Swin-Transformer model M3 obtains the best score on every metric, with a smoothed macro-F1 of **0.815**, exceeding the ResNet-LSTM baseline by 3.4 percentage points and the EfficientNet-TCN by 6.7 points. The accuracy of 0.862 places M3 within the published range of Trans-SVNet on the same dataset, despite training on a single 4 GB GPU rather than data-centre hardware. Figure 4.1 plots each model's performance against its parameter count and shows that M3 dominates without being the largest model — M1 has more parameters yet scores lower, confirming that capacity is not the binding constraint here.

![Architecture comparison](results/figures/thesis/fig_arch_compare.png)

**Figure 4.1** — Capacity versus performance. The Swin-Transformer (M3) achieves the highest smoothed macro-F1 without being the largest model; the EfficientNet-TCN (M2) is the smallest but weakest, partly because its Stage-3 fine-tuning was interrupted by memory pressure (Section 4.1, training dynamics).

**Training dynamics.** Figure 4.2 shows loss and accuracy curves for all three models across the three-stage curriculum. M1 and M3 converge stably, with validation accuracy rising sharply once the backbone is unfrozen in Stage 3. M2's curve is shorter because, under the unfrozen EfficientNet-B3 backbone, per-epoch memory pressure reached roughly 97 % of the 4 GB envelope and only a partial Stage 3 could be completed; the reported M2 numbers should therefore be read as a lower bound on what EfficientNet-TCN can achieve in this setting.

![Training history](results/figures/thesis/fig_train_history.png)

**Figure 4.2** — Training dynamics for the three main models: total loss (top) and accuracy (bottom) for training and validation across epochs. M3 shows the clearest Stage-3 jump; M2's run is truncated by the memory-constrained backbone fine-tuning.

**Per-class analysis of the best model.** Table 4.2 details per-class precision, recall, and F1 for M3 on the raw test predictions.

**Table 4.2 — Per-class precision / recall / F1 for M3 (raw, test set).**

| Phase | Precision | Recall | F1 |
|---|---:|---:|---:|
| Preparation | 0.830 | 0.803 | 0.816 |
| Calot triangle dissection | 0.880 | 0.895 | 0.887 |
| Clipping and cutting | 0.815 | 0.796 | 0.806 |
| Gallbladder dissection | 0.868 | 0.885 | 0.877 |
| Gallbladder packaging | 0.881 | 0.759 | 0.816 |
| Cleaning and coagulation | 0.741 | 0.663 | **0.700** |
| Gallbladder retraction | 0.727 | 0.764 | 0.745 |

The two dominant phases reach the highest F1 scores, as expected. The weakest class is Cleaning and coagulation (recall 0.663): the model frequently confuses cleaning episodes with the surrounding dissection phase, a confusion that also appears in the confusion matrix (Figure 4.3). Critically, all seven phases exceed F1 = 0.70, confirming that the class-weighted loss prevents the collapse onto the dominant phases that naïve training would produce.

![Confusion matrix M3](results/figures/thesis/fig_confusion_m3.png)

**Figure 4.3** — Normalised confusion matrix (row recall) for M3 on the smoothed test predictions. The dominant misclassifications are Cleaning predicted as Dissection and Clipping predicted as Dissection — adjacent phases that share visual context — while no phase collapses toward the majority class.

## 4.2 Ablation Study

Table 4.3 reports the four ablations against the M1 baseline, each trained under the identical optimiser, scheduler, and three-stage schedule.

**Table 4.3 — Ablation study (test-set smoothed macro-F1, Δ relative to M1).**

| ID | Modification | Smoothed Macro-F1 | ΔF1 vs M1 |
|---|---|---:|---:|
| **M1 baseline** | full configuration | **0.781** | — |
| A1 | remove tool detection (single-task) | 0.765 | **−1.6 %** |
| A2 | remove class weighting | 0.761 | **−2.0 %** |
| A3 | sequence length 16 instead of 8 | 0.787 | +0.6 % |
| A4 | disable median smoothing | 0.775 | −0.6 % |

Three findings stand out. First, removing tool detection (A1) costs 1.6 points of macro-F1, confirming the regularising effect of auxiliary tool supervision first reported by EndoNet. Second, removing class weighting (A2) costs 2.0 points, but the damage is unevenly distributed: the two rarest phases — Gallbladder retraction and Gallbladder packaging — lose 5.1 and 2.8 points of per-class F1 respectively, while the dominant Calot phase actually gains 0.7. This is exactly the failure mode that class weighting is designed to prevent on imbalanced data. Third, A4 isolates the post-processing contribution: median smoothing improves macro-F1 only marginally (+0.6 %) but boosts the edit score by roughly 35 % relative, confirming that its dominant benefit is on boundary quality rather than per-frame accuracy. Doubling the sequence length (A3) yields a modest +0.6 %, indicating that the eight-second window already captures most of the useful temporal context at this resolution.

Together, class weighting and multi-task supervision contribute almost four points of macro-F1 — comparable to the 3.4-point gap between the best and worst architecture, the central evidence that the training recipe is as influential as the backbone on Cholec80.

## 4.3 Ensemble Results

Table 4.4 summarises the three ensemble strategies under a consistent non-overlapping evaluation protocol.

**Table 4.4 — Ensemble strategies (smoothed, test set).**

| Strategy | Accuracy | Macro-F1 |
|---|---:|---:|
| M3 alone (best single model) | 0.856 | 0.799 |
| Ensemble — simple softmax average | 0.863 | 0.808 |
| **Ensemble — val-F1-weighted average** | **0.864** | **0.809** |
| Ensemble — majority voting | 0.857 | 0.799 |

The validation-F1-weighted ensemble exceeds the best single model by 1.0 point of macro-F1, with M3 receiving the largest weight (0.355), followed by M1 (0.327) and M2 (0.318). The near-equal weights suggest that the three architectures are individually capable rather than strongly complementary — their disagreements concentrate on the same hard frames (typically at phase boundaries) rather than partitioning the error mass. This observation directly motivates the decoding and calibration work of the next section: if the residual errors are concentrated and structured, a better decoder may help more than a larger ensemble.

## 4.4 Causal Decoding and Calibration

This section evaluates the streaming causal decoder of Section 3.5 against the baselines, using identical per-frame logits for every method so that differences reflect only the decoding stage.

**Calibration.** Table 4.5 reports validation-set ECE and NLL before and after temperature scaling. M1, the most over-confident model, benefits most — its ECE drops by 66 %. Across all three models the optimal temperature exceeds 1.0, the expected over-confidence signature. Figure 4.4 shows the corresponding reliability diagrams: the raw per-bin accuracy (grey) sits below the diagonal for confident bins, and temperature scaling pulls the curve toward perfect calibration.

**Table 4.5 — Validation-set calibration before and after temperature scaling.**

| Model | T\* | ECE raw | ECE cal | NLL raw | NLL cal |
|---|---:|---:|---:|---:|---:|
| M1 — ResNet-LSTM | 1.30 | 0.109 | **0.037** | 0.897 | 0.855 |
| M2 — EffNet-TCN | 1.10 | 0.059 | 0.074 | 0.824 | 0.817 |
| M3 — Swin-Transformer | 1.20 | 0.074 | 0.070 | 0.688 | 0.667 |

![Reliability diagrams](results/figures/fig1_reliability.png)

**Figure 4.4** — Reliability diagrams on the validation set. Grey bars are raw per-bin accuracy; the teal curve is after temperature scaling; the dashed diagonal is perfect calibration. M1 is pulled most strongly toward the diagonal.

**Decoder benchmark.** Table 4.6 reports test-set accuracy, macro-F1, and per-frame latency for each decoder on each model. The proposed causal HMM decoder with calibrated logits is the best **causal** decoder on all three models, while the offline monotonic decoder is shown only as a non-causal upper bound. Figure 4.5 visualises the same comparison.

**Table 4.6 — Decoder comparison (test set). The offline decoder † is a non-causal upper bound.**

| Decoder | M1 Acc / F1 | M2 Acc / F1 | M3 Acc / F1 |
|---|---|---|---|
| Raw argmax | 82.4 / 0.763 | 81.2 / 0.738 | 85.3 / 0.794 |
| Median-15 | 82.7 / 0.768 | 81.7 / 0.746 | 85.5 / 0.800 |
| Causal monotonic + cal | 82.5 / 0.777 | 82.4 / 0.767 | 86.0 / 0.802 |
| **Causal HMM + cal** | **83.0 / 0.771** | **82.9 / 0.764** | **86.0 / 0.806** |
| Offline monotonic † | 91.8 / 0.859 | 91.8 / 0.849 | 90.1 / 0.832 |

![Decoder benchmark](results/figures/fig2_benchmark.png)

**Figure 4.5** — Decoder comparison on the test set: accuracy (top) and macro-F1 (bottom). The proposed causal HMM + calibration (teal) is the best causal-deployable decoder on every model; the offline monotonic decoder (amber, hatched) is the non-causal upper bound.

**Statistical significance.** Because the aggregate gains look modest, a per-video paired test is essential. Table 4.7 reports the paired Wilcoxon signed-rank test and Cohen's *d* over the twenty test videos, comparing the proposed decoder against raw argmax. All three models show a highly significant, large-effect improvement (*p* < 0.001, *d* > 0.97) — the gain is small in absolute terms but consistent across videos rather than driven by outliers. Per-frame decoding latency is 0.011–0.024 ms, roughly three orders of magnitude below the 40 ms budget implied by 25 fps live video. Figure 4.6 shows the learned transition matrix and the per-video gains side by side.

**Table 4.7 — Per-video significance of causal HMM + cal versus raw argmax (20 test videos).**

| Model | ΔAcc (pp) | Wilcoxon *p* | Cohen's *d* |
|---|---:|---:|---:|
| M1 — ResNet-LSTM | +0.62 | < 0.0001 | 1.38 |
| M2 — EffNet-TCN | +1.74 | 0.0001 | 1.25 |
| M3 — Swin-Transformer | +0.76 | 0.0001 | 0.98 |

![Transition and significance](results/figures/fig5_transition_significance.png)

**Figure 4.6** — Left: the learned per-second transition matrix, strongly diagonal with off-diagonal mass on the next phase, reflecting the fixed procedural order. Right: per-video accuracy gain of the proposed decoder over raw argmax, annotated with Cohen's *d* and the paired Wilcoxon *p*-value.

**Qualitative example.** Figure 4.7 shows the predicted phase timeline for one test video. Raw argmax produces many isolated flips, especially early in the procedure; the proposed causal decoder removes most of them and tracks the ground truth far more cleanly.

![Qualitative timeline](results/figures/fig4_timeline.png)

**Figure 4.7** — Qualitative phase timeline for a test video (M3). Raw argmax (middle) flips frequently; the proposed causal HMM + calibration decoder (bottom) is markedly smoother and closely follows the ground truth (top).

## 4.5 Boundary-versus-Interior Analysis

Annotation noise on Cholec80 concentrates at phase transitions, where even human annotators disagree on the exact boundary. To see where each decoder actually helps, test frames are split into **boundary** frames (within ±5 s of a ground-truth transition) and **interior** frames (everywhere else). Table 4.8 reports the split for M3, and Figure 4.8 visualises it.

**Table 4.8 — Boundary versus interior accuracy for M3. The offline decoder † is non-causal.**

| Decoder | Boundary Acc. | Interior Acc. |
|---|---:|---:|
| Raw argmax | 60.7 % | 86.1 % |
| Median-15 | 61.1 % | 86.3 % |
| **Causal HMM + cal** | **63.2 %** | 86.8 % |
| Offline monotonic † | 62.8 % | **91.0 %** |

![Boundary analysis](results/figures/fig3_boundary.png)

**Figure 4.8** — Boundary versus interior accuracy for M3. Boundary frames (solid bars) are hard for every method; the proposed causal decoder is the strongest there, slightly exceeding even the offline upper bound. Interior frames (hatched bars) are where the offline decoder's entire advantage lies.

Two findings stand out. First, the proposed causal decoder is the **strongest on boundary frames** (63.2 %), even outperforming the offline upper bound (62.8 %): the offline decoder's hard monotonicity constraint occasionally forces a delayed transition, whereas the soft learned prior accommodates the labelled transition timing more flexibly. Second, the entire 4.2-point accuracy gap between offline decoding and the best causal decoder lies in **interior** frames (86.8 → 91.0), not at boundaries. The offline decoder's advantage is global error correction inside long stable phases by integrating future evidence — something a causal decoder cannot recover by construction. To our knowledge this decomposition has not previously been quantified on Cholec80, and it identifies a clean direction for future work: a short, bounded-latency lookahead would close most of the gap while remaining deployable.

<div style="page-break-after: always;"></div>

# Chapter 5 — Conclusion and Future Work

## 5.1 Summary of Contributions

This thesis delivered a controlled, hardware-constrained study of surgical phase recognition on Cholec80, organised around two connected contributions.

The first is a **fair architecture comparison**. Three representative families — ResNet-50 + BiLSTM, EfficientNet-B3 + TCN, and Swin-Tiny + Transformer — were trained under a single identical pipeline (three-stage curriculum, automatic class re-weighting, multi-task tool supervision, label smoothing, and mixed-precision optimisation) on a single 4 GB consumer GPU. Holding the training recipe fixed isolates the effect of architecture: the Swin-Transformer reached a smoothed macro-F1 of 0.815, exceeding the ResNet-LSTM by 3.4 points and the EfficientNet-TCN by 6.7 points. Four controlled ablations then attributed roughly +1.6 % macro-F1 to multi-task tool supervision and +2.0 % to class re-weighting, and a validation-F1-weighted ensemble added a further 1.0 point.

The second contribution targets the **decoding and calibration stages** that prior Cholec80 work largely overlooks. A streaming causal Viterbi decoder with a learned transition prior and temperature-scaled emissions was introduced; without any retraining it improved test-set accuracy by +0.6 to +1.7 points across the three models — a small but, by paired Wilcoxon testing, highly significant gain with a large effect size — at a decoding cost of 0.013 ms per frame. A boundary-versus-interior decomposition showed that the proposed decoder is strongest exactly where models are weakest (at phase transitions) and that the residual gap to a non-causal offline upper bound lies entirely in interior frames.

## 5.2 Key Findings

Three findings have implications beyond the specific dataset and models studied.

**The training recipe rivals the architecture.** Class weighting and multi-task supervision together contribute almost four points of macro-F1 — comparable to the gap between the best and worst architecture. On a heavily imbalanced surgical dataset, how the model is trained is at least as important as which backbone it uses. This is easy to overlook when comparisons vary the architecture and the recipe at the same time, and it is precisely what the controlled pipeline of this thesis was designed to expose.

**Calibration unlocks the prior, not just trust.** Temperature scaling is usually motivated by trustworthiness alone, but here it has a second, mechanical role: the Viterbi update combines log-emission and log-transition additively, so over-peaked emissions drown out the transition prior. Calibrating the emissions is what allows the learned phase ordering to contribute, which is why the calibrated decoder consistently beats its uncalibrated counterpart.

**The lookahead premium lives in interior frames.** The advantage of offline decoding over the best causal decoder is almost entirely a matter of correcting momentary mis-classifications inside long, stable phases by integrating future evidence — not of placing phase boundaries more accurately. This reframes what a real-time system gives up: not boundary precision, but the ability to retrospectively fix interior glitches. A short, bounded-latency lookahead is therefore the natural next lever.

## 5.3 Limitations

Several constraints bound the conclusions of this study.

- **Hardware envelope.** Batch size and sequence length were both forced small by the 4 GB GPU. Results on data-centre hardware would likely be one to three points higher, and the EfficientNet-TCN model in particular suffered an interrupted Stage-3 fine-tuning, so its numbers are a lower bound rather than a fair estimate of its potential.
- **Single dataset.** All conclusions are drawn from Cholec80. Procedures with less rigid phase orderings may favour a less restrictive transition prior, and generalisation to other surgeries is untested.
- **Annotation noise.** Phase-boundary labels in Cholec80 are imprecise by construction, placing an upper bound on achievable boundary accuracy that is not formally estimated here.
- **Sub-sampled evaluation in the live demo.** The accompanying interactive demo sub-samples frames for speed; short closing phases can be missed at low sampling density. The reported benchmark numbers, however, use the full 1 fps protocol and are unaffected.

## 5.4 Future Work

The most direct extensions address these limitations.

- **Bounded-latency lookahead.** A decoder that may peek a few seconds ahead would recover most of the interior-frame gap to the offline upper bound while keeping latency clinically acceptable, directly targeting the bottleneck identified in Section 4.5.
- **Data-centre re-training.** Re-running the same three architectures with batch sixteen and sequence length thirty-two would clarify how much of the gap to published state of the art is attributable to hardware rather than architecture.
- **Uncertainty estimation.** Adding Monte-Carlo dropout or deep-ensemble uncertainty, on top of the calibration already in place, would turn each per-frame prediction into a clinically usable confidence signal.
- **Cross-procedure generalisation.** Re-training on other procedures with annotated workflow video would test whether the pipeline and the learned-prior decoder transfer beyond cholecystectomy.

In summary, this thesis shows that competitive surgical-workflow recognition is achievable on consumer-grade hardware when training, decoding, and post-processing are designed jointly — and that the decoding and calibration stages, often treated as an afterthought, carry real, statistically significant, real-time value.

<div style="page-break-after: always;"></div>

## References

[1] Maier-Hein, L. et al. (2017). Surgical data science for next-generation interventions. *Nature Biomedical Engineering*, 1(9):691–696.

[2] Vedula, S. S., Ishii, M., and Hager, G. D. (2017). Objective assessment of surgical technical skill and competency in the operating room. *Annual Review of Biomedical Engineering*, 19:301–325.

[3] Twinanda, A. P., Shehata, S., Mutter, D., Marescaux, J., de Mathelin, M., and Padoy, N. (2017). EndoNet: A deep architecture for recognition tasks on laparoscopic videos. *IEEE Transactions on Medical Imaging*, 36(1):86–97.

[4] Jin, Y., Dou, Q., Chen, H., Yu, L., Qin, J., Fu, C.-W., and Heng, P.-A. (2018). SV-RCNet: Workflow recognition from surgical videos using recurrent convolutional network. *IEEE Transactions on Medical Imaging*, 37(5):1114–1126.

[5] Czempiel, T., Paschali, M., Keicher, M., Simson, W., Feussner, H., Kim, S. T., and Navab, N. (2020). TeCNO: Surgical phase recognition with multi-stage temporal convolutional networks. In *MICCAI*, pp. 343–352.

[6] Gao, X., Jin, Y., Long, Y., Dou, Q., and Heng, P.-A. (2021). Trans-SVNet: Accurate phase recognition from surgical videos via hybrid embedding aggregation Transformer. In *MICCAI*, pp. 593–603.

[7] He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. In *CVPR*, pp. 770–778.

[8] Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *ICML*, pp. 6105–6114.

[9] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., and Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using shifted windows. In *ICCV*, pp. 10012–10022.

[10] Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In *ICLR*.

[11] Buda, M., Maki, A., and Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106:249–259.

[12] Lea, C., Flynn, M. D., Vidal, R., Reiter, A., and Hager, G. D. (2017). Temporal convolutional networks for action segmentation and detection. In *CVPR*, pp. 156–165.

[13] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. In *NeurIPS*, pp. 5998–6008.

[14] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. (2016). Rethinking the Inception architecture for computer vision. In *CVPR*, pp. 2818–2826.

[15] Micikevicius, P. et al. (2018). Mixed precision training. In *ICLR*.

[16] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). A ConvNet for the 2020s. In *CVPR*, pp. 11976–11986.

[17] Lea, C., Vidal, R., Reiter, A., and Hager, G. D. (2016). Temporal convolutional networks: A unified approach to action segmentation. In *ECCV Workshops*, pp. 47–54.

[18] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In *ICML*, pp. 1321–1330.

[19] Cui, Y., Jia, M., Lin, T.-Y., Song, Y., and Belongie, S. (2019). Class-balanced loss based on effective number of samples. In *CVPR*, pp. 9268–9277.

[20] Wightman, R. (2019). PyTorch Image Models. https://github.com/rwightman/pytorch-image-models.

---

*End of report.*
