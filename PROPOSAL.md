# A Systematic Comparison of Three Deep-Learning Architectures for Surgical Phase Recognition on Cholec80

**USTH Scientific Research for Students 2026 — Topic Registration**

**Student:** [Your full name]
**Student ID:** [Your ID]
**Department:** [Department]
**Major:** [Major]
**Supervisor:** [Supervisor name]

---

## ABSTRACT

Surgical phase recognition is the task of labelling each frame of a surgery video with the procedural step taking place — a foundation for operating-room analytics, surgical skill assessment, and intra-operative decision support. The Cholec80 benchmark contains 80 laparoscopic cholecystectomy videos split into seven surgical phases that are heavily imbalanced (the most common phase covers 41 % of frames while the rarest covers under 4 %) and that often look visually similar across short windows of time. Existing work spans three architecture families — CNN + LSTM, CNN + TCN, and CNN + Transformer — but published comparisons mix different training protocols, loss functions, and post-processing choices, making it hard to know how much of the performance gap comes from the architecture itself. This project trains three representative models (ResNet-50 + BiLSTM, EfficientNet-B3 + TCN, and Swin-Tiny + Transformer) under a single, identical pipeline so that any difference reflects architecture only. Four controlled ablations measure the contribution of tool detection, class re-weighting, sequence length, and temporal smoothing, and three ensemble strategies test whether combining the models adds value. Preliminary experiments place the Transformer model at 0.815 smoothed macro-F1 on the held-out test set — competitive with published TCN-based methods.

**Keywords:** surgical phase recognition · Cholec80 · architecture comparison · multi-task learning · ablation study · ensemble

---

## I. INTRODUCTION

Hospitals routinely record video during minimally invasive surgery, but very little of that footage is ever analysed. A system that automatically recognises which procedural step is happening would enable several practical applications: generating operative reports, assessing trainee skill, estimating how long the operation will take, and warning the team when the workflow deviates from the expected sequence [1, 2].

The standard public benchmark for this task is Cholec80 [3]: 80 videos of laparoscopic cholecystectomy, with every frame labelled by which of seven phases is occurring (Preparation, Calot's triangle dissection, Clipping and cutting, Gallbladder dissection, Gallbladder packaging, Cleaning and coagulation, Gallbladder retraction) and which of seven instruments are visible. Two characteristics make the dataset hard. First, the phase distribution is very uneven — one phase fills 41 % of all frames while another fills under 4 %, an 11× ratio that biases naïve training toward the dominant class. Second, transitions between phases often look visually similar for several seconds, so a model needs temporal context, not just a single frame, to decide what is happening.

Research on Cholec80 has gone through three waves. EndoNet [3] used a single CNN with a Hidden Markov Model smoother. SV-RCNet [4] and TeCNO [5] kept a CNN backbone but added a trainable temporal model — an LSTM or a temporal convolutional network — on top of it. Trans-SVNet [6] introduced a Transformer over per-frame embeddings and pushed accuracy further. Reported numbers improved from roughly 0.79 macro-F1 to roughly 0.85 over the same five-year span, but cross-paper comparisons mix different data splits, loss functions, batch sizes, sequence lengths, and post-processing choices — so it is genuinely unclear how much of each improvement is the new architecture and how much is the surrounding training recipe.

This project addresses that gap directly. Three representative architectures — CNN with recurrent temporal model, CNN with convolutional temporal model, and Transformer with attention-based temporal model — are trained under a single, identical protocol on the standard data split. Four ablations isolate the contribution of common design choices, and three ensemble strategies test whether the models complement one another. The questions answered are: (Q1) which architecture family wins under fixed training conditions on Cholec80? (Q2) how much do multi-task tool supervision, class re-weighting, sequence length and temporal smoothing each contribute? (Q3) do the three models combine into a stronger ensemble than the best one alone?

---

## II. OBJECTIVES

The goal is to deliver a controlled comparison of three modern deep-learning architectures for surgical phase recognition on Cholec80 by holding the training pipeline, loss function, dataset split and post-processing identical across all three. Four ablation studies will quantify the contribution of common design choices, and three ensemble strategies will be evaluated.

---

## III. MATERIALS AND METHODS

The Cholec80 dataset is split by patient into 40 training, 20 validation and 20 test videos following the established protocol. Frames are extracted at 1 frame per second, resized to 224 × 224, normalised, and lightly augmented (random flip, rotation up to 10°, random crop and colour jitter). Each input sample is a sequence of 8 consecutive frames — 8 seconds of context. Three architectures share a common multi-task head that predicts both the current phase (seven-way classification) and which tools are present (seven binary outputs). **M1** uses a ResNet-50 backbone with a two-layer bidirectional LSTM (35.7 M parameters). **M2** uses an EfficientNet-B3 backbone with a temporal convolutional network (11.2 M parameters). **M3** uses a Swin-Tiny backbone with a three-layer Transformer encoder (31.6 M parameters).

The training procedure is identical for all three models. The loss combines class-weighted cross-entropy on phase with label smoothing 0.1 and binary cross-entropy on tool presence (weighted 0.5); class weights are computed automatically as the inverse class frequency to counter the 11× imbalance. Training runs in three curriculum stages: backbone frozen for 5 epochs, still frozen for 10 more, then fully fine-tuned for 10 epochs at a smaller learning rate. The optimiser is AdamW with a cosine warm-restart schedule. Early stopping monitors validation macro-F1 with a patience of 5 epochs. At inference, a median filter of length 15 is applied to remove single-frame noise. The four ablations each modify one design choice relative to M1: A1 removes tool detection, A2 removes class weighting, A3 doubles the sequence length to 16, and A4 disables the median filter. Three ensemble strategies — softmax averaging, validation-F1-weighted averaging, and majority voting — combine the three models.

---

## IV. RESULTS AND DISCUSSION

### 1. Preliminary results

All three main models and the four ablations have been trained and evaluated on the held-out test set of twenty videos. The Swin-Transformer model **M3 reaches 0.815 smoothed macro-F1 (accuracy 0.862)**, beating the ResNet-LSTM baseline M1 (0.781) by 3.4 percentage points and the EfficientNet-TCN model M2 (0.748) by 6.7 points. A validation-F1-weighted softmax ensemble of the three models gives a small further gain. The ablations show that removing tool detection costs about 1.6 % macro-F1; removing class weights costs about 2.0 %, and most of that drop concentrates on the two rarest phases (Gallbladder retraction loses 5 points of per-class F1); median smoothing improves macro-F1 only marginally (+0.6 %) but boosts the action-segmentation edit score by roughly 35 % — its main role is cleaning up phase boundaries, not improving per-frame accuracy.

### 2. Discussion

Under a single, controlled training pipeline, the Transformer-based model clearly outperforms the recurrent and convolutional alternatives on Cholec80 — confirming the architectural direction taken by the most recent literature, while also showing that the gap between architecture families is real and not an artefact of different training recipes. The ablations make a complementary point: class weighting and multi-task tool supervision together contribute almost four percentage points of macro-F1, comparable in magnitude to the gap between the best and worst architecture. This indicates that on heavily imbalanced datasets like Cholec80, the loss-function recipe is at least as influential as the architecture choice. The small ensemble gain happens because all three models concentrate their errors on the same ambiguous frames — usually at phase boundaries — rather than on complementary frames, pointing toward calibration and refined temporal modelling as the more promising next step.

---

## V. CONCLUSION AND PERSPECTIVE

This work delivers a fair, controlled comparison of three modern architecture families for surgical phase recognition on Cholec80, accompanied by four ablation studies and three ensemble strategies. The Transformer-based model achieves 0.815 smoothed macro-F1, competitive with published TCN-based work, and the ablation analysis cleanly attributes contributions of individual design choices. Three natural next steps open up: extending the comparison to longer temporal context and larger sequence batches; adding calibrated uncertainty estimates so the predictions can be trusted in a clinical context; and replacing the median post-filter with a causal temporal model so the system could run in real time during surgery. A detailed final report including full confusion matrices, training curves, and qualitative phase-timeline visualisations will be submitted in July.

---

## REFERENCES

[1] Maier-Hein, L. et al. (2017). Surgical data science for next-generation interventions. *Nature Biomedical Engineering*, 1(9):691–696.

[2] Vedula, S. S., Ishii, M., and Hager, G. D. (2017). Objective assessment of surgical technical skill and competency in the operating room. *Annual Review of Biomedical Engineering*, 19:301–325.

[3] Twinanda, A. P. et al. (2017). EndoNet: A deep architecture for recognition tasks on laparoscopic videos. *IEEE Transactions on Medical Imaging*, 36(1):86–97.

[4] Jin, Y. et al. (2018). SV-RCNet: Workflow recognition from surgical videos using recurrent convolutional network. *IEEE Transactions on Medical Imaging*, 37(5):1114–1126.

[5] Czempiel, T. et al. (2020). TeCNO: Surgical phase recognition with multi-stage temporal convolutional networks. In *MICCAI*, pp. 343–352.

[6] Gao, X. et al. (2021). Trans-SVNet: Accurate phase recognition from surgical videos via hybrid embedding aggregation Transformer. In *MICCAI*, pp. 593–603.

[7] He, K. et al. (2016). Deep residual learning for image recognition. In *CVPR*, pp. 770–778.

[8] Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *ICML*, pp. 6105–6114.

[9] Liu, Z. et al. (2021). Swin Transformer: Hierarchical Vision Transformer using shifted windows. In *ICCV*, pp. 10012–10022.

[10] Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In *ICLR*.
