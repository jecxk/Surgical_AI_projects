# 🏥 Surgical Phase Recognition & Tool-Aware Workflow Understanding

> **Nhận diện giai đoạn phẫu thuật và hiểu quy trình thao tác từ video nội soi ổ bụng bằng học sâu**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project implements deep learning models for **real-time surgical phase recognition** from laparoscopic cholecystectomy videos. The system identifies the current surgical phase and detects surgical tools, providing insights into the surgical workflow.

### Key Features
- **4 Model Architectures**: ResNet50+LSTM, ResNet50+TCN (MS-TCN), ResNet50+Transformer, TimeSformer
- **Multi-Task Learning**: Simultaneous phase recognition + tool detection
- **Temporal Smoothing**: Post-processing for smoother predictions
- **Grad-CAM Interpretability**: Visual explanations of model decisions
- **Comprehensive Metrics**: Accuracy, Macro-F1, Edit Score, Phase Transition Error

### Surgical Phases (Cholec80)
| Phase | Name | Description |
|-------|------|-------------|
| 0 | Preparation | Initial setup and trocar insertion |
| 1 | Calot Triangle Dissection | Exposure of cystic duct and artery |
| 2 | Clipping & Cutting | Clip and divide cystic structures |
| 3 | Gallbladder Dissection | Separation from liver bed |
| 4 | Gallbladder Packaging | Bag and prepare for extraction |
| 5 | Cleaning & Coagulation | Hemostasis and irrigation |
| 6 | Gallbladder Retraction | Final extraction |

## 🚀 Quick Start

### Installation
```bash
# Clone and install
cd d:\Surgical_AI
pip install -r requirements.txt
```

### Training with Synthetic Data (for development)
```bash
# Quick test with synthetic data
python scripts/train.py --config configs/resnet_lstm.yaml --synthetic --epochs 5

# Full training
python scripts/train.py --config configs/default.yaml --synthetic
```

### Training with Cholec80 Dataset
```bash
# 1. Prepare data (after downloading from CAMMA)
python data/prepare_data.py --video_dir /path/to/cholec80/videos --output_dir data/cholec80

# 2. Train different architectures
python scripts/train.py --config configs/resnet_lstm.yaml
python scripts/train.py --config configs/resnet_tcn.yaml
python scripts/train.py --config configs/resnet_transformer.yaml
```

### Evaluation & Visualization
```bash
python scripts/evaluate.py --checkpoint results/resnet50_lstm/checkpoints/best_model.pth --synthetic
python scripts/visualize.py --checkpoint results/resnet50_lstm/checkpoints/best_model.pth
```

## 📁 Project Structure
```
Surgical_AI/
├── configs/              # YAML configurations
├── data/                 # Dataset & preparation scripts
├── src/
│   ├── dataset/          # Data loading & transforms
│   ├── models/           # Model architectures
│   │   ├── backbone.py           # CNN feature extractors
│   │   ├── temporal_lstm.py      # Bi-LSTM temporal model
│   │   ├── temporal_tcn.py       # MS-TCN temporal model
│   │   ├── temporal_transformer.py  # Transformer temporal model
│   │   ├── multi_task_head.py    # Phase + Tool heads
│   │   └── surgical_model.py    # Full model wrapper
│   ├── training/         # Training loop & losses
│   ├── evaluation/       # Metrics & evaluator
│   └── visualization/    # Grad-CAM, plots, analysis
├── scripts/              # Training & evaluation scripts
└── results/              # Checkpoints, logs, figures
```

## 🏗️ Model Architectures

### Architecture 1: ResNet50 + Bi-LSTM
```
Frames → ResNet50 → Feature Vectors → Bi-LSTM → Phase + Tool Predictions
```

### Architecture 2: ResNet50 + MS-TCN
```
Frames → ResNet50 → Features → Multi-Stage TCN → Refined Predictions
```

### Architecture 3: ResNet50 + Temporal Transformer
```
Frames → ResNet50 → Features + PosEnc → Transformer Encoder → Predictions
```

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Frame-level classification accuracy |
| Macro-F1 | Per-phase F1, macro-averaged |
| Edit Score | Segment-level edit distance (1 = perfect) |
| Phase Transition Error | Mean error at phase boundaries (seconds) |

## 📖 Dataset

### Cholec80
- **Source**: [CAMMA](https://camma.unistra.fr/datasets/) (registration required)
- 80 cholecystectomy videos, 7 phases, frame-level annotations
- Split: 40 train / 20 val / 20 test

### CholecT50
- 50 videos with surgical action triplets
- Instrument × Verb × Target annotations

## 📚 References

1. Twinanda et al., "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos", IEEE TMI 2017
2. Czempiel et al., "TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks", MICCAI 2020
3. Gao et al., "Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer", MICCAI 2021
4. Nwoye et al., "Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos", MedIA 2022

## 📄 License
MIT License
