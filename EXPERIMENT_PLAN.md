# Surgical AI — Kế hoạch thực nghiệm Thesis/NCKH

**Hardware:** RTX 2050 4GB VRAM (laptop) · Windows 11 · Python 3.13 + PyTorch 2.6 + CUDA 12.4
**Dataset:** Cholec80 — 80 videos cholecystectomy, 7 phases, 7 tools
**Mục tiêu:** 3 model architectures + 3 ablations + ensemble → bài thesis hoàn chỉnh

---

## 📊 Phase distribution (đã audit, imbalance 11.2×)

| Phase | % | Phase | % |
|---|---|---|---|
| CalotTriangleDissection | 40.9% | Cleaning | 8.2% |
| GallbladderDissection | 30.8% | Clipping | 7.7% |
| Preparation | 4.9% | GallbladderPackaging | 4.0% |
| GallbladderRetraction | 3.7% |

→ Tất cả configs đã set `class_weights: "auto"` + `label_smoothing: 0.1` để chống imbalance.

---

## 🎯 Configurations đã sẵn sàng

### Main models (3 thí nghiệm chính)

| ID | Config | Backbone | Temporal | Params | ETA |
|---|---|---|---|---|---|
| **M1** | `configs/m1_resnet_lstm.yaml` | ResNet50 (25M) | LSTM (2-layer Bi) | ~35M | **~15h** |
| **M2** | `configs/m2_efficientnet_tcn.yaml` | EfficientNet-B3 (12M) | TCN (2-stage) | ~20M | **~12h** |
| **M3** | `configs/m3_swin_transformer.yaml` | Swin-Tiny (28M) | Transformer (3-layer) | ~32M | **~20h** |

### Ablations (chứng minh từng design choice)

| ID | Config | So với baseline (M1) | ETA |
|---|---|---|---|
| **A1** | `abl_no_multitask.yaml` | Bỏ tool detection | ~8h |
| **A2** | `abl_no_class_weights.yaml` | Bỏ class weighting | ~8h |
| **A3** | `abl_seqlen16.yaml` | seq_len 16 thay 8 | ~10h |
| **A4** | `abl_no_smoothing.yaml` | Bỏ temporal smoothing | **~5 phút** (chỉ eval lại) |

**Tổng compute time:** ~78h training + ~2h eval = **~80h** (~5-6 ngày)

---

## 📅 SCHEDULE chi tiết (hôm nay = thứ Bảy 23/05/2026)

### **Ngày 1 — Sat 23/05 (chiều/tối)**

| Giờ | Việc |
|---|---|
| 16:30 | Pipeline `prepare_dataset.py` xong (verify zip đã xóa) |
| 16:30–17:00 | Chạy `verify_data.py` + `audit_data.py` lần cuối |
| 17:00 | **Khởi động M1**: `.\start_training.ps1`  (hoặc `.\run_experiments.ps1 -Only m1`) |
| 17:00–23:00 | M1 chạy stage 1 (backbone frozen, 5 epochs) |
| 23:00 | Đi ngủ — M1 tiếp tục chạy qua đêm |

**Mục tiêu cuối ngày:** M1 hoàn thành stage 1 + bắt đầu stage 2.

---

### **Ngày 2 — Sun 24/05**

| Giờ | Việc |
|---|---|
| 08:00 | Check `results/resnet50_lstm/training.log` — kiểm tra loss, val_f1 |
| Sáng | M1 đang chạy stage 2-3 |
| ~13:00 | **M1 dự kiến xong** (early stop có thể kích hoạt sớm hơn) |
| 13:00–14:00 | Verify M1 results, save best checkpoint |
| 14:00 | **Khởi động M2**: `.\run_experiments.ps1 -Only m2` |
| 14:00–23:00 | M2 chạy |
| 23:00 | Đi ngủ — M2 chạy qua đêm |

**Mục tiêu cuối ngày:** M1 done ✓ · M2 đang chạy stage 2.

---

### **Ngày 3 — Mon 25/05**

| Giờ | Việc |
|---|---|
| 08:00 | Check M2 log |
| ~10:00 | **M2 dự kiến xong** |
| 10:00–11:00 | Verify M2 |
| 11:00 | **Khởi động M3**: `.\run_experiments.ps1 -Only m3` (Swin là model nặng nhất) |
| Cả ngày | M3 chạy (~20h) |

**Mục tiêu cuối ngày:** M2 done ✓ · M3 đang chạy.

---

### **Ngày 4 — Tue 26/05**

| Giờ | Việc |
|---|---|
| ~07:00 | **M3 dự kiến xong** (qua đêm) |
| 08:00–09:00 | Verify M3 + tổng kết 3 models chính |
| 09:00 | **Khởi động Ablation 1 (no_multitask)**: `.\run_experiments.ps1 -Only abl` |
|       | (Script sẽ tự chạy A1 → A2 → A3 sequential, tổng ~26h) |
| Cả ngày | A1 chạy (~8h) |
| ~17:00 | A1 xong, A2 bắt đầu |
| Tối + qua đêm | A2 chạy |

---

### **Ngày 5 — Wed 27/05**

| Giờ | Việc |
|---|---|
| ~02:00 (qua đêm) | A2 xong, A3 bắt đầu |
| ~12:00 | A3 dự kiến xong |
| 12:00–13:00 | Verify ablations + tổng kết bảng so sánh |
| 13:00 | **Khởi động Ensemble evaluation**: `.\run_experiments.ps1 -Only ensemble` |
| 13:30 | Ensemble xong (~30 phút) — xem `results/ensemble/ensemble_results.json` |
| 14:00 | **Khởi động A4 (no_smoothing eval)** — eval lại M1 không smoothing (~5 phút) |
| Chiều/tối | **Bắt đầu viết phần Kết quả của thesis** |

---

### **Ngày 6 — Thu 28/05**

| Giờ | Việc |
|---|---|
| Cả ngày | Polishing: |
|         | - Generate confusion matrices: code đã tự làm trong `figures/` |
|         | - Phase timeline visualizations per video |
|         | - Bảng so sánh chi tiết (M1 vs M2 vs M3 vs ablations) |
|         | - Viết Methods + Results sections |

---

### **Ngày 7 — Fri 29/05 (BUFFER DAY)**

Dự phòng cho:
- Re-train nếu có model under-perform
- Thử ensemble strategy khác (stacking, per-phase routing)
- Sửa bug phát hiện muộn
- Polish hình ảnh cho thesis

---

## 🚦 Daily checklist mỗi sáng

```powershell
cd D:\Surgical_AI

# 1. Check training đêm qua thế nào
Get-Content results\*\training.log -Tail 30

# 2. Check GPU temp/util
nvidia-smi

# 3. Verify checkpoint mới nhất
Get-ChildItem results\*\checkpoints\*.pt | Sort LastWriteTime -Desc | Select -First 5

# 4. Disk space
Get-PSDrive D | Select Used, Free
```

---

## 🚨 Khi có vấn đề

### Training crash giữa chừng
```powershell
# Resume từ checkpoint mới nhất — run_experiments.ps1 tự detect
.\run_experiments.ps1
```

### OOM (Out of Memory)
- Giảm `batch_size` trong config xuống 2
- Giảm `sequence_length` xuống 4

### Train chậm hơn ETA quá nhiều
- Check GPU temp > 85°C → thermal throttle
- Đặt laptop chỗ thoáng + cooling pad
- Hoặc `--epochs 15` thay vì để 25 (early stop sẽ kích sớm)

### Val F1 không cải thiện
- Kiểm tra learning rate (có thể quá cao/thấp)
- Verify class weights compute đúng
- Plot loss curves để xem overfit hay underfit

---

## 📁 Output structure cuối cùng

```
results/
├── resnet50_lstm/                  # M1
│   ├── checkpoints/best.pt
│   ├── training.log
│   ├── config.yaml
│   ├── figures/training_curves.png
│   ├── figures/confusion_matrix.png
│   └── test_results/
├── efficientnet_b3_tcn/            # M2
├── swin_tiny_transformer/          # M3
├── resnet50_lstm_no_multitask/     # A1
├── resnet50_lstm_no_class_weights/ # A2
├── resnet50_lstm_seq16/            # A3
└── ensemble/
    └── ensemble_results.json       # final comparison
```

---

## 📝 Sections thesis sẽ viết được sau plan này

1. **Dataset Analysis** — phase distribution, class imbalance, train/val/test split
2. **Methods**
   - Backbone selection (CNN classic vs CNN efficient vs Transformer)
   - Temporal modeling (RNN vs Convolutional vs Self-attention)
   - Multi-task learning (phase + tool)
   - Class weighting + label smoothing
   - Temporal post-processing
3. **Experiments**
   - M1/M2/M3 individual results (Table 1)
   - Ablation studies (Table 2): multi-task, class weights, seq_len, smoothing
   - Ensemble results (Table 3): simple avg, weighted, voting
4. **Per-phase analysis** — F1 by phase, confusion matrix
5. **Qualitative results** — phase timeline visualizations, Grad-CAM (code đã có)

---

## ✅ Pre-flight checklist (làm trước khi bắt đầu chiều nay)

- [ ] Pipeline `prepare_dataset.py` báo 80/80 done + zip đã xóa
- [ ] `verify_data.py` pass
- [ ] `audit_data.py` không có anomaly
- [ ] `smoke_test.py` pass (đã làm rồi)
- [ ] Laptop cắm điện, chỗ thoáng, có cooling pad nếu có
- [ ] Power settings: sleep disabled (script tự làm trong start_training.ps1)
- [ ] Đóng Chrome, Discord, các app nặng

---

## 🎬 Lệnh để start sau khi data extract xong

```powershell
cd D:\Surgical_AI

# Option A — chạy toàn bộ sequential (sẽ block terminal ~5 ngày)
.\run_experiments.ps1

# Option B — từng stage (kiểm soát hơn)
.\run_experiments.ps1 -Only m1     # train M1
.\run_experiments.ps1 -Only m2     # rồi M2
.\run_experiments.ps1 -Only m3     # rồi M3
.\run_experiments.ps1 -Only abl    # ablations
.\run_experiments.ps1 -Only ensemble  # ensemble eval

# Option C — chỉ M1 hôm nay, mai tính tiếp
.\start_training.ps1
```

**Khuyến nghị:** Dùng **Option C** chiều nay (nhẹ nhàng, kiểm soát), từ ngày mai chuyển sang **Option B** từng stage.
