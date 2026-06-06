"""Generate the supplementary thesis figures (front-matter style, à la the
skin-lesion thesis template) from the real Cholec80 artefacts.

Outputs -> results/figures/thesis/*.png (300 dpi) + *.pdf.

Figures:
  fig_class_dist     Phase frequency across train/val/test splits (bar chart).
  fig_pipeline       Training-pipeline block diagram.
  fig_transfer       Backbone + temporal-head + multi-task-head schematic.
  fig_train_history  Loss / accuracy / macro-F1 curves for M1, M2, M3.
  fig_confusion_m3   Normalised confusion matrix for the best model (M3).
  fig_arch_compare   Parameters-vs-macro-F1 scatter for the four-way comparison.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 11, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})

ROOT = Path(__file__).parent.parent
FIG = ROOT / "results" / "figures" / "thesis"
FIG.mkdir(parents=True, exist_ok=True)

PHASE_SHORT = ["Prep", "Calot", "Clip", "Dissect", "Pack", "Clean", "Retract"]
PHASE_FULL = ["Preparation", "Calot triangle dissection", "Clipping & cutting",
              "Gallbladder dissection", "Gallbladder packaging",
              "Cleaning & coagulation", "Gallbladder retraction"]
PHASE_COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]
TEAL, VIOLET, SLATE = "#0d9488", "#6d28d9", "#64748b"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved thesis/{name}")


def load_train_labels(ids, src_fps=25, tgt_fps=1):
    stride = src_fps // tgt_fps
    counts = np.zeros(7, dtype=int)
    for vid in ids:
        ann = ROOT / "data" / "cholec80" / f"video{vid:02d}" / "phase_annotations.txt"
        if not ann.exists():
            continue
        with ann.open() as f:
            next(f, None)
            for i, line in enumerate(f):
                if i % stride:
                    continue
                p = line.strip().split()
                if len(p) >= 2 and p[1].isdigit():
                    counts[int(p[1])] += 1
    return counts


# ============================================================ class dist =====
def fig_class_dist():
    tr = load_train_labels(range(1, 41))
    va = load_train_labels(range(41, 61))
    te = load_train_labels(range(61, 81))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(7); w = 0.26
    ax.bar(x - w, tr, w, label=f"train (n={tr.sum():,})", color=TEAL, edgecolor="white")
    ax.bar(x,     va, w, label=f"val (n={va.sum():,})",   color=VIOLET, edgecolor="white")
    ax.bar(x + w, te, w, label=f"test (n={te.sum():,})",  color="#f59e0b", edgecolor="white")
    for i in range(7):
        ax.text(i - w, tr[i] + tr.max() * 0.01, f"{tr[i]}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(PHASE_SHORT)
    ax.set_ylabel("Frames (1 fps)"); ax.set_xlabel("Surgical phase")
    ax.set_title("Number of frames per phase (train / val / test)")
    ax.legend()
    pct = tr / tr.sum() * 100
    ax.text(0.99, 0.95, f"imbalance ratio  {tr.max()/tr.min():.1f}:1\n"
            f"dominant {PHASE_SHORT[pct.argmax()]} {pct.max():.1f}%  ·  "
            f"rarest {PHASE_SHORT[pct.argmin()]} {pct.min():.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f8fafc", ec="#e2e8f0"))
    save(fig, "fig_class_dist")


# ============================================================ pipeline ========
def _box(ax, x, y, w, h, text, fc, tc="#1c1917", fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                fc=fc, ec="#94a3b8", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color="#475569"))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    ys, h = 1.1, 1.2
    _box(ax, 0.2, ys, 1.9, h, "Input sequence\n8 frames @ 1 fps\n224×224", "#e0f2fe")
    _box(ax, 2.5, ys, 2.0, h, "Augmentation\nflip · rotate\ncrop · jitter", "#ede9fe")
    _box(ax, 4.9, ys, 2.0, h, "Pretrained\nbackbone\n(CNN / ViT)", "#dcfce7")
    _box(ax, 7.3, ys, 2.0, h, "Temporal model\nLSTM / TCN /\nTransformer", "#fef3c7")
    _box(ax, 9.7, ys, 2.1, h, "Multi-task head\nphase (7) +\ntools (7)", "#fee2e2")
    for x in (2.1, 4.5, 6.9, 9.3):
        _arrow(ax, x, ys + h / 2, x + 0.4, ys + h / 2)
    # bottom: optimisation strip
    _box(ax, 4.9, 0.05, 6.9, 0.7,
         "AdamW · cosine warm-restart · class-weighted CE + label smoothing 0.1 · "
         "BCE tools (λ=0.5) · mixed precision",
         "#f1f5f9", fs=8.3)
    _arrow(ax, 8.3, ys, 8.3, 0.75)
    ax.set_title("Training pipeline", fontsize=12, fontweight="bold")
    save(fig, "fig_pipeline")


def fig_transfer():
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3); ax.axis("off")
    _box(ax, 0.3, 1.0, 2.3, 1.1, "ImageNet\npretrained\nweights", "#e0f2fe")
    _box(ax, 3.1, 1.0, 2.6, 1.1, "Frozen → fine-tuned\nbackbone\n(3-stage curriculum)", "#dcfce7")
    _box(ax, 6.2, 1.0, 2.3, 1.1, "Temporal\nencoder\n(8-frame context)", "#fef3c7")
    _box(ax, 9.0, 1.0, 1.7, 1.1, "Phase +\nTool heads", "#fee2e2")
    for x in (2.6, 5.7, 8.5):
        _arrow(ax, x, 1.55, x + 0.5, 1.55)
    ax.set_title("Transfer learning and fine-tuning", fontsize=12, fontweight="bold")
    save(fig, "fig_transfer")


# ============================================================ train history ===
def _load_hist(model):
    h = json.loads((ROOT / "results" / model / "training_history.json").read_text())
    ep = [e["epoch"] for e in h]
    tl = [e["train"]["total_loss"] for e in h]
    vl = [e["val"]["total_loss"] for e in h]
    ta = [e["train"]["accuracy"] for e in h]
    va = [e["val"]["accuracy"] for e in h]
    return ep, tl, vl, ta, va


def fig_train_history():
    models = [("resnet50_lstm", "M1 · ResNet-50 + BiLSTM"),
              ("efficientnet_b3_tcn", "M2 · EfficientNet-B3 + TCN"),
              ("swin_tiny_transformer", "M3 · Swin-Tiny + Transformer")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
    for col, (m, title) in enumerate(models):
        try:
            ep, tl, vl, ta, va = _load_hist(m)
        except Exception as e:
            print(f"  skip {m}: {e}"); continue
        ax = axes[0, col]
        ax.plot(ep, tl, color=TEAL, lw=1.8, label="train")
        ax.plot(ep, vl, color="#f59e0b", lw=1.8, label="val")
        ax.set_title(title, fontsize=10.5); ax.set_ylabel("Total loss" if col == 0 else "")
        ax.legend(fontsize=8)
        ax2 = axes[1, col]
        ax2.plot(ep, ta, color=TEAL, lw=1.8, label="train")
        ax2.plot(ep, va, color="#f59e0b", lw=1.8, label="val")
        ax2.set_ylabel("Accuracy" if col == 0 else ""); ax2.set_xlabel("Epoch")
        ax2.set_ylim(0, 1.0); ax2.legend(fontsize=8)
    fig.suptitle("Training and validation curves for the three models", fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig_train_history")


# ============================================================ confusion =======
def fig_confusion_m3():
    # Recompute M3 smoothed confusion on the full test set from cached logits.
    from scipy.ndimage import median_filter
    z = np.load(ROOT / "results" / "cache" / "logits_M3_swin_transformer_test.npz", allow_pickle=True)
    yt_all, yp_all = [], []
    for k in z.files:
        if not k.startswith("logits_"):
            continue
        vid = k.split("_")[1]
        lg = z[k]; lab = z[f"labels_{vid}"]
        pred = median_filter(lg.argmax(1).astype(int), size=15, mode="nearest")
        m = lab >= 0
        yt_all.append(lab[m]); yp_all.append(pred[m])
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    cm = np.zeros((7, 7))
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(PHASE_SHORT, rotation=45, ha="right"); ax.set_yticklabels(PHASE_SHORT)
    ax.set_xlabel("Predicted phase"); ax.set_ylabel("True phase")
    for i in range(7):
        for j in range(7):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        color="white" if v > 0.55 else "#1c1917", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Recall (row-normalised)")
    ax.set_title("Confusion matrix for M3 (smoothed, test set)")
    save(fig, "fig_confusion_m3")


# ============================================================ arch compare ====
def fig_arch_compare():
    # params (M) vs smoothed macro-F1 from the report's main table
    data = [("M1 ResNet-50+BiLSTM", 35.7, 0.781, "#0ea5e9"),
            ("M2 EffNet-B3+TCN", 11.2, 0.748, "#f59e0b"),
            ("M3 Swin-Tiny+Transformer", 31.6, 0.815, "#ef4444"),
            ("Ensemble (3 models)", 78.5, 0.809, "#10b981")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, p, f1, c in data:
        ax.scatter(p, f1, s=160, color=c, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(name, (p, f1), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Smoothed macro-F1 (test set)")
    ax.set_title("Model size vs. macro-F1")
    ax.set_ylim(0.72, 0.83)
    ax.axhline(0.815, color="#cbd5e1", ls="--", lw=1, zorder=1)
    save(fig, "fig_arch_compare")


def main():
    print("Generating thesis figures …")
    fig_class_dist()
    fig_pipeline()
    fig_transfer()
    fig_train_history()
    fig_confusion_m3()
    fig_arch_compare()
    print("Done ->", FIG)


if __name__ == "__main__":
    main()
