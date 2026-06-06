"""Generate publication-quality figures for the causal-decoding paper.

All figures are rendered from the actual artifacts:
  - results/cache/logits_*.npz   (per-frame logits + labels, val & test)
  - results/causal_benchmark.json
  - results/boundary_breakdown.json
  - results/significance.json

Outputs -> results/figures/*.png (300 dpi) and *.pdf (vector, for LaTeX).

Figures:
  fig1_reliability   Reliability diagrams (calibration) before/after T-scaling, 3 models.
  fig2_benchmark     Grouped bar chart: accuracy + macro-F1 per decoder per model.
  fig3_boundary      Boundary vs interior accuracy (the key finding), M3.
  fig4_timeline      Qualitative video70 timeline: GT vs raw vs causal-decoded.
  fig5_transition    Learned transition-matrix heatmap + significance/effect-size.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.causal_decode import (
    log_softmax, estimate_transition_matrix, make_causal_hmm, decode_video_causal,
)
from src.evaluation.postprocess import monotonic_decode, softmax as _softmax

# ------------------------------------------------------------------ style ---
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# Phase palette (matches the web demo / CSS).
PHASE_COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]
PHASE_SHORT = ["Prep", "Calot", "Clip", "Dissect", "Pack", "Clean", "Retract"]
PHASE_FULL = ["Preparation", "Calot triangle", "Clipping & cutting",
              "Gallbladder dissection", "Packaging", "Cleaning & coag.", "Retraction"]

# Accent colours for methods.
C_RAW = "#94a3b8"       # slate (baseline)
C_MEDIAN = "#64748b"    # darker slate
C_PROPOSED = "#0d9488"  # teal (our method)
C_PROPOSED2 = "#6d28d9" # violet (our method variant)
C_OFFLINE = "#f59e0b"   # amber (upper bound)

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "results" / "cache"
FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

MODELS = ["M1_resnet_lstm", "M2_effnet_tcn", "M3_swin_transformer"]
MODEL_TITLE = {
    "M1_resnet_lstm": "M1 · ResNet-50 + BiLSTM",
    "M2_effnet_tcn": "M2 · EfficientNet-B3 + TCN",
    "M3_swin_transformer": "M3 · Swin-Tiny + Transformer",
}

BENCH = json.loads((ROOT / "results" / "causal_benchmark.json").read_text())
BND = json.loads((ROOT / "results" / "boundary_breakdown.json").read_text())
SIG = json.loads((ROOT / "results" / "significance.json").read_text())
TEMPS = BENCH["temperatures"]


def load_cache(model, split):
    z = np.load(CACHE / f"logits_{model}_{split}.npz", allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith("logits_"):
            vid = int(k.split("_")[1])
            out[vid] = (z[k], z[f"labels_{vid}"])
    return out


def concat_split(model, split):
    data = load_cache(model, split)
    L = np.concatenate([data[v][0] for v in sorted(data)])
    Y = np.concatenate([data[v][1] for v in sorted(data)])
    return L, Y


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# =========================================================== FIG 1 ===========
def fig1_reliability(n_bins=12):
    """Reliability diagrams before/after temperature scaling, per model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = 1.0 / n_bins * 0.92

    for ax, model in zip(axes, MODELS):
        L, Y = concat_split(model, "val")
        mask = Y >= 0
        L, Y = L[mask], Y[mask].astype(int)
        T = TEMPS[model]

        def bin_acc_conf(logits):
            p = np.exp(log_softmax(logits))
            conf = p.max(1); pred = p.argmax(1)
            correct = (pred == Y).astype(float)
            accs = np.full(n_bins, np.nan); confs = np.full(n_bins, np.nan); wts = np.zeros(n_bins)
            for b in range(n_bins):
                m = (conf > edges[b]) & (conf <= edges[b + 1])
                if m.sum():
                    accs[b] = correct[m].mean(); confs[b] = conf[m].mean(); wts[b] = m.sum()
            ece = np.nansum(wts / wts.sum() * np.abs(np.nan_to_num(accs) - np.nan_to_num(confs)))
            return accs, ece

        acc_raw, ece_raw = bin_acc_conf(L)
        acc_cal, ece_cal = bin_acc_conf(L / T)

        ax.plot([0, 1], [0, 1], "--", color="#cbd5e1", lw=1.4, zorder=1, label="perfect")
        ax.bar(centers, np.nan_to_num(acc_raw), width=width, color=C_RAW, alpha=0.55,
               edgecolor="white", linewidth=0.5, label=f"raw (ECE {ece_raw:.3f})", zorder=2)
        ax.plot(centers, acc_cal, "-o", color=C_PROPOSED, lw=2, ms=4,
                label=f"calibrated T={T:.2f} (ECE {ece_cal:.3f})", zorder=3)

        ax.set_title(MODEL_TITLE[model], fontsize=11)
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper left", fontsize=8.3)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Calibration before and after temperature scaling (validation set)",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "fig1_reliability")


# =========================================================== FIG 2 ===========
def fig2_benchmark():
    """Grouped bars: accuracy (top) and macro-F1 (bottom) per decoder per model."""
    decoders = ["argmax_raw", "median15", "causal_monotonic_cal", "causal_hmm_cal", "offline_monotonic"]
    dlabels = ["Raw\npredictions", "Median-15", "Fixed-order\n+ cal", "Proposed\ndecoder", "Offline\n(full video) †"]
    dcolors = [C_RAW, C_MEDIAN, C_PROPOSED2, C_PROPOSED, C_OFFLINE]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True)
    x = np.arange(len(MODELS))
    n = len(decoders)
    bw = 0.16

    for row, (metric, ax, ylab) in enumerate([
            ("acc", axes[0], "Accuracy (%)"),
            ("macroF1", axes[1], "Macro-F1")]):
        for j, (dec, lab, col) in enumerate(zip(decoders, dlabels, dcolors)):
            vals = []
            for m in MODELS:
                v = BENCH["results_per_model"][m][dec][metric]
                vals.append(v * 100 if metric == "acc" else v)
            offset = (j - (n - 1) / 2) * bw
            hatch = "//" if dec == "offline_monotonic" else None
            bars = ax.bar(x + offset, vals, bw, color=col, label=lab if row == 0 else None,
                          edgecolor="white", linewidth=0.7, hatch=hatch, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if metric == "acc" else 0.004),
                        f"{v:.1f}" if metric == "acc" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.2, color="#334155")
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_TITLE[m].split(" · ")[0] + "\n" + MODEL_TITLE[m].split(" · ")[1]
                            for m in MODELS])
        if metric == "acc":
            ax.set_ylim(78, 93)
        else:
            ax.set_ylim(0.70, 0.88)

    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=9)
    fig.suptitle("Decoder comparison on the Cholec80 test set (20 videos)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02, "† Offline decoder uses the whole video and is only an upper bound.",
             ha="center", fontsize=8.5, color="#64748b", style="italic")
    save(fig, "fig2_benchmark")


# =========================================================== FIG 3 ===========
def fig3_boundary():
    """Boundary vs interior accuracy for M3 — the key finding."""
    m3 = BND["M3_swin_transformer"]
    decoders = ["argmax_raw", "median15", "causal_hmm_cal", "offline_monotonic"]
    dlabels = ["Raw predictions", "Median-15", "Proposed decoder", "Offline (full video) †"]
    dcolors = [C_RAW, C_MEDIAN, C_PROPOSED, C_OFFLINE]

    # boundary_breakdown.json stores accuracy as a fraction (0..1); scale to %.
    bnd_acc = [m3[d]["boundary"]["acc"] * 100 for d in decoders]
    int_acc = [m3[d]["interior"]["acc"] * 100 for d in decoders]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(decoders)); bw = 0.36

    b1 = ax.bar(x - bw/2, bnd_acc, bw, color=dcolors, alpha=0.55, edgecolor="white",
                linewidth=0.8, label="Boundary (±5 s of a transition)", zorder=3)
    b2 = ax.bar(x + bw/2, int_acc, bw, color=dcolors, edgecolor="white",
                linewidth=0.8, label="Interior (within a stable phase)", zorder=3,
                hatch="..")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8.5, color="#334155")

    ax.set_xticks(x); ax.set_xticklabels(dlabels)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(55, 95)

    # custom legend (fill vs hatch)
    legend_elems = [
        Patch(facecolor="#888", alpha=0.55, label="Boundary (±5 s of a transition)"),
        Patch(facecolor="#888", hatch="..", label="Interior (within a stable phase)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left")

    # The two findings (proposed is best near phase changes; the offline decoder's
    # advantage is entirely inside phases) are explained in the caption and the text,
    # so no on-plot annotations are drawn -- they would only overlap the tall bars.

    ax.set_title("Accuracy near phase changes vs. inside phases (M3)")
    fig.text(0.5, -0.03, "† Non-causal upper bound. Boundary frames are hard for all methods; "
             "the proposed causal decoder is strongest there.",
             ha="center", fontsize=8.5, color="#64748b", style="italic")
    save(fig, "fig3_boundary")


# =========================================================== FIG 4 ===========
def fig4_timeline(video=70):
    """Qualitative timeline for one test video: GT vs raw argmax vs causal decoded."""
    model = "M3_swin_transformer"
    data = load_cache(model, "test")
    if video not in data:
        # fall back to any available test video
        video = sorted(data)[0]
    logits, _ = data[video]

    # ground truth from annotation file at 1 fps
    ann = ROOT / "data" / "cholec80" / f"video{video:02d}" / "phase_annotations.txt"
    gt = []
    with ann.open() as f:
        next(f)
        for i, line in enumerate(f):
            if i % 25 != 0:
                continue
            p = line.strip().split()
            if len(p) >= 2 and p[1].isdigit():
                gt.append(int(p[1]))
    gt = np.array(gt)
    T = min(len(gt), len(logits))
    gt = gt[:T]; logits = logits[:T]

    raw = logits.argmax(1)
    A = estimate_transition_matrix([gt], smoothing=1.0)  # illustrative; paper uses train-set A
    # use the train-set transition matrix actually used in the benchmark
    A = np.array(BENCH["transition_matrix_learned"])
    dec = decode_video_causal(make_causal_hmm(A, temperature=TEMPS[model]), logits)

    def acc(p): return (p[:T] == gt).mean() * 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 4.2), sharex=True)
    rows = [("Ground truth", gt, None),
            (f"Raw predictions  ({acc(raw):.1f}%)", raw, None),
            (f"Proposed decoder  ({acc(dec):.1f}%)", dec, None)]
    for ax, (label, seq, _) in zip(axes, rows):
        for t in range(T):
            ax.axvspan(t, t + 1, color=PHASE_COLORS[int(seq[t])], lw=0)
        ax.set_yticks([]); ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_xlim(0, T)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("Time (seconds, 1 fps)")
    axes[-1].spines["bottom"].set_visible(True)

    legend_elems = [Patch(facecolor=PHASE_COLORS[i], label=PHASE_FULL[i]) for i in range(7)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=7, fontsize=8.2,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"Phase timeline for test video {video} (M3)",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig4_timeline")


# =========================================================== FIG 5 ===========
def fig5_transition_and_significance():
    """Left: learned transition-matrix heatmap. Right: per-video gain + effect size."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})

    # --- transition matrix heatmap ---
    A = np.array(BENCH["transition_matrix_learned"])
    im = axL.imshow(A, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    axL.set_xticks(range(7)); axL.set_yticks(range(7))
    axL.set_xticklabels(PHASE_SHORT, rotation=45, ha="right")
    axL.set_yticklabels(PHASE_SHORT)
    axL.set_xlabel("To phase  (t+1)"); axL.set_ylabel("From phase  (t)")
    axL.set_title("Learned transition table (per second)")
    for i in range(7):
        for j in range(7):
            v = A[i, j]
            if v >= 0.005:
                axL.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                         color="white" if v < 0.6 else "black", fontsize=7.5)
    axL.grid(False)
    cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)
    cb.set_label("P(phase$_{t+1}$ | phase$_t$)", fontsize=9)

    # --- per-video gain + effect size ---
    models = MODELS
    deltas = [SIG[m]["delta_acc_pct"] for m in models]
    ds = [SIG[m]["cohens_d_acc"] for m in models]
    ps = [SIG[m]["wilcoxon_acc_p"] for m in models]
    colors = [C_PROPOSED2, C_PROPOSED2, C_PROPOSED2]

    y = np.arange(len(models))
    short = {"M1_resnet_lstm": "M1\nResNet+LSTM",
             "M2_effnet_tcn": "M2\nEffNet+TCN",
             "M3_swin_transformer": "M3\nSwin+Transf."}
    bars = axR.barh(y, deltas, color=C_PROPOSED, alpha=0.85, edgecolor="white",
                    height=0.5, zorder=3)
    axR.set_yticks(y)
    axR.set_yticklabels([short[m] for m in models], fontsize=9)
    axR.set_xlabel("Accuracy gain over raw predictions (percentage points)")
    axR.set_xlim(0, max(deltas) * 1.7)
    axR.set_title("Per-video improvement of the proposed decoder")
    for b, dlt, d, p in zip(bars, deltas, ds, ps):
        pstr = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        axR.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2,
                 f"+{dlt:.2f} pp   (d = {d:.2f}, {pstr})",
                 va="center", fontsize=9, color="#334155")
    axR.axvline(0, color="#94a3b8", lw=0.8)

    fig.suptitle("Learned transition table (left) and per-video gains (right)",
                 fontsize=13, fontweight="bold", y=1.0)
    save(fig, "fig5_transition_significance")


def main():
    print("Generating figures into results/figures/ ...")
    fig1_reliability()
    fig2_benchmark()
    fig3_boundary()
    fig4_timeline(video=70)
    fig5_transition_and_significance()
    print("Done. Files:")
    for f in sorted(FIGDIR.glob("*.png")):
        print("  ", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
