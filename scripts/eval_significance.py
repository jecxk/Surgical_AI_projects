"""Per-video paired significance test for causal decoder improvements.

For each model, compute per-video accuracy under (i) argmax_raw and
(ii) causal_hmm_cal, then run a Wilcoxon signed-rank test on the 20 paired
differences. Also report mean +/- std and Cohen's d (effect size).

Reviewer-defense: a +0.8% accuracy gain looks tiny in aggregate but may be
consistent across videos. Significance test answers that.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.causal_decode import (
    estimate_transition_matrix, make_causal_hmm, decode_video_causal,
)
from scripts.eval_causal import MODELS, CACHE_DIR, load_train_labels


def load_cached(name, split):
    z = np.load(CACHE_DIR / f"logits_{name}_{split}.npz", allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith("logits_"):
            vid = int(k.split("_")[1])
            out[vid] = (z[k], z[f"labels_{vid}"])
    return out


def per_video_metric(pred_fn, per_vid):
    accs, f1s = [], []
    for vid, (lg, lab) in sorted(per_vid.items()):
        m = lab >= 0
        if m.sum() == 0:
            continue
        pred = pred_fn(lg)
        accs.append(accuracy_score(lab[m], pred[m]))
        f1s.append(f1_score(lab[m], pred[m], average="macro",
                            labels=list(range(7)), zero_division=0))
    return np.array(accs), np.array(f1s)


def cohens_d(x, y):
    diff = x - y
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def main():
    bench = json.loads(Path("results/causal_benchmark.json").read_text())
    temperatures = bench["temperatures"]
    seqs = load_train_labels(list(range(1, 41)))
    A = estimate_transition_matrix(seqs, smoothing=1.0)

    out = {}
    print(f"\n{'Model':22s} {'dAcc%':>7s} {'dF1':>7s} {'p_acc':>9s} {'p_F1':>9s} {'d_acc':>7s}")
    print("-" * 70)
    for name in MODELS:
        T = temperatures[name]
        per_vid = load_cached(name, "test")
        acc_raw, f1_raw = per_video_metric(lambda lg: lg.argmax(1), per_vid)
        acc_cau, f1_cau = per_video_metric(
            lambda lg: decode_video_causal(make_causal_hmm(A, temperature=T), lg),
            per_vid,
        )
        d_acc = (acc_cau - acc_raw).mean() * 100
        d_f1 = (f1_cau - f1_raw).mean()
        # Wilcoxon paired signed-rank (one-sided, hypothesis: causal > raw)
        try:
            w_acc, p_acc = wilcoxon(acc_cau, acc_raw, alternative="greater")
        except ValueError:
            w_acc, p_acc = float("nan"), 1.0
        try:
            w_f1, p_f1 = wilcoxon(f1_cau, f1_raw, alternative="greater")
        except ValueError:
            w_f1, p_f1 = float("nan"), 1.0
        d = cohens_d(acc_cau, acc_raw)
        out[name] = {
            "n_videos": len(acc_raw),
            "acc_raw_mean": float(acc_raw.mean()), "acc_raw_std": float(acc_raw.std(ddof=1)),
            "acc_causal_mean": float(acc_cau.mean()), "acc_causal_std": float(acc_cau.std(ddof=1)),
            "f1_raw_mean": float(f1_raw.mean()), "f1_causal_mean": float(f1_cau.mean()),
            "delta_acc_pct": float(d_acc), "delta_f1": float(d_f1),
            "wilcoxon_acc_p": float(p_acc), "wilcoxon_f1_p": float(p_f1),
            "cohens_d_acc": float(d),
            "per_video_acc_raw": acc_raw.tolist(),
            "per_video_acc_causal": acc_cau.tolist(),
        }
        print(f"{name:22s} {d_acc:+7.3f} {d_f1:+7.3f} {p_acc:9.4f} {p_f1:9.4f} {d:+7.3f}")

    Path("results/significance.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> results/significance.json")
    print("\nInterpretation:")
    print("  p < 0.05 = significant at 5%; Cohen's d > 0.5 = medium effect, > 0.8 = large.")


if __name__ == "__main__":
    main()
