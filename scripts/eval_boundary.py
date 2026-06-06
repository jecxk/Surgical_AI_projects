"""Boundary vs interior error breakdown for the best causal decoder.

Reuses the cached logits from scripts/eval_causal.py and the same decoder
specifications. For each model and each decoder, reports:
  - boundary_acc / boundary_F1 (frames within 5s of phase transition)
  - interior_acc / interior_F1 (everywhere else)
  - all_acc / all_F1

This is the fair-evaluation pillar of the paper: annotation noise concentrates
at boundaries, so SOTA gaps shrink when interior frames are isolated.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.postprocess import monotonic_decode, softmax
from src.evaluation.causal_decode import (
    estimate_transition_matrix,
    make_causal_argmax, make_causal_monotonic, make_causal_hmm,
    decode_video_causal,
)
from src.evaluation.boundary_analysis import split_metrics

# reuse loaders from eval_causal
from scripts.eval_causal import (
    MODELS, CACHE_DIR, load_train_labels, median_filter,
)


def load_cached(name, split):
    z = np.load(CACHE_DIR / f"logits_{name}_{split}.npz", allow_pickle=True)
    out = {}
    for k in z.files:
        if k.startswith("logits_"):
            vid = int(k.split("_")[1])
            out[vid] = (z[k], z[f"labels_{vid}"])
    return out


def main():
    # learn transition matrix
    seqs = load_train_labels(list(range(1, 41)))
    print(f"Loaded {len(seqs)} train label sequences")
    A = estimate_transition_matrix(seqs, smoothing=1.0)

    # also need temperatures, pull from results json
    bench = json.loads(Path("results/causal_benchmark.json").read_text())
    temperatures = bench["temperatures"]
    print(f"Temperatures: {temperatures}")

    decoders = [
        ("argmax_raw",          lambda lg, T=1.0: lg.argmax(1)),
        ("median15",            lambda lg, T=1.0: median_filter(lg.argmax(1), 15)),
        ("offline_monotonic",   lambda lg, T=1.0: monotonic_decode(softmax(lg))),
        ("causal_monotonic_cal",lambda lg, T=1.0: decode_video_causal(
            make_causal_monotonic(stay=0.95, temperature=T), lg)),
        ("causal_hmm_cal",      lambda lg, T=1.0: decode_video_causal(
            make_causal_hmm(A, temperature=T), lg)),
    ]

    out = {}
    for name in MODELS:
        T = temperatures[name]
        per_vid = load_cached(name, "test")
        print(f"\n=== {name} (T*={T:.3f}) ===")
        out[name] = {}
        for tag, fn in decoders:
            yt_all, yp_all = [], []
            for vid, (lg, lab) in per_vid.items():
                preds = fn(lg, T)
                yt_all.append(lab); yp_all.append(preds)
            yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
            m = split_metrics(yt, yp, window=5)
            out[name][tag] = m
            b, i, a = m["boundary"], m["interior"], m["all"]
            print(f"  {tag:24s}  bnd {b['acc']*100:5.2f}/{b['macroF1']:.3f}  "
                  f"int {i['acc']*100:5.2f}/{i['macroF1']:.3f}  "
                  f"all {a['acc']*100:5.2f}/{a['macroF1']:.3f}")

    Path("results/boundary_breakdown.json").write_text(json.dumps(out, indent=2))
    print("\nSaved -> results/boundary_breakdown.json")


if __name__ == "__main__":
    main()
