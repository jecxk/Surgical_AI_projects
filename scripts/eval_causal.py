"""Benchmark causal decoders + calibration on Cholec80 test set.

Pipeline:
  1. Load 3 models (M1 ResNet-LSTM, M2 EffNet-TCN, M3 Swin-Transformer).
  2. Collect per-frame logits on val (videos 41-60) and test (61-80).
     Cache to results/cache/logits_<model>_<split>.npz for reuse.
  3. Estimate phase transition matrix from train labels (videos 1-40).
  4. Fit per-model temperature on val by minimizing NLL.
  5. Evaluate 6 decoders on test:
       - argmax (raw, baseline)
       - median-15 smoothing (causal-with-lag, current pipeline)
       - offline monotonic Viterbi (upper bound, NON-causal)
       - causal argmax (sanity, == raw)
       - causal monotonic (proposed, streaming Viterbi w/ monotonic prior)
       - causal HMM (proposed, streaming Viterbi w/ learned transition)
       + calibrated variants (T-scaled) for causal-mono and causal-hmm.
  6. Report acc, macro-F1, ECE, NLL, and per-frame latency.
  7. Save results/causal_benchmark.json + a markdown summary.

Designed to reuse the existing logit-collection from scripts/eval_postprocess.py.
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import Cholec80SequenceDataset
from src.dataset.transforms import get_val_transforms
from src.models.surgical_model import SurgicalPhaseModel
from src.evaluation.postprocess import monotonic_decode, softmax, TRAIN_PRIORS
from src.evaluation.causal_decode import (
    estimate_transition_matrix, monotonic_transition_matrix,
    make_causal_argmax, make_causal_monotonic, make_causal_hmm,
    decode_video_causal, fit_temperature, temperature_scale, log_softmax,
    expected_calibration_error, maximum_calibration_error,
    negative_log_likelihood,
)
from sklearn.metrics import accuracy_score, f1_score

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]
MODELS = {
    "M1_resnet_lstm":     "results/resnet50_lstm",
    "M2_effnet_tcn":      "results/efficientnet_b3_tcn",
    "M3_swin_transformer":"results/swin_tiny_transformer",
}
CACHE_DIR = Path("results/cache")
OUT_JSON  = Path("results/causal_benchmark.json")
OUT_MD    = Path("results/causal_benchmark.md")


def log(m): print(m, flush=True)


def load_model(run_dir, device):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = cfg.get("model", {}); mcfg["num_phases"] = 7; mcfg["num_tools"] = 7
    ckpt = torch.load(run_dir / "checkpoints" / "best_model.pth",
                      map_location="cpu", weights_only=False)
    model = SurgicalPhaseModel(mcfg); model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    seq_len = int(cfg.get("data", {}).get("sequence_length", 8))
    vf1 = ckpt.get("best_val_f1") or cfg.get("val_f1", 0.7)
    return model, seq_len, float(vf1)


@torch.no_grad()
def collect_logits(model, video_ids, seq_len, transform, device):
    """Return {vid: (logits [T,7], labels [T])}."""
    out = {}
    for vid in video_ids:
        ds = Cholec80SequenceDataset(
            data_root="data/cholec80", video_ids=[vid],
            transform=transform, sequence_length=seq_len, stride=seq_len, fps=1,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
        lg_all, lab_all = [], []
        for batch in loader:
            frames = batch["images"].to(device)
            o = model(frames)
            lg = o["phase_logits"].cpu().numpy()
            lab = batch["phases"].numpy()
            B, T, _ = lg.shape
            lg_all.append(lg.reshape(B*T, 7)); lab_all.append(lab.reshape(B*T))
        out[vid] = (np.concatenate(lg_all), np.concatenate(lab_all))
    return out


def cached_collect(name, run_dir, video_ids, split_tag, device, transform):
    """Load cached logits; collect any missing videos and append to cache."""
    cache_file = CACHE_DIR / f"logits_{name}_{split_tag}.npz"
    cached = {}
    if cache_file.exists():
        z = np.load(cache_file, allow_pickle=True)
        cached = {int(k.split("_")[1]): (z[f"logits_{k.split('_')[1]}"],
                                          z[f"labels_{k.split('_')[1]}"])
                  for k in z.files if k.startswith("logits_")}
    missing = [v for v in video_ids if v not in cached]
    if not missing:
        return {v: cached[v] for v in video_ids}

    log(f"  collecting {name}/{split_tag}: {len(missing)} missing of {len(video_ids)}")
    model, seq_len, _ = load_model(Path(run_dir), device)
    new_data = collect_logits(model, missing, seq_len, transform, device)
    cached.update(new_data)
    payload = {}
    for vid, (lg, lab) in cached.items():
        payload[f"logits_{vid}"] = lg
        payload[f"labels_{vid}"] = lab
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **payload)
    del model; torch.cuda.empty_cache()
    return {v: cached[v] for v in video_ids}


def median_filter(seq, window=15):
    seq = np.asarray(seq)
    T = len(seq); half = window // 2; out = seq.copy()
    for t in range(T):
        a = max(0, t - half); b = min(T, t + half + 1)
        vals, counts = np.unique(seq[a:b], return_counts=True)
        out[t] = vals[counts.argmax()]
    return out


PHASE_NAME_TO_ID = {
    "Preparation": 0, "CalotTriangleDissection": 1, "ClippingCutting": 2,
    "GallbladderDissection": 3, "GallbladderPackaging": 4,
    "CleaningCoagulation": 5, "GallbladderRetraction": 6,
}


def load_train_labels(video_ids, source_fps: int = 25, target_fps: int = 1):
    """Read phase annotations directly from text files, subsample to target fps.

    Annotations are at 25 fps with header 'Frame<TAB>Phase'. We take every
    25th row to match the 1 fps used by the model.
    """
    stride = source_fps // target_fps
    sequences = []
    for vid in video_ids:
        ann = Path(f"data/cholec80/video{vid:02d}/phase_annotations.txt")
        if not ann.exists():
            log(f"  no annotation for vid {vid}")
            continue
        labs = []
        with ann.open() as f:
            next(f, None)  # skip header
            for i, line in enumerate(f):
                if i % stride != 0:
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                p = parts[1]
                # accept both integer and name annotations
                if p.isdigit():
                    labs.append(int(p))
                elif p in PHASE_NAME_TO_ID:
                    labs.append(PHASE_NAME_TO_ID[p])
        if labs:
            sequences.append(np.array(labs, dtype=int))
    return sequences


def metrics_pair(y_true, y_pred):
    m = y_true >= 0
    yt, yp = y_true[m], y_pred[m]
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", labels=list(range(7)), zero_division=0)
    return acc, f1m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_videos",  default="41-60")
    ap.add_argument("--test_videos", default="61-80")
    ap.add_argument("--train_videos_for_transition", default="1-40")
    ap.add_argument("--skip_cache", action="store_true")
    args = ap.parse_args()

    def parse_range(s):
        lo, hi = s.split("-"); return list(range(int(lo), int(hi)+1))
    val_ids   = parse_range(args.val_videos)
    test_ids  = parse_range(args.test_videos)
    train_ids = parse_range(args.train_videos_for_transition)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Val ids:  {val_ids[:3]}..{val_ids[-1]}  ({len(val_ids)} videos)")
    log(f"Test ids: {test_ids[:3]}..{test_ids[-1]} ({len(test_ids)} videos)")

    transform = get_val_transforms(224)

    # ----- Step 1: collect logits per model per split (cached) -----
    log("\n[1/5] Collect logits (cached if available)")
    logits_by_model = {}
    for name, rd in MODELS.items():
        log(f"  {name}")
        val_data  = cached_collect(name, rd, val_ids,  "val",  device, transform)
        test_data = cached_collect(name, rd, test_ids, "test", device, transform)
        logits_by_model[name] = {"val": val_data, "test": test_data}

    # ----- Step 2: estimate transition matrix from train labels -----
    log("\n[2/5] Estimate transition matrix from train labels")
    train_label_seqs = load_train_labels(train_ids)
    log(f"  loaded {len(train_label_seqs)} train label sequences")
    A_learned = estimate_transition_matrix(train_label_seqs, smoothing=1.0)
    A_mono    = monotonic_transition_matrix(stay=0.95)
    log("  learned transition diag (P(stay)):")
    log("    " + "  ".join([f"{A_learned[i,i]:.3f}" for i in range(7)]))

    # ----- Step 3: fit temperature per model on val -----
    log("\n[3/5] Calibrate temperature per model on val")
    temperatures = {}
    val_metrics_raw = {}
    val_metrics_cal = {}
    for name in MODELS:
        all_logits, all_labels = [], []
        for vid, (lg, lab) in logits_by_model[name]["val"].items():
            all_logits.append(lg); all_labels.append(lab)
        L = np.concatenate(all_logits); Y = np.concatenate(all_labels)
        T = fit_temperature(L, Y)
        temperatures[name] = T
        probs_raw = np.exp(log_softmax(L))
        probs_cal = np.exp(log_softmax(L / T))
        ece_raw = expected_calibration_error(probs_raw, Y)
        ece_cal = expected_calibration_error(probs_cal, Y)
        nll_raw = negative_log_likelihood(probs_raw, Y)
        nll_cal = negative_log_likelihood(probs_cal, Y)
        val_metrics_raw[name] = {"ECE": ece_raw, "NLL": nll_raw}
        val_metrics_cal[name] = {"ECE": ece_cal, "NLL": nll_cal}
        log(f"  {name}: T*={T:.3f}  ECE {ece_raw:.4f} -> {ece_cal:.4f}  NLL {nll_raw:.4f} -> {nll_cal:.4f}")

    # ----- Step 4: evaluate decoders on test set per model -----
    log("\n[4/5] Evaluate decoders on test set")
    decoder_specs = [
        # (tag, builder_fn(name) -> decoder_fn(logits) -> preds, applies_temp)
        ("argmax_raw",        lambda n: lambda lg: lg.argmax(1),                                 False),
        ("median15",          lambda n: lambda lg: median_filter(lg.argmax(1), 15),              False),
        ("offline_monotonic", lambda n: lambda lg: monotonic_decode(softmax(lg)),                False),
        ("causal_argmax",     lambda n: lambda lg: decode_video_causal(make_causal_argmax(), lg), False),
        ("causal_monotonic",  lambda n: lambda lg: decode_video_causal(make_causal_monotonic(stay=0.95), lg), False),
        ("causal_hmm",        lambda n: lambda lg: decode_video_causal(make_causal_hmm(A_learned), lg), False),
        ("causal_monotonic_cal", lambda n: lambda lg: decode_video_causal(
            make_causal_monotonic(stay=0.95, temperature=temperatures[n]), lg), True),
        ("causal_hmm_cal",       lambda n: lambda lg: decode_video_causal(
            make_causal_hmm(A_learned, temperature=temperatures[n]), lg), True),
    ]

    results = defaultdict(dict)
    latency_ms = defaultdict(dict)
    for name in MODELS:
        log(f"\n  --- {name} (T*={temperatures[name]:.3f}) ---")
        per_vid_logits = logits_by_model[name]["test"]
        for tag, builder, _ in decoder_specs:
            decode_fn = builder(name)
            yt_all, yp_all = [], []
            t_sum, n_frames = 0.0, 0
            for vid, (lg, lab) in per_vid_logits.items():
                t0 = time.perf_counter()
                preds = decode_fn(lg)
                t_sum += time.perf_counter() - t0
                n_frames += lg.shape[0]
                yt_all.append(lab); yp_all.append(preds)
            yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
            acc, f1m = metrics_pair(yt, yp)
            ms_per_frame = (t_sum / n_frames) * 1000.0
            results[name][tag] = {"acc": acc, "macroF1": f1m,
                                  "ms_per_frame": ms_per_frame, "n_frames": n_frames}
            latency_ms[name][tag] = ms_per_frame
            log(f"    {tag:24s}  acc={acc*100:5.2f}%  macroF1={f1m:.3f}  {ms_per_frame:6.3f} ms/frame")

    # ----- Step 5: write outputs -----
    log("\n[5/5] Write JSON + Markdown")
    payload = {
        "temperatures": temperatures,
        "val_calibration": {"raw": val_metrics_raw, "calibrated": val_metrics_cal},
        "transition_matrix_learned": A_learned.tolist(),
        "results_per_model": {k: dict(v) for k, v in results.items()},
        "phase_names": PHASE_NAMES,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    log(f"  -> {OUT_JSON}")

    # Markdown summary
    lines = ["# Causal decoding + calibration benchmark", "",
             "## Per-model temperature and calibration (val set)", ""]
    lines.append("| Model | T* | ECE raw | ECE cal | NLL raw | NLL cal |")
    lines.append("|---|---|---|---|---|---|")
    for n in MODELS:
        T = temperatures[n]
        r, c = val_metrics_raw[n], val_metrics_cal[n]
        lines.append(f"| {n} | {T:.3f} | {r['ECE']:.4f} | {c['ECE']:.4f} | {r['NLL']:.4f} | {c['NLL']:.4f} |")
    lines += ["", "## Test-set accuracy / macro-F1 per decoder", ""]
    lines.append("| Model | Decoder | Accuracy | Macro-F1 | ms/frame |")
    lines.append("|---|---|---|---|---|")
    for n in MODELS:
        for tag, _, _ in decoder_specs:
            r = results[n][tag]
            lines.append(f"| {n} | {tag} | {r['acc']*100:.2f}% | {r['macroF1']:.3f} | {r['ms_per_frame']:.3f} |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    log(f"  -> {OUT_MD}")
    log("\nDone.")


if __name__ == "__main__":
    main()
