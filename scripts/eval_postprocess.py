"""Evaluate test videos WITH inference-time post-processing (no retraining).

Runs each model over test videos, collects per-video logit sequences, then
applies monotonic-phase Viterbi decoding + class-prior logit adjustment and
reports accuracy / macro-F1 before vs after. Compares the val_f1-weighted
ensemble too. Results saved to results/test_eval_postprocess.json.
"""
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import Cholec80SequenceDataset
from src.dataset.transforms import get_val_transforms
from src.models.surgical_model import SurgicalPhaseModel
from src.evaluation.postprocess import postprocess_video, softmax, TRAIN_PRIORS
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def log(m): print(m, flush=True)

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]
MODELS = {
    "M1": "results/resnet50_lstm",
    "M2": "results/efficientnet_b3_tcn",
    "M3": "results/swin_tiny_transformer",
}


def load_model(run_dir, device):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = cfg.get("model", {}); mcfg["num_phases"] = 7; mcfg["num_tools"] = 7
    ckpt = torch.load(run_dir / "checkpoints" / "best_model.pth", map_location="cpu", weights_only=False)
    model = SurgicalPhaseModel(mcfg); model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    seq_len = int(cfg.get("data", {}).get("sequence_length", 8))
    vf1 = ckpt.get("best_val_f1") or cfg.get("val_f1", 0.7)
    return model, seq_len, float(vf1)


@torch.no_grad()
def collect_logits_per_video(model, video_ids, seq_len, transform, device):
    """Return {vid: (logits [T,7], labels [T])} using non-overlapping windows."""
    out = {}
    for vid in video_ids:
        ds = Cholec80SequenceDataset(
            data_root="data/cholec80", video_ids=[vid],
            transform=transform, sequence_length=seq_len, stride=seq_len, fps=1,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
        lg_all, lab_all = [], []
        for batch in loader:
            frames = batch["images"].to(device)
            out_m = model(frames)
            lg = out_m["phase_logits"].cpu().numpy()       # [B,T,7]
            lab = batch["phases"].numpy()                  # [B,T]
            B, T, _ = lg.shape
            lg_all.append(lg.reshape(B * T, 7)); lab_all.append(lab.reshape(B * T))
        out[vid] = (np.concatenate(lg_all), np.concatenate(lab_all))
    return out


def metrics(y_true, y_pred):
    m = y_true >= 0
    yt, yp = y_true[m], y_pred[m]
    return (accuracy_score(yt, yp),
            f1_score(yt, yp, average="macro", labels=list(range(7)), zero_division=0),
            f1_score(yt, yp, average="weighted", labels=list(range(7)), zero_division=0))


def eval_config(per_model, weights, video_ids, tau, monotonic, allow_skip):
    """Apply postprocess per video and aggregate. Returns (acc,macroF1,wF1, y_true,y_pred)."""
    yt_all, yp_all = [], []
    wsum = sum(weights.values())
    for vid in video_ids:
        # ensemble probs per video
        labels = None
        ens = None
        for name in MODELS:
            logits, lab = per_model[name][vid]
            labels = lab
            # apply logit adjust then softmax, weight, sum
            adj = logits - tau * np.log(np.clip(TRAIN_PRIORS, 1e-6, None))[None, :]
            p = softmax(adj) * (weights[name] / wsum)
            ens = p if ens is None else ens + p
        if monotonic:
            from src.evaluation.postprocess import monotonic_decode
            pred = monotonic_decode(ens, allow_skip=allow_skip)
        else:
            pred = ens.argmax(1)
        yt_all.append(labels); yp_all.append(pred)
    yt = np.concatenate(yt_all); yp = np.concatenate(yp_all)
    acc, f1m, f1w = metrics(yt, yp)
    return acc, f1m, f1w, yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="61-70")
    args = ap.parse_args()
    lo, hi = (args.videos.split("-") + [args.videos])[:2]
    video_ids = list(range(int(lo), int(hi) + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device {device} | videos {video_ids}")

    transform = get_val_transforms(224)
    per_model, weights = {}, {}
    for name, rd in MODELS.items():
        t0 = time.time()
        model, seq_len, vf1 = load_model(Path(rd), device)
        log(f"[{name}] seq_len={seq_len} collecting logits...")
        per_model[name] = collect_logits_per_video(model, video_ids, seq_len, transform, device)
        weights[name] = max(vf1, 0.01)
        log(f"[{name}] done ({time.time()-t0:.0f}s)")
        del model; torch.cuda.empty_cache()

    results = {}
    configs = [
        ("ensemble_raw",            dict(tau=0.0, monotonic=False, allow_skip=True)),
        ("ensemble_logitadj",       dict(tau=1.0, monotonic=False, allow_skip=True)),
        ("ensemble_monotonic",      dict(tau=0.0, monotonic=True,  allow_skip=True)),
        ("ensemble_mono+logitadj",  dict(tau=1.0, monotonic=True,  allow_skip=True)),
    ]
    log("\n=== Ensemble post-processing comparison ===")
    best = None
    for tag, kw in configs:
        acc, f1m, f1w, yt, yp = eval_config(per_model, weights, video_ids, **kw)
        results[tag] = {"acc": acc, "macroF1": f1m, "weightedF1": f1w, **kw}
        log(f"  {tag:24s}  acc={acc*100:5.2f}%  macroF1={f1m:.3f}  wF1={f1w:.3f}")
        if best is None or acc > best[1]:
            best = (tag, acc, yt, yp)

    tag, acc, yt, yp = best
    cm = confusion_matrix(yt, yp, labels=list(range(7)))
    log(f"\nBest config: {tag} (acc={acc*100:.2f}%)")
    log("Per-class recall:")
    for i in range(7):
        tot = int(cm[i].sum()); rec = cm[i][i]/tot if tot else 0
        log(f"  {PHASE_NAMES[i]:25s} {rec*100:5.1f}%  (n={tot})")
    results["best"] = {"config": tag, "confusion": cm.tolist()}

    Path("results/test_eval_postprocess.json").write_text(json.dumps(results, indent=2))
    log("\nSaved -> results/test_eval_postprocess.json")


if __name__ == "__main__":
    main()
