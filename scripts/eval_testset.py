"""Dense evaluation on held-out Cholec80 test videos.

Unlike the web demo (which sparsely samples ~24-60 points), this evaluates every
1-fps frame via non-overlapping sequence windows — the standard protocol — and
reports accuracy / macro-F1 per model and for the val_f1-weighted ensemble.
"""
import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def log(msg):
    print(msg, flush=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import Cholec80SequenceDataset
from src.dataset.transforms import get_val_transforms
from src.models.surgical_model import SurgicalPhaseModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.ndimage import median_filter

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]

MODELS = {
    "M1 resnet50_lstm": "results/resnet50_lstm",
    "M2 efficientnet_b3_tcn": "results/efficientnet_b3_tcn",
    "M3 swin_tiny_transformer": "results/swin_tiny_transformer",
}


def load_model(run_dir: Path, device):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = cfg.get("model", {})
    mcfg["num_phases"] = 7
    mcfg["num_tools"] = 7
    ckpt = torch.load(run_dir / "checkpoints" / "best_model.pth", map_location="cpu", weights_only=False)
    model = SurgicalPhaseModel(mcfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    val_f1 = ckpt.get("best_val_f1") or ckpt.get("val_f1") or cfg.get("val_f1", 0.5)
    seq_len = int(cfg.get("data", {}).get("sequence_length", 8))
    return model, seq_len, float(val_f1)


@torch.no_grad()
def predict_logits(model, loader, device, n_classes=7):
    """Return (all_logits [N,7], all_labels [N]) flattened over sequence positions."""
    logits_list, labels_list = [], []
    n_batches = len(loader)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        frames = batch["images"].to(device)          # [B, T, C, H, W]
        labels = batch["phases"]                      # [B, T]
        out = model(frames)
        lg = out["phase_logits"].cpu()               # [B, T, 7]
        B, T, _ = lg.shape
        logits_list.append(lg.reshape(B * T, -1))
        labels_list.append(labels.reshape(B * T))
        if bi % 20 == 0 or bi == n_batches - 1:
            log(f"    batch {bi+1}/{n_batches}  ({time.time()-t0:.0f}s)")
    return torch.cat(logits_list).numpy(), torch.cat(labels_list).numpy()


def report(name, y_true, y_pred, smooth_window=0):
    mask = y_true >= 0
    yt, yp = y_true[mask], y_pred[mask]
    if smooth_window and len(yp) >= smooth_window:
        yp = median_filter(yp, size=smooth_window)
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", labels=list(range(7)), zero_division=0)
    f1w = f1_score(yt, yp, average="weighted", labels=list(range(7)), zero_division=0)
    tag = f" (smooth w={smooth_window})" if smooth_window else ""
    log(f"  {name}{tag}:  acc={acc*100:5.2f}%   macroF1={f1m:.3f}   weightedF1={f1w:.3f}")
    return acc, f1m, yp, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=str, default="61-70")
    ap.add_argument("--smooth", type=int, default=15, help="median window in frames (1fps); 0=off")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers for image I/O")
    args = ap.parse_args()

    lo, hi = (args.videos.split("-") + [args.videos])[:2]
    video_ids = list(range(int(lo), int(hi) + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | test videos: {video_ids}\n")

    transform = get_val_transforms(224)
    per_model_logits = {}
    labels_ref = None
    weights = {}
    summary = {"videos": video_ids, "smooth": args.smooth, "device": str(device), "models": {}}

    for name, rd in MODELS.items():
        model, seq_len, vf1 = load_model(Path(rd), device)
        ds = Cholec80SequenceDataset(
            data_root="data/cholec80", video_ids=video_ids,
            transform=transform, sequence_length=seq_len, stride=seq_len, fps=1,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=args.workers)
        log(f"[{name}] seq_len={seq_len} val_f1={vf1:.3f} windows={len(ds)}")
        logits, labels = predict_logits(model, loader, device)
        per_model_logits[name] = logits
        weights[name] = max(vf1, 0.01)
        if labels_ref is None:
            labels_ref = labels
        acc_raw, f1_raw, _, _ = report(name, labels, logits.argmax(1))
        acc_s, f1_s, _, _ = report(name, labels, logits.argmax(1), smooth_window=args.smooth)
        summary["models"][name] = {"val_f1": vf1, "acc_raw": acc_raw, "f1_raw": f1_raw,
                                    "acc_smoothed": acc_s, "f1_smoothed": f1_s}
        log("")
        del model
        torch.cuda.empty_cache()

    # Ensemble — softmax then val_f1-weighted average (matches web demo logic)
    log("[Ensemble val_f1-weighted]")
    probs = []
    wsum = sum(weights.values())
    for name in MODELS:
        p = torch.softmax(torch.tensor(per_model_logits[name]), dim=1).numpy()
        probs.append(p * (weights[name] / wsum))
    ens = np.sum(probs, axis=0)
    acc_raw, f1_raw, _, _ = report("Ensemble", labels_ref, ens.argmax(1))
    acc, f1m, yp, yt = report("Ensemble", labels_ref, ens.argmax(1), smooth_window=args.smooth)
    summary["models"]["Ensemble"] = {"acc_raw": acc_raw, "f1_raw": f1_raw,
                                      "acc_smoothed": acc, "f1_smoothed": f1m}

    cm = confusion_matrix(yt, yp, labels=list(range(7)))
    log("\nConfusion matrix (Ensemble, smoothed) — row=GT, col=pred:")
    log("       " + " ".join(f"{i:>4d}" for i in range(7)))
    for i in range(7):
        log(f"  GT{i}: " + " ".join(f"{cm[i][j]:>4d}" for j in range(7)) + f"   {PHASE_NAMES[i]}")
    log("\nPer-class recall (Ensemble, smoothed):")
    recalls = {}
    for i in range(7):
        tot = int(cm[i].sum())
        rec = cm[i][i] / tot if tot else 0
        recalls[PHASE_NAMES[i]] = {"recall": float(rec), "n": tot}
        log(f"  {PHASE_NAMES[i]:25s} {rec*100:5.1f}%  (n={tot})")
    summary["ensemble_confusion"] = cm.tolist()
    summary["ensemble_recall"] = recalls

    out = Path("results/test_eval_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    log(f"\nSaved summary -> {out}")


if __name__ == "__main__":
    main()
