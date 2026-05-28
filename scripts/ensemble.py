"""
Ensemble Evaluation for Surgical Phase Recognition.

Loads N trained checkpoints, runs each on the test set,
then combines predictions with multiple ensemble strategies:
  1. Simple average of softmax
  2. Weighted average (weights from val F1)
  3. Majority voting

Usage:
    python scripts/ensemble.py \
        --m1 results/resnet50_lstm/checkpoints/best.pt \
        --m2 results/efficientnet_b3_tcn/checkpoints/best.pt \
        --m3 results/swin_tiny_transformer/checkpoints/best.pt \
        --output results/ensemble/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import Cholec80SequenceDataset
from src.dataset.transforms import get_val_transforms
from src.models.surgical_model import SurgicalPhaseModel
from src.evaluation.metrics import compute_all_metrics
from scipy.ndimage import median_filter


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load a model from checkpoint. Returns (model, val_f1, config).

    Trainer saves only the flattened 'training' section under ckpt['config'], so we read
    the full merged config from the sibling config.yaml in the results dir.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg_path = Path(checkpoint_path).parent.parent / 'config.yaml'
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt.get('config', {})

    model_cfg = cfg.get('model', {})
    if not model_cfg:
        raise ValueError(f"Could not find model config in {cfg_path} or checkpoint")

    model_cfg['num_phases'] = 7
    model_cfg['num_tools'] = 7

    model = SurgicalPhaseModel(model_cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_f1 = ckpt.get('best_metric', ckpt.get('best_val_f1', ckpt.get('val_f1', 0.5)))
    return model, val_f1, cfg


@torch.no_grad()
def collect_softmax(model, loader, device):
    """Run model on test loader, return concatenated softmax probs + targets."""
    all_probs = []
    all_targets = []
    for batch in tqdm(loader, desc="Inference"):
        images = batch['images'].to(device)
        out = model(images)
        probs = F.softmax(out['phase_logits'], dim=-1).cpu().numpy()
        all_probs.append(probs.reshape(-1, probs.shape[-1]))
        all_targets.append(batch['phases'].cpu().numpy().reshape(-1))
    return np.concatenate(all_probs), np.concatenate(all_targets)


def apply_smoothing(preds: np.ndarray, window: int = 15) -> np.ndarray:
    return median_filter(preds, size=window)


def report(name: str, preds: np.ndarray, targets: np.ndarray, class_names):
    print(f"\n--- {name} ---")
    metrics = compute_all_metrics(targets, preds, class_names)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"  Edit     : {metrics['edit_score']:.4f}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m1', required=True, help='Checkpoint of model 1')
    ap.add_argument('--m2', required=True, help='Checkpoint of model 2')
    ap.add_argument('--m3', required=True, help='Checkpoint of model 3')
    ap.add_argument('--data_root', default='data/cholec80')
    ap.add_argument('--test_videos', default='61-80', help='Range like 61-80')
    ap.add_argument('--output', default='results/ensemble')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--smoothing_window', type=int, default=15)
    args = ap.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse test videos range
    lo, hi = args.test_videos.split('-')
    test_videos = list(range(int(lo), int(hi) + 1))
    print(f"Testing on videos {test_videos[0]}..{test_videos[-1]} ({len(test_videos)} videos)")

    # Use config from m1 for dataset settings
    print("\nLoading models...")
    models_info = []
    for tag, ck in [('M1', args.m1), ('M2', args.m2), ('M3', args.m3)]:
        m, f1, cfg = load_model(ck, device)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"  {tag}: {Path(ck).parent.parent.name:30s}  val_f1={f1:.4f}  params={n_params:.1f}M")
        models_info.append((tag, m, f1, cfg))

    # Build test loader using m1's seq config (or pick max to be safe)
    seq_lengths = [info[3]['data']['sequence_length'] for info in models_info]
    if len(set(seq_lengths)) > 1:
        print(f"[!] WARNING: models trained with different seq_length: {seq_lengths}")
        print(f"    Using {seq_lengths[0]} from M1 — best to retrain with matching seq for ensemble")

    seq_len = seq_lengths[0]
    print(f"\nUsing sequence_length={seq_len}, stride={seq_len} (no overlap for eval)")

    ds = Cholec80SequenceDataset(
        data_root=args.data_root,
        video_ids=test_videos,
        transform=get_val_transforms(224),
        sequence_length=seq_len,
        stride=seq_len,
        fps=1,
    )
    print(f"Test sequences: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Collect predictions from each model
    print("\n" + "=" * 60)
    print("Running inference (each model on full test set)")
    print("=" * 60)
    all_probs = []
    all_targets = None
    for tag, m, f1, cfg in models_info:
        print(f"\n[{tag}] inference...")
        probs, targets = collect_softmax(m, loader, device)
        all_probs.append(probs)
        all_targets = targets
        # free GPU mem
        m.to('cpu')
        torch.cuda.empty_cache()

    class_names = [
        "Preparation", "CalotTriDissect", "Clipping", "GallbDissect",
        "GallbPackage", "Cleaning", "GallbRetract",
    ]

    # Individual model results
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODELS")
    print("=" * 60)
    individual_metrics = {}
    for i, (tag, _, f1, _) in enumerate(models_info):
        preds = all_probs[i].argmax(axis=-1)
        m = report(f"{tag} raw", preds, all_targets, class_names)
        preds_sm = apply_smoothing(preds, args.smoothing_window)
        m_sm = report(f"{tag} + temporal smoothing", preds_sm, all_targets, class_names)
        individual_metrics[tag] = {'raw': m, 'smoothed': m_sm}

    # Ensemble 1: simple average
    print("\n" + "=" * 60)
    print("ENSEMBLE STRATEGIES")
    print("=" * 60)
    avg_probs = np.mean(all_probs, axis=0)
    avg_preds = avg_probs.argmax(axis=-1)
    ens_avg = report("Ensemble — simple average", avg_preds, all_targets, class_names)
    ens_avg_sm = report("Ensemble — simple avg + smoothing",
                       apply_smoothing(avg_preds, args.smoothing_window), all_targets, class_names)

    # Ensemble 2: weighted by val F1
    val_f1s = np.array([info[2] for info in models_info])
    weights = val_f1s / val_f1s.sum()
    print(f"\nWeights (from val F1): M1={weights[0]:.3f}, M2={weights[1]:.3f}, M3={weights[2]:.3f}")
    weighted_probs = sum(w * p for w, p in zip(weights, all_probs))
    weighted_preds = weighted_probs.argmax(axis=-1)
    ens_w = report("Ensemble — val-F1 weighted", weighted_preds, all_targets, class_names)
    ens_w_sm = report("Ensemble — weighted + smoothing",
                     apply_smoothing(weighted_preds, args.smoothing_window), all_targets, class_names)

    # Ensemble 3: majority voting (hard)
    hard_preds = np.stack([p.argmax(axis=-1) for p in all_probs], axis=0)
    from scipy.stats import mode
    vote_preds = mode(hard_preds, axis=0).mode.squeeze()
    ens_v = report("Ensemble — majority voting", vote_preds, all_targets, class_names)

    # Save results
    final = {
        'individual': {tag: m for tag, m in individual_metrics.items()},
        'ensemble': {
            'simple_avg': ens_avg,
            'simple_avg_smoothed': ens_avg_sm,
            'weighted_avg': ens_w,
            'weighted_avg_smoothed': ens_w_sm,
            'majority_voting': ens_v,
        },
        'weights': weights.tolist(),
        'val_f1s': val_f1s.tolist(),
    }

    # Helper to keep result JSON-serializable
    def to_native(o):
        if isinstance(o, dict):
            return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to_native(x) for x in o]
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    (output / 'ensemble_results.json').write_text(json.dumps(to_native(final), indent=2))
    print(f"\n[OK] Saved results to {output / 'ensemble_results.json'}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY (Macro-F1)")
    print("=" * 60)
    print(f"  M1 (raw):              {individual_metrics['M1']['raw']['macro_f1']:.4f}")
    print(f"  M2 (raw):              {individual_metrics['M2']['raw']['macro_f1']:.4f}")
    print(f"  M3 (raw):              {individual_metrics['M3']['raw']['macro_f1']:.4f}")
    print(f"  Ensemble simple avg:   {ens_avg['macro_f1']:.4f}")
    print(f"  Ensemble weighted:     {ens_w['macro_f1']:.4f}")
    print(f"  Ensemble + smoothing:  {ens_w_sm['macro_f1']:.4f}  <-- typically best")
    print(f"  Majority voting:       {ens_v['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
