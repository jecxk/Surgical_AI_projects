"""
Evaluation Script for Surgical Phase Recognition.

Usage:
    python scripts/evaluate.py --checkpoint results/resnet50_lstm/checkpoints/best_model.pth
"""

import sys
import argparse
import json
import yaml
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import SyntheticCholec80Dataset
from src.dataset.transforms import get_val_transforms
from src.dataset.utils import create_data_loaders
from src.models.surgical_model import SurgicalPhaseModel
from src.evaluation.evaluator import Evaluator
from src.visualization.temporal_plot import plot_confusion_matrix, plot_phase_timeline
from src.visualization.tool_phase_analysis import plot_tool_phase_correlation


def main():
    parser = argparse.ArgumentParser(description="Evaluate Surgical Phase Model")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build model
    model_config = config if 'backbone' in config else config.get('model', config)
    model_config['num_phases'] = 7
    model_config['num_tools'] = 7
    model = SurgicalPhaseModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create test dataset
    val_transform = get_val_transforms(224)
    
    if args.synthetic:
        test_dataset = SyntheticCholec80Dataset(
            num_videos=10, frames_per_video=200,
            sequence_length=16, stride=16,
            transform=val_transform,
        )
    else:
        from src.dataset.cholec80_dataset import Cholec80SequenceDataset
        test_dataset = Cholec80SequenceDataset(
            data_root='data/cholec80',
            video_ids=list(range(61, 81)),
            transform=val_transform,
            sequence_length=16, stride=16,
        )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=0,
    )
    
    # Evaluate
    output_dir = args.output or str(Path(args.checkpoint).parent.parent / 'evaluation')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    class_names = [
        "Preparation", "CalotTriangleDissection", "ClippingCutting",
        "GallbladderDissection", "GallbladderPackaging",
        "CleaningCoagulation", "GallbladderRetraction",
    ]
    
    evaluator = Evaluator(
        model=model, device=device, class_names=class_names,
        temporal_smoothing={'enabled': True, 'method': 'median', 'window_size': 15},
    )
    
    results = evaluator.evaluate(test_loader, save_dir=output_dir)
    
    # Print results
    raw = results['raw_metrics']
    smoothed = results['smoothed_metrics']
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:     {raw['accuracy']:.4f}  →  {smoothed['accuracy']:.4f} (smoothed)")
    print(f"  Macro-F1:     {raw['macro_f1']:.4f}  →  {smoothed['macro_f1']:.4f} (smoothed)")
    print(f"  Edit Score:   {raw['edit_score']:.4f}  →  {smoothed['edit_score']:.4f} (smoothed)")
    print("=" * 60)
    
    # Generate plots
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    cm = np.array(raw['confusion_matrix'])
    plot_confusion_matrix(cm, save_path=str(figures_dir / 'confusion_matrix.png'))
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
