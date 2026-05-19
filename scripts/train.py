"""
Main Training Script for Surgical Phase Recognition.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/resnet_lstm.yaml --synthetic
"""

import os
import sys
import argparse
import logging
import random
import yaml
import json
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import (
    Cholec80SequenceDataset, SyntheticCholec80Dataset
)
from src.dataset.transforms import get_train_transforms, get_val_transforms
from src.dataset.utils import create_data_loaders, compute_class_weights
from src.models.surgical_model import SurgicalPhaseModel
from src.training.trainer import Trainer
from src.training.losses import SurgicalLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load and merge configuration files."""
    # Load default config
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(default_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with specific config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            override = yaml.safe_load(f)
        if override:
            deep_update(config, override)
    
    return config


def deep_update(base: dict, update: dict):
    """Recursively update nested dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def create_datasets(config: dict, use_synthetic: bool = False):
    """Create train/val/test datasets."""
    data_cfg = config.get('data', {})
    img_size = data_cfg.get('img_size', 224)
    seq_len = data_cfg.get('sequence_length', 16)
    stride = data_cfg.get('stride', 8)
    fps = data_cfg.get('fps', 1)
    
    train_transform = get_train_transforms(img_size, data_cfg.get('augmentation'))
    val_transform = get_val_transforms(img_size)
    
    if use_synthetic:
        logger.info("Using SYNTHETIC dataset for development")
        train_dataset = SyntheticCholec80Dataset(
            num_videos=20, frames_per_video=300,
            sequence_length=seq_len, stride=stride,
            transform=train_transform, img_size=img_size,
        )
        val_dataset = SyntheticCholec80Dataset(
            num_videos=8, frames_per_video=200,
            sequence_length=seq_len, stride=stride,
            transform=val_transform, img_size=img_size,
        )
        test_dataset = SyntheticCholec80Dataset(
            num_videos=5, frames_per_video=200,
            sequence_length=seq_len, stride=stride,
            transform=val_transform, img_size=img_size,
        )
    else:
        data_root = data_cfg.get('data_root', 'data/cholec80')
        logger.info(f"Loading Cholec80 from {data_root}")
        
        train_videos = data_cfg.get('train_videos', list(range(1, 41)))
        val_videos = data_cfg.get('val_videos', list(range(41, 61)))
        test_videos = data_cfg.get('test_videos', list(range(61, 81)))
        
        train_dataset = Cholec80SequenceDataset(
            data_root=data_root, video_ids=train_videos,
            transform=train_transform, sequence_length=seq_len,
            stride=stride, fps=fps,
        )
        val_dataset = Cholec80SequenceDataset(
            data_root=data_root, video_ids=val_videos,
            transform=val_transform, sequence_length=seq_len,
            stride=seq_len, fps=fps,  # No overlap for val
        )
        test_dataset = Cholec80SequenceDataset(
            data_root=data_root, video_ids=test_videos,
            transform=val_transform, sequence_length=seq_len,
            stride=seq_len, fps=fps,
        )
    
    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description="Train Surgical Phase Recognition Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=None, help='Override num epochs')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set seed
    set_seed(config.get('project', {}).get('seed', 42))
    
    # Setup output directory
    output_dir = args.output or config.get('project', {}).get('output_dir', 'results')
    model_name = config.get('model', {}).get('temporal_model', 'lstm')
    backbone_name = config.get('model', {}).get('backbone', 'resnet50')
    run_name = f"{backbone_name}_{model_name}"
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Add file handler for logging
    fh = logging.FileHandler(output_dir / 'training.log')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    logger.info(f"Configuration: {json.dumps(config, indent=2, default=str)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(config, args.synthetic)
    logger.info(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_cfg = config.get('training', {})
    loaders = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=train_cfg.get('batch_size', 8),
        num_workers=config.get('project', {}).get('num_workers', 0 if sys.platform == 'win32' else 4),
    )
    
    # Build model
    model_config = config.get('model', {})
    model_config['num_phases'] = config.get('data', {}).get('num_phases', 7)
    model_config['num_tools'] = config.get('data', {}).get('num_tools', 7)
    
    model = SurgicalPhaseModel(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {run_name}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    training_config = {
        **train_cfg,
        'device': config.get('project', {}).get('device', 'cuda'),
        'tool_loss_weight': model_config.get('tool_loss_weight', 0.5),
    }
    
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=training_config,
        output_dir=str(output_dir),
    )
    
    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    trainer.train()
    
    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("Running test set evaluation...")
    logger.info("=" * 60)
    
    from src.evaluation.evaluator import Evaluator
    
    phase_labels = config.get('data', {}).get('phase_labels', {})
    class_names = [phase_labels.get(i, f"Phase_{i}") for i in range(7)]
    
    smoothing_cfg = config.get('evaluation', {}).get('temporal_smoothing', {})
    
    evaluator = Evaluator(
        model=model,
        device=trainer.device,
        class_names=class_names,
        temporal_smoothing=smoothing_cfg,
    )
    
    test_results = evaluator.evaluate(loaders['test'], save_dir=str(output_dir / 'test_results'))
    
    # Print results
    raw = test_results['raw_metrics']
    smoothed = test_results['smoothed_metrics']
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Accuracy:     {raw['accuracy']:.4f}  (smoothed: {smoothed['accuracy']:.4f})")
    print(f"  Macro-F1:     {raw['macro_f1']:.4f}  (smoothed: {smoothed['macro_f1']:.4f})")
    print(f"  Edit Score:   {raw['edit_score']:.4f}  (smoothed: {smoothed['edit_score']:.4f})")
    print(f"  Transition Error: {raw['transition_error']['mean_error']:.2f}s")
    print()
    print("Per-class F1:")
    for name, metrics in raw['per_class'].items():
        print(f"  {name:30s} P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
    print("=" * 60)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    from src.visualization.temporal_plot import plot_training_curves, plot_confusion_matrix
    
    (output_dir / 'figures').mkdir(exist_ok=True)
    
    plot_training_curves(
        trainer.training_history,
        save_path=str(output_dir / 'figures' / 'training_curves.png'),
    )
    
    cm = np.array(raw['confusion_matrix'])
    plot_confusion_matrix(
        cm, class_names=[n.split('\n')[0] for n in class_names],
        save_path=str(output_dir / 'figures' / 'confusion_matrix.png'),
    )
    
    logger.info(f"All results saved to {output_dir}")
    print(f"\n✅ Training complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
