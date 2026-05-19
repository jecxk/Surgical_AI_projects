"""
Dataset utility functions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from collections import Counter


def compute_class_weights(dataset, num_classes: int = 7) -> torch.Tensor:
    """Compute inverse frequency class weights from dataset."""
    phase_counts = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        if 'phases' in sample:
            phases = sample['phases']
            if isinstance(phases, torch.Tensor):
                for p in phases.tolist():
                    phase_counts[p] += 1
            else:
                phase_counts[phases] += 1
        elif 'phase' in sample:
            phase_counts[sample['phase'].item()] += 1
    
    total = sum(phase_counts.values())
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        count = phase_counts.get(i, 1)
        weights[i] = total / (num_classes * count)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    return weights


def create_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset=None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create data loaders for train/val/test splits."""
    loaders = {
        'train': DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        ),
        'val': DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        ),
    }
    if test_dataset is not None:
        loaders['test'] = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    return loaders
