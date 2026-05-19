"""
Temporal Visualization for Surgical Phase Recognition.

Generates timeline plots, training curves, and confusion matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


# Professional color palette for surgical phases
PHASE_COLORS = [
    '#2196F3',  # Preparation - Blue
    '#4CAF50',  # CalotTriangleDissection - Green
    '#FF9800',  # ClippingCutting - Orange
    '#F44336',  # GallbladderDissection - Red
    '#9C27B0',  # GallbladderPackaging - Purple
    '#00BCD4',  # CleaningCoagulation - Cyan
    '#795548',  # GallbladderRetraction - Brown
]

PHASE_NAMES = [
    "Preparation", "CalotTriangle\nDissection", "Clipping\nCutting",
    "Gallbladder\nDissection", "Gallbladder\nPackaging",
    "Cleaning\nCoagulation", "Gallbladder\nRetraction",
]


def plot_phase_timeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Surgical Phase Timeline",
    save_path: Optional[str] = None,
    fps: int = 1,
) -> plt.Figure:
    """
    Plot ground truth vs predicted phase timeline.
    
    Creates a ribbon plot showing phase annotations over time.
    """
    if class_names is None:
        class_names = PHASE_NAMES
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 4), sharex=True)
    
    time_axis = np.arange(len(y_true)) / fps / 60  # Convert to minutes
    
    # Ground truth
    for phase_id in range(7):
        mask = y_true == phase_id
        if mask.any():
            axes[0].fill_between(time_axis, 0, 1, where=mask,
                               color=PHASE_COLORS[phase_id], alpha=0.8)
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_yticks([])
    
    # Predictions
    for phase_id in range(7):
        mask = y_pred == phase_id
        if mask.any():
            axes[1].fill_between(time_axis, 0, 1, where=mask,
                               color=PHASE_COLORS[phase_id], alpha=0.8)
    axes[1].set_ylabel("Predicted")
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (minutes)")
    
    # Legend
    from matplotlib.patches import Patch
    short_names = ["Prep", "Calot", "Clip", "Dissect", "Package", "Clean", "Retract"]
    legend_elements = [Patch(facecolor=PHASE_COLORS[i], label=short_names[i]) for i in range(7)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=7,
              bbox_to_anchor=(0.5, 1.05), fontsize=9)
    
    plt.suptitle(title, y=1.1, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot confusion matrix as heatmap."""
    if class_names is None:
        class_names = ["Prep", "Calot", "Clip", "Dissect", "Package", "Clean", "Retract"]
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f' if normalize else 'd',
               cmap='Blues', xticklabels=class_names, yticklabels=class_names,
               ax=ax, vmin=0, vmax=1 if normalize else None)
    
    ax.set_xlabel('Predicted Phase', fontsize=12)
    ax.set_ylabel('True Phase', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def plot_training_curves(
    history: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation curves."""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['total_loss'] for h in history]
    val_loss = [h['val']['total_loss'] for h in history]
    train_acc = [h['train']['accuracy'] for h in history]
    val_acc = [h['val']['accuracy'] for h in history]
    val_f1 = [h['val'].get('macro_f1', 0) for h in history]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[2].plot(epochs, val_f1, 'g-', label='Val Macro-F1', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Validation F1 Score', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Mark stages
    stage_changes = []
    for i in range(1, len(history)):
        if history[i].get('stage') != history[i-1].get('stage'):
            stage_changes.append(epochs[i])
    
    for ax in axes:
        for sc in stage_changes:
            ax.axvline(x=sc, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig
