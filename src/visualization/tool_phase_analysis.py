"""
Tool-Phase Correlation Analysis.

Analyzes and visualizes the relationship between surgical tools
and surgical phases from the model's predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Dict, List, Optional


PHASE_NAMES_SHORT = ["Prep", "Calot", "Clip", "Dissect", "Package", "Clean", "Retract"]
TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"]


def compute_tool_phase_matrix(
    phases: np.ndarray,
    tools: np.ndarray,
    num_phases: int = 7,
    num_tools: int = 7,
) -> np.ndarray:
    """
    Compute co-occurrence matrix between tools and phases.
    
    Args:
        phases: Phase labels (N,)
        tools: Tool binary labels (N, num_tools)
        
    Returns:
        Co-occurrence matrix (num_phases, num_tools) normalized by phase count
    """
    matrix = np.zeros((num_phases, num_tools))
    phase_counts = np.zeros(num_phases)
    
    for i in range(len(phases)):
        phase = phases[i]
        if 0 <= phase < num_phases:
            phase_counts[phase] += 1
            matrix[phase] += tools[i]
    
    # Normalize by phase count
    for p in range(num_phases):
        if phase_counts[p] > 0:
            matrix[p] /= phase_counts[p]
    
    return matrix


def plot_tool_phase_correlation(
    phases: np.ndarray,
    tools: np.ndarray,
    phase_names: Optional[List[str]] = None,
    tool_names: Optional[List[str]] = None,
    title: str = "Tool-Phase Co-occurrence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot heatmap of tool-phase co-occurrence."""
    if phase_names is None:
        phase_names = PHASE_NAMES_SHORT
    if tool_names is None:
        tool_names = TOOL_NAMES
    
    matrix = compute_tool_phase_matrix(phases, tools)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        matrix, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=tool_names, yticklabels=phase_names,
        ax=ax, vmin=0, vmax=1,
    )
    
    ax.set_xlabel('Surgical Tool', fontsize=12)
    ax.set_ylabel('Surgical Phase', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig
