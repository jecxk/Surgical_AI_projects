"""
Evaluation Metrics for Surgical Phase Recognition.

Includes: accuracy, macro-F1, edit score, phase transition error,
per-class metrics, and confusion matrix.
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from scipy.spatial.distance import squareform


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute frame-level accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1 per class."""
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    results = {}
    for i, name in enumerate(class_names):
        if i < len(precision):
            results[name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
            }
    return results


def compute_edit_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the segment-level edit score (normalized Levenshtein distance).
    
    Measures how well the predicted phase sequence matches the ground truth
    at the segment level (penalizes over-segmentation).
    """
    # Convert frame-level to segment-level
    true_segments = _frames_to_segments(y_true)
    pred_segments = _frames_to_segments(y_pred)
    
    # Compute normalized edit distance
    distance = _levenshtein_distance(true_segments, pred_segments)
    max_len = max(len(true_segments), len(pred_segments))
    
    if max_len == 0:
        return 1.0
    
    edit_score = 1.0 - (distance / max_len)
    return max(0.0, edit_score)


def compute_phase_transition_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fps: int = 1,
) -> Dict[str, float]:
    """
    Compute phase transition error.
    
    Measures the mean absolute error (in seconds) at phase boundaries.
    
    Returns:
        Dictionary with mean, median, and std transition error in seconds.
    """
    # Find ground truth transitions
    true_transitions = []
    for i in range(1, len(y_true)):
        if y_true[i] != y_true[i - 1]:
            true_transitions.append((i, y_true[i - 1], y_true[i]))
    
    if not true_transitions:
        return {'mean_error': 0.0, 'median_error': 0.0, 'std_error': 0.0}
    
    errors = []
    for true_idx, from_phase, to_phase in true_transitions:
        # Find nearest matching transition in predictions
        best_error = len(y_pred)  # worst case
        for j in range(1, len(y_pred)):
            if y_pred[j] != y_pred[j - 1]:
                if y_pred[j - 1] == from_phase and y_pred[j] == to_phase:
                    error = abs(j - true_idx) / fps
                    best_error = min(best_error, error)
        errors.append(best_error / fps)
    
    errors = np.array(errors)
    return {
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'std_error': float(np.std(errors)),
        'num_transitions': len(true_transitions),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 7,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fps: int = 1,
) -> Dict:
    """Compute all evaluation metrics."""
    results = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'macro_f1': compute_macro_f1(y_true, y_pred),
        'per_class': compute_per_class_metrics(y_true, y_pred, class_names),
        'edit_score': compute_edit_score(y_true, y_pred),
        'transition_error': compute_phase_transition_error(y_true, y_pred, fps),
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred).tolist(),
    }
    return results


# ==================== Helper Functions ====================

def _frames_to_segments(y: np.ndarray) -> List[int]:
    """Convert frame-level predictions to segment-level (run-length encoding)."""
    if len(y) == 0:
        return []
    segments = [y[0]]
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            segments.append(y[i])
    return segments


def _levenshtein_distance(s1: List[int], s2: List[int]) -> int:
    """Compute Levenshtein distance between two sequences."""
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    matrix = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        matrix[i, 0] = i
    for j in range(len(s2) + 1):
        matrix[0, j] = j
    
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,      # deletion
                matrix[i, j - 1] + 1,      # insertion
                matrix[i - 1, j - 1] + cost  # substitution
            )
    
    return matrix[len(s1), len(s2)]
