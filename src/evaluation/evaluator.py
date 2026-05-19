"""
Model Evaluator for Surgical Phase Recognition.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import median_filter

from .metrics import compute_all_metrics


class Evaluator:
    """
    Evaluator for surgical phase recognition models.
    
    Performs inference, temporal smoothing, and metric computation.
    
    Args:
        model: Trained model
        device: Computation device
        class_names: List of phase class names
        temporal_smoothing: Smoothing config dict
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        temporal_smoothing: Optional[Dict] = None,
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.smoothing_config = temporal_smoothing or {'enabled': False}
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run full evaluation on a dataset.
        
        Returns:
            Dictionary containing all metrics and predictions.
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_tool_preds = []
        all_tool_targets = []
        video_results = {}
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['images'].to(self.device)
            phases = batch['phases']
            tools = batch['tools']
            video_ids = batch['video_id']
            
            outputs = self.model(images)
            
            phase_preds = outputs['phase_logits'].argmax(dim=-1).cpu()
            
            all_preds.append(phase_preds.flatten())
            all_targets.append(phases.flatten())
            
            if outputs.get('tool_logits') is not None:
                tool_preds = (torch.sigmoid(outputs['tool_logits']) > 0.5).float().cpu()
                all_tool_preds.append(tool_preds.flatten(0, 1))
                all_tool_targets.append(tools.flatten(0, 1))
            
            # Group by video
            for i, vid_id in enumerate(video_ids):
                vid_id = vid_id.item() if isinstance(vid_id, torch.Tensor) else vid_id
                if vid_id not in video_results:
                    video_results[vid_id] = {'preds': [], 'targets': []}
                video_results[vid_id]['preds'].extend(phase_preds[i].tolist())
                video_results[vid_id]['targets'].extend(phases[i].tolist())
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        # Apply temporal smoothing
        if self.smoothing_config.get('enabled', False):
            all_preds_smoothed = self._apply_temporal_smoothing(all_preds)
            # Also smooth per-video
            for vid_id in video_results:
                video_results[vid_id]['preds_smoothed'] = self._apply_temporal_smoothing(
                    np.array(video_results[vid_id]['preds'])
                ).tolist()
        else:
            all_preds_smoothed = all_preds
        
        # Compute metrics
        metrics_raw = compute_all_metrics(all_preds, all_targets, self.class_names)
        metrics_smoothed = compute_all_metrics(all_preds_smoothed, all_targets, self.class_names)
        
        results = {
            'raw_metrics': metrics_raw,
            'smoothed_metrics': metrics_smoothed,
            'per_video': {},
        }
        
        # Per-video metrics
        for vid_id, vr in video_results.items():
            preds = np.array(vr['preds'])
            targets = np.array(vr['targets'])
            results['per_video'][vid_id] = compute_all_metrics(preds, targets, self.class_names)
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _apply_temporal_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to predictions."""
        method = self.smoothing_config.get('method', 'median')
        window = self.smoothing_config.get('window_size', 15)
        
        if method == 'median':
            return median_filter(predictions, size=window).astype(int)
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            # Smooth class probabilities approach
            return median_filter(predictions, size=window).astype(int)
        else:
            return predictions
