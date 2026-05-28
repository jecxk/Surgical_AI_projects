"""
Training Pipeline for Surgical Phase Recognition.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .losses import SurgicalLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for surgical phase recognition models.
    
    Supports:
    - Staged training (freeze backbone -> train temporal -> full fine-tune)
    - Mixed precision training
    - TensorBoard logging
    - Early stopping
    - Checkpoint management
    
    Args:
        model: SurgicalPhaseModel
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        output_dir: Directory to save checkpoints and logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        output_dir: str = "results",
        evaluator=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.evaluator = evaluator
        
        # Setup directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        # Loss
        loss_cfg = config.get('loss', {})
        self.criterion = SurgicalLoss(
            label_smoothing=loss_cfg.get('label_smoothing', 0.1),
            tool_loss_weight=config.get('tool_loss_weight', 0.5),
        )
        
        # Optimizer with differential learning rates
        opt_cfg = config.get('optimizer', {})
        backbone_lr = opt_cfg.get('backbone_lr', 1e-5)
        main_lr = opt_cfg.get('lr', 1e-4)
        weight_decay = opt_cfg.get('weight_decay', 1e-4)
        
        param_groups = [
            {'params': model.get_non_backbone_params(), 'lr': main_lr},
        ]
        backbone_params = list(model.get_backbone_params())
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': backbone_lr})
        
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Scheduler
        sched_cfg = config.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'cosine_warm_restarts')
        if sched_type == 'cosine_warm_restarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_cfg.get('T_0', 10),
                T_mult=sched_cfg.get('T_mult', 2),
                eta_min=sched_cfg.get('eta_min', 1e-6),
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = []
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            images = batch['images'].to(self.device)
            phases = batch['phases'].to(self.device)
            tools = batch['tools'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images)
                    losses = self.criterion(outputs, {'phases': phases, 'tools': tools})
                
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                losses = self.criterion(outputs, {'phases': phases, 'tools': tools})
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Track losses
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            
            # Track accuracy
            phase_preds = outputs['phase_logits'].argmax(dim=-1)
            total_correct += (phase_preds == phases).sum().item()
            total_samples += phases.numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'acc': f"{total_correct / total_samples:.4f}",
            })
        
        # Average losses
        num_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        epoch_losses['accuracy'] = total_correct / total_samples
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        epoch_losses = {}
        all_preds = []
        all_targets = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['images'].to(self.device)
            phases = batch['phases'].to(self.device)
            tools = batch['tools'].to(self.device)
            
            outputs = self.model(images)
            losses = self.criterion(outputs, {'phases': phases, 'tools': tools})
            
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()
            
            phase_preds = outputs['phase_logits'].argmax(dim=-1)
            all_preds.append(phase_preds.cpu())
            all_targets.append(phases.cpu())
        
        # Average losses
        num_batches = len(self.val_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        # Compute metrics
        all_preds = torch.cat(all_preds).flatten()
        all_targets = torch.cat(all_targets).flatten()
        
        accuracy = (all_preds == all_targets).float().mean().item()
        epoch_losses['accuracy'] = accuracy
        
        # Compute macro F1
        from sklearn.metrics import f1_score
        epoch_losses['macro_f1'] = f1_score(
            all_targets.numpy(), all_preds.numpy(), average='macro', zero_division=0
        )
        
        return epoch_losses
    
    def train(self, num_epochs: Optional[int] = None):
        """Full training loop with staged training support."""
        if num_epochs is None:
            num_epochs = self.config.get('epochs', 50)
        
        staged = self.config.get('staged_training', False)
        
        if staged:
            s1 = self.config.get('stage1_epochs', 10)
            s2 = self.config.get('stage2_epochs', 20)
            s3 = self.config.get('stage3_epochs', 20)
            
            # Stage 1: Freeze backbone, train temporal + heads
            logger.info("=== Stage 1: Training temporal model (backbone frozen) ===")
            self.model.freeze_backbone()
            self._train_stage(s1, "stage1")
            
            # Stage 2: Continue training temporal + heads
            logger.info("=== Stage 2: Continued temporal training ===")
            self._train_stage(s2, "stage2")
            
            # Stage 3: Full fine-tuning
            logger.info("=== Stage 3: End-to-end fine-tuning ===")
            self.model.unfreeze_backbone()
            for pg in self.optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.1
            self._train_stage(s3, "stage3")
        else:
            self._train_stage(num_epochs, "full")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        self._save_history()
        self.writer.close()
        
        logger.info(f"Training complete! Best epoch: {self.best_epoch}, Best F1: {self.best_metric:.4f}")
    
    def _train_stage(self, num_epochs: int, stage_name: str):
        """Train for a specific stage."""
        patience = self.config.get('early_stopping', {}).get('patience', 10)
        # Reset patience counter at start of each stage so unfreezing gets a fair chance.
        self.patience_counter = 0
        logger.info(f"[{stage_name}] starting: num_epochs={num_epochs}, early_stopping_patience={patience}")

        for epoch in range(num_epochs):
            self.current_epoch += 1
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # Log
            elapsed = time.time() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            log_msg = (
                f"[{stage_name}] Epoch {self.current_epoch} ({elapsed:.1f}s) | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} | "
                f"LR: {lr:.2e}"
            )
            logger.info(log_msg)
            print(log_msg)
            
            # TensorBoard logging
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, self.current_epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}', v, self.current_epoch)
            self.writer.add_scalar('lr', lr, self.current_epoch)
            
            # Save history (incremental flush so external tools can monitor live)
            self.training_history.append({
                'epoch': self.current_epoch,
                'stage': stage_name,
                'train': train_metrics,
                'val': val_metrics,
                'lr': lr,
            })
            self._save_history()

            # Check for best model
            current_metric = val_metrics.get('macro_f1', val_metrics['accuracy'])
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.best_epoch = self.current_epoch
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print(f"  * New best model! F1={self.best_metric:.4f}")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            save_every = self.config.get('save_every', 5)
            if self.current_epoch % save_every == 0:
                self.save_checkpoint(f"epoch_{self.current_epoch}.pth")
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
    
    def _save_history(self):
        """Save training history to JSON."""
        path = self.output_dir / "training_history.json"
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
