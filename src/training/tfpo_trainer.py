"""
TFPO Trainer for LLMdoctor framework.

Main training coordinator for Token-level Flow-guided Preference Optimization.
Handles the complete training loop with checkpointing, logging, and evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time
from tqdm import tqdm
import numpy as np

from .flow_losses import TFPOLoss, compute_prefix_scores
from ..models.doctor_model import DoctorModel
from ..reward.reward_processor import TokenRewardData

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for TFPO training."""
    # Training hyperparameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Loss configuration
    lambda_value: float = 1.0
    subtb_config: Dict[str, Any] = field(default_factory=dict)
    value_config: Dict[str, Any] = field(default_factory=dict)
    prefix_score_method: str = "cumulative"
    
    # Data configuration
    batch_size: int = 4
    max_sequence_length: int = 1024
    shuffle_data: bool = True
    num_workers: int = 0
    
    # Checkpointing and logging
    save_steps: int = 100
    log_steps: int = 10
    eval_steps: int = 50
    save_total_limit: int = 3
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Device and precision
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False


@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    step: int = 0
    epoch: int = 0
    total_loss: float = 0.0
    subtb_loss: float = 0.0
    value_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    examples_per_second: float = 0.0
    value_reward_correlation: float = 0.0
    mean_flow: float = 0.0
    std_flow: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "subtb_loss": self.subtb_loss,
            "value_loss": self.value_loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "examples_per_second": self.examples_per_second,
            "value_reward_correlation": self.value_reward_correlation,
            "mean_flow": self.mean_flow,
            "std_flow": self.std_flow
        }


class TFPOTrainer:
    """
    Main trainer for TFPO (Token-level Flow-guided Preference Optimization).
    
    Coordinates the training process including:
    - Data loading and batching
    - Loss computation with SubTB and value discrimination
    - Optimization and gradient clipping
    - Checkpointing and logging
    - Evaluation and early stopping
    """
    
    def __init__(
        self,
        model: DoctorModel,
        config: TrainingConfig,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None,
        wandb_project: Optional[str] = None
    ):
        """
        Initialize TFPO trainer.
        
        Args:
            model: DoctorModel instance to train
            config: Training configuration
            output_dir: Directory to save checkpoints and logs
            resume_from_checkpoint: Path to checkpoint to resume from
            wandb_project: Wandb project name for logging
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.loss_fn = TFPOLoss(
            lambda_value=config.lambda_value,
            subtb_config=config.subtb_config,
            value_config=config.value_config
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Metrics tracking
        self.training_metrics = []
        self.eval_metrics = []
        
        # Mixed precision scaler
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup logging
        self._setup_logging(wandb_project)
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"TFPOTrainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train(
        self,
        train_dataset: List[TokenRewardData],
        eval_dataset: Optional[List[TokenRewardData]] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_dataset: Training data
            eval_dataset: Optional evaluation data
            num_epochs: Number of epochs (computed from config if None)
            
        Returns:
            Training results dictionary
        """
        if num_epochs is None:
            # Estimate epochs from max_steps
            steps_per_epoch = len(train_dataset) // self.config.batch_size
            num_epochs = max(1, self.config.max_steps // steps_per_epoch)
        
        logger.info(f"Starting training for {num_epochs} epochs, {self.config.max_steps} max steps")
        
        # Create data loader
        train_loader = self._create_dataloader(train_dataset, shuffle=self.config.shuffle_data)
        
        # Create scheduler
        total_steps = min(self.config.max_steps, num_epochs * len(train_loader))
        self.scheduler = self._create_scheduler(total_steps)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            if self.global_step >= self.config.max_steps:
                break
            
            self.current_epoch = epoch
            epoch_metrics = self._train_epoch(train_loader)
            
            # Evaluation
            if eval_dataset is not None and epoch % (self.config.eval_steps // len(train_loader) + 1) == 0:
                eval_metrics = self.evaluate(eval_dataset)
                self.eval_metrics.append(eval_metrics)
                
                # Early stopping check
                if self._should_early_stop(eval_metrics.total_loss):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Save checkpoint
            if epoch % (self.config.save_steps // len(train_loader) + 1) == 0:
                self.save_checkpoint(f"checkpoint-epoch-{epoch}")
        
        total_time = time.time() - start_time
        
        # Save final model
        self.save_model("final_model")
        
        # Training summary
        results = {
            "total_time": total_time,
            "total_steps": self.global_step,
            "total_epochs": self.current_epoch + 1,
            "best_loss": self.best_loss,
            "final_metrics": epoch_metrics.to_dict() if 'epoch_metrics' in locals() else {},
            "training_metrics": [m.to_dict() for m in self.training_metrics],
            "eval_metrics": [m.to_dict() for m in self.eval_metrics]
        }
        
        logger.info(f"Training completed in {total_time:.2f}s")
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> TrainingMetrics:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = []
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            if self.global_step >= self.config.max_steps:
                break
            
            # Forward pass and loss computation
            metrics = self._training_step(batch)
            epoch_metrics.append(metrics)
            
            # Logging
            if self.global_step % self.config.log_steps == 0:
                self._log_metrics(metrics)
                progress_bar.set_postfix({
                    "loss": f"{metrics.total_loss:.4f}",
                    "lr": f"{metrics.learning_rate:.2e}",
                    "corr": f"{metrics.value_reward_correlation:.3f}"
                })
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint-step-{self.global_step}")
            
            self.global_step += 1
        
        # Compute epoch average
        avg_metrics = self._average_metrics(epoch_metrics)
        self.training_metrics.append(avg_metrics)
        
        return avg_metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Perform a single training step."""
        step_start_time = time.time()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        if self.config.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = self._compute_loss(batch)
        else:
            loss_dict = self._compute_loss(batch)
        
        total_loss = loss_dict["loss"] / self.config.accumulation_steps
        
        # Backward pass
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        # Optimization step
        if (self.global_step + 1) % self.config.accumulation_steps == 0:
            if self.config.mixed_precision and self.scaler is not None:
                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard gradient clipping and step
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        # Compute metrics
        step_time = time.time() - step_start_time
        examples_per_second = batch["input_ids"].shape[0] / step_time
        
        metrics = TrainingMetrics(
            step=self.global_step,
            epoch=self.current_epoch,
            total_loss=float(loss_dict["loss"]),
            subtb_loss=float(loss_dict.get("subtb_loss", 0.0)),
            value_loss=float(loss_dict.get("value_loss", 0.0)),
            learning_rate=float(self.optimizer.param_groups[0]['lr']),
            grad_norm=float(grad_norm),
            examples_per_second=examples_per_second,
            value_reward_correlation=float(loss_dict.get("value_diagnostics", {}).get("value_reward_correlation", 0.0)),
            mean_flow=float(loss_dict.get("subtb_diagnostics", {}).get("mean_flow", 0.0)),
            std_flow=float(loss_dict.get("subtb_diagnostics", {}).get("std_flow", 0.0))
        )
        
        return metrics
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute TFPO loss for a batch."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        token_rewards = batch["token_rewards"]
        
        # Forward pass through model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract outputs
        logits = outputs["logits"]
        values = outputs["values"]
        
        # Compute log probabilities for the actual sequence
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get log probabilities for actual tokens (shifted for next token prediction)
        policy_logprobs = []
        for i in range(input_ids.shape[1] - 1):
            token_logprobs = log_probs[:, i, input_ids[:, i + 1]]
            policy_logprobs.append(token_logprobs)
        
        if policy_logprobs:
            policy_logprobs = torch.stack(policy_logprobs, dim=1)  # [batch_size, seq_len-1]
        else:
            policy_logprobs = torch.zeros((input_ids.shape[0], 0), device=self.device)
        
        # Compute prefix scores from token rewards
        prefix_scores = compute_prefix_scores(
            token_rewards, method=self.config.prefix_score_method
        )
        
        # Adjust dimensions to match
        min_len = min(policy_logprobs.shape[1], values.shape[1], prefix_scores.shape[1], token_rewards.shape[1])
        
        if min_len > 0:
            policy_logprobs = policy_logprobs[:, :min_len]
            values_adjusted = values[:, :min_len] if values.dim() == 2 else values[:, :min_len, :]
            prefix_scores = prefix_scores[:, :min_len]
            token_rewards = token_rewards[:, :min_len]
            
            if attention_mask is not None:
                attention_mask = attention_mask[:, :min_len]
        else:
            # Handle edge case with very short sequences
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
        
        # Compute TFPO loss
        loss_dict = self.loss_fn(
            policy_logprobs=policy_logprobs,
            values=values_adjusted,
            prefix_scores=prefix_scores,
            token_rewards=token_rewards,
            attention_mask=attention_mask,
            return_components=True
        )
        
        return loss_dict
    
    def evaluate(self, eval_dataset: List[TokenRewardData]) -> TrainingMetrics:
        """Evaluate model on evaluation dataset."""
        self.model.eval()
        eval_loader = self._create_dataloader(eval_dataset, shuffle=False)
        
        eval_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                loss_dict = self._compute_loss(batch)
                
                metrics = TrainingMetrics(
                    step=self.global_step,
                    epoch=self.current_epoch,
                    total_loss=float(loss_dict["loss"]),
                    subtb_loss=float(loss_dict.get("subtb_loss", 0.0)),
                    value_loss=float(loss_dict.get("value_loss", 0.0)),
                    value_reward_correlation=float(loss_dict.get("value_diagnostics", {}).get("value_reward_correlation", 0.0)),
                    mean_flow=float(loss_dict.get("subtb_diagnostics", {}).get("mean_flow", 0.0)),
                    std_flow=float(loss_dict.get("subtb_diagnostics", {}).get("std_flow", 0.0))
                )
                
                eval_metrics.append(metrics)
        
        avg_metrics = self._average_metrics(eval_metrics)
        logger.info(f"Evaluation - Loss: {avg_metrics.total_loss:.4f}, Correlation: {avg_metrics.value_reward_correlation:.3f}")
        
        return avg_metrics
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler."""
        if self.config.scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.config.warmup_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        elif self.config.scheduler_type.lower() == "linear":
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
        else:
            return None
    
    def _create_dataloader(
        self, 
        dataset: List[TokenRewardData], 
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader from token reward data."""
        from ..data.tokenized_dataset import TokenizedDataset
        from ..data.data_collators import TFPODataCollator
        
        # Convert to tokenized dataset
        tokenized_dataset = TokenizedDataset(
            reward_data=dataset,
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_sequence_length
        )
        
        # Create collator
        collator = TFPODataCollator(
            tokenizer=self.model.tokenizer,
            max_length=self.config.max_sequence_length
        )
        
        return DataLoader(
            tokenized_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            collate_fn=collator,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )
    
    def _average_metrics(self, metrics_list: List[TrainingMetrics]) -> TrainingMetrics:
        """Compute average of metrics."""
        if not metrics_list:
            return TrainingMetrics()
        
        avg_dict = {}
        for key in metrics_list[0].to_dict().keys():
            if key in ["step", "epoch"]:
                avg_dict[key] = metrics_list[-1].to_dict()[key]  # Use last value
            else:
                values = [m.to_dict()[key] for m in metrics_list if not np.isnan(m.to_dict()[key])]
                avg_dict[key] = np.mean(values) if values else 0.0
        
        return TrainingMetrics(**avg_dict)
    
    def _should_early_stop(self, eval_loss: float) -> bool:
        """Check if training should stop early."""
        if eval_loss < self.best_loss - self.config.early_stopping_threshold:
            self.best_loss = eval_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config.early_stopping_patience
    
    def _setup_logging(self, wandb_project: Optional[str]):
        """Setup logging with optional wandb integration."""
        try:
            if wandb_project:
                import wandb
                wandb.init(
                    project=wandb_project,
                    config=self.config.__dict__,
                    name=f"tfpo-{self.output_dir.name}"
                )
                self.use_wandb = True
            else:
                self.use_wandb = False
        except ImportError:
            logger.warning("wandb not available, skipping wandb logging")
            self.use_wandb = False
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to console and wandb."""
        metrics_dict = metrics.to_dict()
        
        # Console logging
        logger.info(
            f"Step {metrics.step}: Loss={metrics.total_loss:.4f}, "
            f"SubTB={metrics.subtb_loss:.4f}, Value={metrics.value_loss:.4f}, "
            f"LR={metrics.learning_rate:.2e}, Corr={metrics.value_reward_correlation:.3f}"
        )
        
        # Wandb logging
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics_dict, step=metrics.step)
            except:
                pass
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"{checkpoint_name}.pt"
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "early_stopping_counter": self.early_stopping_counter,
            "config": self.config.__dict__,
            "training_metrics": [m.to_dict() for m in self.training_metrics],
            "eval_metrics": [m.to_dict() for m in self.eval_metrics]
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if checkpoint.get("scaler_state_dict") and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.early_stopping_counter = checkpoint["early_stopping_counter"]
        
        # Load metrics
        self.training_metrics = [TrainingMetrics(**m) for m in checkpoint.get("training_metrics", [])]
        self.eval_metrics = [TrainingMetrics(**m) for m in checkpoint.get("eval_metrics", [])]
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
    
    def save_model(self, model_name: str):
        """Save the trained model."""
        model_path = self.output_dir / model_name
        self.model.save_pretrained(str(model_path))
        
        # Save training config
        config_path = model_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved: {model_path}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        checkpoint_files = list(self.output_dir.glob("checkpoint-*.pt"))
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent checkpoints
        for checkpoint_file in checkpoint_files[self.config.save_total_limit:]:
            checkpoint_file.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint_file}")


class TFPOTrainerCallback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer: TFPOTrainer):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: TFPOTrainer):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: TFPOTrainer, epoch: int):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: TFPOTrainer, epoch: int, metrics: TrainingMetrics):
        """Called at the end of each epoch."""
        pass
    
    def on_step_end(self, trainer: TFPOTrainer, step: int, metrics: TrainingMetrics):
        """Called after each training step."""
        pass


class EarlyStoppingCallback(TFPOTrainerCallback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, trainer: TFPOTrainer, epoch: int, metrics: TrainingMetrics):
        if metrics.total_loss < self.best_loss - self.min_delta:
            self.best_loss = metrics.total_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.should_stop = True
                logger.info(f"Early stopping at epoch {epoch}")


class LearningRateSchedulerCallback(TFPOTrainerCallback):
    """Learning rate scheduling callback."""
    
    def __init__(self, scheduler_fn):
        self.scheduler_fn = scheduler_fn
    
    def on_step_end(self, trainer: TFPOTrainer, step: int, metrics: TrainingMetrics):
        new_lr = self.scheduler_fn(step, metrics)
        if new_lr is not None:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr