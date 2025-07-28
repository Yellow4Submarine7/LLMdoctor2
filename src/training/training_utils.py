"""
Training utilities for TFPO training in LLMdoctor framework.

Provides helper functions, data structures, and utilities for training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

from .tfpo_trainer import TrainingConfig, TrainingMetrics

logger = logging.getLogger(__name__)


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> TrainingConfig:
    """Create TrainingConfig from dictionary."""
    # Filter out unknown keys
    valid_keys = set(TrainingConfig.__dataclass_fields__.keys())
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    
    return TrainingConfig(**filtered_dict)


def save_training_config(config: TrainingConfig, filepath: str):
    """Save training configuration to file."""
    config_dict = asdict(config)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Training config saved to {filepath}")


def load_training_config(filepath: str) -> TrainingConfig:
    """Load training configuration from file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return create_training_config_from_dict(config_dict)


def compute_model_size(model: nn.Module) -> Dict[str, int]:
    """Compute model size statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0.0
    }


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    sequence_length: int,
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """Estimate memory usage for training."""
    
    # Model parameters
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Gradients (same size as parameters for trainable params)
    grad_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    # Optimizer states (AdamW needs 2x parameter memory for momentum and variance)
    optimizer_memory = grad_memory * 2
    
    # Activations (rough estimate)
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'hidden_size', 768)
        num_layers = getattr(model.config, 'num_hidden_layers', 12)
    else:
        hidden_size = 768  # Default
        num_layers = 12    # Default
    
    # Rough activation memory estimate
    activation_memory = (
        batch_size * sequence_length * hidden_size * num_layers * 
        (4 if dtype == torch.float32 else 2)
    )
    
    total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
    
    return {
        "parameters_mb": param_memory / (1024 ** 2),
        "gradients_mb": grad_memory / (1024 ** 2),
        "optimizer_mb": optimizer_memory / (1024 ** 2),
        "activations_mb": activation_memory / (1024 ** 2),
        "total_mb": total_memory / (1024 ** 2),
        "total_gb": total_memory / (1024 ** 3)
    }


def setup_distributed_training(
    local_rank: int,
    world_size: int,
    backend: str = "nccl"
) -> bool:
    """Setup distributed training."""
    try:
        import torch.distributed as dist
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            rank=local_rank,
            world_size=world_size
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training setup: rank {local_rank}/{world_size}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        return False


def cleanup_distributed_training():
    """Cleanup distributed training."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except:
        pass


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_names: List[str] = None
) -> List[Dict[str, Any]]:
    """Get parameter groups for optimizer with different weight decay."""
    
    if no_decay_names is None:
        no_decay_names = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(nd_name in name for nd_name in no_decay_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]


def freeze_parameters(model: nn.Module, freeze_patterns: List[str]):
    """Freeze parameters matching patterns."""
    frozen_count = 0
    
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in freeze_patterns):
            param.requires_grad = False
            frozen_count += 1
    
    logger.info(f"Frozen {frozen_count} parameters matching patterns: {freeze_patterns}")


def unfreeze_parameters(model: nn.Module, unfreeze_patterns: List[str]):
    """Unfreeze parameters matching patterns."""
    unfrozen_count = 0
    
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in unfreeze_patterns):
            param.requires_grad = True
            unfrozen_count += 1
    
    logger.info(f"Unfrozen {unfrozen_count} parameters matching patterns: {unfreeze_patterns}")


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
):
    """Create learning rate scheduler with warmup."""
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log_model_info(model: nn.Module, config: TrainingConfig):
    """Log detailed model information."""
    model_stats = compute_model_size(model)
    memory_stats = estimate_memory_usage(
        model=model,
        batch_size=config.batch_size,
        sequence_length=config.max_sequence_length
    )
    
    logger.info("=" * 50)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Total parameters: {model_stats['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_stats['trainable_parameters']:,}")
    logger.info(f"Frozen parameters: {model_stats['frozen_parameters']:,}")
    logger.info(f"Trainable ratio: {model_stats['trainable_ratio']:.2%}")
    
    logger.info("\nMEMORY ESTIMATES")
    logger.info("-" * 30)
    logger.info(f"Parameters: {memory_stats['parameters_mb']:.1f} MB")
    logger.info(f"Gradients: {memory_stats['gradients_mb']:.1f} MB")
    logger.info(f"Optimizer: {memory_stats['optimizer_mb']:.1f} MB")
    logger.info(f"Activations: {memory_stats['activations_mb']:.1f} MB")
    logger.info(f"Total: {memory_stats['total_gb']:.2f} GB")
    logger.info("=" * 50)


def validate_training_config(config: TrainingConfig) -> List[str]:
    """Validate training configuration and return warnings."""
    warnings = []
    
    # Check learning rate
    if config.learning_rate > 1e-3:
        warnings.append(f"Learning rate {config.learning_rate} is quite high, consider reducing")
    
    if config.learning_rate < 1e-6:
        warnings.append(f"Learning rate {config.learning_rate} is quite low, training may be slow")
    
    # Check batch size
    if config.batch_size < 2:
        warnings.append("Batch size < 2 may cause issues with batch normalization")
    
    # Check accumulation steps
    effective_batch_size = config.batch_size * config.accumulation_steps
    if effective_batch_size < 8:
        warnings.append(f"Effective batch size {effective_batch_size} is quite small")
    
    # Check sequence length
    if config.max_sequence_length > 4096:
        warnings.append(f"Sequence length {config.max_sequence_length} is quite long, may cause memory issues")
    
    # Check save frequency
    if config.save_steps > config.max_steps:
        warnings.append("save_steps > max_steps, model will not be saved during training")
    
    # Check evaluation frequency
    if config.eval_steps > config.max_steps:
        warnings.append("eval_steps > max_steps, no evaluation will occur during training")
    
    return warnings


def create_optimizer_groups(
    model: nn.Module,
    config: TrainingConfig
) -> List[Dict[str, Any]]:
    """Create optimizer parameter groups with proper weight decay."""
    
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    value_head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "value_heads" in name:
            # Value heads might need different learning rate
            value_head_params.append(param)
        elif any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = []
    
    if decay_params:
        param_groups.append({
            "params": decay_params,
            "weight_decay": config.weight_decay,
            "lr": config.learning_rate
        })
    
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "weight_decay": 0.0,
            "lr": config.learning_rate
        })
    
    if value_head_params:
        # Value heads might benefit from slightly higher learning rate
        param_groups.append({
            "params": value_head_params,
            "weight_decay": config.weight_decay * 0.5,  # Reduced weight decay
            "lr": config.learning_rate * 1.5  # Slightly higher LR
        })
    
    return param_groups


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for monitoring."""
    total_norm = 0.0
    max_grad = 0.0
    min_grad = float('inf')
    num_params = 0
    zero_grads = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            max_grad = max(max_grad, param.grad.data.abs().max().item())
            min_grad = min(min_grad, param.grad.data.abs().min().item())
            
            num_params += param.grad.data.numel()
            zero_grads += (param.grad.data == 0).sum().item()
    
    total_norm = total_norm ** 0.5
    
    return {
        "grad_norm": total_norm,
        "max_grad": max_grad,
        "min_grad": min_grad if min_grad != float('inf') else 0.0,
        "zero_grad_ratio": zero_grads / num_params if num_params > 0 else 0.0,
        "num_params_with_grad": num_params - zero_grads
    }


def save_training_summary(
    results: Dict[str, Any],
    config: TrainingConfig,
    output_dir: str
):
    """Save comprehensive training summary."""
    summary = {
        "config": asdict(config),
        "results": results,
        "training_completed": True
    }
    
    summary_path = Path(output_dir) / "training_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Training summary saved to {summary_path}")


def check_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check for NaN or Inf values in tensor."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        logger.warning(f"{name} contains {'NaN' if has_nan else ''}{'/' if has_nan and has_inf else ''}{'Inf' if has_inf else ''} values")
        return True
    
    return False


def create_exponential_moving_average(
    model: nn.Module,
    decay: float = 0.999
) -> nn.Module:
    """Create exponential moving average of model parameters."""
    ema_model = type(model)(model.config) if hasattr(model, 'config') else None
    
    if ema_model is None:
        logger.warning("Could not create EMA model, returning None")
        return None
    
    # Initialize EMA with current model parameters
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.copy_(param.data)
    
    # Disable gradients for EMA model
    for param in ema_model.parameters():
        param.requires_grad = False
    
    return ema_model


def update_exponential_moving_average(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = 0.999
):
    """Update exponential moving average."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class TrainingStateManager:
    """Manages training state and resumption."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / "training_state.json"
    
    def save_state(
        self,
        step: int,
        epoch: int,
        best_loss: float,
        metrics: List[TrainingMetrics]
    ):
        """Save training state."""
        state = {
            "step": step,
            "epoch": epoch,
            "best_loss": best_loss,
            "metrics": [m.to_dict() for m in metrics],
            "timestamp": str(torch.tensor(time.time()))
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load training state."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            return None
    
    def cleanup(self):
        """Clean up state file."""
        if self.state_file.exists():
            self.state_file.unlink()


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoint_files = list(checkpoint_dir.glob("checkpoint-*.pt"))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    return str(latest_checkpoint)


def profile_training_step(
    trainer,
    batch: Dict[str, torch.Tensor],
    num_steps: int = 10
) -> Dict[str, float]:
    """Profile training step performance."""
    import time
    
    trainer.model.train()
    
    times = {
        "forward": [],
        "backward": [],
        "optimizer": [],
        "total": []
    }
    
    for _ in range(num_steps):
        batch = {k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        start_time = time.time()
        
        # Forward pass
        forward_start = time.time()
        loss_dict = trainer._compute_loss(batch)
        total_loss = loss_dict["loss"]
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        total_loss.backward()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optimizer_start = time.time()
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start
        
        total_time = time.time() - start_time
        
        times["forward"].append(forward_time)
        times["backward"].append(backward_time)
        times["optimizer"].append(optimizer_time)
        times["total"].append(total_time)
    
    # Compute averages
    avg_times = {k: np.mean(v) for k, v in times.items()}
    
    return avg_times