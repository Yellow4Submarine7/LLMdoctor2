"""
Main training script for LLMDoctor framework.

This script orchestrates the complete training pipeline:
1. Load configuration
2. Process preference dataset to extract token-level rewards
3. Train TFPO model
4. Save results
"""

import argparse
import logging
import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
import random
from typing import Dict, List, Optional, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.patient_model import PatientModel
from src.models.doctor_model import DoctorModel
from src.reward.reward_processor import RewardDataProcessor, RewardProcessingConfig, PreferenceExample
from src.reward.behavioral_variants import BehavioralVariantConfig
from src.reward.token_importance import TokenImportanceConfig
from src.reward.reward_assignment import RewardAssignmentConfig
from src.training.tfpo_trainer import TFPOTrainer, TrainingConfig
from src.training.training_utils import (
    log_model_info, 
    validate_training_config,
    save_training_summary,
    find_latest_checkpoint
)
from src.data.data_loader import PreferenceDataLoader, DatasetConfig
from configs import ExperimentConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration."""
    if config_path.endswith('.yaml'):
        config = ExperimentConfig.load(config_path)
    else:
        # Load from Python module
        if config_path == "hh_rlhf":
            from configs import create_hh_rlhf_config
            config = create_hh_rlhf_config()
        elif config_path == "pku_saferlhf":
            from configs import create_pku_saferlhf_config
            config = create_pku_saferlhf_config()
        elif config_path == "weak_to_strong":
            from configs import create_weak_to_strong_config
            config = create_weak_to_strong_config()
        else:
            raise ValueError(f"Unknown configuration: {config_path}")
    
    logger.info(f"Loaded configuration: {config.experiment_name}")
    return config


def load_models(config: ExperimentConfig) -> tuple[PatientModel, DoctorModel]:
    """Load patient and doctor models."""
    logger.info("Loading models...")
    
    # Load patient model
    patient_model = PatientModel(
        model_name=config.model.patient_model_name,
        device=config.model.patient_device,
        dtype=config.model.patient_dtype,
        load_in_8bit=config.model.patient_load_in_8bit,
        load_in_4bit=config.model.patient_load_in_4bit,
        trust_remote_code=config.model.trust_remote_code,
        use_cache=config.model.use_cache,
        cache_dir=config.model.cache_dir
    )
    
    # Load doctor model
    doctor_model = DoctorModel(
        model_name=config.model.doctor_model_name,
        num_preference_dims=config.model.doctor_num_preference_dims,
        value_head_hidden_size=config.model.doctor_value_head_hidden_size,
        device=config.model.doctor_device,
        dtype=config.model.doctor_dtype,
        freeze_base_model=config.model.doctor_freeze_base_model,
        trust_remote_code=config.model.trust_remote_code,
        use_cache=config.model.use_cache,
        cache_dir=config.model.cache_dir
    )
    
    logger.info(f"Patient model: {patient_model.model_info()['model_name']}")
    logger.info(f"Doctor model: {doctor_model.model_info()['model_name']}")
    
    return patient_model, doctor_model


def process_reward_data(
    patient_model: PatientModel,
    config: ExperimentConfig,
    preference_examples: List[PreferenceExample],
    resume_checkpoint: Optional[str] = None
) -> List:
    """Process preference dataset to extract token-level rewards."""
    logger.info("Processing reward data...")
    
    # Create reward processor configuration
    reward_config = RewardProcessingConfig(
        batch_size=config.data.batch_size,
        max_sequence_length=config.data.max_sequence_length,
        save_intermediate_results=True,
        intermediate_save_interval=100,
        num_workers=config.data.num_workers,
        enable_caching=True,
        cache_dir=config.model.cache_dir,
        log_progress=True,
        dimension=config.data.preference_dimension
    )
    
    # Set sub-component configs
    if config.reward.positive_instruction and config.reward.negative_instruction:
        reward_config.variant_config = BehavioralVariantConfig(
            positive_instruction=config.reward.positive_instruction,
            negative_instruction=config.reward.negative_instruction
        )
    
    reward_config.importance_config = TokenImportanceConfig(
        epsilon=config.reward.epsilon,
        tau=config.reward.tau,
        normalization_method=config.reward.normalization_method,
        smoothing_function=config.reward.smoothing_function,
        batch_size=config.reward.importance_batch_size
    )
    
    reward_config.reward_config = RewardAssignmentConfig(
        sparsity_threshold=config.reward.sparsity_threshold,
        reward_scale=config.reward.reward_scale
    )
    
    # Create reward processor
    processor = RewardDataProcessor(
        patient_model=patient_model,
        config=reward_config,
        device=config.training.device
    )
    
    # Process dataset
    output_path = Path(config.reward.reward_data_path) / f"{config.experiment_name}_rewards.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    reward_data = processor.process_preference_dataset(
        preference_examples=preference_examples,
        output_path=str(output_path),
        resume_from_checkpoint=resume_checkpoint
    )
    
    # Compute and log statistics
    stats = processor.compute_dataset_statistics(reward_data)
    logger.info("Reward data statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Filter by quality if needed
    filtered_data = processor.filter_by_quality(
        reward_data,
        min_tokens=5,
        min_rewarded_tokens=1,
        min_reward_magnitude=0.01
    )
    
    logger.info(f"Filtered data: {len(filtered_data)} examples")
    
    return filtered_data


def train_tfpo(
    doctor_model: DoctorModel,
    train_data: List,
    eval_data: Optional[List],
    config: ExperimentConfig,
    resume_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Train TFPO model."""
    logger.info("Starting TFPO training...")
    
    # Create training configuration
    training_config = TrainingConfig(
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        gradient_clip_norm=config.training.gradient_clip_norm,
        accumulation_steps=config.training.accumulation_steps,
        lambda_value=config.training.lambda_value,
        subtb_config={
            "clamp_min": config.training.subtb_clamp_min,
            "clamp_max": config.training.subtb_clamp_max,
            "eps": config.training.subtb_eps
        },
        value_config={
            "margin": config.training.value_margin,
            "loss_type": config.training.value_loss_type
        },
        prefix_score_method=config.training.prefix_score_method,
        batch_size=config.training.batch_size,
        max_sequence_length=config.training.max_sequence_length,
        save_steps=config.training.save_steps,
        log_steps=config.training.log_steps,
        eval_steps=config.training.eval_steps,
        save_total_limit=config.training.save_total_limit,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_threshold=config.training.early_stopping_threshold,
        device=config.training.device,
        mixed_precision=config.training.mixed_precision,
        gradient_checkpointing=config.training.gradient_checkpointing
    )
    
    # Validate configuration
    warnings = validate_training_config(training_config)
    if warnings:
        logger.warning("Training configuration warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    # Create trainer
    trainer = TFPOTrainer(
        model=doctor_model,
        config=training_config,
        output_dir=config.training.output_dir,
        resume_from_checkpoint=resume_checkpoint,
        wandb_project=config.training.wandb_project
    )
    
    # Log model information
    log_model_info(doctor_model, training_config)
    
    # Train model
    results = trainer.train(
        train_dataset=train_data,
        eval_dataset=eval_data
    )
    
    logger.info("Training completed!")
    logger.info(f"Final loss: {results['final_metrics'].get('total_loss', 'N/A')}")
    logger.info(f"Total time: {results['total_time']:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train LLMDoctor model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file or preset name (hh_rlhf, pku_saferlhf, weak_to_strong)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--skip-reward-processing",
        action="store_true",
        help="Skip reward processing and load from cache"
    )
    parser.add_argument(
        "--reward-data-path",
        type=str,
        default=None,
        help="Path to pre-processed reward data"
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced dataset"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_experiment_config(args.config)
    
    # Override configuration with command line arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.max_samples:
        config.data.max_samples = args.max_samples
    if args.batch_size:
        config.training.batch_size = args.batch_size
        config.data.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_steps:
        config.training.max_steps = args.max_steps
    
    # Debug mode
    if args.debug:
        logger.info("Running in debug mode with reduced dataset")
        config.data.max_samples = 100
        config.training.max_steps = 50
        config.training.eval_steps = 10
        config.training.save_steps = 20
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(output_dir / "training.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Save configuration
    config.save(output_dir / "experiment_config.yaml")
    
    try:
        # Load models
        patient_model, doctor_model = load_models(config)
        
        # Process or load reward data
        if args.skip_reward_processing and args.reward_data_path:
            logger.info(f"Loading pre-processed reward data from {args.reward_data_path}")
            from src.reward.reward_processor import RewardDataProcessor
            processor = RewardDataProcessor(patient_model, device=config.training.device)
            reward_data = processor.load_processed_data(args.reward_data_path)
        else:
            # Load preference dataset
            dataset_config = DatasetConfig(
                dataset_name=config.data.dataset_name,
                dataset_split=config.data.dataset_split,
                max_samples=config.data.max_samples,
                max_sequence_length=config.data.max_sequence_length,
                cache_dir=config.model.cache_dir,
                seed=args.seed
            )
            
            data_loader = PreferenceDataLoader()
            preference_dataset = data_loader.load_dataset(dataset_config)
            
            # Convert to preference examples
            preference_examples = []
            for example in preference_dataset:
                pref_example = PreferenceExample(
                    prompt=example.get("prompt", example.get("question", "")),
                    preferred_response=example.get("chosen", example.get("response_chosen", "")),
                    non_preferred_response=example.get("rejected", example.get("response_rejected", "")),
                    metadata=example.get("metadata", {})
                )
                preference_examples.append(pref_example)
            
            logger.info(f"Loaded {len(preference_examples)} preference examples")
            
            # Process reward data
            reward_resume_checkpoint = None
            if args.resume:
                checkpoint_dir = Path(args.resume)
                if checkpoint_dir.is_dir():
                    reward_checkpoint = checkpoint_dir / "reward_data.checkpoint"
                    if reward_checkpoint.exists():
                        reward_resume_checkpoint = str(reward_checkpoint)
            
            reward_data = process_reward_data(
                patient_model=patient_model,
                config=config,
                preference_examples=preference_examples,
                resume_checkpoint=reward_resume_checkpoint
            )
        
        # Split into train/eval
        num_examples = len(reward_data)
        num_eval = int(num_examples * args.eval_split)
        
        # Shuffle data
        indices = list(range(num_examples))
        np.random.shuffle(indices)
        
        eval_indices = indices[:num_eval]
        train_indices = indices[num_eval:]
        
        train_data = [reward_data[i] for i in train_indices]
        eval_data = [reward_data[i] for i in eval_indices] if num_eval > 0 else None
        
        logger.info(f"Training data: {len(train_data)} examples")
        if eval_data:
            logger.info(f"Evaluation data: {len(eval_data)} examples")
        
        # Find checkpoint to resume from
        resume_checkpoint = None
        if args.resume:
            if Path(args.resume).is_file():
                resume_checkpoint = args.resume
            else:
                resume_checkpoint = find_latest_checkpoint(args.resume)
                if resume_checkpoint:
                    logger.info(f"Found checkpoint to resume from: {resume_checkpoint}")
        
        # Train model
        results = train_tfpo(
            doctor_model=doctor_model,
            train_data=train_data,
            eval_data=eval_data,
            config=config,
            resume_checkpoint=resume_checkpoint
        )
        
        # Save training summary
        save_training_summary(
            results=results,
            config=config.training,
            output_dir=config.training.output_dir
        )
        
        logger.info(f"Training complete! Results saved to {config.training.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()