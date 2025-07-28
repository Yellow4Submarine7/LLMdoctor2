"""
Base configuration for LLMDoctor experiments.

This module defines configuration dataclasses for all components
of the LLMDoctor framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model setup."""
    # Patient model config
    patient_model_name: str = "meta-llama/Llama-2-7b-hf"
    patient_device: str = "auto"
    patient_dtype: str = "float16"
    patient_load_in_8bit: bool = False
    patient_load_in_4bit: bool = False
    
    # Doctor model config
    doctor_model_name: str = "meta-llama/Llama-2-7b-hf"
    doctor_num_preference_dims: int = 1
    doctor_value_head_hidden_size: int = 512
    doctor_device: str = "auto"
    doctor_dtype: str = "float16"
    doctor_load_in_8bit: bool = False
    doctor_load_in_4bit: bool = False
    doctor_freeze_base_model: bool = False
    
    # Common settings
    trust_remote_code: bool = False
    use_cache: bool = True
    cache_dir: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data handling."""
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train"
    dataset_subset: Optional[str] = None
    max_samples: Optional[int] = None
    max_sequence_length: int = 1024
    preference_dimension: str = "helpfulness"
    
    # Data processing
    batch_size: int = 4
    num_workers: int = 0
    shuffle: bool = True
    seed: int = 42
    
    # Validation split
    validation_split: float = 0.1
    validation_max_samples: Optional[int] = 1000


@dataclass
class RewardConfig:
    """Configuration for token-level reward acquisition."""
    # Behavioral variants
    positive_instruction: Optional[str] = None
    negative_instruction: Optional[str] = None
    
    # Token importance calculation
    epsilon: float = 1e-8
    tau: float = 1.0
    normalization_method: str = "mean"  # "mean", "max", "std"
    smoothing_function: str = "tanh"    # "tanh", "sigmoid", "linear"
    
    # Reward assignment
    sparsity_threshold: float = 0.1
    reward_scale: float = 1.0
    
    # Processing
    importance_batch_size: int = 8
    save_reward_data: bool = True
    reward_data_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for TFPO training."""
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # Loss configuration
    lambda_value: float = 1.0
    subtb_clamp_min: float = -10.0
    subtb_clamp_max: float = 10.0
    subtb_eps: float = 1e-8
    value_margin: float = 0.1
    value_loss_type: str = "hinge"  # "hinge", "mse", "ranking"
    
    # Subtrajectory balance
    max_subtrajectory_length: Optional[int] = None
    prefix_score_method: str = "cumulative"  # "cumulative", "mean", "max"
    
    # Training settings
    batch_size: int = 4
    eval_batch_size: int = 8
    max_sequence_length: int = 1024
    shuffle_data: bool = True
    num_workers: int = 0
    
    # Checkpointing and logging
    output_dir: str = "./outputs"
    save_steps: int = 500
    log_steps: int = 10
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # Device and precision
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Experiment tracking
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    
    # Optimization settings
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class InferenceConfig:
    """Configuration for online alignment inference."""
    # Reward-guided decoding
    alpha: float = 1.0  # Weight for base model
    beta: float = 0.5   # Weight for reward model
    
    # Multi-dimensional preference control
    preference_weights: Dict[str, float] = field(default_factory=dict)
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: Optional[int] = None
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Efficiency settings
    batch_size: int = 1
    use_cache: bool = True
    
    # Evaluation
    eval_prompts_file: Optional[str] = None
    output_file: Optional[str] = None
    save_generations: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Evaluation data
    eval_dataset: str = "Anthropic/hh-rlhf"
    eval_split: str = "test"
    eval_max_samples: Optional[int] = 300
    
    # Evaluation method
    eval_method: str = "gpt4"  # "gpt4", "reward_model", "perplexity"
    gpt4_api_key: Optional[str] = None
    gpt4_model: str = "gpt-4"
    
    # Metrics
    compute_diversity: bool = True
    diversity_metric: str = "distinct-4"
    compute_perplexity: bool = False
    
    # Head-to-head comparison
    baseline_methods: List[str] = field(default_factory=lambda: ["dpo", "genarm"])
    num_comparisons: int = 300
    
    # Output
    results_dir: str = "./results"
    save_detailed_results: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment metadata
    experiment_name: str = "llmdoctor_experiment"
    experiment_type: str = "standard"  # "standard", "multi_dim", "weak_to_strong"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Environment
    seed: int = 42
    deterministic: bool = True
    log_level: str = "INFO"
    
    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Recursively create dataclass instances
        def dict_to_dataclass(data_dict, dataclass_type):
            if isinstance(data_dict, dict):
                field_types = dataclass_type.__annotations__
                kwargs = {}
                for field_name, field_type in field_types.items():
                    if field_name in data_dict:
                        if hasattr(field_type, '__annotations__'):
                            # Nested dataclass
                            kwargs[field_name] = dict_to_dataclass(data_dict[field_name], field_type)
                        else:
                            kwargs[field_name] = data_dict[field_name]
                return dataclass_type(**kwargs)
            return data_dict
        
        return dict_to_dataclass(data, cls)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested keys like "model.patient_model_name"
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(self, key, value)


# Utility functions for creating common configurations

def create_hh_rlhf_config(
    patient_model: str = "meta-llama/Llama-2-7b-hf",
    doctor_model: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "./outputs/hh_rlhf",
    **kwargs
) -> ExperimentConfig:
    """Create configuration for HH-RLHF experiments."""
    config = ExperimentConfig()
    
    # Model settings
    config.model.patient_model_name = patient_model
    config.model.doctor_model_name = doctor_model
    
    # Data settings
    config.data.dataset_name = "Anthropic/hh-rlhf"
    config.data.preference_dimension = "helpfulness"
    
    # Training settings
    config.training.output_dir = output_dir
    config.training.max_steps = 10000
    
    # Experiment metadata
    config.experiment_name = "hh_rlhf_tfpo"
    config.experiment_type = "standard"
    
    # Apply any additional kwargs
    config.update_from_dict(kwargs)
    
    return config


def create_pku_saferlhf_config(
    patient_model: str = "meta-llama/Llama-2-7b-hf",
    doctor_model: str = "meta-llama/Llama-2-7b-hf",
    output_dir: str = "./outputs/pku_saferlhf",
    num_preference_dims: int = 2,
    **kwargs
) -> ExperimentConfig:
    """Create configuration for PKU-SafeRLHF experiments."""
    config = ExperimentConfig()
    
    # Model settings
    config.model.patient_model_name = patient_model
    config.model.doctor_model_name = doctor_model
    config.model.doctor_num_preference_dims = num_preference_dims
    
    # Data settings
    config.data.dataset_name = "PKU-Alignment/PKU-SafeRLHF"
    config.data.preference_dimension = "safety_helpfulness"
    
    # Training settings
    config.training.output_dir = output_dir
    config.training.max_steps = 15000
    
    # Inference settings for multi-dimensional control
    config.inference.preference_weights = {
        "helpfulness": 0.5,
        "safety": 0.5
    }
    
    # Experiment metadata
    config.experiment_name = "pku_saferlhf_multi_dim"
    config.experiment_type = "multi_dim"
    
    # Apply any additional kwargs
    config.update_from_dict(kwargs)
    
    return config


def create_weak_to_strong_config(
    patient_model: str = "allenai/tulu-2-70b",
    doctor_model: str = "allenai/tulu-2-7b",
    output_dir: str = "./outputs/weak_to_strong",
    **kwargs
) -> ExperimentConfig:
    """Create configuration for weak-to-strong guidance experiments."""
    config = ExperimentConfig()
    
    # Model settings
    config.model.patient_model_name = patient_model
    config.model.doctor_model_name = doctor_model
    config.model.patient_load_in_8bit = True  # For 70B model
    
    # Data settings
    config.data.dataset_name = "Anthropic/hh-rlhf"
    config.data.max_samples = 10000  # Smaller dataset for efficiency
    
    # Training settings
    config.training.output_dir = output_dir
    config.training.max_steps = 5000
    config.training.batch_size = 2  # Smaller batch for large models
    
    # Evaluation settings
    config.evaluation.eval_method = "alpacaeval"
    config.evaluation.baseline_methods = ["sft", "dpo", "genarm"]
    
    # Experiment metadata
    config.experiment_name = "weak_to_strong_guidance"
    config.experiment_type = "weak_to_strong"
    
    # Apply any additional kwargs
    config.update_from_dict(kwargs)
    
    return config