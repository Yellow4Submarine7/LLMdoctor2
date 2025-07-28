"""
Configuration module for LLMDoctor experiments.

Provides easy access to configuration classes and utility functions
for creating and loading experiment configurations.
"""

from .base_config import (
    ModelConfig,
    DataConfig,
    RewardConfig,
    TrainingConfig,
    InferenceConfig,
    EvaluationConfig,
    ExperimentConfig,
    create_hh_rlhf_config,
    create_pku_saferlhf_config,
    create_weak_to_strong_config
)

__all__ = [
    # Configuration dataclasses
    "ModelConfig",
    "DataConfig", 
    "RewardConfig",
    "TrainingConfig",
    "InferenceConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    
    # Configuration factories
    "create_hh_rlhf_config",
    "create_pku_saferlhf_config",
    "create_weak_to_strong_config"
]


# Convenience functions for loading pre-defined configurations
def load_hh_rlhf_config() -> ExperimentConfig:
    """Load the default HH-RLHF configuration."""
    import os
    config_path = os.path.join(os.path.dirname(__file__), "hh_rlhf_config.yaml")
    return ExperimentConfig.load(config_path)


def load_pku_saferlhf_config() -> ExperimentConfig:
    """Load the default PKU-SafeRLHF configuration."""
    import os
    config_path = os.path.join(os.path.dirname(__file__), "pku_saferlhf_config.yaml")
    return ExperimentConfig.load(config_path)


def load_weak_to_strong_config() -> ExperimentConfig:
    """Load the default weak-to-strong configuration."""
    import os
    config_path = os.path.join(os.path.dirname(__file__), "weak_to_strong_config.yaml")
    return ExperimentConfig.load(config_path)