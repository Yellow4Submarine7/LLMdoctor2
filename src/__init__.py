"""
LLMdoctor: Token-Level Flow-Guided Preference Optimization for Efficient Test-Time Alignment of Large Language Models

This package implements the LLMdoctor framework as described in the paper:
"LLMdoctor: Token-Level Flow-Guided Preference Optimization for Efficient Test-Time Alignment of Large Language Models"

The framework consists of three main stages:
1. Token-Level Reward Acquisition: Extract fine-grained token-level rewards from patient model behavioral variations
2. TFPO Training: Train a doctor model using Token-level Flow-guided Preference Optimization
3. Online Alignment: Use the trained doctor model to guide patient model during inference

Key components:
- models: Patient and Doctor model implementations
- reward: Token-level reward acquisition utilities
- training: TFPO training implementation
- inference: Online alignment and reward-guided decoding
- data: Dataset processing utilities
- evaluation: Evaluation metrics and comparison frameworks
"""

__version__ = "1.0.0"
__author__ = "LLMdoctor Team"

from . import models
from . import reward
from . import training
from . import inference
from . import data
from . import evaluation

__all__ = [
    "models",
    "reward", 
    "training",
    "inference",
    "data",
    "evaluation"
]