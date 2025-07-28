"""
Evaluation framework for LLMDoctor models.

Provides comprehensive evaluation metrics and methods for assessing
model performance including win rates, diversity, perplexity, and more.
"""

from .evaluator import LLMEvaluator, EvaluationConfig, EvaluationResult
from .diversity_metrics import DiversityMetrics
from .perplexity_evaluator import PerplexityEvaluator
from .gpt4_evaluator import GPT4Evaluator

__all__ = [
    "LLMEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    "DiversityMetrics",
    "PerplexityEvaluator",
    "GPT4Evaluator"
]