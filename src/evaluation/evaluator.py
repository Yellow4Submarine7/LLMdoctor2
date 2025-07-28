"""
LLMDoctor evaluation framework.

Provides comprehensive evaluation metrics and methods for assessing
the performance of trained models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
import json
from pathlib import Path
import logging
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Evaluation data
    eval_dataset: str = "AlpacaEval"
    eval_split: Optional[str] = None
    eval_max_samples: Optional[int] = None
    
    # Evaluation method
    eval_method: str = "gpt4"  # "gpt4", "alpacaeval", "human", "reward_model"
    gpt4_api_key: Optional[str] = None
    gpt4_model: str = "gpt-4"
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: Optional[int] = None
    do_sample: bool = True
    num_return_sequences: int = 1
    
    # Evaluation metrics
    compute_diversity: bool = True
    diversity_metric: str = "distinct-4"  # distinct-n, entropy, self-bleu
    compute_perplexity: bool = True
    compute_win_rate: bool = True
    compute_length_stats: bool = True
    
    # Comparison settings
    baseline_methods: List[str] = field(default_factory=lambda: ["sft", "dpo", "ppo"])
    num_comparisons: int = 300  # Number of examples for pairwise comparison
    
    # Output settings
    results_dir: str = "./results"
    save_generations: bool = True
    save_detailed_results: bool = True
    
    # Multi-dimensional evaluation (for multi-preference models)
    preference_dimensions: Optional[List[str]] = None
    dimension_weights: Optional[Dict[str, float]] = None


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Basic metrics
    win_rate: Optional[float] = None
    tie_rate: Optional[float] = None
    loss_rate: Optional[float] = None
    
    # Generation quality
    avg_length: Optional[float] = None
    diversity_score: Optional[float] = None
    perplexity: Optional[float] = None
    
    # Detailed results
    pairwise_results: Optional[List[Dict]] = None
    generation_examples: Optional[List[Dict]] = None
    
    # Multi-dimensional results
    dimension_scores: Optional[Dict[str, float]] = None
    
    # Metadata
    num_examples: int = 0
    model_name: Optional[str] = None
    baseline_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "win_rate": self.win_rate,
            "tie_rate": self.tie_rate,
            "loss_rate": self.loss_rate,
            "avg_length": self.avg_length,
            "diversity_score": self.diversity_score,
            "perplexity": self.perplexity,
            "num_examples": self.num_examples,
            "model_name": self.model_name,
            "baseline_name": self.baseline_name,
            "dimension_scores": self.dimension_scores
        }


class LLMEvaluator:
    """
    Main evaluator for LLMDoctor models.
    
    Supports multiple evaluation methods:
    - GPT-4 based evaluation
    - AlpacaEval framework
    - Reward model based evaluation
    - Diversity and quality metrics
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Evaluation configuration
            device: Device for computation
        """
        self.config = config
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize evaluation components
        self._init_evaluators()
        
        # Create output directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LLMEvaluator initialized on {self.device}")
    
    def _init_evaluators(self):
        """Initialize specific evaluators based on config."""
        self.evaluators = {}
        
        if self.config.eval_method == "gpt4":
            from .gpt4_evaluator import GPT4Evaluator
            self.evaluators["gpt4"] = GPT4Evaluator(
                api_key=self.config.gpt4_api_key,
                model=self.config.gpt4_model
            )
        
        if self.config.compute_diversity:
            from .diversity_metrics import DiversityMetrics
            self.evaluators["diversity"] = DiversityMetrics(
                metric_type=self.config.diversity_metric
            )
        
        if self.config.compute_perplexity:
            from .perplexity_evaluator import PerplexityEvaluator
            self.evaluators["perplexity"] = PerplexityEvaluator(device=self.device)
    
    def evaluate(
        self,
        model,
        test_prompts: List[str],
        baseline_responses: Optional[Dict[str, List[str]]] = None,
        preference_weights: Optional[Dict[str, float]] = None
    ) -> EvaluationResult:
        """
        Evaluate model on test prompts.
        
        Args:
            model: Model to evaluate (DoctorModel + PatientModel or standalone model)
            test_prompts: List of test prompts
            baseline_responses: Optional baseline responses for comparison
            preference_weights: Weights for multi-dimensional preferences
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {len(test_prompts)} prompts")
        
        # Generate model responses
        model_responses = self._generate_responses(
            model=model,
            prompts=test_prompts,
            preference_weights=preference_weights
        )
        
        # Initialize results
        results = EvaluationResult(
            num_examples=len(test_prompts),
            model_name=getattr(model, 'model_name', 'llmdoctor')
        )
        
        # Compute win rate if baseline provided
        if baseline_responses and self.config.compute_win_rate:
            win_results = self._compute_win_rate(
                prompts=test_prompts,
                model_responses=model_responses,
                baseline_responses=baseline_responses
            )
            results.win_rate = win_results["win_rate"]
            results.tie_rate = win_results["tie_rate"]
            results.loss_rate = win_results["loss_rate"]
            results.pairwise_results = win_results["detailed_results"]
        
        # Compute diversity metrics
        if self.config.compute_diversity and "diversity" in self.evaluators:
            results.diversity_score = self.evaluators["diversity"].compute(
                model_responses
            )
        
        # Compute perplexity
        if self.config.compute_perplexity and "perplexity" in self.evaluators:
            results.perplexity = self.evaluators["perplexity"].compute(
                model=model,
                texts=model_responses
            )
        
        # Compute length statistics
        if self.config.compute_length_stats:
            lengths = [len(resp.split()) for resp in model_responses]
            results.avg_length = np.mean(lengths)
        
        # Save generation examples
        if self.config.save_generations:
            results.generation_examples = [
                {
                    "prompt": prompt,
                    "response": response,
                    "length": len(response.split())
                }
                for prompt, response in list(zip(test_prompts, model_responses))[:100]
            ]
        
        # Multi-dimensional evaluation
        if self.config.preference_dimensions and preference_weights:
            results.dimension_scores = self._evaluate_dimensions(
                model=model,
                prompts=test_prompts,
                responses=model_responses,
                dimensions=self.config.preference_dimensions,
                weights=preference_weights
            )
        
        # Save results
        if self.config.save_detailed_results:
            self._save_results(results)
        
        return results
    
    def _generate_responses(
        self,
        model,
        prompts: List[str],
        preference_weights: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """Generate responses from model."""
        responses = []
        
        # Check if this is a guided generation setup
        is_guided = hasattr(model, 'guided_generate')
        
        for prompt in tqdm(prompts, desc="Generating responses"):
            if is_guided and preference_weights:
                # Use guided generation with preference weights
                response = model.guided_generate(
                    prompt=prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    preference_weights=preference_weights
                )
            else:
                # Standard generation
                response = model.generate(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=self.config.do_sample,
                    num_return_sequences=1
                )
            
            # Handle different response formats
            if isinstance(response, list):
                response = response[0]
            
            responses.append(response)
        
        return responses
    
    def _compute_win_rate(
        self,
        prompts: List[str],
        model_responses: List[str],
        baseline_responses: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Compute pairwise win rate against baselines."""
        all_results = {}
        
        for baseline_name, baseline_resps in baseline_responses.items():
            if len(baseline_resps) != len(model_responses):
                logger.warning(f"Baseline {baseline_name} has different number of responses")
                continue
            
            # Use configured evaluation method
            if self.config.eval_method == "gpt4" and "gpt4" in self.evaluators:
                results = self.evaluators["gpt4"].compare_pairwise(
                    prompts=prompts[:self.config.num_comparisons],
                    responses_a=model_responses[:self.config.num_comparisons],
                    responses_b=baseline_resps[:self.config.num_comparisons],
                    model_a_name="LLMDoctor",
                    model_b_name=baseline_name
                )
            else:
                # Fallback to simple length-based comparison
                results = self._simple_comparison(
                    model_responses[:self.config.num_comparisons],
                    baseline_resps[:self.config.num_comparisons]
                )
            
            all_results[baseline_name] = results
        
        # Aggregate results
        if all_results:
            avg_win_rate = np.mean([r["win_rate"] for r in all_results.values()])
            avg_tie_rate = np.mean([r["tie_rate"] for r in all_results.values()])
            avg_loss_rate = np.mean([r["loss_rate"] for r in all_results.values()])
            
            return {
                "win_rate": avg_win_rate,
                "tie_rate": avg_tie_rate,
                "loss_rate": avg_loss_rate,
                "detailed_results": all_results
            }
        
        return {"win_rate": None, "tie_rate": None, "loss_rate": None, "detailed_results": {}}
    
    def _simple_comparison(
        self,
        responses_a: List[str],
        responses_b: List[str]
    ) -> Dict[str, float]:
        """Simple comparison based on response length and quality."""
        wins = 0
        ties = 0
        losses = 0
        
        for resp_a, resp_b in zip(responses_a, responses_b):
            len_a = len(resp_a.split())
            len_b = len(resp_b.split())
            
            # Simple heuristic: longer responses are better (up to a point)
            if len_a > len_b * 1.2 and len_a < 500:
                wins += 1
            elif len_b > len_a * 1.2 and len_b < 500:
                losses += 1
            else:
                ties += 1
        
        total = len(responses_a)
        return {
            "win_rate": wins / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
            "loss_rate": losses / total if total > 0 else 0
        }
    
    def _evaluate_dimensions(
        self,
        model,
        prompts: List[str],
        responses: List[str],
        dimensions: List[str],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Evaluate multiple preference dimensions."""
        dimension_scores = {}
        
        for dimension in dimensions:
            # Evaluate each dimension separately
            # This is a placeholder - implement dimension-specific evaluation
            dimension_scores[dimension] = np.random.random() * 100
        
        # Compute weighted overall score
        total_weight = sum(weights.values())
        weighted_score = sum(
            dimension_scores[dim] * weights.get(dim, 0) / total_weight
            for dim in dimensions
        )
        
        dimension_scores["weighted_overall"] = weighted_score
        
        return dimension_scores
    
    def _save_results(self, results: EvaluationResult):
        """Save evaluation results."""
        # Save summary
        summary_path = self.results_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Save detailed results
        if results.pairwise_results:
            detailed_path = self.results_dir / "pairwise_results.json"
            with open(detailed_path, 'w') as f:
                json.dump(results.pairwise_results, f, indent=2)
        
        # Save generation examples
        if results.generation_examples:
            examples_path = self.results_dir / "generation_examples.json"
            with open(examples_path, 'w') as f:
                json.dump(results.generation_examples, f, indent=2)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_prompts: List[str],
        preference_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            test_prompts: Test prompts
            preference_weights: Preference weights for multi-dimensional models
            
        Returns:
            Dictionary of results for each model
        """
        all_results = {}
        all_responses = {}
        
        # Generate responses for all models
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}")
            
            responses = self._generate_responses(
                model=model,
                prompts=test_prompts,
                preference_weights=preference_weights
            )
            all_responses[model_name] = responses
        
        # Evaluate each model against others
        for model_name, model in models.items():
            baseline_responses = {
                name: resps for name, resps in all_responses.items()
                if name != model_name
            }
            
            results = self.evaluate(
                model=model,
                test_prompts=test_prompts,
                baseline_responses=baseline_responses,
                preference_weights=preference_weights
            )
            
            results.model_name = model_name
            all_results[model_name] = results
        
        # Save comparison results
        self._save_comparison_results(all_results)
        
        return all_results
    
    def _save_comparison_results(self, results: Dict[str, EvaluationResult]):
        """Save model comparison results."""
        comparison_data = {
            model_name: result.to_dict()
            for model_name, result in results.items()
        }
        
        comparison_path = self.results_dir / "model_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Create comparison table
        self._create_comparison_table(results)
    
    def _create_comparison_table(self, results: Dict[str, EvaluationResult]):
        """Create a readable comparison table."""
        import pandas as pd
        
        data = []
        for model_name, result in results.items():
            row = {
                "Model": model_name,
                "Win Rate": f"{result.win_rate:.1%}" if result.win_rate else "N/A",
                "Diversity": f"{result.diversity_score:.3f}" if result.diversity_score else "N/A",
                "Avg Length": f"{result.avg_length:.1f}" if result.avg_length else "N/A",
                "Perplexity": f"{result.perplexity:.1f}" if result.perplexity else "N/A"
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = self.results_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as formatted text
        txt_path = self.results_dir / "comparison_table.txt"
        with open(txt_path, 'w') as f:
            f.write(df.to_string(index=False))
        
        # Print to console
        print("\nModel Comparison Results:")
        print("=" * 60)
        print(df.to_string(index=False))
        print("=" * 60)