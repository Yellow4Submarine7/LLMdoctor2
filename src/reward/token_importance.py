"""
Token importance calculator for LLMdoctor framework.

Computes token-level importance scores by measuring log-likelihood differences
between positive and negative behavioral variants of the patient model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenImportanceConfig:
    """Configuration for token importance calculation."""
    epsilon: float = 1e-8  # Small constant to prevent division by zero
    tau: float = 1.0       # Temperature parameter for smoothing
    normalization_method: str = "mean"  # "mean", "max", "std"
    smoothing_function: str = "tanh"    # "tanh", "sigmoid", "linear"
    min_importance_threshold: float = 0.01  # Minimum importance to consider
    batch_size: int = 8     # Batch size for processing
    max_sequence_length: int = 2048  # Maximum sequence length


class TokenImportanceCalculator:
    """
    Calculates token-level importance scores based on behavioral variants.
    
    Implements the core importance measurement from the paper:
    Δt = |ℓ^pos_t - ℓ^neg_t| - absolute difference in log-likelihoods
    St = tanh(Δ̂t/τ) - normalized and smoothed importance score
    """
    
    def __init__(
        self,
        config: Optional[TokenImportanceConfig] = None,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize the token importance calculator.
        
        Args:
            config: Configuration for importance calculation
            device: Device for computations
        """
        self.config = config if config is not None else TokenImportanceConfig()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"TokenImportanceCalculator initialized on {self.device}")
    
    def compute_token_importance(
        self,
        patient_model,
        prompt: str,
        response: str,
        positive_instruction: str = None,
        negative_instruction: str = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute token-level importance scores for a response.
        
        Args:
            patient_model: PatientModel instance
            prompt: Original prompt
            response: Response to analyze
            positive_instruction: Positive behavioral instruction
            negative_instruction: Negative behavioral instruction
            return_details: Whether to return detailed information
            
        Returns:
            Importance scores tensor or detailed results dictionary
        """
        # Create behavioral variants
        from .behavioral_variants import BehavioralVariantCreator, BehavioralVariantConfig
        
        config = BehavioralVariantConfig()
        if positive_instruction:
            config.positive_instruction = positive_instruction
        if negative_instruction:
            config.negative_instruction = negative_instruction
        
        variant_creator = BehavioralVariantCreator(config)
        pos_prompt, neg_prompt = variant_creator.create_variants(prompt)
        
        # Compute log-likelihoods under both variants
        pos_logprobs, tokens = patient_model.get_sequence_logprobs(
            pos_prompt, response, return_per_token=True
        )
        neg_logprobs, _ = patient_model.get_sequence_logprobs(
            neg_prompt, response, return_per_token=True
        )
        
        # Ensure same length (handle potential tokenization differences)
        min_len = min(len(pos_logprobs), len(neg_logprobs))
        pos_logprobs = pos_logprobs[:min_len]
        neg_logprobs = neg_logprobs[:min_len]
        tokens = tokens[:min_len]
        
        # Compute raw importance scores
        raw_differences = torch.abs(pos_logprobs - neg_logprobs)
        
        # Normalize and smooth
        importance_scores = self._normalize_and_smooth(raw_differences)
        
        if return_details:
            return {
                "importance_scores": importance_scores,
                "raw_differences": raw_differences,
                "pos_logprobs": pos_logprobs,
                "neg_logprobs": neg_logprobs,
                "tokens": tokens,
                "positive_prompt": pos_prompt,
                "negative_prompt": neg_prompt
            }
        
        return importance_scores
    
    def compute_batch_importance(
        self,
        patient_model,
        prompts: List[str],
        responses: List[str],
        dimension: str = "helpfulness",
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Compute token importance for a batch of prompt-response pairs.
        
        Args:
            patient_model: PatientModel instance
            prompts: List of prompts
            responses: List of responses
            dimension: Preference dimension
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of importance score dictionaries
        """
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must match")
        
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            batch_results = []
            for prompt, response in zip(batch_prompts, batch_responses):
                result = self.compute_token_importance(
                    patient_model=patient_model,
                    prompt=prompt,
                    response=response,
                    return_details=True
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if progress_callback:
                progress_callback(min(i + batch_size, len(prompts)), len(prompts))
        
        return results
    
    def compute_comparative_importance(
        self,
        patient_model,
        prompt: str,
        preferred_response: str,
        non_preferred_response: str,
        dimension: str = "helpfulness"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comparative importance between preferred and non-preferred responses.
        
        Args:
            patient_model: PatientModel instance
            prompt: Original prompt
            preferred_response: Preferred response
            non_preferred_response: Non-preferred response
            dimension: Preference dimension
            
        Returns:
            Dictionary with comparative importance scores
        """
        # Compute importance for both responses
        preferred_result = self.compute_token_importance(
            patient_model=patient_model,
            prompt=prompt,
            response=preferred_response,
            return_details=True
        )
        
        non_preferred_result = self.compute_token_importance(
            patient_model=patient_model,
            prompt=prompt,
            response=non_preferred_response,
            return_details=True
        )
        
        return {
            "preferred_importance": preferred_result["importance_scores"],
            "non_preferred_importance": non_preferred_result["importance_scores"],
            "preferred_tokens": preferred_result["tokens"],
            "non_preferred_tokens": non_preferred_result["tokens"],
            "importance_difference": (
                preferred_result["importance_scores"] - 
                non_preferred_result["importance_scores"]
            )
        }
    
    def _normalize_and_smooth(
        self,
        raw_differences: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize and smooth raw importance differences.
        
        Args:
            raw_differences: Raw absolute differences
            
        Returns:
            Normalized and smoothed importance scores
        """
        # Normalize
        if self.config.normalization_method == "mean":
            mean_diff = torch.mean(raw_differences) + self.config.epsilon
            normalized = raw_differences / mean_diff
        elif self.config.normalization_method == "max":
            max_diff = torch.max(raw_differences) + self.config.epsilon
            normalized = raw_differences / max_diff
        elif self.config.normalization_method == "std":
            std_diff = torch.std(raw_differences) + self.config.epsilon
            normalized = raw_differences / std_diff
        else:
            normalized = raw_differences
        
        # Apply smoothing function
        if self.config.smoothing_function == "tanh":
            smoothed = torch.tanh(normalized / self.config.tau)
        elif self.config.smoothing_function == "sigmoid":
            smoothed = torch.sigmoid(normalized / self.config.tau)
        elif self.config.smoothing_function == "linear":
            smoothed = torch.clamp(normalized / self.config.tau, 0, 1)
        else:
            smoothed = normalized
        
        return smoothed
    
    def filter_important_tokens(
        self,
        importance_scores: torch.Tensor,
        tokens: List[str],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Filter tokens based on importance scores.
        
        Args:
            importance_scores: Token importance scores
            tokens: List of tokens
            threshold: Minimum importance threshold
            top_k: Keep only top-k most important tokens
            top_p: Keep tokens with cumulative importance up to top_p
            
        Returns:
            Tuple of (filtered_scores, filtered_tokens, indices)
        """
        if threshold is None:
            threshold = self.config.min_importance_threshold
        
        # Apply threshold filter
        mask = importance_scores >= threshold
        
        if top_k is not None:
            # Keep only top-k tokens
            _, top_indices = torch.topk(importance_scores, min(top_k, len(importance_scores)))
            top_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
            top_mask[top_indices] = True
            mask = mask & top_mask
        
        if top_p is not None:
            # Keep tokens with cumulative importance up to top_p
            sorted_scores, sorted_indices = torch.sort(importance_scores, descending=True)
            cumulative_scores = torch.cumsum(sorted_scores, dim=0)
            normalized_cumsum = cumulative_scores / cumulative_scores[-1]
            
            cutoff_idx = torch.searchsorted(normalized_cumsum, top_p, right=True) + 1
            top_p_indices = sorted_indices[:cutoff_idx]
            
            top_p_mask = torch.zeros_like(importance_scores, dtype=torch.bool)
            top_p_mask[top_p_indices] = True
            mask = mask & top_p_mask
        
        # Apply filter
        indices = torch.where(mask)[0]
        filtered_scores = importance_scores[mask]
        filtered_tokens = [tokens[i] for i in indices.tolist()]
        
        return filtered_scores, filtered_tokens, indices
    
    def analyze_token_patterns(
        self,
        importance_scores: torch.Tensor,
        tokens: List[str],
        min_frequency: int = 2
    ) -> Dict[str, float]:
        """
        Analyze patterns in important tokens.
        
        Args:
            importance_scores: Token importance scores
            tokens: List of tokens
            min_frequency: Minimum frequency for pattern analysis
            
        Returns:
            Dictionary with pattern analysis results
        """
        # Filter important tokens
        filtered_scores, filtered_tokens, _ = self.filter_important_tokens(
            importance_scores, tokens
        )
        
        if len(filtered_tokens) == 0:
            return {"avg_importance": 0.0, "token_diversity": 0.0, "patterns": {}}
        
        # Compute statistics
        avg_importance = float(torch.mean(filtered_scores))
        token_diversity = len(set(filtered_tokens)) / len(filtered_tokens)
        
        # Find frequent tokens
        token_counts = {}
        token_importance_sum = {}
        
        for token, score in zip(filtered_tokens, filtered_scores):
            token_counts[token] = token_counts.get(token, 0) + 1
            token_importance_sum[token] = token_importance_sum.get(token, 0.0) + float(score)
        
        # Filter by frequency and compute average importance
        frequent_patterns = {}
        for token, count in token_counts.items():
            if count >= min_frequency:
                avg_score = token_importance_sum[token] / count
                frequent_patterns[token] = {
                    "frequency": count,
                    "avg_importance": avg_score,
                    "total_importance": token_importance_sum[token]
                }
        
        return {
            "avg_importance": avg_importance,
            "token_diversity": token_diversity,
            "num_important_tokens": len(filtered_tokens),
            "patterns": frequent_patterns
        }
    
    def visualize_importance(
        self,
        importance_scores: torch.Tensor,
        tokens: List[str],
        threshold: Optional[float] = None,
        max_tokens: int = 100
    ) -> str:
        """
        Create a text visualization of token importance.
        
        Args:
            importance_scores: Token importance scores
            tokens: List of tokens
            threshold: Importance threshold for highlighting
            max_tokens: Maximum tokens to display
            
        Returns:
            Formatted string with importance visualization
        """
        if threshold is None:
            threshold = self.config.min_importance_threshold
        
        # Limit display length
        display_scores = importance_scores[:max_tokens]
        display_tokens = tokens[:max_tokens]
        
        # Create visualization
        lines = []
        lines.append("Token Importance Visualization:")
        lines.append("=" * 50)
        
        for i, (token, score) in enumerate(zip(display_tokens, display_scores)):
            score_val = float(score)
            
            # Create importance indicator
            if score_val >= threshold:
                indicator = "🔥" if score_val > 0.7 else "⭐" if score_val > 0.4 else "💡"
                lines.append(f"{i:3d}: {indicator} {token:<15} (importance: {score_val:.3f})")
            else:
                lines.append(f"{i:3d}:    {token:<15} (importance: {score_val:.3f})")
        
        if len(tokens) > max_tokens:
            lines.append(f"... and {len(tokens) - max_tokens} more tokens")
        
        return "\n".join(lines)
    
    def save_importance_data(
        self,
        importance_data: List[Dict[str, torch.Tensor]],
        filepath: str,
        format: str = "json"
    ):
        """
        Save importance data to file.
        
        Args:
            importance_data: List of importance score dictionaries
            filepath: Output file path
            format: Output format ("json", "pickle")
        """
        if format == "json":
            import json
            
            # Convert tensors to lists for JSON serialization
            json_data = []
            for item in importance_data:
                json_item = {}
                for key, value in item.items():
                    if isinstance(value, torch.Tensor):
                        json_item[key] = value.tolist()
                    else:
                        json_item[key] = value
                json_data.append(json_item)
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        elif format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(importance_data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Importance data saved to {filepath}")
    
    def load_importance_data(
        self,
        filepath: str,
        format: str = "json"
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Load importance data from file.
        
        Args:
            filepath: Input file path
            format: Input format ("json", "pickle")
            
        Returns:
            List of importance score dictionaries
        """
        if format == "json":
            import json
            
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            # Convert lists back to tensors
            importance_data = []
            for item in json_data:
                converted_item = {}
                for key, value in item.items():
                    if isinstance(value, list) and key.endswith(('_scores', '_logprobs', '_differences')):
                        converted_item[key] = torch.tensor(value)
                    else:
                        converted_item[key] = value
                importance_data.append(converted_item)
            
            return importance_data
            
        elif format == "pickle":
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Importance data loaded from {filepath}")