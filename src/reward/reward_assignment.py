"""
Reward assignment module for LLMdoctor framework.

Combines token importance scores with human preference signals to create
directional token-level rewards with sparsity control.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PreferenceLabel(Enum):
    """Preference labels for responses."""
    PREFERRED = 1
    NON_PREFERRED = -1
    NEUTRAL = 0


@dataclass
class RewardAssignmentConfig:
    """Configuration for reward assignment."""
    sparsity_threshold: float = 0.1  # θ in the paper - only tokens above this get rewards
    reward_scale: float = 1.0        # Scale factor for reward values
    enable_sparsity: bool = True     # Whether to apply sparsity threshold
    normalize_rewards: bool = True   # Whether to normalize rewards to [-1, 1]
    min_reward_magnitude: float = 0.01  # Minimum reward magnitude
    reward_clipping: bool = True     # Whether to clip extreme rewards
    clip_min: float = -2.0           # Minimum reward value
    clip_max: float = 2.0            # Maximum reward value


class RewardAssigner:
    """
    Assigns directional token-level rewards based on importance scores and preferences.
    
    Implements the reward assignment from the paper:
    rt = sign(y) · St · 1[St > θ]
    
    Where:
    - rt is the token reward
    - sign(y) is the preference signal (+1 for preferred, -1 for non-preferred)
    - St is the importance score
    - θ is the sparsity threshold
    """
    
    def __init__(
        self,
        config: Optional[RewardAssignmentConfig] = None,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize the reward assigner.
        
        Args:
            config: Configuration for reward assignment
            device: Device for computations
        """
        self.config = config if config is not None else RewardAssignmentConfig()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"RewardAssigner initialized on {self.device}")
    
    def assign_token_rewards(
        self,
        importance_scores: torch.Tensor,
        preference_label: Union[int, PreferenceLabel],
        tokens: Optional[List[str]] = None,
        apply_sparsity: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Assign token-level rewards based on importance scores and preference.
        
        Args:
            importance_scores: Token importance scores [seq_len]
            preference_label: Preference label (1 for preferred, -1 for non-preferred)
            tokens: Optional list of tokens for analysis
            apply_sparsity: Whether to apply sparsity threshold (overrides config)
            
        Returns:
            Dictionary with reward information
        """
        if isinstance(preference_label, PreferenceLabel):
            preference_sign = preference_label.value
        else:
            preference_sign = preference_label
        
        # Convert to tensor if needed
        if not isinstance(importance_scores, torch.Tensor):
            importance_scores = torch.tensor(importance_scores, device=self.device)
        else:
            importance_scores = importance_scores.to(self.device)
        
        # Apply sparsity threshold if enabled
        if apply_sparsity is None:
            apply_sparsity = self.config.enable_sparsity
        
        if apply_sparsity:
            # Sparsity mask: 1[St > θ]
            sparsity_mask = (importance_scores > self.config.sparsity_threshold).float()
        else:
            sparsity_mask = torch.ones_like(importance_scores)
        
        # Compute raw rewards: sign(y) · St · 1[St > θ]
        raw_rewards = preference_sign * importance_scores * sparsity_mask
        
        # Apply reward scaling
        scaled_rewards = raw_rewards * self.config.reward_scale
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            normalized_rewards = self._normalize_rewards(scaled_rewards)
        else:
            normalized_rewards = scaled_rewards
        
        # Apply reward clipping if enabled
        if self.config.reward_clipping:
            clipped_rewards = torch.clamp(
                normalized_rewards,
                self.config.clip_min,
                self.config.clip_max
            )
        else:
            clipped_rewards = normalized_rewards
        
        # Final rewards
        final_rewards = clipped_rewards
        
        # Filter out very small rewards
        magnitude_mask = torch.abs(final_rewards) >= self.config.min_reward_magnitude
        final_rewards = final_rewards * magnitude_mask.float()
        
        result = {
            "rewards": final_rewards,
            "raw_rewards": raw_rewards,
            "importance_scores": importance_scores,
            "sparsity_mask": sparsity_mask,
            "preference_sign": torch.tensor(preference_sign, device=self.device),
            "num_rewarded_tokens": int(torch.sum(sparsity_mask)),
            "total_reward": float(torch.sum(final_rewards)),
            "avg_reward": float(torch.mean(final_rewards[final_rewards != 0])) if torch.any(final_rewards != 0) else 0.0
        }
        
        if tokens is not None:
            result["tokens"] = tokens
            result["rewarded_tokens"] = [
                token for i, token in enumerate(tokens) 
                if i < len(sparsity_mask) and sparsity_mask[i] > 0
            ]
        
        return result
    
    def assign_comparative_rewards(
        self,
        preferred_importance: torch.Tensor,
        non_preferred_importance: torch.Tensor,
        preferred_tokens: Optional[List[str]] = None,
        non_preferred_tokens: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Assign rewards for a pair of preferred vs non-preferred responses.
        
        Args:
            preferred_importance: Importance scores for preferred response
            non_preferred_importance: Importance scores for non-preferred response
            preferred_tokens: Tokens for preferred response
            non_preferred_tokens: Tokens for non-preferred response
            
        Returns:
            Dictionary with rewards for both responses
        """
        # Assign rewards for preferred response
        preferred_rewards = self.assign_token_rewards(
            importance_scores=preferred_importance,
            preference_label=PreferenceLabel.PREFERRED,
            tokens=preferred_tokens
        )
        
        # Assign rewards for non-preferred response
        non_preferred_rewards = self.assign_token_rewards(
            importance_scores=non_preferred_importance,
            preference_label=PreferenceLabel.NON_PREFERRED,
            tokens=non_preferred_tokens
        )
        
        return {
            "preferred": preferred_rewards,
            "non_preferred": non_preferred_rewards,
            "reward_gap": float(preferred_rewards["total_reward"] - non_preferred_rewards["total_reward"])
        }
    
    def assign_batch_rewards(
        self,
        batch_importance: List[torch.Tensor],
        batch_preferences: List[Union[int, PreferenceLabel]],
        batch_tokens: Optional[List[List[str]]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Assign rewards for a batch of responses.
        
        Args:
            batch_importance: List of importance score tensors
            batch_preferences: List of preference labels
            batch_tokens: Optional list of token lists
            progress_callback: Optional progress callback
            
        Returns:
            List of reward dictionaries
        """
        if len(batch_importance) != len(batch_preferences):
            raise ValueError("Number of importance scores and preferences must match")
        
        if batch_tokens is not None and len(batch_tokens) != len(batch_importance):
            raise ValueError("Number of token lists must match importance scores")
        
        results = []
        
        for i, (importance, preference) in enumerate(zip(batch_importance, batch_preferences)):
            tokens = batch_tokens[i] if batch_tokens is not None else None
            
            reward_result = self.assign_token_rewards(
                importance_scores=importance,
                preference_label=preference,
                tokens=tokens
            )
            
            results.append(reward_result)
            
            if progress_callback:
                progress_callback(i + 1, len(batch_importance))
        
        return results
    
    def assign_multi_dimensional_rewards(
        self,
        importance_scores_dict: Dict[str, torch.Tensor],
        preference_labels_dict: Dict[str, Union[int, PreferenceLabel]],
        tokens: Optional[List[str]] = None,
        dimension_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Assign rewards for multiple preference dimensions.
        
        Args:
            importance_scores_dict: Dictionary mapping dimensions to importance scores
            preference_labels_dict: Dictionary mapping dimensions to preference labels
            tokens: Optional list of tokens
            dimension_weights: Optional weights for each dimension
            
        Returns:
            Dictionary with rewards for each dimension
        """
        if set(importance_scores_dict.keys()) != set(preference_labels_dict.keys()):
            raise ValueError("Importance scores and preference labels must have same dimensions")
        
        if dimension_weights is None:
            dimension_weights = {dim: 1.0 for dim in importance_scores_dict.keys()}
        
        results = {}
        
        for dimension in importance_scores_dict.keys():
            importance = importance_scores_dict[dimension]
            preference = preference_labels_dict[dimension]
            weight = dimension_weights.get(dimension, 1.0)
            
            # Scale importance by dimension weight
            weighted_importance = importance * weight
            
            reward_result = self.assign_token_rewards(
                importance_scores=weighted_importance,
                preference_label=preference,
                tokens=tokens
            )
            
            results[dimension] = reward_result
        
        # Compute aggregated rewards
        if len(results) > 1:
            aggregated_rewards = torch.zeros_like(list(results.values())[0]["rewards"])
            total_weight = sum(dimension_weights.values())
            
            for dimension, result in results.items():
                weight = dimension_weights.get(dimension, 1.0)
                aggregated_rewards += result["rewards"] * (weight / total_weight)
            
            results["aggregated"] = {
                "rewards": aggregated_rewards,
                "total_reward": float(torch.sum(aggregated_rewards)),
                "avg_reward": float(torch.mean(aggregated_rewards[aggregated_rewards != 0])) if torch.any(aggregated_rewards != 0) else 0.0
            }
        
        return results
    
    def compute_reward_statistics(
        self,
        reward_results: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute statistics over a batch of reward results.
        
        Args:
            reward_results: List of reward dictionaries
            
        Returns:
            Dictionary with reward statistics
        """
        if not reward_results:
            return {}
        
        # Collect statistics
        total_rewards = [result["total_reward"] for result in reward_results]
        avg_rewards = [result["avg_reward"] for result in reward_results]
        num_rewarded = [result["num_rewarded_tokens"] for result in reward_results]
        
        # Compute overall statistics
        stats = {
            "mean_total_reward": float(np.mean(total_rewards)),
            "std_total_reward": float(np.std(total_rewards)),
            "mean_avg_reward": float(np.mean(avg_rewards)),
            "std_avg_reward": float(np.std(avg_rewards)),
            "mean_rewarded_tokens": float(np.mean(num_rewarded)),
            "std_rewarded_tokens": float(np.std(num_rewarded)),
            "min_total_reward": float(np.min(total_rewards)),
            "max_total_reward": float(np.max(total_rewards)),
            "sparsity_ratio": float(np.mean([r / len(result["rewards"]) for result in reward_results for r in [result["num_rewarded_tokens"]]])),
        }
        
        return stats
    
    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalize rewards to [-1, 1] range.
        
        Args:
            rewards: Raw rewards tensor
            
        Returns:
            Normalized rewards
        """
        if torch.all(rewards == 0):
            return rewards
        
        max_abs_reward = torch.max(torch.abs(rewards))
        if max_abs_reward > 0:
            return rewards / max_abs_reward
        else:
            return rewards
    
    def filter_rewards_by_threshold(
        self,
        reward_result: Dict[str, torch.Tensor],
        threshold: float
    ) -> Dict[str, torch.Tensor]:
        """
        Filter rewards by a custom threshold.
        
        Args:
            reward_result: Reward result dictionary
            threshold: Importance threshold
            
        Returns:
            Filtered reward result
        """
        importance_scores = reward_result["importance_scores"]
        preference_sign = float(reward_result["preference_sign"])
        
        # Apply new threshold
        custom_mask = (importance_scores > threshold).float()
        filtered_rewards = preference_sign * importance_scores * custom_mask
        
        # Apply same normalization and clipping as original
        if self.config.normalize_rewards:
            filtered_rewards = self._normalize_rewards(filtered_rewards)
        
        if self.config.reward_clipping:
            filtered_rewards = torch.clamp(
                filtered_rewards,
                self.config.clip_min,
                self.config.clip_max
            )
        
        # Update result
        filtered_result = reward_result.copy()
        filtered_result.update({
            "rewards": filtered_rewards,
            "sparsity_mask": custom_mask,
            "num_rewarded_tokens": int(torch.sum(custom_mask)),
            "total_reward": float(torch.sum(filtered_rewards)),
            "avg_reward": float(torch.mean(filtered_rewards[filtered_rewards != 0])) if torch.any(filtered_rewards != 0) else 0.0
        })
        
        return filtered_result
    
    def analyze_reward_distribution(
        self,
        reward_result: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[float, int]]:
        """
        Analyze the distribution of assigned rewards.
        
        Args:
            reward_result: Reward result dictionary
            
        Returns:
            Dictionary with distribution analysis
        """
        rewards = reward_result["rewards"]
        nonzero_rewards = rewards[rewards != 0]
        
        if len(nonzero_rewards) == 0:
            return {
                "num_tokens": len(rewards),
                "num_rewarded": 0,
                "sparsity": 1.0,
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0
            }
        
        positive_rewards = nonzero_rewards[nonzero_rewards > 0]
        negative_rewards = nonzero_rewards[nonzero_rewards < 0]
        
        return {
            "num_tokens": len(rewards),
            "num_rewarded": len(nonzero_rewards),
            "sparsity": 1.0 - (len(nonzero_rewards) / len(rewards)),
            "mean_reward": float(torch.mean(nonzero_rewards)),
            "std_reward": float(torch.std(nonzero_rewards)),
            "min_reward": float(torch.min(nonzero_rewards)),
            "max_reward": float(torch.max(nonzero_rewards)),
            "positive_ratio": len(positive_rewards) / len(nonzero_rewards) if len(nonzero_rewards) > 0 else 0.0,
            "negative_ratio": len(negative_rewards) / len(nonzero_rewards) if len(nonzero_rewards) > 0 else 0.0,
            "mean_positive_reward": float(torch.mean(positive_rewards)) if len(positive_rewards) > 0 else 0.0,
            "mean_negative_reward": float(torch.mean(negative_rewards)) if len(negative_rewards) > 0 else 0.0,
        }
    
    def visualize_rewards(
        self,
        reward_result: Dict[str, torch.Tensor],
        max_tokens: int = 50
    ) -> str:
        """
        Create a text visualization of token rewards.
        
        Args:
            reward_result: Reward result dictionary
            max_tokens: Maximum tokens to display
            
        Returns:
            Formatted string with reward visualization
        """
        rewards = reward_result["rewards"][:max_tokens]
        tokens = reward_result.get("tokens", [f"token_{i}" for i in range(len(rewards))])[:max_tokens]
        
        lines = []
        lines.append("Token Reward Visualization:")
        lines.append("=" * 50)
        
        for i, (token, reward) in enumerate(zip(tokens, rewards)):
            reward_val = float(reward)
            
            if reward_val > 0:
                indicator = "🟢" if reward_val > 0.5 else "🟡"
                lines.append(f"{i:3d}: {indicator} {token:<15} (reward: +{reward_val:.3f})")
            elif reward_val < 0:
                indicator = "🔴" if reward_val < -0.5 else "🟠"
                lines.append(f"{i:3d}: {indicator} {token:<15} (reward: {reward_val:.3f})")
            else:
                lines.append(f"{i:3d}:    {token:<15} (reward:  0.000)")
        
        if len(reward_result["rewards"]) > max_tokens:
            lines.append(f"... and {len(reward_result['rewards']) - max_tokens} more tokens")
        
        # Add summary statistics
        stats = self.analyze_reward_distribution(reward_result)
        lines.append("")
        lines.append("Summary Statistics:")
        lines.append(f"  Total tokens: {stats['num_tokens']}")
        lines.append(f"  Rewarded tokens: {stats['num_rewarded']}")
        lines.append(f"  Sparsity: {stats['sparsity']:.3f}")
        lines.append(f"  Mean reward: {stats['mean_reward']:.3f}")
        lines.append(f"  Total reward: {reward_result['total_reward']:.3f}")
        
        return "\n".join(lines)