"""
Flow-based loss functions for TFPO training in LLMdoctor framework.

Implements the Subtrajectory Balance (SubTB) loss and Value Discrimination loss
based on GFlowNet principles for token-level flow-guided preference optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class SubtrajectoryBalanceLoss(nn.Module):
    """
    Subtrajectory Balance (SubTB) loss for TFPO training.
    
    Implements the core flow balance loss from the paper:
    L_SubTB = Σ(log(Q(s_m)V_φ(s_n) / Q(s_n)V_φ(s_m)) - Σlog(π̂_θ(y_{k+1}|s_k)))²
    
    This loss enforces flow conservation across all subtrajectories in token sequences.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
        eps: float = 1e-8
    ):
        """
        Initialize SubTB loss.
        
        Args:
            reduction: Reduction method ("mean", "sum", "none")
            clamp_min: Minimum value for log clamping
            clamp_max: Maximum value for log clamping
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.reduction = reduction
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eps = eps
    
    def forward(
        self,
        policy_logprobs: torch.Tensor,
        values: torch.Tensor,
        prefix_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_subtrajectory_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SubTB loss.
        
        Args:
            policy_logprobs: Log probabilities from policy π̂_θ [batch_size, seq_len, vocab_size]
            values: Value estimates V_φ [batch_size, seq_len, num_dims] or [batch_size, seq_len]
            prefix_scores: Q(s_t) scores [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_subtrajectory_length: Maximum subtrajectory length to consider
            
        Returns:
            Dictionary with loss and diagnostics
        """
        batch_size, seq_len = prefix_scores.shape
        device = prefix_scores.device
        
        # Handle multi-dimensional values
        if values.dim() == 3:
            values = values[:, :, 0]  # Use first dimension
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.bool()
        else:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Compute flow: F(s_t) = Q(s_t) * V_φ(s_t)
        flows = prefix_scores * values  # [batch_size, seq_len]
        
        # Clamp flows for numerical stability
        flows = torch.clamp(flows, min=self.eps)
        log_flows = torch.log(flows)
        log_flows = torch.clamp(log_flows, self.clamp_min, self.clamp_max)
        
        # Initialize loss accumulator
        total_loss = torch.zeros(batch_size, device=device)
        num_subtrajectories = torch.zeros(batch_size, device=device)
        
        # Set maximum subtrajectory length
        if max_subtrajectory_length is None:
            max_subtrajectory_length = seq_len
        else:
            max_subtrajectory_length = min(max_subtrajectory_length, seq_len)
        
        # Iterate over all possible subtrajectories
        for m in range(seq_len):
            for n in range(m + 1, min(m + max_subtrajectory_length + 1, seq_len)):
                # Check if subtrajectory is valid (within attention mask)
                subtrajectory_mask = mask[:, m:n].all(dim=1)  # [batch_size]
                
                if not subtrajectory_mask.any():
                    continue
                
                # Extract relevant values for this subtrajectory
                log_flow_m = log_flows[:, m]  # [batch_size]
                log_flow_n = log_flows[:, n] if n < seq_len else log_flows[:, -1]  # [batch_size]
                
                # Compute log policy probabilities for subtrajectory
                # Σ_{k=m}^{n-1} log π̂_θ(y_{k+1} | s_k)
                subtrajectory_logprobs = torch.zeros(batch_size, device=device)
                
                for k in range(m, min(n, seq_len - 1)):
                    # Get the actual token at position k+1
                    if k + 1 < policy_logprobs.shape[1]:
                        # For training, we need to extract the log-prob of the actual next token
                        # This assumes policy_logprobs contains log-probs for the actual sequence
                        subtrajectory_logprobs += policy_logprobs[:, k]
                
                # Compute flow balance term: log(F(s_m) / F(s_n))
                log_flow_ratio = log_flow_m - log_flow_n
                
                # SubTB loss for this subtrajectory: (log_flow_ratio - subtrajectory_logprobs)^2
                subtb_term = (log_flow_ratio - subtrajectory_logprobs) ** 2
                
                # Apply subtrajectory mask and accumulate
                masked_loss = subtb_term * subtrajectory_mask.float()
                total_loss += masked_loss
                num_subtrajectories += subtrajectory_mask.float()
        
        # Normalize by number of valid subtrajectories
        num_subtrajectories = torch.clamp(num_subtrajectories, min=1.0)
        normalized_loss = total_loss / num_subtrajectories
        
        # Apply reduction
        if self.reduction == "mean":
            loss = torch.mean(normalized_loss)
        elif self.reduction == "sum":
            loss = torch.sum(normalized_loss)
        else:
            loss = normalized_loss
        
        return {
            "loss": loss,
            "raw_loss": total_loss,
            "num_subtrajectories": num_subtrajectories,
            "normalized_loss": normalized_loss,
            "mean_flow": torch.mean(flows),
            "std_flow": torch.std(flows)
        }
    
    def compute_subtrajectory_loss_efficient(
        self,
        policy_logprobs: torch.Tensor,
        values: torch.Tensor,
        prefix_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stride: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        More efficient version that uses strided computation.
        
        Args:
            policy_logprobs: Policy log probabilities [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            prefix_scores: Prefix scores [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            stride: Stride for subtrajectory sampling
            
        Returns:
            Dictionary with loss and diagnostics
        """
        batch_size, seq_len = prefix_scores.shape
        device = prefix_scores.device
        
        # Handle multi-dimensional values
        if values.dim() == 3:
            values = values[:, :, 0]
        
        # Compute flows
        flows = torch.clamp(prefix_scores * values, min=self.eps)
        log_flows = torch.clamp(torch.log(flows), self.clamp_min, self.clamp_max)
        
        # Create all valid subtrajectory pairs with stride
        losses = []
        
        for gap in range(1, seq_len, stride):
            for start in range(0, seq_len - gap, stride):
                end = start + gap
                
                # Flow balance term
                log_flow_ratio = log_flows[:, start] - log_flows[:, end]
                
                # Policy probability sum
                policy_sum = torch.sum(policy_logprobs[:, start:end], dim=1)
                
                # SubTB loss term
                subtb_term = (log_flow_ratio - policy_sum) ** 2
                losses.append(subtb_term)
        
        if losses:
            stacked_losses = torch.stack(losses, dim=0)  # [num_subtrajectories, batch_size]
            
            if self.reduction == "mean":
                loss = torch.mean(stacked_losses)
            elif self.reduction == "sum":
                loss = torch.sum(stacked_losses)
            else:
                loss = torch.mean(stacked_losses, dim=0)
        else:
            loss = torch.zeros(batch_size if self.reduction == "none" else [], device=device)
        
        return {
            "loss": loss,
            "num_subtrajectories": len(losses),
            "mean_flow": torch.mean(flows),
            "std_flow": torch.std(flows)
        }


class ValueDiscriminationLoss(nn.Module):
    """
    Value discrimination loss for TFPO training.
    
    Ensures the value function correctly discriminates between tokens
    based on their reward values. Implements a margin-based ranking loss.
    """
    
    def __init__(
        self,
        margin: float = 0.1,
        reduction: str = "mean",
        loss_type: str = "hinge"  # "hinge", "mse", "ranking"
    ):
        """
        Initialize value discrimination loss.
        
        Args:
            margin: Margin for ranking loss
            reduction: Reduction method ("mean", "sum", "none")
            loss_type: Type of discrimination loss
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        self.loss_type = loss_type
    
    def forward(
        self,
        values: torch.Tensor,
        token_rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute value discrimination loss.
        
        Args:
            values: Value estimates [batch_size, seq_len, num_dims] or [batch_size, seq_len]
            token_rewards: Token-level rewards [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary with loss and diagnostics
        """
        batch_size, seq_len = token_rewards.shape
        device = token_rewards.device
        
        # Handle multi-dimensional values
        if values.dim() == 3:
            values = values[:, :, 0]  # Use first dimension
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.bool()
            values = values * mask.float()
            token_rewards = token_rewards * mask.float()
        else:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        
        if self.loss_type == "mse":
            # Direct MSE between values and rewards
            loss_per_token = (values - token_rewards) ** 2
            
        elif self.loss_type == "hinge":
            # Hinge loss for preference pairs
            loss_per_token = torch.zeros_like(values)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    if i != j:
                        # If reward_i > reward_j, then value_i should be > value_j
                        reward_diff = token_rewards[:, i] - token_rewards[:, j]
                        value_diff = values[:, i] - values[:, j]
                        
                        # Hinge loss: max(0, margin - (value_diff * sign(reward_diff)))
                        expected_sign = torch.sign(reward_diff)
                        hinge_term = torch.clamp(
                            self.margin - (value_diff * expected_sign), 
                            min=0
                        )
                        
                        loss_per_token[:, i] += hinge_term
            
            loss_per_token /= max(seq_len - 1, 1)  # Normalize by comparisons
            
        elif self.loss_type == "ranking":
            # Ranking loss based on reward ordering
            loss_per_token = torch.zeros_like(values)
            
            # Sort rewards and values
            sorted_rewards, reward_indices = torch.sort(token_rewards, dim=1, descending=True)
            
            # Gather values according to reward ordering
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
            sorted_values = values[batch_indices, reward_indices]
            
            # Compute ranking loss: penalize inversions
            for i in range(seq_len - 1):
                # Values should be in descending order too
                value_violations = torch.clamp(
                    sorted_values[:, i + 1] - sorted_values[:, i] + self.margin,
                    min=0
                )
                loss_per_token[:, reward_indices[:, i]] += value_violations
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply mask and reduction
        if attention_mask is not None:
            loss_per_token = loss_per_token * mask.float()
            valid_tokens = torch.sum(mask.float(), dim=1)
            batch_loss = torch.sum(loss_per_token, dim=1) / torch.clamp(valid_tokens, min=1.0)
        else:
            batch_loss = torch.mean(loss_per_token, dim=1)
        
        if self.reduction == "mean":
            loss = torch.mean(batch_loss)
        elif self.reduction == "sum":
            loss = torch.sum(batch_loss)
        else:
            loss = batch_loss
        
        # Compute diagnostics
        value_reward_corr = self._compute_correlation(values, token_rewards, mask)
        
        return {
            "loss": loss,
            "value_reward_correlation": value_reward_corr,
            "mean_value": torch.mean(values),
            "std_value": torch.std(values),
            "mean_reward": torch.mean(token_rewards),
            "std_reward": torch.std(token_rewards)
        }
    
    def _compute_correlation(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute correlation between values and rewards."""
        # Flatten and apply mask
        flat_values = values[mask]
        flat_rewards = rewards[mask]
        
        if len(flat_values) < 2:
            return torch.tensor(0.0, device=values.device)
        
        # Compute Pearson correlation
        mean_values = torch.mean(flat_values)
        mean_rewards = torch.mean(flat_rewards)
        
        numerator = torch.sum((flat_values - mean_values) * (flat_rewards - mean_rewards))
        
        values_var = torch.sum((flat_values - mean_values) ** 2)
        rewards_var = torch.sum((flat_rewards - mean_rewards) ** 2)
        
        denominator = torch.sqrt(values_var * rewards_var)
        
        if denominator > 0:
            correlation = numerator / denominator
        else:
            correlation = torch.tensor(0.0, device=values.device)
        
        return correlation


class TFPOLoss(nn.Module):
    """
    Combined TFPO loss that includes both SubTB and Value Discrimination losses.
    
    L_TFPO = L_SubTB + λ * L_value
    """
    
    def __init__(
        self,
        lambda_value: float = 1.0,
        subtb_config: Optional[Dict] = None,
        value_config: Optional[Dict] = None
    ):
        """
        Initialize combined TFPO loss.
        
        Args:
            lambda_value: Weight for value discrimination loss
            subtb_config: Configuration for SubTB loss
            value_config: Configuration for value discrimination loss
        """
        super().__init__()
        self.lambda_value = lambda_value
        
        # Initialize sub-losses
        subtb_config = subtb_config or {}
        value_config = value_config or {}
        
        self.subtb_loss = SubtrajectoryBalanceLoss(**subtb_config)
        self.value_loss = ValueDiscriminationLoss(**value_config)
    
    def forward(
        self,
        policy_logprobs: torch.Tensor,
        values: torch.Tensor,
        prefix_scores: torch.Tensor,
        token_rewards: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_components: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined TFPO loss.
        
        Args:
            policy_logprobs: Policy log probabilities
            values: Value estimates
            prefix_scores: Prefix scores Q(s_t)
            token_rewards: Token-level rewards
            attention_mask: Attention mask
            return_components: Whether to return individual loss components
            
        Returns:
            Dictionary with total loss and components
        """
        # Compute SubTB loss
        subtb_result = self.subtb_loss(
            policy_logprobs=policy_logprobs,
            values=values,
            prefix_scores=prefix_scores,
            attention_mask=attention_mask
        )
        
        # Compute value discrimination loss
        value_result = self.value_loss(
            values=values,
            token_rewards=token_rewards,
            attention_mask=attention_mask
        )
        
        # Combine losses
        total_loss = subtb_result["loss"] + self.lambda_value * value_result["loss"]
        
        result = {"loss": total_loss}
        
        if return_components:
            result.update({
                "subtb_loss": subtb_result["loss"],
                "value_loss": value_result["loss"],
                "subtb_diagnostics": subtb_result,
                "value_diagnostics": value_result
            })
        
        return result
    
    def update_lambda(self, new_lambda: float):
        """Update the lambda weight for value loss."""
        self.lambda_value = new_lambda
        logger.info(f"Updated lambda_value to {new_lambda}")


def compute_prefix_scores(
    token_rewards: torch.Tensor,
    method: str = "cumulative"
) -> torch.Tensor:
    """
    Compute prefix scores Q(s_t) from token-level rewards.
    
    Args:
        token_rewards: Token-level rewards [batch_size, seq_len]
        method: Method for computing prefix scores ("cumulative", "mean", "max")
        
    Returns:
        Prefix scores [batch_size, seq_len]
    """
    if method == "cumulative":
        # Cumulative sum of rewards up to each position
        prefix_scores = torch.cumsum(token_rewards, dim=1)
        
    elif method == "mean":
        # Mean reward up to each position
        cumsum = torch.cumsum(token_rewards, dim=1)
        positions = torch.arange(1, token_rewards.shape[1] + 1, device=token_rewards.device)
        prefix_scores = cumsum / positions.unsqueeze(0)
        
    elif method == "max":
        # Maximum reward up to each position
        prefix_scores = torch.cummax(token_rewards, dim=1)[0]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return prefix_scores