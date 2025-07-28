"""
Flow-guided reward model for LLMdoctor framework.

Converts the trained doctor model into a reward model for online alignment.
Provides token-level preference signals during inference.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..models.doctor_model import DoctorModel

logger = logging.getLogger(__name__)


class FlowGuidedRewardModel:
    """
    Flow-guided reward model that uses the trained doctor model to provide
    token-level preference signals for guiding patient model generation.
    
    This model implements the flow-guided reward formulation from the paper,
    where the doctor model outputs log-probability scores π_r(y_{t+1}|s_t)
    for each potential next token based on the learned flow dynamics.
    """
    
    def __init__(
        self,
        doctor_model: DoctorModel,
        preference_dim: int = 0,
        temperature: float = 1.0,
        use_value_head: bool = True,
        normalize_rewards: bool = True,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize flow-guided reward model.
        
        Args:
            doctor_model: Trained DoctorModel instance
            preference_dim: Which preference dimension to use
            temperature: Temperature for reward scaling
            use_value_head: Whether to incorporate value head outputs
            normalize_rewards: Whether to normalize reward scores
            device: Device for computations
        """
        self.doctor_model = doctor_model
        self.preference_dim = preference_dim
        self.temperature = temperature
        self.use_value_head = use_value_head
        self.normalize_rewards = normalize_rewards
        
        if device == "auto":
            self.device = next(doctor_model.parameters()).device
        else:
            self.device = torch.device(device)
        
        # Set model to evaluation mode
        self.doctor_model.eval()
        
        logger.info(f"FlowGuidedRewardModel initialized on {self.device}")
        logger.info(f"Using preference dimension: {preference_dim}")
        logger.info(f"Temperature: {temperature}")
    
    def compute_token_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        candidate_tokens: Optional[torch.Tensor] = None,
        return_logprobs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute token-level reward scores for next token prediction.
        
        Args:
            input_ids: Current sequence [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            candidate_tokens: Specific tokens to evaluate [batch_size, num_candidates]
            return_logprobs: Whether to return log-probabilities
            
        Returns:
            Dictionary with reward scores and optionally log-probabilities
        """
        with torch.no_grad():
            # Forward pass through doctor model
            outputs = self.doctor_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            # Extract logits and values
            logits = outputs["logits"][:, -1, :]  # Last position logits [batch_size, vocab_size]
            
            if self.use_value_head:
                values = outputs["values"][:, -1, self.preference_dim]  # [batch_size]
            else:
                values = torch.ones(logits.shape[0], device=self.device)
            
            # Compute base log-probabilities
            log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            
            # Incorporate value estimates into reward computation
            if self.use_value_head:
                # Modulate log-probs by value estimates
                # π_r(y_{t+1}|s_t) = π_base(y_{t+1}|s_t) * exp(V_φ(s_t))
                value_weights = torch.exp(values / self.temperature).unsqueeze(1)  # [batch_size, 1]
                reward_log_probs = log_probs + torch.log(value_weights)
            else:
                reward_log_probs = log_probs
            
            # Normalize if requested
            if self.normalize_rewards:
                reward_log_probs = F.log_softmax(reward_log_probs, dim=-1)
            
            result = {
                "reward_logprobs": reward_log_probs,  # [batch_size, vocab_size]
                "values": values,  # [batch_size]
            }
            
            if return_logprobs:
                result["base_logprobs"] = log_probs
            
            # Extract specific candidate token scores if provided
            if candidate_tokens is not None:
                batch_size, num_candidates = candidate_tokens.shape
                batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_candidates)
                
                candidate_rewards = reward_log_probs[batch_indices, candidate_tokens]
                result["candidate_rewards"] = candidate_rewards  # [batch_size, num_candidates]
                
                if return_logprobs:
                    candidate_logprobs = log_probs[batch_indices, candidate_tokens]
                    result["candidate_logprobs"] = candidate_logprobs
        
        return result
    
    def compute_sequence_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_per_token: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reward scores for an entire sequence.
        
        Args:
            input_ids: Input sequence [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_per_token: Whether to return per-token scores
            
        Returns:
            Dictionary with sequence and optionally per-token reward scores
        """
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            # Forward pass through entire sequence
            outputs = self.doctor_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            logits = outputs["logits"]  # [batch_size, seq_len, vocab_size]
            
            if self.use_value_head:
                values = outputs["values"][:, :, self.preference_dim]  # [batch_size, seq_len]
            else:
                values = torch.ones((batch_size, seq_len), device=self.device)
            
            # Compute log-probabilities for actual tokens
            log_probs = F.log_softmax(logits / self.temperature, dim=-1)
            
            # Get log-probs for actual next tokens
            token_log_probs = []
            for i in range(seq_len - 1):
                token_logprobs = log_probs[:, i, input_ids[:, i + 1]]
                token_log_probs.append(token_logprobs)
            
            if token_log_probs:
                token_log_probs = torch.stack(token_log_probs, dim=1)  # [batch_size, seq_len-1]
            else:
                token_log_probs = torch.zeros((batch_size, 0), device=self.device)
            
            # Incorporate value estimates
            if self.use_value_head and token_log_probs.shape[1] > 0:
                value_weights = values[:, :-1] / self.temperature  # [batch_size, seq_len-1]
                reward_log_probs = token_log_probs + value_weights
            else:
                reward_log_probs = token_log_probs
            
            # Compute sequence-level score
            if attention_mask is not None:
                mask = attention_mask[:, 1:].float()  # Skip first token
                if mask.shape[1] > reward_log_probs.shape[1]:
                    mask = mask[:, :reward_log_probs.shape[1]]
                elif mask.shape[1] < reward_log_probs.shape[1]:
                    # Pad mask if needed
                    padding = torch.ones((batch_size, reward_log_probs.shape[1] - mask.shape[1]), device=mask.device)
                    mask = torch.cat([mask, padding], dim=1)
                
                sequence_rewards = torch.sum(reward_log_probs * mask, dim=1)
                valid_lengths = torch.sum(mask, dim=1)
                mean_rewards = sequence_rewards / torch.clamp(valid_lengths, min=1.0)
            else:
                sequence_rewards = torch.sum(reward_log_probs, dim=1)
                mean_rewards = torch.mean(reward_log_probs, dim=1)
            
            result = {
                "sequence_rewards": sequence_rewards,  # [batch_size]
                "mean_rewards": mean_rewards,  # [batch_size]
                "sequence_values": torch.mean(values, dim=1),  # [batch_size]
            }
            
            if return_per_token:
                result.update({
                    "token_rewards": reward_log_probs,  # [batch_size, seq_len-1]
                    "token_values": values,  # [batch_size, seq_len]
                    "base_token_logprobs": token_log_probs  # [batch_size, seq_len-1]
                })
        
        return result
    
    def rank_candidates(
        self,
        input_ids: torch.Tensor,
        candidate_sequences: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_scores: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Rank candidate sequences by their reward scores.
        
        Args:
            input_ids: Context sequence [batch_size, context_len]
            candidate_sequences: Candidate sequences [batch_size, num_candidates, seq_len]
            attention_mask: Attention mask for context [batch_size, context_len]
            return_scores: Whether to return individual scores
            
        Returns:
            Dictionary with rankings and optionally scores
        """
        batch_size, num_candidates, seq_len = candidate_sequences.shape
        
        # Concatenate context with each candidate
        context_len = input_ids.shape[1]
        full_sequences = torch.cat([
            input_ids.unsqueeze(1).expand(-1, num_candidates, -1),
            candidate_sequences
        ], dim=2)  # [batch_size, num_candidates, context_len + seq_len]
        
        # Reshape for batch processing
        flat_sequences = full_sequences.view(batch_size * num_candidates, -1)
        
        # Create attention masks
        if attention_mask is not None:
            candidate_masks = torch.ones((batch_size, num_candidates, seq_len), device=self.device)
            full_masks = torch.cat([
                attention_mask.unsqueeze(1).expand(-1, num_candidates, -1),
                candidate_masks
            ], dim=2)
            flat_masks = full_masks.view(batch_size * num_candidates, -1)
        else:
            flat_masks = None
        
        # Compute rewards for all candidates
        reward_results = self.compute_sequence_rewards(
            input_ids=flat_sequences,
            attention_mask=flat_masks,
            return_per_token=False
        )
        
        # Reshape results
        sequence_rewards = reward_results["sequence_rewards"].view(batch_size, num_candidates)
        
        # Rank candidates by reward scores
        sorted_rewards, ranking_indices = torch.sort(sequence_rewards, dim=1, descending=True)
        
        result = {
            "rankings": ranking_indices,  # [batch_size, num_candidates]
            "best_candidate_idx": ranking_indices[:, 0],  # [batch_size]
        }
        
        if return_scores:
            result.update({
                "candidate_scores": sequence_rewards,  # [batch_size, num_candidates]
                "sorted_scores": sorted_rewards,  # [batch_size, num_candidates]
            })
        
        return result
    
    def compute_token_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "gradient"
    ) -> torch.Tensor:
        """
        Compute importance scores for tokens in the input sequence.
        
        Args:
            input_ids: Input sequence [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            method: Method for computing importance ("gradient", "attention", "value")
            
        Returns:
            Token importance scores [batch_size, seq_len]
        """
        if method == "gradient":
            # Use gradient-based importance
            input_ids.requires_grad_(True)
            
            outputs = self.doctor_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use mean value as the scalar to backpropagate
            if self.use_value_head:
                scalar_output = outputs["values"][:, :, self.preference_dim].mean()
            else:
                scalar_output = outputs["logits"].mean()
            
            scalar_output.backward()
            
            importance_scores = torch.abs(input_ids.grad)
            input_ids.grad = None
            
        elif method == "value":
            # Use value head outputs as importance
            with torch.no_grad():
                outputs = self.doctor_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if self.use_value_head:
                    importance_scores = torch.abs(outputs["values"][:, :, self.preference_dim])
                else:
                    # Fallback: use max logit as importance
                    importance_scores = torch.max(torch.abs(outputs["logits"]), dim=-1)[0]
        
        elif method == "attention":
            # Use attention weights as importance (if available)
            with torch.no_grad():
                outputs = self.doctor_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True
                )
                
                if hasattr(outputs, "attentions") and outputs.attentions:
                    # Average attention across heads and layers
                    attentions = torch.stack(outputs.attentions)  # [num_layers, batch_size, num_heads, seq_len, seq_len]
                    avg_attention = torch.mean(attentions, dim=(0, 2))  # [batch_size, seq_len, seq_len]
                    
                    # Sum attention received by each token
                    importance_scores = torch.sum(avg_attention, dim=1)  # [batch_size, seq_len]
                else:
                    logger.warning("Attention weights not available, falling back to value method")
                    return self.compute_token_importance(input_ids, attention_mask, method="value")
        
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Apply attention mask if provided
        if attention_mask is not None:
            importance_scores = importance_scores * attention_mask.float()
        
        return importance_scores
    
    def update_temperature(self, new_temperature: float):
        """Update the temperature parameter."""
        self.temperature = new_temperature
        logger.info(f"Updated temperature to {new_temperature}")
    
    def update_preference_dim(self, new_dim: int):
        """Update the preference dimension."""
        if new_dim >= self.doctor_model.num_preference_dims:
            raise ValueError(f"Preference dimension {new_dim} >= {self.doctor_model.num_preference_dims}")
        
        self.preference_dim = new_dim
        logger.info(f"Updated preference dimension to {new_dim}")
    
    def get_model_info(self) -> Dict[str, Union[int, float, str]]:
        """Get information about the reward model."""
        return {
            "preference_dim": self.preference_dim,
            "temperature": self.temperature,
            "use_value_head": self.use_value_head,
            "normalize_rewards": self.normalize_rewards,
            "num_preference_dims": self.doctor_model.num_preference_dims,
            "device": str(self.device),
            "vocab_size": self.doctor_model.vocab_size
        }