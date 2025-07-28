"""
Doctor Model implementation for LLMdoctor framework.

The DoctorModel is a smaller model trained via TFPO to internalize preference patterns
and guide the PatientModel during inference. It includes both a policy head and value head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel
)
import logging

logger = logging.getLogger(__name__)


class DoctorModel(nn.Module):
    """
    The Doctor Model for LLMdoctor framework.
    
    This model:
    1. Is trained using TFPO to learn token-level preferences
    2. Has both policy and value heads
    3. Guides the PatientModel during inference
    4. Supports multi-dimensional preference modeling
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        num_preference_dims: int = 1,
        value_head_hidden_size: int = 512,
        device: Union[str, torch.device] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        use_cache: bool = True,
        freeze_base_model: bool = False,
    ):
        """
        Initialize the DoctorModel.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            num_preference_dims: Number of preference dimensions (e.g., helpfulness, safety)
            value_head_hidden_size: Hidden size for value head
            device: Device to load the model on
            torch_dtype: PyTorch data type for the model
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            trust_remote_code: Whether to trust remote code in model
            use_cache: Whether to use key-value cache during generation
            freeze_base_model: Whether to freeze the base model parameters
        """
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.num_preference_dims = num_preference_dims
        self.value_head_hidden_size = value_head_hidden_size
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        self.freeze_base_model = freeze_base_model
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="left"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "use_cache": use_cache,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        
        # Move to device if not using quantization
        if not (load_in_8bit or load_in_4bit):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.base_model = self.base_model.to(device)
        
        # Freeze base model if specified
        if freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Get model configuration
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        
        # Initialize value heads for each preference dimension
        self.value_heads = nn.ModuleList([
            self._create_value_head() for _ in range(num_preference_dims)
        ])
        
        # Move value heads to device
        self.value_heads = self.value_heads.to(self.base_model.device)
        
        logger.info(f"Loaded DoctorModel: {model_name_or_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        logger.info(f"Number of preference dimensions: {num_preference_dims}")
        
    def _create_value_head(self) -> nn.Module:
        """Create a value head for estimating state values."""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.value_head_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.value_head_hidden_size, self.value_head_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.value_head_hidden_size // 2, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the doctor model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss (optional)
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
            - logits: Language modeling logits [batch_size, seq_len, vocab_size]
            - values: Value estimates for each preference dim [batch_size, seq_len, num_dims]
            - hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            - loss: Language modeling loss (if labels provided)
        """
        # Forward pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # Extract hidden states
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Compute value estimates for each preference dimension
        values = []
        for value_head in self.value_heads:
            value = value_head(hidden_states)  # [batch_size, seq_len, 1]
            values.append(value.squeeze(-1))   # [batch_size, seq_len]
        
        values = torch.stack(values, dim=-1)  # [batch_size, seq_len, num_dims]
        
        result = {
            "logits": outputs.logits,
            "values": values,
            "hidden_states": hidden_states,
        }
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss
            
        return result
    
    def get_policy_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get log-probabilities from the policy (language modeling head).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            target_ids: Target token IDs to extract log-probs for [batch_size, target_len]
            
        Returns:
            Log-probabilities [batch_size, seq_len, vocab_size] or 
            [batch_size, target_len] if target_ids provided
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        logits = outputs["logits"]
        log_probs = F.log_softmax(logits, dim=-1)
        
        if target_ids is not None:
            # Extract log-probs for specific tokens
            batch_size, target_len = target_ids.shape
            batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, target_len)
            seq_indices = torch.arange(target_len).unsqueeze(0).expand(batch_size, -1)
            
            # Make sure we don't go out of bounds
            seq_indices = torch.clamp(seq_indices, 0, log_probs.shape[1] - 1)
            
            selected_log_probs = log_probs[batch_indices, seq_indices, target_ids]
            return selected_log_probs
        
        return log_probs
    
    def get_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        preference_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get value estimates from the value heads.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            preference_dim: Specific preference dimension (if None, return all)
            
        Returns:
            Value estimates [batch_size, seq_len] or [batch_size, seq_len, num_dims]
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        values = outputs["values"]  # [batch_size, seq_len, num_dims]
        
        if preference_dim is not None:
            return values[:, :, preference_dim]
        
        return values
    
    def generate_with_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        preference_dim: int = 0,
        return_values: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate tokens while tracking value estimates.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            preference_dim: Which preference dimension to track
            return_values: Whether to return value estimates
            
        Returns:
            Tuple of (generated_ids, values)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated_ids = input_ids.clone()
        all_values = [] if return_values else None
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            
            # Get next token logits and values
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            if return_values:
                next_values = outputs["values"][:, -1, preference_dim]  # [batch_size]
                all_values.append(next_values)
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # Apply the filter
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=1)
            
            # Check for EOS tokens
            if (next_token_ids == self.tokenizer.eos_token_id).all():
                break
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
        
        if return_values and all_values:
            all_values = torch.stack(all_values, dim=1)  # [batch_size, generated_len]
            return generated_ids, all_values
        
        return generated_ids, None
    
    def compute_flow(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_scores: Optional[torch.Tensor] = None,
        preference_dim: int = 0
    ) -> torch.Tensor:
        """
        Compute flow F(s_t) = Q(s_t) * V(s_t) for TFPO.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            prefix_scores: Q(s_t) scores [batch_size, seq_len]
            preference_dim: Which preference dimension to use
            
        Returns:
            Flow values [batch_size, seq_len]
        """
        # Get value estimates
        values = self.get_values(
            input_ids=input_ids,
            attention_mask=attention_mask,
            preference_dim=preference_dim
        )
        
        if prefix_scores is not None:
            # Flow = Q(s_t) * V(s_t)
            flow = prefix_scores * values
        else:
            # If no prefix scores provided, use values as flow
            flow = values
        
        return flow
    
    def save_pretrained(self, save_path: str):
        """Save the doctor model."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(os.path.join(save_path, "base_model"))
        
        # Save value heads
        torch.save({
            'value_heads_state_dict': self.value_heads.state_dict(),
            'num_preference_dims': self.num_preference_dims,
            'value_head_hidden_size': self.value_head_hidden_size,
        }, os.path.join(save_path, "value_heads.pt"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"DoctorModel saved to {save_path}")
    
    @classmethod
    def load_pretrained(
        cls,
        load_path: str,
        device: Union[str, torch.device] = "auto",
        **kwargs
    ):
        """Load a pre-trained doctor model."""
        import os
        
        # Load value head config
        value_heads_path = os.path.join(load_path, "value_heads.pt")
        if os.path.exists(value_heads_path):
            value_config = torch.load(value_heads_path, map_location='cpu')
            kwargs.update({
                'num_preference_dims': value_config['num_preference_dims'],
                'value_head_hidden_size': value_config['value_head_hidden_size'],
            })
        
        # Initialize model
        base_model_path = os.path.join(load_path, "base_model")
        if os.path.exists(base_model_path):
            model_path = base_model_path
        else:
            model_path = load_path
        
        model = cls(model_name_or_path=model_path, device=device, **kwargs)
        
        # Load value heads if available
        if os.path.exists(value_heads_path):
            value_state_dict = value_config['value_heads_state_dict']
            model.value_heads.load_state_dict(value_state_dict)
        
        logger.info(f"DoctorModel loaded from {load_path}")
        return model
    
    @property
    def device_type(self) -> str:
        """Get device type."""
        return str(self.base_model.device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            }
        return {"allocated": 0.0, "reserved": 0.0}