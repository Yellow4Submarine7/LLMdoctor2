"""
Reward-guided decoder for LLMdoctor framework.

Implements the reward-guided decoding algorithm that combines patient model probabilities
with doctor model guidance to generate aligned responses.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass

from ..models.patient_model import PatientModel
from .flow_guided_reward import FlowGuidedRewardModel

logger = logging.getLogger(__name__)


@dataclass
class DecodingConfig:
    """Configuration for reward-guided decoding."""
    # Guidance parameters
    alpha: float = 1.0        # Weight for patient model probabilities
    beta: float = 1.0         # Weight for reward model guidance
    temperature: float = 1.0  # Sampling temperature
    
    # Sampling parameters
    top_k: Optional[int] = None        # Top-k sampling
    top_p: Optional[float] = 0.9       # Nucleus sampling
    do_sample: bool = True             # Whether to use sampling
    repetition_penalty: float = 1.0    # Repetition penalty
    
    # Generation parameters
    max_new_tokens: int = 512          # Maximum tokens to generate
    min_length: int = 1                # Minimum generation length
    early_stopping: bool = True        # Whether to stop at EOS
    
    # Advanced parameters
    use_cache: bool = True             # Whether to use key-value cache
    pad_token_id: Optional[int] = None # Padding token ID
    eos_token_id: Optional[int] = None # End-of-sequence token ID
    
    # Guidance scheduling
    guidance_schedule: str = "constant"  # "constant", "linear_decay", "cosine_decay"
    min_beta: float = 0.1               # Minimum beta value for scheduling


class RewardGuidedDecoder:
    """
    Reward-guided decoder that implements the core decoding algorithm from the paper:
    
    π_decode(y_{t+1}|s_t) ∝ [π_base(y_{t+1}|s_t)]^α · [π_r(y_{t+1}|s_t)]^β
    
    Where:
    - π_base is the patient model
    - π_r is the flow-guided reward model (doctor model)
    - α and β control the balance between fluency and alignment
    """
    
    def __init__(
        self,
        patient_model: PatientModel,
        reward_model: FlowGuidedRewardModel,
        config: Optional[DecodingConfig] = None,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize reward-guided decoder.
        
        Args:
            patient_model: Patient model instance
            reward_model: Flow-guided reward model instance
            config: Decoding configuration
            device: Device for computations
        """
        self.patient_model = patient_model
        self.reward_model = reward_model
        self.config = config if config is not None else DecodingConfig()
        
        if device == "auto":
            self.device = next(patient_model.parameters()).device
        else:
            self.device = torch.device(device)
        
        # Set models to evaluation mode
        self.patient_model.eval()
        self.reward_model.doctor_model.eval()
        
        logger.info(f"RewardGuidedDecoder initialized on {self.device}")
        logger.info(f"Alpha: {self.config.alpha}, Beta: {self.config.beta}")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[DecodingConfig] = None,
        return_dict_in_generate: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate text using reward-guided decoding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            generation_config: Optional generation config override
            return_dict_in_generate: Whether to return detailed generation info
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs or detailed generation dictionary
        """
        config = generation_config if generation_config is not None else self.config
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        
        # Setup attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get special token IDs
        pad_token_id = config.pad_token_id or self.patient_model.tokenizer.pad_token_id
        eos_token_id = config.eos_token_id or self.patient_model.tokenizer.eos_token_id
        
        # Generation loop
        generation_info = {
            "patient_logprobs": [],
            "reward_logprobs": [],
            "combined_logprobs": [],
            "selected_tokens": [],
            "guidance_weights": []
        }
        
        for step in range(config.max_new_tokens):
            # Check if all sequences have generated EOS
            if config.early_stopping and eos_token_id is not None:
                if (generated_ids[:, -1] == eos_token_id).all():
                    break
            
            # Compute guidance weight (with optional scheduling)
            beta = self._compute_guidance_weight(step, config)
            
            # Generate next token
            next_token_ids, step_info = self._generate_next_token(
                generated_ids,
                attention_mask,
                config,
                beta
            )
            
            # Update sequences
            generated_ids = torch.cat([generated_ids, next_token_ids.unsqueeze(1)], dim=1)
            
            # Update attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=1)
            
            # Store generation info
            if return_dict_in_generate:
                generation_info["patient_logprobs"].append(step_info["patient_logprobs"])
                generation_info["reward_logprobs"].append(step_info["reward_logprobs"])
                generation_info["combined_logprobs"].append(step_info["combined_logprobs"])
                generation_info["selected_tokens"].append(next_token_ids)
                generation_info["guidance_weights"].append(beta)
        
        if return_dict_in_generate:
            # Stack tensors
            for key in ["patient_logprobs", "reward_logprobs", "combined_logprobs", "selected_tokens"]:
                if generation_info[key]:
                    generation_info[key] = torch.stack(generation_info[key], dim=1)
            
            generation_info["sequences"] = generated_ids
            generation_info["guidance_weights"] = torch.tensor(generation_info["guidance_weights"])
            
            return generation_info
        
        return generated_ids
    
    def _generate_next_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: DecodingConfig,
        beta: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Generate the next token using reward-guided decoding."""
        
        with torch.no_grad():
            # Get patient model probabilities
            patient_outputs = self.patient_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            patient_logits = patient_outputs.logits[:, -1, :] / config.temperature
            patient_logprobs = F.log_softmax(patient_logits, dim=-1)
            
            # Get reward model probabilities
            reward_results = self.reward_model.compute_token_rewards(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_logprobs=True
            )
            reward_logprobs = reward_results["reward_logprobs"]
            
            # Combine probabilities: π_decode ∝ [π_base]^α · [π_r]^β
            combined_logprobs = (
                config.alpha * patient_logprobs + 
                beta * reward_logprobs
            )
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                combined_logprobs = self._apply_repetition_penalty(
                    combined_logprobs, input_ids, config.repetition_penalty
                )
            
            # Apply sampling constraints
            filtered_logprobs = self._apply_sampling_constraints(
                combined_logprobs, config
            )
            
            # Sample next token
            if config.do_sample:
                probs = F.softmax(filtered_logprobs, dim=-1)
                next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token_ids = torch.argmax(filtered_logprobs, dim=-1)
            
            step_info = {
                "patient_logprobs": patient_logprobs,
                "reward_logprobs": reward_logprobs,
                "combined_logprobs": combined_logprobs
            }
            
            return next_token_ids, step_info
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        repetition_penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if repetition_penalty == 1.0:
            return logits
        
        batch_size, vocab_size = logits.shape
        
        # Get unique tokens in input
        for i in range(batch_size):
            for token_id in input_ids[i].unique():
                if token_id >= 0 and token_id < vocab_size:  # Valid token
                    if logits[i, token_id] < 0:
                        logits[i, token_id] *= repetition_penalty
                    else:
                        logits[i, token_id] /= repetition_penalty
        
        return logits
    
    def _apply_sampling_constraints(
        self,
        logits: torch.Tensor,
        config: DecodingConfig
    ) -> torch.Tensor:
        """Apply top-k and top-p sampling constraints."""
        
        # Apply top-k filtering
        if config.top_k is not None and config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            # Remove tokens with rank higher than top_k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if config.top_p is not None and config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted indices back to unsorted indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _compute_guidance_weight(
        self,
        step: int,
        config: DecodingConfig
    ) -> float:
        """Compute guidance weight with optional scheduling."""
        
        if config.guidance_schedule == "constant":
            return config.beta
        
        elif config.guidance_schedule == "linear_decay":
            # Linear decay from beta to min_beta
            decay_rate = (config.beta - config.min_beta) / config.max_new_tokens
            return max(config.min_beta, config.beta - step * decay_rate)
        
        elif config.guidance_schedule == "cosine_decay":
            # Cosine decay from beta to min_beta
            progress = step / config.max_new_tokens
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            return config.min_beta + (config.beta - config.min_beta) * cosine_factor
        
        else:
            logger.warning(f"Unknown guidance schedule: {config.guidance_schedule}")
            return config.beta
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated responses
        """
        # Tokenize prompts
        inputs = self.patient_model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **kwargs
            )
        
        # Decode responses
        responses = []
        for i, output in enumerate(outputs):
            # Remove input tokens
            generated_tokens = output[inputs.input_ids.shape[1]:]
            response = self.patient_model.tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            responses.append(response)
        
        return responses
    
    def interactive_generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        show_probabilities: bool = False,
        **kwargs
    ) -> Dict[str, Union[str, List]]:
        """
        Interactive generation with step-by-step information.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            show_probabilities: Whether to show token probabilities
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation details
        """
        # Tokenize prompt
        inputs = self.patient_model.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with detailed info
        result = self.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            **kwargs
        )
        
        # Decode generated sequence
        generated_text = self.patient_model.tokenizer.decode(
            result["sequences"][0], skip_special_tokens=True
        )
        
        response_text = generated_text[len(prompt):].strip()
        
        # Prepare detailed info
        generation_details = {
            "prompt": prompt,
            "response": response_text,
            "full_text": generated_text,
            "num_generated_tokens": len(result["selected_tokens"][0]),
            "guidance_weights": result["guidance_weights"].tolist()
        }
        
        if show_probabilities:
            # Add token-by-token probability information
            token_details = []
            
            for i in range(len(result["selected_tokens"][0])):
                token_id = result["selected_tokens"][0, i].item()
                token_text = self.patient_model.tokenizer.decode([token_id])
                
                patient_prob = torch.exp(result["patient_logprobs"][0, i, token_id]).item()
                reward_prob = torch.exp(result["reward_logprobs"][0, i, token_id]).item()
                combined_prob = torch.exp(result["combined_logprobs"][0, i, token_id]).item()
                
                token_details.append({
                    "token": token_text,
                    "token_id": token_id,
                    "patient_prob": patient_prob,
                    "reward_prob": reward_prob,
                    "combined_prob": combined_prob,
                    "guidance_weight": result["guidance_weights"][i].item()
                })
            
            generation_details["token_details"] = token_details
        
        return generation_details
    
    def compare_with_baseline(
        self,
        prompt: str,
        num_samples: int = 5,
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Compare reward-guided generation with baseline patient model.
        
        Args:
            prompt: Input prompt
            num_samples: Number of samples to generate
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with guided and baseline generations
        """
        # Generate with reward guidance
        guided_responses = []
        for _ in range(num_samples):
            response = self.batch_generate([prompt], **kwargs)[0]
            guided_responses.append(response)
        
        # Generate baseline responses (no guidance)
        baseline_responses = []
        original_beta = self.config.beta
        self.config.beta = 0.0  # Disable guidance
        
        try:
            for _ in range(num_samples):
                response = self.batch_generate([prompt], **kwargs)[0]
                baseline_responses.append(response)
        finally:
            self.config.beta = original_beta  # Restore guidance
        
        return {
            "guided_responses": guided_responses,
            "baseline_responses": baseline_responses,
            "prompt": prompt
        }
    
    def update_guidance_weights(self, alpha: float, beta: float):
        """Update the guidance weights."""
        self.config.alpha = alpha
        self.config.beta = beta
        logger.info(f"Updated guidance weights: alpha={alpha}, beta={beta}")
    
    def get_decoding_info(self) -> Dict[str, Union[float, int, str]]:
        """Get information about the decoder configuration."""
        return {
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens,
            "guidance_schedule": self.config.guidance_schedule,
            "patient_model": self.patient_model.model_name_or_path,
            "reward_model_preference_dim": self.reward_model.preference_dim,
            "device": str(self.device)
        }