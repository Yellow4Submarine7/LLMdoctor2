"""
Guided generation module for LLMDoctor framework.

Provides high-level interface for reward-guided text generation
using trained doctor models to guide patient models.
"""

import torch
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..models.patient_model import PatientModel
from ..models.doctor_model import DoctorModel
from .flow_guided_reward import FlowGuidedRewardModel
from .reward_guided_decoder import RewardGuidedDecoder, DecodingConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference with LLMDoctor."""
    # Guidance parameters
    alpha: float = 1.0  # Weight for patient model
    beta: float = 1.0   # Weight for doctor model guidance
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: Optional[int] = None
    do_sample: bool = True
    repetition_penalty: float = 1.0
    
    # Multi-dimensional preferences
    preference_weights: Optional[Dict[str, float]] = None
    aggregation_method: str = "weighted_sum"  # weighted_sum, max, product
    
    # Advanced parameters
    guidance_schedule: str = "constant"
    min_beta: float = 0.1
    use_value_head: bool = True
    normalize_rewards: bool = True
    
    # Efficiency
    batch_size: int = 1
    use_cache: bool = True
    
    # Output settings
    return_detailed_info: bool = False
    save_generations: bool = False
    output_file: Optional[str] = None


class GuidedGenerator:
    """
    Main interface for guided text generation with LLMDoctor.
    
    Combines patient and doctor models to generate text that is both
    fluent (from patient model) and aligned with preferences (from doctor model).
    """
    
    def __init__(
        self,
        patient_model: PatientModel,
        doctor_model: DoctorModel,
        config: Optional[InferenceConfig] = None,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize guided generator.
        
        Args:
            patient_model: Patient model instance
            doctor_model: Trained doctor model instance
            config: Inference configuration
            device: Device for computation
        """
        self.patient_model = patient_model
        self.doctor_model = doctor_model
        self.config = config if config is not None else InferenceConfig()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Move models to device
        self.patient_model = self.patient_model.to(self.device)
        self.doctor_model = self.doctor_model.to(self.device)
        
        # Create reward models for each preference dimension
        self.reward_models = {}
        for dim in range(doctor_model.num_preference_dims):
            self.reward_models[dim] = FlowGuidedRewardModel(
                doctor_model=doctor_model,
                preference_dim=dim,
                temperature=config.temperature,
                use_value_head=config.use_value_head,
                normalize_rewards=config.normalize_rewards,
                device=self.device
            )
        
        # Create decoder
        decoder_config = DecodingConfig(
            alpha=config.alpha,
            beta=config.beta,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            max_new_tokens=config.max_new_tokens,
            guidance_schedule=config.guidance_schedule,
            min_beta=config.min_beta,
            use_cache=config.use_cache
        )
        
        # Use first dimension as default
        self.decoder = RewardGuidedDecoder(
            patient_model=patient_model,
            reward_model=self.reward_models[0],
            config=decoder_config,
            device=self.device
        )
        
        logger.info(f"GuidedGenerator initialized on {self.device}")
        logger.info(f"Number of preference dimensions: {doctor_model.num_preference_dims}")
    
    def generate(
        self,
        prompt: str,
        preference_dim: int = 0,
        preference_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Union[str, Dict[str, any]]:
        """
        Generate text with reward guidance.
        
        Args:
            prompt: Input prompt
            preference_dim: Which preference dimension to use (for single-dim)
            preference_weights: Weights for multi-dimensional preferences
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or detailed generation info
        """
        # Handle multi-dimensional preferences
        if preference_weights is not None and len(preference_weights) > 1:
            return self._generate_multi_dimensional(
                prompt=prompt,
                preference_weights=preference_weights,
                **kwargs
            )
        
        # Single-dimensional generation
        if preference_dim != self.decoder.reward_model.preference_dim:
            self.decoder.reward_model = self.reward_models[preference_dim]
        
        # Update config with kwargs
        generation_config = DecodingConfig(
            alpha=kwargs.get('alpha', self.config.alpha),
            beta=kwargs.get('beta', self.config.beta),
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            top_k=kwargs.get('top_k', self.config.top_k),
            do_sample=kwargs.get('do_sample', self.config.do_sample),
            max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens),
            repetition_penalty=kwargs.get('repetition_penalty', self.config.repetition_penalty)
        )
        
        # Generate
        if self.config.return_detailed_info or kwargs.get('return_detailed_info', False):
            result = self.decoder.interactive_generate(
                prompt=prompt,
                max_new_tokens=generation_config.max_new_tokens,
                show_probabilities=True,
                **kwargs
            )
            return result
        else:
            responses = self.decoder.batch_generate(
                prompts=[prompt],
                generation_config=generation_config
            )
            return responses[0]
    
    def _generate_multi_dimensional(
        self,
        prompt: str,
        preference_weights: Dict[str, float],
        **kwargs
    ) -> Union[str, Dict[str, any]]:
        """Generate with multi-dimensional preference guidance."""
        # Map dimension names to indices
        dim_mapping = {
            "helpfulness": 0,
            "safety": 1,
            "truthfulness": 2,
            # Add more mappings as needed
        }
        
        # Tokenize prompt
        inputs = self.patient_model.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Generate with aggregated guidance
        generated_ids = input_ids.clone()
        
        for step in range(self.config.max_new_tokens):
            # Get patient model probabilities
            with torch.no_grad():
                patient_outputs = self.patient_model.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask
                )
                patient_logits = patient_outputs.logits[:, -1, :] / self.config.temperature
                patient_logprobs = torch.nn.functional.log_softmax(patient_logits, dim=-1)
            
            # Get reward probabilities for each dimension
            dimension_rewards = {}
            for dim_name, weight in preference_weights.items():
                if dim_name in dim_mapping:
                    dim_idx = dim_mapping[dim_name]
                    reward_model = self.reward_models[dim_idx]
                    
                    reward_results = reward_model.compute_token_rewards(
                        input_ids=generated_ids,
                        attention_mask=attention_mask,
                        return_logprobs=True
                    )
                    dimension_rewards[dim_name] = (
                        reward_results["reward_logprobs"],
                        weight
                    )
            
            # Aggregate rewards across dimensions
            if self.config.aggregation_method == "weighted_sum":
                total_weight = sum(w for _, w in dimension_rewards.values())
                aggregated_reward_logprobs = None
                
                for dim_name, (reward_logprobs, weight) in dimension_rewards.items():
                    weighted_logprobs = reward_logprobs * (weight / total_weight)
                    if aggregated_reward_logprobs is None:
                        aggregated_reward_logprobs = weighted_logprobs
                    else:
                        aggregated_reward_logprobs += weighted_logprobs
            
            elif self.config.aggregation_method == "max":
                # Take maximum reward across dimensions
                all_rewards = [r for r, _ in dimension_rewards.values()]
                aggregated_reward_logprobs = torch.stack(all_rewards).max(dim=0)[0]
            
            elif self.config.aggregation_method == "product":
                # Product of probabilities (sum of log-probs)
                aggregated_reward_logprobs = sum(
                    r * w for r, w in dimension_rewards.values()
                )
            
            # Combine with patient model
            combined_logprobs = (
                self.config.alpha * patient_logprobs +
                self.config.beta * aggregated_reward_logprobs
            )
            
            # Sample next token
            if self.config.do_sample:
                probs = torch.nn.functional.softmax(combined_logprobs, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(combined_logprobs, dim=-1, keepdim=True)
            
            # Update sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)
            
            # Check for EOS
            if self.patient_model.tokenizer.eos_token_id is not None:
                if (next_token == self.patient_model.tokenizer.eos_token_id).any():
                    break
        
        # Decode response
        generated_text = self.patient_model.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        response = generated_text[len(prompt):].strip()
        
        if self.config.return_detailed_info:
            return {
                "prompt": prompt,
                "response": response,
                "preference_weights": preference_weights,
                "aggregation_method": self.config.aggregation_method
            }
        
        return response
    
    def batch_generate(
        self,
        prompts: List[str],
        preference_dim: int = 0,
        preference_weights: Optional[Dict[str, float]] = None,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[Union[str, Dict]]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            preference_dim: Preference dimension to use
            preference_weights: Multi-dimensional preference weights
            batch_size: Batch size for generation
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Generate for batch
            batch_responses = []
            for prompt in batch_prompts:
                response = self.generate(
                    prompt=prompt,
                    preference_dim=preference_dim,
                    preference_weights=preference_weights,
                    **kwargs
                )
                batch_responses.append(response)
            
            responses.extend(batch_responses)
        
        # Save if requested
        if self.config.save_generations and self.config.output_file:
            self._save_generations(prompts, responses)
        
        return responses
    
    def guided_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        preference_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """
        Simplified interface for guided generation compatible with evaluation.
        
        This method is designed to be compatible with the evaluation framework.
        """
        # Use provided parameters or defaults
        generation_kwargs = {
            'max_new_tokens': max_new_tokens or self.config.max_new_tokens,
            'temperature': temperature or self.config.temperature,
            'top_p': top_p or self.config.top_p,
            'top_k': top_k or self.config.top_k,
            'do_sample': do_sample if do_sample is not None else self.config.do_sample
        }
        
        if preference_weights:
            generation_kwargs['preference_weights'] = preference_weights
        
        return self.generate(prompt, **generation_kwargs)
    
    def compare_dimensions(
        self,
        prompt: str,
        dimension_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """
        Generate responses using different preference dimensions.
        
        Args:
            prompt: Input prompt
            dimension_names: Names of dimensions to compare
            **kwargs: Generation parameters
            
        Returns:
            Dictionary mapping dimension names to responses
        """
        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(self.doctor_model.num_preference_dims)]
        
        results = {}
        
        for i, dim_name in enumerate(dimension_names[:self.doctor_model.num_preference_dims]):
            response = self.generate(
                prompt=prompt,
                preference_dim=i,
                **kwargs
            )
            results[dim_name] = response
        
        return results
    
    def interactive_chat(self):
        """Run an interactive chat session."""
        print("\n" + "="*60)
        print("LLMDoctor Interactive Chat")
        print("="*60)
        print(f"Patient Model: {self.patient_model.model_name_or_path}")
        print(f"Doctor Dimensions: {self.doctor_model.num_preference_dims}")
        print(f"Guidance: α={self.config.alpha}, β={self.config.beta}")
        print("\nCommands:")
        print("  /dim <n>    - Switch to preference dimension n")
        print("  /weights    - Set multi-dimensional weights")
        print("  /params     - Show/update generation parameters")
        print("  /compare    - Compare responses across dimensions")
        print("  /quit       - Exit chat")
        print("="*60 + "\n")
        
        current_dim = 0
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    if user_input == "/quit":
                        print("Goodbye!")
                        break
                    
                    elif user_input.startswith("/dim"):
                        try:
                            new_dim = int(user_input.split()[1])
                            if 0 <= new_dim < self.doctor_model.num_preference_dims:
                                current_dim = new_dim
                                print(f"Switched to dimension {new_dim}")
                            else:
                                print(f"Invalid dimension. Must be 0-{self.doctor_model.num_preference_dims-1}")
                        except:
                            print("Usage: /dim <number>")
                    
                    elif user_input == "/params":
                        print(f"\nCurrent parameters:")
                        print(f"  Alpha: {self.config.alpha}")
                        print(f"  Beta: {self.config.beta}")
                        print(f"  Temperature: {self.config.temperature}")
                        print(f"  Top-p: {self.config.top_p}")
                        print(f"  Max tokens: {self.config.max_new_tokens}")
                    
                    elif user_input.startswith("/compare"):
                        prompt = input("Enter prompt to compare: ")
                        print("\nComparing across dimensions...")
                        results = self.compare_dimensions(prompt)
                        for dim_name, response in results.items():
                            print(f"\n{dim_name}: {response}")
                    
                    else:
                        print("Unknown command. Type /quit to exit.")
                    
                    continue
                
                # Generate response
                print("\nAssistant: ", end="", flush=True)
                response = self.generate(
                    prompt=user_input,
                    preference_dim=current_dim
                )
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\nError: {e}")
    
    def _save_generations(self, prompts: List[str], responses: List[Union[str, Dict]]):
        """Save generation results to file."""
        output_path = Path(self.config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = []
        for prompt, response in zip(prompts, responses):
            if isinstance(response, dict):
                entry = response
                entry["prompt"] = prompt
            else:
                entry = {
                    "prompt": prompt,
                    "response": response
                }
            data.append(entry)
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data)} generations to {output_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        doctor_model_path: str,
        patient_model_name: str,
        config: Optional[InferenceConfig] = None,
        device: Union[str, torch.device] = "auto"
    ) -> "GuidedGenerator":
        """
        Load guided generator from pretrained models.
        
        Args:
            doctor_model_path: Path to trained doctor model
            patient_model_name: Name or path of patient model
            config: Inference configuration
            device: Device for computation
            
        Returns:
            GuidedGenerator instance
        """
        # Load patient model
        patient_model = PatientModel(
            model_name=patient_model_name,
            device=device
        )
        
        # Load doctor model
        doctor_model = DoctorModel.from_pretrained(doctor_model_path)
        
        return cls(
            patient_model=patient_model,
            doctor_model=doctor_model,
            config=config,
            device=device
        )