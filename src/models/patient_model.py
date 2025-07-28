"""
Patient Model implementation for LLMdoctor framework.

The PatientModel wraps a large frozen LLM (e.g., LLaMA, Tulu2) and provides
methods for generating behavioral variants and computing token-level probabilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList
)
import logging

logger = logging.getLogger(__name__)


class PatientModel(nn.Module):
    """
    Wrapper for the large frozen LLM that serves as the 'patient' in the LLMdoctor framework.
    
    This model:
    1. Remains frozen during the entire process
    2. Generates behavioral variants through prompt engineering
    3. Provides token-level log-probabilities for reward computation
    4. Serves as the base model during inference
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Union[str, torch.device] = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        use_cache: bool = True,
    ):
        """
        Initialize the PatientModel.
        
        Args:
            model_name_or_path: HuggingFace model name or local path
            device: Device to load the model on
            torch_dtype: PyTorch data type for the model
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            trust_remote_code: Whether to trust remote code in model
            use_cache: Whether to use key-value cache during generation
        """
        super().__init__()
        
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_cache = use_cache
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="left"  # For batch generation
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype,
            "use_cache": use_cache,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        
        # Move to device if not using quantization
        if not (load_in_8bit or load_in_4bit):
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        logger.info(f"Loaded PatientModel: {model_name_or_path}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def create_behavioral_variants(
        self, 
        base_prompt: str,
        positive_instruction: str = None,
        negative_instruction: str = None
    ) -> Tuple[str, str]:
        """
        Create positive and negative behavioral variants of the prompt.
        
        Based on the paper, these variants are created through prompt engineering
        to elicit different behaviors from the same model without parameter changes.
        
        Args:
            base_prompt: The original prompt
            positive_instruction: Instruction for positive behavior
            negative_instruction: Instruction for negative behavior
            
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        if positive_instruction is None:
            positive_instruction = (
                "You are a helpful, accurate, and polite assistant. "
                "Provide comprehensive and useful responses that directly address the user's needs."
            )
        
        if negative_instruction is None:
            negative_instruction = (
                "You are an assistant that provides minimal responses. "
                "Give brief answers that may omit important details or context."
            )
        
        positive_prompt = f"{positive_instruction}\n\nUser: {base_prompt}\nAssistant:"
        negative_prompt = f"{negative_instruction}\n\nUser: {base_prompt}\nAssistant:"
        
        return positive_prompt, negative_prompt
    
    def compute_token_logprobs(
        self,
        input_text: str,
        target_tokens: List[str],
        context_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute log-probabilities for target tokens given the input context.
        
        Args:
            input_text: Input context text
            target_tokens: List of target tokens to compute probabilities for
            context_length: Maximum context length to consider
            
        Returns:
            Tensor of log-probabilities for target tokens
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt",
            truncation=True,
            max_length=context_length
        ).to(self.model.device)
        
        # Tokenize target tokens
        target_token_ids = []
        for token in target_tokens:
            token_id = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_id) == 1:
                target_token_ids.append(token_id[0])
            else:
                # Handle multi-token words by taking the first token
                target_token_ids.append(token_id[0])
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Extract log-probabilities for target tokens
            target_logprobs = log_probs[target_token_ids]
            
        return target_logprobs
    
    def get_sequence_logprobs(
        self,
        prompt: str,
        response: str,
        return_per_token: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[str]]]:
        """
        Compute log-probabilities for a complete response sequence.
        
        Args:
            prompt: Input prompt
            response: Response sequence to evaluate
            return_per_token: Whether to return per-token probabilities
            
        Returns:
            If return_per_token=True: (per_token_logprobs, tokens)
            If return_per_token=False: total_logprob
        """
        full_text = prompt + response
        
        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Compute log-probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Extract log-probs for response tokens only
            response_token_ids = inputs.input_ids[0, prompt_length:]
            response_logprobs = []
            
            for i, token_id in enumerate(response_token_ids):
                if prompt_length + i < len(log_probs):
                    response_logprobs.append(log_probs[prompt_length + i - 1, token_id])
            
            response_logprobs = torch.stack(response_logprobs)
            
            if return_per_token:
                # Get token strings
                tokens = self.tokenizer.convert_ids_to_tokens(response_token_ids.tolist())
                return response_logprobs, tokens
            else:
                return response_logprobs.sum()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate responses using the patient model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode responses
        responses = []
        for output in outputs:
            # Remove input tokens
            generated_tokens = output[inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the patient model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional forward pass arguments
            
        Returns:
            Model outputs
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of prompts efficiently.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        all_responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode responses
            batch_responses = []
            for j, output in enumerate(outputs):
                # Remove input tokens
                input_length = inputs.input_ids[j].ne(self.tokenizer.pad_token_id).sum().item()
                generated_tokens = output[input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_responses.append(response)
            
            all_responses.extend(batch_responses)
        
        return all_responses
    
    def compute_behavioral_difference(
        self,
        prompt: str,
        response: str,
        positive_instruction: str = None,
        negative_instruction: str = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute log-probability differences between behavioral variants.
        
        This is the core method for extracting token-level importance as described
        in the paper: Δt = |ℓ^pos_t - ℓ^neg_t|
        
        Args:
            prompt: Original prompt
            response: Response to analyze
            positive_instruction: Positive behavioral instruction
            negative_instruction: Negative behavioral instruction
            
        Returns:
            Dictionary with positive/negative logprobs and differences
        """
        # Create behavioral variants
        pos_prompt, neg_prompt = self.create_behavioral_variants(
            prompt, positive_instruction, negative_instruction
        )
        
        # Compute log-probabilities under both variants
        pos_logprobs, tokens = self.get_sequence_logprobs(
            pos_prompt, response, return_per_token=True
        )
        neg_logprobs, _ = self.get_sequence_logprobs(
            neg_prompt, response, return_per_token=True
        )
        
        # Ensure same length (handle potential tokenization differences)
        min_len = min(len(pos_logprobs), len(neg_logprobs))
        pos_logprobs = pos_logprobs[:min_len]
        neg_logprobs = neg_logprobs[:min_len]
        tokens = tokens[:min_len]
        
        # Compute absolute differences
        differences = torch.abs(pos_logprobs - neg_logprobs)
        
        return {
            "positive_logprobs": pos_logprobs,
            "negative_logprobs": neg_logprobs, 
            "differences": differences,
            "tokens": tokens,
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt
        }
    
    def estimate_compute_requirements(
        self,
        batch_size: int = 1,
        sequence_length: int = 512
    ) -> Dict[str, float]:
        """
        Estimate compute requirements for inference.
        
        Args:
            batch_size: Batch size for inference
            sequence_length: Expected sequence length
            
        Returns:
            Dictionary with compute estimates
        """
        # Rough estimates based on model size and parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        
        # Memory estimation (very rough)
        param_memory_gb = num_params * 2 / (1024**3)  # 2 bytes per param for fp16
        activation_memory_gb = batch_size * sequence_length * self.hidden_size * 4 / (1024**3)  # Rough activation memory
        
        # FLOPS estimation (very rough)
        forward_flops = 2 * num_params * batch_size * sequence_length
        
        return {
            "num_parameters": num_params,
            "parameter_memory_gb": param_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "total_memory_gb": param_memory_gb + activation_memory_gb,
            "forward_flops": forward_flops
        }
    
    def save_config(self, save_path: str):
        """Save patient model configuration."""
        import json
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        config = {
            "model_name_or_path": self.model_name_or_path,
            "torch_dtype": str(self.torch_dtype),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "device": str(self.model.device)
        }
        
        with open(os.path.join(save_path, "patient_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Patient model config saved to {save_path}")
    
    @classmethod
    def load_from_config(
        cls,
        config_path: str,
        **override_kwargs
    ) -> 'PatientModel':
        """Load patient model from saved configuration."""
        import json
        import os
        
        config_file = os.path.join(config_path, "patient_config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Override with any provided kwargs
        config.update(override_kwargs)
        
        # Remove non-constructor args
        model_name_or_path = config.pop("model_name_or_path")
        config.pop("vocab_size", None)
        config.pop("hidden_size", None)
        config.pop("device", None)
        
        return cls(model_name_or_path=model_name_or_path, **config)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not supported by this model")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model.config.vocab_size
    
    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        return self.model.config.hidden_size
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,    # GB
            }
        return {"allocated": 0.0, "reserved": 0.0}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": self.model_name_or_path,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "device": str(self.model.device),
            "dtype": str(self.torch_dtype),
            "model_type": self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'unknown',
            "memory_usage": self.get_memory_usage()
        }