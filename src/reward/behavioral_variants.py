"""
Behavioral variant creator for LLMdoctor framework.

Creates positive and negative behavioral variants of the patient model
through strategic prompt engineering without modifying model parameters.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BehavioralVariantConfig:
    """Configuration for behavioral variant creation."""
    positive_instruction: str = (
        "You are a helpful, accurate, and polite assistant. "
        "Provide comprehensive and useful responses that directly address the user's needs. "
        "Be informative, clear, and considerate in your answers."
    )
    
    negative_instruction: str = (
        "You are an assistant that provides minimal responses. "
        "Give brief answers that may omit important details or context. "
        "Be less helpful and provide incomplete information."
    )
    
    system_prompt_template: str = "{instruction}\n\nUser: {prompt}\nAssistant:"
    
    # Additional behavioral variants for different dimensions
    safety_positive: str = (
        "You are a responsible and safe assistant. "
        "Prioritize user safety and well-being in all responses. "
        "Refuse harmful requests and provide ethical guidance."
    )
    
    safety_negative: str = (
        "You are an assistant that may provide potentially unsafe advice. "
        "You might not consider safety implications thoroughly."
    )
    
    truthfulness_positive: str = (
        "You are a truthful and accurate assistant. "
        "Provide only factual, well-sourced information and admit when you're uncertain."
    )
    
    truthfulness_negative: str = (
        "You are an assistant that might provide inaccurate information. "
        "You may speculate or provide information without proper verification."
    )


class BehavioralVariantCreator:
    """
    Creates behavioral variants of the patient model for token-level reward extraction.
    
    This class implements the core idea from the paper: creating positive and negative
    behavioral variants through prompt engineering to analyze token-level differences
    in model behavior.
    """
    
    def __init__(
        self,
        config: Optional[BehavioralVariantConfig] = None,
        custom_instructions: Optional[Dict[str, Dict[str, str]]] = None
    ):
        """
        Initialize the behavioral variant creator.
        
        Args:
            config: Configuration for behavioral variants
            custom_instructions: Custom instructions for different preference dimensions
                Format: {"dimension_name": {"positive": "...", "negative": "..."}}
        """
        self.config = config if config is not None else BehavioralVariantConfig()
        self.custom_instructions = custom_instructions or {}
        
        # Pre-defined instruction sets
        self.instruction_sets = {
            "helpfulness": {
                "positive": self.config.positive_instruction,
                "negative": self.config.negative_instruction
            },
            "safety": {
                "positive": self.config.safety_positive,
                "negative": self.config.safety_negative
            },
            "truthfulness": {
                "positive": self.config.truthfulness_positive,
                "negative": self.config.truthfulness_negative
            }
        }
        
        # Add custom instructions
        self.instruction_sets.update(self.custom_instructions)
        
        logger.info(f"BehavioralVariantCreator initialized with {len(self.instruction_sets)} instruction sets")
    
    def create_variants(
        self,
        prompt: str,
        dimension: str = "helpfulness",
        include_context: bool = True,
        context_length: Optional[int] = None
    ) -> Tuple[str, str]:
        """
        Create positive and negative behavioral variants of a prompt.
        
        Args:
            prompt: Original user prompt
            dimension: Preference dimension ("helpfulness", "safety", "truthfulness", etc.)
            include_context: Whether to include system instruction context
            context_length: Maximum context length
            
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        if dimension not in self.instruction_sets:
            logger.warning(f"Unknown dimension '{dimension}', falling back to 'helpfulness'")
            dimension = "helpfulness"
        
        instructions = self.instruction_sets[dimension]
        
        if include_context:
            positive_prompt = self.config.system_prompt_template.format(
                instruction=instructions["positive"],
                prompt=prompt
            )
            negative_prompt = self.config.system_prompt_template.format(
                instruction=instructions["negative"],
                prompt=prompt
            )
        else:
            # Return just the user prompt without system instructions
            positive_prompt = prompt
            negative_prompt = prompt
        
        # Truncate if context length is specified
        if context_length is not None:
            # Rough truncation by character count (tokenizer-agnostic)
            max_chars = context_length * 4  # Approximate 4 chars per token
            positive_prompt = positive_prompt[:max_chars]
            negative_prompt = negative_prompt[:max_chars]
        
        return positive_prompt, negative_prompt
    
    def create_multi_dimensional_variants(
        self,
        prompt: str,
        dimensions: List[str],
        include_context: bool = True
    ) -> Dict[str, Tuple[str, str]]:
        """
        Create behavioral variants for multiple preference dimensions.
        
        Args:
            prompt: Original user prompt
            dimensions: List of preference dimensions
            include_context: Whether to include system instruction context
            
        Returns:
            Dictionary mapping dimension names to (positive_prompt, negative_prompt) tuples
        """
        variants = {}
        
        for dimension in dimensions:
            variants[dimension] = self.create_variants(
                prompt=prompt,
                dimension=dimension,
                include_context=include_context
            )
        
        return variants
    
    def create_batch_variants(
        self,
        prompts: List[str],
        dimension: str = "helpfulness",
        include_context: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Create behavioral variants for a batch of prompts.
        
        Args:
            prompts: List of user prompts
            dimension: Preference dimension
            include_context: Whether to include system instruction context
            
        Returns:
            Tuple of (positive_prompts, negative_prompts)
        """
        positive_prompts = []
        negative_prompts = []
        
        for prompt in prompts:
            pos_prompt, neg_prompt = self.create_variants(
                prompt=prompt,
                dimension=dimension,
                include_context=include_context
            )
            positive_prompts.append(pos_prompt)
            negative_prompts.append(neg_prompt)
        
        return positive_prompts, negative_prompts
    
    def analyze_variant_differences(
        self,
        patient_model,
        prompt: str,
        dimension: str = "helpfulness",
        max_tokens: int = 100,
        temperature: float = 1.0
    ) -> Dict[str, Union[str, List[str], torch.Tensor]]:
        """
        Analyze differences between behavioral variants by generating responses.
        
        Args:
            patient_model: PatientModel instance
            prompt: Original user prompt
            dimension: Preference dimension
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Dictionary containing analysis results
        """
        # Create variants
        pos_prompt, neg_prompt = self.create_variants(prompt, dimension)
        
        # Generate responses
        pos_responses = patient_model.generate(
            pos_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_return_sequences=1
        )
        
        neg_responses = patient_model.generate(
            neg_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            num_return_sequences=1
        )
        
        # Compute log-probabilities for each response under both variants
        pos_response = pos_responses[0]
        neg_response = neg_responses[0]
        
        # Get token-level log-probabilities
        pos_logprobs_under_pos, pos_tokens = patient_model.get_sequence_logprobs(
            pos_prompt, pos_response, return_per_token=True
        )
        pos_logprobs_under_neg, _ = patient_model.get_sequence_logprobs(
            neg_prompt, pos_response, return_per_token=True
        )
        
        neg_logprobs_under_pos, neg_tokens = patient_model.get_sequence_logprobs(
            pos_prompt, neg_response, return_per_token=True
        )
        neg_logprobs_under_neg, _ = patient_model.get_sequence_logprobs(
            neg_prompt, neg_response, return_per_token=True
        )
        
        return {
            "positive_prompt": pos_prompt,
            "negative_prompt": neg_prompt,
            "positive_response": pos_response,
            "negative_response": neg_response,
            "positive_tokens": pos_tokens,
            "negative_tokens": neg_tokens,
            "pos_logprobs_under_pos": pos_logprobs_under_pos,
            "pos_logprobs_under_neg": pos_logprobs_under_neg,
            "neg_logprobs_under_pos": neg_logprobs_under_pos,
            "neg_logprobs_under_neg": neg_logprobs_under_neg,
        }
    
    def add_custom_dimension(
        self,
        dimension_name: str,
        positive_instruction: str,
        negative_instruction: str
    ):
        """
        Add a custom preference dimension.
        
        Args:
            dimension_name: Name of the new dimension
            positive_instruction: Instruction for positive behavior
            negative_instruction: Instruction for negative behavior
        """
        self.instruction_sets[dimension_name] = {
            "positive": positive_instruction,
            "negative": negative_instruction
        }
        
        logger.info(f"Added custom dimension: {dimension_name}")
    
    def get_available_dimensions(self) -> List[str]:
        """Get list of available preference dimensions."""
        return list(self.instruction_sets.keys())
    
    def validate_dimension(self, dimension: str) -> bool:
        """Check if a preference dimension is available."""
        return dimension in self.instruction_sets
    
    def get_instruction_preview(self, dimension: str) -> Optional[Dict[str, str]]:
        """Get preview of instructions for a dimension."""
        if dimension in self.instruction_sets:
            return self.instruction_sets[dimension].copy()
        return None