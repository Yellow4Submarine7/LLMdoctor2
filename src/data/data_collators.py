"""
Data collators for LLMdoctor framework.

Handles batching and collation of tokenized examples for efficient training
with proper padding and attention mask handling.
"""

import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorConfig:
    """Configuration for data collators."""
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_length: Optional[int] = None
    padding_side: str = "right"
    include_metadata: bool = False


class TFPODataCollator:
    """
    Data collator for TFPO training.
    
    Handles batching of tokenized examples with proper padding for:
    - Input IDs and attention masks
    - Token-level rewards
    - Labels (if present)
    - Metadata (if requested)
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt",
        label_pad_token_id: int = -100,
        reward_pad_value: float = 0.0,
        include_metadata: bool = False
    ):
        """
        Initialize the TFPO data collator.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for padding
            padding: Padding strategy
            pad_to_multiple_of: Pad to multiple of this value
            return_tensors: Type of tensors to return
            label_pad_token_id: Token ID to use for padding labels
            reward_pad_value: Value to use for padding rewards
            include_metadata: Whether to include metadata in batches
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.label_pad_token_id = label_pad_token_id
        self.reward_pad_value = reward_pad_value
        self.include_metadata = include_metadata
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        # Separate the different types of data
        input_ids = [example["input_ids"] for example in examples]
        attention_masks = [example["attention_mask"] for example in examples]
        token_rewards = [example["token_rewards"] for example in examples]
        
        # Handle labels if present
        labels = None
        if "labels" in examples[0] and examples[0]["labels"] is not None:
            labels = [example["labels"] for example in examples]
        
        # Determine batch max length
        batch_max_length = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            batch_max_length = min(batch_max_length, self.max_length)
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            batch_max_length = (
                (batch_max_length + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        # Pad all sequences
        padded_input_ids = self._pad_sequences(
            input_ids, batch_max_length, self.tokenizer.pad_token_id
        )
        padded_attention_masks = self._pad_sequences(
            attention_masks, batch_max_length, 0
        )
        padded_token_rewards = self._pad_sequences(
            token_rewards, batch_max_length, self.reward_pad_value
        )
        
        # Create batch dictionary
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "token_rewards": torch.stack(padded_token_rewards)
        }
        
        # Add labels if present
        if labels is not None:
            padded_labels = self._pad_sequences(
                labels, batch_max_length, self.label_pad_token_id
            )
            batch["labels"] = torch.stack(padded_labels)
        
        # Add metadata if requested
        if self.include_metadata:
            metadata = self._collate_metadata(examples)
            batch.update(metadata)
        
        return batch
    
    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],
        target_length: int,
        pad_value: Union[int, float]
    ) -> List[torch.Tensor]:
        """Pad sequences to target length."""
        padded_sequences = []
        
        for seq in sequences:
            current_length = len(seq)
            
            if current_length >= target_length:
                # Truncate if too long
                padded_seq = seq[:target_length]
            else:
                # Pad if too short
                pad_length = target_length - current_length
                if isinstance(pad_value, int):
                    padding = torch.full((pad_length,), pad_value, dtype=seq.dtype)
                else:
                    padding = torch.full((pad_length,), pad_value, dtype=torch.float)
                
                padded_seq = torch.cat([seq, padding])
            
            padded_sequences.append(padded_seq)
        
        return padded_sequences
    
    def _collate_metadata(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate metadata from examples."""
        metadata = {}
        
        # Get all metadata keys from first example
        if examples and "metadata" in examples[0]:
            first_metadata = examples[0]["metadata"]
            if first_metadata is not None:
                for key in first_metadata.keys():
                    values = []
                    for example in examples:
                        if "metadata" in example and example["metadata"] is not None:
                            value = example["metadata"].get(key, 0)
                            values.append(value)
                        else:
                            values.append(0)
                    
                    # Convert to tensor if possible
                    try:
                        if all(isinstance(v, (int, float)) for v in values):
                            metadata[key] = torch.tensor(values)
                        elif all(isinstance(v, str) for v in values):
                            # For string metadata, we'll skip tensor conversion
                            # but could implement string encoding if needed
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to collate metadata key '{key}': {e}")
        
        return metadata


class PreferenceDataCollator:
    """
    Data collator for preference learning datasets.
    
    Handles batching of preference pairs (preferred vs non-preferred responses)
    for training preference models or reward models.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        return_tensors: str = "pt"
    ):
        """
        Initialize the preference data collator.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            return_tensors: Type of tensors to return
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of preference examples."""
        # Separate preferred and non-preferred examples
        preferred_examples = []
        non_preferred_examples = []
        
        for example in examples:
            if "preferred_input_ids" in example:
                preferred_examples.append({
                    "input_ids": example["preferred_input_ids"],
                    "attention_mask": example.get("preferred_attention_mask"),
                    "token_rewards": example.get("preferred_token_rewards")
                })
            
            if "non_preferred_input_ids" in example:
                non_preferred_examples.append({
                    "input_ids": example["non_preferred_input_ids"],
                    "attention_mask": example.get("non_preferred_attention_mask"),
                    "token_rewards": example.get("non_preferred_token_rewards")
                })
        
        batch = {}
        
        # Collate preferred examples
        if preferred_examples:
            preferred_batch = self._collate_single_type(preferred_examples, "preferred")
            batch.update(preferred_batch)
        
        # Collate non-preferred examples
        if non_preferred_examples:
            non_preferred_batch = self._collate_single_type(non_preferred_examples, "non_preferred")
            batch.update(non_preferred_batch)
        
        return batch
    
    def _collate_single_type(
        self, 
        examples: List[Dict[str, Any]], 
        prefix: str
    ) -> Dict[str, torch.Tensor]:
        """Collate examples of a single type (preferred or non-preferred)."""
        # Extract sequences
        input_ids = [ex["input_ids"] for ex in examples]
        attention_masks = [ex.get("attention_mask") for ex in examples if ex.get("attention_mask") is not None]
        token_rewards = [ex.get("token_rewards") for ex in examples if ex.get("token_rewards") is not None]
        
        # Pad input_ids
        max_length = max(len(ids) for ids in input_ids)
        if self.max_length is not None:
            max_length = min(max_length, self.max_length)
        
        padded_input_ids = []
        padded_attention_masks = []
        padded_token_rewards = []
        
        for i, ids in enumerate(input_ids):
            # Pad input_ids
            if len(ids) >= max_length:
                padded_ids = ids[:max_length]
            else:
                pad_length = max_length - len(ids)
                padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=ids.dtype)
                padded_ids = torch.cat([ids, padding])
            padded_input_ids.append(padded_ids)
            
            # Pad attention_mask
            if i < len(attention_masks) and attention_masks[i] is not None:
                mask = attention_masks[i]
                if len(mask) >= max_length:
                    padded_mask = mask[:max_length]
                else:
                    pad_length = max_length - len(mask)
                    padding = torch.zeros(pad_length, dtype=mask.dtype)
                    padded_mask = torch.cat([mask, padding])
                padded_attention_masks.append(padded_mask)
            else:
                # Create attention mask from padded input_ids
                padded_mask = (padded_ids != self.tokenizer.pad_token_id).long()
                padded_attention_masks.append(padded_mask)
            
            # Pad token_rewards if present
            if i < len(token_rewards) and token_rewards[i] is not None:
                rewards = token_rewards[i]
                if len(rewards) >= max_length:
                    padded_rewards = rewards[:max_length]
                else:
                    pad_length = max_length - len(rewards)
                    padding = torch.zeros(pad_length, dtype=rewards.dtype)
                    padded_rewards = torch.cat([rewards, padding])
                padded_token_rewards.append(padded_rewards)
        
        # Create batch
        batch = {
            f"{prefix}_input_ids": torch.stack(padded_input_ids),
            f"{prefix}_attention_mask": torch.stack(padded_attention_masks)
        }
        
        if padded_token_rewards:
            batch[f"{prefix}_token_rewards"] = torch.stack(padded_token_rewards)
        
        return batch


class GenerationDataCollator:
    """
    Data collator for generation tasks.
    
    Handles batching of prompts for text generation during evaluation
    or online alignment.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = True,
        return_tensors: str = "pt",
        add_special_tokens: bool = True
    ):
        """
        Initialize the generation data collator.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            return_tensors: Type of tensors to return
            add_special_tokens: Whether to add special tokens
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, examples: List[Union[str, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of prompts for generation."""
        # Extract prompts
        if isinstance(examples[0], str):
            prompts = examples
        else:
            prompts = [ex["prompt"] if "prompt" in ex else ex["text"] for ex in examples]
        
        # Tokenize prompts
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens
        )
        
        return tokenized


# Utility functions for creating data collators

def create_tfpo_collator(
    tokenizer,
    max_length: Optional[int] = None,
    **kwargs
) -> TFPODataCollator:
    """Create a TFPO data collator with common settings."""
    return TFPODataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        **kwargs
    )


def create_preference_collator(
    tokenizer,
    max_length: Optional[int] = None,
    **kwargs
) -> PreferenceDataCollator:
    """Create a preference data collator with common settings."""
    return PreferenceDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        **kwargs
    )


def create_generation_collator(
    tokenizer,
    max_length: Optional[int] = None,
    **kwargs
) -> GenerationDataCollator:
    """Create a generation data collator with common settings."""
    return GenerationDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        **kwargs
    )