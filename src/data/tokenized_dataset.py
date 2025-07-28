"""
Tokenized dataset for LLMdoctor framework.

Handles the conversion of token reward data into a PyTorch dataset
for efficient batching and training with TFPO.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import logging

from ..reward.reward_processor import TokenRewardData

logger = logging.getLogger(__name__)


@dataclass
class TokenizedExample:
    """A single tokenized training example."""
    input_ids: torch.Tensor          # Token IDs for the sequence
    attention_mask: torch.Tensor     # Attention mask
    token_rewards: torch.Tensor      # Token-level rewards
    labels: Optional[torch.Tensor] = None   # Labels for language modeling
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class TokenizedDataset(Dataset):
    """
    Dataset class for tokenized preference data with rewards.
    
    Converts TokenRewardData objects into tokenized examples suitable
    for TFPO training with proper padding and attention masks.
    """
    
    def __init__(
        self,
        reward_data: List[TokenRewardData],
        tokenizer,
        max_length: int = 1024,
        padding: bool = True,
        truncation: bool = True,
        include_labels: bool = True,
        reward_pad_value: float = 0.0,
        filter_empty: bool = True
    ):
        """
        Initialize the tokenized dataset.
        
        Args:
            reward_data: List of TokenRewardData objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            include_labels: Whether to include labels for language modeling
            reward_pad_value: Value to use for padding rewards
            filter_empty: Whether to filter out empty examples
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.include_labels = include_labels
        self.reward_pad_value = reward_pad_value
        
        # Process and tokenize the data
        self.examples = self._process_reward_data(reward_data, filter_empty)
        
        logger.info(f"TokenizedDataset initialized with {len(self.examples)} examples")
        logger.info(f"Max length: {max_length}, Padding: {padding}, Truncation: {truncation}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized example by index."""
        example = self.examples[idx]
        
        result = {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "token_rewards": example.token_rewards
        }
        
        if example.labels is not None:
            result["labels"] = example.labels
        
        if example.metadata is not None:
            # Add metadata that can be converted to tensors
            for key, value in example.metadata.items():
                if isinstance(value, (int, float)):
                    result[key] = torch.tensor(value)
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
                    result[key] = torch.tensor(value)
        
        return result
    
    def _process_reward_data(
        self, 
        reward_data: List[TokenRewardData], 
        filter_empty: bool
    ) -> List[TokenizedExample]:
        """Process raw reward data into tokenized examples."""
        examples = []
        
        for data in reward_data:
            try:
                # Combine prompt and response
                full_text = data.prompt + data.response
                
                # Tokenize the text
                tokenized = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    padding="max_length" if self.padding else False,
                    truncation=self.truncation,
                    return_tensors="pt"
                )
                
                input_ids = tokenized["input_ids"].squeeze(0)
                attention_mask = tokenized["attention_mask"].squeeze(0)
                
                # Process token rewards
                token_rewards = self._process_token_rewards(
                    data.token_rewards, len(input_ids)
                )
                
                # Create labels for language modeling if requested
                labels = None
                if self.include_labels:
                    labels = input_ids.clone()
                    # Mask prompt tokens in labels (only compute loss on response)
                    prompt_tokens = self.tokenizer(
                        data.prompt,
                        add_special_tokens=False
                    )["input_ids"]
                    prompt_length = len(prompt_tokens)
                    if prompt_length < len(labels):
                        labels[:prompt_length] = -100
                
                # Create metadata
                metadata = {
                    "prompt_length": len(data.prompt),
                    "response_length": len(data.response),
                    "preference_dimension": data.preference_dimension,
                    "is_preferred": data.is_preferred
                }
                
                example = TokenizedExample(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_rewards=token_rewards,
                    labels=labels,
                    metadata=metadata
                )
                
                # Filter empty examples if requested
                if filter_empty and self._is_empty_example(example):
                    continue
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Failed to process reward data: {e}")
                continue
        
        return examples
    
    def _process_token_rewards(
        self, 
        token_rewards: torch.Tensor, 
        target_length: int
    ) -> torch.Tensor:
        """Process token rewards to match tokenized sequence length."""
        if len(token_rewards) == target_length:
            return token_rewards
        elif len(token_rewards) > target_length:
            # Truncate if too long
            return token_rewards[:target_length]
        else:
            # Pad if too short
            padding_length = target_length - len(token_rewards)
            padding = torch.full((padding_length,), self.reward_pad_value)
            return torch.cat([token_rewards, padding])
    
    def _is_empty_example(self, example: TokenizedExample) -> bool:
        """Check if an example is effectively empty."""
        # Check if all input_ids are padding tokens
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            non_pad_tokens = (example.input_ids != self.tokenizer.pad_token_id).sum().item()
        else:
            # If no pad token, consider any token as valid
            non_pad_tokens = len(example.input_ids)
        
        # Check if there are any non-zero rewards
        non_zero_rewards = (example.token_rewards != 0).sum().item()
        
        return non_pad_tokens < 2 or non_zero_rewards == 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics."""
        if not self.examples:
            return {}
        
        seq_lengths = [example.attention_mask.sum().item() for example in self.examples]
        reward_stats = [example.token_rewards[example.token_rewards != 0] for example in self.examples]
        
        # Flatten all non-zero rewards
        all_rewards = torch.cat([r for r in reward_stats if len(r) > 0])
        
        stats = {
            "num_examples": len(self.examples),
            "avg_sequence_length": sum(seq_lengths) / len(seq_lengths),
            "max_sequence_length": max(seq_lengths),
            "min_sequence_length": min(seq_lengths),
        }
        
        if len(all_rewards) > 0:
            stats.update({
                "avg_reward": float(torch.mean(all_rewards)),
                "std_reward": float(torch.std(all_rewards)),
                "max_reward": float(torch.max(all_rewards)),
                "min_reward": float(torch.min(all_rewards)),
                "num_non_zero_rewards": len(all_rewards)
            })
        
        return stats
    
    def filter_by_length(self, min_length: int = 10, max_length: Optional[int] = None) -> 'TokenizedDataset':
        """Create a new dataset filtered by sequence length."""
        max_length = max_length or self.max_length
        
        filtered_examples = []
        for example in self.examples:
            seq_len = example.attention_mask.sum().item()
            if min_length <= seq_len <= max_length:
                filtered_examples.append(example)
        
        # Create new dataset with filtered examples
        new_dataset = TokenizedDataset.__new__(TokenizedDataset)
        new_dataset.tokenizer = self.tokenizer
        new_dataset.max_length = self.max_length
        new_dataset.padding = self.padding
        new_dataset.truncation = self.truncation
        new_dataset.include_labels = self.include_labels
        new_dataset.reward_pad_value = self.reward_pad_value
        new_dataset.examples = filtered_examples
        
        logger.info(f"Filtered dataset: {len(self.examples)} -> {len(filtered_examples)} examples")
        return new_dataset
    
    def filter_by_rewards(self, min_reward_count: int = 1) -> 'TokenizedDataset':
        """Create a new dataset filtered by minimum number of non-zero rewards."""
        filtered_examples = []
        for example in self.examples:
            non_zero_rewards = (example.token_rewards != 0).sum().item()
            if non_zero_rewards >= min_reward_count:
                filtered_examples.append(example)
        
        # Create new dataset with filtered examples
        new_dataset = TokenizedDataset.__new__(TokenizedDataset)
        new_dataset.tokenizer = self.tokenizer
        new_dataset.max_length = self.max_length
        new_dataset.padding = self.padding
        new_dataset.truncation = self.truncation
        new_dataset.include_labels = self.include_labels
        new_dataset.reward_pad_value = self.reward_pad_value
        new_dataset.examples = filtered_examples
        
        logger.info(f"Reward filtered dataset: {len(self.examples)} -> {len(filtered_examples)} examples")
        return new_dataset
    
    def split(self, train_ratio: float = 0.8, seed: int = 42) -> tuple['TokenizedDataset', 'TokenizedDataset']:
        """Split the dataset into train and validation sets."""
        torch.manual_seed(seed)
        indices = torch.randperm(len(self.examples))
        
        train_size = int(len(self.examples) * train_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_examples = [self.examples[i] for i in train_indices]
        val_examples = [self.examples[i] for i in val_indices]
        
        # Create train dataset
        train_dataset = TokenizedDataset.__new__(TokenizedDataset)
        train_dataset.tokenizer = self.tokenizer
        train_dataset.max_length = self.max_length
        train_dataset.padding = self.padding
        train_dataset.truncation = self.truncation
        train_dataset.include_labels = self.include_labels
        train_dataset.reward_pad_value = self.reward_pad_value
        train_dataset.examples = train_examples
        
        # Create validation dataset
        val_dataset = TokenizedDataset.__new__(TokenizedDataset)
        val_dataset.tokenizer = self.tokenizer
        val_dataset.max_length = self.max_length
        val_dataset.padding = self.padding
        val_dataset.truncation = self.truncation
        val_dataset.include_labels = self.include_labels
        val_dataset.reward_pad_value = self.reward_pad_value
        val_dataset.examples = val_examples
        
        logger.info(f"Split dataset: {len(self.examples)} -> Train: {len(train_examples)}, Val: {len(val_examples)}")
        
        return train_dataset, val_dataset
    
    def save(self, filepath: str):
        """Save the dataset to disk."""
        torch.save({
            'examples': self.examples,
            'tokenizer_name': getattr(self.tokenizer, 'name_or_path', 'unknown'),
            'max_length': self.max_length,
            'padding': self.padding,
            'truncation': self.truncation,
            'include_labels': self.include_labels,
            'reward_pad_value': self.reward_pad_value
        }, filepath)
        logger.info(f"Dataset saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, tokenizer) -> 'TokenizedDataset':
        """Load a dataset from disk."""
        data = torch.load(filepath, map_location='cpu')
        
        dataset = cls.__new__(cls)
        dataset.tokenizer = tokenizer
        dataset.max_length = data['max_length']
        dataset.padding = data['padding']
        dataset.truncation = data['truncation']
        dataset.include_labels = data['include_labels']
        dataset.reward_pad_value = data['reward_pad_value']
        dataset.examples = data['examples']
        
        logger.info(f"Dataset loaded from {filepath} with {len(dataset.examples)} examples")
        return dataset