"""
Reward data processor for LLMdoctor framework.

Main coordinator for the token-level reward acquisition stage.
Processes preference datasets and extracts token-level rewards.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import logging

from .behavioral_variants import BehavioralVariantCreator, BehavioralVariantConfig
from .token_importance import TokenImportanceCalculator, TokenImportanceConfig
from .reward_assignment import RewardAssigner, RewardAssignmentConfig, PreferenceLabel

logger = logging.getLogger(__name__)


@dataclass
class RewardProcessingConfig:
    """Configuration for reward data processing."""
    batch_size: int = 8
    max_sequence_length: int = 2048
    save_intermediate_results: bool = True
    intermediate_save_interval: int = 100
    num_workers: int = 1
    enable_caching: bool = True
    cache_dir: str = "./cache"
    log_progress: bool = True
    dimension: str = "helpfulness"
    
    # Sub-component configs
    variant_config: Optional[BehavioralVariantConfig] = None
    importance_config: Optional[TokenImportanceConfig] = None
    reward_config: Optional[RewardAssignmentConfig] = None


@dataclass
class PreferenceExample:
    """Single preference example."""
    prompt: str
    preferred_response: str
    non_preferred_response: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TokenRewardData:
    """Token-level reward data for a single example."""
    prompt: str
    response: str
    preference_label: int  # 1 for preferred, -1 for non-preferred
    tokens: List[str]
    importance_scores: torch.Tensor
    rewards: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None


class RewardDataProcessor:
    """
    Main processor for extracting token-level rewards from preference data.
    
    Coordinates the three-step process:
    1. Create behavioral variants
    2. Compute token importance
    3. Assign directional rewards
    """
    
    def __init__(
        self,
        patient_model,
        config: Optional[RewardProcessingConfig] = None,
        device: Union[str, torch.device] = "auto"
    ):
        """
        Initialize the reward data processor.
        
        Args:
            patient_model: PatientModel instance
            config: Processing configuration
            device: Device for computations
        """
        self.patient_model = patient_model
        self.config = config if config is not None else RewardProcessingConfig()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize sub-components
        self.variant_creator = BehavioralVariantCreator(
            config=self.config.variant_config
        )
        
        self.importance_calculator = TokenImportanceCalculator(
            config=self.config.importance_config,
            device=self.device
        )
        
        self.reward_assigner = RewardAssigner(
            config=self.config.reward_config,
            device=self.device
        )
        
        # Create cache directory
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RewardDataProcessor initialized on {self.device}")
    
    def process_preference_dataset(
        self,
        preference_examples: List[PreferenceExample],
        output_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> List[TokenRewardData]:
        """
        Process a complete preference dataset to extract token-level rewards.
        
        Args:
            preference_examples: List of preference examples
            output_path: Optional path to save results
            resume_from_checkpoint: Optional checkpoint to resume from
            
        Returns:
            List of token reward data
        """
        logger.info(f"Processing {len(preference_examples)} preference examples")
        
        # Check for resume
        start_idx = 0
        processed_data = []
        
        if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            processed_data = self.load_processed_data(resume_from_checkpoint)
            start_idx = len(processed_data) // 2  # Each example produces 2 reward entries
        
        # Process remaining examples
        for i in tqdm(
            range(start_idx, len(preference_examples)), 
            desc="Processing examples",
            disable=not self.config.log_progress
        ):
            example = preference_examples[i]
            
            try:
                # Process single example (produces 2 reward entries)
                reward_data_pair = self.process_single_example(example)
                processed_data.extend(reward_data_pair)
                
                # Save intermediate results
                if (self.config.save_intermediate_results and 
                    i % self.config.intermediate_save_interval == 0 and 
                    output_path):
                    self._save_checkpoint(processed_data, f"{output_path}.checkpoint")
                
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_data)} reward entries")
        
        # Save final results
        if output_path:
            self.save_processed_data(processed_data, output_path)
        
        return processed_data
    
    def process_single_example(
        self,
        example: PreferenceExample
    ) -> List[TokenRewardData]:
        """
        Process a single preference example to extract token rewards.
        
        Args:
            example: Preference example
            
        Returns:
            List of token reward data (preferred and non-preferred)
        """
        # Extract token importance for preferred response
        preferred_importance = self.importance_calculator.compute_token_importance(
            patient_model=self.patient_model,
            prompt=example.prompt,
            response=example.preferred_response,
            return_details=True
        )
        
        # Extract token importance for non-preferred response
        non_preferred_importance = self.importance_calculator.compute_token_importance(
            patient_model=self.patient_model,
            prompt=example.prompt,
            response=example.non_preferred_response,
            return_details=True
        )
        
        # Assign rewards for preferred response
        preferred_rewards = self.reward_assigner.assign_token_rewards(
            importance_scores=preferred_importance["importance_scores"],
            preference_label=PreferenceLabel.PREFERRED,
            tokens=preferred_importance["tokens"]
        )
        
        # Assign rewards for non-preferred response
        non_preferred_rewards = self.reward_assigner.assign_token_rewards(
            importance_scores=non_preferred_importance["importance_scores"],
            preference_label=PreferenceLabel.NON_PREFERRED,
            tokens=non_preferred_importance["tokens"]
        )
        
        # Create reward data objects
        preferred_data = TokenRewardData(
            prompt=example.prompt,
            response=example.preferred_response,
            preference_label=1,
            tokens=preferred_importance["tokens"],
            importance_scores=preferred_importance["importance_scores"],
            rewards=preferred_rewards["rewards"],
            metadata={
                "total_reward": preferred_rewards["total_reward"],
                "num_rewarded_tokens": preferred_rewards["num_rewarded_tokens"],
                "avg_reward": preferred_rewards["avg_reward"],
                "original_metadata": example.metadata
            }
        )
        
        non_preferred_data = TokenRewardData(
            prompt=example.prompt,
            response=example.non_preferred_response,
            preference_label=-1,
            tokens=non_preferred_importance["tokens"],
            importance_scores=non_preferred_importance["importance_scores"],
            rewards=non_preferred_rewards["rewards"],
            metadata={
                "total_reward": non_preferred_rewards["total_reward"],
                "num_rewarded_tokens": non_preferred_rewards["num_rewarded_tokens"],
                "avg_reward": non_preferred_rewards["avg_reward"],
                "original_metadata": example.metadata
            }
        )
        
        return [preferred_data, non_preferred_data]
    
    def process_batch_examples(
        self,
        examples: List[PreferenceExample],
        batch_size: Optional[int] = None
    ) -> List[TokenRewardData]:
        """
        Process a batch of preference examples efficiently.
        
        Args:
            examples: List of preference examples
            batch_size: Batch size (uses config default if None)
            
        Returns:
            List of token reward data
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        all_reward_data = []
        
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            batch_reward_data = []
            for example in batch_examples:
                reward_data_pair = self.process_single_example(example)
                batch_reward_data.extend(reward_data_pair)
            
            all_reward_data.extend(batch_reward_data)
            
            if self.config.log_progress:
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(examples) + batch_size - 1)//batch_size}")
        
        return all_reward_data
    
    def compute_dataset_statistics(
        self,
        reward_data: List[TokenRewardData]
    ) -> Dict[str, Any]:
        """
        Compute statistics over the processed reward dataset.
        
        Args:
            reward_data: List of token reward data
            
        Returns:
            Dictionary with dataset statistics
        """
        if not reward_data:
            return {}
        
        # Separate preferred and non-preferred
        preferred_data = [d for d in reward_data if d.preference_label == 1]
        non_preferred_data = [d for d in reward_data if d.preference_label == -1]
        
        # Compute basic statistics
        stats = {
            "total_examples": len(reward_data),
            "preferred_examples": len(preferred_data),
            "non_preferred_examples": len(non_preferred_data),
            "avg_tokens_per_response": np.mean([len(d.tokens) for d in reward_data]),
            "avg_rewarded_tokens_per_response": np.mean([
                int(torch.sum(d.rewards != 0)) for d in reward_data
            ]),
            "overall_sparsity": 1.0 - np.mean([
                int(torch.sum(d.rewards != 0)) / len(d.tokens) for d in reward_data
            ])
        }
        
        # Compute reward statistics
        all_rewards = torch.cat([d.rewards for d in reward_data])
        nonzero_rewards = all_rewards[all_rewards != 0]
        
        if len(nonzero_rewards) > 0:
            stats.update({
                "mean_reward": float(torch.mean(nonzero_rewards)),
                "std_reward": float(torch.std(nonzero_rewards)),
                "min_reward": float(torch.min(nonzero_rewards)),
                "max_reward": float(torch.max(nonzero_rewards)),
                "positive_reward_ratio": float(torch.sum(nonzero_rewards > 0)) / len(nonzero_rewards),
                "negative_reward_ratio": float(torch.sum(nonzero_rewards < 0)) / len(nonzero_rewards)
            })
        
        # Compute preference-specific statistics
        if preferred_data:
            preferred_rewards = torch.cat([d.rewards for d in preferred_data])
            preferred_nonzero = preferred_rewards[preferred_rewards != 0]
            
            if len(preferred_nonzero) > 0:
                stats["preferred_mean_reward"] = float(torch.mean(preferred_nonzero))
                stats["preferred_total_reward"] = float(torch.sum(preferred_rewards))
        
        if non_preferred_data:
            non_preferred_rewards = torch.cat([d.rewards for d in non_preferred_data])
            non_preferred_nonzero = non_preferred_rewards[non_preferred_rewards != 0]
            
            if len(non_preferred_nonzero) > 0:
                stats["non_preferred_mean_reward"] = float(torch.mean(non_preferred_nonzero))
                stats["non_preferred_total_reward"] = float(torch.sum(non_preferred_rewards))
        
        return stats
    
    def save_processed_data(
        self,
        reward_data: List[TokenRewardData],
        filepath: str,
        format: str = "pickle"
    ):
        """
        Save processed reward data to file.
        
        Args:
            reward_data: List of token reward data
            filepath: Output file path
            format: Output format ("pickle", "json")
        """
        if format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(reward_data, f)
        
        elif format == "json":
            # Convert to JSON-serializable format
            json_data = []
            for data in reward_data:
                json_item = {
                    "prompt": data.prompt,
                    "response": data.response,
                    "preference_label": data.preference_label,
                    "tokens": data.tokens,
                    "importance_scores": data.importance_scores.tolist(),
                    "rewards": data.rewards.tolist(),
                    "metadata": data.metadata
                }
                json_data.append(json_item)
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(reward_data)} reward entries to {filepath}")
    
    def load_processed_data(
        self,
        filepath: str,
        format: str = "auto"
    ) -> List[TokenRewardData]:
        """
        Load processed reward data from file.
        
        Args:
            filepath: Input file path
            format: Input format ("pickle", "json", "auto")
            
        Returns:
            List of token reward data
        """
        if format == "auto":
            format = "pickle" if filepath.endswith('.pkl') else "json"
        
        if format == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        elif format == "json":
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            reward_data = []
            for item in json_data:
                data = TokenRewardData(
                    prompt=item["prompt"],
                    response=item["response"],
                    preference_label=item["preference_label"],
                    tokens=item["tokens"],
                    importance_scores=torch.tensor(item["importance_scores"]),
                    rewards=torch.tensor(item["rewards"]),
                    metadata=item.get("metadata")
                )
                reward_data.append(data)
            
            return reward_data
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_checkpoint(self, data: List[TokenRewardData], filepath: str):
        """Save processing checkpoint."""
        self.save_processed_data(data, filepath, format="pickle")
        logger.info(f"Checkpoint saved: {filepath}")
    
    def filter_by_quality(
        self,
        reward_data: List[TokenRewardData],
        min_tokens: int = 5,
        min_rewarded_tokens: int = 1,
        min_reward_magnitude: float = 0.01
    ) -> List[TokenRewardData]:
        """
        Filter reward data by quality criteria.
        
        Args:
            reward_data: List of token reward data
            min_tokens: Minimum number of tokens
            min_rewarded_tokens: Minimum number of rewarded tokens
            min_reward_magnitude: Minimum reward magnitude
            
        Returns:
            Filtered reward data
        """
        filtered_data = []
        
        for data in reward_data:
            # Check minimum tokens
            if len(data.tokens) < min_tokens:
                continue
            
            # Check minimum rewarded tokens
            num_rewarded = int(torch.sum(data.rewards != 0))
            if num_rewarded < min_rewarded_tokens:
                continue
            
            # Check minimum reward magnitude
            max_reward_magnitude = float(torch.max(torch.abs(data.rewards)))
            if max_reward_magnitude < min_reward_magnitude:
                continue
            
            filtered_data.append(data)
        
        logger.info(f"Filtered {len(reward_data)} -> {len(filtered_data)} examples")
        return filtered_data
    
    def create_training_dataset(
        self,
        reward_data: List[TokenRewardData],
        max_length: Optional[int] = None,
        balance_preferences: bool = True
    ) -> Dict[str, List]:
        """
        Create training dataset from reward data for TFPO training.
        
        Args:
            reward_data: List of token reward data
            max_length: Maximum sequence length
            balance_preferences: Whether to balance preferred/non-preferred examples
            
        Returns:
            Dictionary with training data
        """
        if max_length is None:
            max_length = self.config.max_sequence_length
        
        # Separate preferred and non-preferred
        preferred_data = [d for d in reward_data if d.preference_label == 1]
        non_preferred_data = [d for d in reward_data if d.preference_label == -1]
        
        # Balance if requested
        if balance_preferences and len(preferred_data) != len(non_preferred_data):
            min_count = min(len(preferred_data), len(non_preferred_data))
            preferred_data = preferred_data[:min_count]
            non_preferred_data = non_preferred_data[:min_count]
            logger.info(f"Balanced dataset: {min_count} preferred, {min_count} non-preferred")
        
        # Combine and shuffle
        all_data = preferred_data + non_preferred_data
        np.random.shuffle(all_data)
        
        # Extract training data
        prompts = []
        responses = []
        rewards = []
        preference_labels = []
        
        for data in all_data:
            # Truncate if necessary
            if len(data.tokens) > max_length:
                truncated_tokens = data.tokens[:max_length]
                truncated_rewards = data.rewards[:max_length]
            else:
                truncated_tokens = data.tokens
                truncated_rewards = data.rewards
            
            # Combine prompt and response
            full_text = data.prompt + " " + " ".join(truncated_tokens)
            
            prompts.append(data.prompt)
            responses.append(" ".join(truncated_tokens))
            rewards.append(truncated_rewards)
            preference_labels.append(data.preference_label)
        
        return {
            "prompts": prompts,
            "responses": responses,
            "rewards": rewards,
            "preference_labels": preference_labels,
            "num_examples": len(all_data)
        }