"""
Data loading utilities for LLMdoctor framework.

Handles loading and preprocessing of various preference datasets
like HH-RLHF, PKU-SafeRLHF, UltraFeedback, etc.
"""

import torch
import json
import pandas as pd
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass

from ..reward.reward_processor import TokenRewardData
from .tokenized_dataset import TokenizedDataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    split: str = "train"
    subset: Optional[str] = None
    cache_dir: Optional[str] = None
    max_samples: Optional[int] = None
    seed: int = 42
    trust_remote_code: bool = False


class PreferenceDataLoader:
    """
    Loads and preprocesses preference datasets for LLMdoctor training.
    
    Supports various formats:
    - HH-RLHF format (prompt, chosen, rejected)
    - PKU-SafeRLHF format (with multiple dimensions)
    - UltraFeedback format
    - Custom JSON/CSV formats
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize the data loader.
        
        Args:
            cache_dir: Directory for caching datasets
            seed: Random seed for reproducibility
        """
        self.cache_dir = cache_dir
        self.seed = seed
        
        # Dataset format handlers
        self.format_handlers = {
            "hh_rlhf": self._load_hh_rlhf,
            "pku_saferlhf": self._load_pku_saferlhf,
            "ultrafeedback": self._load_ultrafeedback,
            "anthropic_hh": self._load_anthropic_hh,
            "json": self._load_json,
            "csv": self._load_csv,
            "custom": self._load_custom
        }
        
        logger.info("PreferenceDataLoader initialized")
    
    def load_dataset(
        self,
        config: DatasetConfig,
        format_type: str = "auto"
    ) -> Dataset:
        """
        Load a dataset based on configuration.
        
        Args:
            config: Dataset configuration
            format_type: Dataset format type or "auto" for auto-detection
            
        Returns:
            Loaded HuggingFace Dataset
        """
        if format_type == "auto":
            format_type = self._detect_format(config.name)
        
        if format_type not in self.format_handlers:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Loading dataset: {config.name} (format: {format_type})")
        
        dataset = self.format_handlers[format_type](config)
        
        # Apply sampling if requested
        if config.max_samples is not None and len(dataset) > config.max_samples:
            dataset = dataset.shuffle(seed=config.seed).select(range(config.max_samples))
            logger.info(f"Sampled {config.max_samples} examples from dataset")
        
        return dataset
    
    def load_preference_pairs(
        self,
        config: DatasetConfig,
        format_type: str = "auto",
        preference_key: str = "preferred"
    ) -> List[Dict[str, Any]]:
        """
        Load preference pairs from a dataset.
        
        Args:
            config: Dataset configuration
            format_type: Dataset format type
            preference_key: Key indicating preference in the data
            
        Returns:
            List of preference pair dictionaries
        """
        dataset = self.load_dataset(config, format_type)
        
        preference_pairs = []
        for item in dataset:
            pair = self._extract_preference_pair(item, preference_key)
            if pair is not None:
                preference_pairs.append(pair)
        
        logger.info(f"Loaded {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def create_token_reward_data(
        self,
        patient_model,
        preference_pairs: List[Dict[str, Any]],
        importance_calculator,
        preference_dimension: str = "helpfulness",
        batch_size: int = 8,
        progress_callback: Optional[callable] = None
    ) -> List[TokenRewardData]:
        """
        Create TokenRewardData from preference pairs using token importance calculation.
        
        Args:
            patient_model: PatientModel instance
            preference_pairs: List of preference pair dictionaries
            importance_calculator: TokenImportanceCalculator instance
            preference_dimension: Preference dimension to analyze
            batch_size: Batch size for processing
            progress_callback: Optional progress callback
            
        Returns:
            List of TokenRewardData objects
        """
        token_reward_data = []
        
        for i, pair in enumerate(preference_pairs):
            try:
                prompt = pair["prompt"]
                chosen = pair["chosen"]
                rejected = pair["rejected"]
                
                # Compute token importance for chosen response
                chosen_importance = importance_calculator.compute_token_importance(
                    patient_model=patient_model,
                    prompt=prompt,
                    response=chosen,
                    return_details=True
                )
                
                # Compute token importance for rejected response  
                rejected_importance = importance_calculator.compute_token_importance(
                    patient_model=patient_model,
                    prompt=prompt,
                    response=rejected,
                    return_details=True
                )
                
                # Create TokenRewardData for chosen response (positive)
                chosen_data = TokenRewardData(
                    prompt=prompt,
                    response=chosen,
                    token_rewards=chosen_importance["importance_scores"],
                    is_preferred=True,
                    preference_dimension=preference_dimension,
                    metadata={
                        "source": "chosen",
                        "pair_id": i,
                        "tokens": chosen_importance["tokens"]
                    }
                )
                
                # Create TokenRewardData for rejected response (negative)
                rejected_data = TokenRewardData(
                    prompt=prompt,
                    response=rejected,
                    token_rewards=-rejected_importance["importance_scores"],  # Negative rewards
                    is_preferred=False,
                    preference_dimension=preference_dimension,
                    metadata={
                        "source": "rejected",
                        "pair_id": i,
                        "tokens": rejected_importance["tokens"]
                    }
                )
                
                token_reward_data.extend([chosen_data, rejected_data])
                
                if progress_callback:
                    progress_callback(i + 1, len(preference_pairs))
                    
            except Exception as e:
                logger.warning(f"Failed to process preference pair {i}: {e}")
                continue
        
        logger.info(f"Created {len(token_reward_data)} TokenRewardData objects")
        return token_reward_data
    
    def _detect_format(self, dataset_name: str) -> str:
        """Auto-detect dataset format from name."""
        name_lower = dataset_name.lower()
        
        if "hh" in name_lower and "rlhf" in name_lower:
            return "hh_rlhf"
        elif "pku" in name_lower and "safe" in name_lower:
            return "pku_saferlhf"
        elif "ultra" in name_lower and "feedback" in name_lower:
            return "ultrafeedback"
        elif "anthropic" in name_lower:
            return "anthropic_hh"
        elif dataset_name.endswith(".json"):
            return "json"
        elif dataset_name.endswith(".csv"):
            return "csv"
        else:
            return "custom"
    
    def _load_hh_rlhf(self, config: DatasetConfig) -> Dataset:
        """Load HH-RLHF dataset."""
        try:
            dataset = load_dataset(
                "Anthropic/hh-rlhf",
                split=config.split,
                cache_dir=self.cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            # Convert to standard format
            def format_hh_rlhf(example):
                return {
                    "prompt": example.get("prompt", ""),
                    "chosen": example.get("chosen", ""),
                    "rejected": example.get("rejected", ""),
                    "preference_dimension": "helpfulness"
                }
            
            return dataset.map(format_hh_rlhf)
            
        except Exception as e:
            logger.error(f"Failed to load HH-RLHF dataset: {e}")
            raise
    
    def _load_pku_saferlhf(self, config: DatasetConfig) -> Dataset:
        """Load PKU-SafeRLHF dataset."""
        try:
            dataset = load_dataset(
                "PKU-Alignment/PKU-SafeRLHF",
                split=config.split,
                cache_dir=self.cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            # Convert to standard format with multiple dimensions
            def format_pku_saferlhf(example):
                return {
                    "prompt": example.get("prompt", ""),
                    "chosen": example.get("response_0", ""),
                    "rejected": example.get("response_1", ""),
                    "safer_response": example.get("safer_response_id", 0),
                    "better_response": example.get("better_response_id", 0),
                    "preference_dimension": "safety_helpfulness"
                }
            
            return dataset.map(format_pku_saferlhf)
            
        except Exception as e:
            logger.error(f"Failed to load PKU-SafeRLHF dataset: {e}")
            raise
    
    def _load_ultrafeedback(self, config: DatasetConfig) -> Dataset:
        """Load UltraFeedback dataset."""
        try:
            dataset = load_dataset(
                "openbmb/UltraFeedback",
                split=config.split,
                cache_dir=self.cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            
            # Convert to standard format
            def format_ultrafeedback(example):
                # UltraFeedback has multiple responses with ratings
                responses = example.get("completions", [])
                if len(responses) >= 2:
                    # Sort by rating and take best/worst
                    sorted_responses = sorted(responses, key=lambda x: x.get("overall", 0), reverse=True)
                    chosen = sorted_responses[0]["response"]
                    rejected = sorted_responses[-1]["response"]
                else:
                    chosen = responses[0]["response"] if responses else ""
                    rejected = ""
                
                return {
                    "prompt": example.get("instruction", ""),
                    "chosen": chosen,
                    "rejected": rejected,
                    "preference_dimension": "helpfulness"
                }
            
            return dataset.map(format_ultrafeedback)
            
        except Exception as e:
            logger.error(f"Failed to load UltraFeedback dataset: {e}")
            raise
    
    def _load_anthropic_hh(self, config: DatasetConfig) -> Dataset:
        """Load Anthropic HH dataset."""
        # Similar to HH-RLHF but different repository
        return self._load_hh_rlhf(config)
    
    def _load_json(self, config: DatasetConfig) -> Dataset:
        """Load dataset from JSON file."""
        try:
            with open(config.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Ensure data is a list of dictionaries
            if isinstance(data, dict):
                data = [data]
            
            return Dataset.from_list(data)
            
        except Exception as e:
            logger.error(f"Failed to load JSON dataset: {e}")
            raise
    
    def _load_csv(self, config: DatasetConfig) -> Dataset:
        """Load dataset from CSV file."""
        try:
            df = pd.read_csv(config.name)
            data = df.to_dict('records')
            
            return Dataset.from_list(data)
            
        except Exception as e:
            logger.error(f"Failed to load CSV dataset: {e}")
            raise
    
    def _load_custom(self, config: DatasetConfig) -> Dataset:
        """Load custom dataset format."""
        try:
            # Try to load as HuggingFace dataset first
            dataset = load_dataset(
                config.name,
                config.subset,
                split=config.split,
                cache_dir=self.cache_dir,
                trust_remote_code=config.trust_remote_code
            )
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load custom dataset: {e}")
            raise
    
    def _extract_preference_pair(
        self, 
        item: Dict[str, Any], 
        preference_key: str = "preferred"
    ) -> Optional[Dict[str, Any]]:
        """Extract preference pair from dataset item."""
        try:
            # Standard format
            if "chosen" in item and "rejected" in item:
                return {
                    "prompt": item.get("prompt", ""),
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "preference_dimension": item.get("preference_dimension", "helpfulness")
                }
            
            # Alternative format with preference indicators
            elif preference_key in item:
                responses = item.get("responses", [])
                if len(responses) >= 2:
                    preferred_idx = item[preference_key]
                    if isinstance(preferred_idx, bool):
                        chosen = responses[0] if preferred_idx else responses[1]
                        rejected = responses[1] if preferred_idx else responses[0]
                    else:
                        chosen = responses[preferred_idx]
                        rejected = responses[1 - preferred_idx]
                    
                    return {
                        "prompt": item.get("prompt", ""),
                        "chosen": chosen,
                        "rejected": rejected,
                        "preference_dimension": item.get("preference_dimension", "helpfulness")
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract preference pair: {e}")
            return None
    
    def save_processed_data(
        self,
        token_reward_data: List[TokenRewardData],
        filepath: str,
        format: str = "json"
    ):
        """Save processed token reward data."""
        if format == "json":
            data_dicts = []
            for data in token_reward_data:
                data_dict = {
                    "prompt": data.prompt,
                    "response": data.response,
                    "token_rewards": data.token_rewards.tolist(),
                    "is_preferred": data.is_preferred,
                    "preference_dimension": data.preference_dimension,
                    "metadata": data.metadata
                }
                data_dicts.append(data_dict)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_dicts, f, indent=2, ensure_ascii=False)
                
        elif format == "torch":
            torch.save(token_reward_data, filepath)
        else:
            raise ValueError(f"Unsupported save format: {format}")
        
        logger.info(f"Saved {len(token_reward_data)} TokenRewardData to {filepath}")
    
    def load_processed_data(
        self,
        filepath: str,
        format: str = "json"
    ) -> List[TokenRewardData]:
        """Load processed token reward data."""
        if format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                data_dicts = json.load(f)
            
            token_reward_data = []
            for data_dict in data_dicts:
                data = TokenRewardData(
                    prompt=data_dict["prompt"],
                    response=data_dict["response"],
                    token_rewards=torch.tensor(data_dict["token_rewards"]),
                    is_preferred=data_dict["is_preferred"],
                    preference_dimension=data_dict["preference_dimension"],
                    metadata=data_dict.get("metadata")
                )
                token_reward_data.append(data)
            
            return token_reward_data
            
        elif format == "torch":
            return torch.load(filepath, map_location='cpu')
        else:
            raise ValueError(f"Unsupported load format: {format}")


# Utility functions

def create_dataset_config(
    name: str,
    split: str = "train",
    **kwargs
) -> DatasetConfig:
    """Create a dataset configuration with common settings."""
    return DatasetConfig(name=name, split=split, **kwargs)


def load_hh_rlhf(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Dataset:
    """Quick loader for HH-RLHF dataset."""
    loader = PreferenceDataLoader(cache_dir=cache_dir)
    config = DatasetConfig(name="hh_rlhf", split=split, max_samples=max_samples)
    return loader.load_dataset(config, format_type="hh_rlhf")


def load_pku_saferlhf(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Dataset:
    """Quick loader for PKU-SafeRLHF dataset."""
    loader = PreferenceDataLoader(cache_dir=cache_dir)
    config = DatasetConfig(name="pku_saferlhf", split=split, max_samples=max_samples)
    return loader.load_dataset(config, format_type="pku_saferlhf")