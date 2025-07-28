"""
Perplexity evaluator for language model evaluation.

Computes perplexity of generated text to assess fluency and coherence.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """
    Evaluates text perplexity using a language model.
    
    Lower perplexity indicates more fluent and predictable text.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: Union[str, torch.device] = "auto",
        batch_size: int = 8,
        max_length: int = 1024
    ):
        """
        Initialize perplexity evaluator.
        
        Args:
            model_name: Name of model to use for perplexity calculation
            device: Device for computation
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self._load_model(model_name)
        
        logger.info(f"PerplexityEvaluator initialized with {model_name} on {self.device}")
    
    def _load_model(self, model_name: str):
        """Load model and tokenizer for perplexity calculation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def compute(
        self,
        texts: Union[List[str], str],
        model: Optional[nn.Module] = None,
        return_per_text: bool = False
    ) -> Union[float, List[float]]:
        """
        Compute perplexity for text(s).
        
        Args:
            texts: Text or list of texts to evaluate
            model: Optional model to use (otherwise uses loaded model)
            return_per_text: Whether to return perplexity per text
            
        Returns:
            Average perplexity or list of perplexities
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return 0.0 if not return_per_text else []
        
        # Use provided model or default
        eval_model = model if model is not None else self.model
        
        perplexities = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Computing perplexity"):
            batch_texts = texts[i:i + self.batch_size]
            batch_perplexities = self._compute_batch_perplexity(
                batch_texts, eval_model
            )
            perplexities.extend(batch_perplexities)
        
        if return_per_text:
            return perplexities
        else:
            return np.mean(perplexities)
    
    def _compute_batch_perplexity(
        self,
        texts: List[str],
        model: nn.Module
    ) -> List[float]:
        """Compute perplexity for a batch of texts."""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # Compute log probabilities
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Get loss per token
            logits = outputs.logits
            
        # Compute perplexity for each text
        batch_perplexities = []
        
        for idx in range(len(texts)):
            # Get valid tokens (non-padding)
            valid_mask = attention_mask[idx] == 1
            valid_length = valid_mask.sum().item()
            
            if valid_length <= 1:
                batch_perplexities.append(0.0)
                continue
            
            # Compute cross entropy loss
            shift_logits = logits[idx, :-1, :].contiguous()
            shift_labels = input_ids[idx, 1:].contiguous()
            shift_mask = valid_mask[1:].contiguous()
            
            # Only compute loss on valid tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Mask out padding tokens
            masked_losses = token_losses * shift_mask.view(-1).float()
            avg_loss = masked_losses.sum() / shift_mask.sum().float()
            
            # Perplexity = exp(avg_loss)
            perplexity = torch.exp(avg_loss).item()
            
            # Handle edge cases
            if np.isnan(perplexity) or np.isinf(perplexity):
                perplexity = 1e6  # Large but finite value
            
            batch_perplexities.append(perplexity)
        
        return batch_perplexities
    
    def compute_conditional_perplexity(
        self,
        prompts: List[str],
        completions: List[str],
        model: Optional[nn.Module] = None
    ) -> Union[float, List[float]]:
        """
        Compute perplexity of completions conditioned on prompts.
        
        This is useful for evaluating how well the completions
        follow from the prompts.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            model: Optional model to use
            
        Returns:
            Average conditional perplexity
        """
        if len(prompts) != len(completions):
            raise ValueError("Number of prompts and completions must match")
        
        eval_model = model if model is not None else self.model
        
        conditional_perplexities = []
        
        for prompt, completion in zip(prompts, completions):
            # Combine prompt and completion
            full_text = prompt + " " + completion
            
            # Tokenize
            prompt_encoded = self.tokenizer.encode(prompt, add_special_tokens=True)
            full_encoded = self.tokenizer.encode(full_text, add_special_tokens=True)
            
            # Get completion token indices
            completion_start_idx = len(prompt_encoded)
            
            if len(full_encoded) <= completion_start_idx:
                conditional_perplexities.append(0.0)
                continue
            
            # Compute perplexity only on completion tokens
            with torch.no_grad():
                input_ids = torch.tensor([full_encoded]).to(self.device)
                outputs = eval_model(input_ids=input_ids, labels=input_ids)
                
                # Get logits
                logits = outputs.logits[0]  # Remove batch dimension
                
                # Compute loss only on completion tokens
                shift_logits = logits[completion_start_idx-1:-1]
                shift_labels = input_ids[0, completion_start_idx:]
                
                if len(shift_logits) == 0:
                    conditional_perplexities.append(0.0)
                    continue
                
                # Compute cross entropy
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                perplexity = torch.exp(loss).item()
                
                # Handle edge cases
                if np.isnan(perplexity) or np.isinf(perplexity):
                    perplexity = 1e6
                
                conditional_perplexities.append(perplexity)
        
        return np.mean(conditional_perplexities)
    
    def compute_sliding_window_perplexity(
        self,
        text: str,
        window_size: int = 512,
        stride: int = 256,
        model: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Compute perplexity using sliding window for long texts.
        
        Args:
            text: Input text
            window_size: Size of sliding window
            stride: Stride for sliding window
            model: Optional model to use
            
        Returns:
            Dictionary with perplexity statistics
        """
        eval_model = model if model is not None else self.model
        
        # Tokenize full text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) <= window_size:
            # Text fits in one window
            perplexity = self.compute(text, model=eval_model)
            return {
                "mean_perplexity": perplexity,
                "min_perplexity": perplexity,
                "max_perplexity": perplexity,
                "std_perplexity": 0.0,
                "num_windows": 1
            }
        
        # Compute perplexity for each window
        window_perplexities = []
        
        for start_idx in range(0, len(tokens) - window_size + 1, stride):
            window_tokens = tokens[start_idx:start_idx + window_size]
            window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
            
            window_perplexity = self.compute(window_text, model=eval_model)
            window_perplexities.append(window_perplexity)
        
        return {
            "mean_perplexity": np.mean(window_perplexities),
            "min_perplexity": np.min(window_perplexities),
            "max_perplexity": np.max(window_perplexities),
            "std_perplexity": np.std(window_perplexities),
            "num_windows": len(window_perplexities)
        }
    
    def compare_models_perplexity(
        self,
        texts: List[str],
        models: Dict[str, nn.Module]
    ) -> Dict[str, float]:
        """
        Compare perplexity across multiple models.
        
        Args:
            texts: List of texts to evaluate
            models: Dictionary mapping model names to models
            
        Returns:
            Dictionary mapping model names to average perplexity
        """
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Computing perplexity for {model_name}")
            avg_perplexity = self.compute(texts, model=model)
            results[model_name] = avg_perplexity
        
        return results
    
    def plot_perplexity_distribution(
        self,
        texts: List[str],
        model: Optional[nn.Module] = None,
        save_path: Optional[str] = None,
        bins: int = 50
    ):
        """
        Plot distribution of perplexities.
        
        Args:
            texts: List of texts
            model: Optional model to use
            save_path: Path to save plot
            bins: Number of bins for histogram
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return
        
        # Compute perplexities
        perplexities = self.compute(texts, model=model, return_per_text=True)
        
        # Filter out extreme values for better visualization
        perplexities = [p for p in perplexities if p < np.percentile(perplexities, 95)]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(perplexities, bins=bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Perplexity')
        plt.ylabel('Count')
        plt.title('Perplexity Distribution')
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(perplexities, vert=True)
        plt.ylabel('Perplexity')
        plt.title('Perplexity Box Plot')
        
        # Add statistics
        mean_ppl = np.mean(perplexities)
        median_ppl = np.median(perplexities)
        plt.figtext(0.5, 0.02, f'Mean: {mean_ppl:.2f}, Median: {median_ppl:.2f}', 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Perplexity plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()