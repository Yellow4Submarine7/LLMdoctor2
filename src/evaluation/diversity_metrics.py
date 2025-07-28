"""
Diversity metrics for evaluating text generation quality.

Implements various diversity metrics including:
- Distinct-n
- Entropy
- Self-BLEU
- Unique n-grams
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter, defaultdict
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DiversityMetrics:
    """
    Computes various diversity metrics for generated text.
    
    Diversity is important for evaluating whether the model
    generates varied and interesting responses.
    """
    
    def __init__(self, metric_type: str = "distinct-4"):
        """
        Initialize diversity metrics calculator.
        
        Args:
            metric_type: Type of diversity metric to use
        """
        self.metric_type = metric_type
        self.smoothing = SmoothingFunction()
        
        logger.info(f"DiversityMetrics initialized with {metric_type}")
    
    def compute(
        self,
        texts: List[str],
        return_all_metrics: bool = False
    ) -> Union[float, Dict[str, float]]:
        """
        Compute diversity metrics for a list of texts.
        
        Args:
            texts: List of generated texts
            return_all_metrics: Whether to return all metrics
            
        Returns:
            Diversity score or dictionary of metrics
        """
        if not texts:
            return 0.0 if not return_all_metrics else {}
        
        # Tokenize texts
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        if return_all_metrics:
            metrics = {
                "distinct-1": self._compute_distinct_n(tokenized_texts, n=1),
                "distinct-2": self._compute_distinct_n(tokenized_texts, n=2),
                "distinct-3": self._compute_distinct_n(tokenized_texts, n=3),
                "distinct-4": self._compute_distinct_n(tokenized_texts, n=4),
                "entropy-1": self._compute_entropy(tokenized_texts, n=1),
                "entropy-2": self._compute_entropy(tokenized_texts, n=2),
                "entropy-3": self._compute_entropy(tokenized_texts, n=3),
                "entropy-4": self._compute_entropy(tokenized_texts, n=4),
                "self_bleu": self._compute_self_bleu(tokenized_texts),
                "unique_tokens": self._compute_unique_ratio(tokenized_texts, n=1),
                "unique_bigrams": self._compute_unique_ratio(tokenized_texts, n=2),
                "unique_trigrams": self._compute_unique_ratio(tokenized_texts, n=3),
                "msttr": self._compute_msttr(tokenized_texts),
                "avg_length": np.mean([len(tokens) for tokens in tokenized_texts])
            }
            return metrics
        else:
            # Return single metric based on type
            if self.metric_type.startswith("distinct-"):
                n = int(self.metric_type.split("-")[1])
                return self._compute_distinct_n(tokenized_texts, n=n)
            elif self.metric_type.startswith("entropy-"):
                n = int(self.metric_type.split("-")[1])
                return self._compute_entropy(tokenized_texts, n=n)
            elif self.metric_type == "self-bleu":
                return self._compute_self_bleu(tokenized_texts)
            elif self.metric_type == "msttr":
                return self._compute_msttr(tokenized_texts)
            else:
                logger.warning(f"Unknown metric type: {self.metric_type}, using distinct-4")
                return self._compute_distinct_n(tokenized_texts, n=4)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple tokenization - can be replaced with more sophisticated tokenizer
        tokens = text.lower().split()
        # Remove punctuation from tokens
        tokens = [token.strip('.,!?;:"') for token in tokens]
        return [token for token in tokens if token]
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _compute_distinct_n(self, tokenized_texts: List[List[str]], n: int) -> float:
        """
        Compute distinct-n metric.
        
        Distinct-n = (# of unique n-grams) / (# of total n-grams)
        """
        all_ngrams = []
        for tokens in tokenized_texts:
            ngrams = self._get_ngrams(tokens, n)
            all_ngrams.extend(ngrams)
        
        if not all_ngrams:
            return 0.0
        
        unique_ngrams = set(all_ngrams)
        return len(unique_ngrams) / len(all_ngrams)
    
    def _compute_entropy(self, tokenized_texts: List[List[str]], n: int) -> float:
        """
        Compute entropy of n-gram distribution.
        
        Higher entropy indicates more diverse text.
        """
        ngram_counts = Counter()
        total_ngrams = 0
        
        for tokens in tokenized_texts:
            ngrams = self._get_ngrams(tokens, n)
            ngram_counts.update(ngrams)
            total_ngrams += len(ngrams)
        
        if total_ngrams == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in ngram_counts.values():
            prob = count / total_ngrams
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _compute_self_bleu(
        self,
        tokenized_texts: List[List[str]],
        max_n: int = 4
    ) -> float:
        """
        Compute Self-BLEU score.
        
        Lower Self-BLEU indicates more diverse text.
        """
        if len(tokenized_texts) < 2:
            return 0.0
        
        bleu_scores = []
        
        for i, hypothesis in enumerate(tokenized_texts):
            # Use all other texts as references
            references = tokenized_texts[:i] + tokenized_texts[i+1:]
            
            # Compute BLEU score
            if hypothesis and references:
                # Calculate BLEU with different n-gram weights
                weights = []
                for n in range(1, min(max_n + 1, len(hypothesis) + 1)):
                    weight = tuple([1.0/n] * n + [0.0] * (max_n - n))
                    weights.append(weight)
                
                if weights:
                    scores = []
                    for weight in weights:
                        try:
                            score = sentence_bleu(
                                references,
                                hypothesis,
                                weights=weight,
                                smoothing_function=self.smoothing.method1
                            )
                            scores.append(score)
                        except:
                            scores.append(0.0)
                    
                    bleu_score = np.mean(scores)
                    bleu_scores.append(bleu_score)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def _compute_unique_ratio(
        self,
        tokenized_texts: List[List[str]],
        n: int
    ) -> float:
        """
        Compute ratio of unique n-grams to total possible n-grams.
        """
        unique_ngrams = set()
        
        for tokens in tokenized_texts:
            ngrams = self._get_ngrams(tokens, n)
            unique_ngrams.update(ngrams)
        
        # Calculate total possible n-grams
        vocab = set()
        for tokens in tokenized_texts:
            vocab.update(tokens)
        
        max_possible = len(vocab) ** n if n <= len(vocab) else 0
        
        if max_possible == 0:
            return 0.0
        
        return len(unique_ngrams) / max_possible
    
    def _compute_msttr(
        self,
        tokenized_texts: List[List[str]],
        window_size: int = 100
    ) -> float:
        """
        Compute Mean Segmental Type-Token Ratio (MSTTR).
        
        MSTTR is more robust than simple TTR for texts of different lengths.
        """
        sttr_scores = []
        
        for tokens in tokenized_texts:
            if len(tokens) < window_size:
                # For short texts, use simple TTR
                if tokens:
                    ttr = len(set(tokens)) / len(tokens)
                    sttr_scores.append(ttr)
            else:
                # Calculate TTR for each window
                window_ttrs = []
                for i in range(0, len(tokens) - window_size + 1):
                    window = tokens[i:i + window_size]
                    ttr = len(set(window)) / len(window)
                    window_ttrs.append(ttr)
                
                if window_ttrs:
                    sttr_scores.append(np.mean(window_ttrs))
        
        return np.mean(sttr_scores) if sttr_scores else 0.0
    
    def compute_corpus_statistics(
        self,
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Compute comprehensive corpus-level statistics.
        
        Args:
            texts: List of generated texts
            
        Returns:
            Dictionary of corpus statistics
        """
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # Vocabulary statistics
        all_tokens = []
        for tokens in tokenized_texts:
            all_tokens.extend(tokens)
        
        vocab_size = len(set(all_tokens))
        total_tokens = len(all_tokens)
        
        # Length statistics
        lengths = [len(tokens) for tokens in tokenized_texts]
        
        # N-gram statistics
        ngram_stats = {}
        for n in range(1, 5):
            all_ngrams = []
            for tokens in tokenized_texts:
                ngrams = self._get_ngrams(tokens, n)
                all_ngrams.extend(ngrams)
            
            if all_ngrams:
                ngram_stats[f"{n}-gram_unique"] = len(set(all_ngrams))
                ngram_stats[f"{n}-gram_total"] = len(all_ngrams)
                ngram_stats[f"{n}-gram_ratio"] = len(set(all_ngrams)) / len(all_ngrams)
        
        stats = {
            "num_texts": len(texts),
            "vocab_size": vocab_size,
            "total_tokens": total_tokens,
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "type_token_ratio": vocab_size / total_tokens if total_tokens > 0 else 0,
            **ngram_stats
        }
        
        return stats
    
    def plot_diversity_metrics(
        self,
        texts_dict: Dict[str, List[str]],
        save_path: Optional[str] = None
    ):
        """
        Plot diversity metrics comparison for multiple models.
        
        Args:
            texts_dict: Dictionary mapping model names to generated texts
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available for plotting")
            return
        
        # Compute metrics for each model
        metrics_data = defaultdict(dict)
        
        for model_name, texts in texts_dict.items():
            metrics = self.compute(texts, return_all_metrics=True)
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data[metric_name][model_name] = value
        
        # Create subplots
        num_metrics = len(metrics_data)
        fig, axes = plt.subplots(
            nrows=(num_metrics + 2) // 3,
            ncols=3,
            figsize=(15, 5 * ((num_metrics + 2) // 3))
        )
        axes = axes.flatten() if num_metrics > 3 else [axes]
        
        # Plot each metric
        for idx, (metric_name, model_values) in enumerate(metrics_data.items()):
            if idx < len(axes):
                ax = axes[idx]
                models = list(model_values.keys())
                values = list(model_values.values())
                
                ax.bar(models, values)
                ax.set_title(metric_name.replace("_", " ").title())
                ax.set_ylabel("Score")
                ax.set_xticklabels(models, rotation=45, ha='right')
                
                # Add value labels on bars
                for i, v in enumerate(values):
                    ax.text(i, v + max(values) * 0.01, f'{v:.3f}', ha='center')
        
        # Remove empty subplots
        for idx in range(len(metrics_data), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Diversity metrics plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compute_pairwise_diversity(
        self,
        texts: List[str],
        sample_size: Optional[int] = None
    ) -> float:
        """
        Compute average pairwise diversity between texts.
        
        This measures how different texts are from each other.
        
        Args:
            texts: List of texts
            sample_size: Number of pairs to sample (for efficiency)
            
        Returns:
            Average pairwise diversity score
        """
        if len(texts) < 2:
            return 0.0
        
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # Sample pairs if needed
        if sample_size and len(texts) > sample_size:
            import random
            indices = list(range(len(texts)))
            sampled_pairs = []
            for _ in range(sample_size):
                i, j = random.sample(indices, 2)
                sampled_pairs.append((i, j))
        else:
            # All pairs
            sampled_pairs = [(i, j) for i in range(len(texts)) 
                           for j in range(i+1, len(texts))]
        
        diversity_scores = []
        
        for i, j in sampled_pairs:
            tokens_i = set(tokenized_texts[i])
            tokens_j = set(tokenized_texts[j])
            
            if tokens_i or tokens_j:
                # Jaccard distance
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                jaccard_dist = 1 - (intersection / union) if union > 0 else 0
                diversity_scores.append(jaccard_dist)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0