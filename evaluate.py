"""
Evaluation script for LLMDoctor models.

This script evaluates trained models on various benchmarks and metrics.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
import torch
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.doctor_model import DoctorModel
from src.models.patient_model import PatientModel
from src.inference.guided_generation import GuidedGenerator, InferenceConfig
from src.evaluation.evaluator import LLMEvaluator, EvaluationConfig
from src.data.evaluation_datasets import load_evaluation_dataset
from configs import ExperimentConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_prompts(dataset_name: str, max_samples: Optional[int] = None) -> List[str]:
    """Load test prompts from evaluation dataset."""
    if dataset_name.lower() == "alpacaeval":
        # Load AlpacaEval dataset
        try:
            from datasets import load_dataset
            dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")
            prompts = [item["instruction"] for item in dataset]
        except:
            logger.warning("Could not load AlpacaEval dataset, using sample prompts")
            prompts = [
                "What are the main causes of climate change?",
                "Explain quantum computing in simple terms.",
                "How can I improve my public speaking skills?",
                "What is the difference between machine learning and deep learning?",
                "Write a haiku about artificial intelligence."
            ]
    elif dataset_name.lower() == "custom":
        # Load custom prompts from file
        prompts_file = Path("evaluation_prompts.json")
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                prompts = json.load(f)
        else:
            logger.warning("Custom prompts file not found, using sample prompts")
            prompts = [
                "How do I debug a Python program?",
                "What are best practices for API design?",
                "Explain the concept of recursion with an example.",
                "How can I optimize database queries?",
                "What is the difference between TCP and UDP?"
            ]
    else:
        # Default sample prompts
        prompts = [
            "Explain the benefits of regular exercise.",
            "How does photosynthesis work?",
            "What are the key principles of effective leadership?",
            "Describe the process of making coffee.",
            "What is artificial general intelligence?"
        ]
    
    if max_samples and len(prompts) > max_samples:
        prompts = prompts[:max_samples]
    
    logger.info(f"Loaded {len(prompts)} test prompts")
    return prompts


def load_baseline_responses(
    baseline_names: List[str],
    test_prompts: List[str],
    cache_dir: str = "./baseline_cache"
) -> Dict[str, List[str]]:
    """Load or generate baseline responses."""
    baseline_responses = {}
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for baseline_name in baseline_names:
        cache_file = cache_dir / f"{baseline_name}_responses.json"
        
        if cache_file.exists():
            # Load from cache
            with open(cache_file, 'r') as f:
                responses = json.load(f)
            logger.info(f"Loaded {baseline_name} responses from cache")
        else:
            # Generate mock responses for demonstration
            logger.info(f"Generating mock responses for {baseline_name}")
            
            if baseline_name == "sft":
                responses = [f"This is a standard fine-tuned response to: {prompt}" 
                           for prompt in test_prompts]
            elif baseline_name == "dpo":
                responses = [f"This is a DPO-optimized response addressing: {prompt}" 
                           for prompt in test_prompts]
            elif baseline_name == "ppo":
                responses = [f"This is a PPO-trained response for: {prompt}" 
                           for prompt in test_prompts]
            else:
                responses = [f"This is a {baseline_name} response to: {prompt}" 
                           for prompt in test_prompts]
            
            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(responses, f, indent=2)
        
        baseline_responses[baseline_name] = responses
    
    return baseline_responses


def evaluate_model(
    model_path: str,
    patient_model_name: str,
    eval_config: EvaluationConfig,
    test_prompts: List[str],
    baseline_responses: Optional[Dict[str, List[str]]] = None,
    preference_weights: Optional[Dict[str, float]] = None
):
    """Evaluate a trained LLMDoctor model."""
    logger.info(f"Evaluating model: {model_path}")
    
    # Load models
    logger.info("Loading models...")
    
    # Load patient model
    patient_model = PatientModel(
        model_name=patient_model_name,
        device="auto",
        dtype="float16",
        load_in_8bit=False,
        trust_remote_code=False
    )
    
    # Load doctor model
    doctor_model = DoctorModel.from_pretrained(model_path)
    
    # Create guided generator
    inference_config = InferenceConfig(
        alpha=1.0,
        beta=1.0,
        max_new_tokens=eval_config.max_new_tokens,
        temperature=eval_config.temperature,
        top_p=eval_config.top_p,
        top_k=eval_config.top_k,
        do_sample=eval_config.do_sample
    )
    
    guided_generator = GuidedGenerator(
        patient_model=patient_model,
        doctor_model=doctor_model,
        config=inference_config
    )
    
    # Create evaluator
    evaluator = LLMEvaluator(config=eval_config)
    
    # Evaluate
    results = evaluator.evaluate(
        model=guided_generator,
        test_prompts=test_prompts,
        baseline_responses=baseline_responses,
        preference_weights=preference_weights
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Number of examples: {results.num_examples}")
    print("-"*60)
    
    if results.win_rate is not None:
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Tie Rate: {results.tie_rate:.2%}")
        print(f"Loss Rate: {results.loss_rate:.2%}")
    
    if results.diversity_score is not None:
        print(f"Diversity Score: {results.diversity_score:.3f}")
    
    if results.perplexity is not None:
        print(f"Perplexity: {results.perplexity:.2f}")
    
    if results.avg_length is not None:
        print(f"Average Length: {results.avg_length:.1f} tokens")
    
    if results.dimension_scores:
        print("\nDimension Scores:")
        for dim, score in results.dimension_scores.items():
            print(f"  {dim}: {score:.2f}")
    
    print("="*60)
    
    return results


def compare_models(
    model_paths: List[str],
    patient_model_name: str,
    eval_config: EvaluationConfig,
    test_prompts: List[str]
):
    """Compare multiple LLMDoctor models."""
    logger.info(f"Comparing {len(model_paths)} models")
    
    # Load all models
    models = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).name
        
        # Load patient model
        patient_model = PatientModel(
            model_name=patient_model_name,
            device="auto",
            dtype="float16"
        )
        
        # Load doctor model
        doctor_model = DoctorModel.from_pretrained(model_path)
        
        # Create guided generator
        inference_config = InferenceConfig(
            alpha=1.0,
            beta=1.0,
            max_new_tokens=eval_config.max_new_tokens,
            temperature=eval_config.temperature,
            top_p=eval_config.top_p,
            do_sample=eval_config.do_sample
        )
        
        guided_generator = GuidedGenerator(
            patient_model=patient_model,
            doctor_model=doctor_model,
            config=inference_config
        )
        
        models[model_name] = guided_generator
    
    # Create evaluator and compare
    evaluator = LLMEvaluator(config=eval_config)
    comparison_results = evaluator.compare_models(
        models=models,
        test_prompts=test_prompts
    )
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLMDoctor models")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained doctor model"
    )
    parser.add_argument(
        "--patient-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Patient model name"
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="AlpacaEval",
        help="Evaluation dataset (AlpacaEval, custom, or sample)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples"
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["sft", "dpo", "ppo"],
        help="Baseline methods to compare against"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--preference-weights",
        type=str,
        default=None,
        help="JSON string with preference weights for multi-dimensional models"
    )
    parser.add_argument(
        "--compare-models",
        type=str,
        nargs="+",
        default=None,
        help="Additional model paths for comparison"
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline comparisons"
    )
    parser.add_argument(
        "--compute-perplexity",
        action="store_true",
        help="Compute perplexity (requires additional model)"
    )
    parser.add_argument(
        "--gpt4-api-key",
        type=str,
        default=None,
        help="OpenAI API key for GPT-4 evaluation"
    )
    
    args = parser.parse_args()
    
    # Parse preference weights if provided
    preference_weights = None
    if args.preference_weights:
        try:
            preference_weights = json.loads(args.preference_weights)
        except:
            logger.warning("Could not parse preference weights, using defaults")
    
    # Create evaluation config
    eval_config = EvaluationConfig(
        eval_dataset=args.eval_dataset,
        eval_max_samples=args.max_samples,
        eval_method="gpt4" if args.gpt4_api_key else "mock",
        gpt4_api_key=args.gpt4_api_key,
        compute_diversity=True,
        compute_perplexity=args.compute_perplexity,
        compute_win_rate=not args.skip_baselines,
        baseline_methods=args.baselines,
        results_dir=args.output_dir,
        save_generations=True,
        save_detailed_results=True
    )
    
    # Load test prompts
    test_prompts = load_test_prompts(
        dataset_name=args.eval_dataset,
        max_samples=args.max_samples
    )
    
    # Load baseline responses if needed
    baseline_responses = None
    if not args.skip_baselines:
        baseline_responses = load_baseline_responses(
            baseline_names=args.baselines,
            test_prompts=test_prompts
        )
    
    # Evaluate single model or compare multiple
    if args.compare_models:
        # Compare multiple models
        all_models = [args.model_path] + args.compare_models
        comparison_results = compare_models(
            model_paths=all_models,
            patient_model_name=args.patient_model,
            eval_config=eval_config,
            test_prompts=test_prompts
        )
        
        logger.info(f"Comparison results saved to {eval_config.results_dir}")
    else:
        # Evaluate single model
        results = evaluate_model(
            model_path=args.model_path,
            patient_model_name=args.patient_model,
            eval_config=eval_config,
            test_prompts=test_prompts,
            baseline_responses=baseline_responses,
            preference_weights=preference_weights
        )
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()