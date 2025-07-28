"""
Inference script for LLMDoctor models.

This script demonstrates how to use trained doctor models for guided text generation.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompts(prompt_file: Optional[str] = None) -> List[str]:
    """Load prompts from file or return default prompts."""
    if prompt_file and Path(prompt_file).exists():
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {prompt_file}")
    else:
        # Default prompts for demonstration
        prompts = [
            "What are the benefits of regular exercise?",
            "How can I improve my programming skills?",
            "Explain quantum computing in simple terms.",
            "What should I consider when buying a new laptop?",
            "How do I prepare for a job interview?",
            "What are the main causes of climate change?",
            "Can you recommend strategies for better time management?",
            "How does machine learning differ from traditional programming?",
            "What are the key principles of effective communication?",
            "Explain the concept of blockchain technology."
        ]
        logger.info(f"Using {len(prompts)} default prompts")
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate text with LLMDoctor models")
    
    # Model arguments
    parser.add_argument(
        "--doctor-model",
        type=str,
        required=True,
        help="Path to trained doctor model"
    )
    parser.add_argument(
        "--patient-model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Patient model name or path"
    )
    
    # Generation arguments
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )
    
    # Guidance arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for patient model (fluency)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight for doctor model (alignment)"
    )
    parser.add_argument(
        "--preference-dim",
        type=int,
        default=0,
        help="Preference dimension to use"
    )
    parser.add_argument(
        "--preference-weights",
        type=str,
        default=None,
        help="JSON string with multi-dimensional preference weights"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save generations"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare with baseline (no guidance)"
    )
    parser.add_argument(
        "--compare-dimensions",
        action="store_true",
        help="Compare across preference dimensions"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive chat"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed generation info"
    )
    
    args = parser.parse_args()
    
    # Load models
    logger.info("Loading models...")
    
    patient_model = PatientModel(
        model_name=args.patient_model,
        device="auto",
        dtype="float16"
    )
    
    doctor_model = DoctorModel.from_pretrained(args.doctor_model)
    
    # Create inference config
    inference_config = InferenceConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        return_detailed_info=args.verbose,
        save_generations=args.output is not None,
        output_file=args.output
    )
    
    # Parse preference weights if provided
    preference_weights = None
    if args.preference_weights:
        try:
            preference_weights = json.loads(args.preference_weights)
            logger.info(f"Using preference weights: {preference_weights}")
        except:
            logger.warning("Could not parse preference weights, using single dimension")
    
    # Create guided generator
    generator = GuidedGenerator(
        patient_model=patient_model,
        doctor_model=doctor_model,
        config=inference_config
    )
    
    logger.info(f"Initialized generator with {doctor_model.num_preference_dims} preference dimensions")
    
    # Run selected mode
    if args.interactive:
        # Interactive chat mode
        generator.interactive_chat()
        
    elif args.compare_dimensions:
        # Compare across dimensions
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = load_prompts(args.prompts)[:5]  # Limit for comparison
        
        for prompt in prompts:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print('='*60)
            
            results = generator.compare_dimensions(prompt)
            for dim_name, response in results.items():
                print(f"\n{dim_name}:")
                print(response)
        
    elif args.compare_baseline:
        # Compare with baseline
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = load_prompts(args.prompts)[:5]  # Limit for comparison
        
        for prompt in prompts:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print('='*60)
            
            # Generate with guidance
            print("\nWith guidance (α={}, β={}):".format(args.alpha, args.beta))
            guided_response = generator.generate(
                prompt=prompt,
                preference_dim=args.preference_dim,
                preference_weights=preference_weights
            )
            if isinstance(guided_response, dict):
                print(guided_response['response'])
            else:
                print(guided_response)
            
            # Generate without guidance
            print(f"\nWithout guidance (baseline):")
            generator.config.beta = 0.0
            baseline_response = generator.generate(
                prompt=prompt,
                preference_dim=args.preference_dim
            )
            generator.config.beta = args.beta  # Restore
            
            if isinstance(baseline_response, dict):
                print(baseline_response['response'])
            else:
                print(baseline_response)
    
    else:
        # Standard generation
        if args.prompt:
            prompts = [args.prompt]
        else:
            prompts = load_prompts(args.prompts)
        
        responses = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating response {i+1}/{len(prompts)}")
            
            response = generator.generate(
                prompt=prompt,
                preference_dim=args.preference_dim,
                preference_weights=preference_weights
            )
            
            if args.verbose:
                # Detailed output
                print(f"\n{'='*60}")
                print(f"Prompt {i+1}: {prompt}")
                print('-'*60)
                if isinstance(response, dict):
                    print(f"Response: {response['response']}")
                    print(f"Tokens generated: {response.get('num_generated_tokens', 'N/A')}")
                    if 'token_details' in response:
                        print("\nToken details (first 10):")
                        for j, token_info in enumerate(response['token_details'][:10]):
                            print(f"  {j}: '{token_info['token']}' - "
                                  f"patient: {token_info['patient_prob']:.3f}, "
                                  f"reward: {token_info['reward_prob']:.3f}, "
                                  f"combined: {token_info['combined_prob']:.3f}")
                else:
                    print(f"Response: {response}")
            else:
                # Simple output
                print(f"\nPrompt: {prompt}")
                print(f"Response: {response if isinstance(response, str) else response['response']}")
            
            responses.append({
                "prompt": prompt,
                "response": response if isinstance(response, str) else response['response'],
                "preference_dim": args.preference_dim,
                "preference_weights": preference_weights
            })
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(responses)} responses to {output_path}")


if __name__ == "__main__":
    main()