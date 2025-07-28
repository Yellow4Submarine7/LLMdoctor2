"""
Example: Evaluate a trained LLMDoctor model

This example demonstrates how to evaluate model performance
using various metrics and comparisons.
"""

import subprocess
import sys
from pathlib import Path
import json

def main():
    """Run evaluation on a trained model."""
    
    print("=" * 60)
    print("LLMDoctor Evaluation Example")
    print("=" * 60)
    print()
    
    # Check if model exists
    model_path = "./outputs/hh_rlhf_example/final_model"
    if not Path(model_path).exists():
        print("No trained model found!")
        print("Please run train_hh_rlhf.py first to train a model.")
        print()
        print("Alternatively, you can specify a different model path:")
        model_path = input("Enter model path (or press Enter to exit): ").strip()
        if not model_path:
            return
    
    print(f"Using model: {model_path}")
    print()
    
    # Evaluation options
    print("Select evaluation type:")
    print("1. Quick evaluation (10 samples)")
    print("2. Standard evaluation (100 samples)")
    print("3. Compare with baselines")
    print("4. Diversity analysis")
    print("5. Multi-dimensional evaluation (if applicable)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    output_dir = "./evaluation_results/example"
    
    if choice == "1":
        # Quick evaluation
        print("\n--- Quick Evaluation ---")
        print("This will evaluate on 10 sample prompts...")
        
        cmd = [
            sys.executable,
            "evaluate.py",
            "--model-path", model_path,
            "--eval-dataset", "sample",
            "--max-samples", "10",
            "--output-dir", output_dir,
            "--skip-baselines"
        ]
        
        print("\nRunning evaluation...")
        subprocess.run(cmd)
        
        # Show results
        results_file = Path(output_dir) / "evaluation_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print("\n" + "="*40)
            print("Evaluation Results:")
            print("="*40)
            print(f"Number of examples: {results.get('num_examples', 'N/A')}")
            print(f"Average length: {results.get('avg_length', 'N/A'):.1f} tokens")
            print(f"Diversity score: {results.get('diversity_score', 'N/A'):.3f}")
    
    elif choice == "2":
        # Standard evaluation
        print("\n--- Standard Evaluation ---")
        print("This will evaluate on 100 samples...")
        
        cmd = [
            sys.executable,
            "evaluate.py",
            "--model-path", model_path,
            "--eval-dataset", "custom",
            "--max-samples", "100",
            "--output-dir", output_dir,
            "--compute-perplexity"
        ]
        
        print("\nRunning evaluation...")
        subprocess.run(cmd)
    
    elif choice == "3":
        # Compare with baselines
        print("\n--- Baseline Comparison ---")
        print("This will compare with SFT, DPO, and PPO baselines...")
        
        cmd = [
            sys.executable,
            "evaluate.py",
            "--model-path", model_path,
            "--eval-dataset", "sample",
            "--max-samples", "50",
            "--baselines", "sft", "dpo", "ppo",
            "--output-dir", output_dir
        ]
        
        print("\nRunning comparison...")
        subprocess.run(cmd)
    
    elif choice == "4":
        # Diversity analysis
        print("\n--- Diversity Analysis ---")
        print("Generating multiple responses per prompt to analyze diversity...")
        
        # First generate multiple samples
        prompts = [
            "Tell me a story about a robot.",
            "What's your favorite color and why?",
            "Describe a perfect day.",
            "What advice would you give to your younger self?",
            "If you could have any superpower, what would it be?"
        ]
        
        all_responses = []
        
        for prompt in prompts:
            print(f"\nGenerating 5 responses for: {prompt[:50]}...")
            
            responses = []
            for i in range(5):
                cmd = [
                    sys.executable,
                    "inference.py",
                    "--doctor-model", model_path,
                    "--prompt", prompt,
                    "--max-new-tokens", "50",
                    "--temperature", "1.2",  # Higher temperature for diversity
                    "--top-p", "0.95"
                ]
                
                # Capture output
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                
                # Extract response from output
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith("Response:"):
                        response = line[9:].strip()
                        responses.append(response)
                        break
            
            all_responses.extend(responses)
        
        # Simple diversity metrics
        unique_responses = len(set(all_responses))
        total_responses = len(all_responses)
        
        print("\n" + "="*40)
        print("Diversity Analysis Results:")
        print("="*40)
        print(f"Total responses: {total_responses}")
        print(f"Unique responses: {unique_responses}")
        print(f"Diversity ratio: {unique_responses/total_responses:.2%}")
        
        # Show some examples
        print("\nSample responses:")
        for i, resp in enumerate(all_responses[:5]):
            print(f"{i+1}. {resp[:100]}...")
    
    elif choice == "5":
        # Multi-dimensional evaluation
        print("\n--- Multi-Dimensional Evaluation ---")
        
        # Check if model is multi-dimensional
        try:
            # Load model info
            doctor_model_info = Path(model_path) / "config.json"
            if doctor_model_info.exists():
                with open(doctor_model_info, 'r') as f:
                    config = json.load(f)
                num_dims = config.get('num_preference_dims', 1)
                
                if num_dims > 1:
                    print(f"Model has {num_dims} preference dimensions")
                    print("\nEvaluating with different preference weights...")
                    
                    # Test different weight combinations
                    weight_configs = [
                        '{"helpfulness": 1.0, "safety": 0.0}',
                        '{"helpfulness": 0.5, "safety": 0.5}',
                        '{"helpfulness": 0.0, "safety": 1.0}'
                    ]
                    
                    for weights in weight_configs:
                        print(f"\nWeights: {weights}")
                        
                        cmd = [
                            sys.executable,
                            "evaluate.py",
                            "--model-path", model_path,
                            "--eval-dataset", "sample",
                            "--max-samples", "20",
                            "--preference-weights", weights,
                            "--output-dir", f"{output_dir}/weights_{weights.replace(' ', '').replace(':', '_')}",
                            "--skip-baselines"
                        ]
                        
                        subprocess.run(cmd)
                else:
                    print("Model is single-dimensional. Skipping multi-dimensional evaluation.")
            else:
                print("Could not determine model dimensions.")
        except Exception as e:
            print(f"Error checking model configuration: {e}")
    
    else:
        print("Invalid choice.")
        return
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()