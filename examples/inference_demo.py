"""
Example: Generate text with a trained LLMDoctor model

This example shows different ways to use a trained doctor model
for guided text generation.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Demonstrate inference with LLMDoctor."""
    
    print("=" * 60)
    print("LLMDoctor Inference Examples")
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
    
    # Example prompts
    prompts = [
        "What are the benefits of meditation?",
        "How can I learn to code effectively?",
        "Explain the theory of relativity in simple terms.",
        "What should I consider when adopting a pet?",
        "How do I start a vegetable garden?"
    ]
    
    while True:
        print("\nSelect an example:")
        print("1. Basic generation")
        print("2. Compare with baseline (no guidance)")
        print("3. Interactive chat")
        print("4. Adjust guidance strength")
        print("5. Custom prompt")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-5): ").strip()
        
        if choice == "0":
            break
            
        elif choice == "1":
            # Basic generation
            print("\n--- Basic Generation ---")
            prompt = prompts[0]
            
            cmd = [
                sys.executable,
                "inference.py",
                "--doctor-model", model_path,
                "--prompt", prompt,
                "--max-new-tokens", "100"
            ]
            
            print(f"\nPrompt: {prompt}")
            print("\nGenerating response...")
            subprocess.run(cmd)
            
        elif choice == "2":
            # Compare with baseline
            print("\n--- Comparison with Baseline ---")
            prompt = prompts[1]
            
            cmd = [
                sys.executable,
                "inference.py",
                "--doctor-model", model_path,
                "--prompt", prompt,
                "--compare-baseline",
                "--max-new-tokens", "100"
            ]
            
            subprocess.run(cmd)
            
        elif choice == "3":
            # Interactive chat
            print("\n--- Interactive Chat ---")
            print("Starting interactive mode...")
            print("(Use /quit to exit)")
            
            cmd = [
                sys.executable,
                "inference.py",
                "--doctor-model", model_path,
                "--interactive"
            ]
            
            subprocess.run(cmd)
            
        elif choice == "4":
            # Adjust guidance strength
            print("\n--- Guidance Strength Comparison ---")
            prompt = prompts[2]
            
            print(f"\nPrompt: {prompt}")
            
            for beta in [0.0, 0.5, 1.0, 2.0]:
                print(f"\n{'='*40}")
                print(f"Beta = {beta} (guidance strength)")
                print('='*40)
                
                cmd = [
                    sys.executable,
                    "inference.py",
                    "--doctor-model", model_path,
                    "--prompt", prompt,
                    "--beta", str(beta),
                    "--max-new-tokens", "50"
                ]
                
                subprocess.run(cmd)
            
        elif choice == "5":
            # Custom prompt
            print("\n--- Custom Prompt ---")
            prompt = input("Enter your prompt: ").strip()
            if not prompt:
                continue
            
            # Get parameters
            max_tokens = input("Max tokens (default 100): ").strip() or "100"
            temperature = input("Temperature (default 1.0): ").strip() or "1.0"
            beta = input("Guidance strength beta (default 1.0): ").strip() or "1.0"
            
            cmd = [
                sys.executable,
                "inference.py",
                "--doctor-model", model_path,
                "--prompt", prompt,
                "--max-new-tokens", max_tokens,
                "--temperature", temperature,
                "--beta", beta
            ]
            
            print("\nGenerating response...")
            subprocess.run(cmd)
            
        else:
            print("Invalid choice. Please try again.")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()