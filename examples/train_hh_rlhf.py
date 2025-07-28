"""
Example: Train LLMDoctor on HH-RLHF dataset

This example demonstrates how to train an LLMDoctor model
on the Anthropic HH-RLHF dataset for helpfulness alignment.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run training on HH-RLHF dataset."""
    
    print("=" * 60)
    print("LLMDoctor Training Example: HH-RLHF")
    print("=" * 60)
    print()
    print("This example will:")
    print("1. Load the HH-RLHF preference dataset")
    print("2. Extract token-level rewards using behavioral variants")
    print("3. Train a doctor model with TFPO")
    print("4. Save the trained model")
    print()
    
    # Configuration
    config = "hh_rlhf"  # Use pre-defined configuration
    output_dir = "./outputs/hh_rlhf_example"
    max_samples = 1000  # Use subset for quick demo
    max_steps = 500     # Fewer steps for demo
    
    # Build training command
    cmd = [
        sys.executable,
        "train.py",
        "--config", config,
        "--output-dir", output_dir,
        "--max-samples", str(max_samples),
        "--max-steps", str(max_steps),
        "--batch-size", "4",
        "--learning-rate", "5e-5",
        "--seed", "42"
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    # Confirm before running
    response = input("Start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print("\nStarting training...")
    print("-" * 60)
    
    try:
        # Run training
        subprocess.run(cmd, check=True)
        
        print()
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Model saved to: {output_dir}")
        print()
        print("Next steps:")
        print(f"1. Evaluate: python evaluate.py --model-path {output_dir}/final_model")
        print(f"2. Generate: python inference.py --doctor-model {output_dir}/final_model --prompt 'Your question here'")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        print("Check the logs above for details.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")


if __name__ == "__main__":
    main()