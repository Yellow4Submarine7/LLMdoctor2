"""
Example: Train multi-dimensional LLMDoctor on PKU-SafeRLHF

This example demonstrates how to train a multi-dimensional
doctor model that can balance helpfulness and safety.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run multi-dimensional training on PKU-SafeRLHF dataset."""
    
    print("=" * 60)
    print("LLMDoctor Training Example: Multi-Dimensional Preferences")
    print("=" * 60)
    print()
    print("This example will train a doctor model with:")
    print("- 2 preference dimensions (helpfulness & safety)")
    print("- PKU-SafeRLHF dataset with multi-dimensional labels")
    print("- Ability to control generation along both dimensions")
    print()
    
    # Configuration
    config = "pku_saferlhf"  # Multi-dimensional config
    output_dir = "./outputs/pku_multi_dim_example"
    max_samples = 500   # Smaller dataset
    max_steps = 300     # Fewer steps for demo
    
    # Build training command
    cmd = [
        sys.executable,
        "train.py",
        "--config", config,
        "--output-dir", output_dir,
        "--max-samples", str(max_samples),
        "--max-steps", str(max_steps),
        "--batch-size", "2",  # Smaller batch for memory
        "--learning-rate", "3e-5",
        "--seed", "42"
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    # Confirm before running
    response = input("Start multi-dimensional training? (y/n): ")
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
        print("Multi-dimensional training completed!")
        print(f"Model saved to: {output_dir}")
        print()
        print("Example usage with different preference weights:")
        print()
        print("1. Maximize helpfulness:")
        print(f'   python inference.py --doctor-model {output_dir}/final_model \\')
        print('                       --preference-weights \'{"helpfulness": 1.0, "safety": 0.0}\' \\')
        print('                       --prompt "How to make a bomb?"')
        print()
        print("2. Maximize safety:")
        print(f'   python inference.py --doctor-model {output_dir}/final_model \\')
        print('                       --preference-weights \'{"helpfulness": 0.0, "safety": 1.0}\' \\')
        print('                       --prompt "How to make a bomb?"')
        print()
        print("3. Balance both:")
        print(f'   python inference.py --doctor-model {output_dir}/final_model \\')
        print('                       --preference-weights \'{"helpfulness": 0.5, "safety": 0.5}\' \\')
        print('                       --prompt "How to make a bomb?"')
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code {e.returncode}")
        print("Check the logs above for details.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")


if __name__ == "__main__":
    main()