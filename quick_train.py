"""
Quick training script with preset configurations for LLMDoctor experiments.

Example usage:
    # Train with HH-RLHF dataset
    python quick_train.py --experiment hh_rlhf
    
    # Train with PKU-SafeRLHF for multi-dimensional preferences
    python quick_train.py --experiment pku_saferlhf --gpus 2
    
    # Weak-to-strong guidance experiment
    python quick_train.py --experiment weak_to_strong --debug
"""

import subprocess
import sys
import argparse
from pathlib import Path


EXPERIMENTS = {
    "hh_rlhf": {
        "config": "hh_rlhf",
        "description": "Standard helpfulness alignment with HH-RLHF dataset",
        "default_args": {
            "max_samples": 50000,
            "batch_size": 4,
            "max_steps": 10000
        }
    },
    "pku_saferlhf": {
        "config": "pku_saferlhf", 
        "description": "Multi-dimensional alignment (helpfulness + safety)",
        "default_args": {
            "max_samples": 10000,
            "batch_size": 4,
            "max_steps": 15000
        }
    },
    "weak_to_strong": {
        "config": "weak_to_strong",
        "description": "7B doctor model guiding 70B patient model",
        "default_args": {
            "max_samples": 20000,
            "batch_size": 2,
            "max_steps": 5000
        }
    },
    "debug": {
        "config": "hh_rlhf",
        "description": "Debug configuration with small dataset",
        "default_args": {
            "max_samples": 100,
            "batch_size": 2,
            "max_steps": 50,
            "debug": True
        }
    }
}


def main():
    parser = argparse.ArgumentParser(
        description="Quick training script for LLMDoctor experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add experiment selection
    parser.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        required=True,
        help="Preset experiment to run"
    )
    
    # Training arguments
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--skip-reward", action="store_true", help="Skip reward processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Override arguments
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--max-samples", type=int, default=None, help="Override max samples")
    
    args = parser.parse_args()
    
    # Get experiment configuration
    experiment = EXPERIMENTS[args.experiment]
    
    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Description: {experiment['description']}")
    print(f"{'='*60}\n")
    
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--config", experiment["config"],
        "--seed", str(args.seed),
    ]
    
    # Add default arguments
    defaults = experiment["default_args"]
    
    # Override with command line arguments
    if args.batch_size is not None:
        defaults["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        defaults["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        defaults["max_steps"] = args.max_steps
    if args.max_samples is not None:
        defaults["max_samples"] = args.max_samples
    
    # Add arguments to command
    for key, value in defaults.items():
        if key == "debug" and value:
            cmd.append("--debug")
        else:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))
    
    # Add optional arguments
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    else:
        # Create default output directory
        output_dir = f"./outputs/{args.experiment}_exp"
        cmd.extend(["--output-dir", output_dir])
    
    if args.resume:
        cmd.extend(["--resume", args.resume])
    
    if args.skip_reward:
        cmd.append("--skip-reward-processing")
    
    if args.debug:
        cmd.append("--debug")
    
    # Handle multi-GPU training
    if args.gpus > 1:
        # Use torchrun for distributed training
        torchrun_cmd = [
            "torchrun",
            "--nproc_per_node", str(args.gpus),
            "--master_port", "29500",
        ] + cmd
        cmd = torchrun_cmd
    
    # Print command
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Execute
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print(f"Experiment {args.experiment} completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()