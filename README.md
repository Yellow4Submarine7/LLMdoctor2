# LLMDoctor: Token-level Flow-guided Preference Optimization

Official implementation of the LLMDoctor framework for aligning Large Language Models (LLMs) with human preferences using token-level rewards and flow-based optimization.

## Overview

LLMDoctor is a novel preference optimization framework that:
- Uses **token-level rewards** instead of trajectory-level rewards for fine-grained preference modeling
- Employs a **patient-doctor paradigm** where a small trainable "doctor" model guides a large frozen "patient" LLM
- Implements **flow-based optimization** (TFPO) for training the doctor model
- Supports **multi-dimensional preferences** (e.g., helpfulness, safety, truthfulness)
- Enables **weak-to-strong guidance** where smaller models can effectively guide much larger models

## Key Features

- ✅ **Token-level Preference Modeling**: Identifies and rewards important tokens rather than entire sequences
- ✅ **Efficient Training**: Only trains a small doctor model while keeping the large patient model frozen
- ✅ **Multi-dimensional Control**: Balance multiple preference dimensions during generation
- ✅ **Flexible Architecture**: Works with any causal language model (GPT, LLaMA, etc.)
- ✅ **Comprehensive Framework**: Includes data processing, training, evaluation, and inference

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA-capable GPU (recommended)

## Quick Start

### 1. Training

Train an LLMDoctor model on the HH-RLHF dataset:

```bash
# Quick training with default settings
python train.py --config hh_rlhf --output-dir ./outputs/my_model

# Or use the example script
python examples/train_hh_rlhf.py
```

### 2. Inference

Generate text with a trained model:

```bash
# Basic generation
python inference.py \
    --doctor-model ./outputs/my_model/final_model \
    --prompt "What are the benefits of exercise?"

# Interactive chat
python inference.py \
    --doctor-model ./outputs/my_model/final_model \
    --interactive
```

### 3. Evaluation

Evaluate model performance:

```bash
python evaluate.py \
    --model-path ./outputs/my_model/final_model \
    --eval-dataset AlpacaEval \
    --output-dir ./evaluation_results
```

## Architecture

### Patient-Doctor Paradigm

```
┌─────────────────┐     ┌─────────────────┐
│  Patient Model  │     │  Doctor Model   │
│   (Frozen LLM)  │ ←── │  (Small, Trained)│
│   e.g., 70B     │     │   e.g., 7B      │
└─────────────────┘     └─────────────────┘
         ↓                       ↓
    Base Probs              Reward Signal
         ↓                       ↓
         └───────────┬───────────┘
                     ↓
              Guided Generation
```

### Token-level Reward Pipeline

1. **Behavioral Variants**: Create positive/negative instruction variants
2. **Token Importance**: Measure importance via probability differences
3. **Reward Assignment**: Assign directional rewards to important tokens
4. **TFPO Training**: Train doctor model using flow-based objectives

## Advanced Usage

### Multi-dimensional Preferences

Train and use models with multiple preference dimensions:

```bash
# Train multi-dimensional model
python train.py --config pku_saferlhf

# Generate with different preference weights
python inference.py \
    --doctor-model ./outputs/multi_dim_model/final_model \
    --preference-weights '{"helpfulness": 0.8, "safety": 0.2}' \
    --prompt "How do I build a website?"
```

### Weak-to-Strong Guidance

Use a small model to guide a much larger one:

```bash
# Train 7B doctor to guide 70B patient
python train.py --config weak_to_strong

# The 7B model can now effectively guide the 70B model!
```

### Custom Datasets

Process your own preference dataset:

```python
from src.reward.reward_processor import RewardDataProcessor, PreferenceExample

# Create preference examples
examples = [
    PreferenceExample(
        prompt="How do I stay healthy?",
        preferred_response="Maintain a balanced diet, exercise regularly...",
        non_preferred_response="Just eat whatever you want."
    )
]

# Process to extract token rewards
processor = RewardDataProcessor(patient_model)
reward_data = processor.process_preference_dataset(examples)
```

## Configuration

### Pre-defined Configurations

- `hh_rlhf`: Standard helpfulness alignment
- `pku_saferlhf`: Multi-dimensional (helpfulness + safety)
- `weak_to_strong`: 7B doctor guiding 70B patient

### Custom Configuration

Create your own YAML configuration:

```yaml
experiment_name: my_experiment
model:
  patient_model_name: meta-llama/Llama-2-7b-hf
  doctor_model_name: meta-llama/Llama-2-7b-hf
  doctor_num_preference_dims: 1
  
training:
  learning_rate: 5.0e-5
  max_steps: 10000
  batch_size: 4
  
# ... see configs/ for more options
```

## Project Structure

```
llm_doctor/
├── src/
│   ├── models/          # Model implementations
│   ├── reward/          # Reward processing pipeline
│   ├── training/        # TFPO training
│   ├── inference/       # Guided generation
│   ├── evaluation/      # Evaluation metrics
│   └── data/           # Data handling
├── configs/            # Experiment configurations
├── examples/           # Example scripts
├── train.py           # Main training script
├── inference.py       # Generation script
└── evaluate.py        # Evaluation script
```

## Experimental Results

LLMDoctor achieves state-of-the-art results on:
- **AlpacaEval**: 88.1% win rate (7B model)
- **Multi-dimensional Control**: Effective balancing of helpfulness and safety
- **Weak-to-Strong**: 7B doctor successfully guides 70B patient

## Additional Information

- Based on the GFlowNet framework for flow-based optimization
- Uses the Hugging Face Transformers library
- Inspired by weak-to-strong generalization research

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in configuration
- Enable gradient checkpointing: `gradient_checkpointing: true`
- Use 8-bit quantization for large models: `patient_load_in_8bit: true`

### Slow Training

- Ensure CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Use mixed precision training: `mixed_precision: true`
- Reduce `max_sequence_length` if possible

### Poor Results

- Ensure sufficient training steps (usually 5k-20k)
- Check that reward sparsity threshold is appropriate (default: 0.1)
- Verify data quality and preference labels


