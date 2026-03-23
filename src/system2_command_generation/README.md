# System 2: Command Generation

This module trains and evaluates models that generate OS-aware **CommandPlan** objects.

> ⚠️ **Note**: This module does NOT execute commands. It only generates command plans.

## Overview

The Command Generation system uses fine-tuned language models to generate shell commands that are:

- **OS-aware**: Generates appropriate commands for Windows, Linux, or macOS
- **Shell-specific**: Produces valid syntax for PowerShell, Bash, Cmd, or Zsh
- **Structured**: Outputs valid JSON with sequential execution steps

## Status

Implemented:
- Training and evaluation pipeline for CodeT5+ and FLAN-T5
- Dataset preprocessing from command-dataset schema
- Metric suite for match, preservation, validity, and compatibility checks

Pending:
- Dedicated production inference service wrapper
- Cross-platform command dry-run harness integration

## Architecture

### Training Time
Uses the **command-dataset** directly:
```
Dataset Row → Model → CommandPlan
```

### Inference Time
Uses System 1's output:
```
CanonicalIntent (from System 1) → Model → CommandPlan
```

## Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **CodeT5+ (770M)** | Primary model - Salesforce's code-specialized T5 | Best accuracy for code generation |
| **FLAN-T5 Base** | Baseline model - Google's instruction-tuned T5 | Comparison and fallback |

## Data Contracts

### Training Data (command-dataset)

```json
{
  "instruction": "install git",
  "intent_type": "install_package",
  "entities": {
    "package": "git",
    "runtime": null,
    "version": null
  },
  "os": "linux",
  "shell": "bash",
  "command": "sudo apt install git -y",
  "source": "NL2SH-ALFA"
}
```

### Model Input (at training)

```
<intent>install_package</intent>
<entities>{"package":"git","runtime":null,"version":null}</entities>
<os>linux</os>
<shell>bash</shell>
<command>
```

### Model Output: CommandPlan

```json
{
  "intent_type": "install_package",
  "entities": {"package": "git", "runtime": null, "version": null},
  "os": "linux",
  "shell": "bash",
  "steps": [
    {
      "step_number": 1,
      "type": "install",
      "command": "sudo apt install git -y",
      "description": "install git"
    }
  ],
  "confidence": 0.95,
  "requires_elevation": true
}
```

### Inference Input (System 1 → System 2)

At runtime, System 1 produces a **CanonicalIntent** which is converted to the model's input format:

```json
{
  "intent_type": "install_package",
  "entities": {"package": "git", "runtime": null, "version": null},
  "os_hint": "linux",
  "shell_type": "bash",
  ...
}
```

## Installation

```bash
# Navigate to the project root
cd contAIner

# Install dependencies
pip install -r src/system2_command_generation/requirements.txt
```

## Usage

### Training

```bash
# Train CodeT5+ (primary model)
python -m src.system2_command_generation.train --model codet5plus

# Train FLAN-T5 (baseline)
python -m src.system2_command_generation.train --model flan_t5 --baseline

# Train with custom configuration
python -m src.system2_command_generation.train --config config.json

# Resume from checkpoint
python -m src.system2_command_generation.train --resume checkpoint-1000
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `codet5plus` | Model to train: `codet5plus` or `flan_t5` |
| `--baseline` | `false` | Train FLAN-T5 as baseline |
| `--data-source` | `huggingface` | Data source: `huggingface` or `local` |
| `--dataset-name` | `sumit-s-nair/command-dataset` | HuggingFace dataset name |
| `--local-data-dir` | `None` | Local data directory path |
| `--output-dir` | `./outputs/system2_command_generation` | Output directory |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `8` | Training batch size |
| `--learning-rate` | `5e-5` | Learning rate |
| `--seed` | `42` | Random seed for reproducibility |
| `--resume` | `None` | Checkpoint to resume from |
| `--eval-only` | `false` | Only run evaluation |
| `--model-path` | `None` | Path to trained model for evaluation |

### Evaluation Only

```bash
# Evaluate a trained model
python -m src.system2_command_generation.train \
    --eval-only \
    --model-path ./outputs/final_model
```

### Programmatic Usage

```python
from src.system2_command_generation import (
    CommandGenerationTrainer,
    TrainingConfig,
    ModelType,
)

# Create configuration
config = TrainingConfig(
    model_type=ModelType.CODET5_PLUS,
    num_epochs=10,
    batch_size=8,
    learning_rate=5e-5,
)

# Initialize trainer
trainer = CommandGenerationTrainer(config)

# Load data
trainer.load_data(dataset_source="huggingface")

# Train
results = trainer.train()

# Evaluate
eval_results = trainer.evaluate(dataset_split="test")
print(eval_results.summary())
```

### Inference

```python
from src.system2_command_generation import CommandGenerationModel
from src.system2_command_generation.models import generate_command_plan

# Load trained model
model = CommandGenerationModel.load("./outputs/final_model")

# Generate command plan
canonical_intent = {
    "intent_type": "install_package",
    "entities": {"runtime": None, "package": "numpy", "version": None},
    "scope": "user",
    "os_hint": "linux",
    "shell_type": "bash",
    "confidence": 0.95,
    "missing_fields": [],
    "needs_clarification": False,
    "clarification_question": None,
}

command_plan = generate_command_plan(model, canonical_intent)
print(command_plan)
```

## Evaluation Metrics

The training pipeline evaluates models against these metrics:

| Metric | Target | Description |
|--------|--------|-------------|
| Exact Match | ≥ 70% | Predicted JSON exactly matches reference |
| Normalized Match | ≥ 85% | Match after normalizing whitespace/flag order |
| Intent Preservation | 100% | Output intent matches input intent |
| Entity Preservation | ≥ 95% | Output entities match input entities |
| Syntax Validity | ≥ 95% | Commands are syntactically valid |
| OS/Shell Compatibility | 100% | Shell is compatible with target OS |

## Directory Structure

```
src/system2_command_generation/
├── __init__.py              # Module exports
├── config.py                # Configuration and schema definitions
├── data_preprocessing.py    # Data loading and preprocessing
├── models.py                # Model loading and inference
├── metrics.py               # Evaluation metrics
├── train.py                 # Training pipeline
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Configuration

### Training Configuration

Create a `config.json` file for custom training:

```json
{
  "model_type": "codet5plus",
  "output_dir": "./outputs/custom_run",
  "num_epochs": 15,
  "batch_size": 16,
  "learning_rate": 3e-5,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "gradient_accumulation_steps": 2,
  "fp16": true,
  "seed": 42,
  "eval_steps": 500,
  "save_steps": 500,
  "early_stopping_patience": 3
}
```

### Supported Intent Types

- `install_runtime` / `update_runtime` / `remove_runtime`
- `install_package` / `update_package` / `remove_package`
- `list_packages`
- `create_environment` / `activate_environment` / `deactivate_environment`
- `configure_setting`
- `run_script`
- `check_version`

### Supported OS/Shell Combinations

| OS | Compatible Shells |
|----|-------------------|
| Windows | PowerShell, Cmd |
| Linux | Bash, Zsh |
| macOS | Bash, Zsh |

## Outputs

After training, the following files are created:

```
outputs/system2_command_generation/
├── codet5plus_YYYYMMDD_HHMMSS/
│   ├── config.json              # Training configuration
│   ├── training_YYYYMMDD.log    # Training logs
│   ├── checkpoint-*/            # Training checkpoints
│   ├── final_model/             # Final trained model
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   └── training_config.json
│   ├── train_results.json       # Training metrics
│   └── eval_results_test.json   # Evaluation metrics
```

## Hardware Requirements

| Model | VRAM (FP16) | VRAM (FP32) | Training Time (10 epochs) |
|-------|-------------|-------------|---------------------------|
| CodeT5+ 770M | ~6 GB | ~12 GB | ~2-4 hours (V100) |
| FLAN-T5 Base | ~2 GB | ~4 GB | ~1-2 hours (V100) |

For systems with limited VRAM:
- Use `--batch-size 4` or lower
- Enable gradient accumulation with `gradient_accumulation_steps: 4`
- Use 8-bit quantization (requires `bitsandbytes`)

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python -m src.system2_command_generation.train --batch-size 4

# Or use gradient accumulation in config
{
  "batch_size": 4,
  "gradient_accumulation_steps": 4
}
```

### Dataset Not Found

```bash
# Use local dataset
python -m src.system2_command_generation.train \
    --data-source local \
    --local-data-dir ./datasets/command-dataset/data
```

### CUDA Not Available

The training script automatically falls back to CPU if CUDA is not available, but training will be significantly slower.

## License

MIT License - See [LICENSE](../../LICENSE) for details.
