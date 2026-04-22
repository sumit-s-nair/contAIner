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
- Training and evaluation pipeline for Qwen2.5-Coder-1.5B and CodeT5+
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
| **Qwen2.5-Coder-1.5B** | Primary model - Qwen code-focused decoder-only LLM | Best command synthesis and instruction following |
| **CodeT5+ (770M)** | Baseline model - Salesforce's code-specialized T5 | Lightweight comparison baseline |

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
# Default command (run from repo root; includes MCP doc enrichment)
python -m src.system2_command_generation.train \
  --model qwen2_5_coder_1_5b \
  --use-qlora \
  --use-mcp \
  --mcp-url http://localhost:11435 \
  --output-dir ./outputs/system2_command_generation

# Train Qwen2.5-Coder-1.5B (primary model)
python -m src.system2_command_generation.train --model qwen2_5_coder_1_5b

# Recommended on 8-12 GB GPUs: QLoRA + MCP doc enrichment
python -m src.system2_command_generation.train \
  --model qwen2_5_coder_1_5b \
  --use-qlora \
  --use-mcp \
  --mcp-url http://localhost:11435 \
  --output-dir ./outputs/system2_command_generation

# Train CodeT5+ (baseline)
python -m src.system2_command_generation.train --model codet5plus --baseline

# Train with custom configuration
python -m src.system2_command_generation.train --config config.json

# Resume from checkpoint
python -m src.system2_command_generation.train --resume checkpoint-1000
```

Training sessions are saved automatically. Each run creates a timestamped folder
under `./outputs/system2_command_generation/` (for example,
`qwen2_5_coder_1_5b_20260421_110607`) containing `config.json`,
`training_*.log`, `checkpoint-*`, and `final_model/`.

To continue a saved session from a specific checkpoint:

```bash
python -m src.system2_command_generation.train \
  --model qwen2_5_coder_1_5b \
  --resume ./outputs/system2_command_generation/qwen2_5_coder_1_5b_YYYYMMDD_HHMMSS/checkpoint-1000
```

Pressing `Ctrl+C` during training now triggers an emergency checkpoint save
before exit, so you can resume from the latest saved state.

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `qwen2_5_coder_1_5b` | Model to train: `qwen2_5_coder_1_5b` or `codet5plus` |
| `--baseline` | `false` | Train CodeT5+ as baseline |
| `--data-source` | `huggingface` | Data source: `huggingface` or `local` |
| `--dataset-name` | `sumit-s-nair/command-dataset` | HuggingFace dataset name |
| `--local-data-dir` | `None` | Local data directory path |
| `--output-dir` | `./outputs/system2_command_generation` | Output directory |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `8` | Training batch size |
| `--eval-batch-size` | `16` | Evaluation batch size |
| `--grad-accum-steps` | `4` | Gradient accumulation steps |
| `--max-input-length` | `512` | Maximum prompt/input token length |
| `--max-output-length` | `1024` | Maximum target/completion token length |
| `--generation-num-beams` | `1` | Beam width used during evaluation generation |
| `--eval-steps` | `500` | Run evaluation every N optimizer steps |
| `--save-steps` | `100` | Save checkpoint every N optimizer steps |
| `--save-total-limit` | `3` | Maximum number of checkpoints to retain |
| `--warmup-steps` | `0` | Warmup steps (overrides warmup ratio when > 0) |
| `--no-gradient-checkpointing` | `false` | Disable gradient checkpointing |
| `--disable-auto-memory-tuning` | `false` | Disable automatic low-VRAM tuning for Qwen |
| `--learning-rate` | `5e-5` | Learning rate |
| `--seed` | `42` | Random seed for reproducibility |
| `--use-qlora` | `auto` | Enable QLoRA (default behavior for Qwen2.5-Coder-1.5B) |
| `--no-qlora` | `false` | Disable QLoRA and train full model weights |
| `--lora-r` | `16` | LoRA rank |
| `--lora-alpha` | `32` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--lora-target-modules` | Qwen proj layers | LoRA target module names |
| `--qlora-compute-dtype` | `float16` | QLoRA compute dtype |
| `--qlora-quant-type` | `nf4` | 4-bit quantization type |
| `--qlora-no-double-quant` | `false` | Disable nested quantization |
| `--use-mcp` | `false` | Enrich each input with live MCP docs |
| `--mcp-url` | `http://localhost:11435` | MCP server base URL |
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
  model_type=ModelType.QWEN2_5_CODER_1_5B,
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
  "model_type": "qwen2_5_coder_1_5b",
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
├── qwen2_5_coder_1_5b_YYYYMMDD_HHMMSS/
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
| Qwen2.5-Coder-1.5B | ~10 GB | ~20 GB | ~3-6 hours (V100) |
| CodeT5+ 770M | ~6 GB | ~12 GB | ~2-4 hours (V100) |

For systems with limited VRAM:
- Use `--batch-size 4` or lower
- Enable gradient accumulation with `gradient_accumulation_steps: 4`
- Prefer `--use-qlora` for Qwen2.5-Coder-1.5B training on 8-12 GB GPUs

## Troubleshooting

### Out of Memory

Qwen2.5-Coder-1.5B can exceed VRAM quickly at long sequence lengths.
The trainer now auto-applies conservative settings on lower-memory GPUs
unless `--disable-auto-memory-tuning` is set.

```bash
# Explicit memory-safe run (works on many 8-12 GB GPUs)
python -m src.system2_command_generation.train \
  --model qwen2_5_coder_1_5b \
  --batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 16 \
  --max-input-length 256 \
  --max-output-length 256
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
