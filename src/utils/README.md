# Dataset Loader Utility

This utility provides a API for fetching Hugging Face datasets used in the contAIner project.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

#### As a Module

```python
from src.utils.load_datasets import load_datasets

# Load all datasets with all available splits
intent_data, command_data = load_datasets()

# Access specific splits
train_intent = intent_data['train']
test_command = command_data['test']

# Load specific splits only
intent_data, command_data = load_datasets(splits=['train', 'test'])

# With authentication (for private datasets)
intent_data, command_data = load_datasets(hf_token="your_hf_token_here")
```

#### Standalone Script

```bash
# Run as standalone to test dataset loading
python src/utils/load_datasets.py
```

## Datasets

1. **Intent Dataset**: `sumit-s-nair/intent-dataset`
   - Used for intent understanding tasks

2. **Command Dataset**: `sumit-s-nair/command-dataset`
   - Used for command generation tasks

## Status

- `load_datasets.py` is ready for both scripted and import-based usage.
- `env_loader.py` provides centralized `.env` discovery and loading.
- Utilities are actively used by training and dataset upload workflows.

## API Reference

### `load_datasets()`

Main function to load both datasets.

**Parameters:**
- `intent_dataset` (str): Name of intent dataset (default: "sumit-s-nair/intent-dataset")
- `command_dataset` (str): Name of command dataset (default: "sumit-s-nair/command-dataset")
- `splits` (list, optional): List of splits to load (e.g., ['train', 'test']). If None, loads all.
- `hf_token` (str, optional): Hugging Face token for authentication
- `print_stats` (bool): Whether to print dataset statistics (default: True)

**Returns:**
- Tuple of (intent_dataset, command_dataset) as DatasetDict objects

**Example:**
```python
intent_data, command_data = load_datasets(
    splits=['train', 'validation'],
    print_stats=True
)
```

## Authentication

For private datasets:

```bash
# Option 1: Use CLI
huggingface-cli login

# Option 2: Pass token in code
intent_data, command_data = load_datasets(hf_token="your_token")
```

## Caching

Datasets are automatically cached by Hugging Face in:
- **Linux/Mac**: `~/.cache/huggingface/datasets`
- **Windows**: `%USERPROFILE%\.cache\huggingface\datasets`

## Troubleshooting

**Problem**: Import error for `datasets` or `huggingface_hub`
```bash
pip install datasets huggingface_hub
```

**Problem**: Authentication error for private datasets
```bash
huggingface-cli login
```

**Problem**: Missing split
The script will skip unavailable splits and load only the available ones.

## Integration Example

```python
# In your training script
from src.utils.load_datasets import load_datasets

def train_model():
    # Load datasets
    intent_data, command_data = load_datasets(splits=['train'])
    
    # Use in your training loop
    for example in intent_data['train']:
        # Your training code here
        pass
```
