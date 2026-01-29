"""
Dataset Loader Utility for contAIner Project

This script provides a API to fetch and work with Hugging Face datasets for intent understanding and command generation tasks.

Datasets:
- sumit-s-nair/intent-dataset
- sumit-s-nair/command-dataset

Usage:
    from src.utils.load_datasets import load_datasets

    intent_data, command_data = load_datasets()
    print(intent_data['train'][0])
"""

import sys
import os
from typing import Optional, Tuple
from datasets import load_dataset, DatasetDict
from huggingface_hub import login

# Load environment variables from .env files
from .env_loader import ensure_env_loaded
ensure_env_loaded()


# Dataset identifiers
INTENT_DATASET = "sumit-s-nair/intent-dataset"
COMMAND_DATASET = "sumit-s-nair/command-dataset"


def check_and_install_dependencies():
    # Check if required packages are installed
    try:
        import datasets
        import huggingface_hub
    except ImportError as e:
        print(f"⚠️  Missing required package: {e.name}")
        print("\nPlease install dependencies using:")
        print("  pip install datasets huggingface_hub")
        sys.exit(1)


def authenticate_huggingface(token: Optional[str] = None, auto_prompt: bool = True):
    """
    Authenticate with Hugging Face if needed.

    Behavior:
      - If `token` is provided, use it.
      - Else, check environment variables (`HF_TOKEN`).
      - If not found and `auto_prompt` is True, attempt an interactive
      `huggingface_hub.login()` which prompts the user to paste a token.

    Args:
        token: Optional HF token. If provided, will be used immediately.
        auto_prompt: If True, prompt interactively when no token is found.

    Our datasets are public so auth is optional but still do login.
    """
    try:
        if token:
            login(token=token)
            print("✓ Authenticated with Hugging Face using provided token")
            return

        # Check environment var
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            login(token=env_token)
            print("✓ Authenticated with Hugging Face using token from environment")
            return

        # Prompt hf login
        if auto_prompt:
            try:
                login()
                print("✓ Authenticated with Hugging Face via interactive login")
            except Exception as e:
                print(f"⚠️ Interactive login failed: {e}")
                print(
                    "If datasets are private, please authenticate with: huggingface-cli login"
                )
        else:
            print("ℹ️  No Hugging Face token found; continuing without authentication.")

    except Exception as e:
        print(f"⚠️  Authentication note: {e}")
        print(
            "If datasets are private, please authenticate with: huggingface-cli login"
        )


def load_single_dataset(
    dataset_name: str, splits: Optional[list] = None
) -> DatasetDict:
    """
    Load a single dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        splits: List of splits to load (e.g., ['train', 'validation', 'test'])
                If None, loads all available splits.

    Returns:
        DatasetDict containing the requested splits

    Raises:
        Exception: If dataset cannot be loaded
    """
    try:
        print(f"\n📦 Loading dataset: {dataset_name}")

        if splits:
            # Load specific splits
            dataset_dict = {}
            for split in splits:
                try:
                    dataset_dict[split] = load_dataset(dataset_name, split=split)
                    print(f"  ✓ Loaded split: {split}")
                except Exception as e:
                    print(f"  ⚠️  Split '{split}' not available: {e}")

            if not dataset_dict:
                raise Exception(f"No valid splits found for {dataset_name}")

            return DatasetDict(dataset_dict)
        else:
            # Load all available splits
            dataset = load_dataset(dataset_name)
            print(f"  ✓ Loaded all available splits: {list(dataset.keys())}")
            return dataset

    except Exception as e:
        print(f"  ✗ Error loading dataset {dataset_name}: {e}")
        raise


def print_dataset_statistics(dataset: DatasetDict, dataset_name: str):
    """
    Print basic statistics about a dataset.

    Args:
        dataset: The loaded dataset
        dataset_name: Name of the dataset for display
    """
    print(f"\n📊 Statistics for {dataset_name}:")
    print("=" * 60)

    total_samples = 0
    for split_name, split_data in dataset.items():
        num_samples = len(split_data)
        total_samples += num_samples
        print(f"  {split_name:12s}: {num_samples:6d} samples")

    print(f"  {'TOTAL':12s}: {total_samples:6d} samples")

    # Print column information from first available split
    if dataset:
        first_split = list(dataset.keys())[0]
        columns = dataset[first_split].column_names
        print(f"\n  Available fields: {', '.join(columns)}")

        # Show example from first split
        if len(dataset[first_split]) > 0:
            print(f"\n  Example from '{first_split}' split:")
            example = dataset[first_split][0]
            for key, value in example.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                print(f"    {key}: {value_str}")

    print("=" * 60)


def load_datasets(
    intent_dataset: str = INTENT_DATASET,
    command_dataset: str = COMMAND_DATASET,
    splits: Optional[list] = None,
    hf_token: Optional[str] = None,
    hf_auto_login: bool = True,
    print_stats: bool = True,
) -> Tuple[DatasetDict, DatasetDict]:
    """
    Load both intent and command datasets from Hugging Face.

    Args:
        intent_dataset: Name of the intent dataset (default: sumit-s-nair/intent-dataset)
        command_dataset: Name of the command dataset (default: sumit-s-nair/command-dataset)
        splits: List of splits to load (e.g., ['train', 'test']).
                If None, loads all available splits.
        hf_token: Optional Hugging Face token for authentication
        print_stats: Whether to print dataset statistics (default: True)

    Returns:
        Tuple of (intent_dataset, command_dataset) as DatasetDict objects

    Example:
        >>> intent_data, command_data = load_datasets()
        >>> print(intent_data['train'][0])
        >>> print(command_data['train'][0])

        # Load specific splits only
        >>> intent_data, command_data = load_datasets(splits=['train', 'test'])
    """
    # Check dependencies
    check_and_install_dependencies()

    # Attempt authentication flow: token -> env -> cache -> optional prompt
    authenticate_huggingface(hf_token, auto_prompt=hf_auto_login)

    print("\n" + "=" * 60)
    print("🚀 contAIner Dataset Loader")
    print("=" * 60)

    # Load intent dataset
    try:
        intent_data = load_single_dataset(intent_dataset, splits)
        if print_stats:
            print_dataset_statistics(intent_data, "Intent Dataset")
    except Exception as e:
        print(f"\n❌ Failed to load intent dataset: {e}")
        raise

    # Load command dataset
    try:
        command_data = load_single_dataset(command_dataset, splits)
        if print_stats:
            print_dataset_statistics(command_data, "Command Dataset")
    except Exception as e:
        print(f"\n❌ Failed to load command dataset: {e}")
        raise

    print("\n✅ All datasets loaded successfully!")
    print("=" * 60)

    return intent_data, command_data


def main():
    """
    Main function for demonstration and testing.

    Run this script directly to verify dataset loading:
        python src/utils/load_datasets.py
    """
    print("Running dataset loader in standalone mode...\n")

    try:
        # Load all datasets with default settings
        intent_data, command_data = load_datasets()

        print("\n" + "=" * 60)
        print("✓ Dataset Loading Test Successful!")
        print("=" * 60)
        print("\nYou can now use these datasets in your training scripts:")
        print("  from src.utils.load_datasets import load_datasets")
        print("  intent_data, command_data = load_datasets()")

    except Exception as e:
        print(f"\n❌ Dataset loading failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have internet connectivity")
        print("2. Check if datasets are accessible (public/private)")
        print("3. If private, authenticate: huggingface-cli login")
        print("4. Verify dataset names are correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
