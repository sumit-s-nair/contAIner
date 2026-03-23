#!/usr/bin/env python3
"""Upload local dataset folders to Hugging Face dataset repositories.

Usage examples:

# Upload both datasets using HF_TOKEN from environment:
HF_TOKEN=xxx python scripts/push_datasets_to_hf.py --all

# Upload a single dataset and pass token explicitly:
python scripts/push_datasets_to_hf.py \
    --local-path datasets/intent-dataset \
    --repo-id sumit-s-nair/intent-dataset \
    --token <HF_TOKEN>

Notes:
- Requires `huggingface_hub` (pip install huggingface_hub).
- The script creates the dataset repo if it doesn't exist.
- Files are uploaded with `upload_folder` (no git LFS required for small text files).
"""

import argparse
import os
import sys
from pathlib import Path

# Load environment variables from .env files
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.env_loader import ensure_env_loaded
ensure_env_loaded()

try:
    from huggingface_hub import create_repo, upload_folder, HfApi
except Exception as e:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    raise


def push_folder(local_path: Path, repo_id: str, token: str):
    local_path = local_path.resolve()
    if not local_path.exists():
        print(f"Local path not found: {local_path}")
        return 1

    # Ensure the destination dataset repository exists before upload.
    print(f"Creating or checking repo: {repo_id} (type=dataset)")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Warning: create_repo may have failed or repo already exists: {e}")

    # Upload folder contents to repository root.
    print(f"Uploading folder {local_path} to {repo_id}...")
    try:
        upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo="",
            token=token,
        )
        print(f"✓ Uploaded {local_path} to {repo_id}")
        return 0
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return 2


def main():
    parser = argparse.ArgumentParser(description="Push local dataset folders to Hugging Face Hub")
    parser.add_argument("--local-path", type=str, help="Local dataset folder to upload")
    parser.add_argument("--repo-id", type=str, help="Hub repo id, e.g. username/repo-name")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--all", action="store_true", help="Upload both datasets using default paths and repo ids")

    args = parser.parse_args()

    # Accept any of the common HF token env var names
    env_vars = ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"]
    token = args.token or next((os.environ.get(k) for k in env_vars if os.environ.get(k)), None)
    if not token:
        print(
            "Error: No Hugging Face token provided. Set one of HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HUGGINGFACE_TOKEN or pass --token"
        )
        sys.exit(1)

    tasks = []
    if args.all:
        tasks.append((Path("datasets/intent-dataset"), "sumit-s-nair/intent-dataset"))
        tasks.append((Path("datasets/command-dataset"), "sumit-s-nair/command-dataset"))
    else:
        if not args.local_path or not args.repo_id:
            print("Error: --local-path and --repo-id are required unless --all is used")
            sys.exit(1)
        tasks.append((Path(args.local_path), args.repo_id))

    exit_codes = []
    for local_path, repo_id in tasks:
        code = push_folder(local_path, repo_id, token)
        exit_codes.append(code)

    if any(c != 0 for c in exit_codes):
        print("One or more uploads failed")
        sys.exit(2)

    print("All uploads completed successfully.")


if __name__ == "__main__":
    main()
