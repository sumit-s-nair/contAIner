"""
Centralized environment variable loading utility for contAIner project.

This module provides a consistent way to load environment variables from .env files
across all project components. It automatically searches for .env files in the project
root and loads them into os.environ.

Usage:
    from src.utils.env_loader import load_env
    
    load_env()  # Call this before accessing any environment variables
    
    # Then access env vars normally
    token = os.environ.get("HF_TOKEN")
"""

import os
from pathlib import Path
from typing import Optional, List


def find_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the project root by looking for common project markers.
    
    Args:
        start_path: Starting path to search from (default: current file's directory)
        
    Returns:
        Path to project root, or None if not found
    """
    if start_path is None:
        start_path = Path(__file__).parent
    
    current = Path(start_path).resolve()
    
    # Look for common project root markers
    markers = [".git", "pyproject.toml", "setup.py", "requirements.txt", ".env"]
    
    while current != current.parent:
        if any((current / marker).exists() for marker in markers):
            return current
        current = current.parent
    
    return None


def load_env_file(env_path: Path) -> int:
    """
    Load environment variables from a single .env file.
    
    Args:
        env_path: Path to the .env file
        
    Returns:
        Number of variables loaded
    """
    if not env_path.exists():
        return 0
    
    loaded_count = 0
    try:
        content = env_path.read_text(encoding='utf-8')
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Parse KEY=value format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Only set if not already present (don't override existing env vars)
                if key not in os.environ:
                    os.environ[key] = value
                    loaded_count += 1
                    
    except Exception as e:
        print(f"Warning: Failed to load {env_path}: {e}")
    
    return loaded_count


def load_env(search_paths: Optional[List[Path]] = None, verbose: bool = False) -> int:
    """
    Load environment variables from .env files.
    
    Searches for .env files in:
    1. Current working directory
    2. Project root directory (auto-detected)
    3. Additional paths specified in search_paths
    
    Args:
        search_paths: Additional paths to search for .env files
        verbose: Whether to print loading information
        
    Returns:
        Total number of environment variables loaded
        
    Note:
        - Existing environment variables are not overridden
        - Uses python-dotenv if available, fallback to manual parsing
        - Automatically detects project root
    """
    total_loaded = 0
    
    # Try using python-dotenv first
    try:
        from dotenv import load_dotenv
        
        # Load from current directory
        if load_dotenv(verbose=verbose):
            if verbose:
                print("✓ Loaded .env using python-dotenv")
        
        # Load from project root if different from current directory
        project_root = find_project_root()
        if project_root and project_root != Path.cwd():
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file, verbose=verbose)
                if verbose:
                    print(f"✓ Loaded {env_file} using python-dotenv")
        
        # For dotenv, we can't easily count loaded vars, so return success indicator
        return 1 if any(Path(p).exists() for p in [".env", project_root / ".env" if project_root else None] if p) else 0
        
    except ImportError:
        # Fallback to manual loading
        if verbose:
            print("ℹ️ python-dotenv not available, using manual .env parsing")
    
    # Manual loading fallback
    search_locations = []
    
    # Add current working directory
    search_locations.append(Path.cwd() / ".env")
    
    # Add project root
    project_root = find_project_root()
    if project_root:
        search_locations.append(project_root / ".env")
    
    # Add custom search paths
    if search_paths:
        for path in search_paths:
            if path.is_dir():
                search_locations.append(path / ".env")
            else:
                search_locations.append(path)
    
    # Load from all found locations
    for env_path in search_locations:
        if env_path.exists():
            loaded = load_env_file(env_path)
            total_loaded += loaded
            if verbose and loaded > 0:
                print(f"✓ Loaded {loaded} variables from {env_path}")
    
    if verbose and total_loaded == 0:
        print("ℹ️ No .env files found or no new variables loaded")
    
    return total_loaded


def ensure_env_loaded():
    """
    Convenience function to ensure environment variables are loaded.
    
    This is safe to call multiple times - it won't reload if already loaded.
    Call this at the start of any module that uses environment variables.
    """
    # Simple check - if we haven't loaded before, do it now
    if not getattr(ensure_env_loaded, '_loaded', False):
        load_env()
        ensure_env_loaded._loaded = True
