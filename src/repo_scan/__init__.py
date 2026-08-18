"""
src/repo_scan
=============
Static, deterministic, LLM-free repository manifest extraction.

Public surface:
    from src.repo_scan import scan_repo, RepoManifest
"""

from .models import (
    Dependency,
    EcosystemManifest,
    EnvironmentConfig,
    InferredDependency,
    RepoManifest,
    SourceRef,
)
from .scanner import scan_repo

__all__ = [
    "scan_repo",
    "RepoManifest",
    "EcosystemManifest",
    "Dependency",
    "EnvironmentConfig",
    "InferredDependency",
    "SourceRef",
]
