# System 2: Repository Scanner (`repo_scan`)

## Overview
The `repo_scan` module provides the ground-truth environment state used by System 2. It evaluates project repositories to detect dependencies, ecosystems, configuration constraints, and known conflicts, packaging them into a deterministic `RepoManifest`.

## Data Models (Schema)
The scanning logic produces pure data structures to isolate parsing from analysis. Key classes defined in `src/repo_scan/models.py`:

```python
@dataclass
class SourceRef:
    file: str
    line: Optional[int] = None
    raw_line: Optional[str] = None

@dataclass
class Dependency:
    name: str
    declared_constraint: str
    resolved_version: Optional[str] = None
    sources: List[SourceRef] = field(default_factory=list)

@dataclass
class InferredDependency:
    import_name: str
    guessed_package_name: Optional[str]
    confidence: str  # "mapped" | "unmapped_guess"
    sources: List[SourceRef] = field(default_factory=list)

@dataclass
class EcosystemManifest:
    ecosystem: str
    manifest_files: List[str] = field(default_factory=list)
    lock_files: List[str] = field(default_factory=list)
    dependencies: Dict[str, Dependency] = field(default_factory=dict)
    inferred_dependencies: List[InferredDependency] = field(default_factory=list)

@dataclass
class RepoManifest:
    ecosystems: Dict[str, EcosystemManifest] = field(default_factory=dict)
    environment_configs: List[EnvironmentConfig] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
```

## Parsing and Traceability
- **Manifest parsing**: Uses a dual-pass approach (e.g., `tomllib` plus regular expressions for `pyproject.toml`) to extract both the AST structures and the raw physical file line numbers for source traceability.

## Conflict Detection Rules
The module implements conflict detection (e.g., identifying when a user attempts to globally install dependencies that conflict with system packages). 
- **Example Conflict**: Multi-manifest mismatch (e.g., requirements.txt and pyproject.toml specifying divergent versions), competing package managers, or version pin vs. manifest conflicts. 

## Import-Based Inference Fallback (AST Extraction)
For Python codebases lacking manifests (e.g., raw scripts), `src/repo_scan/import_scan.py` implements an AST-based static analysis fallback:
1. **Extraction**: Collects all top-level `import` and `from ... import` statements.
2. **Filtering**: Excludes Python standard library modules (dependent on the running interpreter version) and local file imports.
3. **Resolution**: Maps the extracted import name to PyPI package names using a ~200-entry internal static mapping table (e.g., `IMPORT_TO_PACKAGE = {"cv2": "opencv-python"}`).
4. **Confidence Tagging**: Tags results as `"mapped"` (found in table) or `"unmapped_guess"` (falling back to the import string directly).

**Known Limitations**: 
- Standard library filtering accuracy depends on the scanning interpreter's Python version (e.g., `urlparse` triggered a false positive because it's stdlib in Python 2 but tested under Python 3). 
- Tested primarily on top-level imports; conditional/deferred imports (`try/except ImportError`) remain unverified by the initial test suite.

## Concrete Example: Explain Output
The `RepoManifest.explain()` function provides line-level traceability. Here is an example of an inferred dependency mapping from a verified real-repo test:

**Input Query**: `manifest.explain("opencv-python")`

**Output**:
```text
cv2: inferred from 'import cv2' in src/main.py:5 (no manifest found — mapped to package 'opencv-python')
```
