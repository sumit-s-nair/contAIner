"""
src/repo_scan/import_scan.py
============================
AST-based Python import inference — fallback dependency detection.

Used by the scanner when ``.py`` files are present but no manifest file
(``pyproject.toml`` / ``requirements.txt``) is found.

Public API
----------
    scan_imports(root: Path) -> List[InferredDependency]
    is_local_module(name: str, root: Path) -> bool   # exported for tests

Design rules
------------
- AST-only import extraction: no regex on source text.
- Stdlib filtered via ``sys.stdlib_module_names`` (Python ≥ 3.10) or a
  bundled static frozenset (Python 3.8/3.9 fallback).
- Local modules excluded only when a ``<name>.py`` file OR a proper package
  directory ``<name>/__init__.py`` exists at the repo root.
  A bare directory *without* ``__init__.py`` is NOT treated as local.
- Aggregation: the same import name seen in N files → one
  ``InferredDependency`` with N ``SourceRef`` entries.
- Per-file parse failures are logged (skipped) without crashing the scan.
"""

from __future__ import annotations

import ast
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import InferredDependency, SourceRef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stdlib module set
# ---------------------------------------------------------------------------

def _stdlib_modules() -> frozenset:
    """
    Return the set of top-level stdlib module names for the running Python.

    Uses ``sys.stdlib_module_names`` on Python ≥ 3.10.  On older versions
    falls back to a statically bundled list that covers the Python 3.8/3.9
    standard library (the lowest versions this codebase targets).
    """
    if hasattr(sys, "stdlib_module_names"):
        return frozenset(sys.stdlib_module_names)

    # Static fallback — Python 3.8 / 3.9 stdlib top-level names.
    # Source: https://docs.python.org/3.9/py-modindex.html
    return frozenset({
        "__future__", "_thread", "abc", "aifc", "argparse", "array",
        "ast", "asynchat", "asyncio", "asyncore", "atexit", "audioop",
        "base64", "bdb", "binascii", "binhex", "bisect", "builtins",
        "bz2", "calendar", "cgi", "cgitb", "chunk", "cmath", "cmd",
        "code", "codecs", "codeop", "collections", "colorsys",
        "compileall", "concurrent", "configparser", "contextlib",
        "contextvars", "copy", "copyreg", "cProfile", "csv", "ctypes",
        "curses", "dataclasses", "datetime", "dbm", "decimal",
        "difflib", "dis", "distutils", "doctest", "email",
        "encodings", "enum", "errno", "faulthandler", "fcntl",
        "filecmp", "fileinput", "fnmatch", "fractions", "ftplib",
        "functools", "gc", "getopt", "getpass", "gettext", "glob",
        "grp", "gzip", "hashlib", "heapq", "hmac", "html", "http",
        "idlelib", "imaplib", "imghdr", "imp", "importlib", "inspect",
        "io", "ipaddress", "itertools", "json", "keyword", "lib2to3",
        "linecache", "locale", "logging", "lzma", "mailbox",
        "mailcap", "marshal", "math", "mimetypes", "mmap",
        "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
        "numbers", "operator", "optparse", "os", "ossaudiodev",
        "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil",
        "platform", "plistlib", "poplib", "posix", "posixpath",
        "pprint", "profile", "pstats", "pty", "pwd", "py_compile",
        "pyclbr", "pydoc", "queue", "quopri", "random", "re",
        "readline", "reprlib", "resource", "rlcompleter", "runpy",
        "sched", "secrets", "select", "selectors", "shelve", "shlex",
        "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr",
        "socket", "socketserver", "spwd", "sqlite3", "sre_compile",
        "sre_constants", "sre_parse", "ssl", "stat", "statistics",
        "string", "stringprep", "struct", "subprocess", "sunau",
        "symtable", "sys", "sysconfig", "syslog", "tabnanny",
        "tarfile", "telnetlib", "tempfile", "termios", "test",
        "textwrap", "threading", "time", "timeit", "tkinter",
        "token", "tokenize", "tomllib", "trace", "traceback",
        "tracemalloc", "tty", "turtle", "turtledemo", "types",
        "typing", "unicodedata", "unittest", "urllib", "uu",
        "uuid", "venv", "warnings", "wave", "weakref", "webbrowser",
        "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
        "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
        # Common private/internal top-levels exposed at runtime
        "_collections_abc", "_weakrefset",
    })


_STDLIB: frozenset = _stdlib_modules()


# ---------------------------------------------------------------------------
# Import-name → PyPI package-name mapping table
# ---------------------------------------------------------------------------

# Keys are the import-statement top-level names; values are PyPI package names.
# Sourced from pip, pipreqs, isort, and the PyPI package index.
IMPORT_TO_PACKAGE: Dict[str, str] = {
    # Image / computer vision
    "cv2":              "opencv-python",
    "PIL":              "Pillow",
    "skimage":          "scikit-image",
    "imageio":          "imageio",

    # Data science / ML
    "sklearn":          "scikit-learn",
    "tensorflow":       "tensorflow",
    "tf":               "tensorflow",
    "torch":            "torch",
    "torchvision":      "torchvision",
    "torchaudio":       "torchaudio",
    "xgboost":          "xgboost",
    "lightgbm":         "lightgbm",
    "catboost":         "catboost",
    "shap":             "shap",

    # Scientific computing
    "numpy":            "numpy",
    "np":               "numpy",       # common alias
    "pandas":           "pandas",
    "pd":               "pandas",      # common alias
    "scipy":            "scipy",
    "matplotlib":       "matplotlib",
    "plt":              "matplotlib",  # common alias
    "seaborn":          "seaborn",
    "statsmodels":      "statsmodels",
    "sympy":            "sympy",

    # Web / HTTP
    "requests":         "requests",
    "httpx":            "httpx",
    "aiohttp":          "aiohttp",
    "urllib3":          "urllib3",
    "bs4":              "beautifulsoup4",
    "flask":            "Flask",
    "fastapi":          "fastapi",
    "uvicorn":          "uvicorn",
    "starlette":        "starlette",
    "django":           "Django",
    "tornado":          "tornado",
    "sanic":            "sanic",
    "bottle":           "bottle",
    "falcon":           "falcon",

    # Configuration / environment
    "dotenv":           "python-dotenv",
    "decouple":         "python-decouple",
    "yaml":             "PyYAML",
    "toml":             "toml",
    "dynaconf":         "dynaconf",

    # CLI
    "click":            "click",
    "typer":            "typer",
    "rich":             "rich",
    "colorama":         "colorama",
    "tqdm":             "tqdm",
    "tabulate":         "tabulate",
    "prompt_toolkit":   "prompt-toolkit",

    # Database / ORM
    "sqlalchemy":       "SQLAlchemy",
    "alembic":          "alembic",
    "pymongo":          "pymongo",
    "motor":            "motor",
    "redis":            "redis",
    "psycopg2":         "psycopg2",
    "psycopg":          "psycopg",
    "pymysql":          "PyMySQL",
    "peewee":           "peewee",
    "tortoise":         "tortoise-orm",

    # Serialisation / validation
    "pydantic":         "pydantic",
    "attr":             "attrs",
    "attrs":            "attrs",
    "marshmallow":      "marshmallow",
    "cerberus":         "cerberus",

    # Auth / crypto
    "jwt":              "PyJWT",
    "cryptography":     "cryptography",
    "nacl":             "PyNaCl",
    "bcrypt":           "bcrypt",
    "passlib":          "passlib",

    # Testing
    "pytest":           "pytest",
    "hypothesis":       "hypothesis",
    "faker":            "Faker",
    "factory":          "factory-boy",
    "responses":        "responses",

    # Cloud / infra SDKs
    "boto3":            "boto3",
    "botocore":         "botocore",
    "google":           "google-cloud-core",
    "azure":            "azure-core",

    # Async / concurrency
    "trio":             "trio",
    "anyio":            "anyio",
    "celery":           "celery",
    "dramatiq":         "dramatiq",

    # Misc utilities
    "dateutil":         "python-dateutil",
    "arrow":            "arrow",
    "pendulum":         "pendulum",
    "humanize":         "humanize",
    "loguru":           "loguru",
    "structlog":        "structlog",
    "pkg_resources":    "setuptools",
    "setuptools":       "setuptools",
    "six":              "six",
    "future":           "future",
    "more_itertools":   "more-itertools",
    "toolz":            "toolz",
    "cytoolz":          "cytoolz",
    "chardet":          "chardet",
    "charset_normalizer": "charset-normalizer",
    "lxml":             "lxml",
    "xlrd":             "xlrd",
    "xlwt":             "xlwt",
    "openpyxl":         "openpyxl",
    "docx":             "python-docx",
    "pptx":             "python-pptx",
    "paramiko":         "paramiko",
    "fabric":           "fabric",
    "invoke":           "invoke",
    "sh":               "sh",
    "psutil":           "psutil",
    "watchdog":         "watchdog",
    "schedule":         "schedule",
    "apscheduler":      "APScheduler",
    "Crypto":           "pycryptodome",
    "nacl":             "PyNaCl",
    "pyarrow":          "pyarrow",
    "dask":             "dask",
    "numba":            "numba",
    "cython":           "Cython",
    "networkx":         "networkx",
    "nltk":             "nltk",
    "spacy":            "spacy",
    "transformers":     "transformers",
    "gensim":           "gensim",
    "plotly":           "plotly",
    "bokeh":            "bokeh",
    "altair":           "altair",
    "streamlit":        "streamlit",
    "gradio":           "gradio",
    "jinja2":           "Jinja2",
    "Jinja2":           "Jinja2",
    "mako":             "Mako",
    "markupsafe":       "MarkupSafe",
    "itsdangerous":     "itsdangerous",
    "werkzeug":         "Werkzeug",
    "multipart":        "python-multipart",
    "magic":            "python-magic",
    "slugify":          "python-slugify",
    "unidecode":        "Unidecode",
    "pycurl":           "pycurl",
    "selenium":         "selenium",
    "playwright":       "playwright",
    "scrapy":           "Scrapy",
    "feedparser":       "feedparser",
    "pygments":         "Pygments",
    "mistune":          "mistune",
    "markdown":         "Markdown",
    "docutils":         "docutils",
    "sphinx":           "Sphinx",
    "nox":              "nox",
    "black":            "black",
    "ruff":             "ruff",
    "mypy":             "mypy",
    "pylint":           "pylint",
    "flake8":           "flake8",
    "isort":            "isort",
    "bandit":           "bandit",
    "coverage":         "coverage",
    "codecov":          "codecov",
    "pre_commit":       "pre-commit",
    "pep8":             "pep8",
    "autopep8":         "autopep8",
}


# ---------------------------------------------------------------------------
# Directory / path exclusions
# ---------------------------------------------------------------------------

# Directory *names* (any level) that should never be walked.
_EXCLUDED_DIR_NAMES: frozenset = frozenset({
    ".venv", "venv", "env", ".env",
    "__pycache__",
    ".git",
    "site-packages",
    "node_modules",
    ".tox", ".nox",
    "dist", "build", "eggs", ".eggs",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
})

# Repo-relative path *prefixes* that should be skipped wholesale.
# These are matched against Path.relative_to(root) as-is.
_EXCLUDED_REL_PREFIXES: Tuple[str, ...] = (
    "tests/fixtures",
    "tests\\fixtures",  # Windows variant
)


def _iter_py_files(root: Path):
    """
    Yield absolute :class:`Path` objects for every ``.py`` file under
    *root*, respecting all exclusion rules.
    """
    for path in root.rglob("*.py"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue

        rel_str = rel.as_posix()

        # Skip excluded relative prefixes (e.g. tests/fixtures/)
        if any(rel_str.startswith(prefix) for prefix in (
            "tests/fixtures/",
        )):
            continue

        # Skip any path component that is an excluded directory name
        if any(part in _EXCLUDED_DIR_NAMES for part in rel.parts):
            continue

        yield path


# ---------------------------------------------------------------------------
# Local-module detection
# ---------------------------------------------------------------------------

def is_local_module(name: str, root: Path) -> bool:
    """
    Return ``True`` if *name* resolves to a local module or package inside
    *root*.

    Rules
    -----
    * **Single-file module**: ``<root>/<name>.py`` exists → local.
    * **Local package**: ``<root>/<name>/`` directory *and*
      ``<root>/<name>/__init__.py`` both exist → local.
    * **Bare directory** (no ``__init__.py``): ``<root>/<name>/`` exists
      but has no ``__init__.py`` → **not** treated as local (avoids
      false-excluding external imports sharing a name with an unrelated
      directory).
    """
    if (root / f"{name}.py").is_file():
        return True
    pkg_dir = root / name
    if pkg_dir.is_dir() and (pkg_dir / "__init__.py").is_file():
        return True
    return False


# ---------------------------------------------------------------------------
# AST import extraction
# ---------------------------------------------------------------------------

def _extract_imports(
    path: Path,
    root: Path,
    stdlib: frozenset,
) -> List[Tuple[str, SourceRef]]:
    """
    Parse *path* with ``ast`` and return a list of
    ``(top_level_module_name, SourceRef)`` tuples for every external import.

    Returns an empty list if the file cannot be parsed (syntax error,
    encoding error), logging a warning instead of raising.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        logger.warning("import_scan: skipping %s — SyntaxError: %s", path, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("import_scan: skipping %s — %s: %s", path, type(exc).__name__, exc)
        return []

    results: List[Tuple[str, SourceRef]] = []
    # Use a relative path for SourceRef.file for portability.
    try:
        rel_file = str(path.relative_to(root))
    except ValueError:
        rel_file = str(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                _maybe_add(top, node.lineno, rel_file, root, stdlib, results)

        elif isinstance(node, ast.ImportFrom):
            # Relative imports (from . import x, from ..pkg import y)
            if node.level and node.level > 0:
                continue
            if node.module is None:
                continue
            top = node.module.split(".")[0]
            _maybe_add(top, node.lineno, rel_file, root, stdlib, results)

    return results


def _maybe_add(
    top: str,
    lineno: int,
    rel_file: str,
    root: Path,
    stdlib: frozenset,
    results: List[Tuple[str, SourceRef]],
) -> None:
    """Filter and append one import candidate to *results*."""
    if not top:
        return
    if top in stdlib:
        return
    if is_local_module(top, root):
        return

    # Reconstruct a plausible raw_line for traceability
    raw = f"import {top}"
    results.append((top, SourceRef(file=rel_file, line=lineno, raw_line=raw)))


# ---------------------------------------------------------------------------
# Aggregation and mapping
# ---------------------------------------------------------------------------

def _map_import(import_name: str) -> Tuple[Optional[str], str]:
    """
    Return ``(guessed_package_name, confidence)`` for *import_name*.

    Looks up the static mapping table first; falls back to using the
    import name itself as the guessed package name.
    """
    pkg = IMPORT_TO_PACKAGE.get(import_name)
    if pkg is not None:
        return pkg, "mapped"
    return import_name, "unmapped_guess"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def scan_imports(root: Path) -> List[InferredDependency]:
    """
    Walk all ``.py`` files under *root* (excluding stdlib, local modules,
    and ignored directories) and return a list of
    :class:`InferredDependency` objects.

    * Each unique top-level import name produces exactly **one** entry.
    * Multiple import sites for the same name are merged into a single
      entry with multiple :class:`SourceRef` entries.
    * The list is sorted by ``import_name`` for deterministic output.

    Parameters
    ----------
    root:
        Absolute path to the repository root.

    Returns
    -------
    List[InferredDependency]
        Sorted list of inferred external dependencies.
    """
    # Accumulate: import_name → list[SourceRef]
    aggregated: Dict[str, List[SourceRef]] = defaultdict(list)

    for py_file in _iter_py_files(root):
        pairs = _extract_imports(py_file, root, _STDLIB)
        for import_name, ref in pairs:
            aggregated[import_name].append(ref)

    deps: List[InferredDependency] = []
    for import_name, refs in sorted(aggregated.items()):
        pkg_name, confidence = _map_import(import_name)
        deps.append(
            InferredDependency(
                import_name=import_name,
                guessed_package_name=pkg_name,
                confidence=confidence,
                sources=refs,
            )
        )

    return deps
