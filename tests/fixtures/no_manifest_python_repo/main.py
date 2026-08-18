"""
tests/fixtures/no_manifest_python_repo/main.py

- from utils import helper  → local single-file module (excluded)
- import myapp              → local package with __init__.py (excluded)
- import numpy              → external, mapped → numpy
- import requests           → external, mapped → requests (also in app.py → aggregation test)
"""
from utils import helper
import myapp

import numpy
import requests
