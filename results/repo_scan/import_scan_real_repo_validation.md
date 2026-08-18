# Real-Repo Validation & PyPI Spot-Check

**Methodology:**
Validated the AST-based import scanner against 3 real public no-manifest Python repos, plus a 15-entry random spot-check of the mapping table against the live PyPI API.

**Results:**
- **Real-Repo Validation**: Correctly inferred and mapped dependencies (e.g., `cv2` → `opencv-python`) across all 3 repos. Correctly filtered stdlib and local imports.
- **False Positives**: 1 found (`urlparse`, which is valid stdlib in Python 2, was flagged as external when scanned under a Python 3 interpreter).
- **PyPI Spot-Check**: 15/15 randomly sampled mapping-table entries (out of ~200 total) were confirmed valid against PyPI.
- **Unmapped Guesses**: Several `unmapped_guess` entries in real repos (e.g., `tweepy`, `twitter`, `hurry`, `wand`) resolved correctly because the import name equals the package name. (Note: unmapped-guess accuracy was not separately measured).
