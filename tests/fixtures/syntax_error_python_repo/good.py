# tests/fixtures/syntax_error_python_repo/good.py
# Valid Python. import_scan must collect 'requests' from this file
# even though bad.py in the same repo has a SyntaxError.

import requests
