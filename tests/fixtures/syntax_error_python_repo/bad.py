# tests/fixtures/syntax_error_python_repo/bad.py
# This file contains an intentional SyntaxError.
# import_scan must log a warning and skip this file without crashing.

def broken(:
    pass
