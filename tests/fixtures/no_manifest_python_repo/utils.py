# tests/fixtures/no_manifest_python_repo/utils.py
# This is a LOCAL single-file module.
# Any `import utils` in the repo should be excluded from inferred dependencies.

def helper(x):
    return x * 2
