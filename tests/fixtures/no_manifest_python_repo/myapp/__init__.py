# tests/fixtures/no_manifest_python_repo/myapp/__init__.py
# This is a LOCAL PACKAGE (directory + __init__.py).
# Any `import myapp` in the repo should be excluded from inferred dependencies.

APP_NAME = "myapp"
