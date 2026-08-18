"""
tests/fixtures/no_manifest_python_repo/app.py

External imports to test:
- cv2      → opencv-python  (mapped)
- requests → requests       (mapped, also in main.py → aggregation test)
- yaml     → PyYAML         (mapped)
- unusuallib               (unmapped_guess)

Stdlib imports to be filtered:
- os, sys
"""
import os
import sys

import cv2
import requests
import yaml
import unusuallib
