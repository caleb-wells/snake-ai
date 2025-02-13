"""
Configuration module for pytest testing setup.

This module adds the project root directory to the Python path, ensuring that
modules can be imported correctly during testing. It modifies the sys.path to include
the parent directory of the tests folder.

The module is automatically loaded by pytest when running tests, as it follows
the pytest convention of using conftest.py for shared test configuration.
"""

import os
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
