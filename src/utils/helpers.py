"""
This module provides helper functions for setting up the training environment
in the Snake AI project, such as creating necessary directories.
"""

from pathlib import Path
from typing import Dict


def setup_training_dirs() -> Dict[str, str]:
    """
    Create and ensure the existence of directories for training checkpoints and logs.

    This function sets up the required directories by creating them if they do not exist.
    It returns a dictionary containing the paths for the checkpoints and logs directories.

    Returns:
        Dict[str, str]: A dictionary with keys 'checkpoints' and 'logs' mapping to their
        directory paths.
    """
    dirs = {"checkpoints": "checkpoints", "logs": "logs"}
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dirs
