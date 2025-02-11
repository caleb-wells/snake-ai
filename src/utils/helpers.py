from pathlib import Path
from typing import Dict


def setup_training_dirs() -> Dict[str, str]:
    dirs = {"checkpoints": "checkpoints", "logs": "logs"}
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dirs
