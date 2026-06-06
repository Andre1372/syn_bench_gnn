"""Utility module for logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_console_logging(project_root: Path, experiment_name: str) -> None:
    """Configures the root python logger.

    Sets up a logger that streams to both stdout and a file located in the 
    'logs' directory of the project root.

    Args:
        project_root: The root directory path of the project.
        experiment_name: The name used as a prefix for the log file.
    """
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{experiment_name}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicate logs in some environments
    if not root_logger.handlers:
        formatter = logging.Formatter('%(asctime)s - [%(name)s:%(lineno)d] - %(levelname)s - %(message)s')
        
        # Console handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        root_logger.addHandler(stdout_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
