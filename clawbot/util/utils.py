"""Utility functions for Clawbot."""

import uuid
from pathlib import Path

# Async loop polling interval (seconds)
POLL_INTERVAL = 0.1
# Queue operation timeout (seconds)
QUEUE_TIMEOUT = 0.1


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return uuid.uuid4().hex


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path_obj = Path(path).expanduser()
    if path_obj.exists() and not path_obj.is_dir():
        raise ValueError(f"Path exists but is not a directory: {path_obj}")
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
