"""Utility functions for Clawbot."""

from pathlib import Path


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Returns:
        The Path object for the directory.

    Raises:
        ValueError: If path exists but is not a directory.
    """
    path_obj = Path(path).expanduser()
    if path_obj.exists() and not path_obj.is_dir():
        raise ValueError(f"Path exists but is not a directory: {path_obj}")
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
