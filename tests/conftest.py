"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def sample_config_dict() -> dict:
    """Sample configuration dictionary."""
    return {
        "agents": {
            "default": {
                "workspace": "~/.clawbot/workspace",
                "model": "glm-4.7",
                "max_tokens": 8192,
                "temperature": 0.1,
                "max_steps": 40,
                "memory_window": 100,
            },
            "code": {"model": "gpt-4", "temperature": 0.0},
        },
        "providers": {
            "openai": {
                "api_key": "sk-test",
                "api_base": "https://api.openai.com/v1",
                "extra_headers": None,
            }
        },
    }


@pytest.fixture
def example_config_path() -> Path:
    """Path to the example configuration file."""
    return Path(__file__).parent.parent / "clawbot.yaml.example"
