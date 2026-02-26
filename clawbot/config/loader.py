"""Configuration loader for Clawbot."""

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings.sources import PydanticBaseSettingsSource

from clawbot.config.schema import ClawbotConfig


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom YAML settings source for pydantic-settings."""

    def __init__(self, yaml_file: Path | None) -> None:
        self.yaml_file = Path(yaml_file) if yaml_file else None

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        """Get field value from YAML."""
        if self.yaml_file and self.yaml_file.exists():
            with open(self.yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            field_value = data.get(field_name)
            return field_value, field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if self.yaml_file and self.yaml_file.exists():
            with open(self.yaml_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {k: v for k, v in data.items() if v is not None}
        return {}


def find_config_file() -> Path | None:
    """Find configuration file in standard locations.

    Search order:
    1. Current directory: ./clawbot.yaml
    2. Current directory: ./clawbot.yml
    3. User config directory: ~/.clawbot/config.yaml
    4. User config directory: ~/.clawbot/config.yml
    """
    search_paths = [
        Path.cwd() / "clawbot.yaml",
        Path.cwd() / "clawbot.yml",
        Path.home() / ".clawbot" / "config.yaml",
        Path.home() / ".clawbot" / "config.yml",
    ]

    for path in search_paths:
        if path.exists():
            return path
    return None


def load_config(
    config_file: str | Path | None = None,
    *,
    ensure_workspace: bool = False,
) -> ClawbotConfig:
    """Load Clawbot configuration."""
    if config_file is None:
        config_file = find_config_file()

    if config_file is not None and not Path(config_file).exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load configuration using custom settings sources
    try:
        yaml_source = YamlConfigSettingsSource(config_file)

        class _ClawbotConfigWithYaml(ClawbotConfig):
            @classmethod
            def settings_customise_sources(
                cls,
                settings_cls: type[ClawbotConfig],
                init_settings: PydanticBaseSettingsSource,
                env_settings: PydanticBaseSettingsSource,
                dotenv_settings: PydanticBaseSettingsSource,
                file_secret_settings: PydanticBaseSettingsSource,
            ):
                return (
                    init_settings,  # Constructor arguments (highest priority)
                    env_settings,  # Environment variables
                    dotenv_settings,  # .env file
                    yaml_source,  # YAML config file
                    file_secret_settings,
                )

        config = _ClawbotConfigWithYaml()
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}") from e

    # Optionally ensure workspace directory exists
    if ensure_workspace:
        from clawbot.util.utils import ensure_directory

        ensure_directory(config.workspace_path)

    return config
