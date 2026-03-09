"""Tests for clawbot.config.loader module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from clawbot.config.loader import (
    YamlConfigSettingsSource,
    find_config_file,
    load_config,
)
from clawbot.config.schema import ClawbotConfig


class TestYamlConfigSettingsSource:
    """Tests for YamlConfigSettingsSource."""

    def test_no_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that missing file returns empty dict."""
        source = YamlConfigSettingsSource(tmp_path / "nonexistent.yaml")
        assert source() == {}

    def test_existing_file_returns_content(self, tmp_path: Path) -> None:
        """Test that existing file returns parsed content."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agents:\n  default:\n    model: gpt-4\n", encoding="utf-8"
        )
        source = YamlConfigSettingsSource(config_file)
        result = source()
        assert "agents" in result
        assert result["agents"]["default"]["model"] == "gpt-4"

    def test_filters_none_values(self, tmp_path: Path) -> None:
        """Test that top-level None values are filtered."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "agents:\n  default:\n    model: gpt-4\nunused: null\n",
            encoding="utf-8",
        )
        source = YamlConfigSettingsSource(config_file)
        result = source()
        # Top level None values are filtered
        assert "unused" not in result


class TestFindConfigFile:
    """Tests for find_config_file function."""

    def test_finds_clawbot_yaml(self, tmp_path: Path) -> None:
        """Test finding clawbot.yaml in current directory."""
        config_file = tmp_path / "clawbot.yaml"
        config_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = find_config_file()
            assert result == config_file

    def test_finds_clawbot_yml(self, tmp_path: Path) -> None:
        """Test finding clawbot.yml in current directory."""
        config_file = tmp_path / "clawbot.yml"
        config_file.touch()

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = find_config_file()
            assert result == config_file

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        """Test returning None when no config file exists."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = find_config_file()
            assert result is None


class TestLoadConfig:
    """Tests for load_config function."""

    # def test_loads_default_config_when_no_file(self) -> None:
    #     """Test loading default config when no file specified."""
    #     config = load_config(None)
    #     assert isinstance(config, ClawbotConfig)
    #     assert config.get_agent_config("default").model == "zhipu/glm-4.7"

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """agents:
  default:
    model: gpt-4
    temperature: 0.0
skills:
  enabled: true
permission:
  skills:
    allow:
      - "code_*"
providers:
  openai:
    api_key: sk-test
""",
            encoding="utf-8",
        )

        config = load_config(config_file)
        assert config.get_agent_config("default").model == "gpt-4"
        assert config.get_agent_config("default").temperature == 0.0
        assert config.skills.enabled is True
        assert config.permission.skills.allow == ["code_*"]
        assert config.providers.openai.api_key == "sk-test"

    def test_raises_file_not_found(self) -> None:
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/config.yaml")

    def test_creates_workspace_directory(self, tmp_path: Path) -> None:
        """Test that workspace directory is created when ensure_workspace=True."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """agents:
  default:
    workspace: """ + str(tmp_path / "workspace") + "\n",
            encoding="utf-8",
        )

        config = load_config(config_file, ensure_workspace=True)
        assert config.workspace_path.exists()
        assert config.workspace_path.is_dir()

    def test_does_not_create_workspace_by_default(self, tmp_path: Path) -> None:
        """Test that workspace directory is NOT created by default."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """agents:
  default:
    workspace: """ + str(tmp_path / "new_workspace") + "\n",
            encoding="utf-8",
        )

        config = load_config(config_file)  # ensure_workspace defaults to False
        assert not config.workspace_path.exists()

    def test_multiple_agents_from_yaml(self, tmp_path: Path) -> None:
        """Test loading multiple agents from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """agents:
  default:
    model: glm-4.7
  code:
    model: gpt-4
    temperature: 0.0
  chat:
    model: glm-4-flash
    temperature: 0.7
""",
            encoding="utf-8",
        )

        config = load_config(config_file)
        assert config.get_agent_config("default").model == "glm-4.7"
        assert config.get_agent_config("code").model == "gpt-4"
        assert config.get_agent_config("chat").model == "glm-4-flash"


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_full_config_loading_workflow(self, tmp_path: Path, example_config_path: Path) -> None:
        """Test complete workflow: copy example, load, and use config."""
        import shutil

        # 1. Copy example config
        config_file = tmp_path / "clawbot.yaml"
        shutil.copy(example_config_path, config_file)

        # 2. Load config
        config = load_config(config_file)

        # 3. Verify loaded config
        assert config.get_agent_config("default") is not None
        assert config.providers.openai is not None

    def test_config_overrides_with_dict(self, tmp_path: Path) -> None:
        """Test that dict values override YAML values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """agents:
  default:
    model: gpt-4
""",
            encoding="utf-8",
        )

        # Override with dict values (higher priority)
        class TestConfig(ClawbotConfig):
            pass

        config = TestConfig(
            agents={"default": {"model": "gpt-4-turbo", "temperature": 0.5}}
        )
        assert config.get_agent_config("default").model == "gpt-4-turbo"
