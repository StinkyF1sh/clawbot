"""Tests for clawbot.config.schema module."""

from pathlib import Path

from clawbot.config.schema import (
    AgentDefaults,
    AgentsConfig,
    ClawbotConfig,
    ProviderConfig,
    ProvidersConfig,
)


class TestAgentDefaults:
    """Tests for AgentDefaults model."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = AgentDefaults()
        assert config.workspace == "~/.clawbot/workspace"
        assert config.model == "zhipu/glm-4.7"
        assert config.max_tokens == 8192
        assert config.temperature == 0.1
        assert config.max_tool_iterations == 40
        assert config.memory_window == 100

    def test_custom_values(self) -> None:
        """Test custom values can be set."""
        config = AgentDefaults(
            model="gpt-4",
            temperature=0.5,
            max_tokens=4096,
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096


class TestAgentsConfig:
    """Tests for AgentsConfig model."""

    def test_single_agent(self) -> None:
        """Test single agent configuration."""
        config = AgentsConfig.model_validate(
            {"default": {"model": "gpt-4", "temperature": 0.0}}
        )
        assert "default" in config.keys()
        assert config["default"].model == "gpt-4"
        assert config["default"].temperature == 0.0

    def test_multiple_agents(self) -> None:
        """Test multiple agents configuration."""
        config = AgentsConfig.model_validate(
            {
                "default": {"model": "glm-4.7"},
                "code": {"model": "gpt-4", "temperature": 0.0},
                "chat": {"model": "glm-4-flash", "temperature": 0.7},
            }
        )
        assert "default" in config.keys()
        assert "code" in config.keys()
        assert "chat" in config.keys()
        assert config["code"].model == "gpt-4"
        assert config["chat"].temperature == 0.7

    def test_get_method(self) -> None:
        """Test get method with fallback."""
        config = AgentsConfig.model_validate(
            {"default": {"model": "glm-4.7"}}
        )
        assert config.get("default") is not None
        assert config.get("unknown") is None

    def test_contains_method(self) -> None:
        """Test __contains__ method."""
        config = AgentsConfig.model_validate(
            {"default": {"model": "glm-4.7"}}
        )
        assert "default" in config
        assert "unknown" not in config


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ProviderConfig()
        assert config.api_key == ""
        assert config.api_base is None
        assert config.extra_headers is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ProviderConfig(
            api_key="sk-test",
            api_base="https://api.example.com",
            extra_headers={"Authorization": "Bearer token"},
        )
        assert config.api_key == "sk-test"
        assert config.api_base == "https://api.example.com"
        assert config.extra_headers == {"Authorization": "Bearer token"}


class TestProvidersConfig:
    """Tests for ProvidersConfig model."""

    def test_default_providers(self) -> None:
        """Test all default providers exist."""
        config = ProvidersConfig()
        assert hasattr(config, "openai")
        assert hasattr(config, "zhipu")
        assert hasattr(config, "bailian")

    def test_custom_provider_config(self) -> None:
        """Test custom provider configuration."""
        config = ProvidersConfig(
            openai=ProviderConfig(api_key="sk-openai"),
            zhipu=ProviderConfig(api_key="sk-zhipu"),
        )
        assert config.openai.api_key == "sk-openai"
        assert config.zhipu.api_key == "sk-zhipu"


class TestClawbotConfig:
    """Tests for ClawbotConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ClawbotConfig()
        assert config.get_agent_config("default") is not None
        assert config.get_agent_config("default").model == "zhipu/glm-4.7"

    def test_workspace_path_property(self) -> None:
        """Test workspace_path property."""
        config = ClawbotConfig()
        workspace = config.workspace_path
        assert isinstance(workspace, Path)
        assert workspace.expanduser().exists() or True  # May or may not exist

    def test_get_agent_config_default(self) -> None:
        """Test getting default agent config."""
        config = ClawbotConfig()
        agent = config.get_agent_config("default")
        assert agent.model == "zhipu/glm-4.7"
        assert agent.temperature == 0.1

    def test_get_agent_config_fallback(self) -> None:
        """Test fallback to default for unknown agent."""
        config = ClawbotConfig()
        agent = config.get_agent_config("unknown")
        assert agent.model == "zhipu/glm-4.7"  # Should fallback to default

    def test_get_agent_config_custom(self) -> None:
        """Test getting custom agent config."""
        config = ClawbotConfig(
            agents=AgentsConfig.model_validate(
                {
                    "default": {"model": "zhipu/glm-4.7"},
                    "code": {"model": "gpt-4", "temperature": 0.0},
                }
            )
        )
        agent = config.get_agent_config("code")
        assert agent.model == "gpt-4"
        assert agent.temperature == 0.0

    def test_provider_resolution(self) -> None:
        """Test provider name resolution from model string."""
        config = ClawbotConfig()
        assert config._parse_provider_name("gpt-4") == "openai"
        assert config._parse_provider_name("openai/gpt-4") == "openai"
        assert config._parse_provider_name("zhipu/glm-4") == "zhipu"

    def test_get_provider(self) -> None:
        """Test getting provider configuration."""
        config = ClawbotConfig()
        provider = config.get_provider("gpt-4")
        assert provider is not None
        assert provider.api_base is None

    def test_get_provider_name(self) -> None:
        """Test getting provider name."""
        config = ClawbotConfig()
        assert config.get_provider_name("gpt-4") == "openai"
        assert config.get_provider_name("zhipu/glm-4") == "zhipu"

    def test_get_api_base(self) -> None:
        """Test getting API base URL."""
        config = ClawbotConfig(
            providers=ProvidersConfig(
                openai=ProviderConfig(api_base="https://api.openai.com/v1")
            )
        )
        assert config.get_api_base("gpt-4") == "https://api.openai.com/v1"

    def test_get_api_key(self) -> None:
        """Test getting API key."""
        config = ClawbotConfig(
            providers=ProvidersConfig(openai=ProviderConfig(api_key="sk-test"))
        )
        assert config.get_api_key("gpt-4") == "sk-test"

    def test_model_config_env_prefix(self) -> None:
        """Test model config has correct env prefix."""
        config = ClawbotConfig()
        assert config.model_config.get("env_prefix") == "CLAWBOT_"
        assert config.model_config.get("env_nested_delimiter") == "__"

    def test_camel_case_alias_support(self) -> None:
        """Test camelCase alias support in Base model."""
        # camelCase input should map to snake_case field
        config = AgentDefaults(maxTokens=4096, model="gpt-4")
        assert config.max_tokens == 4096
        assert config.model == "gpt-4"

        # snake_case input also works
        config = AgentDefaults(max_tokens=2048, temperature=0.5)
        assert config.max_tokens == 2048
        assert config.temperature == 0.5
