"Configuration schema definitions for Clawbot."

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

class AgentDefaults(Base):
    """Default agent configuration."""

    workspace: str = "~/.clawbot/workspace"
    model: str = "zhipu/glm-4.7"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100

class AgentsConfig(RootModel[dict[str, AgentDefaults]]):
    """Agent configurations mapping."""

    def __getitem__(self, key: str) -> AgentDefaults:
        return self.root[key]

    def get(self, key: str, default: AgentDefaults | None = None) -> AgentDefaults | None:
        return self.root.get(key, default)

    def keys(self):
        return self.root.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.root


class ProviderConfig(Base):
    """LLM provider configuration."""

    api_key: str = ""
    api_base: str | None = None
    extra_headers: dict[str, str] | None = None

class ProvidersConfig(Base):
    """LLM providers configuration."""

    openai: ProviderConfig = Field(default_factory=ProviderConfig) # OpenAI-compatible endpoint
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig) # Zhipu AI endpoint
    bailian: ProviderConfig = Field(default_factory=ProviderConfig) # Bailian endpoint

class ClawbotConfig(BaseSettings):
    """Main configuration for Clawbot."""

    agents: AgentsConfig = Field(
        default_factory=lambda: AgentsConfig.model_validate({"default": AgentDefaults()})
    )
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)

    def get_agent_config(self, agent_name: str = "default") -> AgentDefaults:
        """Get agent configuration by name."""
        # First try to get the requested agent directly
        if agent_name in self.agents:
            return self.agents[agent_name]
        # Fall back to "default" agent if it exists
        if "default" in self.agents:
            return self.agents["default"]
        # Last resort: return a new instance with defaults
        return AgentDefaults()

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path for the default agent."""
        return Path(self.get_agent_config().workspace).expanduser()


    def _parse_provider_name(self, model: str) -> str:
        """Extract provider name from model string."""
        if not model:
            agent_config = self.get_agent_config()
            model = agent_config.model

        model_lower = model.lower()
        provider = model_lower.split("/")[0].replace("-", "_")

        if hasattr(self.providers, provider):
            return provider
        return "openai"  # Default to openai for unknown/bare model names

    def get_provider(self, model: str) -> ProviderConfig | None:
        """Get provider configuration by name."""
        provider_name = self._parse_provider_name(model)
        return getattr(self.providers, provider_name, None)


    def get_provider_name(self, model: str) -> str:
        """Get provider name by model."""
        return self._parse_provider_name(model)

    def get_api_base(self, model: str) -> str | None:
        """Get API base URL for the provider."""
        provider = self.get_provider(model)
        return provider.api_base if provider else None
    def get_api_key(self, model: str) -> str:
        """Get API key for the provider."""
        provider = self.get_provider(model)
        return provider.api_key if provider else ""

    model_config = ConfigDict(env_prefix="CLAWBOT_", env_nested_delimiter="__")
