"""CLI initialization and component setup for Clawbot."""

from typing import TYPE_CHECKING

from clawbot.agent.context import ContextBuilder
from clawbot.agent.loop import CliHandler
from clawbot.config.schema import ClawbotConfig
from clawbot.provider.litellm import LiteLLMProvider
from clawbot.provider.openai_compatible import OpenAICompatibleProvider
from clawbot.storage.session import SessionStorage

if TYPE_CHECKING:
    from clawbot.provider.base import BaseProvider


def initialize_providers(config: ClawbotConfig) -> dict[str, "BaseProvider"]:
    """Initialize LLM providers from configuration."""
    providers: dict[str, "BaseProvider"] = {}

    provider_configs = {
        "openai": config.providers.openai,
        "zhipu": config.providers.zhipu,
        "bailian": config.providers.bailian,
    }

    for provider_name, provider_config in provider_configs.items():
        if provider_config.api_key:
            providers[provider_name] = OpenAICompatibleProvider(
                api_key=provider_config.api_key,
                api_base_url=provider_config.api_base,
                default_model=config.get_agent_config().model,
            )

    if not providers:
        providers["default"] = LiteLLMProvider()

    return providers


def create_cli_handler(config: ClawbotConfig) -> CliHandler:
    """Create CLI handler with all required components."""
    storage = SessionStorage(workspace=config.workspace_path)
    context_builder = ContextBuilder(storage=storage, default_workspace=str(config.workspace_path))
    providers = initialize_providers(config)

    return CliHandler(
        global_config=config,
        storage=storage,
        context_builder=context_builder,
        providers=providers,
    )
