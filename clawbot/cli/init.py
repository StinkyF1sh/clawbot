"""CLI initialization and component setup for Clawbot."""

from pathlib import Path
from typing import TYPE_CHECKING

from clawbot.agent.context import ContextBuilder
from clawbot.agent.loop import CliHandler
from clawbot.config.schema import ClawbotConfig
from clawbot.provider.litellm import LiteLLMProvider
from clawbot.provider.openai_compatible import OpenAICompatibleProvider
from clawbot.storage.session import SessionStorage
from clawbot.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from clawbot.tools.registry import ToolRegistry
from clawbot.tools.shell import ExecTool

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


def initialize_tool_registry(workspace: str | Path) -> ToolRegistry:
    """Initialize tool registry with built-in tools."""
    registry = ToolRegistry()

    workspace_path = Path(workspace).expanduser() if isinstance(workspace, str) else workspace

    registry.register(ReadFileTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(WriteFileTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(EditFileTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(ListDirTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(ExecTool(working_dir=str(workspace_path), restrict_to_workspace=True))

    return registry


def create_cli_handler(config: ClawbotConfig) -> CliHandler:
    """Create CLI handler with all required components."""
    storage = SessionStorage(workspace=config.workspace_path)
    context_builder = ContextBuilder(storage=storage, default_workspace=str(config.workspace_path))
    providers = initialize_providers(config)
    tool_registry = initialize_tool_registry(config.workspace_path)

    return CliHandler(
        global_config=config,
        storage=storage,
        context_builder=context_builder,
        providers=providers,
        tool_registry=tool_registry,
    )
