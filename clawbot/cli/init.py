"""CLI initialization and component setup for Clawbot."""

from pathlib import Path
from typing import TYPE_CHECKING

from clawbot.agent.context import ContextBuilder
from clawbot.agent.loop import CliHandler
from clawbot.config.schema import ClawbotConfig
from clawbot.provider.litellm import LiteLLMProvider
from clawbot.provider.openai_compatible import OpenAICompatibleProvider
from clawbot.skills.store import SkillStore
from clawbot.storage.session import SessionStorage
from clawbot.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from clawbot.tools.registry import ToolRegistry
from clawbot.tools.shell import ExecTool
from clawbot.tools.skill import SkillTool

if TYPE_CHECKING:
    from clawbot.agent.config import AgentRuntimeConfig
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


def initialize_tool_registry(
    workspace: str | Path,
    skill_store: SkillStore | None = None,
) -> ToolRegistry:
    """Initialize tool registry with built-in tools."""
    registry = ToolRegistry()

    workspace_path = Path(workspace).expanduser() if isinstance(workspace, str) else workspace

    registry.register(ReadFileTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(WriteFileTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(EditFileTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(ListDirTool(workspace=workspace_path, allowed_dir=workspace_path))
    registry.register(ExecTool(working_dir=str(workspace_path), restrict_to_workspace=True))
    if skill_store:
        registry.register(SkillTool(store=skill_store))

    return registry


def create_cli_handler(config: ClawbotConfig) -> CliHandler:
    """Create CLI handler with all required components."""
    storage = SessionStorage(workspace=config.workspace_path)
    skill_store_cache: dict[str, SkillStore] = {}

    def resolve_skill_store(workspace: str | Path) -> SkillStore | None:
        if not config.skills.enabled:
            return None

        workspace_path = Path(workspace).expanduser()
        cache_key = str(workspace_path)
        if cache_key not in skill_store_cache:
            skill_store_cache[cache_key] = SkillStore(
                workspace=workspace_path,
                discovery_paths=config.skills.discovery_paths,
                include_home=config.skills.include_home,
                max_description_length=config.skills.max_description_length,
                allow_patterns=config.permission.skills.allow,
                deny_patterns=config.permission.skills.deny,
            )
        return skill_store_cache[cache_key]

    def skill_catalog_provider(workspace: str) -> list[tuple[str, str]]:
        store = resolve_skill_store(workspace)
        if store is None:
            return []
        return store.get_catalog_entries()

    context_builder = ContextBuilder(
        storage=storage,
        default_workspace=str(config.workspace_path),
        skill_catalog_provider=skill_catalog_provider,
    )
    providers = initialize_providers(config)
    tool_registry = initialize_tool_registry(
        config.workspace_path,
        skill_store=resolve_skill_store(config.workspace_path),
    )

    def tool_registry_factory(agent_config: "AgentRuntimeConfig") -> ToolRegistry:
        return initialize_tool_registry(
            agent_config.workspace,
            skill_store=resolve_skill_store(agent_config.workspace),
        )

    return CliHandler(
        global_config=config,
        storage=storage,
        context_builder=context_builder,
        providers=providers,
        tool_registry=tool_registry,
        tool_registry_factory=tool_registry_factory,
    )
