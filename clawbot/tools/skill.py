"""Skill tool for progressive skill loading."""

from typing import Any

from clawbot.skills.store import (
    SkillMetadataError,
    SkillNotFoundError,
    SkillPermissionError,
    SkillStore,
)
from clawbot.tools.base import Tool


class SkillTool(Tool):
    """Load skill instructions on demand by skill name."""

    def __init__(self, store: SkillStore):
        super().__init__(workspace=None)
        self.store = store

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        return (
            "Load a skill by name. Use this when you need detailed instructions "
            "from an available skill."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name from <available_skills>.",
                    "minLength": 1,
                }
            },
            "required": ["name"],
        }

    async def execute(self, **kwargs: Any) -> str:
        name = str(kwargs.get("name", "")).strip()
        if not name:
            return "Error: Invalid parameters for tool 'skill': missing required name"

        try:
            loaded = self.store.load_skill(name)
        except SkillNotFoundError:
            return f"Error: Skill '{name}' not found"
        except SkillPermissionError:
            return f"Error: Skill '{name}' denied by policy"
        except SkillMetadataError as exc:
            return f"Error: {exc}"

        return f"[Loaded skill: {loaded.metadata.name}]\n\n{loaded.instructions}"
