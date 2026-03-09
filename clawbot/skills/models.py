"""Skill data models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SkillMeta:
    """Tier-1 metadata exposed to the model as catalog."""

    name: str
    description: str
    version: str
    source: Path


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """Tier-2 loaded skill instructions."""

    metadata: SkillMeta
    instructions: str
