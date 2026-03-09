"""Skill store with discovery, permission checks, and tiered loading."""

from fnmatch import fnmatchcase
from pathlib import Path

from clawbot.skills.discovery import discover_skills, parse_skill_document
from clawbot.skills.models import LoadedSkill, SkillMeta


class SkillNotFoundError(ValueError):
    """Raised when a skill is not found in catalog."""


class SkillPermissionError(PermissionError):
    """Raised when a skill is blocked by policy."""


class SkillMetadataError(ValueError):
    """Raised when a skill file is malformed."""


class SkillStore:
    """Discover and load skills with progressive loading semantics."""

    def __init__(
        self,
        workspace: str | Path,
        discovery_paths: list[str],
        include_home: bool,
        max_description_length: int,
        allow_patterns: list[str],
        deny_patterns: list[str],
    ):
        self.workspace = Path(workspace).expanduser()
        self.discovery_paths = list(discovery_paths)
        self.include_home = include_home
        self.max_description_length = max_description_length
        self.allow_patterns = list(allow_patterns)
        self.deny_patterns = list(deny_patterns)

        self._catalog: dict[str, SkillMeta] | None = None
        self._loaded: dict[str, LoadedSkill] = {}

    def _ensure_catalog(self) -> None:
        if self._catalog is None:
            self._catalog = discover_skills(
                workspace=self.workspace,
                discovery_paths=self.discovery_paths,
                include_home=self.include_home,
                max_description_length=self.max_description_length,
            )

    def refresh_catalog(self) -> None:
        """Refresh catalog and clear loaded tier-2 cache."""
        self._catalog = discover_skills(
            workspace=self.workspace,
            discovery_paths=self.discovery_paths,
            include_home=self.include_home,
            max_description_length=self.max_description_length,
        )
        self._loaded.clear()

    def is_allowed(self, name: str) -> bool:
        """Check permission with deny-first matching."""
        if any(fnmatchcase(name, pattern) for pattern in self.deny_patterns):
            return False

        if not self.allow_patterns:
            return False

        return any(fnmatchcase(name, pattern) for pattern in self.allow_patterns)

    def list_available_skills(self) -> list[SkillMeta]:
        """Return tier-1 skill catalog filtered by permission."""
        self._ensure_catalog()
        assert self._catalog is not None
        return [
            self._catalog[name]
            for name in sorted(self._catalog.keys())
            if self.is_allowed(name)
        ]

    def get_catalog_entries(self) -> list[tuple[str, str]]:
        """Return (name, description) pairs for prompt injection."""
        return [
            (meta.name, meta.description)
            for meta in self.list_available_skills()
        ]

    def load_skill(self, name: str) -> LoadedSkill:
        """Load tier-2 skill instructions by name."""
        self._ensure_catalog()
        assert self._catalog is not None

        if not self.is_allowed(name):
            raise SkillPermissionError(f"Skill '{name}' denied by policy")

        meta = self._catalog.get(name)
        if meta is None:
            raise SkillNotFoundError(f"Skill '{name}' not found")

        cached = self._loaded.get(name)
        if cached is not None:
            return cached

        frontmatter, body = parse_skill_document(meta.source)
        if frontmatter is None or not body:
            raise SkillMetadataError(f"Invalid skill metadata in '{meta.source}'")

        loaded = LoadedSkill(metadata=meta, instructions=body)
        self._loaded[name] = loaded
        return loaded
