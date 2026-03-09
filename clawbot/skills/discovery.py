"""Skill discovery and parsing utilities."""

import re
from pathlib import Path
from typing import Any

import yaml

from clawbot.skills.models import SkillMeta

_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def resolve_discovery_roots(
    workspace: str | Path,
    discovery_paths: list[str],
    include_home: bool,
) -> list[Path]:
    """Resolve skill discovery roots in a stable order."""
    workspace_root = Path(workspace).expanduser()
    roots: list[Path] = []
    seen: set[str] = set()

    for path_text in discovery_paths:
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = workspace_root / path
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)

    if include_home:
        home_root = Path("~/.clawbot/skills").expanduser()
        key = str(home_root)
        if key not in seen:
            roots.append(home_root)

    return roots


def parse_skill_document(skill_file: Path) -> tuple[dict[str, Any] | None, str]:
    """Parse SKILL.md frontmatter and body."""
    try:
        content = skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, ""

    lines = content.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return None, ""

    end_idx = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = idx
            break

    if end_idx is None:
        return None, ""

    frontmatter_raw = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).strip()
    if not body:
        return None, ""

    try:
        parsed = yaml.safe_load(frontmatter_raw) or {}
    except yaml.YAMLError:
        return None, ""

    if not isinstance(parsed, dict):
        return None, ""

    return parsed, body


def parse_skill_metadata(
    skill_file: Path,
    max_description_length: int,
) -> SkillMeta | None:
    """Parse and validate skill metadata from SKILL.md."""
    metadata, _ = parse_skill_document(skill_file)
    if metadata is None:
        return None

    name = str(metadata.get("name", "")).strip()
    description = str(metadata.get("description", "")).strip()
    version = str(metadata.get("version", "")).strip() or "0.1.0"

    if not name or not _SKILL_NAME_PATTERN.fullmatch(name):
        return None
    if not description or len(description) > max_description_length:
        return None

    return SkillMeta(
        name=name,
        description=description,
        version=version,
        source=skill_file,
    )


def discover_skills(
    workspace: str | Path,
    discovery_paths: list[str],
    include_home: bool,
    max_description_length: int,
) -> dict[str, SkillMeta]:
    """Discover skills from known roots and return first-win catalog by name."""
    catalog: dict[str, SkillMeta] = {}
    roots = resolve_discovery_roots(workspace, discovery_paths, include_home)

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue

        try:
            children = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
        except OSError:
            continue

        for child in children:
            skill_file = child / "SKILL.md"
            if not skill_file.exists() or not skill_file.is_file():
                continue

            meta = parse_skill_metadata(
                skill_file=skill_file,
                max_description_length=max_description_length,
            )
            if meta is None:
                continue

            if meta.name in catalog:
                continue
            catalog[meta.name] = meta

    return catalog
