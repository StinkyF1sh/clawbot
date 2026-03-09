"""Tests for skill discovery, store, and skill tool."""

from pathlib import Path

import pytest

from clawbot.skills.store import SkillPermissionError, SkillStore
from clawbot.tools.skill import SkillTool


def _write_skill(
    root: Path,
    folder: str,
    *,
    name: str,
    description: str,
    body: str,
    version: str = "0.1.0",
) -> None:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"version: {version}\n"
            "---\n\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )


def _create_store(tmp_path: Path, *, allow: list[str] | None = None, deny: list[str] | None = None):
    return SkillStore(
        workspace=tmp_path,
        discovery_paths=[".clawbot/skills", ".opencode/skills", ".agents/skills"],
        include_home=False,
        max_description_length=200,
        allow_patterns=allow if allow is not None else ["*"],
        deny_patterns=deny if deny is not None else [],
    )


class TestSkillStoreDiscovery:
    def test_first_win_precedence_by_discovery_order(self, tmp_path: Path) -> None:
        opencode_root = tmp_path / ".opencode" / "skills"
        claw_root = tmp_path / ".clawbot" / "skills"

        _write_skill(
            opencode_root,
            "shared_opencode",
            name="shared",
            description="from opencode",
            body="opencode body",
        )
        _write_skill(
            claw_root,
            "shared_clawbot",
            name="shared",
            description="from clawbot",
            body="clawbot body",
        )

        store = _create_store(tmp_path)
        available = store.list_available_skills()

        assert len(available) == 1
        assert available[0].name == "shared"
        assert available[0].description == "from clawbot"

    def test_invalid_skill_metadata_is_skipped(self, tmp_path: Path) -> None:
        invalid_dir = tmp_path / ".clawbot" / "skills" / "invalid"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        (invalid_dir / "SKILL.md").write_text("no frontmatter", encoding="utf-8")

        store = _create_store(tmp_path)
        assert store.list_available_skills() == []


class TestSkillStoreLoad:
    def test_load_skill_instructions(self, tmp_path: Path) -> None:
        skill_root = tmp_path / ".clawbot" / "skills"
        _write_skill(
            skill_root,
            "code_review",
            name="code_review",
            description="review code changes",
            body="Use strict review checklist.",
        )

        store = _create_store(tmp_path)
        loaded = store.load_skill("code_review")

        assert loaded.metadata.name == "code_review"
        assert loaded.instructions == "Use strict review checklist."

    def test_deny_has_priority_over_allow(self, tmp_path: Path) -> None:
        skill_root = tmp_path / ".clawbot" / "skills"
        _write_skill(
            skill_root,
            "secret_ops",
            name="secret_ops",
            description="restricted",
            body="restricted body",
        )

        store = _create_store(tmp_path, allow=["*"], deny=["secret_*"])
        with pytest.raises(SkillPermissionError, match="denied by policy"):
            store.load_skill("secret_ops")


class TestSkillTool:
    @pytest.mark.asyncio
    async def test_execute_loads_skill_by_name(self, tmp_path: Path) -> None:
        skill_root = tmp_path / ".clawbot" / "skills"
        _write_skill(
            skill_root,
            "writer",
            name="writer",
            description="writing helper",
            body="Write concise and direct responses.",
        )

        store = _create_store(tmp_path)
        tool = SkillTool(store=store)
        result = await tool.execute(name="writer")

        assert "[Loaded skill: writer]" in result
        assert "Write concise and direct responses." in result

    @pytest.mark.asyncio
    async def test_execute_returns_not_found_error(self, tmp_path: Path) -> None:
        store = _create_store(tmp_path)
        tool = SkillTool(store=store)
        result = await tool.execute(name="missing")

        assert result == "Error: Skill 'missing' not found"
