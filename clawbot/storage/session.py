"""Session history storage for the Clawbot agent."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clawbot.util.utils import ensure_directory, generate_session_id

if TYPE_CHECKING:
    pass


META_CREATED_AT_PREFIX = "# clawbot_session_created_at:"


@dataclass
class SessionConfig:
    """Session storage configuration."""

    workspace: str
    session_dir: str | None = None
    max_window: int = 100

    @property
    def resolved_session_dir(self) -> Path:
        """Resolve session directory path."""
        if self.session_dir:
            return Path(self.session_dir).expanduser()
        return Path(self.workspace).expanduser() / ".session"


@dataclass
class SessionMeta:
    """Session metadata."""

    session_id: str
    created_at: float
    updated_at: float
    message_count: int
    file_path: str


class SessionStorage:
    """Session storage manager using JSONL format."""

    def __init__(
        self,
        config: SessionConfig | None = None,
        workspace: str | Path | None = None,
    ):
        """Initialize storage manager."""
        if config:
            self._config = config
            self.session_dir = config.resolved_session_dir
        elif workspace:
            workspace_path = Path(workspace).expanduser()
            self._config = SessionConfig(workspace=str(workspace_path))
            self.session_dir = workspace_path / ".session"
        else:
            raise ValueError("Must provide config or workspace parameter")

        ensure_directory(self.session_dir)

    def _get_session_path(self, session_id: str) -> Path:
        """Get session file path."""
        return self.session_dir / f"{session_id}.jsonl"

    def _read_created_at_from_file(self, session_id: str) -> float | None:
        """Read creation timestamp from file."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line.startswith(META_CREATED_AT_PREFIX):
                    try:
                        return float(first_line[len(META_CREATED_AT_PREFIX):].strip())
                    except (ValueError, IndexError):
                        pass
        except (IOError, OSError):
            pass
        return None

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        """Append a message to session file."""
        path = self._get_session_path(session_id)
        try:
            if not path.exists() or path.stat().st_size == 0:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"{META_CREATED_AT_PREFIX} {time.time()}\n")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message, ensure_ascii=False) + "\n")
        except (IOError, OSError):
            pass

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all messages from session."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return []

        messages: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except (IOError, OSError):
            return []
        return messages

    def delete_session(self, session_id: str) -> bool:
        """Delete session file."""
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[str]:
        """List all session IDs sorted by update time (descending)."""
        sessions = []
        for path in self.session_dir.glob("*.jsonl"):
            session_id = path.stem
            meta = self.get_session_meta(session_id)
            if meta:
                sessions.append((meta.updated_at, session_id))
        sessions.sort(reverse=True)
        return [sid for _, sid in sessions]

    def get_session_meta(self, session_id: str) -> SessionMeta | None:
        """Get session metadata."""
        path = self._get_session_path(session_id)
        if not path.exists():
            return None

        try:
            stat = path.stat()
            message_count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        message_count += 1

            created_at = self._read_created_at_from_file(session_id)
            if created_at is None:
                created_at = getattr(stat, "st_birthtime", stat.st_ctime)

            return SessionMeta(
                session_id=session_id,
                created_at=created_at,
                updated_at=stat.st_mtime,
                message_count=message_count,
                file_path=str(path),
            )
        except (IOError, OSError):
            return None

    def create_session(self, session_id: str | None = None) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = generate_session_id()

        path = self._get_session_path(session_id)
        try:
            if not path.exists():
                path.touch()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"{META_CREATED_AT_PREFIX} {time.time()}\n")
        except (IOError, OSError):
            pass

        return session_id

    def append_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Append multiple messages to session."""
        path = self._get_session_path(session_id)
        try:
            if not path.exists() or path.stat().st_size == 0:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"{META_CREATED_AT_PREFIX} {time.time()}\n")
            with open(path, "a", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except (IOError, OSError):
            pass

    def truncate_session(
        self,
        session_id: str,
        max_messages: int,
    ) -> list[dict[str, Any]]:
        """Truncate session to keep only latest N messages."""
        messages = self.load_session(session_id)
        if len(messages) <= max_messages:
            return []

        removed = messages[:-max_messages]
        kept = messages[-max_messages:]

        path = self._get_session_path(session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                created_at = self._read_created_at_from_file(session_id)
                if created_at is None:
                    created_at = time.time()
                f.write(f"{META_CREATED_AT_PREFIX} {created_at}\n")
                for msg in kept:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except (IOError, OSError):
            return []

        return removed
