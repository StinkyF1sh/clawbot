"""Tests for session storage module."""

from pathlib import Path

from clawbot.storage.session import (
    META_CREATED_AT_PREFIX,
    SessionConfig,
    SessionStorage,
)


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_session_dir(self) -> None:
        """Test default session directory resolution."""
        config = SessionConfig(workspace="/tmp/workspace")
        assert config.resolved_session_dir == Path("/tmp/workspace/.session")

    def test_custom_session_dir(self) -> None:
        """Test custom session directory."""
        config = SessionConfig(
            workspace="/tmp/workspace",
            session_dir="/custom/sessions",
        )
        assert config.resolved_session_dir == Path("/custom/sessions")

    def test_session_dir_with_home_expansion(self) -> None:
        """Test session directory with ~ expansion."""
        config = SessionConfig(workspace="/tmp", session_dir="~/.clawbot")
        assert config.resolved_session_dir == Path.home() / ".clawbot"


class TestSessionStorageInit:
    """Tests for SessionStorage initialization."""

    def test_init_with_config(self, tmp_path) -> None:
        """Test initialization with SessionConfig."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        assert storage.session_dir == tmp_path / ".session"
        assert storage.session_dir.exists()

    def test_init_with_workspace(self, tmp_path) -> None:
        """Test initialization with workspace path."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        storage = SessionStorage(workspace=workspace)
        assert storage.session_dir == workspace / ".session"
        assert storage.session_dir.exists()

    def test_init_without_params_raises_error(self) -> None:
        """Test initialization without params raises ValueError."""
        try:
            SessionStorage()
        except ValueError as e:
            assert "Must provide config or workspace parameter" in str(e)


class TestSessionStorageGetPath:
    """Tests for _get_session_path method."""

    def test_get_session_path(self, tmp_path) -> None:
        """Test getting session file path."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        path = storage._get_session_path("test-session-123")
        assert path == tmp_path / ".session" / "test-session-123.jsonl"


class TestSessionStorageAppendAndLoad:
    """Tests for append_message and load_session methods."""

    def test_append_message_creates_file(self, tmp_path) -> None:
        """Test appending message creates session file."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session"

        storage.append_message(
            session_id,
            {"role": "user", "content": "Hello"},
        )

        path = storage._get_session_path(session_id)
        assert path.exists()

    def test_append_message_writes_metadata(self, tmp_path) -> None:
        """Test appending message writes creation timestamp."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)

        storage.append_message(
            "session-1",
            {"role": "user", "content": "Hello"},
        )

        content = storage._get_session_path("session-1").read_text()
        assert META_CREATED_AT_PREFIX in content

    def test_append_multiple_messages(self, tmp_path) -> None:
        """Test appending multiple messages."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "test-session"

        storage.append_message(session_id, {"role": "user", "content": "Hello"})
        storage.append_message(session_id, {"role": "assistant", "content": "Hi"})
        storage.append_message(session_id, {"role": "user", "content": "How are you?"})

        messages = storage.load_session(session_id)
        assert len(messages) == 3

    def test_load_session_empty_file(self, tmp_path) -> None:
        """Test loading from non-existent session returns empty list."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        messages = storage.load_session("non-existent")
        assert messages == []

    def test_load_session_roundtrip(self, tmp_path) -> None:
        """Test messages survive roundtrip append/load."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "roundtrip-test"

        original_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        for msg in original_messages:
            storage.append_message(session_id, msg)

        loaded = storage.load_session(session_id)
        assert loaded == original_messages

    def test_load_session_with_chinese_content(self, tmp_path) -> None:
        """Test loading messages with Chinese characters."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)

        storage.append_message(
            "chinese-test",
            {"role": "user", "content": "你好，世界"},
        )

        messages = storage.load_session("chinese-test")
        assert messages[0]["content"] == "你好，世界"


class TestSessionStorageDelete:
    """Tests for delete_session method."""

    def test_delete_existing_session(self, tmp_path) -> None:
        """Test deleting existing session."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "to-delete"

        storage.append_message(session_id, {"role": "user", "content": "Temp"})
        assert storage._get_session_path(session_id).exists()

        result = storage.delete_session(session_id)
        assert result is True
        assert not storage._get_session_path(session_id).exists()

    def test_delete_nonexistent_session(self, tmp_path) -> None:
        """Test deleting non-existent session returns False."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        result = storage.delete_session("non-existent")
        assert result is False


class TestSessionStorageList:
    """Tests for list_sessions method."""

    def test_list_empty_sessions(self, tmp_path) -> None:
        """Test listing when no sessions exist."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        sessions = storage.list_sessions()
        assert sessions == []

    def test_list_sessions_sorted_by_update_time(self, tmp_path) -> None:
        """Test sessions are sorted by update time descending."""
        import time

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)

        # Create first session
        storage.append_message("session-a", {"role": "user", "content": "A"})
        time.sleep(0.01)

        # Create second session (newer)
        storage.append_message("session-b", {"role": "user", "content": "B"})
        time.sleep(0.01)

        # Update first session
        storage.append_message("session-a", {"role": "user", "content": "A2"})

        sessions = storage.list_sessions()
        # session-a should be first (most recently updated)
        assert sessions[0] == "session-a"
        assert sessions[1] == "session-b"


class TestSessionStorageGetMeta:
    """Tests for get_session_meta method."""

    def test_get_meta_nonexistent_session(self, tmp_path) -> None:
        """Test getting metadata for non-existent session."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        meta = storage.get_session_meta("non-existent")
        assert meta is None

    def test_get_meta_basic_info(self, tmp_path) -> None:
        """Test getting basic session metadata."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "meta-test"

        storage.append_message(session_id, {"role": "user", "content": "Hello"})
        storage.append_message(session_id, {"role": "assistant", "content": "Hi"})

        meta = storage.get_session_meta(session_id)
        assert meta is not None
        assert meta.session_id == session_id
        assert meta.message_count == 2
        assert meta.file_path.endswith(f"{session_id}.jsonl")

    def test_get_meta_timestamps(self, tmp_path) -> None:
        """Test session timestamps are recorded."""
        import time

        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "timestamp-test"

        before_create = time.time()
        storage.append_message(session_id, {"role": "user", "content": "Hello"})
        after_create = time.time()

        meta = storage.get_session_meta(session_id)
        assert meta is not None
        assert before_create <= meta.created_at <= after_create
        assert meta.updated_at >= meta.created_at


class TestSessionStorageCreate:
    """Tests for create_session method."""

    def test_create_session_with_auto_id(self, tmp_path) -> None:
        """Test creating session with auto-generated ID."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = storage.create_session()
        assert session_id is not None
        assert len(session_id) > 0
        assert storage._get_session_path(session_id).exists()

    def test_create_session_with_custom_id(self, tmp_path) -> None:
        """Test creating session with custom ID."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = storage.create_session("my-custom-id")
        assert session_id == "my-custom-id"
        assert storage._get_session_path(session_id).exists()

    def test_create_session_writes_metadata(self, tmp_path) -> None:
        """Test creating session writes creation timestamp."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = storage.create_session()

        content = storage._get_session_path(session_id).read_text()
        assert META_CREATED_AT_PREFIX in content


class TestSessionStorageAppendMessages:
    """Tests for append_messages method."""

    def test_append_multiple_messages_at_once(self, tmp_path) -> None:
        """Test appending multiple messages in one call."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "bulk-append"

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]

        storage.append_messages(session_id, messages)
        loaded = storage.load_session(session_id)
        assert loaded == messages


class TestSessionStorageTruncate:
    """Tests for truncate_session method."""

    def test_truncate_no_op_when_under_limit(self, tmp_path) -> None:
        """Test truncate does nothing when under limit."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "truncate-test"

        storage.append_message(session_id, {"role": "user", "content": "Hello"})
        removed = storage.truncate_session(session_id, max_messages=10)

        assert removed == []
        messages = storage.load_session(session_id)
        assert len(messages) == 1

    def test_truncate_keeps_latest_messages(self, tmp_path) -> None:
        """Test truncate keeps only latest N messages."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "truncate-test"

        for i in range(5):
            storage.append_message(session_id, {"role": "user", "content": f"Msg {i}"})

        removed = storage.truncate_session(session_id, max_messages=3)

        assert len(removed) == 2  # First 2 removed
        assert removed[0]["content"] == "Msg 0"
        assert removed[1]["content"] == "Msg 1"

        remaining = storage.load_session(session_id)
        assert len(remaining) == 3
        assert remaining[0]["content"] == "Msg 2"
        assert remaining[2]["content"] == "Msg 4"

    def test_truncate_preserves_metadata(self, tmp_path) -> None:
        """Test truncate preserves creation timestamp."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "truncate-meta"

        storage.append_message(session_id, {"role": "user", "content": "Hello"})
        storage.append_message(session_id, {"role": "assistant", "content": "Hi"})

        original_meta = storage.get_session_meta(session_id)
        storage.truncate_session(session_id, max_messages=1)
        new_meta = storage.get_session_meta(session_id)

        assert new_meta is not None
        # Timestamps should be approximately equal (within 1 second)
        assert abs(new_meta.created_at - original_meta.created_at) < 1.0


class TestSessionStorageReadCreatedAt:
    """Tests for _read_created_at_from_file method."""

    def test_read_created_at_from_valid_file(self, tmp_path) -> None:
        """Test reading creation timestamp from valid file."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        session_id = "read-meta-test"

        storage.append_message(session_id, {"role": "user", "content": "Hello"})

        created_at = storage._read_created_at_from_file(session_id)
        assert created_at is not None
        assert isinstance(created_at, float)

    def test_read_created_at_from_nonexistent_file(self, tmp_path) -> None:
        """Test reading from non-existent file returns None."""
        config = SessionConfig(workspace=str(tmp_path))
        storage = SessionStorage(config=config)
        created_at = storage._read_created_at_from_file("non-existent")
        assert created_at is None
