# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

clawbot is a Python personal AI assistant.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run linting
ruff check .

# Run tests
pytest

# Run single test
pytest tests/test_specific.py -v

# Run CLI
clawbot
```

## Architecture

**Core Modules:**
- `agent/` - Agent loop (`GlobalAgentLoop`, `SingleSessionAgentLoop`), conversation context, and runtime configuration
- `channels/` - External channel integration (messaging platforms), message polling and response routing
- `cli/` - Command-line interface with interactive and chat modes
- `config/` - Pydantic-based configuration schema and YAML loading
- `provider/` - LLM provider implementations (`BaseProvider`, `LiteLLMProvider`, `OpenAICompatibleProvider`)
- `queue/` - Dual-queue task management (`InputQueue`, `OutputQueue`, `TaskQueueManager`)
- `storage/` - Session persistence using JSONL format
- `tools/` - Tool abstraction and execution framework (`Tool`, `ToolRegistry`, built-in tools)

**Entry Point:** `clawbot` CLI command → `clawbot.cli.app:app`

## Code Style

**Comments:**
- Only comment when necessary - self-explanatory code needs no comments
- Use single-line comments for clarity
- Docstrings: single-line for simple functions, multi-line only when Args/Returns need explanation

## Testing

**Guidelines:**
- **Avoid mocks as much as possible** - Test actual implementation, do not duplicate logic into tests
- Use real implementations for dependencies when feasible
- Mock only external services (APIs, databases) that cannot be run in tests
- Prefer integration tests over unit tests when the complexity is manageable

## Notes

- Package built with hatchling, outputs to `clawbot/` namespace
- Configuration supports both YAML file and environment variables (prefix: `CLAWBOT_`)
- Dual-loop architecture: `GlobalAgentLoop` (dispatcher) + `SingleSessionAgentLoop` (per-session)
- Dual-queue design: `InputQueue` (channel → agent) + `OutputQueue` (agent → channel)

## Tools

**Built-in Tools:**
- `ReadFileTool` - Read file contents
- `WriteFileTool` - Write content to file (creates parent directories)
- `EditFileTool` - Edit file by replacing text (with diff-based error messages)
- `ListDirTool` - List directory contents
- `ExecTool` - Execute shell commands (with safety guards)

**Safety Features:**
- Path sandboxing via `allowed_dir` restriction
- Command deny patterns block dangerous operations (rm -rf, format, etc.)
- Optional workspace restriction for shell commands
- Command timeout (default 60s)