# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

clawbot is a Python personal AI assistant framework built on litellm for LLM provider abstraction.

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
myclaw
```

## Architecture

**Core Modules:**
- `agent/` - Agent loop and conversation context
- `cli/` - Command-line interface entry point
- `config/` - Configuration loading
- `provider/` - LLM provider implementations (base, litellm, openai_compatible)
- `queue/` - Task queue management
- `storage/` - Session persistence
- `tools/` - Tool/base functionality

**Entry Point:** `myclaw` CLI command → `clawbot.cli.app:app`

## Notes

- This is a new project skeleton; most module files are empty placeholders
- Package built with hatchling, outputs to `clawbot/` namespace
