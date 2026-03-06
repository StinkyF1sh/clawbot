# Repository Guidelines

## Project Structure & Module Organization
- `clawbot/` is the main package. Key modules: `agent/` (loop and context), `provider/` (LLM backends), `tools/` (tool registry and built-ins), `config/` (schema and loader), `cli/` (Typer entrypoints), `queue/`, `channels/`, and `storage/`.
- `tests/` mirrors runtime modules with focused test files such as `test_agent_loop.py`, `test_config.py`, and `test_shell.py`.
- Root-level configs: `pyproject.toml` (packaging, Ruff), `pytest.ini` (test discovery), and `clawbot.yaml.example` (runtime config template).

## Build, Test, and Development Commands
- `pip install -e ".[dev]"`: install package in editable mode with lint/test dependencies.
- `ruff check .`: run linting (style, imports, naming, errors).
- `pytest`: run full test suite (`-v --tb=short` defaults from `pytest.ini`).
- `pytest tests/test_config.py -v`: run a single test module while iterating.
- `clawbot interactive --agent default`: start interactive CLI session.
- `clawbot chat "hello" --agent default`: run one-shot chat from terminal.

## Coding Style & Naming Conventions
- Target Python `3.11+`, 4-space indentation, max line length `100`.
- Follow Ruff rules configured in `pyproject.toml` (`E`, `F`, `I`, `N`, `W`).
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, and explicit typed signatures when practical.
- Prefer small reusable functions; avoid changing public function signatures unless all call sites are updated.

## Testing Guidelines
- Frameworks: `pytest` + `pytest-asyncio` (`asyncio_mode = auto`).
- Naming is enforced: files `test_*.py`, functions `test_*`, classes `Test*`.
- Add or update tests for every behavior change, especially in agent loop, config loading, and tool execution paths.
- Prefer real-path tests; mock only true external boundaries (LLM/network providers).

## Commit & Pull Request Guidelines
- Follow conventional-style messages seen in history, e.g. `feat(agent): improve termination logic`.
- Use `<type>(<scope>): <summary>` when scope is clear (`feat`, `fix`, `refactor`, `test`, `docs`).
- PRs should include: purpose, key changes, test evidence (`pytest`/`ruff` output), and linked issue (if any).
- For CLI-visible changes, include short command examples and output snippets.

## Security & Configuration Tips
- Copy `clawbot.yaml.example` to `clawbot.yaml` (or use `~/.clawbot/config.yaml`) for local config.
- Prefer environment variables for secrets, e.g. `CLAWBOT_PROVIDERS__OPENAI__API_KEY`.
- Never commit API keys, tokens, or credential files.
