"""CLI application entry point for Clawbot."""

import asyncio
import sys

import typer

from clawbot import __version__
from clawbot.cli.init import create_cli_handler
from clawbot.config.loader import find_config_file, load_config

app = typer.Typer(
    name="clawbot",
    help="Clawbot: A Python personal AI assistant.",
    no_args_is_help=True,
)


@app.callback()
def main_callback(ctx: typer.Context):
    """Clawbot CLI application."""
    ctx.ensure_object(dict)
    config_file = find_config_file()
    if config_file:
        config = load_config(config_file)
        ctx.obj["config"] = config
    else:
        typer.echo("No configuration file found. Please create a config.yaml.")
        raise typer.Exit(1)


@app.command("interactive")
def interactive(
    ctx: typer.Context,
    agent: str = typer.Option("default", "--agent", "-a", help="Agent profile to use"),
    session: str = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID (auto-generated if not provided)",
    ),
):
    """Start interactive conversation session."""
    config = ctx.obj.get("config")
    if not config:
        typer.echo("Configuration not loaded.")
        raise typer.Exit(1)

    handler = create_cli_handler(config)

    typer.echo(f"Clawbot v{__version__} - Interactive Mode")
    typer.echo(f"Agent: {agent}")
    typer.echo(f"Session: {session or 'auto-generated'}")
    typer.echo("Type 'exit' or 'quit' to end the session.")
    typer.echo("-" * 50)

    asyncio.run(_run_interactive(handler, agent, session))


async def _run_interactive(handler, agent_name: str, session_id: str | None):
    """Run interactive conversation loop."""
    from clawbot.util.utils import generate_session_id

    if not session_id:
        session_id = generate_session_id()

    try:
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    typer.echo("Goodbye!")
                    break

                typer.echo("Clawbot: ", nl=False)
                response = await handler.run_turn(
                    session_id=session_id,
                    agent_name=agent_name,
                    user_input=user_input,
                )
                typer.echo(response)

            except KeyboardInterrupt:
                typer.echo("\nInterrupted. Type 'exit' to quit.")
            except EOFError:
                typer.echo("\nGoodbye!")
                break
            except Exception as e:
                typer.echo(f"Error: {e}", err=True)

    except Exception as e:
        typer.echo(f"Fatal error: {e}", err=True)
        sys.exit(1)


@app.command("chat")
def chat(
    ctx: typer.Context,
    message: str = typer.Argument(..., help="Message to send"),
    agent: str = typer.Option("default", "--agent", "-a", help="Agent profile to use"),
    session: str = typer.Option(
        None,
        "--session",
        "-s",
        help="Session ID (auto-generated if not provided)",
    ),
):
    """Send a single message and get response."""
    config = ctx.obj.get("config")
    if not config:
        typer.echo("Configuration not loaded.")
        raise typer.Exit(1)

    handler = create_cli_handler(config)

    asyncio.run(_run_chat(handler, message, agent, session))


async def _run_chat(handler, message: str, agent_name: str, session_id: str | None):
    """Run single message chat."""
    from clawbot.util.utils import generate_session_id

    if not session_id:
        session_id = generate_session_id()

    try:
        response = await handler.run_turn(
            session_id=session_id,
            agent_name=agent_name,
            user_input=message,
        )
        typer.echo(response)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)


@app.command("version")
def version():
    """Show Clawbot version."""
    typer.echo(f"Clawbot version {__version__}")


if __name__ == "__main__":
    app()
