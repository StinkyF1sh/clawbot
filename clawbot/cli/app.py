"""CLI application entry point for Clawbot."""

import typer

from clawbot import __version__
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
    config_file=find_config_file()
    if config_file:
        config = load_config(config_file)
        ctx.obj["config"] = config
    else:
        typer.echo("No configuration file found. Please create a config.yaml.")
    return

@app.command("start")
def main(ctx: typer.Context):
    """Start Clawbot interactive session."""
    config = ctx.obj.get("config", {})
    typer.echo("Clawbot - Coming soon!")
    typer.echo("Configuration loaded:")
    typer.echo(config)

@app.command("version")
def version():
    """Show Clawbot version."""
    typer.echo(f"Clawbot version {__version__}")

if __name__ == "__main__":
    app()
