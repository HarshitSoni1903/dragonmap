"""Command line interface for :mod:`ssslm`."""

import click

__all__ = [
    "main",
]


@click.group()
def main() -> None:
    """CLI for SSSLM."""


@main.command()
@click.argument("path")
def web(path: str) -> None:
    """Run a grounding app."""
    from ssslm.web import run_app

    run_app(path)


if __name__ == "__main__":
    main()
