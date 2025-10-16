"""A CLI for the FAIRsharing client."""

import click


@click.command()
@click.option("--force", is_flag=True)
def main(force: bool):
    """Download the FAIRsharing data."""
    from fairsharing_client import ensure_fairsharing

    ensure_fairsharing(force_download=force)


if __name__ == "__main__":
    main()
