"""Command line interface for :mod:`clinicaltrials_downloader`."""

import click

__all__ = [
    "main",
]


@click.command()
@click.option("--force", is_flag=True, help="Force re-download")
def main(force: bool) -> None:
    """Download the ClinicalTrials.gov data."""
    from .api import get_studies, get_studies_slim

    # yes, this is wasteful in time, but more efficient in memory
    get_studies_slim(force=force)
    get_studies(force=force)


if __name__ == "__main__":
    main()
