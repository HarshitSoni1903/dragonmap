"""Download ClinicalTrials.gov."""

import gzip
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeAlias, cast

import pystow
import requests
from tqdm import tqdm

__all__ = [
    "get_studies",
    "get_studies_slim",
    "iterate_download_studies",
]

logger = logging.getLogger(__name__)

#: Fields used in the slim dump. See all fields in
#: https://clinicaltrials.gov/data-api/about-api/study-data-structure
SLIM_FIELDS = [
    "NCTId",
    "BriefTitle",
    "Condition",
    "ConditionMeshTerm",  # ConditionMeshTerm is the name of the disease
    "ConditionMeshId",
    "InterventionName",  # InterventionName is the name of the drug/vaccine
    "InterventionType",
    "InterventionMeshTerm",
    "InterventionMeshId",
    "StudyType",
    "DesignAllocation",
    "OverallStatus",
    "Phase",
    "WhyStopped",
    "SecondaryIdType",
    "SecondaryId",
    "StartDate",  # Month [day], year: "November 1, 2023", "May 1984" or NaN
    "StartDateType",  # "Actual" or "Anticipated" (or NaN)
    # these are tagged as relevant by the author, but not necessarily about the trial
    "ReferencePMID",
]

#: The API endpoint for studies
STUDIES_ENDPOINT_URL = "https://clinicaltrials.gov/api/v2/studies"

MODULE = pystow.module("bio", "clinicaltrials")

FULL_PATH = MODULE.join(name="results.json.gz")
FULL_SAMPLE_PATH = MODULE.join(name="results_sample.json")

SLIM_PATH = MODULE.join(name="results_slim.json.gz")
SLIM_SAMPLE_PATH = MODULE.join(name="results_slim_sample.json")

#: This is the maximum page size allowed by the API
MAXIMUM_PAGE_SIZE = 1000

#: The number of studies to save in the sample file
N_SAMPLE_ROWS = 5

#: A type annotation representing a raw study
RawStudy: TypeAlias = dict[str, Any]


def get_studies(*, force: bool = False) -> list[RawStudy]:
    """Get the full ClinicalTrials.gov data dump.

    :param force: Should the data be re-downloaded (and the cache invalidated?)
    :return:
        A list of raw dictionaries representing studies in ClinicalTrials.gov,
        as they are returned by the `ClinicalTrials.gov API <https://clinicaltrials.gov/data-api/api>`_

    .. warning::

        The results are cached to :data:`PATH`. You should
        periodically set force=True to re-download the database,
        since new records are added on a daily basis.

        Order is not guaranteed, so, unfortunately, this can't
        be done incrementally.

    If you want more control over how downloading works, see
    :func:`iterate_download_studies`.
    """
    return _help_get_studies(
        path=FULL_PATH,
        sample_path=FULL_SAMPLE_PATH,
        fields=None,  # none means use everything
        force=force,
    )


def get_studies_slim(*, force: bool = False) -> list[RawStudy]:
    """Get a slimmed-down set of studies based on :data:`SLIM_FIELDS`."""
    return _help_get_studies(
        path=SLIM_PATH,
        sample_path=SLIM_SAMPLE_PATH,
        fields=SLIM_FIELDS,
        force=force,
    )


def _help_get_studies(
    *, path: Path, sample_path: Path, fields: list[str] | None, force: bool = False
) -> list[RawStudy]:
    if path.exists() and not force:
        with gzip.open(path, "rt") as file:
            t = time.time()
            logger.info("loading cached ClinicalTrials.gov")
            rv = cast(list[RawStudy], json.load(file))
            logger.info("loaded cached ClinicalTrials.gov in %f seconds", time.time() - t)
            return rv

    logger.info("download results to %s", path)
    studies = list(iterate_download_studies(page_size=MAXIMUM_PAGE_SIZE, fields=fields))

    with sample_path.open("w") as file:
        json.dump(studies[:N_SAMPLE_ROWS], file, indent=2)

    with gzip.open(path, "wt") as file:
        json.dump(studies, file)

    return studies


def iterate_download_studies(
    *,
    page_size: int | None = None,
    fields: list[str] | None = None,
) -> Iterable[RawStudy]:
    """Download studies iteratively by paging through the ClinicalTrials.gov API.

    :param page_size: The page size when hitting the API
    :param fields:
        the fields to download. See a full list at
        https://clinicaltrials.gov/data-api/about-api/study-data-structure.
        For example, a small field list of ``["NCTId", "BriefTitle"]`` is useful
        for quickly checking the database.
    :yields: Individual dictionaries corresponding to studies

    .. seealso:: `ClinicalTrials.gov API documentation <https://clinicaltrials.gov/data-api/api>`_
    """
    if page_size is None:
        page_size = MAXIMUM_PAGE_SIZE

    if page_size > MAXIMUM_PAGE_SIZE:
        page_size = MAXIMUM_PAGE_SIZE

    parameters: dict[str, str | int] = {
        "pageSize": page_size,
    }
    if fields is not None:
        parameters["fields"] = ",".join(fields)

    # on the first get, we need to return the count so we can make a progress bar
    # note that countTotal needs to be encoded with a string, and not a proper boolean
    res = requests.get(
        STUDIES_ENDPOINT_URL, params={**parameters, "countTotal": "true"}, timeout=5
    ).json()

    total = res["totalCount"]
    yield from res["studies"]

    with tqdm(
        desc="Downloading ClinicalTrials.gov",
        total=total,
        unit="trial",
        unit_scale=True,
    ) as pbar:
        # update the progress bar based on the initial result
        # size
        pbar.update(page_size)

        # While there's a nextPageToken field available in the result,
        # get a new page. We don't need the countTotal anymore. Make
        # sure to overwrite the original variable name each time.
        while next_token := res.get("nextPageToken"):
            res = requests.get(
                STUDIES_ENDPOINT_URL,
                params={**parameters, "pageToken": next_token},
                timeout=5,
            ).json()
            studies = res["studies"]
            yield from studies
            pbar.update(len(studies))
