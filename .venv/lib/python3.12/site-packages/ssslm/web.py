"""A FastAPI-based web wrapper around the grounder."""

from pathlib import Path
from typing import Annotated

import fastapi
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel

import ssslm
from ssslm import Annotation, Grounder, Match

__all__ = [
    "get_app",
    "run_app",
]

api_router = fastapi.APIRouter()


def run_app(grounder: ssslm.Grounder | str | Path) -> None:
    """Construct a FastAPI app from a grounder and run with :mod:`uvicorn`."""
    if isinstance(grounder, str | Path):
        grounder = ssslm.make_grounder(ssslm.read_literal_mappings(grounder))

    import uvicorn

    uvicorn.run(get_app(grounder))


def get_app(grounder: ssslm.Grounder) -> FastAPI:
    """Construct a FastAPI app from a grounder."""
    app = FastAPI(title="SSSLM Grounder")
    app.state = grounder  # type:ignore
    app.include_router(api_router, prefix="/api")
    return app


def _get_grounder(request: Request) -> Grounder:
    return request.app.state  # type:ignore


@api_router.get("/ground/{text}", response_model=list[Match])
def ground(
    grounder: Annotated[Grounder, Depends(_get_grounder)],
    text: str = fastapi.Path(..., description="Text to be grounded."),
) -> list[Match]:
    """Ground text."""
    return grounder.get_matches(text)


class AnnotationRequest(BaseModel):
    """An annotation request."""

    text: str


@api_router.post("/annotate/", response_model=list[Annotation])
def annotate(
    grounder: Annotated[Grounder, Depends(_get_grounder)],
    annotation_request: AnnotationRequest,
) -> list[Annotation]:
    """Annotate text."""
    return grounder.annotate(annotation_request.text)
