"""Load the manually curated metaregistry."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from curies import Converter
from curies.preprocessing import PreprocessingConverter, PreprocessingRules

from .resources import load_goc_map

__all__ = [
    "get_rules",
    "wrap",
]

HERE = Path(__file__).parent.resolve()
RULES_PATH = HERE.joinpath("rules.json")


def wrap(converter: Converter, **kwargs: Any) -> PreprocessingConverter:
    """Wrap a converter with processing rules."""
    return PreprocessingConverter.from_converter(
        converter=converter,
        rules=get_rules(),
        **kwargs,
    )


@lru_cache(1)
def get_rules() -> PreprocessingRules:
    """Get the CURIE/URI string preprocessing rules."""
    rules = PreprocessingRules.model_validate_json(RULES_PATH.read_text())
    rules.rewrites.full.update(load_goc_map())
    return rules


if __name__ == "__main__":
    PreprocessingRules.lint_file(RULES_PATH)
