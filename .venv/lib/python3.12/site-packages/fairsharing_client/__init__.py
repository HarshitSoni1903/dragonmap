# -*- coding: utf-8 -*-

"""A client to the FAIRsharing API."""

from .api import FairsharingClient, ensure_fairsharing, load_fairsharing

__all__ = [
    "load_fairsharing",
    "ensure_fairsharing",
    "FairsharingClient",
]
