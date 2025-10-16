"""Extra functionality that isn't part of the core obographs package."""

from __future__ import annotations

from typing import overload

from .model import Graph, GraphDocument
from .standardized import StandardizedGraph, StandardizedGraphDocument

__all__ = [
    "guess_primary_graph",
]

CANONICAL = {
    "mamo": "http://identifiers.org/mamo",
    "swo": "http://www.ebi.ac.uk/swo/swo.json",
    "ito": "https://identifiers.org/ito:ontology",
    "apollosv": "http://purl.obolibrary.org/obo/apollo_sv.owl",
    "cheminf": "http://semanticchemistry.github.io/semanticchemistry/ontology/cheminf.owl",
    "dideo": "http://purl.obolibrary.org/obo/dideo/release/2022-06-14/dideo.owl",
    "micro": "http://purl.obolibrary.org/obo/MicrO.owl",
    "ogsf": "http://purl.obolibrary.org/obo/ogsf-merged.owl",
    "mfomd": "http://purl.obolibrary.org/obo/MF.owl",
    "one": "http://purl.obolibrary.org/obo/ONE",
    "ons": "https://raw.githubusercontent.com/enpadasi/Ontology-for-Nutritional-Studies/master/ons.owl",
    "ontie": "https://ontology.iedb.org/ontology/ontie.owl",
}


# docstr-coverage:excused `overload`
@overload
def guess_primary_graph(
    graph_document: GraphDocument,
    prefix: str,
) -> Graph: ...


# docstr-coverage:excused `overload`
@overload
def guess_primary_graph(
    graph_document: StandardizedGraphDocument,
    prefix: str,
) -> StandardizedGraph: ...


def guess_primary_graph(
    graph_document: GraphDocument | StandardizedGraphDocument,
    prefix: str,
) -> Graph | StandardizedGraph:
    """Guess the primary graph from a graph document."""
    if 0 == len(graph_document.graphs):
        raise ValueError("Could not automatically identify the primary graph from empty list")
    elif 1 == len(graph_document.graphs):
        return graph_document.graphs[0]

    id_to_graph = {graph.id: graph for graph in graph_document.graphs if graph.id}

    # Check for standard construction of OBO PURLs
    for suffix in ["owl", "obo", "json"]:
        standard_id = f"http://purl.obolibrary.org/obo/{prefix.lower()}.{suffix}"
        if standard_id in id_to_graph:
            return id_to_graph[standard_id]

    # Check if we've manually curated a mapping
    if prefix in CANONICAL and CANONICAL[prefix] in id_to_graph:
        return id_to_graph[CANONICAL[prefix]]

    raise ValueError(
        f"Could not automatically identify the primary graph for {prefix=} from "
        f"{len(id_to_graph):,} graphs:\n\n{sorted(id_to_graph)}"
    )
