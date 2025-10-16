"""Data structures for representing OBO Graphs.

.. seealso::

    - the defining repository https://github.com/geneontology/obographs
    - the JSON schema
      https://github.com/geneontology/obographs/blob/master/schema/obographs-schema.json
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, overload

import curies
from curies.vocabulary import SynonymScopeOIO
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .standardized import StandardizedGraph, StandardizedGraphDocument

__all__ = [
    "Definition",
    "DomainRangeAxiom",
    "Edge",
    "EquivalentNodeSet",
    "ExistentialRestrictionExpression",
    "Graph",
    "GraphDocument",
    "LogicalDefinition",
    "Meta",
    "Node",
    "NodeType",
    "Property",
    "PropertyChainAxiom",
    "PropertyType",
    "Synonym",
    "Xref",
    "read",
]

logger = logging.getLogger(__name__)

OBO_URI_PREFIX = "http://purl.obolibrary.org/obo/"
OBO_URI_PREFIX_LEN = len(OBO_URI_PREFIX)

NodeType: TypeAlias = Literal["CLASS", "PROPERTY", "INDIVIDUAL"]

#: When node type is ``PROPERTY``, this is extra information
PropertyType: TypeAlias = Literal["ANNOTATION", "OBJECT", "DATA"]

TimeoutHint = int | float | None


class Property(BaseModel):
    """Represent a property inside a metadata element."""

    pred: str
    val: str | None = Field(
        None,
        description="Stores the value of the property. This can be a string representing a "
        "literal or IRI. This isn't supposed to be nullable, but it happens a lot - might be a "
        "bug in OWLAPI or ROBOT",
    )
    xrefs: list[str] | None = None
    meta: Meta | None = None


class Definition(BaseModel):
    """Represents a definition for a node."""

    val: str | None = Field(default=None)
    xrefs: list[str] | None = Field(default=None)  # Just a list of CURIEs/IRIs


class Xref(BaseModel):
    """Represents a cross-reference."""

    val: str


class Synonym(BaseModel):
    """Represents a synonym inside an object meta."""

    val: str | None = Field(default=None)
    pred: SynonymScopeOIO = Field(default="hasExactSynonym")
    synonymType: str | None = Field(None, examples=["OMO:0003000"])  # noqa:N815
    xrefs: list[str] = Field(
        default_factory=list,
        description="A list of CURIEs/IRIs for provenance for the synonym",
    )
    meta: Meta | None = None


class Meta(BaseModel):
    """Represents the metadata about a node or ontology."""

    definition: Definition | None = None
    subsets: list[str] | None = None
    xrefs: list[Xref] | None = None
    synonyms: list[Synonym] | None = None
    comments: list[str] | None = None
    version: str | None = None
    basicPropertyValues: list[Property] | None = Field(None)  # noqa:N815
    deprecated: bool = False


class Edge(BaseModel):
    """Represents an edge in an OBO Graph."""

    sub: str = Field(..., examples=["http://purl.obolibrary.org/obo/CHEBI_99998"])
    pred: str = Field(..., examples=["is_a"])
    obj: str = Field(..., examples=["http://purl.obolibrary.org/obo/CHEBI_24995"])
    meta: Meta | None = None


class Node(BaseModel):
    """Represents a node in an OBO Graph."""

    id: str = Field(..., description="The IRI for the node")
    lbl: str | None = Field(None, description="The name of the node")
    meta: Meta | None = None
    type: NodeType | None = Field(None, description="Type of node")
    propertyType: PropertyType | None = Field(  # noqa:N815
        None, description="Type of property, if the node type is a property"
    )


class DomainRangeAxiom(BaseModel):
    """Represents a domain/range axiom."""

    predicateId: str  # noqa:N815
    domainClassIds: list[str] | None = None  # noqa:N815
    rangeClassIds: list[str] | None = None  # noqa:N815
    allValuesFromEdges: list[Edge] | None = None  # noqa:N815
    meta: Meta | None = None


class PropertyChainAxiom(BaseModel):
    """Represents a property chain axiom."""

    predicateId: str  # noqa:N815
    chainPredicateIds: list[str]  # noqa:N815
    meta: Meta | None = None


class ExistentialRestrictionExpression(BaseModel):
    """Represents an existential restriction."""

    propertyId: str  # noqa:N815
    fillerId: str  # noqa:N815


class LogicalDefinition(BaseModel):
    """Represents a logical definition chain axiom."""

    definedClassId: str  # noqa:N815
    genusIds: list[str] | None = None  # noqa:N815
    restrictions: list[ExistentialRestrictionExpression] | None = None
    meta: Meta | None = None


class EquivalentNodeSet(BaseModel):
    """Represents a set of equivalent nodes."""

    representativeNodeId: str  # noqa:N815
    nodeIds: list[str]  # noqa:N815
    meta: Meta | None = None


class Graph(BaseModel):
    """A graph corresponds to an ontology."""

    id: str | None = None
    meta: Meta | None = None
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    equivalentNodesSets: list[EquivalentNodeSet] = Field(default_factory=list)  # noqa:N815
    logicalDefinitionAxioms: list[LogicalDefinition] = Field(default_factory=list)  # noqa:N815
    domainRangeAxioms: list[DomainRangeAxiom] = Field(default_factory=list)  # noqa:N815
    propertyChainAxioms: list[PropertyChainAxiom] = Field(default_factory=list)  # noqa:N815

    def _get_property(self, predicate: str) -> str | None:
        if self.meta is not None:
            for prop in self.meta.basicPropertyValues or []:
                if prop.pred == predicate and prop.val:
                    return prop.val
        return None

    @property
    def name(self) -> str | None:
        """Get the title."""
        return self._get_property("http://purl.org/dc/terms/title")

    @property
    def version(self) -> str | None:
        """Get the version."""
        return self._get_property("http://www.w3.org/2002/07/owl#versionInfo")

    def standardize(
        self, converter: curies.Converter, *, strict: bool = False
    ) -> StandardizedGraph:
        """Standardize the graph."""
        from .standardized import StandardizedGraph

        return StandardizedGraph.from_obograph_raw(self, converter, strict=strict)


class GraphDocument(BaseModel):
    """Represents a list of OBO graphs."""

    graphs: list[Graph]
    meta: Meta | None = None

    def standardize(self, converter: curies.Converter) -> StandardizedGraphDocument:
        """Standardize the graph."""
        from .standardized import StandardizedGraphDocument

        return StandardizedGraphDocument.from_obograph_raw(self, converter)


def get_id_to_node(graph: Graph) -> dict[str, Node]:
    """Get a dictionary from node ID to nodes."""
    return {node.id: node for node in graph.nodes or []}


def get_id_to_edges(graph: Graph) -> dict[str, list[tuple[str, str]]]:
    """Get a dictionary from node ID to nodes."""
    dd = defaultdict(set)
    for edge in graph.edges or []:
        dd[edge.sub].add((edge.pred, edge.obj))
    return {node_id: list(predicate_object_pairs) for node_id, predicate_object_pairs in dd.items()}


# docstr-coverage:excused `overload`
@overload
def read(
    source: str | Path, *, timeout: TimeoutHint = ..., squeeze: Literal[False] = ...
) -> GraphDocument: ...


# docstr-coverage:excused `overload`
@overload
def read(
    source: str | Path, *, timeout: TimeoutHint = ..., squeeze: Literal[True] = ...
) -> Graph: ...


def read(
    source: str | Path, *, timeout: TimeoutHint = None, squeeze: bool = True
) -> Graph | GraphDocument:
    """Read an OBO Graph document.

    :param source: A file path or URL to an OBO Graph JSON
    :param timeout: The timeout for getting a URL
    :param squeeze: By default, will unpack the first graph from a graph document that
        only has a single graph and return a :class:`Graph` object. If `true` and
        multiple graphs are received, will raise an error. Set this to `false` to return
        a GraphDocument containing all graphs.

    :returns: A graph or graph document

    :raises ValueError: If squeeze is set to true and multiple graphs are received
    """
    if isinstance(source, str) and (source.startswith("https://") or source.startswith("http://")):
        import requests

        if source.endswith(".gz"):
            raise NotImplementedError
        else:
            res = requests.get(source, timeout=timeout)
            res_json = res.json()
            graph_document = GraphDocument.model_validate(res_json)

    elif isinstance(source, str | Path):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError
        if path.suffix.endswith(".gz"):
            with gzip.open(path, mode="rt") as file:
                graph_document = GraphDocument.model_validate(json.load(file))
        else:
            with path.open() as file:
                graph_document = GraphDocument.model_validate(json.load(file))
    else:
        raise TypeError(f"Unhandled source: {source}")

    if not squeeze:
        return graph_document
    elif len(graph_document.graphs) != 1:
        raise ValueError(
            f"graph document has {len(graph_document.graphs)} graphs, "
            f"so can not squeeze. set squeeze=False"
        )
    else:
        return graph_document.graphs[0]
