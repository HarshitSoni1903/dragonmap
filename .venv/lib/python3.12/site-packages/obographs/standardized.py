"""Standardize an OBO graph."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

import curies.preprocessing
from curies import Converter, Reference, Triple, vocabulary
from curies.vocabulary import SynonymScopeOIO, has_title, owl_version_info
from pydantic import BaseModel, Field
from typing_extensions import Self

from obographs.model import (
    Definition,
    DomainRangeAxiom,
    Edge,
    EquivalentNodeSet,
    ExistentialRestrictionExpression,
    Graph,
    GraphDocument,
    LogicalDefinition,
    Meta,
    Node,
    NodeType,
    Property,
    PropertyChainAxiom,
    PropertyType,
    Synonym,
    Xref,
)

__all__ = [
    "StandardizedBaseModel",
    "StandardizedDefinition",
    "StandardizedDomainRangeAxiom",
    "StandardizedEdge",
    "StandardizedEquivalentNodeSet",
    "StandardizedExistentialRestriction",
    "StandardizedGraph",
    "StandardizedGraphDocument",
    "StandardizedLogicalDefinition",
    "StandardizedMeta",
    "StandardizedNode",
    "StandardizedProperty",
    "StandardizedPropertyChainAxiom",
    "StandardizedSynonym",
    "StandardizedXref",
]

logger = logging.getLogger(__name__)


def _expand_list(references: list[Reference] | None, converter: Converter) -> list[str] | None:
    if references is None or not references:
        return None
    return [converter.expand_reference(r, strict=True) for r in references]


X = TypeVar("X")


class StandardizedBaseModel(BaseModel, ABC, Generic[X]):
    """A standardized property."""

    @classmethod
    @abstractmethod
    def from_obograph_raw(
        cls, obj: X, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Instantiate by standardizing a raw OBO Graph object."""
        raise NotImplementedError

    @abstractmethod
    def to_raw(self, converter: Converter) -> X:
        """Create a raw object."""
        raise NotImplementedError


class StandardizedProperty(StandardizedBaseModel[Property]):
    """A standardized property."""

    predicate: Reference
    value: Reference | str = Field(
        ..., description="Parsed into a Reference if a CURIE or IRI, or a string if it's a literal"
    )
    xrefs: list[Reference] | None = None
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, prop: Property, converter: Converter, *, strict: bool = False
    ) -> Self:
        """Instantiate by standardizing a raw OBO Graph object."""
        if not prop.val or not prop.pred:
            raise ValueError
        value: Reference | str | None

        if (
            prop.val.startswith("http://")
            or prop.val.startswith("https")
            or converter.is_curie(prop.val)
            or prop.val in BUILTINS
        ):
            value = _curie_or_uri_to_ref(prop.val, converter, strict=False) or prop.val
        else:
            value = prop.val
        return cls(
            predicate=_curie_or_uri_to_ref(prop.pred, converter, strict=strict),
            value=value,
        )

    def to_raw(self, converter: Converter) -> Property:
        """Create a raw object."""
        return Property(
            pred=converter.expand_reference(self.predicate),
            val=converter.expand_reference(self.value)
            if isinstance(self.value, Reference)
            else self.value,
            xrefs=_expand_list(self.xrefs, converter),
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


class StandardizedDefinition(StandardizedBaseModel[Definition]):
    """A standardized definition."""

    value: str | None = Field(default=None)
    xrefs: list[Reference] | None = Field(default=None)

    @classmethod
    def from_obograph_raw(
        cls, definition: Definition | None, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Parse a raw object."""
        if definition is None:
            return None
        return cls(
            value=definition.val,
            xrefs=_parse_list(definition.xrefs, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> Definition:
        """Create a raw object."""
        return Definition(
            val=self.value,
            xrefs=_expand_list(self.xrefs, converter),
        )


class StandardizedXref(StandardizedBaseModel[Xref]):
    """A standardized database cross-reference."""

    reference: Reference

    @classmethod
    def from_obograph_raw(cls, xref: Xref, converter: Converter, *, strict: bool = False) -> Self:
        """Instantiate by standardizing a raw OBO Graph object."""
        reference = _curie_or_uri_to_ref(xref.val, converter, strict=strict)
        if reference is None:
            raise ValueError(f"could not parse xref: {xref.val}")
        return cls(reference=reference)

    def to_raw(self, converter: Converter) -> Xref:
        """Create a raw object."""
        return Xref(val=self.reference.curie)


class StandardizedSynonym(StandardizedBaseModel[Synonym]):
    """A standardized synonym."""

    text: str
    predicate: Reference
    type: Reference | None = None
    xrefs: list[Reference] | None = None

    @classmethod
    def from_obograph_raw(
        cls, synonym: Synonym, converter: Converter, *, strict: bool = False
    ) -> Self:
        """Instantiate by standardizing a raw OBO Graph object."""
        return cls(
            text=synonym.val,
            predicate=Reference(prefix="oboInOwl", identifier=synonym.pred),
            type=synonym.synonymType
            and _curie_or_uri_to_ref(synonym.synonymType, converter, strict=strict),
            xrefs=_parse_list(synonym.xrefs, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> Synonym:
        """Create a raw object."""
        if self.predicate.prefix.lower() != "oboinowl":
            raise ValueError
        return Synonym(
            val=self.text,
            pred=cast(SynonymScopeOIO, self.predicate.identifier),
            synonymType=converter.expand_reference(self.type) if self.type is not None else None,
            xrefs=_expand_list(self.xrefs, converter) or [],
        )


class StandardizedMeta(StandardizedBaseModel[Meta]):
    """A standardized meta object."""

    definition: StandardizedDefinition | None = None
    subsets: list[Reference] | None = None
    xrefs: list[StandardizedXref] | None = None
    synonyms: list[StandardizedSynonym] | None = None
    comments: list[str] | None = None
    deprecated: bool = False
    version_iri: str | None = None
    properties: list[StandardizedProperty] | None = None

    @classmethod
    def from_obograph_raw(  # noqa:C901
        cls, meta: Meta | None, converter: Converter, flag: str = "", strict: bool = False
    ) -> Self | None:
        """Instantiate by standardizing a raw OBO Graph object."""
        if meta is None:
            return None

        xrefs = []
        for raw_xref in meta.xrefs or []:
            if raw_xref.val:
                try:
                    st_xref = StandardizedXref.from_obograph_raw(raw_xref, converter, strict=strict)
                except ValueError:
                    if strict:
                        raise
                    logger.debug("[%s] failed to standardize xref: %s", flag, raw_xref)
                else:
                    xrefs.append(st_xref)

        synonyms = []
        for raw_synonym in meta.synonyms or []:
            if raw_synonym.val:
                try:
                    s = StandardizedSynonym.from_obograph_raw(raw_synonym, converter, strict=strict)
                except ValueError:
                    if strict:
                        raise
                    logger.debug("[%s] failed to standardize synonym: %s", flag, raw_synonym)
                else:
                    synonyms.append(s)

        props = []
        for raw_prop in meta.basicPropertyValues or []:
            if raw_prop.val and raw_prop.pred:
                try:
                    prop = StandardizedProperty.from_obograph_raw(
                        raw_prop, converter, strict=strict
                    )
                except ValueError:
                    if strict:
                        raise
                    logger.debug("[%s] failed to standardize property: %s", flag, raw_prop)
                else:
                    props.append(prop)

        return cls(
            definition=StandardizedDefinition.from_obograph_raw(
                meta.definition, converter, strict=strict
            )
            if meta.definition is not None
            else None,
            subsets=[
                _curie_or_uri_to_ref(subset, converter, strict=strict) for subset in meta.subsets
            ]
            if meta.subsets
            else None,
            xrefs=xrefs or None,
            synonyms=synonyms or None,
            comments=meta.comments,
            version_iri=meta.version,
            deprecated=meta.deprecated,
            properties=props or None,
        )

    def to_raw(self, converter: Converter) -> Meta:
        """Create a raw object."""
        return Meta(
            definition=self.definition.to_raw(converter)
            if self.definition and self.definition.value
            else None,
            subsets=_expand_list(self.subsets, converter),
            xrefs=[xref.to_raw(converter) for xref in self.xrefs] if self.xrefs else None,
            synonyms=[s.to_raw(converter) for s in self.synonyms] if self.synonyms else None,
            comments=self.comments,
            version=self.version_iri,
            deprecated=self.deprecated,
            basicPropertyValues=[p.to_raw(converter) for p in self.properties]
            if self.properties
            else None,
        )


class StandardizedNode(StandardizedBaseModel[Node]):
    """A standardized node."""

    reference: Reference
    label: str | None = Field(None)
    meta: StandardizedMeta | None = None
    type: NodeType | None = Field(None, description="Type of node")
    property_type: PropertyType | None = Field(
        None, description="Type of property, if the node type is a property"
    )

    @classmethod
    def from_obograph_raw(
        cls, node: Node, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Instantiate by standardizing a raw OBO Graph object."""
        reference = _curie_or_uri_to_ref(node.id, converter, strict=strict)
        if reference is None:
            if strict:
                raise ValueError(f"failed to parse node's ID: {node.id}")
            logger.warning("failed to parse node's ID %s", node.id)
            return None

        return cls(
            reference=reference,
            label=node.lbl,
            meta=StandardizedMeta.from_obograph_raw(
                node.meta, converter, flag=reference.curie, strict=strict
            ),
            type=node.type,
            property_type=node.propertyType,
        )

    def to_raw(self, converter: Converter) -> Node:
        """Create a raw object."""
        return Node(
            id=converter.expand_reference(self.reference),
            lbl=self.label,
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
            type=self.type,
            propertyType=self.property_type,
        )


class StandardizedEdge(Triple, StandardizedBaseModel[Edge]):
    """A standardized edge."""

    subject: Reference
    predicate: Reference
    object: Reference
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, edge: Edge, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Instantiate by standardizing a raw OBO Graph object."""
        subject = _curie_or_uri_to_ref(edge.sub, converter, strict=strict)
        if not subject:
            if strict:
                raise ValueError
            logger.warning("failed to parse edge's subject %s", edge.sub)
            return None
        predicate = _curie_or_uri_to_ref(edge.pred, converter, strict=strict)
        if not predicate:
            if strict:
                raise ValueError
            logger.warning("failed to parse edge's predicate %s", edge.pred)
            return None
        obj = _curie_or_uri_to_ref(edge.obj, converter, strict=strict)
        if not obj:
            if strict:
                raise ValueError
            logger.warning("failed to parse edge's object %s", edge.obj)
            return None
        return cls(
            subject=subject,
            predicate=predicate,
            object=obj,
            meta=StandardizedMeta.from_obograph_raw(
                edge.meta,
                converter,
                flag=f"{subject.curie} {predicate.curie} {obj.curie}",
                strict=strict,
            ),
        )

    def to_raw(self, converter: Converter) -> Edge:
        """Create a raw object."""
        if self.predicate in REVERSE_BUILTINS:
            predicate = REVERSE_BUILTINS[self.predicate]
        else:
            predicate = converter.expand_reference(self.predicate, strict=True)

        return Edge(
            sub=converter.expand_reference(self.subject),
            pred=predicate,
            obj=converter.expand_reference(self.object),
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


class StandardizedDomainRangeAxiom(StandardizedBaseModel[DomainRangeAxiom]):
    """Represents a domain/range axiom."""

    predicate: Reference
    domains: list[Reference] = Field(default_factory=list)
    ranges: list[Reference] = Field(default_factory=list)
    all_values_from_edges: list[StandardizedEdge] = Field(default_factory=list)
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, obj: DomainRangeAxiom, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Parse a raw object."""
        return cls(
            predicate=_curie_or_uri_to_ref(obj.predicateId, converter, strict=strict),
            domains=_parse_list(obj.domainClassIds, converter, strict=strict) or [],
            ranges=_parse_list(obj.rangeClassIds, converter, strict=strict) or [],
            all_values_from_edges=[
                StandardizedEdge.from_obograph_raw(edge, converter, strict=strict)
                for edge in obj.allValuesFromEdges or []
            ],
            meta=StandardizedMeta.from_obograph_raw(obj.meta, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> DomainRangeAxiom:
        """Create a raw object."""
        return DomainRangeAxiom(
            predicateId=converter.expand_reference(self.predicate),
            domainClassIds=_expand_list(self.domains, converter),
            rangeClassIds=_expand_list(self.ranges, converter),
            allValuesFromEdges=[edge.to_raw(converter) for edge in self.all_values_from_edges]
            if self.all_values_from_edges
            else None,
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


class StandardizedPropertyChainAxiom(StandardizedBaseModel[PropertyChainAxiom]):
    """Represents a property chain axiom."""

    predicate: Reference
    chain: list[Reference] = Field(default_factory=list)
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, obj: PropertyChainAxiom, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Parse a raw object."""
        return cls(
            predicate=_curie_or_uri_to_ref(obj.predicateId, converter, strict=strict),
            chain=_parse_list(obj.chainPredicateIds, converter, strict=strict),
            meta=StandardizedMeta.from_obograph_raw(obj.meta, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> PropertyChainAxiom:
        """Create a raw object."""
        return PropertyChainAxiom(
            predicateId=converter.expand_reference(self.predicate),
            chainPredicateIds=_expand_list(self.chain, converter),
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


class StandardizedEquivalentNodeSet(StandardizedBaseModel[EquivalentNodeSet]):
    """Represents an equivalence set."""

    node: Reference
    equivalents: list[Reference] = Field(default_factory=list)
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, obj: EquivalentNodeSet, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Parse a raw object."""
        return cls(
            node=_curie_or_uri_to_ref(obj.representativeNodeId, converter, strict=strict),
            equivalents=_parse_list(obj.nodeIds, converter, strict=strict),
            meta=StandardizedMeta.from_obograph_raw(obj.meta, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> EquivalentNodeSet:
        """Create a raw object."""
        return EquivalentNodeSet(
            representativeNodeId=converter.expand_reference(self.node),
            nodeIds=_expand_list(self.equivalents, converter),
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


class StandardizedExistentialRestriction(StandardizedBaseModel[ExistentialRestrictionExpression]):
    """Represents an existential restriction expression."""

    predicate: Reference
    target: Reference

    @classmethod
    def from_obograph_raw(
        cls, obj: ExistentialRestrictionExpression, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Parse a raw object."""
        return cls(
            predicate=_curie_or_uri_to_ref(obj.propertyId, converter, strict=strict),
            target=_curie_or_uri_to_ref(obj.fillerId, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> ExistentialRestrictionExpression:
        """Create a raw object."""
        return ExistentialRestrictionExpression(
            propertyId=converter.expand_reference(self.predicate),
            fillerId=converter.expand_reference(self.target),
        )


class StandardizedLogicalDefinition(StandardizedBaseModel[LogicalDefinition]):
    """Represents a logical definition axiom."""

    node: Reference
    geni: list[Reference] = Field(default_factory=list)
    restrictions: list[StandardizedExistentialRestriction] = Field(default_factory=list)
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, obj: LogicalDefinition, converter: Converter, *, strict: bool = False
    ) -> Self | None:
        """Parse a raw object."""
        return cls(
            node=_curie_or_uri_to_ref(obj.definedClassId, converter, strict=strict),
            geni=_parse_list(obj.genusIds, converter, strict=strict),
            restrictions=[
                StandardizedExistentialRestriction.from_obograph_raw(r, converter, strict=strict)
                for r in obj.restrictions or []
            ],
            meta=StandardizedMeta.from_obograph_raw(obj.meta, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> LogicalDefinition:
        """Create a raw object."""
        return LogicalDefinition(
            definedClassId=converter.expand_reference(self.node),
            genusIds=_expand_list(self.geni, converter),
            restrictions=[r.to_raw(converter) for r in self.restrictions],
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


class StandardizedGraph(StandardizedBaseModel[Graph]):
    """A standardized graph."""

    id: str | None = None
    meta: StandardizedMeta | None = None
    nodes: list[StandardizedNode] = Field(default_factory=list)
    edges: list[StandardizedEdge] = Field(default_factory=list)

    equivalent_node_sets: list[StandardizedEquivalentNodeSet] = Field(default_factory=list)
    logical_definition_axioms: list[StandardizedLogicalDefinition] = Field(default_factory=list)
    domain_range_axioms: list[StandardizedDomainRangeAxiom] = Field(default_factory=list)
    property_chain_axioms: list[StandardizedPropertyChainAxiom] = Field(default_factory=list)

    @classmethod
    def from_obograph_raw(cls, graph: Graph, converter: Converter, *, strict: bool = False) -> Self:
        """Instantiate by standardizing a raw OBO Graph object."""
        return cls(
            id=graph.id,
            meta=StandardizedMeta.from_obograph_raw(
                graph.meta, converter, flag=graph.id or "", strict=strict
            ),
            nodes=[
                s_node
                for node in graph.nodes
                if (s_node := StandardizedNode.from_obograph_raw(node, converter, strict=strict))
            ],
            edges=[
                s_edge
                for edge in graph.edges
                if (s_edge := StandardizedEdge.from_obograph_raw(edge, converter, strict=strict))
            ],
            equivalent_node_sets=[
                StandardizedEquivalentNodeSet.from_obograph_raw(e, converter, strict=strict)
                for e in graph.equivalentNodesSets or []
            ],
            logical_definition_axioms=[
                StandardizedLogicalDefinition.from_obograph_raw(e, converter, strict=strict)
                for e in graph.logicalDefinitionAxioms or []
            ],
            property_chain_axioms=[
                StandardizedPropertyChainAxiom.from_obograph_raw(e, converter, strict=strict)
                for e in graph.propertyChainAxioms or []
            ],
            domain_range_axioms=[
                StandardizedDomainRangeAxiom.from_obograph_raw(e, converter, strict=strict)
                for e in graph.domainRangeAxioms or []
            ],
        )

    def to_raw(self, converter: Converter) -> Graph:
        """Create a raw object."""
        return Graph(
            id=self.id,
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
            nodes=[node.to_raw(converter) for node in self.nodes],
            edges=[edge.to_raw(converter) for edge in self.edges],
            logicalDefinitionAxioms=[
                axiom.to_raw(converter) for axiom in self.logical_definition_axioms
            ],
            propertyChainAxioms=[axiom.to_raw(converter) for axiom in self.property_chain_axioms],
            domainRangeAxioms=[axiom.to_raw(converter) for axiom in self.domain_range_axioms],
            equivalentNodesSets=[axiom.to_raw(converter) for axiom in self.equivalent_node_sets],
        )

    def _get_property(self, predicate: Reference) -> str | Reference | None:
        if self.meta is None:
            return None

        for p in self.meta.properties or []:
            if p.predicate == predicate:
                return p.value

        return None

    @property
    def name(self) -> str | None:
        """Look up the name of the graph."""
        r = self._get_property(has_title)
        if isinstance(r, Reference):
            raise TypeError
        return r

    @property
    def version(self) -> str | None:
        """Get the version."""
        r = self._get_property(owl_version_info)
        if isinstance(r, Reference):
            raise TypeError
        return r


class StandardizedGraphDocument(StandardizedBaseModel[GraphDocument]):
    """A standardized graph document."""

    graphs: list[StandardizedGraph]
    meta: StandardizedMeta | None = None

    @classmethod
    def from_obograph_raw(
        cls, graph_document: GraphDocument, converter: Converter, *, strict: bool = False
    ) -> Self:
        """Instantiate by standardizing a raw OBO Graph Document object."""
        return cls(
            graphs=[
                StandardizedGraph.from_obograph_raw(graph, converter, strict=strict)
                for graph in graph_document.graphs
            ],
            meta=StandardizedMeta.from_obograph_raw(graph_document.meta, converter, strict=strict),
        )

    def to_raw(self, converter: Converter) -> GraphDocument:
        """Create a raw object."""
        return GraphDocument(
            graphs=[graph.to_raw(converter) for graph in self.graphs],
            meta=self.meta.to_raw(converter) if self.meta is not None else None,
        )


def _parse_list(
    curie_or_uris: list[str] | None, converter: Converter, *, strict: bool
) -> list[Reference] | None:
    if not curie_or_uris:
        return None
    return [
        reference
        for curie_or_uri in curie_or_uris
        if (reference := _curie_or_uri_to_ref(curie_or_uri, converter, strict=strict))
    ]


#: defined in https://github.com/geneontology/obographs/blob/6676b10a5cce04707d75b9dd46fa08de70322b0b/obographs-owlapi/src/main/java/org/geneontology/obographs/owlapi/FromOwl.java#L36-L39
#: this list is complete.
BUILTINS: dict[str, Reference] = {
    "is_a": vocabulary.is_a,
    "subPropertyOf": vocabulary.subproperty_of,
    "type": vocabulary.rdf_type,
    "inverseOf": Reference(prefix="owl", identifier="inverseOf"),
}

"""maybe add these later?
    # predicates, see https://github.com/geneontology/obographs/blob/6676b10a5cce04707d75b9dd46fa08de70322b0b/obographs-core/src/test/java/org/geneontology/obographs/core/model/axiom/PropertyChainAxiomTest.java#L12-L14
    # "part_of": vocabulary.part_of,
    # "has_part": vocabulary.has_part,
    # "overlaps": Reference(prefix="RO", identifier="0002131"),
"""

REVERSE_BUILTINS: dict[Reference, str] = {v: k for k, v in BUILTINS.items()}


def _curie_or_uri_to_ref(s: str, converter: Converter, *, strict: bool) -> Reference | None:
    if s in BUILTINS:
        return BUILTINS[s]
    try:
        reference_tuple = converter.parse(s, strict=False)
    except curies.preprocessing.BlocklistError:
        return None
    if reference_tuple is not None:
        return reference_tuple.to_pydantic()
    if strict:
        raise ValueError(f"could not parse {s}")
    return None
