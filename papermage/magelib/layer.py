"""

Layers are collections of Entities. Supports indexing and slicing.

@kylel

"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .box import Box
from .entity import Entity
from .indexer import EntityBoxIndexer, EntitySpanIndexer
from .metadata import Metadata
from .span import Span

if TYPE_CHECKING:
    from .document import Document


class Layer:
    """A fancy list of Entities. Manages <list> things like indexing and slicing,
    but also gives access to things like reading order and other metadata."""

    __slots__ = ["entities", "metadata", "span_indexer", "box_indexer", "_name", "_doc"]

    def __init__(self, entities: List[Entity], metadata: Optional[Metadata] = None):
        self.entities = entities
        for i, entity in enumerate(entities):
            entity.layer = self
            entity.id = i
        self.metadata = metadata if metadata else Metadata()
        self.span_indexer = EntitySpanIndexer(entities=entities)
        self.box_indexer = EntityBoxIndexer(entities=entities)
        self._name = None
        self._doc = None

    def __repr__(self):
        entity_repr = "\n".join([f"\t{e}" for e in self.entities])
        return f"Layer with {len(self)} Entities:\n{entity_repr}"

    def __getitem__(self, key):
        return self.entities[key]

    def __len__(self):
        return len(self.entities)

    def __iter__(self):
        return iter(self.entities)

    def __contains__(self, item):
        return item in self.entities

    def __eq__(self, other: "Layer") -> bool:
        """Layers are equal if all elements are equal"""
        return self.entities == other.entities

    def to_json(self):
        return [entity.to_json() for entity in self.entities]

    @classmethod
    def from_json(cls, layer_json):
        return cls(entities=[Entity.from_json(entity_json) for entity_json in layer_json])

    def find(self, query: Union[Span, Box]) -> List[Entity]:
        logger = logging.getLogger(__name__)
        logger.warning(
            "This method is deprecated due to ambiguity and will be removed in a future release."
            "Please use Layer.intersect_by_span or Layer.intersect_by_box instead."
        )
        if isinstance(query, Span):
            return self.intersect_by_span(query=Entity(spans=[query]))
        elif isinstance(query, Box):
            return self.intersect_by_box(query=Entity(boxes=[query]))
        else:
            raise TypeError(f"Unsupported query type {type(query)}")

    def intersect_by_span(self, query: Entity) -> List[Entity]:
        return self.span_indexer.find(query=query)

    def intersect_by_box(self, query: Entity) -> List[Entity]:
        return self.box_indexer.find(query=query)

    @property
    def doc(self) -> Optional["Document"]:
        return self._doc

    @doc.setter
    def doc(self, doc: Optional["Document"]) -> None:
        """This method attaches a Document to this Layer, allowing the Layer
        to access things beyond itself within the Document (e.g. other layers)"""
        if self.doc and doc:
            raise AttributeError(
                "Already has an attached Document. Since Layer should correspond"
                "to only a specific Document, we recommend creating a new"
                "Layer from scratch and then attaching your Document."
            )
        self._doc = doc

    @property
    def name(self) -> Optional[int]:
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        if self.name and name:
            raise AttributeError(f"This Layer already has a name: {self.name}")
        self._name = name

    def __getattr__(self, name: str) -> List["Entity"]:
        if not self.doc:
            raise AttributeError("This Layer is missing a Document")

        if name not in self.doc.layers:
            raise AttributeError(f"Layer {name} not found in Document")

        return self.crossref_layers_by_span(source=self, target=self.doc.get_layer(name=name))

    @classmethod
    def crossref_layers_by_span(cls, source: "Layer", target: "Layer") -> List[Entity]:
        """This method allows you to access overlapping Entities between two Layers"""
        raise NotImplementedError
