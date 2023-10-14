"""

Layers are collections of Entities. Supports indexing and slicing.

@kylel

"""

from typing import List, Optional

from .entity import Entity
from .metadata import Metadata


class Layer:
    """A fancy list of Entities. Manages <list> things like indexing and slicing,
    but also gives access to things like reading order and other metadata."""

    __slots__ = ["entities", "metadata"]

    def __init__(self, entities: List[Entity], metadata: Optional[Metadata] = None):
        self.entities = entities
        self.metadata = metadata if metadata else Metadata()

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

    def to_json(self):
        return [entity.to_json() for entity in self.entities]

    @classmethod
    def from_json(cls, layer_json):
        return cls(entities=[Entity.from_json(entity_json) for entity_json in layer_json])
