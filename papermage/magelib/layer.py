"""

Layers are collections of Entities. Supports indexing and slicing.

@kylel

"""

from typing import List

from .entity import Entity


class Layer:
    """Views into a document. Immutable. Lightweight."""

    __slots__ = ["entities"]

    def __init__(self, entities: List[Entity]):
        self.entities = entities

    def __repr__(self):
        return f"Layer{self.entities}"

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
