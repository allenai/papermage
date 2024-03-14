"""


@kylel

"""

import logging
from itertools import chain
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from .box import Box
from .entity import Entity
from .image import Image
from .layer import Layer
from .metadata import Metadata
from .names import (
    EntitiesFieldName,
    ImagesFieldName,
    MetadataFieldName,
    RelationsFieldName,
    SymbolsFieldName,
    TokensFieldName,
)
from .span import Span


class Prediction(NamedTuple):
    name: str
    entities: List[Entity]


class Document:
    tokens: Layer
    rows: Layer
    blocks: Layer
    words: Layer
    sentences: Layer
    paragraphs: Layer
    pages: Layer

    SPECIAL_FIELDS = [SymbolsFieldName, ImagesFieldName, MetadataFieldName]

    def __init__(
        self,
        symbols: Optional[str] = None,
        images: Optional[List[Image]] = None,
        metadata: Optional[Metadata] = None,
    ):
        self.symbols = symbols if symbols else None
        self.images = images if images else None
        if not self.symbols and not self.images:
            raise ValueError("Document must have at least one of `symbols` or `images`")
        self.metadata = metadata if metadata else Metadata()
        self._layers: List[str] = []

    @property
    def layers(self) -> List[str]:
        return self.SPECIAL_FIELDS + self._layers

    def validate_layer_name_availability(self, name: str) -> None:
        if name in self.layers:
            raise AssertionError(f"{name} not allowed Document.SPECIAL_FIELDS.")
        if name in self.layers:
            raise AssertionError(f'{name} already exists. Try `doc.remove_layer("{name}")` first.')
        if name in dir(self):
            raise AssertionError(f"{name} clashes with Document class properties.")

    def get_layer(self, name: str) -> Layer:
        """Gets a layer by name. For example, `doc.get_layer("sentences")` returns sentences."""
        return getattr(self, name)

    def annotate(self, *predictions: Union[Prediction, Tuple[Prediction, ...]]) -> None:
        """Annotates the document with predictions."""
        all_preds = chain.from_iterable([p] if isinstance(p, Prediction) else p for p in predictions)
        for prediction in all_preds:
            self.annotate_layer(name=prediction.name, entities=prediction.entities)

    def annotate_layer(self, name: str, entities: Union[List[Entity], Layer]) -> None:
        self.validate_layer_name_availability(name=name)

        if isinstance(entities, list):
            layer = Layer(entities=entities)
        else:
            layer = entities

        layer.doc = self
        layer.name = name
        setattr(self, name, layer)
        self._layers.append(name)

    def remove_layer(self, name: str):
        if name not in self.layers:
            pass
        else:
            getattr(self, name).doc = None
            getattr(self, name).name = None
            delattr(self, name)
            self._layers.remove(name)

    def get_relation(self, name: str) -> List["Relation"]:
        raise NotImplementedError

    def annotate_relation(self, name: str) -> None:
        self.validate_layer_name_availability(name=name)
        raise NotImplementedError

    def remove_relation(self, name: str) -> None:
        raise NotImplementedError

    def annotate_images(self, images: List[Image]) -> None:
        if len(images) == 0:
            raise ValueError("No images were provided")

        image_types = {type(image) for image in images}
        if len(image_types) > 1:
            raise TypeError(f"Images contain multiple types: {image_types}")
        image_type = image_types.pop()

        if not issubclass(image_type, Image):
            raise NotImplementedError(f"Unsupported image type {image_type} for {ImagesFieldName}")

        setattr(self, ImagesFieldName, images)

    def remove_images(self) -> None:
        raise NotImplementedError

    def to_json(self, layers: Optional[List[str]] = None, with_images: bool = False) -> Dict:
        """Returns a dictionary that's suitable for serialization

        Use `fields` to specify a subset of groups in the Document to include (e.g. 'sentences')

        Output format looks like
            {
                symbols: "...",
                entities: {...},
                relations: {...},
                metadata: {...}
            }
        """
        # 1) instantiate basic Document dict
        doc_dict = {
            SymbolsFieldName: self.symbols,
            MetadataFieldName: self.metadata.to_json(),
            EntitiesFieldName: {},
            RelationsFieldName: {},
        }

        # 2) serialize each layer to JSON
        layers = self._layers if layers is None else layers
        for layer in layers:
            doc_dict[EntitiesFieldName][layer] = [entity.to_json() for entity in getattr(self, layer)]

        # 3) serialize images if `with_images == True`
        if with_images:
            doc_dict[ImagesFieldName] = [image.to_base64() for image in getattr(self, ImagesFieldName)]

        return doc_dict

    @classmethod
    def from_json(cls, doc_json: Dict) -> "Document":
        # 1) instantiate basic Document
        symbols = doc_json[SymbolsFieldName]
        doc = cls(symbols=symbols, metadata=Metadata(**doc_json.get(MetadataFieldName, {})))

        # 2) instantiate entities
        for field_name, entity_jsons in doc_json[EntitiesFieldName].items():
            entities = [Entity.from_json(entity_json=entity_json) for entity_json in entity_jsons]
            doc.annotate_layer(name=field_name, entities=entities)

        return doc

    def __repr__(self):
        return f"Document with {len(self.layers)} layers: {self.layers}"

    def find(self, query: Union[Span, Box], name: str) -> List[Entity]:
        """Finds all entities that intersect with the query"""
        logger = logging.getLogger(__name__)
        logger.warning(
            "This method is deprecated due to ambiguity and will be removed in a future release."
            "Please use Document.intersect_by_span or Document.intersect_by_box instead."
        )
        if isinstance(query, Span):
            return self.intersect_by_span(query=Entity(spans=[query]), name=name)
        elif isinstance(query, Box):
            return self.intersect_by_box(query=Entity(boxes=[query]), name=name)
        else:
            raise TypeError(f"Unsupported query type {type(query)}")

    def intersect_by_span(self, query: Entity, name: str) -> List[Entity]:
        """Finds all entities that intersect with the query"""
        return self.get_layer(name=name).intersect_by_span(query=query)

    def intersect_by_box(self, query: Entity, name: str) -> List[Entity]:
        """Finds all entities that intersect with the query"""
        return self.get_layer(name=name).intersect_by_box(query=query)
